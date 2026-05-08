"""
core/slam_pipeline.py

Main Visual SLAM Pipeline — orchestrates all components.

Flow per frame:
  1. Extract ORB features from new frame
  2. Match against previous frame features
  3. Estimate relative pose (Essential Matrix or PnP)
  4. Update map: trajectory, keyframes, map points
  5. Return frame state for visualisation

States:
  INIT        → waiting for enough features to initialise
  TRACKING    → healthy, pose being estimated every frame
  LOST        → too few matches; trying to relocate
  RELOC       → attempting loop closure / re-initialisation
"""

import cv2
import numpy as np
import torch
import time
from enum import Enum

from .feature_extractor import ORBExtractor
from .pose_estimator    import PoseEstimator
from .slam_map          import SLAMMap


class TrackingState(Enum):
    INIT     = 'INITIALISING'
    TRACKING = 'TRACKING'
    LOST     = 'LOST'
    RELOC    = 'RELOCATING'


class SLAMPipeline:
    """
    Visual SLAM pipeline using ORB features + Essential Matrix + PnP.

    Args:
        camera_matrix : [3,3] numpy float64 intrinsic matrix
                        If None, uses estimates from frame size
        device        : 'cuda' or 'cpu'
        save_path     : directory to auto-save map on exit
    """

    def __init__(
        self,
        camera_matrix: np.ndarray = None,
        device:        str        = 'cuda',
        save_path:     str        = './slam_map_save',
    ):
        self.device    = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.save_path = save_path

        print(f"[SLAM] Running on device: {self.device}")

        # ── Components ──
        self.extractor = ORBExtractor(
            n_features    = 2000,   # detect up to 2000 candidates
            ini_threshold = 8,      # low threshold — finds corners on any surface
            min_threshold = 3,      # very low fallback
            grid_rows     = 4,      # 4x6 grid, 8 per cell = up to 192 features
            grid_cols     = 6,
            device        = str(self.device),
        )

        # Intrinsics — will be refined once we know frame size
        self._K_override = camera_matrix
        self.estimator   = None   # built after first frame (need frame size)
        self.map         = SLAMMap(device=str(self.device))

        # ── Tracking state ──
        self.state = TrackingState.INIT

        # Previous frame data
        self._prev_kps   = None
        self._prev_desc  = None
        self._prev_gray  = None
        self._prev_R     = np.eye(3)
        self._prev_t     = np.zeros((3,1))

        # Initialisation buffer
        self._init_kps  = None
        self._init_desc = None
        self._init_gray = None

        # Stats
        self._frame_idx    = 0
        self._match_count  = 0
        self._lost_streak  = 0
        self._fps_timer    = time.time()
        self._fps          = 0.0

        # For drawing: keep last matched keypoints
        self.last_matched_pts1 = None
        self.last_matched_pts2 = None
        self.last_n_matches    = 0
        self.last_inliers      = 0

    # ──────────────────────────────────────────────────────────
    # PROCESS FRAME  (main entry point)
    # ──────────────────────────────────────────────────────────
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Process one camera frame through the SLAM pipeline.

        Args:
            frame_bgr : [H, W, 3] uint8 BGR frame from OpenCV

        Returns:
            dict with keys:
              state        : TrackingState
              position     : [3] numpy XYZ world position
              rotation     : [3,3] numpy rotation matrix
              n_features   : int
              n_matches    : int
              n_keyframes  : int
              n_map_points : int
              fps          : float
              trajectory   : [N,3] numpy all positions
              map_points   : [M,3] numpy all landmark positions
        """
        t0 = time.time()
        H, W = frame_bgr.shape[:2]

        # Build pose estimator once we know frame size
        if self.estimator is None:
            self._build_estimator(W, H)

        # ── Extract features ──────────────────────────────────
        kps, desc, gray = self.extractor.detect_and_compute(frame_bgr)
        n_feats = len(kps)

        # ── Route by state ────────────────────────────────────
        if self.state == TrackingState.INIT:
            self._handle_init(kps, desc, gray, frame_bgr)

        elif self.state == TrackingState.TRACKING:
            self._handle_tracking(kps, desc, gray, frame_bgr)

        elif self.state in (TrackingState.LOST, TrackingState.RELOC):
            self._handle_lost(kps, desc, gray)

        # ── Update map trajectory ─────────────────────────────
        pos = self.estimator.position
        self.map.update_trajectory(pos)
        self.map.frame_count = self._frame_idx

        # Periodic map maintenance (less aggressive for dense 3D map)
        if self._frame_idx % 60 == 0:
            self.map.prune_map_points(max_age=10000)

        # ── FPS ───────────────────────────────────────────────
        now = time.time()
        self._fps = 1.0 / max(now - t0, 1e-6)
        self._frame_idx += 1

        # ── Return state dict ─────────────────────────────────
        return {
            'state':        self.state,
            'position':     self.estimator.position.cpu().numpy(),
            'rotation':     self.estimator.rotation.cpu().numpy(),
            'n_features':   n_feats,
            'n_matches':    self.last_n_matches,
            'n_inliers':    self.last_inliers,
            'n_keyframes':  self.map.n_keyframes,
            'n_map_points': self.map.n_map_points,
            'fps':          self._fps,
            'trajectory':   self.map.get_trajectory(),
            'map_points':   self.map.get_map_positions(),
            'map_colors':   self.map.get_map_colors(),
            'kf_poses':     [kf.pose.cpu().numpy() for kf in self.map.keyframes],
            'total_dist':   self.map.total_distance,
            'frame_idx':    self._frame_idx,
        }

    # ──────────────────────────────────────────────────────────
    # STATE HANDLERS
    # ──────────────────────────────────────────────────────────

    def _handle_init(self, kps, desc, gray, frame_bgr):
        """
        Initialisation state: collect two frames with enough features
        to bootstrap the map via Essential Matrix triangulation.
        """
        if desc is None or len(kps) < 15:
            print(f"[INIT] Too few features: {len(kps) if kps else 0} — need 15+", end='\r')
            return

        if self._init_kps is None:
            # Save first frame
            self._init_kps  = kps
            self._init_desc = desc
            self._init_gray = gray
            print(f"[INIT] First frame saved — {len(kps)} features. Now move camera SIDEWAYS.")
            return

        # Try to initialise with second frame
        matches = self.extractor.match(self._init_desc, desc, ratio=0.80)
        print(f"[INIT] Features:{len(kps)}  Matches:{len(matches)}/12 needed", end='\r')

        if len(matches) < 12:
            # Update first frame to current so baseline keeps growing
            self._init_kps  = kps
            self._init_desc = desc
            self._init_gray = gray
            return

        print(f"\n[INIT] {len(matches)} matches — attempting Essential Matrix...")
        R, t, mask = self.estimator.estimate_relative_pose(
            self._init_kps, kps, matches, min_matches=10
        )

        if R is None:
            print("[INIT] Essential matrix failed — not enough baseline yet, keep moving sideways")
            return

        # Bootstrap map: triangulate initial map points
        pts1 = np.float32([self._init_kps[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps[m.trainIdx].pt            for m in matches])

        pts3d = self.estimator.triangulate(
            np.eye(3), np.zeros((3,1)),
            R.cpu().numpy(), t.cpu().numpy(),
            pts1, pts2
        )

        if len(pts3d) < 5:
            print("[INIT] Too few 3D points — keep moving camera to build baseline")
            # DO NOT reset _init_kps here! We need the baseline between the first frame and current frame to grow.
            return

        # Extract colors from the image at the keypoint locations
        colors = []
        for pt in pts2:
            x, y = int(pt[0]), int(pt[1])
            # BGR to RGB
            if 0 <= y < frame_bgr.shape[0] and 0 <= x < frame_bgr.shape[1]:
                b, g, r = frame_bgr[y, x]
                colors.append([r, g, b])
            else:
                colors.append([200, 200, 200])
        colors = np.array(colors, dtype=np.uint8)

        # Add initial keyframe at origin
        self.map.add_keyframe(
            pose        = torch.eye(4, device=self.device),
            position    = torch.zeros(3, device=self.device),
            descriptors = self._init_desc,
            keypoints   = self._init_kps,
            frame_idx   = 0,
        )
        self.map.add_map_points(pts3d, colors=colors)

        # Integrate first motion
        self.estimator.integrate_pose(R, t)
        pos = self.estimator.position
        self.map.add_keyframe(
            pose        = self.estimator.current_pose,
            position    = pos,
            descriptors = desc,
            keypoints   = kps,
            frame_idx   = self._frame_idx,
        )

        # Set prev frame data and transition to tracking
        self._prev_kps   = kps
        self._prev_desc  = desc
        self._prev_gray  = gray
        self._prev_R     = R.cpu().numpy()
        self._prev_t     = t.cpu().numpy()
        self.map.is_initialised = True
        self.state = TrackingState.TRACKING
        self.last_n_matches = len(matches)
        print(f"[SLAM] Initialised! {len(pts3d)} map points, {len(matches)} matches")

    def _handle_tracking(self, kps, desc, gray, frame_bgr):
        """
        Normal tracking: match features, estimate pose,
        optionally add keyframe.
        """
        if desc is None or self._prev_desc is None:
            self._go_lost()
            return

        matches = self.extractor.match(self._prev_desc, desc, ratio=0.80)
        self.last_n_matches = len(matches)

        if len(matches) < 8:
            self._lost_streak += 1
            if self._lost_streak > 8:
                self._go_lost()
            return

        self._lost_streak = 0

        # Estimate relative pose
        R, t, mask = self.estimator.estimate_relative_pose(
            self._prev_kps, kps, matches, min_matches=12
        )

        if R is None:
            self._lost_streak += 1
            return

        # Count inliers
        inlier_mask = mask.ravel() if mask is not None else []
        self.last_inliers = int(np.sum(inlier_mask == 1)) if len(inlier_mask) else 0

        # Store matched points for overlay drawing
        self.last_matched_pts1 = np.float32([self._prev_kps[m.queryIdx].pt for m in matches])
        self.last_matched_pts2 = np.float32([kps[m.trainIdx].pt            for m in matches])

        # Integrate pose
        self.estimator.integrate_pose(R, t)
        pos = self.estimator.position

        # Keyframe insertion decision
        if self.map.should_add_keyframe(pos, min_dist=0.05, min_frames=8):
            self.map.add_keyframe(
                pose        = self.estimator.current_pose,
                position    = pos,
                descriptors = desc,
                keypoints   = kps,
                frame_idx   = self._frame_idx,
            )
            # Triangulate new map points from this keyframe pair
            # Extract absolute world-to-camera poses for proper triangulation
            T_c2w_1 = self.estimator.poses[-2]
            T_c2w_2 = self.estimator.poses[-1]
            
            T_w2c_1 = torch.inverse(T_c2w_1).cpu().numpy()
            T_w2c_2 = torch.inverse(T_c2w_2).cpu().numpy()

            pts1 = np.float32([self._prev_kps[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kps[m.trainIdx].pt            for m in matches])
            
            pts3d = self.estimator.triangulate(
                T_w2c_1[:3, :3], T_w2c_1[:3, 3:4],
                T_w2c_2[:3, :3], T_w2c_2[:3, 3:4],
                pts1, pts2,
            )
            if len(pts3d) > 0:
                colors = []
                for pt in pts2:
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= y < frame_bgr.shape[0] and 0 <= x < frame_bgr.shape[1]:
                        b, g, r = frame_bgr[y, x]
                        colors.append([r, g, b])
                    else:
                        colors.append([200, 200, 200])
                colors = np.array(colors, dtype=np.uint8)
                self.map.add_map_points(pts3d, colors=colors)

        # Advance prev frame
        self._prev_kps  = kps
        self._prev_desc = desc
        self._prev_gray = gray
        self._prev_R    = R.cpu().numpy()
        self._prev_t    = t.cpu().numpy()

    def _handle_lost(self, kps, desc, gray):
        """
        Tracking lost: try to re-initialise using current features.
        After enough lost frames, attempt full re-init from scratch.
        """
        self.map.lost_frames += 1

        # Try matching against most recent keyframe
        if self.map.keyframes and desc is not None:
            last_kf = self.map.keyframes[-1]
            if last_kf.descriptors is not None:
                matches = self.extractor.match(last_kf.descriptors, desc, ratio=0.70)
                if len(matches) >= 20:
                    # Recovered! Resume tracking
                    self._prev_kps  = kps
                    self._prev_desc = desc
                    self._prev_gray = gray
                    self.state = TrackingState.TRACKING
                    self.map.lost_frames = 0
                    self._lost_streak = 0
                    print(f"[SLAM] Relocalized! {len(matches)} matches with last keyframe")
                    return

        # Hard re-init after extended loss
        if self.map.lost_frames > 30:
            print("[SLAM] Re-initialising from scratch (tracking lost too long)")
            self._init_kps  = None
            self._init_desc = None
            self.state = TrackingState.INIT
            self.map.lost_frames = 0

    def _go_lost(self):
        self.state = TrackingState.LOST
        self.map.tracking_lost = True
        print(f"[SLAM] Tracking lost at frame {self._frame_idx}")

    # ──────────────────────────────────────────────────────────
    # INTRINSICS
    # ──────────────────────────────────────────────────────────

    def _build_estimator(self, W: int, H: int):
        """Build PoseEstimator with correct intrinsics for this frame size."""
        if self._K_override is not None:
            K = self._K_override
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
        else:
            # Realistic estimate: focal length ≈ 0.7 * image_width
            # (matches typical webcam/phone FOV of ~70 degrees)
            # For best accuracy calibrate with a checkerboard pattern.
            f  = max(W, H) * 0.7
            fx = fy = f
            cx, cy = W / 2.0, H / 2.0
            print(f"[SLAM] Using estimated intrinsics: fx={fx:.0f} cx={cx:.0f} cy={cy:.0f}")
            print(f"       For better accuracy, run: python calibrate_camera.py")

        self.estimator = PoseEstimator(
            fx=fx, fy=fy, cx=cx, cy=cy,
            device=str(self.device),
        )

    # ──────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ──────────────────────────────────────────────────────────

    def save_map(self, path: str = None):
        self.map.save(path or self.save_path)

    def load_map(self, path: str = None):
        self.map.load(path or self.save_path)
        self.state = TrackingState.TRACKING

    def reset(self):
        self.map.reset()
        self.estimator.reset() if self.estimator else None
        self._prev_kps = self._prev_desc = self._prev_gray = None
        self._init_kps = self._init_desc = None
        self._frame_idx = self._lost_streak = 0
        self.state = TrackingState.INIT
        print("[SLAM] Full reset.")