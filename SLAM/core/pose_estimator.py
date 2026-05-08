"""
core/pose_estimator.py

Estimates camera pose (R, t) between consecutive frames using:
  1. Essential Matrix from matched keypoints  →  relative R, t  (initialization)
  2. PnP + RANSAC from 3D map points          →  absolute pose   (tracking)

All rotation matrices and translation vectors are stored as PyTorch tensors
so they can be composed and stored efficiently on GPU.
"""

import cv2
import numpy as np
import torch


class PoseEstimator:
    """
    Computes camera-to-world transforms from feature matches.

    Coordinate convention (NED-like, right-hand):
        X → right (East)
        Y → up    (away from ground)
        Z → back  (opposite of camera look direction)

    Args:
        fx, fy, cx, cy : camera intrinsics (pixels)
                         If unknown, run calibrate_from_video() first
        device         : torch device
    """

    def __init__(
        self,
        fx:     float = 600.0,
        fy:     float = 600.0,
        cx:     float = 320.0,
        cy:     float = 240.0,
        device: str   = 'cuda',
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Camera intrinsic matrix  [3,3]
        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=np.float64)

        self.K_tensor = torch.tensor(self.K, dtype=torch.float32, device=self.device)

        # Pose history: list of 4x4 world-to-camera SE(3) matrices (torch)
        self.poses: list[torch.Tensor] = []

        # Current absolute pose  [4, 4]  (accumulated)
        self.current_pose = torch.eye(4, device=self.device)

        # Scale factor — mono can't recover absolute scale, so we fix it
        # to 1.0 initially and can be updated by external IMU/barometer
        self.scale = 1.0

    # ──────────────────────────────────────────────────────────
    # ESSENTIAL MATRIX POSE  (between two frames, no map)
    # Used for the first two keyframes to bootstrap the map
    # ──────────────────────────────────────────────────────────
    def estimate_relative_pose(
        self,
        kps1: list,   # cv2.KeyPoint list
        kps2: list,
        matches: list,  # cv2.DMatch list
        min_matches: int = 15,
    ):
        """
        Estimate relative rotation R and translation t between two frames
        using the Essential Matrix + RANSAC.

        Returns:
            R   : [3,3] torch.Tensor  rotation matrix
            t   : [3,1] torch.Tensor  unit translation (scale unknown)
            mask: inlier boolean mask (numpy)
            None, None, None if failed
        """
        if len(matches) < min_matches:
            return None, None, None

        # Extract matched pixel coordinates
        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

        # Essential matrix via RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None or mask is None:
            return None, None, None

        # Recover R and t from E
        inliers1 = pts1[mask.ravel() == 1]
        inliers2 = pts2[mask.ravel() == 1]

        _, R, t, pose_mask = cv2.recoverPose(E, inliers1, inliers2, self.K)

        R_t = torch.from_numpy(R.astype(np.float32)).to(self.device)
        t_t = torch.from_numpy(t.astype(np.float32)).to(self.device)

        return R_t, t_t, mask

    # ──────────────────────────────────────────────────────────
    # PnP POSE  (frame vs existing 3D map points)
    # Used after map has been bootstrapped — more accurate
    # ──────────────────────────────────────────────────────────
    def estimate_pnp_pose(
        self,
        pts3d:   np.ndarray,   # [N, 3]  3D map point coords
        pts2d:   np.ndarray,   # [N, 2]  corresponding pixel coords
        min_pts: int = 8,
    ):
        """
        Estimate absolute camera pose from 3D-2D correspondences
        using PnP + RANSAC (EPnP algorithm).

        Returns:
            R_mat : [3,3] torch.Tensor
            t_vec : [3,1] torch.Tensor
            inlier_mask : numpy bool array
            None on failure
        """
        if len(pts3d) < min_pts:
            return None, None, None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float64),
            pts2d.astype(np.float64),
            self.K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=4.0,
            confidence=0.99,
            iterationsCount=200,
        )

        if not success or inliers is None or len(inliers) < min_pts:
            return None, None, None

        R_mat, _ = cv2.Rodrigues(rvec)

        R_t = torch.from_numpy(R_mat.astype(np.float32)).to(self.device)
        t_t = torch.from_numpy(tvec.astype(np.float32)).to(self.device)

        return R_t, t_t, inliers

    # ──────────────────────────────────────────────────────────
    # POSE COMPOSITION — chain relative R,t into absolute pose
    # ──────────────────────────────────────────────────────────
    def integrate_pose(self, R: torch.Tensor, t: torch.Tensor):
        """
        Compose relative pose [R|t] with current absolute pose.
        Updates self.current_pose and appends to self.poses.

        The translation is scaled by self.scale (can be set from IMU).

        Returns:
            position : [3] world-space XYZ of camera
        """
        # Build 4x4 relative transform
        T_rel = torch.eye(4, device=self.device)
        T_rel[:3, :3] = R
        T_rel[:3,  3] = (t * self.scale).squeeze()

        # Accumulate: T_world = T_world @ inv(T_rel)
        # (camera moves in world space)
        self.current_pose = self.current_pose @ torch.inverse(T_rel)
        self.poses.append(self.current_pose.clone())

        # Extract world position (camera centre in world coords)
        position = self.current_pose[:3, 3]
        return position

    # ──────────────────────────────────────────────────────────
    # TRIANGULATE new 3D map points from two views
    # ──────────────────────────────────────────────────────────
    def triangulate(
        self,
        R1, t1,   # pose of frame 1
        R2, t2,   # pose of frame 2
        pts1: np.ndarray,   # [N,2] pixels in frame 1
        pts2: np.ndarray,   # [N,2] pixels in frame 2
    ):
        """
        Triangulate 3D positions of matched points.

        Returns:
            pts3d : [N, 3] float32 numpy array of world-space landmarks
        """
        P1 = self.K @ np.hstack([R1, t1])          # [3,4] projection
        P2 = self.K @ np.hstack([R2, t2])

        pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)   # [4, N]

        # Homogeneous → Euclidean
        pts3d = (pts4d[:3] / pts4d[3]).T            # [N, 3]

        # Filter points behind camera or too far away
        valid = (pts4d[2] > 0) & (pts4d[3] != 0)
        pts3d = pts3d[valid.flatten()]

        return pts3d.astype(np.float32)

    # ──────────────────────────────────────────────────────────
    # ROTATION → EULER ANGLES  (for display)
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def rotation_to_euler(R: torch.Tensor):
        """
        Convert [3,3] rotation matrix to (roll, pitch, yaw) in degrees.
        Uses ZYX convention.
        """
        R_np = R.cpu().numpy()
        sy = np.sqrt(R_np[0,0]**2 + R_np[1,0]**2)
        singular = sy < 1e-6

        if not singular:
            roll  = np.degrees(np.arctan2( R_np[2,1], R_np[2,2]))
            pitch = np.degrees(np.arctan2(-R_np[2,0], sy))
            yaw   = np.degrees(np.arctan2( R_np[1,0], R_np[0,0]))
        else:
            roll  = np.degrees(np.arctan2(-R_np[1,2], R_np[1,1]))
            pitch = np.degrees(np.arctan2(-R_np[2,0], sy))
            yaw   = 0.0

        return roll, pitch, yaw

    # ──────────────────────────────────────────────────────────
    # CURRENT XYZ POSITION
    # ──────────────────────────────────────────────────────────
    @property
    def position(self) -> torch.Tensor:
        """Current camera position [X, Y, Z] in world frame."""
        return self.current_pose[:3, 3]

    @property
    def rotation(self) -> torch.Tensor:
        """Current camera rotation [3,3] in world frame."""
        return self.current_pose[:3, :3]

    def reset(self):
        """Reset to origin."""
        self.current_pose = torch.eye(4, device=self.device)
        self.poses.clear()