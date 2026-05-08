"""
core/slam_map.py

The SLAM Map — the persistent coordinate system.

Stores:
  - Keyframes   : saved camera poses at important moments
  - Map points  : triangulated 3D landmarks in world space
  - Trajectory  : all camera positions (for path visualisation)

This is the "memory" of the SLAM system. Everything is expressed
in the coordinate frame set at initialisation (origin = first pose).

PyTorch tensors are used throughout so the map state can be
serialised and reloaded between sessions.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import json


# ── Data structures ──────────────────────────────────────────

@dataclass
class Keyframe:
    """A saved snapshot of the camera at a significant moment."""
    id:          int
    pose:        torch.Tensor          # [4,4] world-to-camera SE3
    position:    torch.Tensor          # [3]   XYZ in world frame
    descriptors: torch.Tensor          # [N,32] ORB descriptors
    keypoints:   list                  # cv2.KeyPoint list
    map_point_ids: list = field(default_factory=list)  # IDs of visible map points
    frame_idx:   int   = 0


@dataclass
class MapPoint:
    """A triangulated 3D landmark in world space."""
    id:          int
    position:    np.ndarray            # [3] XYZ world coords
    descriptor:  np.ndarray            # [32] representative ORB descriptor
    color:       np.ndarray = field(default_factory=lambda: np.array([200, 200, 200], dtype=np.uint8)) # [3] RGB color
    observations: int = 1              # how many keyframes see this point
    last_seen:   int  = 0             # frame index when last observed


# ── SLAM Map ─────────────────────────────────────────────────

class SLAMMap:
    """
    Maintains the global map and coordinate system.

    The coordinate system is defined as:
        Origin (0,0,0) = camera position at first keyframe
        X = right direction of first camera view
        Y = up direction
        Z = away from first camera look direction

    All positions are in 'relative units' (mono camera can't
    recover absolute scale without IMU). Scale factor can be
    set externally from IMU/barometer to convert to metres.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Core data
        self.keyframes:  list[Keyframe]  = []
        self.map_points: list[MapPoint]  = []
        self.trajectory: list[np.ndarray] = []  # [X,Y,Z] per frame

        # ID counters
        self._kf_id  = 0
        self._mp_id  = 0

        # State flags
        self.is_initialised = False
        self.tracking_lost  = False
        self.lost_frames    = 0

        # Stats
        self.frame_count  = 0
        self.total_distance = 0.0
        self._last_pos    = None

    # ── Keyframe management ───────────────────────────────────

    def add_keyframe(
        self,
        pose:        torch.Tensor,
        position:    torch.Tensor,
        descriptors: torch.Tensor,
        keypoints:   list,
        frame_idx:   int,
    ) -> Keyframe:
        """Add a new keyframe. Returns the created Keyframe."""
        kf = Keyframe(
            id           = self._kf_id,
            pose         = pose.clone(),
            position     = position.clone(),
            descriptors  = descriptors.clone() if descriptors is not None else None,
            keypoints    = keypoints,
            frame_idx    = frame_idx,
        )
        self.keyframes.append(kf)
        self._kf_id += 1
        return kf

    def should_add_keyframe(
        self,
        current_pos: torch.Tensor,
        min_dist:    float = 0.05,
        min_frames:  int   = 8,
    ) -> bool:
        """
        Decide if a new keyframe should be inserted.
        Criteria:
          - Enough frames have passed since last keyframe
          - Camera has moved far enough from last keyframe
        """
        if len(self.keyframes) == 0:
            return True

        # Frame gap check
        last_kf = self.keyframes[-1]
        if self.frame_count - last_kf.frame_idx < min_frames:
            return False

        # Distance check
        dist = torch.norm(current_pos - last_kf.position).item()
        return dist > min_dist

    # ── Map point management ──────────────────────────────────

    def add_map_points(self, positions: np.ndarray, descriptors: np.ndarray = None, colors: np.ndarray = None):
        """
        Add a batch of new 3D map points.

        Args:
            positions   : [N, 3] world-space coords
            descriptors : [N, 32] ORB descriptors (optional)
            colors      : [N, 3] RGB colors (optional)
        """
        for i, pos in enumerate(positions):
            # Skip degenerate points
            if not np.all(np.isfinite(pos)):
                continue
            if np.linalg.norm(pos) > 5000:   # too far from origin
                continue

            desc = descriptors[i] if descriptors is not None else np.zeros(32, np.uint8)
            col = colors[i] if colors is not None else np.array([200, 200, 200], dtype=np.uint8)
            mp = MapPoint(
                id          = self._mp_id,
                position    = pos.copy(),
                descriptor  = desc,
                color       = col.copy(),
                observations = 1,
                last_seen    = self.frame_count,
            )
            self.map_points.append(mp)
            self._mp_id += 1

    def prune_map_points(self, max_age: int = 300):
        """Remove map points not seen for a long time."""
        cutoff = self.frame_count - max_age
        self.map_points = [
            mp for mp in self.map_points
            if mp.last_seen > cutoff or mp.observations > 3
        ]

    def get_map_positions(self) -> np.ndarray:
        """Return all map point positions as [N, 3] numpy array."""
        if not self.map_points:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([mp.position for mp in self.map_points], dtype=np.float32)

    def get_map_colors(self) -> np.ndarray:
        """Return all map point colors as [N, 3] numpy array."""
        if not self.map_points:
            return np.zeros((0, 3), dtype=np.uint8)
        return np.array([mp.color for mp in self.map_points], dtype=np.uint8)

    # ── Trajectory tracking ───────────────────────────────────

    def update_trajectory(self, position: torch.Tensor):
        """Record current camera position to trajectory."""
        pos_np = position.cpu().numpy()
        self.trajectory.append(pos_np.copy())

        # Track total distance travelled
        if self._last_pos is not None:
            self.total_distance += float(np.linalg.norm(pos_np - self._last_pos))
        self._last_pos = pos_np.copy()

        self.frame_count += 1

    def get_trajectory(self) -> np.ndarray:
        """Return full trajectory as [N, 3] numpy array."""
        if not self.trajectory:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(self.trajectory, dtype=np.float32)

    # ── Coordinate system info ────────────────────────────────

    @property
    def origin(self) -> np.ndarray:
        """The world origin — always (0, 0, 0)."""
        return np.zeros(3, dtype=np.float32)

    @property
    def current_position(self) -> np.ndarray:
        """Most recent camera position."""
        if not self.trajectory:
            return np.zeros(3, dtype=np.float32)
        return self.trajectory[-1]

    @property
    def n_keyframes(self) -> int:
        return len(self.keyframes)

    @property
    def n_map_points(self) -> int:
        return len(self.map_points)

    # ── Save / Load the map ───────────────────────────────────

    def save(self, path: str):
        """
        Save the SLAM map to disk so it can be reloaded in future sessions.
        Map points and trajectory are saved as numpy arrays.
        Keyframe poses saved as tensors.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Trajectory
        traj = self.get_trajectory()
        np.save(path / 'trajectory.npy', traj)

        # Map points
        mp_pos = self.get_map_positions()
        np.save(path / 'map_points.npy', mp_pos)

        # Map colors
        mp_col = self.get_map_colors()
        np.save(path / 'map_colors.npy', mp_col)

        # Keyframe poses
        kf_poses = np.array([
            kf.pose.cpu().numpy() for kf in self.keyframes
        ])
        np.save(path / 'keyframe_poses.npy', kf_poses)

        # Metadata
        meta = {
            'n_keyframes':   self.n_keyframes,
            'n_map_points':  self.n_map_points,
            'total_distance': self.total_distance,
            'frame_count':   self.frame_count,
        }
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[SLAM Map] Saved → {path}")
        print(f"  Keyframes : {self.n_keyframes}")
        print(f"  Map points: {self.n_map_points}")
        print(f"  Distance  : {self.total_distance:.3f} units")

    def load(self, path: str):
        """Reload a previously saved map."""
        path = Path(path)

        traj = np.load(path / 'trajectory.npy')
        self.trajectory = list(traj)

        mp_pos = np.load(path / 'map_points.npy')
        try:
            mp_col = np.load(path / 'map_colors.npy')
        except FileNotFoundError:
            mp_col = np.full((len(mp_pos), 3), 200, dtype=np.uint8)

        self.map_points = [
            MapPoint(id=i, position=mp_pos[i], descriptor=np.zeros(32, np.uint8), color=mp_col[i])
            for i in range(len(mp_pos))
        ]
        self._mp_id = len(self.map_points)

        kf_poses = np.load(path / 'keyframe_poses.npy')
        self.keyframes = [
            Keyframe(
                id=i,
                pose=torch.tensor(kf_poses[i], device=self.device),
                position=torch.tensor(kf_poses[i][:3, 3], device=self.device),
                descriptors=None,
                keypoints=[],
            )
            for i in range(len(kf_poses))
        ]
        self._kf_id = len(self.keyframes)

        with open(path / 'meta.json') as f:
            meta = json.load(f)
        self.total_distance = meta['total_distance']
        self.frame_count    = meta['frame_count']
        self.is_initialised = True

        print(f"[SLAM Map] Loaded ← {path}")
        print(f"  Keyframes : {self.n_keyframes}")
        print(f"  Map points: {self.n_map_points}")

    def reset(self):
        """Clear the map back to empty state."""
        self.keyframes.clear()
        self.map_points.clear()
        self.trajectory.clear()
        self._kf_id = 0
        self._mp_id = 0
        self.is_initialised = False
        self.tracking_lost  = False
        self.lost_frames    = 0
        self.frame_count    = 0
        self.total_distance = 0.0
        self._last_pos      = None
        print("[SLAM Map] Reset to empty.")