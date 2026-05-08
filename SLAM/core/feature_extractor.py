"""
core/feature_extractor.py

ORB feature extraction pipeline.
- Detects ORB keypoints on CPU (OpenCV)
- Converts descriptors to PyTorch GPU tensors for fast matching
- Grid-based uniform distribution ensures features spread across the frame
  (critical for Mars-analog / low-texture terrain)
"""

import cv2
import numpy as np
import torch


class ORBExtractor:
    """
    Wraps OpenCV ORB with:
      - Grid-based feature selection  →  even spatial distribution
      - GPU tensor output             →  fast descriptor matching downstream
      - Adaptive thresholding         →  handles low-contrast terrain

    Args:
        n_features   : max features to keep per frame
        scale_factor : pyramid scale (1.2 is ORB default)
        n_levels     : pyramid levels
        ini_threshold: initial FAST threshold (lower = finds more on flat terrain)
        min_threshold: fallback if ini finds too few features
        grid_rows/cols: grid cells for uniform distribution
        device       : torch device ('cuda', 'cpu')
    """

    def __init__(
        self,
        n_features:    int   = 10000,
        scale_factor:  float = 1.2,
        n_levels:      int   = 8,
        ini_threshold: int   = 15,
        min_threshold: int   = 5,
        grid_rows:     int   = 4,
        grid_cols:     int   = 6,
        device:        str   = 'cuda',
    ):
        self.n_features    = n_features
        self.ini_threshold = ini_threshold
        self.min_threshold = min_threshold
        self.grid_rows     = grid_rows
        self.grid_cols     = grid_cols
        self.device        = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Primary extractor
        self.orb = cv2.ORB_create(
            nfeatures   = n_features,
            scaleFactor = scale_factor,
            nlevels     = n_levels,
            fastThreshold = ini_threshold,
        )
        # Fallback extractor with lower threshold for low-texture frames
        self.orb_fallback = cv2.ORB_create(
            nfeatures     = n_features,
            scaleFactor   = scale_factor,
            nlevels       = n_levels,
            fastThreshold = min_threshold,
        )

        # BFMatcher on GPU-transferred descriptors (Hamming for binary ORB)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ──────────────────────────────────────────
    def detect_and_compute(self, frame_bgr: np.ndarray):
        """
        Detect ORB features in a BGR frame.

        Returns:
            keypoints   : list of cv2.KeyPoint
            descriptors : torch.Tensor  [N, 32]  uint8  on self.device
            gray        : np.ndarray  [H, W]  uint8  (for optical flow fallback)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to boost contrast — critical for indoor/low-light scenes
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # Primary detection
        kps, desc = self.orb.detectAndCompute(gray_eq, None)

        # Fallback if too few features found
        if desc is None or len(kps) < 20:
            kps, desc = self.orb_fallback.detectAndCompute(gray_eq, None)

        if desc is None or len(kps) == 0:
            return [], None, gray

        # Grid-based uniform selection
        kps, desc = self._grid_select(kps, desc, gray.shape[1], gray.shape[0])

        # Move descriptors to GPU tensor for fast downstream matching
        desc_tensor = torch.from_numpy(desc.astype(np.float32)).to(self.device)

        return kps, desc_tensor, gray

    # ──────────────────────────────────────────
    def _grid_select(self, keypoints, descriptors, W, H):
        """
        Divide frame into a grid and keep the best K keypoints
        per cell. This prevents feature clustering while still
        allowing enough features for matching.
        """
        keep_per_cell = max(10, self.n_features // (self.grid_rows * self.grid_cols))
        cell_w = W / self.grid_cols
        cell_h = H / self.grid_rows

        # grid[row][col] = list of (response, idx)
        grid = [[[] for _ in range(self.grid_cols)]
                for _ in range(self.grid_rows)]

        for idx, kp in enumerate(keypoints):
            col = min(int(kp.pt[0] / cell_w), self.grid_cols - 1)
            row = min(int(kp.pt[1] / cell_h), self.grid_rows - 1)
            grid[row][col].append((kp.response, idx))

        # From each cell keep top keep_per_cell by response
        selected = []
        for row in grid:
            for cell in row:
                if cell:
                    cell.sort(key=lambda x: x[0], reverse=True)
                    selected += [idx for _, idx in cell[:keep_per_cell]]

        kps_sel  = [keypoints[i]    for i in selected]
        desc_sel = descriptors[selected]

        return kps_sel, desc_sel

    # ──────────────────────────────────────────
    def match(self, desc1: torch.Tensor, desc2: torch.Tensor, ratio: float = 0.75):
        """
        Lowe's ratio test matching between two descriptor sets.

        Args:
            desc1, desc2 : [N, 32] float32 tensors (on any device)
            ratio        : Lowe's ratio threshold (0.75 is standard)

        Returns:
            matches : list of cv2.DMatch (sorted by distance)
        """
        if desc1 is None or desc2 is None:
            return []

        # Back to CPU numpy for BFMatcher (OpenCV doesn't take CUDA tensors)
        d1 = desc1.cpu().numpy().astype(np.uint8)
        d2 = desc2.cpu().numpy().astype(np.uint8)

        raw = self.matcher.knnMatch(d1, d2, k=2)

        # Lowe's ratio test — rejects ambiguous matches
        good = []
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)

        return sorted(good, key=lambda x: x.distance)   