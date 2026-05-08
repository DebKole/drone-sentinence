# Visual SLAM — ORB Features + PyTorch

A complete Visual SLAM pipeline that builds a coordinate system and 3D map
from a moving camera using ORB feature detection and Essential Matrix pose estimation.

---

## Project Structure

```
visual_slam/
├── run_slam.py                  ← ENTRY POINT — run this
├── requirements.txt
├── core/
│   ├── feature_extractor.py     ← ORB detector + grid selection + GPU matching
│   ├── pose_estimator.py        ← Essential Matrix + PnP + triangulation
│   └── slam_map.py              ← Keyframes + landmarks + coordinate system
├── viz/
│   └── visualiser.py            ← Camera feed + top-down map + stats HUD
└── utils/
    └── camera_source.py         ← Webcam + phone IP camera abstraction
```

---

## Install

### 1. Create conda / venv environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
```

### 2. Install PyTorch with CUDA (Windows + NVIDIA)

```bash
# PyTorch CUDA 12.1 — check https://pytorch.org for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install remaining dependencies

```bash
pip install opencv-python numpy
```

---

## Run

### Webcam
```bash
python -m visual_slam.run_slam
```

### Phone Camera (Android)
1. Install **IP Webcam** from Google Play Store
2. Open app → scroll to bottom → **Start server**
3. Note the URL shown (e.g. `http://192.168.1.5:8080`)
4. Run:
```bash
python -m visual_slam.run_slam --source http://192.168.1.5:8080/video
```

### Phone Camera (iPhone)
1. Install **Camo** app (or **EpocCam**)
2. Note the stream URL from the app
```bash
python -m visual_slam.run_slam --source http://192.168.1.5:8080/live
```

### Load a previously saved map
```bash
python -m visual_slam.run_slam --load
```

---

## Controls

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit and save map |
| `R` | Reset map |
| `S` | Save map now |
| `+` | Zoom in map view |
| `-` | Zoom out map view |

---

## What you see

```
┌──────────────────┬──────────────────┬──────────┐
│  Camera feed     │  Top-down map    │  Stats   │
│  + ORB features  │  XZ plane        │  Pose    │
│  + flow vectors  │  trajectory      │  Rotation│
│                  │  + landmarks     │  Counts  │
└──────────────────┴──────────────────┴──────────┘
```

- **Green dots** = ORB keypoints detected in current frame
- **Coloured lines** = optical flow vectors (green=slow, red=fast motion)
- **Amber dots** = triangulated 3D map points (top-down view)
- **Blue line** = camera trajectory (path traced so far)
- **Green dot (map)** = current camera position
- **Red/Blue axes** = X and Z axes of the SLAM coordinate system

---

## Tips for best results

1. **Move slowly** — fast motion blurs features and loses tracking
2. **Favour textured surfaces** — point at walls with texture, not blank white walls
3. **Keep camera steady** then pan — avoid shaky random motion
4. **Good lighting** — SLAM needs visible surface detail
5. **Wide baseline moves** — moving sideways is better than rotating in place

---

## Coordinate system

The coordinate system is set at the moment SLAM initialises (first two frames):

```
Origin (0, 0, 0)  =  camera position at startup
X axis            =  right direction of initial camera view
Y axis            =  up direction
Z axis            =  depth (away from camera)
```

All units are **relative** (monocular camera cannot recover absolute scale).
To convert to metres, set `slam.estimator.scale = metres_per_unit` using
an external IMU or known object size.

---

## Saving and reusing the map

The map is automatically saved on quit to `./slam_map_save/`:
- `trajectory.npy`       — all camera positions
- `map_points.npy`       — all 3D landmark positions
- `keyframe_poses.npy`   — all keyframe SE(3) poses
- `meta.json`            — stats

Reload with `--load` flag to continue where you left off.+