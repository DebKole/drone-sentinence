"""
run_slam.py — Visual SLAM entry point

Two windows:
  Left  (OpenCV)      — camera feed + ORB feature overlay
  Right (matplotlib)  — live 3D path growing as camera moves

Usage:
    # Webcam
    python -m visual_slam.run_slam

    # Phone (Android — install 'IP Webcam' app, start server)
    python -m visual_slam.run_slam --source http://192.168.X.X:8080/video

    # iPhone (install 'Camo' or 'EpocCam')
    python -m visual_slam.run_slam --source http://192.168.X.X:8080/live

    # Load a previously saved map and continue
    python -m visual_slam.run_slam --load

Controls (focus the camera window):
    Q / ESC  → quit
    R        → reset map + clear path
    S        → save map now
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.slam_pipeline  import SLAMPipeline
from viz.visualiser      import Visualiser
from utils.camera_source import CameraSource


def load_kitti_calib(filepath):
    """Parse KITTI calib.txt to extract the left color camera intrinsic matrix K."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('P2:'):
                values = [float(v) for v in line.strip().split()[1:]]
                P2 = np.array(values).reshape(3, 4)
                return P2[:, :3]
        
        # If no labels found, assume the first line is the left camera projection matrix
        if len(lines) > 0 and len(lines[0].split()) >= 12:
            values = [float(v) for v in lines[0].strip().split()[:12]]
            P0 = np.array(values).reshape(3, 4)
            return P0[:, :3]

    return None


def parse_args():
    p = argparse.ArgumentParser(description='Visual SLAM — ORB + matplotlib path trace')
    p.add_argument('--source',    default=0,
                   help='0=webcam, URL=phone stream, path=video file')
    p.add_argument('--width',     type=int, default=640,
                   help='Capture width (0 = keep original, required for KITTI)')
    p.add_argument('--height',    type=int, default=480)
    p.add_argument('--calib',     type=str, default=None,
                   help='Path to KITTI calib.txt file. If set, --width/--height default to 0')
    p.add_argument('--fps',       type=int, default=30)
    p.add_argument('--save-path', default='./slam_map_save')
    p.add_argument('--load',      action='store_true',
                   help='Resume from previously saved map')
    p.add_argument('--device',    default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--no-flip',   action='store_true',
                   help='Disable horizontal flip (front cam mirrors by default)')
    return p.parse_args()


def main():
    args = parse_args()

    K = None
    if args.calib:
        if os.path.exists(args.calib):
            K = load_kitti_calib(args.calib)
            if K is not None:
                print(f"[SLAM] Loaded camera matrix from {args.calib}")
                # Force original resolution if calibration is provided, to avoid mismatch
                if args.width == 640 and args.height == 480:
                    args.width = 0
                    args.height = 0
            else:
                print(f"[WARN] Could not find 'P2:' in {args.calib}")
        else:
            print(f"[WARN] Calibration file not found: {args.calib}")

    # Convert digit string to int for webcam index
    source = (int(args.source)
              if isinstance(args.source, str) and args.source.isdigit()
              else args.source)

    # Flip front-facing webcam; don't flip phone/file/dir
    flip = -1 if (args.no_flip or isinstance(source, str)) else 1

    print("=" * 54)
    print("  Visual SLAM — ORB features + matplotlib path trace")
    print("=" * 54)
    print(f"  Source    : {source}")
    print(f"  Capture   : {args.width}x{args.height} @ {args.fps}fps")
    print(f"  Device    : {args.device}")
    print(f"  Save path : {args.save_path}")
    print()
    print("  HOW TO GET A GOOD PATH:")
    print("  1. Point camera at a textured surface (wall, floor, desk)")
    print("  2. Move SIDEWAYS slowly — don't rotate in place")
    print("  3. Watch for green dots appearing in camera window")
    print("  4. Once you see KF counter rising, SLAM is running")
    print()
    print("  Controls (focus camera window first):")
    print("    Q / ESC  -> quit")
    print("    R        -> reset path")
    print("    S        -> save map")
    print("=" * 54)

    # ── Camera ────────────────────────────────────────────────
    # NOTE: We request 640x480 even from the phone.
    # The phone streams 1920x1080 but we downscale immediately —
    # ORB runs much faster and more reliably at lower resolution.
    cam = CameraSource(
        source = source,
        width  = args.width,
        height = args.height,
        fps    = args.fps,
        flip   = flip,
    )

    try:
        actual_w, actual_h = cam.open()
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        print("\nPhone camera tips:")
        print("  Android -> install 'IP Webcam' (Play Store)")
        print("             Start server, note URL")
        print("             python -m visual_slam.run_slam --source http://IP:PORT/video")
        print("  iPhone  -> install 'Camo' app")
        print("             python -m visual_slam.run_slam --source http://IP:PORT/live")
        sys.exit(1)

    # ── SLAM pipeline ─────────────────────────────────────────
    slam = SLAMPipeline(
        camera_matrix = K,
        device    = args.device,
        save_path = args.save_path,
    )

    if args.load:
        try:
            slam.load_map(args.save_path)
            print(f"[SLAM] Loaded map from {args.save_path}")
        except Exception as e:
            print(f"[WARN] Could not load map: {e} — starting fresh.")

    # ── Visualiser ────────────────────────────────────────────
    # Use the requested display size (not actual capture size) unless 0
    vis_w = args.width if args.width > 0 else actual_w
    vis_h = args.height if args.height > 0 else actual_h
    vis = Visualiser(cam_w=vis_w, cam_h=vis_h)

    # vis.run() hands control to matplotlib on the main thread.
    # The SLAM loop runs inside a background thread managed by vis.
    # This call blocks until the user closes either window or presses Q.
    vis.run(slam, cam)

    # ── Cleanup ───────────────────────────────────────────────
    cam.release()

    print("\n[SLAM] Session complete.")
    print(f"  Path points : {len(vis._path_pts)}")
    print(f"  Keyframes   : {slam.map.n_keyframes}")
    print(f"  Map points  : {slam.map.n_map_points}")
    print(f"  Distance    : {slam.map.total_distance:.3f} units")
    if slam.map.n_keyframes > 0:
        print(f"  Map saved   -> {args.save_path}")


if __name__ == '__main__':
    main()