import argparse
import sys
import os
import cv2
import numpy as np
import torch
import threading
import time
import collections

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.feature_extractor import ORBExtractor
from core.pose_estimator import PoseEstimator
from utils.camera_source import CameraSource


def load_kitti_calib(filepath):
    """Parse KITTI calib.txt to extract the left color camera intrinsic matrix K (from P2)."""
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('P2:'):
                values = [float(v) for v in line.strip().split()[1:]]
                P2 = np.array(values).reshape(3, 4)
                return P2[:, :3]
    return None

class VisualOdometryApp:
    def __init__(self, source, width, height, fps, calib=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[VO] Using device: {self.device}")

        # Initialize Camera
        self.cam = CameraSource(source=source, width=width, height=height, fps=fps, flip=-1)
        self.actual_w, self.actual_h = self.cam.open()
        
        K = None
        if calib and os.path.exists(calib):
            K = load_kitti_calib(calib)
            
        if K is not None:
            print(f"[VO] Loaded camera matrix from {calib}")
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
        else:
            # We estimate fx, fy based on width/height (approx 70 deg FOV)
            f = max(self.actual_w, self.actual_h) * 0.7
            fx, fy = f, f
            cx, cy = self.actual_w / 2.0, self.actual_h / 2.0
            print(f"[VO] Intrinsics estimate: f={f:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
        print(f"[VO] Camera opened. Resolution: {self.actual_w}x{self.actual_h}")

        # Initialize VO Components
        self.extractor = ORBExtractor(
            n_features=1500,
            ini_threshold=10,
            min_threshold=5,
            device=self.device
        )
        self.estimator = PoseEstimator(
            fx=fx, fy=fy, cx=cx, cy=cy, device=self.device
        )

        # VO State
        self.prev_kps = None
        self.prev_desc = None
        self.trajectory = collections.deque(maxlen=5000)
        self.trajectory.append((0.0, 0.0, 0.0))  # Start at origin

        # Threading/Sync
        self.lock = threading.Lock()
        self.quit = False

    def run(self):
        """Starts the background processing thread and main visualization thread."""
        bg_thread = threading.Thread(target=self.process_loop, daemon=True)
        bg_thread.start()

        self.build_plot()
        plt.show()  # Blocks here

        self.quit = True
        bg_thread.join()
        self.cam.release()
        cv2.destroyAllWindows()
        print("[VO] Session ended.")

    def process_loop(self):
        cv2.namedWindow('Visual Odometry', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Visual Odometry', self.actual_w, self.actual_h)

        frame_idx = 0
        while not self.quit:
            ok, frame = self.cam.read()
            if not ok:
                if getattr(self.cam, 'is_file', False) or getattr(self.cam, 'is_dir', False):
                    # Pause at end of video/images
                    key = cv2.waitKey(100) & 0xFF
                    if key in (ord('q'), 27):
                        self.quit = True
                        plt.close('all')
                        break
                    continue
                time.sleep(0.05)
                continue
            
            # 1. Feature Extraction
            kps, desc, gray = self.extractor.detect_and_compute(frame)

            display_frame = frame.copy()

            if self.prev_desc is not None and desc is not None:
                # 2. Feature Tracking
                matches = self.extractor.match(self.prev_desc, desc, ratio=0.80)
                
                # 3. Motion Estimation
                R, t, mask = self.estimator.estimate_relative_pose(
                    self.prev_kps, kps, matches, min_matches=12
                )

                if R is not None:
                    # 4. Pose Integration
                    pos = self.estimator.integrate_pose(R, t)
                    
                    with self.lock:
                        p_np = pos.cpu().numpy()
                        self.trajectory.append((float(p_np[0]), float(p_np[1]), float(p_np[2])))

                    # Draw inliers
                    inliers = mask.ravel() == 1
                    valid_matches = [m for i, m in enumerate(matches) if inliers[i]]
                    
                    for m in valid_matches:
                        p1 = self.prev_kps[m.queryIdx].pt
                        p2 = kps[m.trainIdx].pt
                        cv2.line(display_frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 1)
                        cv2.circle(display_frame, (int(p2[0]), int(p2[1])), 3, (0, 0, 255), -1)
                        
                    cv2.putText(display_frame, f"Tracked: {len(valid_matches)}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Tracking Lost", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw keypoints if first frame
            if self.prev_desc is None and kps is not None:
                for kp in kps:
                    cv2.circle(display_frame, (int(kp.pt[0]), int(kp.pt[1])), 2, (255, 0, 0), -1)
                cv2.putText(display_frame, f"Initialized: {len(kps)} features", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            self.prev_kps = kps
            self.prev_desc = desc
            frame_idx += 1

            if getattr(self.cam, 'is_dir', False):
                time.sleep(0.05)  # Simulate framerate for images

            cv2.imshow('Visual Odometry', display_frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                self.quit = True
                plt.close('all')
                break

    def build_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Z')
        self.ax.set_title('Visual Odometry Trajectory (Top-Down)')
        self.ax.grid(True)
        
        self.path_line, = self.ax.plot([], [], 'b-', linewidth=2, label='Trajectory')
        self.curr_pos = self.ax.scatter([0], [0], c='r', s=50, label='Current Pose')
        self.ax.legend()
        
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)

        self.anim = animation.FuncAnimation(
            self.fig, self.update_plot, interval=50, cache_frame_data=False
        )

    def update_plot(self, _frame):
        with self.lock:
            pts = list(self.trajectory)
        
        if len(pts) < 2:
            return

        pts_np = np.array(pts)
        xs = pts_np[:, 0]
        zs = pts_np[:, 2]  # We plot X-Z plane

        self.path_line.set_data(xs, zs)
        self.curr_pos.set_offsets(np.c_[xs[-1:], zs[-1:]])

        pad = 2.0
        self.ax.set_xlim(xs.min() - pad, xs.max() + pad)
        self.ax.set_ylim(zs.min() - pad, zs.max() + pad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='Path to video file or image directory')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--calib', type=str, default=None, help='Path to KITTI calib.txt file')
    p.add_argument('--fps', type=int, default=30)
    args = p.parse_args()

    if args.calib:
        args.width = 0
        args.height = 0

    app = VisualOdometryApp(
        source=args.source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        calib=args.calib
    )
    app.run()


if __name__ == '__main__':
    main()
