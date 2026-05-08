"""
viz/visualiser.py

Live 3D SLAM visualiser using Matplotlib 3D.
Guaranteed to work without advanced GPU drivers.

  Background thread -> SLAM processing & OpenCV camera feed
  Main thread       -> Matplotlib 3D interactive plot
"""

import cv2
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from core.slam_pipeline import TrackingState

# ── OpenCV colours (BGR) ──────────────────────────────────────
CV_GREEN = (0,  210, 120)
CV_AMBER = (30, 180, 240)
CV_RED   = (50,  50, 220)
CV_DIM   = (80,  90, 100)
CV_WHITE = (210, 210, 210)


class Visualiser:
    def __init__(self, cam_w: int = 640, cam_h: int = 480):
        self.cam_w = cam_w
        self.cam_h = cam_h

        self._path_pts = []
        self._quit = False
        self._lock = threading.Lock()
        
        self._slam_state = {}
        self._kps = []
        self._matched_pts1 = None
        self._matched_pts2 = None

    def run(self, slam, cam):
        # Start background thread
        bg = threading.Thread(target=self._slam_camera_thread, args=(slam, cam), daemon=True)
        bg.start()

        # Build and run matplotlib 3D window on main thread
        self._build_mpl_window()
        plt.show()

        self._quit = True
        bg.join(timeout=3.0)

        if slam.map.n_keyframes > 0:
            slam.save_map()
        cv2.destroyAllWindows()

    def _slam_camera_thread(self, slam, cam):
        cv2.namedWindow('SLAM -- Camera Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SLAM -- Camera Feed', self.cam_w, self.cam_h)

        try:
            while not self._quit:
                ok, frame = cam.read()
                if not ok:
                    if getattr(cam, 'is_file', False) or getattr(cam, 'is_dir', False):
                        print("[VIZ] End of sequence reached. Saving map and exiting...")
                        self._quit = True
                        break
                    time.sleep(0.05)
                    continue

                state_dict = slam.process_frame(frame)

                if getattr(cam, 'is_dir', False):
                    time.sleep(0.02) 

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kps  = slam.extractor.orb.detect(gray, None)

                pos   = state_dict.get('position', None)
                state = state_dict.get('state', TrackingState.INIT)

                if pos is not None and state == TrackingState.TRACKING:
                    with self._lock:
                        if (not self._path_pts or np.linalg.norm(np.array(pos) - np.array(self._path_pts[-1])) > 0.008):
                            self._path_pts.append((float(pos[0]), float(pos[1]), float(pos[2])))

                with self._lock:
                    self._slam_state = state_dict
                    self._kps = kps
                    self._matched_pts1 = slam.last_matched_pts1
                    self._matched_pts2 = slam.last_matched_pts2

                cam_img = self._draw_camera(frame, state_dict, kps, slam.last_matched_pts1, slam.last_matched_pts2)
                cv2.imshow('SLAM -- Camera Feed', cam_img)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), ord('Q'), 27):
                    self._quit = True
                    plt.close('all')
                    break
                elif key in (ord('r'), ord('R')):
                    slam.reset()
                    with self._lock:
                        self._path_pts.clear()
                elif key in (ord('s'), ord('S')):
                    slam.save_map()
                    
        except KeyboardInterrupt:
            print("\n[VIZ] Interrupted by user! Saving map before exit...")
            self._quit = True
            plt.close('all')
        finally:
            if slam.map.n_keyframes > 0:
                slam.save_map()
            cv2.destroyAllWindows()

    def _build_mpl_window(self):
        self._fig = plt.figure(figsize=(9, 7), facecolor='white', num='ORB-SLAM3: Live Map Viewer')
        self._ax = self._fig.add_subplot(111, projection='3d')
        
        self._ax.set_facecolor('white')
        self._fig.patch.set_facecolor('white')
        
        self._ax.set_xlabel('X')
        self._ax.set_ylabel('Y')
        self._ax.set_zlabel('Z')
        
        # Style axis for ORB-SLAM3 light mode
        self._ax.xaxis.label.set_color('black')
        self._ax.yaxis.label.set_color('black')
        self._ax.zaxis.label.set_color('black')
        self._ax.tick_params(colors='black')

        self._map_pts_plot = self._ax.scatter([], [], [], s=1, c='k', marker='.', alpha=0.6)
        self._path_plot, = self._ax.plot([], [], [], color='lime', linewidth=1.5)
        
        self._frustum_col = None

        self._ani = FuncAnimation(self._fig, self._update_plot, interval=100, cache_frame_data=False)

    def _update_plot(self, frame):
        if self._quit:
            return

        with self._lock:
            state = self._slam_state.copy()
            path = list(self._path_pts)

        if not state:
            return

        # Update 3D path
        if path:
            path_arr = np.array(path)
            self._path_plot.set_data(path_arr[:, 0], path_arr[:, 1])
            self._path_plot.set_3d_properties(path_arr[:, 2])

        # Update 3D points
        pts = state.get('map_points', np.zeros((0, 3)))
        if len(pts) > 0:
            self._map_pts_plot._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        # Update Keyframe Frustums
        kf_poses = state.get('kf_poses', [])
        if len(kf_poses) > 0:
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            frustum_lines = []
            scale = 0.5
            for pose in kf_poses:
                pts_cam = np.array([
                    [0, 0, 0],
                    [scale, scale, scale*2],
                    [scale, -scale, scale*2],
                    [-scale, -scale, scale*2],
                    [-scale, scale, scale*2]
                ])
                pts_homo = np.hstack((pts_cam, np.ones((5, 1))))
                pts_world = (pose @ pts_homo.T).T[:, :3]
                
                idx = [(0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1)]
                for i, j in idx:
                    frustum_lines.append([pts_world[i], pts_world[j]])
            
            if self._frustum_col is not None:
                self._frustum_col.remove()
            self._frustum_col = Line3DCollection(frustum_lines, colors='b', linewidths=0.6)
            self._ax.add_collection3d(self._frustum_col)
        
        # Auto scale aspect ratio to keep it looking cubic
        if len(pts) > 0:
            try:
                max_range = np.array([np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])]).max() / 2.0
                mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
                mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
                mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5
                self._ax.set_xlim(mid_x - max_range, mid_x + max_range)
                self._ax.set_ylim(mid_y - max_range, mid_y + max_range)
                self._ax.set_zlim(mid_z - max_range, mid_z + max_range)
            except: pass

    def _draw_camera(self, frame, slam_state, kps, pts1, pts2):
        panel = cv2.resize(frame, (self.cam_w, self.cam_h))
        panel = (panel * 0.72).astype(np.uint8)

        state = slam_state.get('state', TrackingState.INIT)
        pos   = slam_state.get('position', [0, 0, 0])
        fps   = slam_state.get('fps', 0)
        feats = slam_state.get('n_features', 0)
        kf    = slam_state.get('n_keyframes', 0)
        mp    = slam_state.get('n_map_points', 0)

        if pts1 is not None and pts2 is not None:
            for p1, p2 in zip(pts1, pts2):
                x1,y1 = int(p1[0]), int(p1[1])
                x2,y2 = int(p2[0]), int(p2[1])
                mag = np.hypot(x2-x1, y2-y1)
                g = max(0,   int(200 - mag*10))
                r = min(220, int(mag*18))
                cv2.line(panel, (x1,y1), (x2,y2), (0,g,r), 1, cv2.LINE_AA)

        for kp in (kps or []):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(panel, (x,y), 2, CV_GREEN, -1, cv2.LINE_AA)

        col = {
            TrackingState.TRACKING: CV_GREEN,
            TrackingState.LOST:     CV_RED,
            TrackingState.INIT:     CV_AMBER,
            TrackingState.RELOC:    CV_AMBER,
        }.get(state, CV_WHITE)
        badge = f'  {state.value}  '
        cv2.rectangle(panel, (6,6), (6+len(badge)*8, 26), (0,0,0), -1)
        cv2.putText(panel, badge, (8,21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, col, 1, cv2.LINE_AA)

        n_pts = len(self._path_pts)
        dist  = slam_state.get('total_dist', 0.0)
        info  = (f'  FPS:{fps:.0f}  FEAT:{feats}  KF:{kf}  MP:{mp}  '
                 f'PTS:{n_pts}  DIST:{dist:.2f}u  '
                 f'X:{pos[0]:+.2f} Y:{pos[1]:+.2f} Z:{pos[2]:+.2f}')
        cv2.rectangle(panel, (0, self.cam_h-22), (self.cam_w, self.cam_h), (0,0,0), -1)
        cv2.putText(panel, info, (4, self.cam_h-7), cv2.FONT_HERSHEY_SIMPLEX, 0.30, CV_DIM, 1, cv2.LINE_AA)

        cv2.putText(panel, 'Q:quit  R:reset  S:save', (self.cam_w-158, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.30, CV_DIM, 1)

        if state != TrackingState.TRACKING:
            hint = 'Point at textured surface and move SIDEWAYS slowly'
            tw   = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            cx   = (self.cam_w - tw) // 2
            cv2.putText(panel, hint, (cx, self.cam_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CV_AMBER, 1, cv2.LINE_AA)

        return panel