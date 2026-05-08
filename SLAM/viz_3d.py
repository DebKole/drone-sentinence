import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def main():
    p = argparse.ArgumentParser(description="3D Map Visualizer using Matplotlib")
    p.add_argument('--map-dir', type=str, default='slam_map_save', help='Path to saved map directory')
    args = p.parse_args()

    map_dir = args.map_dir
    if not os.path.exists(map_dir):
        print(f"[ERROR] Map directory '{map_dir}' not found. Run SLAM first to generate a map.")
        return

    pts_path = os.path.join(map_dir, 'map_points.npy')
    colors_path = os.path.join(map_dir, 'map_colors.npy')
    traj_path = os.path.join(map_dir, 'trajectory.npy')

    if not os.path.exists(pts_path):
        print("[ERROR] Map points not found in the directory.")
        return

    pts = np.load(pts_path)
    print(f"Loaded {len(pts)} map points.")

    try:
        colors = np.load(colors_path)
    except FileNotFoundError:
        print("[WARN] No colors found, using default gray.")
        colors = np.full((len(pts), 3), 200, dtype=np.uint8)

    try:
        traj = np.load(traj_path)
    except FileNotFoundError:
        traj = np.empty((0, 3))

    try:
        kf_poses = np.load(os.path.join(map_dir, 'keyframe_poses.npy'))
    except FileNotFoundError:
        kf_poses = np.empty((0, 4, 4))

    # Build Matplotlib 3D Figure (ORB-SLAM3 Style: White background)
    fig = plt.figure(figsize=(10, 8), facecolor='white', num='ORB-SLAM3: Map Viewer')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    # ── 1. Plot Map Points (Black) ──
    if len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='k', s=1, marker='.', alpha=0.6)

    # ── 2. Plot Trajectory (Green) ──
    if len(traj) > 0:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='lime', linewidth=2.0)

    # ── 3. Plot Keyframe Frustums (Blue) ──
    if len(kf_poses) > 0:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        frustum_lines = []
        scale = 0.5  # Size of the camera pyramid
        for pose in kf_poses:
            # Pyramid corners in camera frame (looking down +Z)
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
                
        lc = Line3DCollection(frustum_lines, colors='b', linewidths=0.6)
        ax.add_collection3d(lc)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # ORB-SLAM3 uses light grids
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.tick_params(colors='black')

    # Try to set equal aspect ratio
    try:
        max_range = np.array([np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])]).max() / 2.0
        mid_x = (pts[:, 0].max() + pts[:, 0].min()) * 0.5
        mid_y = (pts[:, 1].max() + pts[:, 1].min()) * 0.5
        mid_z = (pts[:, 2].max() + pts[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    except:
        pass

    print("[VIZ] Opening 3D Map Window...")
    plt.show()

if __name__ == '__main__':
    main()
