"""
land_regular_aruco.py
=====================
Autonomous drone landing using a single (regular) ArUco marker.

Flow
----
1. Connect to the flight-controller via MAVLink.
2. Arm & take-off to a user-defined hover altitude (optional – skip if already airborne).
3. Read frames from the downward-facing camera.
4. Detect the ArUco marker; compute its X/Y offset from the camera centre.
5. Send NED velocity commands to null the offset.
6. Once the drone is centred AND close enough to the ground, command LAND mode.

Dependencies
------------
    pip install pymavlink opencv-python opencv-contrib-python numpy

Hardware assumptions
--------------------
- Flight controller running ArduPilot / PX4 and reachable via serial or UDP.
- A downward-facing camera (index 0 by default, change CAMERA_INDEX).
- Camera intrinsics (focal length, principal point) calibrated and filled in below.
"""

import time
import math
import cv2
import numpy as np
from pymavlink import mavutil

# ─────────────────────────────────────────────
#  USER CONFIGURATION
# ─────────────────────────────────────────────
CONNECTION_STRING = "udp:127.0.0.1:14550"   # MAVLink endpoint
CAMERA_INDEX      = 0                         # /dev/video0 or USB index
FRAME_W, FRAME_H  = 640, 480                  # Camera resolution

# ArUco dictionary used when printing / generating the marker
ARUCO_DICT        = cv2.aruco.DICT_4X4_50

# Physical marker side length in metres (needed for pose estimation)
MARKER_SIZE_M     = 0.30                      # 30 cm marker

# Camera intrinsic matrix – replace with your calibration values
# Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
CAMERA_MATRIX = np.array([[500,   0, 320],
                           [  0, 500, 240],
                           [  0,   0,   1]], dtype=np.float64)
DIST_COEFFS   = np.zeros((5, 1), dtype=np.float64)   # assume undistorted

# Control gains  (tune for your airframe)
KP_XY         = 0.4    # proportional gain for horizontal error (m/s per metre)
MAX_VEL_XY    = 1.0    # m/s horizontal limit
DESCENT_RATE  = 0.3    # m/s downward when centred
MAX_VEL_Z     = 0.5    # m/s descent limit

# Thresholds
CENTRE_THRESHOLD_M   = 0.08   # horizontal error below this → start descending
LAND_ALTITUDE_M      = 0.40   # switch to LAND mode when estimated dist < this
LOST_MARKER_TIMEOUT  = 3.0    # seconds without detection before hovering
# ─────────────────────────────────────────────


def connect_mavlink(connection_string: str):
    """Connect and wait for heartbeat."""
    print(f"[MAVLink] Connecting to {connection_string} …")
    mav = mavutil.mavlink_connection(connection_string)
    mav.wait_heartbeat()
    print(f"[MAVLink] Heartbeat received  (system {mav.target_system}, "
          f"component {mav.target_component})")
    return mav


def set_mode(mav, mode_name: str):
    """Change flight mode by name (ArduPilot style)."""
    mode_id = mav.mode_mapping().get(mode_name)
    if mode_id is None:
        raise ValueError(f"Unknown mode: {mode_name}")
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )
    print(f"[MAVLink] Mode → {mode_name}")


def arm(mav):
    """Arm the vehicle."""
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0,
    )
    mav.motors_armed_wait()
    print("[MAVLink] Armed")


def takeoff(mav, altitude_m: float):
    """Command takeoff to target altitude (metres, relative)."""
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude_m,
    )
    print(f"[MAVLink] Takeoff → {altitude_m} m")
    time.sleep(altitude_m / 1.5 + 2)  # rough wait; replace with telemetry check


def send_velocity_ned(mav, vx: float, vy: float, vz: float):
    """
    Send a body-frame velocity setpoint via SET_POSITION_TARGET_LOCAL_NED.
    vx = forward (North), vy = right (East), vz = down (positive = descend).
    """
    mav.mav.set_position_target_local_ned_send(
        0,                                     # time_boot_ms (ignored)
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000_1111_1100_0111,                 # type mask: velocity only
        0, 0, 0,                               # position (ignored)
        vx, vy, vz,                            # velocity (m/s)
        0, 0, 0,                               # acceleration (ignored)
        0, 0,                                  # yaw, yaw_rate (ignored)
    )


def land_mode(mav):
    """Switch to LAND flight mode."""
    set_mode(mav, "LAND")
    print("[MAVLink] LAND mode engaged – descending to touchdown.")


def pixel_error_to_metres(err_px: float, focal_length_px: float, dist_m: float) -> float:
    """Convert pixel offset to real-world offset at a known distance."""
    return err_px * dist_m / focal_length_px


def estimate_distance(tvec) -> float:
    """Euclidean distance from camera to marker centre (metres)."""
    return float(np.linalg.norm(tvec))


def main():
    # ── MAVLink setup ──────────────────────────────────────────────────────────
    mav = connect_mavlink(CONNECTION_STRING)

    # If drone is on the ground, arm & take off; comment out if already airborne
    set_mode(mav, "GUIDED")
    arm(mav)
    takeoff(mav, altitude_m=5.0)

    # ── Camera & ArUco setup ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    cx, cy = FRAME_W / 2.0, FRAME_H / 2.0   # image centre
    fx     = CAMERA_MATRIX[0, 0]             # focal length (px)

    last_seen = time.time()
    landed    = False

    print("[Landing] Starting precision landing loop …")

    try:
        while not landed:
            ret, frame = cap.read()
            if not ret:
                print("[Camera] Frame read failed – skipping")
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                last_seen = time.time()

                # Use first detected marker
                corner = corners[0]
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corner], MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS
                )
                tvec = tvec[0][0]  # shape (3,)

                # Marker centre in pixel space
                pts    = corner[0]
                mx, my = pts[:, 0].mean(), pts[:, 1].mean()

                # Pixel offsets from image centre
                err_x_px = mx - cx   # + → marker is right of centre
                err_y_px = my - cy   # + → marker is below centre

                dist = estimate_distance(tvec)

                # Convert pixel error to metres
                err_x_m = pixel_error_to_metres(err_x_px, fx, dist)
                err_y_m = pixel_error_to_metres(err_y_px, fx, dist)

                horiz_err = math.hypot(err_x_m, err_y_m)

                print(f"[Detection] dist={dist:.2f}m  err_x={err_x_m:.3f}m  "
                      f"err_y={err_y_m:.3f}m  horiz={horiz_err:.3f}m")

                if dist < LAND_ALTITUDE_M and horiz_err < CENTRE_THRESHOLD_M:
                    # Close enough and centred → engage LAND mode
                    land_mode(mav)
                    landed = True
                    break

                # Proportional velocity commands
                # Camera frame: x=right, y=down → body frame: x=fwd, y=right
                # NED body offset: vx=forward, vy=right
                vel_right   = np.clip( KP_XY * err_x_m, -MAX_VEL_XY, MAX_VEL_XY)
                vel_forward = np.clip( KP_XY * err_y_m, -MAX_VEL_XY, MAX_VEL_XY)

                if horiz_err < CENTRE_THRESHOLD_M:
                    vel_down = np.clip(DESCENT_RATE, 0, MAX_VEL_Z)
                else:
                    vel_down = 0.0   # hover while correcting position

                send_velocity_ned(mav, vel_forward, vel_right, vel_down)

                # Debug overlay
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.circle(frame, (int(mx), int(my)), 5, (0, 255, 0), -1)
                cv2.line(frame, (int(cx), int(cy)), (int(mx), int(my)), (0, 0, 255), 2)

            else:
                # Marker not visible
                elapsed = time.time() - last_seen
                print(f"[Detection] Marker not found ({elapsed:.1f}s since last seen)")
                send_velocity_ned(mav, 0, 0, 0)   # hover in place

            cv2.imshow("Precision Landing – Regular ArUco", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[User] Quit key pressed – aborting landing")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Landing] Script exiting.")


if __name__ == "__main__":
    main()

