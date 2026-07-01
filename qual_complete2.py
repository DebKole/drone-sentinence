#!/usr/bin/env python3

import time
import sys
import math
import cv2
import numpy as np
from pymavlink import mavutil

# ──────────────────────────────────────────────────────────
# 1. HARDWARE & MAVLINK CONFIG
# ──────────────────────────────────────────────────────────
PORT = '/dev/ttyACM0'
BAUD = 921600
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480

# ──────────────────────────────────────────────────────────
# 2. FLIGHT & RC CONFIG
# ──────────────────────────────────────────────────────────
FLIGHT_MODE     = 'LOITER'
RC_MID          = 1500
THROTTLE_ZERO   = 1000
THROTTLE_CLIMB  = 1620
THROTTLE_HOVER  = 1500
TARGET_ALT_M    = 1.5  # Altitude to begin ArUco search
SETPOINT_HZ     = 10

# ──────────────────────────────────────────────────────────
# 3. ARUCO & PRECISION LANDING CONFIG
# ──────────────────────────────────────────────────────────
ARUCO_DICT      = cv2.aruco.DICT_4X4_50
MARKER_SIZE_M   = 0.30 
# Replace with your actual calibration
CAMERA_MATRIX   = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
DIST_COEFFS     = np.zeros((5, 1))

KP_XY           = 0.35  # Gain for horizontal centering
MAX_VEL_XY      = 0.8   # m/s
DESCENT_RATE    = 0.25  # m/s
LAND_ALT_LIMIT  = 0.4   # Final Landing trigger height (m)
CENTRE_THRESH   = 0.10  # Allowed error in meters before descending

# ──────────────────────────────────────────────────────────
# TELEMETRY & UTILS
# ──────────────────────────────────────────────────────────
_last_rng = None
_last_mode = None

def log_info(msg): print(f"[INFO] {msg}")
def log_warn(msg): print(f"[WARN] {msg}")

def get_rangefinder(mav):
    global _last_rng
    msg = mav.recv_match(type='DISTANCE_SENSOR', blocking=False)
    if msg: _last_rng = msg.current_distance / 100.0
    return _last_rng

def handle_pause(mav):
    """If pilot changes mode on RC, script stops sending overrides."""
    global _last_mode
    msg = mav.recv_match(type='HEARTBEAT', blocking=False)
    if msg: _last_mode = mavutil.mode_string_v10(msg)
    if _last_mode is not None and _last_mode != FLIGHT_MODE:
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 0,0,0,0,0,0,0,0)
        return True
    return False

def send_velocity_ned(mav, vx, vy, vz):
    """Body-frame velocity: vx=fwd, vy=right, vz=down."""
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000_1111_1100_0111, # Velocity only
        0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0
    )

# ──────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────
def main():
    # --- Initialization ---
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    mav.wait_heartbeat()
    log_info("Connected to Pixhawk")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
        cv2.aruco.DetectorParameters()
    )

    # --- Takeoff Phase (using RC Overrides) ---
    input(f"Verify mode is {FLIGHT_MODE}. Press Enter to Arm & Takeoff...")
    
    # Arm
    mav.mav.command_long_send(mav.target_system, mav.target_component,
                             mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0,0,0,0,0,0)
    time.sleep(2)

    log_info("Climbing to Search Altitude...")
    while True:
        if handle_pause(mav): continue
        rng = get_rangefinder(mav)
        if rng and rng >= TARGET_ALT_M:
            break
        # RC Override for climb
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 
                                         RC_MID, RC_MID, THROTTLE_CLIMB, RC_MID, 0,0,0,0)
        time.sleep(0.1)

    # Clean RC overrides before switching to MAVLink velocity commands
    mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 0,0,0,0,0,0,0,0)
    log_info("Search altitude reached. Initiating ArUco detection.")

    # --- Precision Landing Loop ---
    landed = False
    last_seen = time.time()
    cx, cy = FRAME_W / 2.0, FRAME_H / 2.0
    fx = CAMERA_MATRIX[0, 0]

    try:
        while not landed:
            if handle_pause(mav):
                log_warn("Manual Override active...")
                time.sleep(0.5)
                continue

            ret, frame = cap.read()
            if not ret: continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                last_seen = time.time()
                # Pose estimation
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS)
                tvec = tvecs[0][0]
                dist = float(np.linalg.norm(tvec))

                # Logic for offsets
                pts = corners[0][0]
                mx, my = pts[:, 0].mean(), pts[:, 1].mean()
                err_x_m = (mx - cx) * dist / fx
                err_y_m = (my - cy) * dist / fx
                horiz_err = math.hypot(err_x_m, err_y_m)

                # Decisions
                if dist < LAND_ALT_LIMIT and horiz_err < CENTRE_THRESH:
                    log_info("Thresholds met! Sending LAND command.")
                    mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9) # 9 = Land in ArduPilot
                    landed = True
                    break

                # Calculate Velocities
                vel_right = np.clip(KP_XY * err_x_m, -MAX_VEL_XY, MAX_VEL_XY)
                vel_fwd   = np.clip(KP_XY * err_y_m, -MAX_VEL_XY, MAX_VEL_XY)
                vel_down  = DESCENT_RATE if horiz_err < CENTRE_THRESH else 0.0
                
                send_velocity_ned(mav, vel_fwd, vel_right, vel_down)
                print(f"Aligning: Dist={dist:.2f} Err={horiz_err:.2f}", end='\r')

            else:
                # Lost marker - hover
                if time.time() - last_seen > 3.0:
                    log_warn("Marker lost! Hovering...")
                    send_velocity_ned(mav, 0, 0, 0)
            
            cv2.imshow("Landing Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        log_warn("Emergency Abort")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_info("Exiting script.")

if __name__ == "__main__":
    main()
