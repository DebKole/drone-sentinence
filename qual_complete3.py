#!/usr/bin/env python3
"""
drone_integrated_final.py
=========================
Optimized for Jetson Orin Nano (dustynv container) + SJ4000 + Pixhawk.
Features: 6x6 ArUco tracking, extended search window, and precision centering.
"""

import time
import math
import sys
import traceback
import cv2
import numpy as np
from pymavlink import mavutil

# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# -- MAVLink & Port --
PORT = '/dev/ttyACM0'
BAUD = 921600

# -- Camera (SJ4000) --
CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480

# -- ArUco (UPDATED FOR YOUR 6x6 MARKER) --
ARUCO_DICT = cv2.aruco.DICT_6X6_250 
MARKER_SIZE_M = 0.10  # Set to 0.10 if printed at 10cm, 0.30 if 30cm

# -- Camera Intrinsics (SJ4000 Wide FOV) --
CAMERA_MATRIX = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]], dtype=np.float64)
DIST_COEFFS = np.zeros((5, 1))

# -- Control Gains (Tuned for Centering) --
KP_XY = 0.5            # Speed of attraction to marker
MAX_VEL_XY = 0.7       # m/s horizontal cap
DESCENT_RATE_ARUCO = 0.25 
MAX_VEL_Z = 0.4        

# -- Landing & Search Thresholds --
CENTRE_THRESHOLD_M = 0.12    # Must be within 12cm of center to descend
LAND_ALT_ARUCO_M = 0.40      # Switch to LAND mode at this height
LOST_MARKER_TIMEOUT = 20.0   # 20 second "Patience" window to find marker

# -- RC Overrides & Flight Targets --
THROTTLE_ZERO = 1000
THROTTLE_CLIMB = 1620
THROTTLE_HOVER = 1500
THROTTLE_LAND = 1380
TARGET_ALT_M = 2.5           # Higher altitude = wider camera search window
ALT_LANDED_M = 0.15          
FLIGHT_MODE = 'LOITER'
SHOW_VIDEO = False           # Set False for better performance in Docker

# ═══════════════════════════════════════════════════════════════════

_last_rng = None
_last_mode = None

def log(tag, msg):
    print(f"[{tag} {_ts()}] {msg}")

def _ts(): return time.strftime("%H:%M:%S")

def get_rangefinder(mav):
    global _last_rng
    msg = mav.recv_match(type='DISTANCE_SENSOR', blocking=False)
    if msg: _last_rng = msg.current_distance / 100.0
    return _last_rng

def send_velocity_ned(mav, vx, vy, vz):
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000_1111_1100_0111, 0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0
    )

def handle_pause(mav):
    global _last_mode
    msg = mav.recv_match(type='HEARTBEAT', blocking=False)
    if msg: _last_mode = mavutil.mode_string_v10(msg)
    if _last_mode and _last_mode != FLIGHT_MODE:
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 0,0,0,0,0,0,0,0)
        return True
    return False

def aruco_precision_land(mav, cap, detector):
    log("LAND", "Starting Tracking & Centering Phase...")
    cx, cy = FRAME_W / 2.0, FRAME_H / 2.0
    fx = CAMERA_MATRIX[0, 0]
    last_seen = time.time()
    marker_ever_seen = False

    while True:
        if handle_pause(mav):
            send_velocity_ned(mav, 0, 0, 0)
            time.sleep(0.5); continue

        ret, frame = cap.read()
        if not ret: continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            last_seen = time.time()
            marker_ever_seen = True
            
            # Pose Estimation
            _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS)
            tvec = tvecs[0][0]
            dist = float(np.linalg.norm(tvec))

            # Calculate Center of Marker
            pts = corners[0][0]
            mx, my = pts[:, 0].mean(), pts[:, 1].mean()
            
            # Error in Meters
            err_x_m = (mx - cx) * dist / fx
            err_y_m = (my - cy) * dist / fx
            horiz_err = math.hypot(err_x_m, err_y_m)

            # --- Centering Logic ---
            vel_right = np.clip(KP_XY * err_x_m, -MAX_VEL_XY, MAX_VEL_XY)
            vel_fwd   = np.clip(KP_XY * err_y_m, -MAX_VEL_XY, MAX_VEL_XY)
            
            # Only descend if we are directly above the marker
            if horiz_err < CENTRE_THRESHOLD_M:
                vel_down = DESCENT_RATE_ARUCO
                status = "CENTERED - Descending"
            else:
                vel_down = 0.0
                status = "TRACKING - Moving to Center"

            send_velocity_ned(mav, vel_fwd, vel_right, vel_down)
            print(f"[{status}] Alt: {dist:.2f}m Err: {horiz_err:.3f}m", end='\r')

            if dist < LAND_ALT_ARUCO_M and horiz_err < CENTRE_THRESHOLD_M:
                log("LAND", "Directly above target. Switching to LAND mode.")
                mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)
                while get_rangefinder(mav) > ALT_LANDED_M: time.sleep(0.2)
                return True
        else:
            elapsed = time.time() - last_seen
            send_velocity_ned(mav, 0, 0, 0) # Hover while searching
            if elapsed > LOST_MARKER_TIMEOUT:
                log("WARN", "Search Window Expired.")
                return False
            print(f"[SEARCHING] Marker lost for {elapsed:.1f}s...", end='\r')

def main():
    # Setup MAVLink
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    mav.wait_heartbeat()
    log("INIT", "Connected to Pixhawk. Telemetry OK.")

    # Setup Camera (Force V4L2 for Jetson Container)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    # Setup 6x6 Detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    try:
        input(f"[*] Mode is {FLIGHT_MODE}. Press Enter to FLY...")
        
        # Arm & Takeoff
        mav.mav.command_long_send(mav.target_system, mav.target_component, 
                                 mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0,0,0,0,0,0)
        log("FLIGHT", "Armed. Climbing...")
        
        while (get_rangefinder(mav) or 0) < TARGET_ALT_M:
            if handle_pause(mav): continue
            mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 
                                             1500, 1500, THROTTLE_CLIMB, 1500, 0,0,0,0)
            time.sleep(0.1)

        # Transition to Precision Landing
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 0,0,0,0,0,0,0,0)
        
        if not aruco_precision_land(mav, cap, detector):
            log("WARN", "ArUco Failed. Performing Fallback Landing...")
            mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 1500, 1500, THROTTLE_LAND, 1500, 0,0,0,0)
            while (get_rangefinder(mav) or 1) > ALT_LANDED_M: time.sleep(0.1)

        # Disarm
        log("FINISH", "Touchdown. Disarming.")
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 1500, 1500, THROTTLE_ZERO, 1500, 0,0,0,0)
        time.sleep(1)
        mav.mav.command_long_send(mav.target_system, mav.target_component, 
                                 mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0,0,0,0,0,0)

    except KeyboardInterrupt:
        log("EMERGENCY", "Manual Abort!")
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 1500, 1500, THROTTLE_ZERO, 1500, 0,0,0,0)
    finally:
        cap.release()
        mav.close()

if __name__ == "__main__":
    main()
