#!/usr/bin/env python3
"""
drone_integrated.py
===================
Integrated autonomous flight + ArUco precision landing script.

Hardware
--------
  - Jetson Orin Nano Super  →  runs this script
  - Pixhawk (ArduPilot)     →  connected via /dev/ttyACM0 @ 921600
  - SJ4000 camera           →  downward-facing, USB index 0
  - MTF-P01 optical flow + LiDAR module

Flight sequence
---------------
  1. Connect to Pixhawk, verify telemetry + rangefinder
  2. Arm in LOITER mode
  3. Spin up → climb to TARGET_ALT_M → hover
  4. Start ArUco detection loop:
       - centre over marker using proportional velocity commands
       - descend slowly once centred
       - when close + centred → engage MAVLink LAND mode
  5. If marker is never found / times out → fall back to throttle-based descent
  6. Disarm after landing confirmed

Manual override
---------------
  - Pilot switching out of LOITER at any time pauses RC overrides immediately
  - Script resumes automatically when pilot returns to LOITER
  - Ctrl-C triggers emergency stop (throttle cut + disarm)

Dependencies
------------
  pip install pymavlink opencv-python opencv-contrib-python numpy
"""

import time
import math
import sys
import threading
import traceback

import cv2
import numpy as np
from pymavlink import mavutil


# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  – tune everything here
# ═══════════════════════════════════════════════════════════════════

# ── MAVLink ──────────────────────────────────────────────────────
PORT  = '/dev/ttyACM0'
BAUD  = 921600

# ── Camera (SJ4000 on Jetson) ─────────────────────────────────────
CAMERA_INDEX      = 0
FRAME_W, FRAME_H  = 640, 480

# ── ArUco ────────────────────────────────────────────────────────
ARUCO_DICT    = cv2.aruco.DICT_6X6_250
MARKER_SIZE_M = 0.10        # physical side length of printed marker (metres)

# Camera intrinsics – replace with values from your calibration
# For SJ4000 at 640×480 these are rough starting values
CAMERA_MATRIX = np.array([[500,   0, 320],
                           [  0, 500, 240],
                           [  0,   0,   1]], dtype=np.float64)
DIST_COEFFS   = np.zeros((5, 1), dtype=np.float64)

# ── Control gains ────────────────────────────────────────────────
KP_XY              = 0.4    # P-gain: horizontal error → velocity (m/s per metre)
MAX_VEL_XY         = 1.0    # m/s horizontal clamp
DESCENT_RATE_ARUCO = 0.25   # m/s downward while centred over marker
MAX_VEL_Z          = 0.5    # m/s descent clamp

# ── Landing thresholds ───────────────────────────────────────────
CENTRE_THRESHOLD_M  = 0.08  # horizontal error below this → begin descent
LAND_ALT_ARUCO_M    = 0.40  # switch to LAND mode when dist-to-marker < this
LOST_MARKER_TIMEOUT = 15.0   # seconds without detection → hover, then fallback

# ── RC override values ───────────────────────────────────────────
RC_MID          = 1500
RC_MIN          = 1000
RC_MAX          = 1900
THROTTLE_ZERO   = 1000
THROTTLE_SPINUP = 1200
THROTTLE_HOVER  = 1500
THROTTLE_CLIMB  = 1650
THROTTLE_LAND   = 1380

# ── Flight timing ────────────────────────────────────────────────
SPINUP_TIME   = 3.0   # seconds
CLIMB_TIME    = 6.0   # seconds (max; exits early via rangefinder)
HOVER_TIME    = 300.0   # seconds of stable hover before landing phase
DESCEND_TIME  = 8.0   # seconds max for throttle-based fallback descent
SETPOINT_HZ   = 10    # RC override send rate

# ── Altitude targets ─────────────────────────────────────────────
TARGET_ALT_M  = 2.0   # hover altitude (metres AGL, from rangefinder)
ALT_LANDED_M  = 0.20  # rangefinder threshold to confirm touchdown

# ── Flight mode ──────────────────────────────────────────────────
FLIGHT_MODE = 'LOITER'   # change to 'ALT_HOLD' if LOITER unavailable

# ── Show live camera window (set False on headless Jetson) ───────
SHOW_VIDEO = False

# ═══════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────
#  Shared state (updated by telemetry helpers)
# ───────────────────────────────────────────────────────────────────
_last_alt  = None   # baro alt from VFR_HUD
_last_rng  = None   # rangefinder from DISTANCE_SENSOR
_last_mode = None   # flight mode string


# ═══════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════

def _ts():
    return time.strftime("%H:%M:%S")

def log_info(msg):     print(f"  [INFO     {_ts()}] {msg}")
def log_warn(msg):     print(f"  [WARN     {_ts()}] {msg}")
def log_error(msg):    print(f"  [ERROR    {_ts()}] {msg}", file=sys.stderr)
def log_override(msg): print(f"\n  [OVERRIDE {_ts()}] {msg}")
def log_land(msg):     print(f"  [LAND     {_ts()}] {msg}")


# ═══════════════════════════════════════════════════════════════════
#  MAVLink helpers
# ═══════════════════════════════════════════════════════════════════

def safe_recv(mav, msg_type, blocking=False, timeout=1):
    retries = 10 if blocking else 3
    for _ in range(retries):
        try:
            return mav.recv_match(type=msg_type, blocking=blocking, timeout=timeout)
        except TypeError:
            continue
        except Exception as e:
            log_error(f"safe_recv({msg_type}): {e}")
            return None
    return None


def request_streams(mav):
    log_info("Requesting telemetry streams …")
    try:
        for stream_id, rate in [
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 10),
            (mavutil.mavlink.MAV_DATA_STREAM_ALL,    4),
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA3, 10),
        ]:
            mav.mav.request_data_stream_send(
                mav.target_system, mav.target_component, stream_id, rate, 1
            )
        time.sleep(0.5)
        log_info("Stream requests sent")
    except Exception as e:
        log_error(f"Stream request failed: {e}")


def connect(port, baud):
    print(f"\n[*] Connecting to Pixhawk on {port} @ {baud} baud …")
    try:
        mav = mavutil.mavlink_connection(port, baud=baud)
        mav.wait_heartbeat()
        log_info(f"Connected  system={mav.target_system}  component={mav.target_component}")
        request_streams(mav)
        return mav
    except Exception as e:
        log_error(f"Connection failed: {e}")
        sys.exit(1)


def verify_telemetry(mav):
    global _last_alt
    log_info("Verifying VFR_HUD stream …")
    for attempt in range(20):
        msg = safe_recv(mav, 'VFR_HUD', blocking=True, timeout=1)
        if msg:
            _last_alt = msg.alt
            log_info(f"VFR_HUD OK — alt={msg.alt:.2f} m")
            return True
        log_warn(f"No VFR_HUD ({attempt+1}/20) …")
    return False


def verify_rangefinder(mav):
    global _last_rng
    log_info("Verifying DISTANCE_SENSOR stream …")
    for attempt in range(20):
        msg = safe_recv(mav, 'DISTANCE_SENSOR', blocking=True, timeout=1)
        if msg:
            _last_rng = msg.current_distance / 100.0
            log_info(f"Rangefinder OK — {_last_rng:.2f} m")
            return True
        log_warn(f"No DISTANCE_SENSOR ({attempt+1}/20) …")
    log_warn("Rangefinder not detected — will fall back to baro altitude")
    return False


def update_flight_mode(mav, blocking=False):
    global _last_mode
    if blocking:
        for _ in range(20):
            msg = safe_recv(mav, 'HEARTBEAT', blocking=True, timeout=1)
            if msg and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                _last_mode = mavutil.mode_string_v10(msg)
                return _last_mode
    while True:
        msg = safe_recv(mav, 'HEARTBEAT', blocking=False)
        if msg is None:
            break
        if msg.type != mavutil.mavlink.MAV_TYPE_GCS:
            _last_mode = mavutil.mode_string_v10(msg)
    if _last_mode is None and 'HEARTBEAT' in mav.messages:
        msg = mav.messages['HEARTBEAT']
        if msg.type != mavutil.mavlink.MAV_TYPE_GCS:
            _last_mode = mavutil.mode_string_v10(msg)
    return _last_mode


def is_armed(mav):
    if 'HEARTBEAT' in mav.messages:
        msg = mav.messages['HEARTBEAT']
        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return False


def get_altitude(mav):
    """Barometric altitude from VFR_HUD (metres)."""
    global _last_alt
    while True:
        msg = safe_recv(mav, 'VFR_HUD', blocking=False)
        if msg is None:
            break
        _last_alt = msg.alt
    return _last_alt


def get_rangefinder(mav):
    """
    Height AGL from MTF-P01 LiDAR via DISTANCE_SENSOR (metres).
    Falls back to barometric altitude if no rangefinder data available.
    """
    global _last_rng
    while True:
        msg = safe_recv(mav, 'DISTANCE_SENSOR', blocking=False)
        if msg is None:
            break
        _last_rng = msg.current_distance / 100.0
    return _last_rng if _last_rng is not None else get_altitude(mav)


# ═══════════════════════════════════════════════════════════════════
#  RC override helpers
# ═══════════════════════════════════════════════════════════════════

def send_rc_override(mav, throttle, roll=RC_MID, pitch=RC_MID, yaw=RC_MID):
    try:
        mav.mav.rc_channels_override_send(
            mav.target_system, mav.target_component,
            roll, pitch, throttle, yaw, 0, 0, 0, 0
        )
    except Exception as e:
        log_error(f"send_rc_override: {e}")


def clear_rc_override(mav):
    try:
        mav.mav.rc_channels_override_send(
            mav.target_system, mav.target_component,
            0, 0, 0, 0, 0, 0, 0, 0
        )
    except Exception as e:
        log_error(f"clear_rc_override: {e}")


# ═══════════════════════════════════════════════════════════════════
#  MAVLink velocity command (used by ArUco landing)
# ═══════════════════════════════════════════════════════════════════

def send_velocity_ned(mav, vx: float, vy: float, vz: float):
    """
    Body-frame NED velocity setpoint.
    vx = forward, vy = right, vz = down (positive = descend).
    Uses SET_POSITION_TARGET_LOCAL_NED with velocity-only type mask.
    """
    mav.mav.set_position_target_local_ned_send(
        0,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000_1111_1100_0111,          # velocity only
        0, 0, 0,                        # position (ignored)
        vx, vy, vz,                     # velocity m/s
        0, 0, 0,                        # acceleration (ignored)
        0, 0,                           # yaw, yaw_rate (ignored)
    )


def set_flight_mode(mav, mode_name: str):
    """Change flight mode by name (ArduPilot)."""
    mode_id = mav.mode_mapping().get(mode_name)
    if mode_id is None:
        raise ValueError(f"Unknown mode: {mode_name}")
    mav.mav.set_mode_send(
        mav.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
    )
    log_info(f"Mode → {mode_name}")


# ═══════════════════════════════════════════════════════════════════
#  Manual-override pause logic
# ═══════════════════════════════════════════════════════════════════

def handle_pause(mav):
    """
    If the pilot has switched out of FLIGHT_MODE, release RC overrides
    and block until they return.  Returns True if a pause occurred.
    """
    current_mode = update_flight_mode(mav, blocking=False)
    if current_mode is not None and current_mode != FLIGHT_MODE:
        clear_rc_override(mav)
        log_override(f"Mode changed to {current_mode} — script PAUSED, pilot has control.")
        while True:
            time.sleep(0.1)
            mode = update_flight_mode(mav, blocking=False)
            if mode == FLIGHT_MODE:
                break
        log_override(f"Returned to {FLIGHT_MODE} — script RESUMING in 1 s …")
        time.sleep(1.0)
        return True
    return False


# ═══════════════════════════════════════════════════════════════════
#  Arm / disarm
# ═══════════════════════════════════════════════════════════════════

def arm(mav):
    print("\n[*] Arming …")
    mode = update_flight_mode(mav, blocking=True)
    log_info(f"Current mode: {mode}")
    if FLIGHT_MODE not in str(mode).upper().replace('_', ''):
        log_error(f"Expected {FLIGHT_MODE}, got {mode}")
        sys.exit(1)
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    for _ in range(20):
        update_flight_mode(mav, blocking=False)
        if is_armed(mav):
            log_info("ARMED successfully")
            return
        time.sleep(0.5)
    log_error("Arming failed — check Mission Planner for pre-arm errors")
    sys.exit(1)


def disarm(mav):
    print("\n[*] Disarming …")
    for _ in range(10):
        send_rc_override(mav, throttle=THROTTLE_ZERO)
        time.sleep(0.05)
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 0, 0, 0, 0, 0, 0, 0
    )
    for _ in range(20):
        update_flight_mode(mav, blocking=False)
        if not is_armed(mav):
            log_info("DISARMED successfully")
            return
        time.sleep(0.5)
    log_warn("Disarm not confirmed — use RC kill-switch if needed")


def emergency_stop(mav):
    print("\n[!] EMERGENCY STOP — cutting throttle immediately")
    for _ in range(20):
        send_rc_override(mav, throttle=THROTTLE_ZERO)
        time.sleep(0.05)
    disarm(mav)


# ═══════════════════════════════════════════════════════════════════
#  Throttle ramp / hold (used for climb, spinup, fallback descent)
# ═══════════════════════════════════════════════════════════════════

def ramp_throttle(mav, start, end, duration, label):
    print(f"\n[*] {label}")
    interval = 1.0 / SETPOINT_HZ
    steps    = max(1, int(duration * SETPOINT_HZ))
    i = 0
    while i <= steps:
        if handle_pause(mav):
            continue
        throttle = int(start + (end - start) * (i / steps))
        send_rc_override(mav, throttle=throttle)
        rng = get_rangefinder(mav)
        rng_str = f"{rng:.2f}m" if rng is not None else "?.??m"
        print(f"    throttle={throttle} | rng={rng_str}", end='\r')
        time.sleep(interval)
        i += 1
    print()


def hold_throttle(mav, throttle, duration, label, exit_condition=None):
    print(f"\n[*] {label}")
    interval   = 1.0 / SETPOINT_HZ
    iterations = max(1, int(duration * SETPOINT_HZ))
    i = 0
    while i < iterations:
        if handle_pause(mav):
            continue
        send_rc_override(mav, throttle=throttle)
        rng = get_rangefinder(mav)
        remaining = duration - (i * interval)
        rng_str = f"{rng:.2f}m" if rng is not None else "?.??m"
        print(f"    throttle={throttle} | rng={rng_str} | {remaining:.1f}s left", end='\r')
        if exit_condition and rng is not None and exit_condition(rng):
            print(f"\n[+] Exit condition met at {rng_str}")
            return
        time.sleep(interval)
        i += 1
    print()


# ═══════════════════════════════════════════════════════════════════
#  ArUco geometry helpers
# ═══════════════════════════════════════════════════════════════════

def pixel_error_to_metres(err_px: float, focal_length_px: float, dist_m: float) -> float:
    """Convert a pixel offset to a real-world offset at a known distance."""
    return err_px * dist_m / focal_length_px


def estimate_distance(tvec) -> float:
    """Euclidean distance from camera origin to marker centre (metres)."""
    return float(np.linalg.norm(tvec))


# ═══════════════════════════════════════════════════════════════════
#  ArUco precision landing
# ═══════════════════════════════════════════════════════════════════

def aruco_precision_land(mav, cap, detector):
    """
    Precision landing loop using a downward-facing ArUco marker.

    Strategy
    --------
    - Detect marker → compute horizontal offset in metres
    - Send NED velocity commands to null the offset
    - Descend slowly once centred (horiz error < CENTRE_THRESHOLD_M)
    - Engage MAVLink LAND mode when close enough AND centred
    - If marker lost for > LOST_MARKER_TIMEOUT → hover, then fallback

    Returns
    -------
    True  – landed via ArUco
    False – marker never found / timed out → caller should use fallback
    """
    log_land("Starting ArUco precision landing …")

    cx, cy = FRAME_W / 2.0, FRAME_H / 2.0
    fx     = CAMERA_MATRIX[0, 0]

    last_seen        = time.time()
    marker_ever_seen = False

    while True:
        # ── Manual override check ───────────────────────────────────
        # During NED velocity phase we don't send RC overrides,
        # but we still respect pilot mode switches.
        current_mode = update_flight_mode(mav, blocking=False)
        if current_mode is not None and current_mode != FLIGHT_MODE:
            # Pilot took over – stop sending velocity commands
            send_velocity_ned(mav, 0, 0, 0)
            log_override(f"Mode changed to {current_mode} — ArUco loop PAUSED.")
            while True:
                time.sleep(0.1)
                mode = update_flight_mode(mav, blocking=False)
                if mode == FLIGHT_MODE:
                    break
            log_override(f"Returned to {FLIGHT_MODE} — ArUco loop RESUMING in 1 s …")
            time.sleep(1.0)
            last_seen = time.time()   # reset timeout after pause
            continue

        # ── Camera frame ────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret:
            log_warn("Camera frame read failed — skipping")
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            last_seen        = time.time()
            marker_ever_seen = True

            # Pose estimation on first detected marker
            corner = corners[0]
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corner], MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS
            )
            tvec = tvec[0][0]   # shape (3,)

            # Marker pixel centre
            pts    = corner[0]
            mx, my = pts[:, 0].mean(), pts[:, 1].mean()

            # Pixel error from image centre
            err_x_px = mx - cx   # + = marker right of centre
            err_y_px = my - cy   # + = marker below centre

            dist = estimate_distance(tvec)

            # Convert to metres
            err_x_m = pixel_error_to_metres(err_x_px, fx, dist)
            err_y_m = pixel_error_to_metres(err_y_px, fx, dist)
            horiz_err = math.hypot(err_x_m, err_y_m)

            log_land(f"dist={dist:.2f}m  err_x={err_x_m:+.3f}m  "
                     f"err_y={err_y_m:+.3f}m  horiz={horiz_err:.3f}m")

            # ── Land condition ──────────────────────────────────────
            if dist < LAND_ALT_ARUCO_M and horiz_err < CENTRE_THRESHOLD_M:
                send_velocity_ned(mav, 0, 0, 0)
                log_land("Centred and close — engaging MAVLink LAND mode")
                set_flight_mode(mav, "LAND")
                # Wait for rangefinder to confirm touchdown
                log_land("Waiting for touchdown confirmation …")
                while True:
                    rng = get_rangefinder(mav)
                    if rng is not None and rng <= ALT_LANDED_M:
                        log_land(f"Touchdown confirmed at {rng:.2f} m")
                        break
                    time.sleep(0.2)
                return True

            # ── Velocity commands ────────────────────────────────────
            # Camera x=right → body vy (east/right)
            # Camera y=down  → body vx (forward/north) – needs sign flip
            # Note: err_y_px positive means marker is south of centre in
            # a downward-facing camera, so we fly forward (positive vx).
            vel_right   = np.clip( KP_XY * err_x_m, -MAX_VEL_XY, MAX_VEL_XY)
            vel_forward = np.clip( KP_XY * err_y_m, -MAX_VEL_XY, MAX_VEL_XY)
            vel_down    = np.clip(DESCENT_RATE_ARUCO, 0, MAX_VEL_Z) \
                          if horiz_err < CENTRE_THRESHOLD_M else 0.0

            send_velocity_ned(mav, vel_forward, vel_right, vel_down)

            # ── Debug overlay ────────────────────────────────────────
            if SHOW_VIDEO:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.circle(frame, (int(mx), int(my)), 6, (0, 255, 0), -1)
                cv2.line(frame, (int(cx), int(cy)), (int(mx), int(my)), (0, 0, 255), 2)
                cv2.putText(frame, f"dist={dist:.2f}m  err={horiz_err:.2f}m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        else:
            # Marker not visible
            elapsed = time.time() - last_seen
            send_velocity_ned(mav, 0, 0, 0)   # hover in place

            if not marker_ever_seen and elapsed > LOST_MARKER_TIMEOUT:
                log_warn("Marker never detected — falling back to throttle descent")
                return False

            if marker_ever_seen and elapsed > LOST_MARKER_TIMEOUT:
                log_warn(f"Marker lost for {elapsed:.1f} s — hovering, then fallback")
                return False

            log_land(f"Marker not found  ({elapsed:.1f}s since last seen)")

        if SHOW_VIDEO:
            cv2.imshow("Precision Landing — ArUco", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log_warn("User pressed Q — aborting ArUco landing")
                send_velocity_ned(mav, 0, 0, 0)
                return False

    # Should not reach here
    return False


# ═══════════════════════════════════════════════════════════════════
#  Throttle-based fallback landing
# ═══════════════════════════════════════════════════════════════════

def fallback_throttle_land(mav):
    """
    Simple throttle-based descent when ArUco landing is not available.
    Mirrors the original flight script's descent sequence.
    """
    log_land("Fallback: throttle-based descent …")
    ramp_throttle(mav, THROTTLE_HOVER, THROTTLE_LAND, 3.0, "Reducing throttle for descent")
    hold_throttle(mav, THROTTLE_LAND, DESCEND_TIME, "Descending (fallback)",
                  exit_condition=lambda r: r <= ALT_LANDED_M)
    ramp_throttle(mav, THROTTLE_LAND, THROTTLE_ZERO, 1.0, "Cutting throttle")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Connect & verify ──────────────────────────────────────────
    mav = connect(PORT, BAUD)

    if not verify_telemetry(mav):
        mav.close()
        sys.exit(1)

    verify_rangefinder(mav)   # non-fatal

    # ── Pre-flight summary ────────────────────────────────────────
    print("\n[*] Pre-flight checks …")
    mode = update_flight_mode(mav, blocking=True)
    alt  = get_altitude(mav)
    rng  = get_rangefinder(mav)
    log_info(f"Mode            : {mode}")
    log_info(f"Baro altitude   : {alt:.2f}m" if alt is not None else "Baro altitude   : unknown")
    log_info(f"Rangefinder     : {rng:.2f}m" if rng is not None else "Rangefinder     : not available")
    log_info(f"Target altitude : {TARGET_ALT_M:.2f}m")
    log_info(f"Hover throttle  : {THROTTLE_HOVER}")
    log_info(f"ArUco dict      : DICT_4X4_50  |  marker size: {MARKER_SIZE_M*100:.0f} cm")
    log_info(f"Camera index    : {CAMERA_INDEX}  ({FRAME_W}×{FRAME_H})")

    input(f"\n[*] Verify mode is {FLIGHT_MODE}. Press ENTER to arm & fly (Ctrl+C to abort) … ")

    # ── Camera setup ──────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        log_error(f"Cannot open camera index {CAMERA_INDEX}")
        mav.close()
        sys.exit(1)
    log_info("Camera opened successfully")

    # ── ArUco detector setup ──────────────────────────────────────
    aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    try:
        # ── Arm ───────────────────────────────────────────────────
        arm(mav)
        time.sleep(1)

        # ── Spin up → climb ───────────────────────────────────────
        ramp_throttle(mav, THROTTLE_ZERO, THROTTLE_SPINUP, SPINUP_TIME,
                      "Spinning up motors")
        ramp_throttle(mav, THROTTLE_SPINUP, THROTTLE_CLIMB, CLIMB_TIME,
                      "Climbing")
        hold_throttle(mav, THROTTLE_CLIMB, CLIMB_TIME,
                      f"Climbing to {TARGET_ALT_M:.1f}m (rangefinder)",
                      exit_condition=lambda r: r >= TARGET_ALT_M)

        # ── Hover ─────────────────────────────────────────────────
        hold_throttle(mav, THROTTLE_HOVER, HOVER_TIME,
                      "Hovering — optical flow active")

        # ── Precision landing (ArUco) with throttle fallback ──────
        aruco_success = aruco_precision_land(mav, cap, detector)

        if not aruco_success:
            log_warn("ArUco landing failed or timed out — using throttle fallback")
            fallback_throttle_land(mav)
        else:
            log_land("ArUco precision landing complete ✓")

        # ── Disarm ────────────────────────────────────────────────
        time.sleep(1)
        disarm(mav)
        log_info("Flight complete ✓")

    except KeyboardInterrupt:
        log_warn("Ctrl-C received")
        emergency_stop(mav)

    except Exception as e:
        log_error(f"Unhandled exception: {e}")
        log_error(traceback.format_exc())
        emergency_stop(mav)

    finally:
        cap.release()
        if SHOW_VIDEO:
            cv2.destroyAllWindows()
        mav.close()
        print("[*] Script exited cleanly.")


if __name__ == "__main__":
    main()
