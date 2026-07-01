"""
land_nested_aruco.py
====================
Autonomous drone landing using a NESTED ArUco marker.

Outer (large) marker  → lateral (X/Y) position control
Inner (small) marker  → yaw alignment

Flow
----
1. Connect via MAVLink.
2. Arm & take-off (skip if already airborne).
3. Detect OUTER marker → compute X/Y offset → send NED velocity commands.
4. Simultaneously detect INNER marker → compute its heading relative to the
   outer marker → command yaw to align with it.
5. Once centred, aligned, AND below LAND_ALTITUDE_M → engage LAND mode.

Dictionary assumptions
----------------------
Both markers use the SAME ArUco dictionary but DIFFERENT IDs:
    OUTER_MARKER_ID – large marker, used for position
    INNER_MARKER_ID – small nested marker, used for yaw

Adjust these IDs to match the markers you generated / printed.

Dependencies
------------
    pip install pymavlink opencv-python opencv-contrib-python numpy
"""

import time
import math
import cv2
import numpy as np
from pymavlink import mavutil

# ─────────────────────────────────────────────
#  USER CONFIGURATION
# ─────────────────────────────────────────────
CONNECTION_STRING = "udp:127.0.0.1:14550"

CAMERA_INDEX       = 0
FRAME_W, FRAME_H   = 640, 480

ARUCO_DICT         = cv2.aruco.DICT_4X4_50

OUTER_MARKER_ID    = 0      # ID of the large (outer) marker
INNER_MARKER_ID    = 1      # ID of the small (inner/nested) marker
OUTER_MARKER_SIZE_M = 0.30  # physical side length of outer marker (metres)
INNER_MARKER_SIZE_M = 0.06  # physical side length of inner marker (metres)

# Camera intrinsics – replace with calibrated values
CAMERA_MATRIX = np.array([[500,   0, 320],
                           [  0, 500, 240],
                           [  0,   0,   1]], dtype=np.float64)
DIST_COEFFS   = np.zeros((5, 1), dtype=np.float64)

# ── Control gains ──────────────────────────────────────────────────────────────
KP_XY             = 0.4    # position gain (m/s per metre error)
KP_YAW            = 1.0    # yaw rate gain (rad/s per radian error)
MAX_VEL_XY        = 1.0    # m/s horizontal limit
MAX_YAW_RATE      = 30.0   # deg/s yaw-rate limit
DESCENT_RATE      = 0.3    # m/s when centred & aligned

# ── Thresholds ─────────────────────────────────────────────────────────────────
CENTRE_THRESHOLD_M  = 0.08  # metres – horizontal error to start descending
YAW_THRESHOLD_DEG   = 5.0   # degrees – yaw error considered "aligned"
LAND_ALTITUDE_M     = 0.40  # metres – switch to LAND mode below this distance

# Desired yaw offset between inner and outer marker.
# 0 means the inner marker's top edge should point in the same direction
# as the outer marker's top edge (i.e. no deliberate rotation).
DESIRED_YAW_OFFSET_DEG = 0.0
# ─────────────────────────────────────────────


# ══════════════════════════════════════════════
#  MAVLink helpers
# ══════════════════════════════════════════════

def connect_mavlink(connection_string: str):
    print(f"[MAVLink] Connecting to {connection_string} …")
    mav = mavutil.mavlink_connection(connection_string)
    mav.wait_heartbeat()
    print(f"[MAVLink] Heartbeat from system {mav.target_system}")
    return mav


def set_mode(mav, mode_name: str):
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
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0,
    )
    mav.motors_armed_wait()
    print("[MAVLink] Armed")


def takeoff(mav, altitude_m: float):
    mav.mav.command_long_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude_m,
    )
    print(f"[MAVLink] Takeoff → {altitude_m} m")
    time.sleep(altitude_m / 1.5 + 2)


def send_velocity_ned(mav, vx: float, vy: float, vz: float, yaw_rate_deg: float = 0.0):
    """
    Body-frame NED velocity + yaw rate setpoint.
    yaw_rate_deg: positive = clockwise rotation when viewed from above.
    """
    yaw_rate_rad = math.radians(yaw_rate_deg)

    # Type mask: ignore position & acceleration; use velocity + yaw rate
    TYPE_MASK = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )

    mav.mav.set_position_target_local_ned_send(
        0,
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        TYPE_MASK,
        0, 0, 0,                    # position (ignored)
        vx, vy, vz,                 # velocity
        0, 0, 0,                    # acceleration (ignored)
        0,                          # yaw (ignored)
        yaw_rate_rad,               # yaw rate
    )


def land_mode(mav):
    set_mode(mav, "LAND")
    print("[MAVLink] LAND mode – descending to touchdown.")


# ══════════════════════════════════════════════
#  Geometry helpers
# ══════════════════════════════════════════════

def pixel_error_to_metres(err_px: float, focal_length_px: float, dist_m: float) -> float:
    return err_px * dist_m / focal_length_px


def estimate_distance(tvec: np.ndarray) -> float:
    return float(np.linalg.norm(tvec))


def rvec_to_yaw_deg(rvec: np.ndarray) -> float:
    """
    Extract the rotation around the camera's Z-axis (yaw in image plane)
    from an ArUco rvec.  Positive = counter-clockwise in image.
    """
    R, _ = cv2.Rodrigues(rvec)
    # The marker's X-axis in camera space
    x_axis = R[:, 0]
    yaw_rad = math.atan2(x_axis[1], x_axis[0])
    return math.degrees(yaw_rad)


def angle_diff_deg(target: float, current: float) -> float:
    """Signed angular difference in [-180, 180]."""
    diff = (target - current + 180) % 360 - 180
    return diff


# ══════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════

def main():
    # ── MAVLink ────────────────────────────────────────────────────────────────
    mav = connect_mavlink(CONNECTION_STRING)
    set_mode(mav, "GUIDED")
    arm(mav)
    takeoff(mav, altitude_m=5.0)

    # ── Camera & ArUco ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    cx, cy = FRAME_W / 2.0, FRAME_H / 2.0
    fx     = CAMERA_MATRIX[0, 0]

    landed     = False
    last_seen  = time.time()

    # Running state (kept across frames for filtering / logging)
    state = {
        "horiz_err_m" : 999.0,
        "yaw_err_deg" : 999.0,
        "dist_m"      : 999.0,
    }

    print("[Landing] Starting nested ArUco precision landing loop …")

    try:
        while not landed:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            # Build a convenient {id: corner_array} mapping
            detected: dict[int, np.ndarray] = {}
            if ids is not None:
                for i, mid in enumerate(ids.flatten()):
                    detected[int(mid)] = corners[i]

            # ── Outer marker: position control ─────────────────────────────────
            vel_forward = vel_right = vel_down = 0.0
            outer_dist  = None

            if OUTER_MARKER_ID in detected:
                last_seen = time.time()
                corner    = detected[OUTER_MARKER_ID]

                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corner], OUTER_MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS
                )
                tvec = tvec[0][0]
                outer_dist = estimate_distance(tvec)

                pts    = corner[0]
                mx, my = pts[:, 0].mean(), pts[:, 1].mean()

                err_x_m = pixel_error_to_metres(mx - cx, fx, outer_dist)
                err_y_m = pixel_error_to_metres(my - cy, fx, outer_dist)

                state["horiz_err_m"] = math.hypot(err_x_m, err_y_m)
                state["dist_m"]      = outer_dist

                vel_right   = float(np.clip( KP_XY * err_x_m, -MAX_VEL_XY, MAX_VEL_XY))
                vel_forward = float(np.clip( KP_XY * err_y_m, -MAX_VEL_XY, MAX_VEL_XY))

                # Draw outer marker
                cv2.aruco.drawDetectedMarkers(frame, [corner],
                                              np.array([[OUTER_MARKER_ID]]))
                cv2.circle(frame, (int(mx), int(my)), 6, (0, 255, 0), -1)
                cv2.line(frame, (int(cx), int(cy)), (int(mx), int(my)), (0, 0, 255), 2)

            else:
                elapsed = time.time() - last_seen
                print(f"[Detection] Outer marker lost ({elapsed:.1f}s) – hovering")

            # ── Inner marker: yaw control ──────────────────────────────────────
            yaw_rate_cmd = 0.0

            if INNER_MARKER_ID in detected:
                inner_corner = detected[INNER_MARKER_ID]

                rvec_inner, tvec_inner, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [inner_corner], INNER_MARKER_SIZE_M, CAMERA_MATRIX, DIST_COEFFS
                )
                inner_yaw_deg = rvec_to_yaw_deg(rvec_inner[0][0])

                # If outer is also visible, compute yaw of outer and find relative offset
                if OUTER_MARKER_ID in detected:
                    rvec_outer, _, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [detected[OUTER_MARKER_ID]], OUTER_MARKER_SIZE_M,
                        CAMERA_MATRIX, DIST_COEFFS
                    )
                    outer_yaw_deg = rvec_to_yaw_deg(rvec_outer[0][0])
                    current_relative_yaw = angle_diff_deg(inner_yaw_deg, outer_yaw_deg)
                else:
                    # Only inner visible – align inner to a fixed world reference
                    current_relative_yaw = inner_yaw_deg

                yaw_err_deg           = angle_diff_deg(DESIRED_YAW_OFFSET_DEG,
                                                        current_relative_yaw)
                state["yaw_err_deg"]  = yaw_err_deg

                yaw_rate_cmd = float(np.clip(
                    KP_YAW * yaw_err_deg,
                    -MAX_YAW_RATE, MAX_YAW_RATE
                ))

                # Draw inner marker
                cv2.aruco.drawDetectedMarkers(frame, [inner_corner],
                                              np.array([[INNER_MARKER_ID]]),
                                              borderColor=(255, 0, 0))

                # Draw heading arrow on inner marker
                ipts = inner_corner[0]
                ic   = (int(ipts[:, 0].mean()), int(ipts[:, 1].mean()))
                tip  = (int(ic[0] + 40 * math.cos(math.radians(inner_yaw_deg))),
                        int(ic[1] + 40 * math.sin(math.radians(inner_yaw_deg))))
                cv2.arrowedLine(frame, ic, tip, (255, 0, 255), 2, tipLength=0.4)

            # ── Descent logic ──────────────────────────────────────────────────
            centred = state["horiz_err_m"] < CENTRE_THRESHOLD_M
            aligned = abs(state["yaw_err_deg"]) < YAW_THRESHOLD_DEG
            close   = (outer_dist is not None) and (outer_dist < LAND_ALTITUDE_M)

            if close and centred and aligned:
                land_mode(mav)
                landed = True
                break
            elif centred and aligned:
                vel_down = DESCENT_RATE
            else:
                vel_down = 0.0   # hold altitude while correcting XY / yaw

            # ── Send command ───────────────────────────────────────────────────
            send_velocity_ned(mav, vel_forward, vel_right, vel_down, yaw_rate_cmd)

            # ── HUD overlay ───────────────────────────────────────────────────
            hud_lines = [
                f"Dist:     {state['dist_m']:.2f} m",
                f"Horiz err:{state['horiz_err_m']:.3f} m",
                f"Yaw err:  {state['yaw_err_deg']:.1f} deg",
                f"Centred:  {'YES' if centred else 'NO'}",
                f"Aligned:  {'YES' if aligned else 'NO'}",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(frame, line, (10, 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

            cv2.imshow("Precision Landing – Nested ArUco", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[User] Quit – aborting.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Landing] Script exiting.")


if __name__ == "__main__":
    main()

