#!/usr/bin/env python3
"""
claude_gemini_fix.py
===================
Version 2.4-robust-cam

Changes vs 2.3:
  - Fixed Camera Thread re-open lockouts by introducing a proper teardown sequence.
  - Added a 0.5s resource release delay to let the OS release /dev/video0.
  - Set self._cap explicitly to None during resets to prevent resource leakage.
"""

import time
import math
import cv2
import numpy as np
from pymavlink import mavutil
import threading
from flask import Flask, Response
import serial.serialutil

VERSION = "2.4-robust-cam"

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PORT  = '/dev/ttyACM0'
BAUD  = 921600

TARGET_ALT_M     = 1.5
ALT_LANDED_M     = 0.15
FLIGHT_MODE      = 'LOITER'

RC_MID           = 1500
THROTTLE_CLIMB   = 1620
THROTTLE_HOVER   = 1500
THROTTLE_DESCEND = 1400 

CAM_USB_INDEX    = 0
CAM_WIDTH        = 1280
CAM_HEIGHT       = 720
ARUCO_DICT       = cv2.aruco.DICT_6X6_250
TARGET_MARKER_ID = 1

CAM_OFFSET_X     = 0    
CAM_OFFSET_Y     = 0    

KP_RC            = 0.05
RC_MAX_NUDGE     = 25
CENTER_TOLERANCE_PX     = 80
RC_SMOOTH               = 0.6
MARKER_LOST_DEBOUNCE_FRAMES = 8

CAM_STALE_TIMEOUT_S = 1.0
MAV_RECONNECT_DELAY    = 2.0
MAV_RECONNECT_ATTEMPTS = 10

# ═══════════════════════════════════════════════════════════════════

_last_rng  = None
_last_mode = None
mav_lock   = threading.Lock()
app = Flask(__name__)

SHUTTING_DOWN = False

def log(tag, msg):
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}")

# ─── CAMERA THREAD ───────────────────────────────────────────────
class CameraThread(threading.Thread):
    def __init__(self, index, width, height):
        super().__init__(daemon=True)
        self._index  = index
        self._width  = width
        self._height = height
        self._lock   = threading.Lock()
        self._frame  = None
        self._frame_time = 0.0
        self._running = True
        self._cap    = None

    def _close_current_capture(self):
        """Safely tears down the capture object and forces an OS resource release."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            # CRITICAL: Allow the Linux V4L2 driver subsystem a moment 
            # to release the file lock on /dev/video0
            time.sleep(0.5)

    def _open(self):
        self._close_current_capture()
        
        log("CAM", f"Attempting to open /dev/video{self._index}...")
        cap = cv2.VideoCapture(self._index, cv2.CAP_V4L2)
        
        if not cap.isOpened():
            return False
            
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap = cap
        return True

    def run(self):
        if not self._open():
            log("CAM", "Initial open failed — entering retry loop.")
            
        while self._running and not SHUTTING_DOWN:
            try:
                if self._cap is None or not self._cap.isOpened():
                    time.sleep(1.0)
                    self._open()
                    continue
                
                ret, frame = self._cap.read()
                if ret and frame is not None and frame.size > 0:
                    with self._lock:
                        self._frame      = frame
                        self._frame_time = time.monotonic()
                else:
                    log("CAM", "Empty frame or select() timeout — rebuilding pipeline...")
                    time.sleep(0.5)
                    self._open()
            except cv2.error as e:
                # Catching internal MJPEG imdecode_ alignment faults
                log("CAM", f"Internal OpenCV exception caught: {e}")
                log("CAM", "Resetting V4L2 capture device context...")
                time.sleep(0.5)
                self._open()
            except Exception as e:
                log("CAM", f"Unexpected runtime error: {e}")
                time.sleep(0.5)
                self._open()

    def get_frame(self):
        with self._lock:
            if self._frame is None:
                return None, 9999.0
            age = time.monotonic() - self._frame_time
            return self._frame.copy(), age

    def stop(self):
        self._running = False
        self._close_current_capture()

# ─── STREAMING SERVER ────────────────────────────────────────────
_stream_frame      = None
_stream_frame_lock = threading.Lock()

@app.route('/video')
def video_stream():
    def generate():
        while not SHUTTING_DOWN:
            f = None
            with _stream_frame_lock:
                if _stream_frame is not None:
                    f = _stream_frame.copy()
            if f is not None:
                ret, buf = cv2.imencode('.jpg', f)
                if ret:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                           + buf.tobytes() + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_streaming_server():
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5000, threaded=True)

def push_stream_frame(frame):
    global _stream_frame
    with _stream_frame_lock:
        _stream_frame = frame.copy()

# ─── MAVLINK HELPERS ─────────────────────────────────────────────
def connect_mav(port, baud, label="Connecting"):
    for attempt in range(1, MAV_RECONNECT_ATTEMPTS + 1):
        if SHUTTING_DOWN: return None
        try:
            log("MAV", f"{label} (attempt {attempt}/{MAV_RECONNECT_ATTEMPTS})...")
            mav = mavutil.mavlink_connection(port, baud=baud)
            mav.wait_heartbeat(timeout=5)
            log("MAV", "Connected to Pixhawk.")
            return mav
        except Exception as e:
            log("MAV", f"Failed: {e}")
            if attempt < MAV_RECONNECT_ATTEMPTS:
                time.sleep(MAV_RECONNECT_DELAY)
    raise RuntimeError(f"Could not connect after {MAV_RECONNECT_ATTEMPTS} attempts.")

def _try_reconnect(old_mav):
    log("MAV", "USB disconnect — drone holds LOITER. Reconnecting...")
    try:
        old_mav.close()
    except Exception:
        pass
    time.sleep(MAV_RECONNECT_DELAY)
    return connect_mav(PORT, BAUD, label="Reconnecting")

def update_telemetry(mav):
    global _last_rng, _last_mode
    try:
        with mav_lock:
            while True:
                msg = mav.recv_match(blocking=False)
                if not msg:
                    break
                t = msg.get_type()
                if t == 'DISTANCE_SENSOR':
                    _last_rng = msg.current_distance / 100.0
                elif t == 'HEARTBEAT':
                    _last_mode = mavutil.mode_string_v10(msg)
        return mav
    except (serial.serialutil.SerialException, OSError) as e:
        log("MAV", f"Serial error: {e}")
        return _try_reconnect(mav)

def send_rc_override(mav, roll, pitch, throttle):
    try:
        with mav_lock:
            mav.mav.rc_channels_override_send(
                mav.target_system, mav.target_component,
                int(roll), int(pitch), int(throttle),
                1500, 0, 0, 0, 0
            )
        return mav
    except (serial.serialutil.SerialException, OSError) as e:
        log("MAV", f"Serial error sending RC: {e}")
        return _try_reconnect(mav)

def get_rangefinder():
    return _last_rng if _last_rng is not None else 0.0

def check_pause(mav, expected_mode):
    global _last_mode
    mav = update_telemetry(mav)
    if _last_mode and _last_mode != expected_mode:
        log("PAUSE", f"Pilot override (Mode: {_last_mode}). Paused.")
        mav = send_rc_override(mav, 0, 0, 0)
        while not SHUTTING_DOWN:
            mav = update_telemetry(mav)
            if _last_mode == expected_mode:
                log("RESUME", f"Restored to {expected_mode}. Resuming.")
                break
            time.sleep(0.05)
        return True, mav
    return False, mav

# ─── MAIN ────────────────────────────────────────────────────────
def main():
    global SHUTTING_DOWN
    log("INIT", f"Version {VERSION}")

    mav = connect_mav(PORT, BAUD)
    cam = CameraThread(CAM_USB_INDEX, CAM_WIDTH, CAM_HEIGHT)
    cam.start()
    time.sleep(1.0)
    frame, age = cam.get_frame()
    if frame is None:
        log("WARN", "Camera thread initializing asynchronously.")
    else:
        log("INIT", f"Camera engine linked ({CAM_WIDTH}x{CAM_HEIGHT}).")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    params     = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(aruco_dict, params)

    cx = CAM_WIDTH  / 2.0 + CAM_OFFSET_X
    cy = CAM_HEIGHT / 2.0 + CAM_OFFSET_Y

    threading.Thread(target=start_streaming_server, daemon=True).start()
    log("INIT", "Stream: http://<jetson-ip>:5000/video")

    try:
        input(f"[*] Set switch to {FLIGHT_MODE}. Press Enter to START MISSION...")
        for _ in range(10):
            mav = update_telemetry(mav)
            time.sleep(0.05)

        with mav_lock:
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0
            )

        log("FLIGHT", f"Armed. Climbing to {TARGET_ALT_M}m...")

        # ── PHASE 1: CLIMB ───────────────────────────────────────
        while get_rangefinder() < TARGET_ALT_M:
            paused, mav = check_pause(mav, FLIGHT_MODE)
            if paused: continue
            frame, _ = cam.get_frame()
            if frame is not None:
                push_stream_frame(frame)
            mav = send_rc_override(mav, RC_MID, RC_MID, THROTTLE_CLIMB)
            print(f"[CLIMBING] Alt: {get_rangefinder():.2f}m     ", end='\r')
            time.sleep(0.1)

        print()
        log("FLIGHT", "Target altitude reached. Initiating visual search...")

        # ── PHASE 2: ALIGN & DESCEND ─────────────────────────────
        marker_visible = False
        lost_frames    = 0
        smooth_roll    = float(RC_MID)
        smooth_pitch   = float(RC_MID)

        while get_rangefinder() > ALT_LANDED_M:
            paused, mav = check_pause(mav, FLIGHT_MODE)
            if paused: continue

            frame, frame_age = cam.get_frame()

            if frame is None or frame_age > CAM_STALE_TIMEOUT_S:
                mav = send_rc_override(mav, RC_MID, RC_MID, THROTTLE_HOVER)
                mav = update_telemetry(mav)
                status = "no frame yet" if frame is None else f"stale {frame_age:.1f}s"
                print(f"[CAM WAIT] Camera unavailable ({status}) — holding hover...  ", end='\r')
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            marker_found = (ids is not None and TARGET_MARKER_ID in ids.flatten())

            if marker_found:
                lost_frames = 0
                if not marker_visible:
                    print()
                    log("VISION", f"Marker {TARGET_MARKER_ID} acquired!")
                    marker_visible = True

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                idx = ids.flatten().tolist().index(TARGET_MARKER_ID)
                mc  = corners[idx][0]
                mx, my = mc[:, 0].mean(), mc[:, 1].mean()

                err_x = mx - cx
                err_y = my - cy

                raw_roll  = RC_MID + (err_x * KP_RC)
                raw_pitch = RC_MID + (err_y * KP_RC)

                smooth_roll  = RC_SMOOTH * smooth_roll  + (1 - RC_SMOOTH) * raw_roll
                smooth_pitch = RC_SMOOTH * smooth_pitch + (1 - RC_SMOOTH) * raw_pitch

                rc_roll  = max(RC_MID - RC_MAX_NUDGE, min(RC_MID + RC_MAX_NUDGE, smooth_roll))
                rc_pitch = max(RC_MID - RC_MAX_NUDGE, min(RC_MID + RC_MAX_NUDGE, smooth_pitch))

                err_magnitude = math.hypot(err_x, err_y)
                
                if err_magnitude < CENTER_TOLERANCE_PX:
                    descent_factor = 1.0 - (err_magnitude / CENTER_TOLERANCE_PX)
                    rc_throttle = int(THROTTLE_HOVER - ((THROTTLE_HOVER - THROTTLE_DESCEND) * descent_factor))
                    action = f"DESCENDING ({rc_throttle} PWM)"
                else:
                    rc_throttle = THROTTLE_HOVER
                    action = "CENTERING"

                mav = send_rc_override(mav, rc_roll, rc_pitch, rc_throttle)
                print(
                    f"[{action}] Alt: {get_rangefinder():.2f}m | "
                    f"ErrX: {err_x:.0f}px  ErrY: {err_y:.0f}px  "
                    f"Roll: {rc_roll:.0f}  Pitch: {rc_pitch:.0f}  ",
                    end='\r'
                )

            else:
                lost_frames += 1
                if lost_frames < MARKER_LOST_DEBOUNCE_FRAMES:
                    rc_roll  = max(RC_MID - RC_MAX_NUDGE, min(RC_MID + RC_MAX_NUDGE, smooth_roll))
                    rc_pitch = max(RC_MID - RC_MAX_NUDGE, min(RC_MID + RC_MAX_NUDGE, smooth_pitch))
                    mav = send_rc_override(mav, rc_roll, rc_pitch, THROTTLE_HOVER)
                    print(f"[COASTING] Marker lost {lost_frames}/{MARKER_LOST_DEBOUNCE_FRAMES} frames...  ", end='\r')
                else:
                    if marker_visible:
                        print()
                        log("VISION", "Marker lost! Leveling to reacquire...")
                        marker_visible = False
                    smooth_roll  = RC_SMOOTH * smooth_roll  + (1 - RC_SMOOTH) * RC_MID
                    smooth_pitch = RC_SMOOTH * smooth_pitch + (1 - RC_SMOOTH) * RC_MID
                    mav = send_rc_override(mav, RC_MID, RC_MID, THROTTLE_HOVER)
                    print(f"[SEARCHING] Hovering... Alt: {get_rangefinder():.2f}m  ", end='\r')

            mav = update_telemetry(mav)
            push_stream_frame(frame)

        # ── PHASE 3: TOUCHDOWN & DISARM ──────────────────────────
        print()
        log("FINISH", "Touchdown altitude. Landing.")
        mav = send_rc_override(mav, 0, 0, 0)
        time.sleep(0.1)

        with mav_lock:
            mav.mav.set_mode_send(
                mav.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9
            )
            time.sleep(2)
            mav.mav.command_long_send(
                mav.target_system, mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0
            )

    except KeyboardInterrupt:
        SHUTTING_DOWN = True
        log("EMERGENCY", "Script aborted! Centering sticks.")
        try:
            mav = send_rc_override(mav, RC_MID, RC_MID, THROTTLE_HOVER)
        except Exception:
            pass
    finally:
        SHUTTING_DOWN = True
        log("INIT", "Stopping camera thread...")
        cam.stop()
        cam.join(timeout=3.0)   
        try:
            mav.close()
        except Exception:
            pass
        log("INIT", "Shutdown complete.")

if __name__ == "__main__":
    main()
