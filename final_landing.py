#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np
import threading
from pymavlink import mavutil
from flask import Flask, Response, render_template_string

# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PORT = '/dev/ttyACM0'     # Jetson serial port
BAUD = 921600

# Camera Setup
# If using a USB WebCam, leave as 0. 
# If using a CSI/Pi Cam on Jetson, uncomment the GSTREAMER_PIPELINE and use that instead.
CAMERA_INDEX = 0 
# GSTREAMER_PIPELINE = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"

# Flight Parameters
TAKEOFF_ALT_M = 1.5            # Altitude to climb to before searching
LANDING_ALT_THRESHOLD_M = 0.3  # Alt to switch from iterative descent to strict LAND mode
DESCENT_SPEED_M_S = 0.2        # Speed of downward descent
MAX_XY_SPEED_M_S = 1.0         # Safety cap for horizontal velocity
KP = 0.005                     # Proportional gain for pixel-to-velocity conversion

AUTO_MODE = 'GUIDED'

# ═══════════════════════════════════════════════════════════════════
#  GLOBALS, TELEMETRY & THREADING
# ═══════════════════════════════════════════════════════════════════

_current_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
_current_mode = None

# Thread-safe variables for the Flask video stream
app = Flask(__name__)
shared_frame = None
frame_lock = threading.Lock()

def log(tag, msg):
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {msg}")

def update_telemetry(mav):
    global _current_pos, _current_mode
    while True:
        msg = mav.recv_match(blocking=False)
        if not msg:
            break
        msg_type = msg.get_type()
        if msg_type == 'LOCAL_POSITION_NED':
            _current_pos['x'] = msg.x
            _current_pos['y'] = msg.y
            _current_pos['z'] = msg.z
        elif msg_type == 'HEARTBEAT':
            _current_mode = mavutil.mode_string_v10(msg)

# ═══════════════════════════════════════════════════════════════════
#  FLASK WEB SERVER
# ═══════════════════════════════════════════════════════════════════

def generate_video_stream():
    """Generator function to yield JPEG frames for the Flask web server."""
    global shared_frame
    while True:
        with frame_lock:
            if shared_frame is None:
                time.sleep(0.1)
                continue
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', shared_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        
        # Yield multipart HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05) # Cap stream at ~20 FPS to save Jetson CPU

@app.route('/')
def index():
    # Simple HTML page to display the video feed
    html = """
    <html>
    <head><title>Jetson Optical Flow Landing</title></head>
    <body style="background-color: black; color: white; text-align: center; font-family: sans-serif;">
        <h2>Drone Live Vision Feed</h2>
        <img src="/video_feed" width="800" style="border: 2px solid #4CAF50;">
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    """Runs the Flask app on port 5000."""
    # use_reloader=False is critical when running inside a thread
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ═══════════════════════════════════════════════════════════════════
#  FLIGHT CONTROL & TAKEOFF
# ═══════════════════════════════════════════════════════════════════

def send_body_velocity(mav, velocity_x, velocity_y, velocity_z):
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED, 0b110111000111,
        0, 0, 0, velocity_x, velocity_y, velocity_z, 0, 0, 0, 0, 0
    )

def arm_and_takeoff(mav, target_alt):
    """Switches to GUIDED, arms the drone, and takes off."""
    log("FLIGHT", "Requesting GUIDED mode...")
    # 4 is GUIDED mode in ArduCopter
    mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
    time.sleep(1)

    log("FLIGHT", "Arming motors...")
    mav.mav.command_long_send(mav.target_system, mav.target_component,
                              mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    
    # Wait for motors to spin up
    time.sleep(2) 

    log("FLIGHT", f"Initiating Takeoff to {target_alt}m...")
    mav.mav.command_long_send(mav.target_system, mav.target_component,
                              mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, target_alt)

    while True:
        update_telemetry(mav)
        current_alt = -_current_pos['z']
        print(f"  [TAKEOFF] Climbing... Alt: {current_alt:.2f}m / {target_alt:.2f}m   ", end='\r')
        
        # When we reach 95% of target altitude, break loop
        if current_alt >= target_alt * 0.95:
            print("\n")
            log("FLIGHT", "Takeoff altitude reached. Stabilizing hover.")
            time.sleep(2) # Let Optical Flow settle
            break
        time.sleep(0.2)

def trigger_land_mode(mav):
    log("LAND", "Altitude threshold reached. Triggering native LAND mode.")
    # 9 is LAND mode in ArduCopter
    mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)

# ═══════════════════════════════════════════════════════════════════
#  MAIN LOGIC
# ═══════════════════════════════════════════════════════════════════

def main():
    global shared_frame

    log("INIT", "Starting Web Server thread on port 5000...")
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    log("INIT", "Connecting to Flight Controller...")
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    mav.wait_heartbeat()
    log("INIT", "Heartbeat received.")

    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10, 1
    )

    # Setup ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    # Initialize Camera (Use GSTREAMER_PIPELINE here if using a CSI camera)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        log("ERROR", "Failed to open camera. Check permissions or GStreamer string.")
        return

    cam_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    center_x = cam_width / 2
    center_y = cam_height / 2

    # Execute Takeoff
    arm_and_takeoff(mav, TAKEOFF_ALT_M)

    log("FLIGHT", "Beginning ArUco Search and Landing Sequence.")
    landing_complete = False

    try:
        while not landing_complete:
            update_telemetry(mav)
            current_alt = -_current_pos['z']

            ret, frame = cap.read()
            if not ret:
                log("WARN", "Failed to grab frame. Is camera disconnected?")
                time.sleep(0.1)
                continue

            # Update shared frame for the Flask web server
            with frame_lock:
                shared_frame = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray)

            if ids is not None:
                # Optional: Draw marker on the shared frame so you see it in the web feed
                cv2.aruco.drawDetectedMarkers(shared_frame, corners, ids)

                c = corners[0][0] 
                marker_center_x = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
                marker_center_y = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)

                error_x = marker_center_x - center_x
                error_y = marker_center_y - center_y

                velocity_x = KP * (-error_y)
                velocity_y = KP * (error_x)
                velocity_z = DESCENT_SPEED_M_S 

                velocity_x = np.clip(velocity_x, -MAX_XY_SPEED_M_S, MAX_XY_SPEED_M_S)
                velocity_y = np.clip(velocity_y, -MAX_XY_SPEED_M_S, MAX_XY_SPEED_M_S)

                send_body_velocity(mav, velocity_x, velocity_y, velocity_z)

                print(f"  [TARGETING] Alt: {current_alt:.2f}m | Err_X: {error_x:4.0f}, Err_Y: {error_y:4.0f} | Vx: {velocity_x:+.2f}, Vy: {velocity_y:+.2f}      ", end='\r')

                if current_alt < LANDING_ALT_THRESHOLD_M:
                    print("\n")
                    trigger_land_mode(mav)
                    landing_complete = True

            else:
                send_body_velocity(mav, 0, 0, 0)
                print(f"  [SEARCHING] Hovering at Alt: {current_alt:.2f}m... Marker lost.                          ", end='\r')

    except KeyboardInterrupt:
        print("\n")
        log("EMERGENCY", "Script aborted! Drone holding position.")
        send_body_velocity(mav, 0, 0, 0)

    finally:
        print("\n")
        log("FINISH", "Cleaning up resources...")
        cap.release()
        mav.close()

if __name__ == "__main__":
    main()
