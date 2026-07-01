#!/usr/bin/env python3
"""
simple_forward_test.py
======================
A straightforward test script to verify basic autonomous capabilities:
1. Takeoff to 1.0m.
2. Hover in place for 4 seconds.
3. Move exactly 1.0m Forward (+X in local frame) at 20cm/s.
4. Hover in place for 4 seconds.
5. Auto Land.
"""

import time
import math
from pymavlink import mavutil

# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PORT = '/dev/ttyACM0'
BAUD = 921600

TARGET_ALT_M = 1.0           
FLIGHT_SPEED_CM_S = 20       
FORWARD_DIST_M = 1.0
HOVER_DURATION_S = 4.0
TOLERANCE_M = 0.2            

AUTO_MODE = 'GUIDED'         

# ═══════════════════════════════════════════════════════════════════

_current_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
_current_yaw = 0.0 

def log(tag, msg):
    print(f"[{tag} {time.strftime('%H:%M:%S')}] {msg}")

def flush_telemetry(mav):
    global _current_pos, _current_yaw
    while True:
        msg = mav.recv_match(blocking=False)
        if not msg:
            break
            
        msg_type = msg.get_type()
        
        if msg_type == 'LOCAL_POSITION_NED':
            _current_pos['x'] = msg.x  
            _current_pos['y'] = msg.y  
            _current_pos['z'] = msg.z  
            
        elif msg_type == 'ATTITUDE':
            _current_yaw = msg.yaw 

def set_fake_origin(mav):
    log("INIT", "Injecting local origin (0,0)...")
    mav.mav.set_gps_global_origin_send(mav.target_system, 0, 0, 0)
    mav.mav.set_home_position_send(
        mav.target_system,
        0, 0, 0, 0, 0, 0, 
        [1.0, 0.0, 0.0, 0.0], 0, 0, 0
    )
    time.sleep(1) 

def set_flight_parameters(mav, speed_cm_s):
    log("INIT", f"Limiting autonomous speed to {speed_cm_s} cm/s.")
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        b'WPNAV_SPEED', float(speed_cm_s), mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        b'WP_YAW_BEHAVIOR', 0.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )

def send_position_target(mav, target_x, target_y, target_z, target_yaw):
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        int(0b100111111000), 
        target_x, target_y, target_z,
        0, 0, 0, 0, 0, 0, 
        target_yaw, 
        0    
    )

def wait_for_takeoff(mav, target_alt):
    log("TAKEOFF", f"Climbing to {target_alt}m...")
    while True:
        flush_telemetry(mav)
        current_alt = -_current_pos['z']
        print(f"    [DEBUG] Climbing... Alt: {current_alt:.2f}m / {target_alt:.2f}m     ", end='\r')
        
        if current_alt >= (target_alt - 0.2):
            print("") 
            log("TAKEOFF", "Target altitude reached.")
            break
        time.sleep(0.1) 

def hover(mav, x, y, z_alt, target_yaw, duration, label):
    log("HOVER", f"{label} - Holding position for {duration} seconds...")
    start_time = time.time()
    
    while time.time() - start_time < duration:
        flush_telemetry(mav)
        elapsed = time.time() - start_time
        print(f"    [HOVER {elapsed:.1f}s] Live EKF -> X: {_current_pos['x']:.2f}m, Y: {_current_pos['y']:.2f}m     ", end='\r')
        
        # Continuously send the lock target at 10Hz to fight wind/drift
        send_position_target(mav, x, y, -z_alt, target_yaw)
        time.sleep(0.1)
    print("")

def go_to_waypoint(mav, x, y, z_alt, target_yaw, label):
    z_ned = -z_alt 
    log("NAV", f"Moving to {label} (X:{x:.2f}, Y:{y:.2f})...")
    
    while True:
        flush_telemetry(mav)
        
        dx = x - _current_pos['x']
        dy = y - _current_pos['y']
        dz = z_ned - _current_pos['z']
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        print(f"    [DEBUG] Dist to target: {distance:.2f}m     ", end='\r')

        if distance < TOLERANCE_M:
            print("") 
            break
            
        send_position_target(mav, x, y, z_ned, target_yaw)
        time.sleep(0.1) 

def main():
    log("INIT", "Connecting to Pixhawk...")
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    mav.wait_heartbeat()
    
    # Request data streams
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10, 1
    )
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 10, 1
    )
    
    set_flight_parameters(mav, FLIGHT_SPEED_CM_S)
    set_fake_origin(mav)

    try:
        input(f"[*] Set RC switch to {AUTO_MODE}. Press Enter to START FLIGHT...")
        
        log("FLIGHT", "Arming drone...")
        mav.mav.command_long_send(
            mav.target_system, mav.target_component, 
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )
        time.sleep(2) 
        
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, TARGET_ALT_M
        )
        wait_for_takeoff(mav, TARGET_ALT_M)

        # Grab the locked origin and initial yaw after takeoff
        flush_telemetry(mav)
        start_x = _current_pos['x']
        start_y = _current_pos['y']
        start_yaw = _current_yaw
        
        log("INIT", f"Locked Origin X:{start_x:.2f}, Y:{start_y:.2f}, Yaw:{math.degrees(start_yaw):.1f}°")

        # 1. First Hover at Origin
        hover(mav, start_x, start_y, TARGET_ALT_M, start_yaw, HOVER_DURATION_S, "Takeoff Point")

        # 2. Calculate Forward 1 Meter using Compass-Agnostic rotation matrix
        # (Translating local +Y Forward into absolute MAVLink NED coordinates)
        drone_forward = FORWARD_DIST_M
        drone_right = 0.0

        ned_dx = (drone_forward * math.cos(start_yaw)) - (drone_right * math.sin(start_yaw))
        ned_dy = (drone_forward * math.sin(start_yaw)) + (drone_right * math.cos(start_yaw))
        
        target_x = start_x + ned_dx
        target_y = start_y + ned_dy

        # 3. Move Forward
        go_to_waypoint(mav, target_x, target_y, TARGET_ALT_M, start_yaw, "Forward 1.0m Mark")

        # 4. Second Hover at Destination
        hover(mav, target_x, target_y, TARGET_ALT_M, start_yaw, HOVER_DURATION_S, "Destination Point")

        # 5. Land
        log("LAND", "Test complete. Initiating landing.")
        mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9) 
        
        while True:
            flush_telemetry(mav)
            if _current_pos['z'] > -0.15: 
                break
            time.sleep(0.5)

        log("FINISH", "Touchdown. Disarming.")
        mav.mav.command_long_send(
            mav.target_system, mav.target_component, 
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0
        )

    except KeyboardInterrupt:
        print("")
        log("EMERGENCY", "Script aborted! Drone holding position.")
    finally:
        mav.close()

if __name__ == "__main__":
    main()
