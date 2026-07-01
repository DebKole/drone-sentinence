
#!/usr/bin/env python3
"""
lawnmower_optical_final.py
==========================
Professional closed-loop autonomous workflow for Non-GPS Drones.
Fixed: Takeoff phase separation and on-ground EKF drift compensation.
"""

import time
import math
from pymavlink import mavutil

# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

PORT = '/dev/ttyACM0'
BAUD = 921600

TARGET_ALT_M = 0.8           
ARENA_LENGTH_M = 2.0         
ARENA_WIDTH_M = 2.0          
SWATH_WIDTH_M = 1.0          
TOLERANCE_M = 0.2            
FLIGHT_SPEED_CM_S = 20       

AUTO_MODE = 'GUIDED'         
SAFE_MODES = ['STABILIZE', 'LOITER', 'ALT_HOLD']

# ═══════════════════════════════════════════════════════════════════

_current_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
_current_mode = None

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

def check_safety_pause(mav):
    global _current_mode
    update_telemetry(mav)
    
    if _current_mode and _current_mode != AUTO_MODE:
        log("RC OVERRIDE", f"Pilot switched to {_current_mode}! Script yielding control.")
        log("RC OVERRIDE", f"Switch back to {AUTO_MODE} to resume the mission.")
        
        while True:
            update_telemetry(mav)
            if _current_mode == AUTO_MODE:
                log("RESUME", f"Mode restored to {AUTO_MODE}. Resuming.")
                time.sleep(1)
                break
            time.sleep(0.5)

def set_fake_origin(mav):
    log("INIT", "Injecting local origin (0,0) to unlock GUIDED mode...")
    mav.mav.set_gps_global_origin_send(mav.target_system, 0, 0, 0)
    mav.mav.set_home_position_send(
        mav.target_system, 
        0, 0, 0, 
        0, 0, 0, 
        [1.0, 0.0, 0.0, 0.0], 
        0, 0, 0
    )
    time.sleep(1) 

def send_position_target(mav, target_x, target_y, target_z):
    mav.mav.set_position_target_local_ned_send(
        0, mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        int(0b110111111000), 
        target_x, target_y, target_z,
        0, 0, 0, 0, 0, 0, 0, 0
    )

def wait_for_takeoff(mav, target_alt):
    """
    Monitors altitude without sending position targets. 
    This allows the drone to execute its internal takeoff sequence.
    """
    log("TAKEOFF", f"Waiting for drone to climb to {target_alt}m...")
    while True:
        check_safety_pause(mav)
        update_telemetry(mav)
        
        current_alt = -_current_pos['z'] # Z is negative for UP in NED
        print(f"    [DEBUG] Climbing... Alt: {current_alt:.2f}m / {target_alt:.2f}m     ", end='\r')
        
        # If we are within 20cm of the target altitude, takeoff is complete
        if current_alt >= (target_alt - 0.2):
            print("") 
            log("TAKEOFF", "Target altitude reached. Stabilizing...")
            time.sleep(2) # Give it 2 seconds to brake and hover nicely
            break
        time.sleep(0.2)

def go_to_waypoint(mav, x, y, z_alt, label):
    z_ned = -z_alt 
    log("TARGET", f"Moving to {label} -> Target X: {x:.2f}m, Y: {y:.2f}m")
    
    while True:
        check_safety_pause(mav)
        update_telemetry(mav)
        
        dx = x - _current_pos['x']
        dy = y - _current_pos['y']
        dz = z_ned - _current_pos['z']
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        print(f"    [DEBUG] Cur_X: {_current_pos['x']:.2f} | Cur_Y: {_current_pos['y']:.2f} | Dist: {distance:.2f}m     ", end='\r')
        
        if distance < TOLERANCE_M:
            print("") 
            log("TARGET", f"Reached {label}.")
            break
            
        send_position_target(mav, x, y, z_ned)
        time.sleep(0.2)

def set_drone_speed(mav, speed_cm_s):
    log("INIT", f"Limiting autonomous speed to {speed_cm_s} cm/s.")
    mav.mav.param_set_send(
        mav.target_system, mav.target_component,
        b'WPNAV_SPEED', float(speed_cm_s), mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )

def main():
    log("INIT", "Connecting to Pixhawk...")
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    mav.wait_heartbeat()
    
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10, 1
    )
    
    log("INIT", "Telemetry established. Setting flight parameters.")
    set_drone_speed(mav, FLIGHT_SPEED_CM_S)
    set_fake_origin(mav)

    try:
        input(f"[*] Set RC switch to {AUTO_MODE}. Press Enter to START MISSION...")
        
        # 1. ARM
        mav.mav.command_long_send(
            mav.target_system, mav.target_component, 
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )
        log("FLIGHT", "Armed. Motors spinning up...")
        time.sleep(2) # Give motors time to idle
        
        # 2. TAKEOFF COMMAND (Standard MAV_CMD_NAV_TAKEOFF)
        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, TARGET_ALT_M
        )
        
        # Monitor the climb
        wait_for_takeoff(mav, TARGET_ALT_M)

        # 3. DRIFT COMPENSATION
        # Capture the true hovering X/Y position after EKF stabilizes in the air
        start_x = _current_pos['x']
        start_y = _current_pos['y']
        log("INIT", f"Locking grid origin to actual hover position X:{start_x:.2f}, Y:{start_y:.2f}")

        # 4. 3x3m LAWNMOWER ALGORITHM (Coordinate Based + Drift Offset)
        # Pass 1: Forward
        go_to_waypoint(mav, start_x + ARENA_LENGTH_M, start_y, TARGET_ALT_M, "Pass 1 (Forward)")
        # Step Right
        go_to_waypoint(mav, start_x + ARENA_LENGTH_M, start_y + SWATH_WIDTH_M, TARGET_ALT_M, "Step Right 1")
        
        # Pass 2: Backward
        go_to_waypoint(mav, start_x, start_y + SWATH_WIDTH_M, TARGET_ALT_M, "Pass 2 (Backward)")
        # Step Right
        go_to_waypoint(mav, start_x, start_y + (SWATH_WIDTH_M * 2), TARGET_ALT_M, "Step Right 2")
        
        # Pass 3: Forward
        go_to_waypoint(mav, start_x + ARENA_LENGTH_M, start_y + (SWATH_WIDTH_M * 2), TARGET_ALT_M, "Pass 3 (Forward)")
        # Step Right
        go_to_waypoint(mav, start_x + ARENA_LENGTH_M, start_y + (SWATH_WIDTH_M * 3), TARGET_ALT_M, "Step Right 3")
        
        # Pass 4: Backward
        go_to_waypoint(mav, start_x, start_y + (SWATH_WIDTH_M * 3), TARGET_ALT_M, "Pass 4 (Backward)")

        # 5. AUTONOMOUS LANDING
        log("LAND", "Arena coverage complete. Initiating landing.")
        mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9) # 9 = LAND mode
        
        while True:
            update_telemetry(mav)
            if _current_pos['z'] > -0.15: # Almost at ground (NED is negative Z for UP)
                break
            time.sleep(0.5)

        log("FINISH", "Touchdown. Disarming.")
        mav.mav.command_long_send(
            mav.target_system, mav.target_component, 
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0
        )

    except KeyboardInterrupt:
        log("EMERGENCY", "Script aborted via keyboard! Drone holding position.")
    finally:
        mav.close()

if __name__ == "__main__":
    main()
