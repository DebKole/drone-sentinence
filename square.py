#!/usr/bin/env python3
"""
drone_square_loop.py
=========================
Optimized for Jetson Orin Nano + Pixhawk.
Features: Automated takeoff, 2x perfect square loops (1m x 1m), auto-landing,
and complete safety RC pause/resume functionality.
"""

import time
import sys
import math
from pymavlink import mavutil

# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# -- MAVLink & Port --
PORT = '/dev/ttyACM0'
BAUD = 921600

# -- Flight Settings --
TARGET_ALT_M = 1.5           # Altitude the drone will climb to (meters)
SQUARE_SIDE_M = 1.0          # Length of each side of the square (meters)
TOTAL_LOOPS = 2              # Number of squares to execute before landing
WAYPOINT_TIMEOUT_S = 7.0     # Max time allowed to reach each corner before forcing next

# -- RC Overrides & Flight Targets --
THROTTLE_ZERO = 1000
THROTTLE_CLIMB = 1620
ALT_LANDED_M = 0.15          # Altitude threshold considered "touched down"
FLIGHT_MODE = 'LOITER'       # Switch to this mode on your RC to RUN/RESUME the script

# ═══════════════════════════════════════════════════════════════════

_last_rng = None
_last_mode = None

def log(tag, msg):
    print(f"[{tag} {_ts()}] {msg}")

def _ts(): 
    return time.strftime("%H:%M:%S")

def update_telemetry(mav):
    """Pumps the MAVLink buffer to keep telemetry fresh."""
    global _last_rng, _last_mode
    
    while True:
        msg = mav.recv_match(blocking=False)
        if not msg:
            break
            
        msg_type = msg.get_type()
        if msg_type == 'DISTANCE_SENSOR':
            _last_rng = msg.current_distance / 100.0
        elif msg_type == 'HEARTBEAT':
            _last_mode = mavutil.mode_string_v10(msg)

def get_rangefinder():
    return _last_rng

def set_guided_mode(mav):
    """Switches Pixhawk to GUIDED mode to accept position targets."""
    log("MODE", "Switching to GUIDED mode for structural movement.")
    # Custom Mode 4 = GUIDED in ArduCopter
    mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4)
    time.sleep(0.5)

def send_local_ned_target(mav, north, east, down):
    """
    Sends relative coordinate movements using MAVLink.
    Uses MAV_FRAME_BODY_OFFSET_NED so directions are always relative to where the drone is.
    """
    mav.mav.set_position_target_local_ned_send(
        0,                                              # boot_time
        mav.target_system, mav.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,      # Relative to current position/heading
        0b0000111111111000,                             # Mask looking only at X, Y, Z positions
        north, east, down,                              # Positions (m)
        0, 0, 0,                                        # Velocities (m/s)
        0, 0, 0,                                        # Accelerations
        0, 0                                            # Yaw, Yaw rate
    )

def check_pause(mav, is_landing_phase=False, is_guided_phase=False):
    """
    Monitors flight mode. If the pilot switches away from the required autonomous
    state, it strips controls and blocks until the switch returns to LOITER.
    """
    global _last_mode
    update_telemetry(mav)
    
    should_pause = False
    if is_landing_phase:
        if _last_mode and _last_mode not in ['LAND', 'LOITER']:
            should_pause = True
    elif is_guided_phase:
        if _last_mode and _last_mode not in ['GUIDED', 'LOITER']:
            should_pause = True
    else:
        if _last_mode and _last_mode != FLIGHT_MODE:
            should_pause = True
            
    if should_pause:
        log("PAUSE", f"Pilot manually took control (Mode: {_last_mode}). Script paused.")
        # Release all RC overrides immediately
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 0, 0, 0, 0, 0, 0, 0, 0)
        
        while True:
            update_telemetry(mav)
            if _last_mode == FLIGHT_MODE:
                log("RESUME", f"Switch turned back to {FLIGHT_MODE}. Resuming script control.")
                if is_landing_phase:
                    log("LAND", "Re-engaging autonomous landing sequence.")
                    mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9) # LAND
                elif is_guided_phase:
                    log("GUIDED", "Re-engaging GUIDED navigation.")
                    mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4) # GUIDED
                time.sleep(0.5)
                break
            time.sleep(0.1)
        return True
    return False

def execute_relative_move(mav, north, east, down, label):
    """Moves the drone relative to its position and waits for it to settle."""
    log("MOVE", f"Executing: {label} ({north}m N, {east}m E)")
    
    # Send the movement command
    send_local_ned_target(mav, north, east, down)
    
    # Give the drone time to reach the destination while monitoring for manual pauses
    start_time = time.time()
    while time.time() - start_time < WAYPOINT_TIMEOUT_S:
        # If a pause happens, we re-send the command on resume to make sure Pixhawk remembers it
        if check_pause(mav, is_landing_phase=False, is_guided_phase=True):
            send_local_ned_target(mav, north, east, down)
            start_time = time.time() # Reset timeout clock on manual resume
            
        time.sleep(0.1)

def main():
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    mav.wait_heartbeat()
    log("INIT", "Connected to Pixhawk. Telemetry OK.")

    try:
        input(f"[*] Flight Mode must be {FLIGHT_MODE}. Press Enter to START MISSION...")
        
        for _ in range(10):
            update_telemetry(mav)
            time.sleep(0.05)
        
        # 1. ARM THE MOTORS
        mav.mav.command_long_send(
            mav.target_system, mav.target_component, 
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )
        log("FLIGHT", "Armed. Climbing...")
        
        # 2. CLIMB TO TARGET ALTITUDE (Using RC Throttle Override)
        while (get_rangefinder() or 0) < TARGET_ALT_M:
            check_pause(mav, is_landing_phase=False, is_guided_phase=False)
            mav.mav.rc_channels_override_send(
                mav.target_system, mav.target_component, 
                1500, 1500, THROTTLE_CLIMB, 1500, 0, 0, 0, 0
            )
            time.sleep(0.1)

        log("FLIGHT", "Target altitude reached. Preparing for square loops.")
        # Clear throttle overrides so GUIDED mode can take full ownership of flight controls
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 0, 0, 0, 0, 0, 0, 0, 0)
        time.sleep(0.1)
        
        # Switch to GUIDED mode for coordinate control
        set_guided_mode(mav)

        # 3. SQUARE LOOP EXECUTION
        for loop_count in range(1, TOTAL_LOOPS + 1):
            log("LOOP", f"Starting Square Loop {loop_count} of {TOTAL_LOOPS}")
            
            # Step A: Move 1m Forward
            execute_relative_move(mav, north=SQUARE_SIDE_M, east=0.0, down=0.0, label="1m Forward")
            
            # Step B: Move 1m Left (In NED coordinates, East is positive, so Left is negative East)
            execute_relative_move(mav, north=0.0, east=-SQUARE_SIDE_M, down=0.0, label="1m Left")
            
            # Step C: Move 1m Backward (Negative North)
            execute_relative_move(mav, north=-SQUARE_SIDE_M, east=0.0, down=0.0, label="1m Backward")
            
            # Step D: Move 1m Right (Positive East)
            execute_relative_move(mav, north=0.0, east=SQUARE_SIDE_M, down=0.0, label="1m Right (Home)")

        # 4. AUTONOMOUS LANDING
        log("LAND", "All loops completed. Initializing landing sequence.")
        mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9) # 9 = LAND mode
        
        while (get_rangefinder() or 1) > ALT_LANDED_M:
            check_pause(mav, is_landing_phase=True, is_guided_phase=False)
            time.sleep(0.1)

        # 5. DISARM MOTORS
        log("FINISH", "Touchdown detected. Disarming.")
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 1500, 1500, THROTTLE_ZERO, 1500, 0, 0, 0, 0)
        time.sleep(1)
        mav.mav.command_long_send(
            mav.target_system, mav.target_component, 
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0
        )

    except KeyboardInterrupt:
        log("EMERGENCY", "Script aborted via keyboard!")
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 1500, 1500, THROTTLE_ZERO, 1500, 0, 0, 0, 0)
    finally:
        mav.close()

if __name__ == "__main__":
    main()
