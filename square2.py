#!/usr/bin/env python3
"""
drone_square_loop_loiter.py
=========================
Optimized for Jetson Orin Nano + Pixhawk.
Features: Automated takeoff, 2x square loops purely in LOITER using RC overrides 
(ideal for Optical Flow), auto-landing, and RC pause/resume functionality.
"""

import time
import sys
from pymavlink import mavutil

# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# -- MAVLink & Port --
PORT = '/dev/ttyACM0'
BAUD = 921600

# -- Flight Settings --
TARGET_ALT_M = 1.5           # Altitude the drone will climb to (meters)
TOTAL_LOOPS = 2              # Number of squares to execute before landing

# -- Movement Tuning (LOITER Mode) --
# Distance = Speed (dictated by RC_TILT) * Time (MOVE_DURATION_S)
RC_NEUTRAL = 1500
RC_TILT_PWM = 100            # How hard to push the stick (e.g., 100 = 1400 or 1600 PWM)
MOVE_DURATION_S = 10.0        # How long to hold the stick to travel ~1 meter
SETTLE_TIME_S = 2.0          # How long to wait after centering sticks for drone to brake

# -- RC Overrides --
THROTTLE_ZERO = 1000
THROTTLE_CLIMB = 1620
ALT_LANDED_M = 0.15          # Altitude threshold considered "touched down"
FLIGHT_MODE = 'LOITER'       # Drone stays in this mode the entire time

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

def check_pause(mav, is_landing_phase=False):
    """
    Monitors flight mode. If the pilot switches away from the required autonomous
    state, it strips controls and blocks until the switch returns.
    """
    global _last_mode
    update_telemetry(mav)
    
    should_pause = False
    if is_landing_phase:
        if _last_mode and _last_mode not in ['LAND', 'LOITER']:
            should_pause = True
    else:
        if _last_mode and _last_mode != FLIGHT_MODE:
            should_pause = True
            
    if should_pause:
        log("PAUSE", f"Pilot manually took control (Mode: {_last_mode}). Script paused.")
        # Release all RC overrides immediately to give pilot full control
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 0, 0, 0, 0, 0, 0, 0, 0)
        
        while True:
            update_telemetry(mav)
            if _last_mode == FLIGHT_MODE:
                log("RESUME", f"Switch turned back to {FLIGHT_MODE}. Resuming script control.")
                if is_landing_phase:
                    log("LAND", "Re-engaging autonomous landing sequence.")
                    mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9) # LAND
                time.sleep(0.5)
                break
            time.sleep(0.1)
        return True
    return False

def execute_rc_move(mav, roll, pitch, label):
    """
    Pushes the virtual RC sticks to move the drone in LOITER, 
    then centers them to brake using optical flow.
    """
    log("MOVE", f"Executing: {label} (Roll: {roll}, Pitch: {pitch})")
    
    start_time = time.time()
    
    # 1. Hold the sticks to move
    while time.time() - start_time < MOVE_DURATION_S:
        if check_pause(mav, is_landing_phase=False):
            start_time = time.time() # Reset movement timer if paused
            
        # Send Ch1 (Roll), Ch2 (Pitch), Ch3 (Throttle = Neutral to hold Alt), Ch4 (Yaw = Neutral)
        mav.mav.rc_channels_override_send(
            mav.target_system, mav.target_component, 
            roll, pitch, RC_NEUTRAL, RC_NEUTRAL, 0, 0, 0, 0
        )
        time.sleep(0.1)
        
    # 2. Center sticks to brake and settle
    log("BRAKE", "Centering sticks to brake...")
    start_settle = time.time()
    while time.time() - start_settle < SETTLE_TIME_S:
        check_pause(mav, is_landing_phase=False)
        mav.mav.rc_channels_override_send(
            mav.target_system, mav.target_component, 
            RC_NEUTRAL, RC_NEUTRAL, RC_NEUTRAL, RC_NEUTRAL, 0, 0, 0, 0
        )
        time.sleep(0.1)

def main():
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    mav.wait_heartbeat()
    log("INIT", "Connected to Pixhawk. Telemetry OK.")

    # Calculate PWM targets based on neutral + tilt
    pitch_fwd = RC_NEUTRAL - RC_TILT_PWM     # Ch2 < 1500 is Forward
    pitch_bwd = RC_NEUTRAL + RC_TILT_PWM     # Ch2 > 1500 is Backward
    roll_left = RC_NEUTRAL - RC_TILT_PWM     # Ch1 < 1500 is Left
    roll_right = RC_NEUTRAL + RC_TILT_PWM    # Ch1 > 1500 is Right

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
            check_pause(mav, is_landing_phase=False)
            mav.mav.rc_channels_override_send(
                mav.target_system, mav.target_component, 
                RC_NEUTRAL, RC_NEUTRAL, THROTTLE_CLIMB, RC_NEUTRAL, 0, 0, 0, 0
            )
            time.sleep(0.1)

        log("FLIGHT", "Target altitude reached. Stabilizing...")
        
        # Explicitly center all sticks to stop climbing and hold altitude
        for _ in range(20):
            mav.mav.rc_channels_override_send(
                mav.target_system, mav.target_component, 
                RC_NEUTRAL, RC_NEUTRAL, RC_NEUTRAL, RC_NEUTRAL, 0, 0, 0, 0
            )
            time.sleep(0.1)

        # 3. SQUARE LOOP EXECUTION (Pure LOITER RC Overrides)
        for loop_count in range(1, TOTAL_LOOPS + 1):
            log("LOOP", f"Starting Square Loop {loop_count} of {TOTAL_LOOPS}")
            
            # Step A: Forward
            execute_rc_move(mav, roll=RC_NEUTRAL, pitch=pitch_fwd, label="Forward")
            
            # Step B: Left
            execute_rc_move(mav, roll=roll_left, pitch=RC_NEUTRAL, label="Left")
            
            # Step C: Backward
            execute_rc_move(mav, roll=RC_NEUTRAL, pitch=pitch_bwd, label="Backward")
            
            # Step D: Right (Back to start)
            execute_rc_move(mav, roll=roll_right, pitch=RC_NEUTRAL, label="Right")

        # 4. AUTONOMOUS LANDING
        log("LAND", "All loops completed. Initializing landing sequence.")
        mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9) # 9 = LAND mode
        
        # Release RC overrides to let LAND mode descend naturally
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 0, 0, 0, 0, 0, 0, 0, 0)
        
        while (get_rangefinder() or 1) > ALT_LANDED_M:
            check_pause(mav, is_landing_phase=True)
            time.sleep(0.1)

        # 5. DISARM MOTORS
        log("FINISH", "Touchdown detected. Disarming.")
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, RC_NEUTRAL, RC_NEUTRAL, THROTTLE_ZERO, RC_NEUTRAL, 0, 0, 0, 0)
        time.sleep(1)
        mav.mav.command_long_send(
            mav.target_system, mav.target_component, 
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 0, 0, 0, 0, 0, 0
        )

    except KeyboardInterrupt:
        log("EMERGENCY", "Script aborted via keyboard!")
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, RC_NEUTRAL, RC_NEUTRAL, THROTTLE_ZERO, RC_NEUTRAL, 0, 0, 0, 0)
    finally:
        mav.close()

if __name__ == "__main__":
    main()
