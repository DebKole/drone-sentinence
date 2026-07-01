#!/usr/bin/env python3
"""
drone_timed_hover_land.py
=========================
Optimized for Jetson Orin Nano + Pixhawk.
Features: Timed autonomous hover, automatic landing, and safety RC pause/resume.
"""

import time
import sys
import traceback
from pymavlink import mavutil

# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# -- MAVLink & Port --
PORT = '/dev/ttyACM0'
BAUD = 921600

# -- Timed Hover Settings --
TARGET_ALT_M = 1.0           # Altitude the drone will climb to (meters)
HOVER_DURATION_S = 10.0      # How long the drone will hover before landing

# -- RC Overrides & Flight Targets --
THROTTLE_ZERO = 1000
THROTTLE_CLIMB = 1620
THROTTLE_HOVER = 1550
THROTTLE_LAND = 1380
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
    """
    Pumps the MAVLink buffer to keep all message caches fresh.
    Prevents message starvation across functions.
    """
    global _last_rng, _last_mode
    
    while True:
        msg = mav.recv_match(blocking=False)
        if not msg:
            break  # Buffer is empty, move on
            
        msg_type = msg.get_type()
        if msg_type == 'DISTANCE_SENSOR':
            _last_rng = msg.current_distance / 100.0
        elif msg_type == 'HEARTBEAT':
            _last_mode = mavutil.mode_string_v10(msg)

def get_rangefinder():
    return _last_rng

def check_pause(mav, is_landing_phase=False):
    """
    Monitors flight mode. If the pilot switches away from the script's intended
    mode, it strips overrides and enters a blocking pause loop until the pilot
    flicks the switch back to LOITER.
    """
    global _last_mode
    update_telemetry(mav)
    
    should_pause = False
    if is_landing_phase:
        # During landing, ArduPilot mode is overridden to LAND. If you switch to 
        # a manual mode (like ALT_HOLD or STABILIZE) to take over, we pause.
        if _last_mode and _last_mode not in ['LAND', 'LOITER']:
            should_pause = True
    else:
        # During climb/hover phases, the flight mode must stay in LOITER.
        if _last_mode and _last_mode != FLIGHT_MODE:
            should_pause = True
            
    if should_pause:
        log("PAUSE", f"Pilot manually took control (Mode: {_last_mode}). Script paused.")
        # Release RC overrides immediately to give 100% manual stick control to pilot
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Blocking loop until pilot returns switch to LOITER
        while True:
            update_telemetry(mav)
            if _last_mode == FLIGHT_MODE:
                log("RESUME", f"Switch turned back to {FLIGHT_MODE}. Resuming script control.")
                if is_landing_phase:
                    # Re-engage Pixhawk autonomous LAND mode if we were in the middle of landing
                    log("LAND", "Re-engaging autonomous landing sequence.")
                    mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)
                    time.sleep(0.5)
                break
            time.sleep(0.1)
        return True # Signaling a pause/resume event occurred
    return False

def main():
    # Setup MAVLink Connection
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    mav.wait_heartbeat()
    log("INIT", "Connected to Pixhawk. Telemetry OK.")

    try:
        input(f"[*] Flight Mode must be {FLIGHT_MODE}. Press Enter to START MISSION...")
        
        # Prime the telemetry data
        for _ in range(10):
            update_telemetry(mav)
            time.sleep(0.05)
        
        # 1. ARM THE MOTORS
        mav.mav.command_long_send(
            mav.target_system, mav.target_component, 
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0
        )
        log("FLIGHT", "Armed. Climbing...")
        
        # 2. CLIMB TO TARGET ALTITUDE
        while (get_rangefinder() or 0) < TARGET_ALT_M:
            check_pause(mav, is_landing_phase=False)
                
            mav.mav.rc_channels_override_send(
                mav.target_system, mav.target_component, 
                1500, 1500, THROTTLE_CLIMB, 1500, 0, 0, 0, 0
            )
            time.sleep(0.1)

        # 3. TIMED HOVER PHASE
        log("FLIGHT", f"Target altitude reached. Hovering for {HOVER_DURATION_S} seconds...")
        accumulated_hover_time = 0.0
        last_loop_time = time.time()
        
        while accumulated_hover_time < HOVER_DURATION_S:
            # If a pause happened, check_pause blocks here until resume
            if check_pause(mav, is_landing_phase=False):
                # Reset clock tracking immediately upon resume so pause time is ignored
                last_loop_time = time.time()
            
            now = time.time()
            dt = now - last_loop_time
            last_loop_time = now
            
            accumulated_hover_time += dt
            
            # Maintain hover throttle overrides
            mav.mav.rc_channels_override_send(
                mav.target_system, mav.target_component, 
                1500, 1500, THROTTLE_HOVER, 1500, 0, 0, 0, 0
            )
            print(f"[HOVERING] Time remaining: {max(0.0, HOVER_DURATION_S - accumulated_hover_time):.1f}s", end='\r')
            time.sleep(0.1)

        print() # Clear progress line

        # 4. TIMED AUTONOMOUS LANDING
        log("LAND", "Hover timer complete. Initializing landing sequence.")
        
        # Clear manual overrides to switch Pixhawk directly to autonomous LAND mode
        mav.mav.rc_channels_override_send(mav.target_system, mav.target_component, 0, 0, 0, 0, 0, 0, 0, 0)
        time.sleep(0.1)
        
        # Send Land Mode Command (Custom Mode 9 = LAND)
        mav.mav.set_mode_send(mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 9)
        
        # Wait for Touchdown
        while (get_rangefinder() or 1) > ALT_LANDED_M:
            check_pause(mav, is_landing_phase=True)
            time.sleep(0.1)

        # 5. DISARM MOTORS AFTER TOUCHDOWN
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
