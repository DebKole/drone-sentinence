#!/usr/bin/env python3
"""
calibrate_optical_flow.py
=========================
A manual, disarmed calibration tool for Optical Flow X/Y scalers.
Features live tracking feedback and accounts for a 10cm ground offset.
"""

import time
from pymavlink import mavutil

PORT = '/dev/ttyACM0'
BAUD = 921600

_current_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}

def log(tag, msg):
    print(f"\n[{tag} {time.strftime('%H:%M:%S')}] {msg}")

def flush_telemetry(mav):
    """Pumps the MAVLink buffer to keep position data fresh."""
    global _current_pos
    while True:
        msg = mav.recv_match(blocking=False)
        if not msg:
            break
        if msg.get_type() == 'LOCAL_POSITION_NED':
            _current_pos['x'] = msg.x  
            _current_pos['y'] = msg.y  
            _current_pos['z'] = msg.z  

def get_stable_reading(mav, duration=1.5):
    """Reads telemetry for a short duration to ensure EKF has settled."""
    start = time.time()
    while time.time() - start < duration:
        flush_telemetry(mav)
        time.sleep(0.1)

def set_fake_origin(mav):
    """Sets a 0,0 origin so the EKF starts tracking hand movements."""
    mav.mav.set_gps_global_origin_send(mav.target_system, 0, 0, 0)
    mav.mav.set_home_position_send(
        mav.target_system,
        0, 0, 0, 0, 0, 0, 
        [1.0, 0.0, 0.0, 0.0], 0, 0, 0
    )

def live_tracking_phase(mav, axis, start_val, duration=15.0):
    """Provides real-time terminal feedback while the user moves the drone."""
    log("READING", f"Live tracking for {duration}s. Move SMOOTHLY and keep the drone LEVEL!")
    start_time = time.time()
    
    while time.time() - start_time < duration:
        flush_telemetry(mav)
        time_left = duration - (time.time() - start_time)
        
        if axis == 'X':
            moved = _current_pos['x'] - start_val
            print(f"    [Time Left: {time_left:.1f}s] Live Forward (+X) travel: {moved:.3f}m      ", end='\r')
        elif axis == 'Y':
            moved = _current_pos['y'] - start_val
            print(f"    [Time Left: {time_left:.1f}s] Live Right (+Y) travel: {moved:.3f}m      ", end='\r')
            
        time.sleep(0.1)
        
    print("") # Clear the line after countdown finishes
    flush_telemetry(mav)
    return _current_pos['x'] if axis == 'X' else _current_pos['y']

def main():
    log("INIT", "Connecting to Pixhawk...")
    mav = mavutil.mavlink_connection(PORT, baud=BAUD)
    mav.wait_heartbeat()
    log("INIT", "Connected. Requesting EKF Position stream...")
    
    # Request EKF position data
    mav.mav.request_data_stream_send(
        mav.target_system, mav.target_component, 
        mavutil.mavlink.MAV_DATA_STREAM_POSITION, 20, 1
    )

    try:
        print("\n" + "="*60)
        print(" OPTICAL FLOW (X/Y) SCALING CALIBRATION")
        print(" -> Ensure the drone's nose faces the same direction the whole time.")
        print(" -> DO NOT TILT THE DRONE while moving it.")
        print("="*60)
        
        # --- POINT 1: SET ORIGIN ---
        input("\n>> Step 1: Hold the drone steady at your desired test height. Point nose Forward. Press Enter to set Origin...")
        log("EKF", "Zeroing coordinates... Hold still for 3 seconds...")
        set_fake_origin(mav)
        get_stable_reading(mav, 3.0) 
        
        start_x = _current_pos['x']
        start_y = _current_pos['y']
        
        # Account for 10cm (0.10m) physical ground offset
        adjusted_height = (-_current_pos['z']) - 0.10
        
        log("SENSOR", f"Origin Set at X: {start_x:.3f}, Y: {start_y:.3f}, Z (Height): {adjusted_height:.3f}m")

        # --- X AXIS (FORWARD) ---
        print("\n>> Step 2: X-AXIS (FORWARD) MOVEMENT")
        print("   You will have 15 seconds to smoothly slide the drone EXACTLY 1.0m FORWARD and hold it there.")
        input("   >> Press Enter to start the 15-second live tracking...")
        
        final_x = live_tracking_phase(mav, 'X', start_x, duration=15.0)
        moved_x = final_x - start_x
        log("RESULT", f"EKF measured Forward (+X) travel: {moved_x:.3f} m (Expected: 1.000 m)")
        
        fx_adj = (1.0 - moved_x) * 100
        
        print("\n" + "-"*40)
        print(f" -> X-AXIS (FORWARD) DIAGNOSIS:")
        if abs(1.0 - moved_x) < 0.05:
            print("    Perfect! No changes needed for FLOW_FXSCALER.")
        elif moved_x < 1.0:
            print(f"    Drone thinks it moved TOO SHORT. Increase FLOW_FXSCALER.")
            print(f"    Suggested adjustment: Add +{fx_adj:.0f} to your current FLOW_FXSCALER.")
        else:
            print(f"    Drone thinks it moved TOO FAR. Decrease FLOW_FXSCALER.")
            print(f"    Suggested adjustment: Add {fx_adj:.0f} (a negative number) to your current FLOW_FXSCALER.")
        print("-"*40)

        # Update start position for Y movement
        get_stable_reading(mav, 1.0)
        start_x = _current_pos['x']
        start_y = _current_pos['y']

        # --- Y AXIS (RIGHT) ---
        print("\n>> Step 3: Y-AXIS (RIGHT) MOVEMENT")
        print("   You will have 15 seconds to smoothly slide the drone EXACTLY 1.0m RIGHT and hold it there.")
        input("   >> Press Enter to start the 15-second live tracking...")
        
        final_y = live_tracking_phase(mav, 'Y', start_y, duration=15.0)
        moved_y = final_y - start_y
        log("RESULT", f"EKF measured Right (+Y) travel: {moved_y:.3f} m (Expected: 1.000 m)")
        
        fy_adj = (1.0 - moved_y) * 100
        
        print("\n" + "-"*40)
        print(f" -> Y-AXIS (RIGHT) DIAGNOSIS:")
        if abs(1.0 - moved_y) < 0.05:
            print("    Perfect! No changes needed for FLOW_FYSCALER.")
        elif moved_y < 1.0:
            print(f"    Drone thinks it moved TOO SHORT. Increase FLOW_FYSCALER.")
            print(f"    Suggested adjustment: Add +{fy_adj:.0f} to your current FLOW_FYSCALER.")
        else:
            print(f"    Drone thinks it moved TOO FAR. Decrease FLOW_FYSCALER.")
            print(f"    Suggested adjustment: Add {fy_adj:.0f} (a negative number) to your current FLOW_FYSCALER.")
        print("-"*40)

        print("\n" + "="*60)
        print(" CALIBRATION COMPLETE.")
        print(" Update parameters in Mission Planner, reboot the flight controller, and test again!")
        print("="*60)

    except KeyboardInterrupt:
        log("EMERGENCY", "Calibration aborted.")
    finally:
        mav.close()

if __name__ == "__main__":
    main()
