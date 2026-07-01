#!/usr/bin/env python3

import time
import sys
import traceback
from pymavlink import mavutil

# ─── config ───────────────────────────────────────────────
PORT            = '/dev/ttyACM0'
BAUD            = 921600

RC_MID          = 1500
RC_MIN          = 1000
RC_MAX          = 2000

THROTTLE_ZERO   = 1000
THROTTLE_SPINUP = 1200
THROTTLE_HOVER  = 1550
THROTTLE_LAND   = 1350

SPINUP_TIME     = 3.0
CLIMB_TIME      = 6.0
HOVER_TIME      = 8.0
DESCEND_TIME    = 8.0

ALT_LANDED_M    = 0.3
SETPOINT_HZ     = 10
# ──────────────────────────────────────────────────────────

# ─── state cache ──────────────────────────────────────────
_last_alt = None
_last_mode = None
# ──────────────────────────────────────────────────────────

# ─── logger ───────────────────────────────────────────────
def log_info(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"  [INFO     {ts}] {msg}")

def log_warn(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"  [WARN     {ts}] {msg}")

def log_error(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"  [ERROR    {ts}] {msg}", file=sys.stderr)

def log_override(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"\n  [OVERRIDE {ts}] {msg}")
# ──────────────────────────────────────────────────────────

# ─── safe recv wrapper ────────────────────────────────────
def safe_recv(mav, msg_type, blocking=False, timeout=1):
    retries = 10 if blocking else 3
    for _ in range(retries):
        try:
            return mav.recv_match(type=msg_type, blocking=blocking, timeout=timeout)
        except TypeError:
            continue
        except Exception as e:
            log_error(f"safe_recv({msg_type}) unexpected error: {e}")
            return None
    return None
# ──────────────────────────────────────────────────────────

def request_streams(mav):
    log_info("Requesting data streams from FCU...")
    try:
        mav.mav.request_data_stream_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 10, 1
        )
        mav.mav.request_data_stream_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1
        )
        time.sleep(0.5)
        log_info("Data stream requests sent")
    except Exception as e:
        log_error(f"Stream request failed: {e}")

def verify_telemetry(mav):
    log_info("Verifying VFR_HUD telemetry stream...")
    global _last_alt
    for attempt in range(20):
        msg = safe_recv(mav, 'VFR_HUD', blocking=True, timeout=1)
        if msg:
            _last_alt = msg.alt
            log_info(f"VFR_HUD confirmed — alt={msg.alt:.2f}m")
            return True
        log_warn(f"No VFR_HUD yet ({attempt+1}/20)...")
    return False

def connect(port, baud):
    print(f"\n[*] Connecting to Pixhawk on {port} @ {baud}baud...")
    try:
        mav = mavutil.mavlink_connection(port, baud=baud)
        mav.wait_heartbeat()
        log_info(f"Connected | system={mav.target_system} component={mav.target_component}")
        request_streams(mav)
        return mav
    except Exception as e:
        log_error(f"Connection failed: {e}")
        sys.exit(1)

# ─── FLIGHT MODE & PAUSE LOGIC ────────────────────────────
def update_flight_mode(mav, blocking=False):
    """
    Fetches the current flight mode from HEARTBEAT messages.
    Uses non-blocking calls by default to prevent lagging the loops.
    """
    global _last_mode
    
    if blocking:
        for _ in range(20):
            msg = safe_recv(mav, 'HEARTBEAT', blocking=True, timeout=1)
            if msg and msg.type != mavutil.mavlink.MAV_TYPE_GCS:
                _last_mode = mavutil.mode_string_v10(msg)
                return _last_mode
                
    # Non-blocking update: drain the buffer of any pending heartbeats
    while True:
        msg = safe_recv(mav, 'HEARTBEAT', blocking=False)
        if msg is None:
            break
        if msg.type != mavutil.mavlink.MAV_TYPE_GCS:
            _last_mode = mavutil.mode_string_v10(msg)
            
    # Fallback to internal pymavlink cache if we missed it in the stream
    if _last_mode is None and 'HEARTBEAT' in mav.messages:
        msg = mav.messages['HEARTBEAT']
        if msg.type != mavutil.mavlink.MAV_TYPE_GCS:
            _last_mode = mavutil.mode_string_v10(msg)

    return _last_mode

def is_armed(mav):
    # Relies on the pymavlink cache for instant lookup
    if 'HEARTBEAT' in mav.messages:
        msg = mav.messages['HEARTBEAT']
        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
    return False

def get_altitude(mav):
    global _last_alt
    while True:
        msg = safe_recv(mav, 'VFR_HUD', blocking=False)
        if msg is None:
            break
        _last_alt = msg.alt
    return _last_alt

def send_rc_override(mav, throttle, roll=RC_MID, pitch=RC_MID, yaw=RC_MID):
    try:
        mav.mav.rc_channels_override_send(
            mav.target_system, mav.target_component,
            roll, pitch, throttle, yaw, 0, 0, 0, 0
        )
    except Exception as e:
        log_error(f"send_rc_override failed: {e}")

def clear_rc_override(mav):
    try:
        mav.mav.rc_channels_override_send(
            mav.target_system, mav.target_component,
            0, 0, 0, 0, 0, 0, 0, 0
        )
    except Exception as e:
        log_error(f"clear_rc_override failed: {e}")

def handle_pause(mav):
    """
    Checks the flight mode. If switched out of STABILIZE (e.g., to ALT_HOLD),
    releases RC overrides and blocks execution until switched back.
    """
    current_mode = update_flight_mode(mav, blocking=False)
    
    if current_mode is not None and current_mode != 'STABILIZE':
        clear_rc_override(mav)
        log_override(f"Mode is {current_mode}. Script PAUSED. Pilot has control.")
        
        # Block script execution here until mode returns to STABILIZE
        while True:
            time.sleep(0.1)
            mode = update_flight_mode(mav, blocking=False)
            if mode == 'STABILIZE':
                break
                
        log_override("Mode returned to STABILIZE. Script RESUMING in 1 second...")
        time.sleep(1.0) 
        return True 
        
    return False
# ──────────────────────────────────────────────────────────

def arm(mav):
    print("\n[*] Arming...")
    try:
        mode = update_flight_mode(mav, blocking=True)
        log_info(f"Current mode: {mode}")

        if mode is None:
            log_error("Could not read mode from FCU")
            sys.exit(1)

        if 'STAB' not in str(mode).upper():
            log_error(f"Expected STABILIZE mode, got {mode}")
            sys.exit(1)

        mav.mav.command_long_send(
            mav.target_system, mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )

        for _ in range(20):
            # Update cache to catch the arming state change
            update_flight_mode(mav, blocking=False) 
            if is_armed(mav):
                log_info("ARMED successfully")
                return
            time.sleep(0.5)

        log_error("Arming failed — check Mission Planner for prearm errors")
        sys.exit(1)

    except SystemExit:
        raise
    except Exception as e:
        log_error(f"arm() exception: {e}")
        sys.exit(1)

def disarm(mav):
    print("\n[*] Disarming...")
    try:
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

        log_warn("Disarm not confirmed — use RC killswitch")
    except Exception as e:
        log_error(f"disarm() exception: {e}")

def ramp_throttle(mav, start, end, duration, label):
    print(f"\n[*] {label}")
    interval = 1.0 / SETPOINT_HZ
    steps    = int(duration * SETPOINT_HZ)
    throttle = start

    i = 0
    while i <= steps:
        if handle_pause(mav):
            continue 

        progress = i / steps
        throttle = int(start + (end - start) * progress)
        send_rc_override(mav, throttle=throttle)
        print(f"    throttle={throttle}", end='\r')
        
        time.sleep(interval)
        i += 1
    print()

def hold_throttle(mav, throttle, duration, label, exit_condition=None):
    print(f"\n[*] {label}")
    interval   = 1.0 / SETPOINT_HZ
    iterations = int(duration * SETPOINT_HZ)

    i = 0
    while i < iterations:
        if handle_pause(mav):
            continue

        send_rc_override(mav, throttle=throttle)
        alt = get_altitude(mav)
        remaining = duration - (i * interval)

        if alt is not None:
            print(f"    throttle={throttle} | alt={alt:.2f}m | {remaining:.1f}s left", end='\r')
            if exit_condition and exit_condition(alt):
                print(f"\n[+] Exit condition met at {alt:.2f}m")
                return
        else:
            print(f"    throttle={throttle} | alt=?.??m | {remaining:.1f}s left", end='\r')

        time.sleep(interval)
        i += 1
    print()

def emergency_stop(mav):
    print("\n[!] EMERGENCY STOP — cutting throttle")
    try:
        for _ in range(20):
            send_rc_override(mav, throttle=THROTTLE_ZERO)
            time.sleep(0.05)
        disarm(mav)
    except Exception as e:
        log_error(f"emergency_stop() exception: {e}")

def main():
    mav = connect(PORT, BAUD)

    if not verify_telemetry(mav):
        mav.close()
        sys.exit(1)

    print("\n[*] Pre-flight checks...")
    mode = update_flight_mode(mav, blocking=True)
    alt  = get_altitude(mav)
    log_info(f"Mode            : {mode}")
    log_info(f"Altitude        : {alt:.2f}m" if alt is not None else "Altitude        : unknown")
    log_info(f"Hover throttle  : {THROTTLE_HOVER}")
    
    input("\n[*] Verify mode is STABILIZE. Press ENTER to arm and fly (Ctrl+C to abort)...")

    try:
        arm(mav)
        time.sleep(1)

        ramp_throttle(mav, THROTTLE_ZERO, THROTTLE_SPINUP, SPINUP_TIME, "Spinning up motors")
        ramp_throttle(mav, THROTTLE_SPINUP, THROTTLE_HOVER, CLIMB_TIME, "Climbing")
        hold_throttle(mav, THROTTLE_HOVER, HOVER_TIME, "Hovering")
        ramp_throttle(mav, THROTTLE_HOVER, THROTTLE_LAND, 3.0, "Reducing throttle for descent")
        hold_throttle(mav, THROTTLE_LAND, DESCEND_TIME, "Descending", exit_condition=lambda a: a <= ALT_LANDED_M)
        ramp_throttle(mav, THROTTLE_LAND, THROTTLE_ZERO, 1.0, "Cutting throttle")

        time.sleep(1)
        disarm(mav)
        log_info("Flight complete")

    except KeyboardInterrupt:
        emergency_stop(mav)
    except Exception as e:
        log_error(f"Unhandled exception in flight sequence: {e}")
        log_error(traceback.format_exc())
        emergency_stop(mav)
    finally:
        mav.close()

if __name__ == "__main__":
    main()
