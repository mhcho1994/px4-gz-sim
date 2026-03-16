import math
import time
from pymavlink import mavutil


def _nan_to_zero(x: float) -> float:
    """Convert NaN to 0.0 for MAVLink mission item fields."""
    if isinstance(x, float) and math.isnan(x):
        return 0.0
    return x




# def verify_takeoff_at_seq0(m) -> bool:
#     """
#     Read back mission item 0 and verify that it is NAV_TAKEOFF.
#     """
#     msg = read_mission_item(m, 0)
#     if msg is None:
#         print("Failed to read back mission item 0")
#         return False

#     print(f"Read-back seq0 command={msg.command}, frame={msg.frame}")

#     return msg.command == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF


# def reset_mission_current(m) -> None:
#     """
#     Reset mission current index to 0.
#     """
#     # Older/common pymavlink way
#     m.mav.mission_set_current_send(
#         m.target_system,
#         m.target_component,
#         0,
#     )
#     print("Requested mission current reset to seq=0")


def set_mode_auto(m) -> bool:
    """
    Switch vehicle to AUTO mode and wait for COMMAND_ACK.
    """
    mode_id = m.mode_mapping().get("AUTO")
    if mode_id is None:
        print("AUTO mode is not available in mode mapping")
        return False

    m.mav.command_long_send(
        m.target_system,
        m.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_MODE,
        0,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
        0, 0, 0, 0, 0,
    )

    return wait_command_ack(m, mavutil.mavlink.MAV_CMD_DO_SET_MODE, timeout=5.0)


    ==============================================================================

def prepare_and_start_auto(m, mission_items: list[dict]) -> bool:
    """
    Robust sequence for Copter AUTO mission start:
      1) clear mission
      2) upload mission
      3) read back seq 0 and verify TAKEOFF
      4) optionally set MIS_RESTART=1
      5) reset mission current to 0
      6) switch to AUTO
    """

    print("=== Clearing previous mission ===")
    if not clear_mission(m):
        return False

    print("=== Uploading new mission ===")
    if not upload_mission_items(m, mission_items):
        return False

    print("=== Verifying mission seq 0 ===")
    if not verify_takeoff_at_seq0(m):
        print("Mission seq 0 is not NAV_TAKEOFF on vehicle")
        return False

    print("=== Checking MIS_RESTART ===")
    mis_restart = get_param(m, "MIS_RESTART")
    print(f"MIS_RESTART current value: {mis_restart}")

    # Recommended for debugging: force restart from beginning on AUTO entry
    if mis_restart is None or abs(mis_restart - 1.0) > 1e-3:
        ok = set_param_int(m, "MIS_RESTART", 1)
        if not ok:
            print("Failed to set MIS_RESTART=1")
            return False

    print("=== Resetting mission current ===")
    reset_mission_current(m)
    time.sleep(0.5)

    print("=== Switching to AUTO ===")
    if not set_mode_auto(m):
        return False

    print("AUTO mode request accepted")
    return True


connect = "udpin:127.0.0.1:14550"

m = mavutil.mavlink_connection(
    connect,
    source_system=246,
    source_component=190,
)

print("Waiting heartbeat...")
msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=20)
if msg is None:
    raise RuntimeError("Heartbeat timeout")

m.target_system = 1
m.target_component = 1

print("Setting GUIDED...")
m.set_mode("GUIDED")
time.sleep(1.0)

print("Arming...")
m.arducopter_arm()
m.motors_armed_wait()

ok = prepare_and_start_auto(m, mission_items)
print(f"prepare_and_start_auto => {ok}")


=========================================================================================


# 1) heartbeat
self._wait_heartbeat(m)

m.target_system = 1
m.target_component = 1

# 2) GUIDED
self._set_status(MissionState.SETTING_GUIDED, "switching to GUIDED")
self._set_mode(m, "GUIDED")

# 3) arm
self._arm(m)

# 4) clear/upload/verify/reset/AUTO
self._set_status(MissionState.CLEARING_MISSION, "clearing old mission")
if not clear_mission(m):
    raise RuntimeError("failed to clear mission")

self._set_status(MissionState.UPLOADING_MISSION, "uploading mission")
if not upload_mission_items(m, mission_items):
    raise RuntimeError("failed to upload mission")

if not verify_takeoff_at_seq0(m):
    raise RuntimeError("vehicle mission seq0 is not NAV_TAKEOFF")

mis_restart = get_param(m, "MIS_RESTART")
print(f"MIS_RESTART={mis_restart}")
if mis_restart is None or abs(mis_restart - 1.0) > 1e-3:
    if not set_param_int(m, "MIS_RESTART", 1):
        raise RuntimeError("failed to set MIS_RESTART=1")

reset_mission_current(m)
time.sleep(0.5)

if not set_mode_auto(m):
    raise RuntimeError("failed to switch to AUTO")


============================================================================================

def dump_vehicle_mission(m, n=6, timeout=3.0):
    for seq in range(n):
        m.mav.mission_request_int_send(
            m.target_system,
            m.target_component,
            seq,
        )

        t0 = time.time()
        got = None
        while time.time() - t0 < timeout:
            msg = m.recv_match(
                type=["MISSION_ITEM_INT", "MISSION_ITEM", "STATUSTEXT"],
                blocking=True,
                timeout=0.5,
            )
            if msg is None:
                continue

            if msg.get_type() == "STATUSTEXT":
                print(f"AP: {msg.text}")
                continue

            if getattr(msg, "seq", None) == seq:
                got = msg
                break

        if got is None:
            print(f"vehicle[{seq}] = <read timeout>")
        else:
            print(
                f"vehicle[{seq}] "
                f"cmd={got.command} frame={got.frame}"
            )



========================================================================

def _monitor_current_mission(self, m, mission_items: list[dict], duration: float = 30.0):
    """
    Monitor current mission progress for a fixed duration.
    """
    seq_to_wp = self._build_waypoint_seq_map(mission_items)

    t0 = time.time()
    last_seq = None

    while time.time() - t0 < duration:
        msg = m.recv_match(type="MISSION_CURRENT", blocking=True, timeout=1.0)
        if msg is None:
            continue

        seq = msg.seq
        if seq == last_seq:
            continue

        last_seq = seq
        cmd = mission_items[seq]["command"] if 0 <= seq < len(mission_items) else None
        wp_idx = seq_to_wp.get(seq)

        if wp_idx is not None:
            print(f"[MISSION_CURRENT] seq={seq}, cmd={cmd}, waypoint #{wp_idx}")
        else:
            print(f"[MISSION_CURRENT] seq={seq}, cmd={cmd}, non-waypoint item")


self._set_status(MissionState.AUTO, "switching to AUTO")
self._set_mode(m, "AUTO")

time.sleep(1.0)

msg = m.recv_match(type="MISSION_CURRENT", blocking=True, timeout=2.0)
if msg is not None:
    seq = msg.seq
    print(f"Current mission seq={seq}")
else:
    print("MISSION_CURRENT not received")


def _wait_mission_item_reached(self, m, timeout: float = 10.0):
    """
    Wait for mission item reached notification.
    Returns reached seq or None.
    """
    msg = m.recv_match(type="MISSION_ITEM_REACHED", blocking=True, timeout=timeout)
    if msg is None:
        return None
    return msg.seq


=========================================================================================

def wait_vehicle_ready(m):

    print("Waiting heartbeat...")
    m.wait_heartbeat()

    print("Waiting pre-arm checks...")

    while True:

        msg = m.recv_match(type=["SYS_STATUS", "STATUSTEXT"], blocking=True)

        if msg.get_type() == "STATUSTEXT":
            if "PreArm:" in msg.text:
                print(msg.text)

        if msg.get_type() == "SYS_STATUS":
            if msg.onboard_control_sensors_health & mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK:
                print("Vehicle ready to arm")
                return
            

from pymavlink import mavutil

MAV_SYS_STATUS_PREARM_CHECK = mavutil.mavlink.MAV_SYS_STATUS_PREARM_CHECK


def wait_prearm_ok(m, timeout=30):
    """
    Wait until vehicle passes pre-arm checks.
    """
    import time
    t0 = time.time()

    while time.time() - t0 < timeout:

        msg = m.recv_match(type="SYS_STATUS", blocking=True, timeout=1)

        if msg is None:
            continue

        health = msg.onboard_control_sensors_health

        if health & MAV_SYS_STATUS_PREARM_CHECK:
            print("Pre-arm check OK")
            return True

    print("Pre-arm check timeout")
    return False


def monitor_prearm_messages(m, duration=10):
    import time
    t0 = time.time()

    while time.time() - t0 < duration:

        msg = m.recv_match(type="STATUSTEXT", blocking=True, timeout=1)

        if msg is None:
            continue

        text = msg.text

        if "PreArm:" in text:
            print("PREARM ERROR:", text)
        else:
            print("AP:", text)




=====================================================================

def _get_current_mission_seq(self, m, timeout: float = 1.0):
    """
    Read current mission item index from MISSION_CURRENT.
    Returns seq or None if timeout.
    """
    msg = m.recv_match(type="MISSION_CURRENT", blocking=True, timeout=timeout)
    if msg is None:
        return None
    return msg.seq

self._set_status(MissionState.AUTO, "switching to AUTO")
self._set_mode(m, "AUTO")

seq = self._get_current_mission_seq(m, timeout=2.0)
print(f"Current mission seq = {seq}")

def _build_waypoint_seq_map(self, mission_items: list[dict]):
    """
    Build mapping from mission seq -> waypoint number.
    Only NAV_WAYPOINT items are counted as waypoints.
    """
    wp_num = 0
    seq_to_wp = {}

    for seq, item in enumerate(mission_items):
        if item["command"] == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
            wp_num += 1
            seq_to_wp[seq] = wp_num

    return seq_to_wp

seq = self._get_current_mission_seq(m, timeout=2.0)

if seq is None:
    print("Could not read MISSION_CURRENT")
else:
    wp_idx = seq_to_wp.get(seq)
    if wp_idx is not None:
        print(f"Currently executing waypoint #{wp_idx} (mission seq={seq})")
    else:
        print(f"Currently executing non-waypoint mission item seq={seq}")

def _monitor_current_mission(self, m, mission_items: list[dict], duration: float = 30.0):
    """
    Monitor current mission progress for a fixed duration.
    """
    seq_to_wp = self._build_waypoint_seq_map(mission_items)

    t0 = time.time()
    last_seq = None

    while time.time() - t0 < duration:
        msg = m.recv_match(type="MISSION_CURRENT", blocking=True, timeout=1.0)
        if msg is None:
            continue

        seq = msg.seq
        if seq == last_seq:
            continue

        last_seq = seq
        cmd = mission_items[seq]["command"] if 0 <= seq < len(mission_items) else None
        wp_idx = seq_to_wp.get(seq)

        if wp_idx is not None:
            print(f"[MISSION_CURRENT] seq={seq}, cmd={cmd}, waypoint #{wp_idx}")
        else:
            print(f"[MISSION_CURRENT] seq={seq}, cmd={cmd}, non-waypoint item")

def _wait_mission_item_reached(self, m, timeout: float = 10.0):
    """
    Wait for mission item reached notification.
    Returns reached seq or None.
    """
    msg = m.recv_match(type="MISSION_ITEM_REACHED", blocking=True, timeout=timeout)
    if msg is None:
        return None
    return msg.seq


reached_seq = self._wait_mission_item_reached(m, timeout=30.0)
if reached_seq is not None:
    print(f"Reached mission item seq={reached_seq}")