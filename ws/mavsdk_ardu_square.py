#!/usr/bin/env python3
import time
import math
from pymavlink import mavutil

EARTH_RADIUS_M = 6378137.0

def add_north_east_m_to_gps(lat_deg, lon_deg, north_m, east_m):
    lat_rad = math.radians(lat_deg)
    dlat = north_m / EARTH_RADIUS_M
    dlon = east_m / (EARTH_RADIUS_M * math.cos(lat_rad))
    return (lat_deg + math.degrees(dlat), lon_deg + math.degrees(dlon))

def wait_heartbeat(m):
    m.wait_heartbeat()
    print(f"Heartbeat OK (sys={m.target_system}, comp={m.target_component})")

def get_home_latlon(m, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type=["HOME_POSITION", "GLOBAL_POSITION_INT"], blocking=True, timeout=1.0)
        if not msg:
            continue
        if msg.get_type() == "HOME_POSITION":
            return msg.latitude / 1e7, msg.longitude / 1e7
        if msg.get_type() == "GLOBAL_POSITION_INT":
            return msg.lat / 1e7, msg.lon / 1e7
    raise TimeoutError("No home/global position")

def clear_mission(m):
    m.mav.mission_clear_all_send(m.target_system, m.target_component)
    ack = m.recv_match(type="MISSION_ACK", blocking=True, timeout=5.0)
    if not ack:
        raise RuntimeError("MISSION_CLEAR_ALL: no ACK")

def upload_mission_int(m, items):
    m.mav.mission_count_send(m.target_system, m.target_component, len(items))

    for _ in range(len(items)):
        req = m.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST"], blocking=True, timeout=10.0)
        if not req:
            raise RuntimeError("MISSION upload: request timeout")
        seq = req.seq
        it = items[seq]

        m.mav.mission_item_int_send(
            m.target_system, m.target_component,
            seq,
            it["frame"],
            it["command"],
            0,  # current=0 (set_current로 따로 지정)
            it["autocontinue"],
            it["p1"], it["p2"], it["p3"], it["p4"],
            int(it["lat"] * 1e7),
            int(it["lon"] * 1e7),
            float(it["alt"]),
            0
        )

    ack = m.recv_match(type="MISSION_ACK", blocking=True, timeout=10.0)
    if not ack:
        raise RuntimeError("MISSION upload: no ACK")
    print("Mission upload OK:", ack)

def set_mode(m, mode_name: str):
    m.set_mode(mode_name)
    time.sleep(0.5)

def wait_altitude(m, target_rel_alt_m: float, timeout=25.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
        if not msg:
            continue
        rel_alt_m = msg.relative_alt / 1000.0
        if rel_alt_m >= 0.8 * target_rel_alt_m:
            print(f"Reached rel_alt ~ {rel_alt_m:.2f} m")
            return
    raise TimeoutError("Did not reach target altitude")

def wait_disarmed(m, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
        if not hb:
            continue
        armed = (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
        if not armed:
            print("Disarmed")
            return
    raise TimeoutError("Did not disarm in time")

def guided_takeoff(m, takeoff_alt_m: float):
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0,
        0, 0,
        takeoff_alt_m
    )
    ack = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=3.0)
    if ack:
        print("TAKEOFF ACK:", ack)

def main():
    # ArduPilot SITL Port (can be different from PX4's 14540)
    m = mavutil.mavlink_connection("udp:127.0.0.1:14550")
    wait_heartbeat(m)

    home_lat, home_lon = get_home_latlon(m)
    print("Home:", home_lat, home_lon)

    SIZE = 5.0
    ALT = 5.0

    # Square: (5,0)->(5,5)->(0,5)->(0,0)
    corners = [
        (SIZE, 0.0),
        (SIZE, SIZE),
        (0.0, SIZE),
        (0.0, 0.0),
    ]

    items = []

    # 4 WAYPOINTs
    for north_m, east_m in corners:
        lat, lon = add_north_east_m_to_gps(home_lat, home_lon, north_m, east_m)
        items.append(dict(
            frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            autocontinue=1,
            p1=0,           # hold time (s)
            p2=1.0,         # acceptance radius (m)
            p3=0.0,         # pass radius (m)
            p4=float("nan"),# yaw
            lat=lat, lon=lon, alt=ALT
        ))

    # Final: LAND (Land at current position: suppose lat/lon=0,0)
    items.append(dict(
        frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        command=mavutil.mavlink.MAV_CMD_NAV_LAND,
        autocontinue=1,
        p1=0, p2=0, p3=0, p4=float("nan"),
        lat=0.0, lon=0.0, alt=0.0
    ))

    clear_mission(m)
    upload_mission_int(m, items)

    # Start from the first waypoint (0-based index)
    m.mav.mission_set_current_send(m.target_system, m.target_component, 0)

    # GUIDED → Arm → Takeoff
    set_mode(m, "GUIDED")

    m.arducopter_arm()
    m.motors_armed_wait()
    print("Armed")

    guided_takeoff(m, ALT)
    wait_altitude(m, ALT, timeout=25.0)

    # AUTO -> waypoint and LAND
    set_mode(m, "AUTO")
    print("AUTO mission running (will LAND at end).")

    # After landing, wait for disarm (optional)
    wait_disarmed(m, timeout=180.0)
    print("Mission complete.")

if __name__ == "__main__":
    main()