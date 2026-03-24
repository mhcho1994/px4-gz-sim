#!/usr/bin/env python3
"""
PX4 Mission Runner (YAML-driven, pymavlink-based)

Purpose
-------
MAVLink mission runner for PX4 SITL experiments.

This script reads a scenario.yaml file and executes the mission using
pymavlink with PX4 SITL. 

  1) Parses scenario.yaml
  2) Connects to PX4 via pymavlink
  3) Waits for heartbeat and basic readiness
  4) Builds raw MAVLink mission items
  5) Clears and uploads mission
  6) Arms vehicle
  7) Switches to AUTO.MISSION
  8) Sends MISSION_START
  9) Monitors mission progress and waits for disarm

Notes
-----
- This version is closer in spirit to your ArduPilot commander.
- It uses raw MAVLink mission upload rather than MAVSDK high-level APIs.
- QGC can run independently for monitoring.
"""

from __future__ import annotations

import time
import threading
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pymavlink import mavutil


class MissionState(Enum):
    """
    Finite state machine describing the mission runner lifecycle.
    """

    IDLE = auto()               # Runner created but not started
    CONNECTING = auto()         # MAVLink connection in progress
    HEARTBEAT_OK = auto()       # Vehicle heartbeat received
    CLEARING_MISSION = auto()   # Clear previous mission data
    UPLOADING_MISSION = auto()  # Upload mission data
    MISSION_UPLOADED = auto()   # Mission uploaded
    ARMING = auto()             # Sending arm command
    ARMED = auto()              # Vehicle successfully armed
    SETTING_AUTO = auto()       # Setting auto mode
    STARTING_MISSION = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()             # Error occurred


@dataclass
class MissionStatus:
    """
    Snapshot of mission runner state.

    This object is returned to external callers so they can
    monitor progress without accessing internal variables.
    """

    state: MissionState             # Current FSM state
    message: str = ""               # Human-readable status message
    done: bool = False              # True when runner finished execution
    success: bool = False           # True if mission completed successfully
    error: Optional[str] = None     # Error message if FAILED


def _get(d: Dict[str, Any], path: str, default=None):
    """
    Utility function to retrieve nested dictionary values.

    Example
    -------
    _get(config, "autopilots.px4.mavlink.connect_url")

    Equivalent to:
        config["autopilots"]["px4"]["mavlink"]["connect_url"]

    but safe against missing keys.
    """

    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_connect_url(scn: Dict[str, Any]) -> str:
    """
    Convert scenario MAVLink connection URLs into pymavlink format.

    Supported input examples
    ------------------------
    Scenario format:
        udp:127.0.0.1:14550

    Also accepts:
        udp://127.0.0.1:14550
        udpin:127.0.0.1:14550
        tcp:127.0.0.1:5760

    Output examples
    ---------------
    pymavlink format:
        udpin:127.0.0.1:14550

    If the scenario file does not explicitly define a connection URL,
    this function falls back to the default SITL UDP output port.
    """

    url = _get(scn, "autopilots.px4.mavlink.connect_url")

    if isinstance(url, str) and url.strip():
        url = url.strip()

        # Scenario style: udp:127.0.0.1:14550
        if url.startswith("udp:"):
            hostport = url[len("udp:"):]
            return f"udpin:{hostport}"

        # Legacy style: udp://127.0.0.1:14550
        if url.startswith("udp://"):
            hostport = url[len("udp://"):]
            return f"udpin:{hostport}"

        # Already pymavlink-compatible
        if url.startswith(("udpin:", "udpout:", "tcp:", "tcpin:", "tcpout:", "serial:")):
            return url
        
        raise ValueError(
            f"Unsupported MAVLink connect_url format: {url!r}. "
            "Expected forms like 'udp:127.0.0.1:14550' or 'udpin:127.0.0.1:14550'."
        )

    # Default connection port used by SITL
    return f"udpin:127.0.0.1:14550"


def build_items_from_scenario(scn: Dict[str, Any]) -> Tuple[list[dict], float, bool]:
    """
    Build raw MAVLink mission items from scenario.yaml.

    Supported commands:
      22  MAV_CMD_NAV_TAKEOFF
      16  MAV_CMD_NAV_WAYPOINT
      178 MAV_CMD_DO_CHANGE_SPEED
      21  MAV_CMD_NAV_LAND

    Returns
    -------
    items : list[dict]
        Raw mission item definitions.
    takeoff_alt : float
        Scenario takeoff altitude.
    has_land : bool
        Whether mission includes explicit land command.
    """

    home = _get(scn, "common.scenario.home_lla")
    if home is None:
        raise ValueError("common.scenario.home_lla is missing")
    home_lat = float(home[0])
    home_lon = float(home[1])
    home_alt = float(home[2])

    takeoff_alt = float(_get(scn, "common.scenario.takeoff_alt_m", 10.0))
    do_land = bool(_get(scn, "common.scenario.land", True))
    commands = _get(scn, "common.scenario.command", [])
    waypoints = _get(scn, "common.scenario.waypoints_lla", [])
    speeds = _get(scn, "common.scenario.speed_m_s", [])

    items = []
    has_takeoff = False

    for i, cmd in enumerate(commands):
        if cmd is None:
            continue

        cmd = int(cmd)

        p1 = p2 = p3 = p4 = 0
        p5 = p6 = p7 = 0
        frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT

        wp = waypoints[i] if i < len(waypoints) else None

        if cmd == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
            has_takeoff = True
            p1, p2, p3, p4 = 0.0, 0.0, 0.0, float("nan")
            p5, p6, p7 = home_lat, home_lon, takeoff_alt

        elif cmd == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
            if wp is None:
                raise ValueError(f"Waypoint missing for command index {i}")
            lat, lon, alt = float(wp[0]), float(wp[1]), float(wp[2])
            p1, p2, p3, p4 = 0.0, 1.0, 0.0, float("nan")
            p5, p6, p7 = lat, lon, alt

        elif cmd == mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED:
            frame = mavutil.mavlink.MAV_FRAME_MISSION
            speed = speeds[i] if i < len(speeds) else None
            if speed is None:
                raise ValueError(f"Speed missing for command index {i}")
            p1 = float(mavutil.mavlink.SPEED_TYPE_GROUNDSPEED)
            p2 = float(speed)
            p3 = -1.0
            p4 = p5 = p6 = p7 = 0

        elif cmd == mavutil.mavlink.MAV_CMD_NAV_LAND:
            p1, p2, p3, p4 = 0.0, 0.0, 0.0, float("nan")
            p5, p6, p7 = home_lat, home_lon, 0.0

        else:
            raise ValueError(f"Unsupported command for PX4 raw mission: {cmd}")

        items.append(
            dict(
                frame=frame,
                command=cmd,
                autocontinue=1,
                p1=p1, p2=p2, p3=p3, p4=p4,
                p5=p5, p6=p6, p7=p7,
            )
        )

    if not items:
        raise ValueError("No mission items were generated")

    return items, takeoff_alt, has_takeoff, do_land


class PX4MissionRunner:
    """
    Thread-backed PX4 mission runner using pymavlink.

    Responsibilities
    ----------------
    - Establish MAVLink connection
    - Wait for heartbeat
    - Set AUTO mode
    - Arm the vehicle

    Non-responsibilities
    --------------------
    - Mission upload
    - Takeoff
    - Waypoint navigation
    - AUTO mode

    These features are intentionally excluded so that this runner can
    be reused as a minimal control primitive inside larger systems.
    """

    def __init__(self, scenario_path: Path):
        """
        Parameters
        ----------
        scenario_path : Path
            Path to scenario.yaml configuration file.
        """

        self.scenario_path = Path(scenario_path)

        # Current mission status
        self.status = MissionStatus(MissionState.IDLE, "initialized")

        # Lock protecting shared status object
        self._lock = threading.Lock()

        # Prevent re-running of mission runner
        self._has_started_run = False

        # Worker thread executing the mission
        self._thread: Optional[threading.Thread] = None

        # External stop request flag
        self._stop_requested = False

        # MAVLink connection handle
        self._m = None

    def _set_status(
        self,
        state: MissionState,
        message: str = "",
        done: bool = False,
        success: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """
        Atomically update mission status.

        A lock is used because status may be read from other threads.
        """

        with self._lock:
            self.status = MissionStatus(
                state=state,
                message=message,
                done=done,
                success=success,
                error=error,
            )

    def get_status(self) -> MissionStatus:
        """
        Return a thread-safe snapshot of the current mission status.
        """

        with self._lock:
            return MissionStatus(
                state=self.status.state,
                message=self.status.message,
                done=self.status.done,
                success=self.status.success,
                error=self.status.error,
            )

    def request_stop(self) -> None:
        """
        Request graceful termination of the mission runner.
        """
        self._stop_requested = True

    def is_done(self) -> bool:
        """
        Check whether the runner has completed execution.
        """
        return self.get_status().done

    def start(self) -> None:
        """
        Start mission execution in a background thread.
        """

        if self._thread is not None:
            raise RuntimeError("Mission runner already started")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _check_stop(self) -> None:
        """
        Raise exception if an external stop was requested.
        """

        if self._stop_requested:
            raise RuntimeError("Mission runner stop requested")

    def _wait_heartbeat(self, m) -> None:
        """
        Wait until the autopilot sends a MAVLink heartbeat.

        Heartbeat confirms that:
        - MAVLink connection is alive
        - autopilot is running
        """
        self._set_status(MissionState.CONNECTING, "waiting for heartbeat")

        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=60)

        if msg is None:
            raise RuntimeError("Heartbeat timeout (60s)")
        
        print(f"[HEARTBEAT_CHECKING] Heartbeat OK (sys={msg.get_srcSystem()}, comp={msg.get_srcComponent()})")

        self._set_status(MissionState.HEARTBEAT_OK, "heartbeat received")

    def _wait_command_ack(self, m, command_id: int, timeout: float = 5.0) -> bool:
        """
        Wait for COMMAND_ACK of a specific command.

        Returns
        -------
        True  : if accepted
        False : if denied / failed / timeout
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
            if msg is None:
                continue

            if msg.command != command_id:
                continue

            result = msg.result
            if result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"COMMAND_ACK OK: cmd={command_id}")
                return True

            print(f"COMMAND_ACK FAILED: cmd={command_id}, result={result}")
            return False

        print(f"COMMAND_ACK TIMEOUT: cmd={command_id}")
        return False

    def _clear_mission(self, m, timeout: float = 5.0) -> bool:
        """
        Clear all mission items stored in the vehicle.
        """
        self._set_status(MissionState.CLEARING_MISSION, "clearing mission")

        m.mav.mission_clear_all_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = m.recv_match(type=["MISSION_ACK", "STATUSTEXT"], blocking=True, timeout=0.5)
            if msg is None:
                continue

            if msg.get_type() == "MISSION_ACK":
                return True

            if msg.get_type() == "STATUSTEXT":
                print(f"AP: {msg.text}")

        print("MISSION_CLEAR timeout")
        return False
    
    def _upload_mission_items(self, m, items: list[dict], timeout: float = 30.0) -> bool:
        """
        Upload mission items using MAVLink mission protocol.

        Each item dict should contain:
            frame, command, autocontinue, p1..p7
        """
        self._set_status(MissionState.UPLOADING_MISSION, f"uploading {len(items)} items")

        m.mav.mission_count_send(
            m.target_system,
            m.target_component,
            len(items),
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )

        sent = set()
        t0 = time.time()

        while time.time() - t0 < timeout:
            for _ in range(len(items)):

                req = m.recv_match(
                    type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK", "STATUSTEXT"],
                    blocking=True,
                    timeout=0.5,
                )

                if not req:
                    raise RuntimeError("MISSION upload timeout")

                if req.get_type() == "STATUSTEXT":
                    print(f"AP: {req.text}")
                    continue

                if req.get_type() == "MISSION_ACK":
                    ack_type = req.type
                    self._set_status(MissionState.MISSION_UPLOADED, "mission uploaded")
                    return ack_type == mavutil.mavlink.MAV_MISSION_ACCEPTED

                if req.get_type() in ("MISSION_REQUEST", "MISSION_REQUEST_INT"):
                    seq = req.seq
                    if seq < 0 or seq >= len(items):
                        print(f"[MISSION_UPLOAD] Invalid mission request seq={seq}")
                        return False

                    it = items[seq]

                    m.mav.mission_item_int_send(
                        m.target_system,
                        m.target_component,
                        seq,
                        it["frame"],
                        it["command"],
                        1 if seq == 0 else 0,
                        it["autocontinue"],
                        it["p1"], it["p2"], it["p3"], it["p4"],
                        int(it["p5"] * 1e7),
                        int(it["p6"] * 1e7),
                        float(it["p7"]),
                        mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
                    )

                    sent.add(seq)
                    print(f"[MISSION_UPLOAD] Sent mission item seq={seq}, cmd={it['command']}")

        print("[MISSION_UPLOAD] Mission upload timeout")
        return False
        
    def _read_mission_item(self, m, seq: int, timeout: float = 5.0):
        """
        Read back one mission item from the vehicle.
        Returns the MAVLink message or None.
        """
        m.mav.mission_request_int_send(
            m.target_system,
            m.target_component,
            seq,
        )

        t0 = time.time()
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
                return msg

        return None
    


    def _wait_sensor_health_ok(self, m, timeout: float = 60.0) -> bool:
        """
        Wait for PX4 sensor readiness.

        Minimal heuristic:
        - SYS_STATUS health available
        """

        t0 = time.time()

        while time.time() - t0 < timeout:

            msg = m.recv_match(type=["SYS_STATUS"], blocking=True, timeout=1.0)

            if msg is None:
                continue

            enabled = msg.onboard_control_sensors_enabled
            health = msg.onboard_control_sensors_health

            missing_health = enabled & ~health

            if missing_health == 0:
                print("[MONITOR] sensor health check passed")
                return True
            elif missing_health == 65536:
                print("[MONITOR] sensor health check passed (no radio control)")
                return True

        print("[READINESS] timeout waiting for global position")
        return False




    def _wait_global_position_ok(self, m, timeout: float = 60.0) -> bool:
        """
        Wait for PX4 estimator / global position readiness.

        Minimal heuristic:
        - GLOBAL_POSITION_INT observed
        """

        t0 = time.time()

        while time.time() - t0 < timeout:

            msg = m.recv_match(type=["GLOBAL_POSITION_INT"], blocking=True, timeout=1.0)

            if msg is None:
                continue

            if msg.get_type() == "GLOBAL_POSITION_INT":
                print("[MONITOR] global position information check passed")
                return True

        print("[MONITOR] timeout waiting for global position")
        return False





    def _arm(self, m) -> bool:
        self._set_status(MissionState.ARMING, "arming vehicle")

        m.mav.command_long_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0,
        )

        if not self._wait_command_ack(m, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, timeout=10.0):
            return False

        t0 = time.time()
        while time.time() - t0 < 10.0:
            hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
            if hb is None:
                continue

            armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if armed:
                self._set_status(MissionState.ARMED, "vehicle armed")
                print("[ARM] vehicle armed")
                return True

        print("[ARM] timeout waiting for armed state")
        return False

    def _set_mode_auto_mission(self, m) -> bool:
        """
        Force PX4 into AUTO.MISSION using MAVLink SET_MODE with
        base_mode + PX4 custom_mode.

        PX4 packing:
            custom_mode = (sub_mode << 24) | (main_mode << 16)

        AUTO.MISSION:
            main_mode = 4   # PX4_CUSTOM_MAIN_MODE_AUTO
            sub_mode  = 4   # PX4_CUSTOM_SUB_MODE_AUTO_MISSION
        """
        self._set_status(MissionState.SETTING_AUTO, "switching to AUTO.MISSION")

        # PX4 custom mode values
        PX4_CUSTOM_MAIN_MODE_AUTO = 4
        PX4_CUSTOM_SUB_MODE_AUTO_MISSION = 4

        # Enable custom mode interpretation in base_mode
        base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED

        # PX4 packs main/sub mode into custom_mode like this
        custom_mode = (
            (PX4_CUSTOM_SUB_MODE_AUTO_MISSION << 24)
            | (PX4_CUSTOM_MAIN_MODE_AUTO << 16)
        )

        try:
            m.mav.set_mode_send(
                m.target_system,
                base_mode,
                custom_mode,
            )
        except Exception as e:
            print(f"[MODE CHANGE] SET_MODE AUTO.MISSION failed: {e}")
            return False

        # Wait briefly for heartbeat update
        deadline = time.time() + 2.0
        while time.time() < deadline:
            hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=0.5)
            if hb is None:
                continue

            # base_mode must indicate custom mode is enabled
            custom_enabled = bool(
                hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
            )

            # Decode PX4 custom_mode from heartbeat
            hb_custom = int(hb.custom_mode)
            main_mode = (hb_custom >> 16) & 0xFF
            sub_mode = (hb_custom >> 24) & 0xFF

            if custom_enabled and main_mode == PX4_CUSTOM_MAIN_MODE_AUTO \
                    and sub_mode == PX4_CUSTOM_SUB_MODE_AUTO_MISSION:
                print("[MODE CHANGE] PX4 is now in AUTO.MISSION")
                return True

        print("[MODE CHANGE] AUTO.MISSION mode change not confirmed from HEARTBEAT")
        return False

    def _mission_start(self, m, first_item: int = 0, last_item: int = 0xFFFF) -> bool:
        self._set_status(MissionState.STARTING_MISSION, "sending mission start")

        m.mav.command_long_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_CMD_MISSION_START,
            0,
            float(first_item),
            float(last_item),
            0, 0, 0, 0, 0,
        )

        return self._wait_command_ack(m, mavutil.mavlink.MAV_CMD_MISSION_START, timeout=8.0)

    def _build_waypoint_seq_map(self, mission_items: List[dict]) -> Dict[int, int]:
        wp_num = 0
        seq_to_wp: Dict[int, int] = {}

        for seq, item in enumerate(mission_items):
            if item["command"] == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
                wp_num += 1
                seq_to_wp[seq] = wp_num

        return seq_to_wp

    def _find_last_land_seq(self, mission_items: List[dict]) -> Optional[int]:
        last_land_seq = None
        for seq, item in enumerate(mission_items):
            if item["command"] == mavutil.mavlink.MAV_CMD_NAV_LAND:
                last_land_seq = seq
        return last_land_seq

    def _monitor_current_mission(self, m, mission_items: List[dict], duration: float = 300.0) -> bool:
        seq_to_wp = self._build_waypoint_seq_map(mission_items)
        last_land_seq = self._find_last_land_seq(mission_items)

        t0 = time.time()
        last_seq = None
        self._set_status(MissionState.RUNNING, "mission running")

        while True:
            self._check_stop()

            if time.time() - t0 >= duration:
                raise TimeoutError(f"Mission monitor timeout ({duration:.1f}s)")

            msg = m.recv_match(
                type=["MISSION_CURRENT", "STATUSTEXT"],
                blocking=True,
                timeout=1.0,
            )

            if msg is None:
                continue

            if msg.get_type() == "STATUSTEXT":
                print(f"[PX4] {msg.text}")
                continue

            seq = msg.seq
            if seq == last_seq:
                continue

            last_seq = seq
            cmd = mission_items[seq]["command"] if 0 <= seq < len(mission_items) else None
            wp_idx = seq_to_wp.get(seq)

            if wp_idx is not None:
                print(f"[MONITOR] seq={seq}, cmd={cmd}, waypoint #{wp_idx}")
            else:
                print(f"[MONITOR] seq={seq}, cmd={cmd}, non-waypoint item")

            if last_land_seq is not None and seq >= last_land_seq:
                print(f"[MONITOR] reached/passed land sequence seq={seq}")
                return True

    def _wait_disarmed(self, m, timeout: float = 180.0) -> bool:
        t0 = time.time()

        while time.time() - t0 < timeout:
            self._check_stop()

            hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if hb is None:
                continue

            armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            if not armed:
                print("[DISARM] vehicle disarmed")
                self._set_status(MissionState.COMPLETED, "mission completed", done=True, success=True)
                return True

        raise TimeoutError("Vehicle did not disarm")

    def _run(self) -> None:
        """
        Main mission execution logic.

        Execution order:
            connect -> heartbeat -> GUIDED -> arm
        """

        if self._has_started_run:
            raise RuntimeError("Mission runner _run already entered")
        self._has_started_run = True

        print(f"[ENTER] PX4 pymavlink runner self_id={id(self)} thread={threading.get_ident()}")

        try:
            # Load scenario configuration and parse
            scn = yaml.safe_load(self.scenario_path.read_text(encoding="utf-8")) or {}
            items, takeoff_alt, has_takeoff, do_land = build_items_from_scenario(scn)

            # Get scenario connection URL to pymavlink format
            connect = parse_connect_url(scn)

            # Establish MAVLink connection assign instance
            self._set_status(MissionState.CONNECTING, f"connecting to {connect}")

            m = mavutil.mavlink_connection(
                connect,
                source_system=255,
                source_component=0,
            )
            self._m = m

            # Explicitly set target IDs (common practice in ArduPilot SITL)
            m.target_system = 1
            m.target_component = 1

            # Wait for autopilot heartbeat
            self._wait_heartbeat(m)

            # Stop if requested
            self._check_stop()

            # Check readiness
            self._wait_sensor_health_ok(m, timeout=60.0)
            self._wait_global_position_ok(m, timeout=60.0)

            # Clear and upload mission
            self._clear_mission(m)
            self._upload_mission_items(m, items, timeout=10.0)

            # Stop if requested
            self._check_stop()

            # Set auto mode and arming
            self._set_mode_auto_mission(m)
            self._arm(m)

            # Stop if requested
            self._check_stop()

            # Start mission
            self._mission_start(m)

            # Monitor current mission
            self._monitor_current_mission(m, items, duration=300.0)

            # Stop if requested
            self._check_stop()

            # Waiting for disarm
            self._wait_disarmed(m, timeout=180.0)

        except Exception as e:
            self._set_status(
                MissionState.FAILED,
                message="execution failed, check errors",
                done=True,
                success=False,
                error=str(e),
            )


def main():
    """
    Standalone execution entry point for testing.
    """

    import argparse

    parser = argparse.ArgumentParser(description="ArduPilot Mission Runner")
    parser.add_argument(
        "--scenario",
        type=Path,
        help="Path to scenario.yaml configuration file",
    )
    args = parser.parse_args()

    runner = PX4MissionRunner(args.scenario)
    runner._run()

    status = runner.get_status()
    print(f"Mission finished: {status.state.name}, success={status.success}")

if __name__ == "__main__":
    main()