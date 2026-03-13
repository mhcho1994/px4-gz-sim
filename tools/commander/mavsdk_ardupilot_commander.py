#!/usr/bin/env python3
"""
ArduPilot Mission Runner (YAML-driven)

Purpose
-------
MAVLink mission runner used for SITL experiments.

This script reads a scenario.yaml file and executes the mission using
pymavlink with ArduPilot SITL. 

    1. Parse scenario configuration from YAML
    2. Construct MAVLink mission items
    3. Upload mission to ArduPilot
    4. Arm vehicle and optionally perform guided takeoff
    5. Switch to AUTO mode and execute mission
    6. Wait until vehicle disarms after landing

Design Philosophy
-----------------
This is intentionally minimal because it is used as a building block
inside larger experiment pipelines (e.g., SITL orchestration scripts).

Typical usage:
    runner = ArduPilotMissionRunner("scenario.yaml")
    runner.start()
    runner.join()

Threading
---------
The runner executes in a background thread so that the outer simulation
framework can poll status asynchronously.
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

    IDLE = auto()              # Runner created but not started
    CONNECTING = auto()        # MAVLink connection in progress
    HEARTBEAT_OK = auto()      # Vehicle heartbeat received
    CLEARING_MISSION = auto()
    UPLOADING_MISSION = auto()
    MISSION_UPLOADED = auto()
    SETTING_GUIDED = auto()    # Requesting GUIDED mode
    ARMING = auto()            # Sending arm command
    ARMED = auto()             # Vehicle successfully armed
    TAKING_OFF = auto()
    AUTO = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()            # Error occurred


@dataclass
class MissionStatus:
    """
    Immutable snapshot of mission runner state.

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
    _get(config, "autopilots.ardupilot.mavlink.connect_url")

    Equivalent to:
        config["autopilots"]["ardupilot"]["mavlink"]["connect_url"]

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
    Convert MAVSDK-style connection URLs into pymavlink format.

    Example
    -------
    MAVSDK format:
        udp://127.0.0.1:14550

    pymavlink format:
        udpin:127.0.0.1:14550

    If the scenario file does not explicitly define a connection URL,
    this function falls back to the default SITL UDP output port.
    """

    url = _get(scn, "autopilots.ardupilot.mavlink.connect_url")

    if url:
        # Convert MAVSDK-style URL to pymavlink format
        if url.startswith("udp://"):
            hostport = url[len("udp://"):]
            return f"udpin:{hostport}"
        return url

    # Default connection port used by SITL
    port = _get(scn, "autopilots.ardupilot.sim.out_udp_port", 14550)
    return f"udpin:127.0.0.1:{port}"


def build_items_from_scenario(scn: Dict[str, Any]) -> Tuple[list, float, bool]:
    home = _get(scn, "common.scenario.home_lla")
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
            has_takeoff = False
            p1, p2, p3, p4 = 0, 0, 0, float('nan')
            p5, p6, p7 = home_lat, home_lon, takeoff_alt

        elif cmd == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT:
            if wp is None:
                raise ValueError(f"Waypoint missing for command index {i}")
            lat, lon, alt = wp
            p1, p2, p3, p4 = 0, 1.0, 0, float('nan')
            p5, p6, p7 = lat, lon, alt

        elif cmd == mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED:
            frame = mavutil.mavlink.MAV_FRAME_MISSION
            p1 = mavutil.mavlink.SPEED_TYPE_GROUNDSPEED
            p2 = float(speeds[i])
            p3 = -1.0
            p4 = p5 = p6 = p7 = 0

        elif cmd == mavutil.mavlink.MAV_CMD_NAV_LAND:
            p1, p2, p3, p4 = 0, 0, 0, float('nan')
            p5, p6, p7 = home_lat, home_lon, 0

        items.append(
            dict(
                frame=frame,
                command=cmd,
                autocontinue=1,
                p1=p1, p2=p2, p3=p3, p4=p4,
                p5=p5, p6=p6, p7=p7,
            )
        )

    return items, takeoff_alt, has_takeoff, do_land


class ArduPilotMissionRunner:
    """
    Thread-backed mission runner for ArduPilot SITL.

    Responsibilities
    ----------------
    - Establish MAVLink connection
    - Wait for heartbeat
    - Set GUIDED mode
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

    def join(self, timeout: Optional[float] = None) -> None:
        """
        Wait for the mission thread to finish.
        """

        if self._thread is not None:
            self._thread.join(timeout)

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

        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=20)

        if msg is None:
            raise RuntimeError("Heartbeat timeout (20s)")

        print(f"Heartbeat OK (sys={msg.get_srcSystem()}, comp={msg.get_srcComponent()})")

        self._set_status(MissionState.HEARTBEAT_OK, "heartbeat received")

    def _wait_command_ack(
        m,
        command_id: int,
        timeout: float = 5.0,
    ) -> bool:
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
                        print(f"Invalid mission request seq={seq}")
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
                    print(f"Sent mission item seq={seq}, cmd={it['command']}")

        print("Mission upload timeout")
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

    def _set_mode(self, m, mode_name: str) -> None:
        """
        Request a flight mode change.

        ArduPilot modes include:
            STABILIZE
            GUIDED
            AUTO
            LOITER
            etc.
        """

        m.set_mode(mode_name)

        # Short delay to allow the autopilot to process the request
        time.sleep(0.5)

    def _arm(self, m) -> None:
        """
        Send arm command and wait until motors are armed.

        Arming enables the vehicle's motors and allows flight commands.
        """

        self._set_status(MissionState.ARMING, "arming vehicle")

        m.arducopter_arm()

        # Block until autopilot reports motors armed
        m.motors_armed_wait()

        self._set_status(
            MissionState.ARMED,
            "vehicle armed",
            done=True,
            success=True,
        )

    def _guided_takeoff(self, m, takeoff_alt_m: float) -> None:
        self._set_status(MissionState.TAKING_OFF, f"takeoff to {takeoff_alt_m:.1f} m")
        m.mav.command_long_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0,
            0, 0,
            takeoff_alt_m,
        )

        _ = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=3)

        t0 = time.time()
        while time.time() - t0 < 30:
            self._check_stop()
            msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
            if not msg:
                continue
            rel_alt = msg.relative_alt / 1000.0
            if rel_alt >= 0.8 * takeoff_alt_m:
                return

        raise TimeoutError("Target altitude not reached")

    def _wait_disarmed(self, m) -> None:
        self._set_status(MissionState.RUNNING, "mission running")
        t0 = time.time()
        while time.time() - t0 < 180:
            self._check_stop()
            hb = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1.0)
            if not hb:
                continue

            armed = (
                hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            ) != 0

            if not armed:
                self._set_status(MissionState.COMPLETED, "mission completed", done=True, success=True)
                return

        raise TimeoutError("Vehicle did not disarm")

    def _set_param(self, m, name: str, value: int, timeout: float = 5.0) -> bool:
        """
        Set an ArduPilot parameter as integer/real via PARAM_SET,
        then verify using PARAM_VALUE.
        """
        m.mav.param_set_send(
            m.target_system,
            m.target_component,
            name.encode("utf-8"),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )

        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg is None:
                continue

            def normalize_param_id(param_id) -> str:
                """
                Normalize PARAM_VALUE.param_id to a plain Python string.
                Handles both bytes and str cases.
                """
                if isinstance(param_id, bytes):
                    return param_id.decode("utf-8", errors="ignore").rstrip("\x00")
                return str(param_id).rstrip("\x00")

            param_id = normalize_param_id(msg.param_id)
            if param_id != name:
                continue

            actual = msg.param_value
            print(f"PARAM {name} = {actual}")
            return abs(actual - float(value)) < 1e-3

        print(f"PARAM_SET timeout: {name}")
        return False

    def _get_param(self, m, name: str, timeout: float = 5.0):
        """
        Request a parameter and return its value, or None on timeout.
        """
        m.mav.param_request_read_send(
            m.target_system,
            m.target_component,
            name.encode("utf-8"),
            -1,
        )

        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg is None:
                continue

            def normalize_param_id(param_id) -> str:
                """
                Normalize PARAM_VALUE.param_id to a plain Python string.
                Handles both bytes and str cases.
                """
                if isinstance(param_id, bytes):
                    return param_id.decode("utf-8", errors="ignore").rstrip("\x00")
                return str(param_id).rstrip("\x00")

            param_id = normalize_param_id(msg.param_id)
            if param_id != name:
                continue

            return msg.param_value

        return None




    def _get_current_mission_seq(self, m, timeout: float = 1.0):
        """
        Read current mission item index from MISSION_CURRENT.
        Returns seq or None if timeout.
        """
        msg = m.recv_match(type="MISSION_CURRENT", blocking=True, timeout=timeout)
        if msg is None:
            return None
        return msg.seq



    def _run(self) -> None:
        """
        Main mission execution logic.

        Execution order:
            connect -> heartbeat -> GUIDED -> arm
        """

        try:

            # Load scenario configuration and parse
            scn = yaml.safe_load(self.scenario_path.read_text())
            items, takeoff_alt, has_takeoff, do_land = build_items_from_scenario(scn)

            # Convert scenario MAVSDK connection URL to pymavlink format
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

            # Clear and upload mission
            self._clear_mission(m)
            self._upload_mission_items(m, items, timeout=10)

            # Switch to guided mode and arm
            self._set_status(MissionState.SETTING_GUIDED, "switching to GUIDED")
            self._set_mode(m, "GUIDED")

            self._arm(m)

            if has_takeoff:
                # Enable takeoff in auto mode (but no longer supported by Arducopter, seems deprecated)
                # self._set_param(m, 'AUTO_OPTIONS', 1)
                # self._get_param(m, 'AUTO_OPTIONS', 1)

                # Verify takeoff @ sequence 0
                msg_seq0 = self._read_mission_item(m,0)

                if msg_seq0 is None:
                    raise RuntimeError("Failed to read back mission item 0")
                if msg_seq0.command != mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
                    raise RuntimeError(
                        f"Mission seq0 is not TAKEOFF "
                        f"(got command={msg_seq0.command})"
                    )

            else:
                self._guided_takeoff(m, takeoff_alt)

            # Stop if requested
            self._check_stop()

            self._set_status(MissionState.AUTO, "switching to AUTO")
            self._set_mode(m, "AUTO")

            self._wait_disarmed(m)

            #------------

            self._check_stop()

            # Arm vehicle
            self._arm(m)

        except Exception as e:
            # Any exception is treated as mission failure
            self._set_status(
                MissionState.FAILED,
                message="arming failed",
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

    runner = ArduPilotMissionRunner(args.scenario)
    runner.start()
    runner.join()

    status = runner.get_status()
    print(f"Mission finished: {status.state.name}, success={status.success}")

if __name__ == "__main__":
    main()