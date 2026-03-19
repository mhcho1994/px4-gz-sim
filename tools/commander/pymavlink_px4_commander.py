#!/usr/bin/env python3
"""
PX4 Mission Runner (YAML-driven, pymavlink-based)

Purpose
-------
Raw MAVLink mission runner for PX4 SITL experiments.

This script:
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

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import yaml
from pymavlink import mavutil


class MissionState(Enum):
    IDLE = auto()
    CONNECTING = auto()
    HEARTBEAT_OK = auto()
    CLEARING_MISSION = auto()
    UPLOADING_MISSION = auto()
    MISSION_UPLOADED = auto()
    ARMING = auto()
    ARMED = auto()
    SETTING_AUTO = auto()
    STARTING_MISSION = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class MissionStatus:
    state: MissionState
    message: str = ""
    done: bool = False
    success: bool = False
    error: Optional[str] = None


def _get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_connect_url_px4(scn: Dict[str, Any]) -> str:
    """
    Convert YAML URL into pymavlink format.

    Example:
      udp://127.0.0.1:14650 -> udpin:127.0.0.1:14650
    """
    url = _get(scn, "autopilots.px4.mavsdk.connect_url")
    if not url:
        port = int(_get(scn, "autopilots.px4.sim.out_udp_port", 14540))
        return f"udpin:127.0.0.1:{port}"

    if url.startswith("udp://"):
        return f"udpin:{url[len('udp://'):]}"
    return url


def build_items_from_scenario_px4(scn: Dict[str, Any]) -> Tuple[List[dict], float, bool]:
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

    takeoff_alt = float(_get(scn, "common.scenario.takeoff_alt_m", 10.0))
    commands = _get(scn, "common.scenario.command", [])
    waypoints = _get(scn, "common.scenario.waypoints_lla", [])
    speeds = _get(scn, "common.scenario.speed_m_s", [])

    items: List[dict] = []
    has_land = False

    for i, cmd in enumerate(commands):
        if cmd is None:
            continue

        cmd = int(cmd)
        frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT

        p1 = p2 = p3 = p4 = 0.0
        p5 = p6 = p7 = 0.0

        wp = waypoints[i] if i < len(waypoints) else None

        if cmd == mavutil.mavlink.MAV_CMD_NAV_TAKEOFF:
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
            p1 = float(mavutil.mavlink.SPEED_TYPE_AIRSPEED)
            p2 = float(speed)
            p3 = -1.0
            p4 = p5 = p6 = p7 = 0.0

        elif cmd == mavutil.mavlink.MAV_CMD_NAV_LAND:
            has_land = True
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

    return items, takeoff_alt, has_land


class PX4MissionRunner:
    """
    Thread-backed PX4 mission runner using pymavlink.
    """

    def __init__(self, scenario_path: Path):
        self.scenario_path = Path(scenario_path)

        self.status = MissionStatus(MissionState.IDLE, "initialized")
        self._lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._has_started_run = False

        self._m = None

    def _set_status(
        self,
        state: MissionState,
        message: str = "",
        done: bool = False,
        success: bool = False,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.status = MissionStatus(
                state=state,
                message=message,
                done=done,
                success=success,
                error=error,
            )

    def get_status(self) -> MissionStatus:
        with self._lock:
            return MissionStatus(
                state=self.status.state,
                message=self.status.message,
                done=self.status.done,
                success=self.status.success,
                error=self.status.error,
            )

    def request_stop(self) -> None:
        self._stop_requested = True

    def is_done(self) -> bool:
        return self.get_status().done

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Mission runner already started")

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _check_stop(self) -> None:
        if self._stop_requested:
            raise RuntimeError("Mission runner stop requested")

    def _wait_heartbeat(self, m, timeout: float = 30.0) -> None:
        self._set_status(MissionState.CONNECTING, "waiting for heartbeat")
        msg = m.recv_match(type="HEARTBEAT", blocking=True, timeout=timeout)
        if msg is None:
            raise RuntimeError(f"Heartbeat timeout ({timeout:.1f}s)")

        m.target_system = msg.get_srcSystem()
        m.target_component = msg.get_srcComponent()

        print(f"[HEARTBEAT] sys={m.target_system}, comp={m.target_component}")
        self._set_status(MissionState.HEARTBEAT_OK, "heartbeat received")

    def _normalize_param_id(self, param_id) -> str:
        if isinstance(param_id, bytes):
            return param_id.decode("utf-8", errors="ignore").rstrip("\x00")
        return str(param_id).rstrip("\x00")

    def _wait_global_position_ok(self, m, timeout: float = 60.0) -> bool:
        """
        Wait for PX4 estimator / global position readiness.

        Minimal heuristic:
        - GLOBAL_POSITION_INT observed
        - SYS_STATUS health available
        """
        t0 = time.time()
        got_global_pos = False

        while time.time() - t0 < timeout:
            self._check_stop()
            msg = m.recv_match(
                type=["GLOBAL_POSITION_INT", "SYS_STATUS", "STATUSTEXT"],
                blocking=True,
                timeout=1.0,
            )
            if msg is None:
                continue

            if msg.get_type() == "STATUSTEXT":
                print(f"[PX4] {msg.text}")
                continue

            if msg.get_type() == "GLOBAL_POSITION_INT":
                got_global_pos = True
                print("[READINESS] GLOBAL_POSITION_INT observed")
                return True

        print("[READINESS] timeout waiting for global position")
        return got_global_pos

    def _wait_command_ack(self, m, command_id: int, timeout: float = 8.0) -> bool:
        t0 = time.time()

        while time.time() - t0 < timeout:
            msg = m.recv_match(type="COMMAND_ACK", blocking=True, timeout=0.5)
            if msg is None:
                continue

            if msg.command != command_id:
                continue

            result = msg.result
            if result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"[ACK] cmd={command_id} accepted")
                return True

            print(f"[ACK] cmd={command_id} failed result={result}")
            return False

        print(f"[ACK] cmd={command_id} timeout")
        return False

    def _clear_mission(self, m, timeout: float = 5.0) -> bool:
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

            if msg.get_type() == "STATUSTEXT":
                print(f"[PX4] {msg.text}")
                continue

            if msg.get_type() == "MISSION_ACK":
                print("[MISSION] clear acknowledged")
                return True

        print("[MISSION] clear timeout")
        return False

    def _upload_mission_items(self, m, items: List[dict], timeout: float = 30.0) -> bool:
        self._set_status(MissionState.UPLOADING_MISSION, f"uploading {len(items)} items")

        m.mav.mission_count_send(
            m.target_system,
            m.target_component,
            len(items),
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
        )

        t0 = time.time()

        while time.time() - t0 < timeout:
            req = m.recv_match(
                type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK", "STATUSTEXT"],
                blocking=True,
                timeout=1.0,
            )

            if req is None:
                continue

            if req.get_type() == "STATUSTEXT":
                print(f"[PX4] {req.text}")
                continue

            if req.get_type() == "MISSION_ACK":
                ack_type = req.type
                if ack_type == mavutil.mavlink.MAV_MISSION_ACCEPTED:
                    self._set_status(MissionState.MISSION_UPLOADED, "mission uploaded")
                    print("[MISSION] upload accepted")
                    return True

                print(f"[MISSION] upload failed ack_type={ack_type}")
                return False

            if req.get_type() in ("MISSION_REQUEST", "MISSION_REQUEST_INT"):
                seq = req.seq
                if seq < 0 or seq >= len(items):
                    print(f"[MISSION] invalid request seq={seq}")
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
                    float(it["p1"]),
                    float(it["p2"]),
                    float(it["p3"]),
                    float(it["p4"]),
                    int(float(it["p5"]) * 1e7),
                    int(float(it["p6"]) * 1e7),
                    float(it["p7"]),
                    mavutil.mavlink.MAV_MISSION_TYPE_MISSION,
                )

                print(f"[MISSION] sent seq={seq}, cmd={it['command']}")

        print("[MISSION] upload timeout")
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
        Try PX4 AUTO.MISSION mode change using set_mode first.
        """
        self._set_status(MissionState.SETTING_AUTO, "switching to AUTO.MISSION")

        try:
            m.set_mode("AUTO.MISSION")
        except Exception as e:
            print(f"[MODE] set_mode(AUTO.MISSION) raised: {e}")

        # Give PX4 a moment.
        time.sleep(1.0)

        # Check heartbeats for custom_mode text is not directly available here,
        # so we rely on mission start ack to confirm practical success.
        return True

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
        if self._has_started_run:
            raise RuntimeError("Mission runner _run already entered")
        self._has_started_run = True

        print(f"[ENTER] PX4 pymavlink runner self_id={id(self)} thread={threading.get_ident()}")

        try:
            scn = yaml.safe_load(self.scenario_path.read_text(encoding="utf-8")) or {}
            items, _takeoff_alt, _has_land = build_items_from_scenario_px4(scn)
            connect = parse_connect_url_px4(scn)

            self._set_status(MissionState.CONNECTING, f"connecting to {connect}")

            m = mavutil.mavlink_connection(
                connect,
                source_system=255,
                source_component=190,
            )
            self._m = m

            self._wait_heartbeat(m)
            self._check_stop()

            self._wait_global_position_ok(m, timeout=60.0)
            self._check_stop()

            if not self._clear_mission(m, timeout=5.0):
                raise RuntimeError("Failed to clear mission")

            if not self._upload_mission_items(m, items, timeout=30.0):
                raise RuntimeError("Failed to upload mission")

            self._check_stop()

            if not self._arm(m):
                raise RuntimeError("Failed to arm vehicle")

            self._check_stop()

            if not self._set_mode_auto_mission(m):
                raise RuntimeError("Failed to switch to AUTO.MISSION")

            self._check_stop()

            if not self._mission_start(m):
                raise RuntimeError("Failed to start mission")

            self._monitor_current_mission(m, items, duration=300.0)
            self._check_stop()

            self._wait_disarmed(m, timeout=180.0)

        except Exception as e:
            self._set_status(
                MissionState.FAILED,
                message="execution failed, check errors",
                done=True,
                success=False,
                error=str(e),
            )


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="PX4 pymavlink Mission Runner")
    ap.add_argument("--scenario", type=Path, required=True, help="Path to scenario.yaml")
    args = ap.parse_args()

    runner = PX4MissionRunner(args.scenario)
    runner.start()
    runner.join()

    status = runner.get_status()
    print(f"Mission finished: {status.state.name}, success={status.success}, error={status.error}")
    return 0 if status.success else 1


if __name__ == "__main__":
    raise SystemExit(main())