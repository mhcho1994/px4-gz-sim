#!/usr/bin/env python3
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
    IDLE = auto() 
    CONNECTING = auto()
    HEARTBEAT_OK = auto()
    SETTING_GUIDED = auto()
    ARMING = auto()
    ARMED = auto()
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


def parse_connect_url(scn: Dict[str, Any]) -> str:
    """
    Convert scenario MAVSDK-style connection URL into a pymavlink connection string.

    Supported example:
      udp://127.0.0.1:14550 -> udpin:127.0.0.1:14550
    """
    url = _get(scn, "autopilots.ardupilot.mavsdk.connect_url")
    if url:
        if url.startswith("udp://"):
            hostport = url[len("udp://"):]
            return f"udpin:{hostport}"
        return url

    port = _get(scn, "autopilots.ardupilot.sim.out_udp_port", 14550)
    return f"udpin:127.0.0.1:{port}"


class ArduPilotMissionRunner:
    """
    Thread-backed runner that only performs:
      1) MAVLink connection
      2) heartbeat wait
      3) GUIDED mode switch
      4) arming

    No mission upload, takeoff, or AUTO execution is performed.
    """

    def __init__(self, scenario_path: Path):
        self.scenario_path = Path(scenario_path)
        self.status = MissionStatus(MissionState.IDLE, "initialized")
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_requested = False
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
            self._thread.join(timeout)

    def _check_stop(self) -> None:
        if self._stop_requested:
            raise RuntimeError("Mission runner stop requested")

    def _wait_heartbeat(self, m) -> None:
        self._set_status(MissionState.CONNECTING, "waiting for heartbeat")
        m.wait_heartbeat(timeout=20)
        self._set_status(MissionState.HEARTBEAT_OK, "heartbeat received")

    def _set_mode(self, m, mode_name: str) -> None:
        """
        Request a flight mode change and wait briefly.
        """
        m.set_mode(mode_name)
        time.sleep(0.5)

    def _arm(self, m) -> None:
        """
        Send arm command and wait until motors are reported armed.
        """
        self._set_status(MissionState.ARMING, "arming vehicle")
        m.arducopter_arm()
        m.motors_armed_wait()
        self._set_status(
            MissionState.ARMED,
            "vehicle armed",
            done=True,
            success=True,
        )

    def _run(self) -> None:
        try:
            scn = yaml.safe_load(self.scenario_path.read_text())
            connect = parse_connect_url(scn)

            self._set_status(MissionState.CONNECTING, f"connecting to {connect}")

            m = mavutil.mavlink_connection(
                connect,
                source_system=246,
                source_component=190,
            )
            self._m = m

            self._check_stop()
            self._wait_heartbeat(m)

            # In many SITL setups this is already discovered automatically after heartbeat,
            # but keeping it explicit is often convenient for ArduPilot.
            m.target_system = 1
            m.target_component = 1

            self._check_stop()
            self._set_status(MissionState.SETTING_GUIDED, "switching to GUIDED")
            self._set_mode(m, "GUIDED")

            self._check_stop()
            self._arm(m)

        except Exception as e:
            self._set_status(
                MissionState.FAILED,
                message="arming failed",
                done=True,
                success=False,
                error=str(e),
            )