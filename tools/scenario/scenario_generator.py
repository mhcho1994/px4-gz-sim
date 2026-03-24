#!/usr/bin/env python3
"""
Scenario Generator (Pattern-Specific) + Mission Plan Metadata Embedder

Purpose:
- Generate multiple run directories (run_000, run_001, ...)
- For each run, write a scenario.yaml file
- The mission geometry (waypoints) is common
- Autopilot-specific simulation settings are separated (ArduPilot / PX4)

Design philosophy:
- Keep scenario geometry minimal and pattern-specific
- Avoid overly generic scenario generators
- Enable fair comparison between PX4 and ArduPilot using identical trajectories
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import yaml


# ----------------------------------------------------------------------
# Type definition
# ----------------------------------------------------------------------
# Waypoint representation in LLA frame
# Convert (north, east, down) to (lat, lon, alt) for mission upload
# NOTE:
#   Down is positive in NED.
#   If altitude above home is +5m,
#   then Down = -5.0
NED = Tuple[float, float, float]
# NOTE:
# Alt is meters above WGS84 ellipsoid (or AMSL depending on your convention)
# AltitudeMode meaning in QGC mission plan
# 0 RelativeToHome:
# 1	RelativeToHome: (default mission) QGC mission waypoint
# 2	AMSL: mean sea level altitude
# 3	Terrain: relative to terrain altitude
LLA = Tuple[float, float, float]
# NOTE:
# Specify waypoint speed in m/s (float) for each waypoint
SPD = float

# ----------------------------------------------------------------------
# MAVLink Command IDs
# ----------------------------------------------------------------------
# MAVLink command IDs
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_RETURN_TO_LAUNCH = 20
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_CHANGE_SPEED = 178

# ----------------------------------------------------------------------
# NED → LLA conversion
# ----------------------------------------------------------------------
# WGS84 constants
_WGS84_A = 6378137.0                  # semi-major axis [m]
_WGS84_E2 = 6.69437999014e-3          # eccentricity squared

def ned_to_lla(ned: NED, home_lla: LLA) -> LLA:
    """
    Convert local NED offset (m) to geodetic LLA using a small angle approximation.
    Accurate for small distances (typ. <~10-20km) around home.
    If large, please use a proper transformation: NED -> ECEF -> Geodetic.
    If needed, notify mhcho to implement a more accurate conversion.

    Args:
        ned: (north_m, east_m, down_m)
        home_lla: (lat_deg, lon_deg, alt_m(AMSL or WGS84))

    Returns:
        (lat_deg, lon_deg, alt_rel_m)

    Notes:
        - down is positive in NED; altitude increases upward.
        - alt_rel_m here is computed as (-1) * down.
        - This is accurate for small distances (typ. <~10-20km) around home.
    """
    north, east, down = ned
    lat0_deg, lon0_deg, h0 = home_lla

    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)

    sin_lat0 = np.sin(lat0)
    denom = np.sqrt(1.0 - _WGS84_E2 * sin_lat0 * sin_lat0)

    # radii of curvature
    Rn = _WGS84_A * (1.0 - _WGS84_E2) / (denom ** 3)    # meridian radius
    Re = _WGS84_A / denom                               # prime-vertical radius

    dlat = north / (Rn + h0)
    dlon = east  / ((Re + h0) * np.cos(lat0))

    lat = lat0 + dlat
    lon = lon0 + dlon
    alt_rel = -down # because down positive -> altitude decreases and relative to home position

    return (float(np.degrees(lat)), float(np.degrees(lon)), float(alt_rel))


# ----------------------------------------------------------------------
# Mission Specification Container
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MissionSpec:
    """
    Container describing mission geometry.

    name            : mission identifier
    takeoff_alt_m   : takeoff altitude (positive above home)
    speed_m_s       : cruise speed between waypoints
    waypoints_ned   : list of (N, E, D) waypoints
    land            : whether to land at end of mission

    Note: 
      - Altitude mode fixed to RelativeToHome (1)
      - Lists below are aligned by mission item index:
            len(command) == len(waypoints_lla) == len(waypoints_ned) == len(speed_m_s)
      - For commands that don't have a position (e.g., DO_CHANGE_SPEED),
        waypoints_* entry is None.
      - speed_m_s is only meaningful for DO_CHANGE_SPEED; otherwise None.
    """
    name: str
    home_position: List[LLA]    # list of home positions (lat, lon, alt) for each run
    command: List[int]          # list of MAVLink command IDs for each waypoint
    takeoff_alt_m: float        # takeoff altitude (positive above home)
    waypoints_ned: List[Optional[NED]]      # aligned with command list (None if no position)
    waypoints_lla: List[Optional[LLA]]      # aligned with command list (None if no position)
    speed_m_s: List[Optional[SPD]]          # aligned with command list (None if unused)
    land: bool = True


# ----------------------------------------------------------------------
# Pattern 1: 3-Point Turn Observation
# ----------------------------------------------------------------------
def make_turn_3pts(
    home_position: List[LLA],
    settle_m: float,
    leg1_m: float,
    leg2_m: float,
    turn_deg: float,
    alt_m: float,
    speed_m_s: float,
    land: bool = True, 
) -> MissionSpec:
    """
    3-point turn with configurable turn angle.

    Convention:
      - NED frame: +N forward, +E right, +D down
      - Heading angle psi is measured clockwise from North toward East (deg).

    Geometry:
        P1 = (settle_m, 0, -alt): velocity/control settle pre-roll (leg1_m) segment
            Why different from TO position? - from TO position to first waypoint, we cannot control flight speed.
        P2 = P1 + leg_m * [cos(0), sin(0)] = (settle_m + leg_m, 0, -alt) (north leg)
        P3 = P2 + leg2_m * [cos(turn), sin(turn)] (turned leg)

    Mission layout:
      TAKEOFF
      WP P1 (settle) : speed uncontrolled
      DO_CHANGE_SPEED(speed_m_s) : applies for subsequent legs
      WP P2
      WP P3
      RTL or LAND
    """

    if not home_position:
        raise ValueError("home_position must be non-empty")
    
    d = -float(alt_m)
    settle = float(settle_m)
    l1 = float(leg1_m)
    l2 = float(leg2_m)

    psi = np.radians(float(turn_deg))
    dN2 = l2 * np.cos(psi)
    dE2 = l2 * np.sin(psi)

    P1: NED = (settle, 0.0, d)
    P2: NED = (settle + l1, 0.0, d)
    P3: NED = (P2[0] + dN2, P2[1] + dE2, d)

    home = home_position

    # ------------------------------------------------------------------
    # Build aligned mission-item lists (same length)
    # ------------------------------------------------------------------
    commands: List[int] = [
        MAV_CMD_NAV_TAKEOFF,      # idx 0
        MAV_CMD_NAV_WAYPOINT,     # idx 1: P1
        MAV_CMD_DO_CHANGE_SPEED,  # idx 2
        MAV_CMD_NAV_WAYPOINT,     # idx 3: P2
        MAV_CMD_NAV_WAYPOINT,     # idx 4: P3
    ]
    if land:
        commands.append(MAV_CMD_NAV_LAND)  # idx 5

    waypoints_ned: List[Optional[NED]] = [
        None,   # TAKEOFF (we keep it None to avoid forcing fake points)
        P1,     # WP
        None,   # DO_CHANGE_SPEED
        P2,     # WP
        P3,     # WP
    ]
    if land:
        waypoints_ned.append(None)  # LAND: you may choose to land at home or last wp; keep None here

    waypoints_lla: List[Optional[LLA]] = [
        None,
        ned_to_lla(P1, home),
        None,
        ned_to_lla(P2, home),
        ned_to_lla(P3, home),
    ]
    if land:
        waypoints_lla.append(None)

    speeds: List[Optional[float]] = [
        None,        # TAKEOFF
        None,        # WP P1
        speed_m_s,   # DO_CHANGE_SPEED
        None,        # WP P2
        None,        # WP P3
    ]
    if land:
        speeds.append(None)

    # sanity check: all aligned
    assert len(commands) == len(waypoints_ned) == len(waypoints_lla) == len(speeds)

    return MissionSpec(
        name=f"turn3_settle_{int(round(turn_deg))}deg",
        home_position=home_position,
        command=commands,
        takeoff_alt_m=alt_m,
        waypoints_ned=waypoints_ned,
        waypoints_lla=waypoints_lla,
        speed_m_s=speeds,
        land=land,
    )

# ----------------------------------------------------------------------
# Pattern 2: 4-Point Square
# ----------------------------------------------------------------------
# def make_square4(size_m: float, alt_m: float, speed_m_s: float) -> MissionSpec:
#     """
#     Create a 4-point square trajectory in NED frame.

#     After takeoff from origin (0,0):
#         WP1: (size, 0)
#         WP2: (size, size)
#         WP3: (0, size)
#         WP4: (0, 0)

#     size_m : side length
#     alt_m  : altitude above home (positive)
#     """

#     # Convert altitude to NED Down convention
#     d = -float(alt_m)

#     s = float(size_m)

#     # Define square corners
#     wps: List[NED] = [
#         (s, 0.0, d),
#         (s, s, d),
#         (0.0, s, d),
#         (0.0, 0.0, d),
#     ]

#     return MissionSpec(
#         name="square4",
#         takeoff_alt_m=alt_m,
#         speed_m_s=speed_m_s,
#         waypoints_ned=wps,
#         land=True,
#     )




# ----------------------------------------------------------------------
# YAML Writer
# ----------------------------------------------------------------------
def write_scenario_yaml(
    run_dir: Path,
    mission: MissionSpec,
    *,
    # HERE  
    ardupilot_dir: str,
    ardupilot_vehicle: str,
    ardupilot_frame: str,
    ardupilot_model: str,
    ardupilot_world: str,
    ardupilot_location: str,
    ardupilot_mavproxy_outport: int,
    ardupilot_connect_port: int,
    px4_dir: str,
    px4_vehicle: int,
    px4_frame: str,
    px4_world: str,
    px4_location: str,
    px4_qgc_outport: int,
    px4_connect_port: int,
) -> None:
    """
    Write scenario.yaml for one run directory.

    MissionSpec alignment:
      len(command) == len(waypoints_ned) == len(waypoints_lla) == len(speed_m_s)

    For commands without position/speed, entries are None -> YAML null.
    """

    run_id = int(run_dir.name.split("_")[-1])

    # Pick home position for this run (if multiple homes are provided)
    if not mission.home_position:
        raise ValueError("MissionSpec.home_position must be a non-empty list")

    home_idx = min(run_id, len(mission.home_position) - 1)
    home = mission.home_position  # [lat, lon, alt]

    def ned_to_yaml(wp: Optional[tuple[float, float, float]]):
        if wp is None:
            return None
        n, e, d = wp
        return [float(n), float(e), float(d)]

    def lla_to_yaml(wp: Optional[tuple[float, float, float]]):
        if wp is None:
            return None
        lat, lon, alt = wp
        return [float(lat), float(lon), float(alt)]

    def spd_to_yaml(v: Optional[float]):
        return None if v is None else float(v)

    # Optional: sanity check alignment
    L = len(mission.command)
    if not (len(mission.waypoints_ned) == len(mission.waypoints_lla) == len(mission.speed_m_s) == L):
        raise ValueError(
            "MissionSpec lists must be aligned: "
            "len(command)=len(waypoints_ned)=len(waypoints_lla)=len(speed_m_s)"
        )

    scenario = {

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------
        "meta": {
            "run_id": run_id,
        },

        # --------------------------------------------------------------
        # Common configuration (shared mission geometry)
        # --------------------------------------------------------------
        "common": {
            "world": {
                "kind": "sdf",
                "path": world_sdf
            },
            "location": location,

            "scenario": {
                "name": mission.name,

                # Home for this run (LLA)
                "home_lla": [float(home[0]), float(home[1]), float(home[2])],

                # Mission-level flags/params
                "takeoff_alt_m": float(mission.takeoff_alt_m),
                "altitude_mode": 1,  # RelativeToHome fixed (as you stated)
                "land": bool(mission.land),

                # Mission items (aligned lists)
                "command": [int(c) for c in mission.command],

                # Optional per-item speed (only meaningful for DO_CHANGE_SPEED)
                "speed_m_s": [spd_to_yaml(v) for v in mission.speed_m_s],

                # Optional per-item waypoints (None -> null)
                "waypoints_ned": [ned_to_yaml(wp) for wp in mission.waypoints_ned],
                "waypoints_lla": [lla_to_yaml(wp) for wp in mission.waypoints_lla],
            },
        },

        # --------------------------------------------------------------
        # Autopilot-specific simulation settings
        # --------------------------------------------------------------
        "autopilots": {

            # ---------------- ArduPilot ----------------
            "ardupilot": {
                "sim": {
                    "vehicle": "ArduCopter",
                    "frame": "gazebo-iris",
                    "model": "JSON",
                    "instance": run_id,
                    "out_udp_port": int(ap_port),
                    "startup_delay_s": 5,
                    "max_run_s": 180,
                    "grace_s": 10,
                },
                "mavsdk": {
                    "connect_url": f"udp://127.0.0.1:{ap_port}"
                },
            },

            # ---------------- PX4 ----------------
            "px4": {
                "sim": {
                    "px4_dir": px4_dir,
                    "world": px4_world,
                    "autostart": 4001,
                    "model": "gz_x500",
                    "out_udp_port": int(px4_port),
                    "startup_delay_s": 5,
                    "max_run_s": 180,
                    "grace_s": 10,
                },
                "mavsdk": {
                    "connect_url": f"udp://127.0.0.1:{px4_port}"
                },
            },
        },
    }

    (run_dir / "scenario.yaml").write_text(
        yaml.safe_dump(scenario, sort_keys=False),
        encoding="utf-8",
    )

# ----------------------------------------------------------------------
# Main Entry
# ----------------------------------------------------------------------
def main() -> int:

    # home = (40.4117616, -86.9335208, 0.0)  # (lat, lon, alt) 

    # ned_wpts = [
    #     (0.0, 0.0, -5.0),     # 5m up
    #     (10.0, 0.0, -5.0),
    #     (10.0, 10.0, -5.0),
    # ]

    # lla_wpts = [ned_to_lla(w, home) for w in ned_wpts]
    # print ("NED waypoints:", ned_wpts)
    # print ("LLA waypoints:", lla_wpts)

    """
    CLI entry point.

    Responsibilities:
        1. Parse arguments
        2. Build mission geometry
        3. Create run directories
        4. Generate scenario.yaml for each run
    """

    ap = argparse.ArgumentParser(
        description="Scenario generator (pattern-specific)"
    )

    # Output settings
    ap.add_argument("--outdir", type=Path, default=Path("./data"))
    ap.add_argument("--runs", type=int, default=1)

    # Mission pattern selection
    ap.add_argument(
        "--pattern",
        choices=["turn3pts", "square4pts"],
        default="turn3pts",
    )

    # Common mission parameters
    ap.add_argument("--home-lla", type=float, nargs=3,
    metavar=("LAT", "LON", "ALT"), default=[40.41176161953683, -86.93352081596879, 0.0],
    help="Home position (lat lon alt)")

    # Pattern-specific geometry knobs: 3-point turn
    def parse_turn_deg(v: str):
        if v.lower() == "random":
            return "random"
        try:
            return float(v)
        except ValueError:
            raise argparse.ArgumentTypeError(
                'turn-deg must be a float or "random"'
            )
    
    ap.add_argument("--settle-m", type=float, default=10.0)
    ap.add_argument("--leg1-m", type=float, default=15.0)
    ap.add_argument("--leg2-m", type=float, default=15.0)
    ap.add_argument("--turn-deg", type=parse_turn_deg, default="random", help='Turn angle (deg) or "random"')
    ap.add_argument('--alt-m', type=float, default=10.0)
    ap.add_argument('--speed-m-s', type=float, default=6.0)
    ap.add_argument('--land', type=bool, default=True, help="Whether to land at the end of the mission")

    # # Environment configuration
    # ap.add_argument("--location", type=str, default="Purdue")
    ap.add_argument("--world-sdf", type=str, default="worlds/iris_runway.sdf")

    # # UDP ports (offset per run)
    ap.add_argument("--ap-port-base", type=int, default=14550)
    ap.add_argument("--px4-port-base", type=int, default=14650)

    # # PX4-specific defaults
    ap.add_argument(
        "--px4-dir",
        type=str,
        default="/home/mhcho/ws/flightstack_sim/ap/px4/PX4-Autopilot"
    )
    ap.add_argument("--px4-world", type=str, default="windy")

    args = ap.parse_args()

    # Creating output directory (if not exists)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # Generate runs
    # --------------------------------------------------------------
    for i in range(args.runs):

        run_dir = args.outdir / f"run_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if args.turn_deg == "random":
            turn_deg = np.random.uniform(0, 360)
        else:
            turn_deg = args.turn_deg

        mission = make_turn_3pts(
            home_position=args.home_lla,
            settle_m=args.settle_m,
            leg1_m=args.leg1_m,
            leg2_m=args.leg2_m,
            turn_deg=turn_deg,
            alt_m=args.alt_m,
            speed_m_s=args.speed_m_s,
            land=args.land,
        )

        write_scenario_yaml(
            run_dir,
            mission,
            location="Purdue",
            world_sdf=args.world_sdf,
            ap_port=args.ap_port_base,
            px4_port=args.px4_port_base,
            px4_dir=args.px4_dir,
            px4_world=args.px4_world,
        )

    print(
        f"Generated {args.runs} runs "
        f"with pattern='{args.pattern}' "
        f"under {args.outdir}"
    )
        
    return 0


if __name__ == "__main__":
    raise SystemExit(main())