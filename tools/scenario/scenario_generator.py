#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import math
import yaml


NED = Tuple[float, float, float]


@dataclass(frozen=True)
class MissionSpec:
    name: str
    takeoff_alt_m: float
    speed_m_s: float
    waypoints_ned: List[NED]
    land: bool = True


def make_square4(size_m: float, alt_m: float, speed_m_s: float) -> MissionSpec:
    """
    4-point square in NED frame, starting from origin (0,0) after takeoff.
    size_m: side length
    alt_m: positive altitude above home; Down = -alt_m
    """
    d = -float(alt_m)
    s = float(size_m)
    wps: List[NED] = [
        (s, 0.0, d),
        (s, s, d),
        (0.0, s, d),
        (0.0, 0.0, d),
    ]
    return MissionSpec(name="square4", takeoff_alt_m=alt_m, speed_m_s=speed_m_s, waypoints_ned=wps, land=True)


def make_turn3(
    leg1_m: float,
    lateral_m: float,
    leg2_m: float,
    alt_m: float,
    speed_m_s: float,
) -> MissionSpec:
    """
    Simple "3 points turn observation" pattern.

    After takeoff at origin:
      P1: forward along North (leg1_m)
      P2: move laterally East (lateral_m) while holding North
      P3: move forward again North by leg2_m while holding East

    This yields a clear "straight -> turn -> straight" signature.
    """
    d = -float(alt_m)
    l1 = float(leg1_m)
    lat = float(lateral_m)
    l2 = float(leg2_m)

    wps: List[NED] = [
        (l1, 0.0, d),
        (l1, lat, d),
        (l1 + l2, lat, d),
    ]
    return MissionSpec(name="turn3", takeoff_alt_m=alt_m, speed_m_s=speed_m_s, waypoints_ned=wps, land=True)


def write_scenario_yaml(
    run_dir: Path,
    mission: MissionSpec,
    *,
    location: str,
    world_sdf: str,
    ap_port: int,
    px4_port: int,
    px4_dir: str,
    px4_world: str,
) -> None:
    """
    Write one scenario.yaml that can be used by both ArduPilot and PX4 runs.
    Only the mission geometry is "common"; sim settings differ per autopilot.
    """
    scenario = {
        "meta": {
            "run_id": int(run_dir.name.split("_")[-1]),
        },
        "common": {
            "world": {"kind": "sdf", "path": world_sdf},
            "location": location,
            "scenario": {
                "name": mission.name,
                "takeoff_alt_m": float(mission.takeoff_alt_m),
                "speed_m_s": float(mission.speed_m_s),
                # store as lists to keep YAML clean
                "waypoints_ned": [[float(n), float(e), float(d)] for (n, e, d) in mission.waypoints_ned],
                "land": bool(mission.land),
            },
        },
        "autopilots": {
            "ardupilot": {
                "sim": {
                    "vehicle": "ArduCopter",
                    "frame": "gazebo-iris",
                    "model": "JSON",
                    "instance": int(run_dir.name.split("_")[-1]),
                    "out_udp_port": int(ap_port),
                    "startup_delay_s": 5,
                    "max_run_s": 180,
                    "grace_s": 10,
                },
                "mavsdk": {"connect_url": f"udp://127.0.0.1:{ap_port}"},
            },
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
                "mavsdk": {"connect_url": f"udp://127.0.0.1:{px4_port}"},
            },
        },
    }

    (run_dir / "scenario.yaml").write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Scenario generator (pattern-specific)")
    ap.add_argument("--outdir", type=Path, default=Path("./data"))
    ap.add_argument("--runs", type=int, default=10)

    # choose scenario pattern
    ap.add_argument("--pattern", choices=["turn3", "square4"], default="square4")

    # minimal common parameters
    ap.add_argument("--alt-m", type=float, default=5.0)
    ap.add_argument("--speed-m-s", type=float, default=2.0)

    # pattern geometry knobs (keep few)
    ap.add_argument("--size-m", type=float, default=10.0, help="square side length (for square4)")
    ap.add_argument("--leg1-m", type=float, default=15.0, help="turn3 first straight length")
    ap.add_argument("--lateral-m", type=float, default=8.0, help="turn3 lateral offset")
    ap.add_argument("--leg2-m", type=float, default=10.0, help="turn3 second straight length")

    # environment defaults
    ap.add_argument("--location", type=str, default="Purdue")
    ap.add_argument("--world-sdf", type=str, default="worlds/iris_runway.sdf")

    # ports (keep simple)
    ap.add_argument("--ap-port-base", type=int, default=14550)
    ap.add_argument("--px4-port-base", type=int, default=14650)

    # px4 defaults
    ap.add_argument("--px4-dir", type=str, default="/home/mhcho/ws/flightstack_sim/ap/px4/PX4-Autopilot")
    ap.add_argument("--px4-world", type=str, default="windy")

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # build mission spec once (same geometry for all runs)
    if args.pattern == "square4":
        mission = make_square4(size_m=args.size_m, alt_m=args.alt_m, speed_m_s=args.speed_m_s)
    else:
        mission = make_turn3(
            leg1_m=args.leg1_m,
            lateral_m=args.lateral_m,
            leg2_m=args.leg2_m,
            alt_m=args.alt_m,
            speed_m_s=args.speed_m_s,
        )

    for i in range(args.runs):
        run_dir = args.outdir / f"run_{i:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        ap_port = args.ap_port_base + i
        px4_port = args.px4_port_base + i

        write_scenario_yaml(
            run_dir,
            mission,
            location=args.location,
            world_sdf=args.world_sdf,
            ap_port=ap_port,
            px4_port=px4_port,
            px4_dir=args.px4_dir,
            px4_world=args.px4_world,
        )

    print(f"Generated {args.runs} runs with pattern='{args.pattern}' under {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())