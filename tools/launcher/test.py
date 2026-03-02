#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

import yaml


@dataclass
class ProcHandle:
    name: str
    proc: subprocess.Popen
    log_file: TextIO
    log_path: Path


def popen_pg(name: str, cmd: list[str], cwd: Path, log_path: Path) -> ProcHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", buffering=1, encoding="utf-8")
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )
    return ProcHandle(name, p, f, log_path)


def kill_tree(ph: Optional[ProcHandle], grace_s: float) -> None:
    if ph is None or ph.proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(ph.proc.pid), signal.SIGTERM)
        else:
            ph.proc.terminate()
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if ph.proc.poll() is not None:
            break
        time.sleep(0.1)

    if ph.proc.poll() is None:
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(ph.proc.pid), signal.SIGKILL)
            else:
                ph.proc.kill()
        except Exception:
            pass

    try:
        ph.proc.wait(timeout=2)
    except Exception:
        pass
    try:
        ph.log_file.flush()
        ph.log_file.close()
    except Exception:
        pass


def build_sitl_cmd(cfg: dict) -> list[str]:
    # NOTE: map/console은 sim_vehicle가 별도 창 띄울 수 있으니 여기서는 끄는 게 배치에 안정적
    return [
        "sim_vehicle.py",
        "-v", cfg.get("vehicle", "ArduCopter"),
        "-f", cfg.get("model", "gazebo-iris"),
        "--model", "JSON",
        f"--out=udp:127.0.0.1:{int(cfg.get('out_udp_port', 14550))}",
        f"--location={cfg.get('location', 'Purdue')}",
        "-I", str(int(cfg.get("instance", 0))),
        "--no-rebuild",
        "--no-mavproxy",
        "--no-console",
        "--no-map",
    ]


def build_gz_cmd(world_path: str) -> list[str]:
    return ["gz", "sim", "-v4", "-r", world_path]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="data/run_xxx directory containing scenario.yaml")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    scenario_path = run_dir / "scenario_000.yaml"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    sim = cfg["sim"]
    world = sim.get("world", "worlds/iris_runway.sdf")
    startup_delay_s = float(sim.get("startup_delay_s", 5))
    max_run_s = float(sim.get("max_run_s", 120))
    grace_s = float(sim.get("grace_s", 10))

    sitl = None
    gz = None
    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True
        kill_tree(gz, grace_s)
        kill_tree(sitl, grace_s)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    sitl = popen_pg("sitl", build_sitl_cmd(sim), cwd=run_dir, log_path=logs_dir / "sitl.log")
    time.sleep(startup_delay_s)
    gz = popen_pg("gazebo", build_gz_cmd(str(Path(world).resolve() if Path(world).exists() else world)),
                 cwd=run_dir, log_path=logs_dir / "gazebo.log")

    t0 = time.time()
    rc = 0
    while True:
        if stop["flag"]:
            rc = 130
            break
        if sitl.proc.poll() is not None:
            rc = sitl.proc.returncode or 0
            break
        if gz.proc.poll() is not None:
            rc = gz.proc.returncode or 0
            break
        if time.time() - t0 > max_run_s:
            rc = 0
            break
        time.sleep(0.2)

    kill_tree(gz, grace_s)
    kill_tree(sitl, grace_s)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())