#!/usr/bin/env python3
"""
Multi-run launcher for ArduPilot ArduCopter SITL + Gazebo (gz sim).

What this script does:
  - Repeats N simulation runs.




  - For each run:
      1) Launch ArduPilot SITL via sim_vehicle.py (gazebo-iris, JSON model)
      2) Wait a bit for SITL to initialize
      3) Launch Gazebo (gz sim) with the provided SDF world
      4) Keep both processes alive until:
           - one of them exits, OR
           - max runtime is reached
      5) Terminate both process trees cleanly (SIGTERM then SIGKILL)

Key design choice:
  - Each launched command runs in its own *process group* (Linux/macOS).
    This allows us to kill the whole subtree (sim_vehicle.py typically spawns
    multiple children: mavproxy, arducopter SITL, etc.).
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO


@dataclass
class ProcHandle:
    """Container holding a running process and its log file info."""
    name: str
    proc: subprocess.Popen
    log_path: Path
    log_file: TextIO  # keep handle so it doesn't get GC'ed early


def _popen(
    name: str,
    cmd: list[str],
    cwd: Optional[Path],
    log_path: Path,
) -> ProcHandle:
    """
    Start a subprocess and redirect stdout/stderr into a log file.

    - The process is started in a new process group (POSIX) so we can terminate
      the whole tree (parent + children) later.
    - Logs are line-buffered for real-time tailing (tail -f).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Line-buffered text logs (buffering=1 works with text=True).
    f = open(log_path, "w", buffering=1, encoding="utf-8")

    # preexec_fn=os.setsid: create a new process group/session (POSIX).
    # On Windows, os.setsid is not available; we fall back to default behavior.
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid if os.name != "nt" else None, # new process group
    )

    return ProcHandle(name=name, proc=proc, log_path=log_path, log_file=f)


def _kill_tree(ph: ProcHandle, grace_s: float = 5.0) -> None:
    """
    Terminate a process *and its child processes*.

    Strategy:
      1) Send SIGTERM to the process group (POSIX) / terminate() on Windows.
      2) Wait up to `grace_s` for a clean exit.
      3) If still alive, force kill (SIGKILL / kill()).

    This is important for sim_vehicle.py because it often spawns:
      - mavproxy.py
      - the SITL binary (arducopter)
      - auxiliary helper processes
    """
    # If already exited, nothing to do.
    if ph.proc.poll() is not None:
        return

    # Soft terminate
    try:
        if os.name != "nt":
            # Kill the entire process group
            os.killpg(os.getpgid(ph.proc.pid), signal.SIGTERM)
        else:
            ph.proc.terminate()
    except Exception:
        # We ignore exceptions because sometimes processes exit between checks.
        pass

    # Wait for graceful shutdown
    t0 = time.time()
    while time.time() - t0 < grace_s:
        if ph.proc.poll() is not None:
            return
        time.sleep(0.1)

    # Hard kill
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(ph.proc.pid), signal.SIGKILL)
        else:
            ph.proc.kill()
    except Exception:
        pass


def _finalize_proc(ph: Optional[ProcHandle]) -> None:
    """Best-effort: ensure process is dead and log file is closed."""
    if ph is None:
        return
    try:
        _kill_tree(ph)
    except Exception:
        pass
    try:
        ph.proc.wait(timeout=2)
    except Exception:
        pass
    try:
        ph.log_file.flush()
    except Exception:
        pass
    try:
        ph.log_file.close()
    except Exception:
        pass


def build_sitl_cmd(instance: int, out_port: int, location: str) -> list[str]:
    """
    Build the sim_vehicle.py command for ArduCopter SITL.

    Notes:
      - `-I <instance>` helps separate multiple runs (some paths/ports are derived).
      - `--out=udp:127.0.0.1:<port>` is useful if you want to connect QGC/MAVSDK.
      - `--no-rebuild` makes repeated runs faster.
    """
    return [
        "sim_vehicle.py",
        "-v", "ArduCopter",
        "-f", "gazebo-iris",
        "--model", "JSON",
        "--map",
        "--console",
        f"--out=udp:127.0.0.1:{out_port}",
        f"--location={location}",
        "-I", str(instance),
        "--no-rebuild",
    ]


def build_gz_cmd(world_sdf: str, verbose: str = "-v4") -> list[str]:
    """
    Build the Gazebo Sim command.

    Example:
      gz sim -v4 -r iris_runway.sdf
    """
    return ["gz", "sim", verbose, "-r", world_sdf]


def run_once(
    run_id: int,
    base_dir: Path,
    world_sdf: str,
    location: str,
    instance_base: int,
    out_port_base: int,
    startup_delay_s: float,
    max_run_s: float,
    # NEW: pass in stop flag and current process holders for signal handler
    stop: dict,
    current: dict,
) -> int:
    run_dir = base_dir / f"run_{run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    instance = instance_base + run_id
    out_port = out_port_base + run_id

    sitl_log = run_dir / "sitl.log"
    gz_log = run_dir / "gazebo.log"

    gz = _popen("gazebo", build_gz_cmd(world_sdf), cwd=run_dir, log_path=gz_log)
    current["gz"] = gz

    time.sleep(startup_delay_s)

    sitl = _popen("sitl", build_sitl_cmd(instance, out_port, location), cwd=run_dir, log_path=sitl_log)
    current["sitl"] = sitl

    t0 = time.time()
    rc = 0

    while True:
        # NEW: if user pressed Ctrl+C, exit loop immediately
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

    # Cleanup (always)
    _finalize_proc(current.get("gz"))
    _finalize_proc(current.get("sitl"))
    current["gz"] = None
    current["sitl"] = None

    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="./data/run_xxx directory containing scenario.yaml")
    ap.add_argument("--instance-base", type=int, default=0)
    args = ap.parse_args()
    
    
    run_dir = args.run_dir.resolve()
    scenario_path = run_dir / "scenario.yaml"
    logs_dir = run_dir / "ardu_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)



    # ap.add_argument("--out-dir", type=Path, default=Path("./data/run_xxx/ardu_logs"))

    # ap.add_argument("--world", type=str, default="iris_runway.sdf")
    # ap.add_argument("--location", type=str, default="Purdue")
    
    # ap.add_argument("--out-port-base", type=int, default=14550)
    # ap.add_argument("--startup-delay", type=float, default=5.0)
    # ap.add_argument("--max-run-s", type=float, default=60.0)

    # args.outdir.mkdir(parents=True, exist_ok=True)

    stop = {"flag": False}
    current = {"sitl": None, "gz": None}

    def _sig(_signum, _frame):
        # Mark stop
        stop["flag"] = True
        # IMPORTANT: immediately kill running processes (so ArduPilot doesn't linger)
        _finalize_proc(current.get("gz"))
        _finalize_proc(current.get("sitl"))
        current["gz"] = None
        current["sitl"] = None

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    for i in range(args.runs):
        if stop["flag"]:
            print("Interrupted. Exiting.")
            return 130

        print(f"\n=== RUN {i+1}/{args.runs} ===")
        rc = run_once(
            run_id=i,
            base_dir=args.outdir,
            world_sdf=args.world,
            location=args.location,
            instance_base=args.instance_base,
            out_port_base=args.out_port_base,
            startup_delay_s=args.startup_delay,
            max_run_s=args.max_run_s,
            stop=stop,
            current=current,
        )
        print(f"Run {i+1} finished with rc={rc}. Logs in {args.outdir / f'run_{i:03d}'}")

        if stop["flag"]:
            return 130

        time.sleep(2.0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())