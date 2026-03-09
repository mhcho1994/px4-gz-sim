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
import sys
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from tracemalloc import stop
from typing import Any, Optional, TextIO
import yaml

_THIS_FILE = Path(__file__).resolve()
_TOOLS_DIR = _THIS_FILE.parents[1]          # .../tools
_COMMANDER_DIR = _TOOLS_DIR / "commander"

if str(_COMMANDER_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMANDER_DIR))

# from mavsdk_ardupilot_commander import ArduPilotMissionRunner, MissionState

@dataclass
class ScenarioConfig:
    # sim instance parameters
    instance: int = 0
    outport: int = 14550
    startup_delay_s: float = 0.0
    max_run_s: float = 60.0

    # sim setup
    vehicle: str = "ArduCopter"
    frame: str = "gazebo-iris"
    model: str = "JSON"
    world: str = "iris_runway"
    location: str = "Purdue"
    scenario_name: str = "turn3pts"


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
    verbose: bool = False
) -> ProcHandle:
    """
    Start a subprocess and redirect stdout/stderr into a log file.

    - The process is started in a new process group (POSIX) so we can terminate
      the whole tree (parent + children) later.
    - Logs are line-buffered for real-time tailing (tail -f).
    """
    print(f"[LAUNCH] {name}: {' '.join(cmd)}")
    print(f"[CWD]    {name}: {cwd if cwd else os.getcwd()}")
    print(f"[LOG]    {name}: {log_path}")

    if verbose:
        # TODO: add virtual env or other environment variables if needed
        print(f"[ENV]    PATH={os.environ.get('PATH','')}")

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


def _run_quiet(cmd: str) -> None:
    """Run a shell command quietly, ignoring all errors."""
    try:
        subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        pass


def _pkill_patterns(patterns: list[str], sig: str) -> None:
    """
    Send a signal to processes matched by full command line.

    Parameters
    ----------
    patterns : list[str]
        Patterns passed to `pkill -f`.
    sig : str
        Signal name such as 'TERM' or 'KILL'.
    """
    if os.name == "nt":
        return

    for pattern in patterns:
        _run_quiet(f"pkill -{sig} -f {shlex.quote(pattern)}")


def _cleanup_gz_processes() -> None:
    """
    Best-effort cleanup for Gazebo / gz sim related processes that may survive
    after killing the parent shell/process group.

    Why needed:
    - Killing the bash PID or even its process group does not always terminate
      the actual gz server process.
    - Depending on the setup, Gazebo may appear as:
        * gz sim
        * gzserver / gzclient
        * ign gazebo / ignition gazebo
    """
    if os.name == "nt":
        return

    patterns = [
        r"gz sim",
        r"gz sim server",
        r"gz sim gui",
    ]

    # Graceful termination first
    _pkill_patterns(patterns, "INT")

def _cleanup_sitl_processes() -> None:
    """Best-effort cleanup for common ArduPilot SITL related processes."""
    if os.name == "nt":
        return

    patterns = [
        r"sim_vehicle.py",
        r"arducopter",
        r"ArduCopter",
        r"arduplane",
        r"ArduPlane",
        r"ardurover",
        r"ArduRover",
        r"ardusub",
        r"ArduSub",
        r"antennatracker",
        r"AntennaTracker",
    ]

    _pkill_patterns(patterns, "INT")


def _cleanup_mavproxy_processes() -> None:
    """
    Best-effort cleanup for MAVProxy related processes that may survive
    after killing the parent shell/process group.

    Why needed:
    - Killing the bash PID or even its process group does not always terminate
      the actual mavproxy process.
    - Depending on the setup, MAVProxy may appear as:
        * mavproxy
    """
    if os.name == "nt":
        return

    patterns = [
        r"mavproxy",
    ]

    # Graceful termination first
    _pkill_patterns(patterns, "INT")


# def _kill_tree(ph: ProcHandle, grace_s: float = 5.0) -> None:
    # """
    # Terminate a process *and its child processes*.

    # Strategy:
    #   1) Send SIGTERM to the process group (POSIX) / terminate() on Windows.
    #   2) Wait up to `grace_s` for a clean exit.
    #   3) If still alive, force kill (SIGKILL / kill()).
    #   4) Additionally clean up stray gz/Gazebo processes that may survive
    #      outside the expected process group.

    # This is important for sim_vehicle.py because it often spawns:
    #   - mavproxy.py
    #   - the SITL binary (arducopter)
    #   - auxiliary helper processes

    # And for gz sim because:
    #   - killing the wrapper bash PID is sometimes not enough
    #   - the actual simulator process may remain alive
    # """
    # # If already exited, still do a best-effort gz cleanup because the child
    # # wrapper may be gone while gz itself is still alive.
    # already_exited = (ph.proc.poll() is not None)

    # if not already_exited:
    #     # Soft terminate
    #     try:
    #         if os.name != "nt":
    #             # Kill the entire process group
    #             os.killpg(os.getpgid(ph.proc.pid), signal.SIGTERM)
    #         else:
    #             ph.proc.terminate()
    #     except Exception:
    #         # Process may have exited between checks
    #         pass

    #     # Wait for graceful shutdown
    #     t0 = time.time()
    #     while time.time() - t0 < grace_s:
    #         if ph.proc.poll() is not None:
    #             break
    #         time.sleep(0.1)

    #     # Hard kill if still alive
    #     if ph.proc.poll() is None:
    #         try:
    #             if os.name != "nt":
    #                 os.killpg(os.getpgid(ph.proc.pid), signal.SIGKILL)
    #             else:
    #                 ph.proc.kill()
    #         except Exception:
    #             pass



def _kill_tree(ph: ProcHandle, grace_s: float = 5.0) -> None:
    """
    Terminate a process and its descendants.

    For interactive tools such as sim_vehicle.py, we try SIGINT first
    because it is closer to Ctrl-C and may allow cleaner shutdown.
    Then we escalate to SIGTERM and finally SIGKILL.
    """
    if ph.proc.poll() is not None:
        return

    if os.name != "nt":
        try:
            pgid = os.getpgid(ph.proc.pid)
        except Exception:
            pgid = None
    else:
        pgid = None

    # 1) Try SIGINT first (best for sim_vehicle.py / SITL style processes)
    try:
        if os.name != "nt" and pgid is not None:
            os.killpg(pgid, signal.SIGINT)
        else:
            ph.proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < min(grace_s, 2.0):
        if ph.proc.poll() is not None:
            return
        time.sleep(0.1)

    # 2) Escalate to SIGTERM
    try:
        if os.name != "nt" and pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            ph.proc.terminate()
    except Exception:
        pass

    t1 = time.time()
    while time.time() - t1 < max(grace_s - 2.0, 1.0):
        if ph.proc.poll() is not None:
            return
        time.sleep(0.1)

    # 3) Final hard kill
    try:
        if os.name != "nt" and pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            ph.proc.kill()
    except Exception:
        pass

    # Final fallback cleanup for Gazebo/gz leftovers
    if "gz" in ph.name:
        _cleanup_gz_processes()

    if "sitl" in ph.name:
        _cleanup_sitl_processes()

    if "mavproxy" in ph.name:
        _cleanup_mavproxy_processes()

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
      TODO: adding the following options
        "-I", str(instance),
        "--no-rebuild",
    """
    return [
        "sim_vehicle.py",
        "-v", "ArduCopter",
        "-f", "gazebo-iris",
        "--model", "JSON",
        f"--location={location}",
        f"--out=udp:127.0.0.1:{out_port}",
    ]


def build_mavproxy_cmd(out_port: int) -> list[str]:
    """
    Build the mavproxy command to connect to the SITL instance.
    Open the mavconsole and map to monitor MAVLink messages.

    Example:
      mavproxy.py --master=udp:127.0.0.1:14550
    """

    return [
        "mavproxy.py", 
        f"--master=udp:127.0.0.1:{out_port}",
        "--map",
        "--console",
    ]


def build_gz_cmd(world_sdf: str, verbose: str = "-v4") -> list[str]:
    """
    Build the Gazebo Sim command.

    Example:
      gz sim -v4 -r iris_runway.sdf
    """
    return ["gz", "sim", verbose, "-r", world_sdf]


# def build_launch_cmd(run_dir: Path, scenario_path: Path) -> ProcHandle:
#     cmd = [
#         "python3",
#         "tools/commander/mavsdk_ardupilot_commander.py",
#         "--scenario",
#         str(scenario_path),
#     ]
#     return launch_process(
#         name="commander",
#         cmd=cmd,
#         cwd=Path("."),
#         log_path=run_dir / "commander.log",
#     )


def run_once(
    instance: int,
    logs_dir: Path,
    outport: int,
    startup_delay_s: float,
    max_run_s: float,
    stop: dict,
    current: dict,
    vehicle: str,
    frame: str,
    model: str, 
    world: str,
    location: str,
) -> int:

    sitl_log = logs_dir / "sitl.log"
    gz_log = logs_dir / "gazebo.log"
    mavproxy_log = logs_dir / "mavproxy.log"

    # run_dir = logs_dir.parent

    if current.get("gz") is None:
        time.sleep(startup_delay_s)
        print(f"[INFO] Waiting {startup_delay_s:.1f}s for Gazebo to initialize...")

        gz = _popen("gazebo", build_gz_cmd(f"{world}.sdf", "-v4"), cwd=logs_dir, log_path=gz_log)
        current["gz"] = gz

    if current.get("mavproxy") is None:
        time.sleep(startup_delay_s)
        print(f"[INFO] Waiting {startup_delay_s:.1f}s for MAVProxy to initialize...")

        mavproxy = _popen("mavproxy", build_mavproxy_cmd(outport), cwd=logs_dir, log_path=mavproxy_log)
        current["mavproxy"] = mavproxy

    time.sleep(startup_delay_s)
    print(f"[INFO] Waiting {startup_delay_s:.1f}s for SITL to initialize...")

    sitl = _popen("sitl", build_sitl_cmd(instance, outport, location), cwd=logs_dir, log_path=sitl_log)
    current["sitl"] = sitl

    runner = ArduPilotMissionRunner(scenario_path=run_dir / "scenario.yaml")
    runner.start()

    # t0 = time.time()
    # rc = -1

    while True:
        if stop["flag"]:
            print("[STOP] user interrupt")
            runner.request_stop()
            rc = 130
            break

    #     if time.time() - t0 > max_run_s:
    #         print(f"[TIMEOUT] exceeded {max_run_s:.1f}s")
    #         runner.request_stop()
    #         rc = 124
    #         break

    #     status = runner.get_status()

    #     print(
    #         f"[MISSION] state={status.state.name} "
    #         f"done={status.done} success={status.success} "
    #         f"msg={status.message}"
    #     )

    #     if status.done:
    #         if status.success:
    #             rc = 0
    #         else:
    #             rc = 1
    #             print(f"[MISSION] error: {status.error}")
    #         break

    #     time.sleep(0.5)

    #     runner.join(timeout=2.0)

    #     _finalize_proc(current.get("sitl"))
    #     current["sitl"] = None

    # t0 = time.time()

    # while True:
    #     # 1) user Ctrl+C
    #     if stop["flag"]:
    #         rc = 130
    #         break

    #     # 2) max runtime exceeded
    #     if time.time() - t0 > max_run_s:
    #         print(f"[TIMEOUT] exceeded {max_run_s:.1f}s")
    #         rc = 124
    #         break

    #     # 3) commander finished
    #     cmd_ph = current.get("commander")
    #     if cmd_ph is not None:
    #         cmd_rc = cmd_ph.proc.poll()
    #         if cmd_rc is not None:
    #             print(f"[INFO] commander finished with rc={cmd_rc}")
    #             rc = cmd_rc
    #             break

    #     time.sleep(0.2)

    #     _finalize_proc(current.get("sitl"))
    #     current["sitl"] = None

    rc = 0

    return rc


def _load_scenario_yaml(run_dir: Path) -> ScenarioConfig:
    """
    Load scenario.yaml from run_dir.

    Supported keys (example):
      instance_base: 0
      world: iris_runway.sdf
      location: Purdue
    """
    scenario_path = run_dir / "scenario.yaml"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario.yaml: {scenario_path}")

    data: dict[str, Any] = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}

    # Accept both top-level keys and nested (e.g., {"sim": {...}})
    # You can extend this mapping if your scenario.yaml has a different schema.
    sim = data.get("autopilots",{}).get("ardupilot",{}).get("sim",{})
    scenario = data.get("common",{}).get("scenario",{})
    cfg = ScenarioConfig(
        instance=int(sim.get("instance", ScenarioConfig.instance)),
        vehicle=str(sim.get("vehicle", ScenarioConfig.vehicle)),
        frame=str(sim.get("frame", ScenarioConfig.frame)),
        model=str(sim.get("model", ScenarioConfig.model)),
        world=str(sim.get("world", ScenarioConfig.world)),
        location=str(sim.get("location", ScenarioConfig.location)),
        scenario_name=str(scenario.get("name", ScenarioConfig.scenario_name))
    )
    return cfg


def _iter_run_dirs(data_root: Path) -> list[Path]:
    """
    Find ./data/run_* directories, sorted by name.
    """
    if not data_root.exists():
        return []
    return sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def _should_skip_run_dir(run_dir: Path, force: bool) -> bool:
    """
    Skip if 'ardu_logs' exists.
    """
    logs_dir = run_dir / "ardu_logs"
    if force:
        return False
    bin_files = list(logs_dir.rglob("*.BIN"))
    return logs_dir.exists() and (len(bin_files) > 0)


def main() -> int:
    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("./data"), help="Root folder containing run_xxx/scenario.yaml and run_xxx/ardu_logs")
    ap.add_argument("--force", action="store_true", help="Re-run even if run_xxx/ardu_logs exists")
    ap.add_argument("--outport", type=int, default=14550)
    ap.add_argument("--startup-delay-s", type=float, default=5.0)
    ap.add_argument("--max-run-s", type=float, default=60.0)
    args = ap.parse_args()

    # Access to the resolved run root path
    data_root = args.run_root.resolve()

    stop = {"flag": False}
    current: dict[str, Optional["ProcHandle"]] = {"gz": None, "mavproxy": None, "sitl": None, "commander": None}  

    # Mark stop flag and terminate running processes on SIGINT/SIGTERM
    def _sig(_signum, _frame):
        stop["flag"] = True
        # immediately kill running processes (so ArduPilot doesn't linger)
        _finalize_proc(current.get("gz"))
        _finalize_proc(current.get("mavproxy"))
        _finalize_proc(current.get("sitl"))
        # _finalize_proc(current.get("commander"))
        current["gz"] = None
        current["mavproxy"] = None
        current["sitl"] = None
        # current["commander"] = None

    # Terminate on Ctrl+C or SIGTERM
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Iterations for run directories
    run_dirs = _iter_run_dirs(data_root)
    if not run_dirs:
        print(f"No run_xxx directories found under: {data_root}")
        return 2
    
    overall_rc = 0

    for run_dir in run_dirs:
        if stop["flag"]:
            print("Interrupted. Exiting.")
            return 130

        if _should_skip_run_dir(run_dir, force=args.force):
            print(f"[SKIP] {run_dir} (ardu_logs exists)")
            continue

        # Load scenario.yaml
        try:
            cfg = _load_scenario_yaml(run_dir)
        except Exception as e:
            print(f"[ERROR] {run_dir}: failed to load scenario.yaml: {e}")
            overall_rc = 2
            continue

        # Create output directory for this run
        logs_root = run_dir / "ardu_logs"
        logs_root.mkdir(parents=True, exist_ok=True)

        # Set other configurations
        cfg.outport = args.outport
        cfg.startup_delay_s = args.startup_delay_s
        cfg.max_run_s = args.max_run_s

        print(f"\n=== SCENARIO: {run_dir.name} ===")
        print(f"  world={cfg.world} location={cfg.location}")
        print(f"  instance={cfg.instance} outport={cfg.outport}")
        print(f"  startup_delay={cfg.startup_delay_s} max_run_s={cfg.max_run_s}")
        print(f"  logs_root={logs_root}")

        # Execute the scenario
        print(f"\n--- {run_dir.name}: RUN {cfg.scenario_name} Scenario ---")
        rc = run_once(
            instance=cfg.instance,
            logs_dir=logs_root,
            outport=cfg.outport,
            startup_delay_s=cfg.startup_delay_s,
            max_run_s=cfg.max_run_s,
            stop=stop,
            current=current,
            vehicle=cfg.vehicle,
            frame=cfg.frame,
            model=cfg.model,
            world=cfg.world,
            location=cfg.location
        )

    _finalize_proc(current.get("gz"))
    current["gz"] = None
    _finalize_proc(current.get("mavproxy"))
    current["mavproxy"] = None

    return 0


if __name__ == "__main__":
    raise SystemExit(main())