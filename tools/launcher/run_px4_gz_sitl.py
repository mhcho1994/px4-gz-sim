#!/usr/bin/env python3
"""
Multi-run launcher for PX4 SITL + Gazebo GZ.

What this script does
---------------------
- Scans run_xxx directories under a run-root
- For each scenario:
    1) Loads scenario.yaml
    2) Starts Gazebo separately
    3) Starts PX4 SITL using an already-built PX4 binary
    4) Starts a MAVSDK commander thread
    5) Monitors timeout / failures / completion
    6) Cleans up process trees

Design notes
------------
- QGC is expected to be run separately by the user.
- Gazebo is launched in standalone mode.
- PX4 binary is assumed to be already built.
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
from typing import Any, Optional, TextIO, Tuple
import yaml
import math
import shutil
import re

_THIS_FILE = Path(__file__).resolve()
_TOOLS_DIR = _THIS_FILE.parents[1]
_COMMANDER_DIR = _TOOLS_DIR / "commander"
_QGC_DIR = _TOOLS_DIR / "QGC"

if str(_COMMANDER_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMANDER_DIR))

from pymavlink_px4_commander import PX4MissionRunner


@dataclass
class PX4ScenarioConfig:
    # sim instance parameters
    px4_dir: str = "${FLIGHTSTACK_SIM_ROOT}/ap/px4"
    instance: int = 0
    mavlink_url: str = "udp://127.0.0.1:14650"
    startup_delay_s: float = 0.0
    max_run_s: float = 60.0

    # sim setup
    vehicle: int = 4001
    frame: str = "gz_x500"
    world: str = "default"
    location: str = "Purdue"
    scenario_name: str = "unnamed"
    qgc_outport: int = 14650


@dataclass
class ProcHandle:
    """Container holding a running process and its log file info."""
    name: str
    proc: subprocess.Popen
    log_path: Path
    log_file: TextIO


def _popen(
    name: str,
    cmd: list[str],
    cwd: Optional[Path],
    log_path: Path,
    env: Optional[dict[str, str]] = None,
    verbose: bool = False,
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

    # Print environmental variables
    if verbose and env is not None:
        print(f"[ENV]    {name}:")
        for k in sorted(env.keys()):
            if k.startswith("PX4") or k.startswith("GZ") or k in ("PATH", "HOME"):
                print(f"         {k}={env[k]}")

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
        env=env,
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
    """
    if os.name == "nt":
        return

    patterns = [
        r"gz sim",
        r"gz sim server",
        r"gz sim gui",
    ]
    _pkill_patterns(patterns, "INT")


def _cleanup_sitl_processes() -> None:
    """Best-effort cleanup for common PX4 SITL related processes."""
    if os.name == "nt":
        return

    patterns = [
        r"px4_sitl_default/bin/px4",
    ]

    _pkill_patterns(patterns, "INT")


def _cleanup_qgc_processes() -> None:
    """
    Best-effort cleanup for QGroundControl related processes that may survive
    after killing the parent shell/process group.

    Why needed:
    - Killing the bash PID or even its process group does not always terminate
      the actual QGroundControl process.
    """
    if os.name == "nt":
        return

    patterns = [
        r"QGroundControl",
        r"bin/QGroundControl",
        r"QGroundControl-x86_64.AppImage",
    ]

    _pkill_patterns(patterns, "INT")


def _kill_tree(ph: Optional[ProcHandle], grace_s: float = 5.0) -> None:
    """
    Terminate a process and its descendants.

    For interactive tools such as sim_vehicle.py, we try SIGINT first
    because it is closer to Ctrl-C and may allow cleaner shutdown.
    Then we escalate to SIGTERM and finally SIGKILL.
    """
    if ph is None or ph.proc.poll() is not None:
        return

    if os.name != "nt":
        try:
            pgid = os.getpgid(ph.proc.pid)
        except Exception:
            pgid = None
    else:
        pgid = None

    # 1) Try SIGINT first
    try:
        if os.name != "nt" and pgid is not None:
            os.killpg(pgid, signal.SIGINT)
        else:
            ph.proc.terminate()
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < min(grace_s, 2.0):
        if ph.proc.poll() is not None:
            break
        time.sleep(0.1)

    # 2) Escalate to SIGTERM
    if ph.proc.poll() is None:
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
            break
        time.sleep(0.1)

    # 3) SIGKILL
    if ph.proc.poll() is None:
        try:
            if os.name != "nt" and pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                ph.proc.kill()
        except Exception:
            pass

    # Final fallback cleanup for leftovers
    if "gz" in ph.name:
        _cleanup_gz_processes()

    if "sitl" in ph.name:
        _cleanup_sitl_processes()

    if "qgc" in ph.name:
        _cleanup_qgc_processes()


def _finalize_proc(ph: Optional[ProcHandle], grace_s: float = 5.0) -> None:
    """Best-effort: ensure process is dead and log file is closed."""

    if ph is None:
        return

    try:
        _kill_tree(ph)
    except Exception:
        pass

    try:
        ph.proc.wait(timeout=grace_s)
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


def ensure_px4_built(px4_dir: Path) -> Path:
    """
    Ensure PX4 SITL binary exists. If not, build it.

    Returns
    -------
    Path to PX4 SITL binary
    """

    binary = px4_dir / "build" / "px4_sitl_default" / "bin" / "px4"

    # already built
    if binary.exists():
        print(f"[INFO] PX4 already built: {binary}")
        return binary

    print("[INFO] PX4 not built. Building SITL...")

    # PX4 build (default SITL target)
    subprocess.run(
        ["make", "px4_sitl"],
        cwd=px4_dir,
        check=True,
    )

    if not binary.exists():
        raise RuntimeError("Build finished but PX4 binary not found.")

    print(f"[INFO] Build complete: {binary}")
    return binary


def find_px4_binary(px4_dir: Path) -> Path:
    """
    Prefer already-built PX4 SITL binary.

    Common locations:
      build/px4_sitl_default/bin/px4
      build/px4_sitl_default/bin/px4-rc (script, not binary)
    """
    candidates = [
        px4_dir / "build" / "px4_sitl_default" / "bin" / "px4",
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    raise FileNotFoundError(
        f"PX4 binary not found. Expected one of: {[str(c) for c in candidates]}"
    )


def find_px4_rc_script(px4_dir: Path) -> Path:
    """
    PX4 SITL startup script commonly used by the binary.
    rcS: PX4 nsh-style initialization script
    e.g.) param set MAV_SYS_ID $((px4_instance+1))
          param set UXRCE_DDS_KEY $((px4_instance+1))
    """
    candidates = [
        px4_dir / "ROMFS" / "px4fmu_common" / "init.d-posix" / "rcS",
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    raise FileNotFoundError(
        f"PX4 rcS script not found. Expected one of: {[str(c) for c in candidates]}"
    )


def find_px4_etc(px4_dir: Path) -> Path:
    """
    Locate the PX4 runtime `etc` directory for SITL.

    This function searches common PX4 build output locations to find
    the `etc` directory required to run the PX4 binary.

    Common locations:
      build/px4_sitl_default/etc
      build/px4_sitl_default/etc/
    """
    candidates = [
        px4_dir / "build" / "px4_sitl_default" / "etc",
        # Rare: nested ROMFS (older / edge cases)
        px4_dir / "build/px4_sitl_default" / "ROMFS" / "px4fmu_common",
    ]

    for path  in candidates:
        if path.exists() and path.is_dir():
            return path.resolve()

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"PX4 etc directory not found. Expected one of: \n{searched}\n"
    )


def run_sitl_cmd(px4_bin: Path, px4_etc: Path, logs_dir: Path) -> list[str]:
    cmd = [
        str(px4_bin)
    ]  
    return cmd


def read_location_from_txt(txt_path: Path, name: str) -> Tuple[float, float, float, float]:
    """
    Parse a line like:
      Purdue=40.41176161953683,-86.93352081596879,0,0

    Returns:
      lat, lon, alt, heading_deg
    """
    if not txt_path.is_file():
        raise FileNotFoundError(f"locations file not found: {txt_path}")

    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        key, sep, value = line.partition("=")
        if sep and key.strip() == name:
            parts = [p.strip() for p in value.split(",")]
            if len(parts) < 4:
                raise ValueError(f"Invalid location line for {name}: {line}")

            lat = float(parts[0])
            lon = float(parts[1])
            alt = float(parts[2])
            heading_deg = float(parts[3])
            return lat, lon, alt, heading_deg

    raise KeyError(f"Location '{name}' not found in {txt_path}")


def run_qgc_cmd(out_port: int) -> list[str]:
    """
    Launch QGroundControl (GCS).
    QGC automatically listens on UDP port 14550.
    """
    qgc_bin = _QGC_DIR / "QGroundControl-x86_64.AppImage"
    return [str(qgc_bin)]


def run_gz_cmd(world_sdf: str, verbose: str = "-v4") -> list[str]:
    """
    Run the Gazebo Sim command.

    Example:
      gz sim -v4 -r iris_runway.sdf
    """
    return ["gz", "sim", verbose, "-r", world_sdf]


def _finalize_runner(runner, timeout: float = 2.0) -> None:
    """
    Best-effort shutdown for a mission runner thread.
    """
    if runner is None:
        return

    try:
        runner.request_stop()
    except Exception:
        pass


def run_once(
    px4_dir: Path,
    instance: int,
    scenario_path: Path,
    logs_dir: Path,
    qgc_outport: int,
    mavlink_url: str,
    startup_delay_s: float,
    max_run_s: float,
    stop: dict,
    current: dict,
    vehicle: str,
    frame: str,
    world: str,
    location: str,
) -> int:
    """
    Execute a single PX4 SITL + Gazebo simulation episode with mission execution.

    This function orchestrates the full lifecycle of one simulation run, including:
      1) Launching auxiliary processes (QGroundControl)
      2) Configuring and starting PX4 SITL in standalone mode
      3) Injecting environment variables for simulation (model, world, home position)
      4) Running a mission via PX4MissionRunner
      5) Monitoring execution until completion, timeout, or user interruption
      6) Gracefully terminating all spawned processes

    Parameters
    ----------
    px4_dir : Path
        Root directory of the PX4-Autopilot repository.

    instance : int
        PX4 instance ID (used for multi-run separation).

    scenario_path : Path
        Path to scenario.yaml describing the mission to execute.

    logs_dir : Path
        Directory where SITL, Gazebo, and QGC logs are stored.

    qgc_outport : int
        UDP port used by QGroundControl for MAVLink communication.

    mavlink_url : str
        MAVLink endpoint (e.g., udp://127.0.0.1:14540).

    startup_delay_s : float
        Delay (seconds) between launching components to ensure proper initialization.

    max_run_s : float
        Maximum allowed runtime for the mission before timeout.

    stop : dict
        Shared stop flag (e.g., {"flag": True}) used for external interruption.

    current : dict
        Dictionary tracking running subprocesses (e.g., {"sitl": Popen, "gz": Popen}).

    vehicle : str
        PX4 autostart ID (e.g., "4001" for quadrotor).

    frame : str
        Simulator and model identifier in the form "<sim>_<model>".
        Example: "gz_x500" → simulator="gz", model="x500".

    world : str
        Gazebo world name (without .sdf extension).

    location : str
        Named location used to set home position (resolved via locations.txt).

    Returns
    -------
    int
        Return code indicating simulation outcome:
          0   : success (mission completed)
          1   : mission error
          124 : timeout exceeded
          130 : user interrupt

    Notes
    -----
    - PX4 is launched in standalone SITL mode using environment variables instead of
      the standard `make px4_sitl_<model>` workflow.
    - Home position is set via PX4_HOME_* variables using values from locations.txt.
    - Heading is converted from NED (ArduPilot-style) to ENU (Gazebo/PX4):
          yaw_enu = pi/2 - yaw_ned
    - Gazebo resources (models/worlds) are injected via GZ_SIM_RESOURCE_PATH.
    - Mission execution is handled asynchronously via PX4MissionRunner.

    Execution Flow
    --------------
    QGroundControl → PX4 SITL → MissionRunner → Monitor loop → Cleanup

    Cleanup ensures all spawned processes are terminated before returning.
    """

    # set logs dir
    sitl_log = logs_dir / "sitl.log"
    gz_log = logs_dir / "gazebo.log"
    qgc_log = logs_dir / "qgc.log"

    # find PX4 built binary and startup run commands
    px4_bin = find_px4_binary(px4_dir)
    rc_script = find_px4_rc_script(px4_dir)
    px4_etc = find_px4_etc(px4_dir)

    # Gazebo first
    # TODO: gazebo standalone mode
    if False:
        time.sleep(startup_delay_s)
        print(f"[INFO] Waiting {startup_delay_s:.1f}s for Gazebo to initialize...")

        # Helpful Gazebo resource path if worlds/models live inside PX4 repo
        gz_env = os.environ.copy()
        existing = gz_env.get("GZ_SIM_RESOURCE_PATH", "")
        paths = [p for p in existing.split(":") if p]
        
        px4_models = px4_dir / "Tools" / "simulation" / "gz" / "models"
        px4_worlds = px4_dir / "Tools" / "simulation" / "gz" / "worlds"
        
        if px4_models not in paths and px4_worlds not in paths:
            gz_env["GZ_SIM_RESOURCE_PATH"] = (
                f"{px4_models}:{px4_worlds}:{existing}" if existing else str(px4_models)
            )
        elif px4_models not in paths and px4_worlds in paths:
            gz_env["GZ_SIM_RESOURCE_PATH"] = (
                f"{px4_models}:{existing}" if existing else str(px4_models)
            )
        elif px4_models in paths and px4_worlds not in paths:
            gz_env["GZ_SIM_RESOURCE_PATH"] = (
                f"{px4_worlds}:{existing}" if existing else str(px4_models)
            )

        gz = _popen("gazebo", run_gz_cmd(f"{world}.sdf", "-v4"), cwd=logs_dir, log_path=gz_log, env=gz_env)
        current["gz"] = gz

    # QGroundControl second
    if current.get("QGC") is None:
        time.sleep(startup_delay_s)
        print(f"[INFO] Waiting {startup_delay_s:.1f}s for QGroundControl to initialize...")

        QGC = _popen("QGC", run_qgc_cmd(qgc_outport), cwd=logs_dir, log_path=qgc_log)
        current["QGC"] = QGC


    # PX4 standalone SITL third
    # TODO: px4 standalone mode with calling etc posix
    if current.get("sitl") is None:
        time.sleep(startup_delay_s)
        print(f"[INFO] Waiting {startup_delay_s:.1f}s for SITL to initialize...")

        lat, lon, alt, heading_deg = read_location_from_txt(
            Path(_THIS_FILE.parent / "locations.txt"), location
        )

        # pass environmental variables for SITL instance
        px4_env = os.environ.copy()
        px4_env["PX4_SYS_AUTOSTART"] = vehicle
        sim_engine, model = frame.split("_", 1)
        px4_env["PX4_SIMULATOR"] = sim_engine
        px4_env["PX4_GZ_MODEL"] = model
        px4_env["PX4_GZ_WORLD"] = world
        # px4_env["PX4_GZ_STANDALONE"] = "1"
        px4_env["PX4_SIM_SPEED_FACTOR"] = "1"

        # geodetic home
        px4_env["PX4_HOME_LAT"] = str(lat)
        px4_env["PX4_HOME_LON"] = str(lon)
        px4_env["PX4_HOME_ALT"] = str(alt)

        # optional: yaw only, spawn at local ENU origin -> conversion from NED heading
        yaw_ned = heading_deg/180*math.pi
        yaw_enu = math.pi/2 - yaw_ned
        px4_env["PX4_GZ_MODEL_POSE"] = f"0.0,0.0,0.0,0.0,0.0,{yaw_enu}"

        # add model and world sdf
        existing = px4_env.get("GZ_SIM_RESOURCE_PATH", "")
        paths = [p for p in existing.split(":") if p]
                
        px4_models = str(px4_dir / "Tools" / "simulation" / "gz" / "models")
        px4_worlds = str(px4_dir / "Tools" / "simulation" / "gz" / "worlds")

        new_paths = []
        if px4_models not in paths:
            new_paths.append(px4_models)
        if px4_worlds not in paths:
            new_paths.append(px4_worlds)
        new_paths.extend(paths)

        px4_env["GZ_SIM_RESOURCE_PATH"] = ":".join(new_paths)

        sitl = _popen("sitl", run_sitl_cmd(px4_bin, px4_etc, logs_dir), cwd=logs_dir, log_path=sitl_log, env=px4_env)
        current["sitl"] = sitl

    time.sleep(startup_delay_s)
    runner = PX4MissionRunner(scenario_path=scenario_path)
    runner.start()

    t0 = time.time()

    while True:
        if stop["flag"]:
            print("[STOP] user interrupt")
            _finalize_runner(runner)
            rc = 130
            break

        # 2) max runtime exceeded
        if time.time() - t0 > max_run_s:
            print(f"[TIMEOUT] exceeded {max_run_s:.1f}s")
            _finalize_runner(runner)
            rc = 124
            break

        # 3) commander finished
        status = runner.get_status()

        print(
            f"[RUNNER] state={status.state.name} "
            f"done={status.done} success={status.success} "
            f"msg={status.message}"
        )

        if runner.is_done():
            if status.success:
                print("[RUNNER] normally terminated")
                rc = 0
            else:
                print(f"[RUNNER] error: {status.error}")
                rc = 1
            _finalize_runner(runner)
            break

        time.sleep(1.0)

    _finalize_proc(current.get("sitl"))
    current["sitl"] = None
    _finalize_proc(current.get("gz"))
    current["gz"] = None

    print('[HIT] move to next iteration')

    return rc


def collect_px4_logs(px4_dir: Path, run_dir: Path):
    """
    Extract PX4 ULog file path from sitl.log and move it to run_dir.

    This method parses the PX4 SITL stdout/stderr log (sitl.log)
    to find the exact `.ulg` file generated during the run,
    ensuring deterministic and race-free log collection.

    Parameters
    ----------
    px4_dir : Path
        PX4 repository root (contains build/px4_sitl_default).

    run_dir : Path
        Run-specific log directory containing px4_logs folder and corresponding sitl.log
        (for example, `data/run_000`).
    """
    dst_root = run_dir / "px4_logs"
    dst_root.mkdir(parents=True, exist_ok=True)

    sitl_log = dst_root / "sitl.log"

    if not sitl_log.exists():
        print(f"[WARN] sitl.log not found: {sitl_log}")
        return
    
    text = sitl_log.read_text(errors="ignore")
    matches = re.findall(r"\./log/[^\s]+\.ulg", text)

    if not matches:
        print(f"[WARN] No .ulg path found in {sitl_log}")
        return

    src_root = px4_dir / "build" / "px4_sitl_default" / "rootfs"

    if not src_root.exists():
        print("[WARN] No PX4 log directory found")
        return

    rel_path = Path(matches[-1].replace("./", ""))
    src_ulg_file = src_root / rel_path

    if not src_ulg_file:
        print("[WARN] No .ulg files found")
        return

    dst_ulg_file = dst_root / rel_path.name
    shutil.move(str(src_ulg_file), dst_ulg_file)

    print(f"[INFO] PX4 log moved: {dst_ulg_file}")


def _load_scenario_yaml_px4(run_dir: Path) -> PX4ScenarioConfig:
    """
    Load scenario.yaml from run_dir and extract PX4-specific configuration.

    Supported keys in scenario.yaml:
      px4_dir: ${FLIGHTSTACK_SIM_ROOT}/ap/px4
      instance: 0
      vehicle: 4001
      frame: gz_x500
      world: default
      location: Purdue
      qgc_outport: 14650
      connect_url (mavlink): udp:127.0.0.1:14650
    """
    scenario_path = run_dir / "scenario.yaml"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario.yaml: {scenario_path}")

    data: dict[str, Any] = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}

    # Accept both top-level keys and nested (e.g., {"sim": {...}})
    # You can extend this mapping if your scenario.yaml has a different schema.
    sim = data.get("autopilots", {}).get("px4", {}).get("sim", {})
    scenario = data.get("common", {}).get("scenario", {})
    mavlink = data.get("autopilots",{}).get("px4",{}).get("mavlink",{})

    cfg = PX4ScenarioConfig(
        px4_dir=Path(os.path.expandvars(str(sim.get("px4_dir", PX4ScenarioConfig.px4_dir)))).resolve(),
        instance=int(sim.get("instance", PX4ScenarioConfig.instance)),
        vehicle=str(sim.get("vehicle", PX4ScenarioConfig.vehicle)),
        frame=str(sim.get("frame", PX4ScenarioConfig.frame)),
        world=str(sim.get("world", PX4ScenarioConfig.world)),
        location=str(sim.get("location", PX4ScenarioConfig.location)),
        scenario_name=str(scenario.get("name", PX4ScenarioConfig.scenario_name)),
        qgc_outport=str(sim.get("mavproxy_outport", PX4ScenarioConfig.qgc_outport)),
        mavlink_url=str(mavlink.get("connect_url", PX4ScenarioConfig.mavlink_url)),
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
    Skip if 'px4_logs' exists.
    """
    logs_dir = run_dir / "px4_logs"
    if force:
        return False
    ulg_files = list(logs_dir.rglob("*.ulg"))
    return logs_dir.exists() and (len(ulg_files) > 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("./data/sitl_logs"), help="Root folder containing run_xxx/scenario.yaml")
    ap.add_argument("--force", action="store_true", help="Re-run even if px4_logs already exist")
    ap.add_argument("--startup-delay-s", type=float, default=5.0)
    ap.add_argument("--max-run-s", type=float, default=60.0)
    args = ap.parse_args()

    # Access to the resolved run root path
    data_root = args.run_root.resolve()

    stop = {"flag": False}
    current: dict[str, Optional[ProcHandle]] = {
        "gz": None,
        "QGC": None,
        "sitl": None,
    }

    def _sig(_signum, _frame):
        stop["flag"] = True
        _finalize_proc(current.get("gz"))
        _finalize_proc(current.get("QGC"))
        _finalize_proc(current.get("sitl"))
        current["gz"] = None
        current["QGC"] = None
        current["sitl"] = None

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
            print(f"[SKIP] {run_dir} (px4_logs exists and contains .ulg)")
            continue

        # Load scenario.yaml
        try:
            cfg = _load_scenario_yaml_px4(run_dir)
        except Exception as e:
            print(f"[ERROR] {run_dir}: failed to load scenario.yaml: {e}")
            overall_rc = 2
            continue

        # Create output directory for this run
        logs_root = run_dir / "px4_logs"
        logs_root.mkdir(parents=True, exist_ok=True)

        # Get scenario path
        scenario_path = run_dir / "scenario.yaml"

        # Set other configurations
        cfg.startup_delay_s = args.startup_delay_s
        cfg.max_run_s = args.max_run_s

        # Confirm that PX4 is built
        ensure_px4_built(px4_dir=cfg.px4_dir)

        # Print configuration info
        print(f"\n---- {run_dir.name}: RUN {cfg.scenario_name} Scenario ----")
        print(f"  px4_dir={cfg.px4_dir} vehicle={cfg.vehicle}")
        print(f"  world={cfg.world} location={cfg.location}")
        print(f"  instance={cfg.instance} qgc_outport={cfg.qgc_outport}")
        print(f"  startup_delay={cfg.startup_delay_s} max_run_s={cfg.max_run_s}")
        print(f"  logs_root={logs_root} scenario_path={scenario_path} mavlink_url={cfg.mavlink_url}")

        # Execute the scenario
        rc = run_once(
            px4_dir=cfg.px4_dir,
            instance=cfg.instance,
            scenario_path=scenario_path,
            logs_dir=logs_root,
            qgc_outport=cfg.qgc_outport,
            mavlink_url=cfg.mavlink_url,
            startup_delay_s=cfg.startup_delay_s,
            max_run_s=cfg.max_run_s,
            stop=stop,
            current=current,
            vehicle=cfg.vehicle,
            frame=cfg.frame,
            world=cfg.world,
            location=cfg.location
        )

        # Move sitl log from px4 default directory to log folder
        collect_px4_logs(cfg.px4_dir, run_dir)

        overall_rc = max(overall_rc, 1 if rc != 0 else 0)

    # Kill QGC and end SITL
    _finalize_proc(current.get("QGC"))
    current["QGC"] = None

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())