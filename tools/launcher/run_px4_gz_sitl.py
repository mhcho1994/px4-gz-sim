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
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, TextIO

import yaml

_THIS_FILE = Path(__file__).resolve()
_TOOLS_DIR = _THIS_FILE.parents[1]
_COMMANDER_DIR = _TOOLS_DIR / "commander"

if str(_COMMANDER_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMANDER_DIR))

from mavsdk_px4_commander import PX4MissionRunner  # noqa: E402


@dataclass
class PX4ScenarioConfig:
    px4_dir: Path = Path("/path/to/PX4-Autopilot")
    world: str = "default"
    autostart: int = 4001
    model: str = "gz_x500"
    out_udp_port: int = 14540
    startup_delay_s: float = 5.0
    max_run_s: float = 180.0
    grace_s: float = 10.0
    scenario_name: str = "unnamed"


@dataclass
class ProcHandle:
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
    print(f"[LAUNCH] {name}: {' '.join(cmd)}")
    print(f"[CWD]    {name}: {cwd if cwd else os.getcwd()}")
    print(f"[LOG]    {name}: {log_path}")

    if verbose and env is not None:
        print(f"[ENV]    {name}:")
        for k in sorted(env.keys()):
            if k.startswith("PX4") or k.startswith("GZ") or k in ("PATH", "HOME"):
                print(f"         {k}={env[k]}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", buffering=1, encoding="utf-8")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        preexec_fn=os.setsid if os.name != "nt" else None,
    )

    return ProcHandle(name=name, proc=proc, log_path=log_path, log_file=f)


def _run_quiet(cmd: str) -> None:
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
    if os.name == "nt":
        return

    for pattern in patterns:
        _run_quiet(f"pkill -{sig} -f {shlex.quote(pattern)}")


def _cleanup_gz_processes() -> None:
    if os.name == "nt":
        return

    patterns = [
        r"gz sim",
        r"gz sim server",
        r"gz sim gui",
        r"ign gazebo",
    ]
    _pkill_patterns(patterns, "INT")


def _cleanup_px4_processes() -> None:
    if os.name == "nt":
        return

    patterns = [
        r"/bin/px4",
        r"px4-rc",
        r"gz_x500",
        r"PX4-Autopilot/build/.*/bin/px4",
    ]
    _pkill_patterns(patterns, "INT")


def _kill_tree(ph: Optional[ProcHandle], grace_s: float = 5.0) -> None:
    if ph is None or ph.proc.poll() is not None:
        return

    if os.name != "nt":
        try:
            pgid = os.getpgid(ph.proc.pid)
        except Exception:
            pgid = None
    else:
        pgid = None

    # 1) SIGINT
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

    # 2) SIGTERM
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

    if "gz" in ph.name or "gazebo" in ph.name:
        _cleanup_gz_processes()
    if "px4" in ph.name:
        _cleanup_px4_processes()


def _finalize_proc(ph: Optional[ProcHandle], grace_s: float = 5.0) -> None:
    if ph is None:
        return

    try:
        _kill_tree(ph, grace_s=grace_s)
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


def _finalize_runner(runner, timeout: float = 2.0) -> None:
    if runner is None:
        return

    try:
        runner.request_stop()
    except Exception:
        pass

    try:
        runner.join(timeout=timeout)
    except Exception:
        pass


def build_gz_cmd(world_sdf: str, verbose: str = "-v4") -> list[str]:
    return ["gz", "sim", verbose, "-r", world_sdf]


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


def build_px4_cmd(px4_bin: Path, rc_script: Path) -> list[str]:
    return [str(px4_bin), str(rc_script)]


def _load_scenario_yaml_px4(run_dir: Path) -> PX4ScenarioConfig:
    scenario_path = run_dir / "scenario.yaml"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario.yaml: {scenario_path}")

    data: dict[str, Any] = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}

    sim = data.get("autopilots", {}).get("px4", {}).get("sim", {})
    scenario = data.get("common", {}).get("scenario", {})

    cfg = PX4ScenarioConfig(
        px4_dir=Path(str(sim.get("px4_dir", PX4ScenarioConfig.px4_dir))),
        world=str(sim.get("world", PX4ScenarioConfig.world)),
        autostart=int(sim.get("autostart", PX4ScenarioConfig.autostart)),
        model=str(sim.get("model", PX4ScenarioConfig.model)),
        out_udp_port=int(sim.get("out_udp_port", PX4ScenarioConfig.out_udp_port)),
        startup_delay_s=float(sim.get("startup_delay_s", PX4ScenarioConfig.startup_delay_s)),
        max_run_s=float(sim.get("max_run_s", PX4ScenarioConfig.max_run_s)),
        grace_s=float(sim.get("grace_s", PX4ScenarioConfig.grace_s)),
        scenario_name=str(scenario.get("name", PX4ScenarioConfig.scenario_name)),
    )
    return cfg


def _iter_run_dirs(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    return sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("run_")])


def _should_skip_run_dir(run_dir: Path, force: bool) -> bool:
    logs_dir = run_dir / "px4_logs"
    if force:
        return False

    ulg_files = list(logs_dir.rglob("*.ulg"))
    return logs_dir.exists() and (len(ulg_files) > 0)


def run_once(
    scenario_path: Path,
    logs_dir: Path,
    stop: dict,
    current: dict,
    cfg: PX4ScenarioConfig,
    verbose: bool = False,
) -> int:
    gz_log = logs_dir / "gazebo.log"
    px4_log = logs_dir / "px4.log"

    px4_dir = cfg.px4_dir.resolve()
    px4_bin = find_px4_binary(px4_dir)
    rc_script = find_px4_rc_script(px4_dir)

    # Gazebo first
    if current.get("gz") is None:
        print(f"[INFO] Starting Gazebo world={cfg.world}")
        gz_env = os.environ.copy()

        # Helpful Gazebo resource path if worlds/models live inside PX4 repo
        px4_models = px4_dir / "Tools" / "simulation" / "gazebo-classic" / "sitl_gazebo-classic"
        existing = gz_env.get("GZ_SIM_RESOURCE_PATH", "")
        if px4_models.exists():
            gz_env["GZ_SIM_RESOURCE_PATH"] = (
                f"{px4_models}:{existing}" if existing else str(px4_models)
            )

        current["gz"] = _popen(
            "gazebo",
            build_gz_cmd(f"{cfg.world}.sdf"),
            cwd=logs_dir,
            log_path=gz_log,
            env=gz_env,
            verbose=verbose,
        )

    time.sleep(cfg.startup_delay_s)

    # PX4 standalone SITL
    if current.get("px4") is None:
        print(f"[INFO] Starting PX4 SITL model={cfg.model} autostart={cfg.autostart}")

        px4_env = os.environ.copy()
        px4_env["PX4_SYS_AUTOSTART"] = str(cfg.autostart)
        px4_env["PX4_SIM_MODEL"] = cfg.model
        px4_env["PX4_GZ_WORLD"] = cfg.world
        px4_env["PX4_GZ_STANDALONE"] = "1"
        px4_env["PX4_SIM_SPEED_FACTOR"] = "1"
        px4_env["PX4_HOME_LAT"] = "40.41176161953683"
        px4_env["PX4_HOME_LON"] = "-86.93352081596879"
        px4_env["PX4_HOME_ALT"] = "0.0"

        # Common PX4 working dir
        px4_run_dir = logs_dir / "px4_run"
        px4_run_dir.mkdir(parents=True, exist_ok=True)

        current["px4"] = _popen(
            "px4",
            build_px4_cmd(px4_bin, rc_script),
            cwd=px4_run_dir,
            log_path=px4_log,
            env=px4_env,
            verbose=verbose,
        )

    time.sleep(cfg.startup_delay_s)

    runner = PX4MissionRunner(scenario_path=scenario_path)
    runner.start()

    t0 = time.time()
    rc = 0

    while True:
        if stop["flag"]:
            print("[STOP] user interrupt")
            _finalize_runner(runner)
            rc = 130
            break

        if time.time() - t0 > cfg.max_run_s:
            print(f"[TIMEOUT] exceeded {cfg.max_run_s:.1f}s")
            _finalize_runner(runner)
            rc = 124
            break

        if current.get("gz") and current["gz"].proc.poll() is not None:
            print("[EXIT] gazebo terminated unexpectedly")
            _finalize_runner(runner)
            rc = 1
            break

        if current.get("px4") and current["px4"].proc.poll() is not None:
            print("[EXIT] px4 terminated unexpectedly")
            _finalize_runner(runner)
            rc = 1
            break

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

    _finalize_proc(current.get("px4"), grace_s=cfg.grace_s)
    current["px4"] = None

    _finalize_proc(current.get("gz"), grace_s=cfg.grace_s)
    current["gz"] = None

    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-root",
        type=Path,
        default=Path("./data"),
        help="Root folder containing run_xxx/scenario.yaml",
    )
    ap.add_argument("--force", action="store_true", help="Re-run even if px4_logs already exist")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data_root = args.run_root.resolve()

    stop = {"flag": False}
    current: dict[str, Optional[ProcHandle]] = {
        "gz": None,
        "px4": None,
    }

    def _sig(_signum, _frame):
        stop["flag"] = True
        _finalize_proc(current.get("px4"))
        _finalize_proc(current.get("gz"))
        current["px4"] = None
        current["gz"] = None

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

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
            print(f"[SKIP] {run_dir} (px4_logs contains .ulg)")
            continue

        try:
            cfg = _load_scenario_yaml_px4(run_dir)
        except Exception as e:
            print(f"[ERROR] {run_dir}: failed to load scenario.yaml: {e}")
            overall_rc = 2
            continue

        logs_root = run_dir / "px4_logs"
        logs_root.mkdir(parents=True, exist_ok=True)
        scenario_path = run_dir / "scenario.yaml"

        print(f"\n---- {run_dir.name}: RUN {cfg.scenario_name} Scenario ----")
        print(f"  px4_dir={cfg.px4_dir}")
        print(f"  world={cfg.world} model={cfg.model} autostart={cfg.autostart}")
        print(f"  out_udp_port={cfg.out_udp_port}")
        print(f"  startup_delay={cfg.startup_delay_s} max_run_s={cfg.max_run_s} grace_s={cfg.grace_s}")
        print(f"  logs_root={logs_root} scenario_path={scenario_path}")

        rc = run_once(
            scenario_path=scenario_path,
            logs_dir=logs_root,
            stop=stop,
            current=current,
            cfg=cfg,
            verbose=args.verbose,
        )

        overall_rc = max(overall_rc, 1 if rc != 0 else 0)

    _finalize_proc(current.get("px4"))
    current["px4"] = None
    _finalize_proc(current.get("gz"))
    current["gz"] = None

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())