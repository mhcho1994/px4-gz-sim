import os
import subprocess
import time
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
import run_arducopter_gz_sitl as launcher
import yaml

# @dataclass
# class ScenarioConfig:
#     # sim instance parameters
#     firmware: str = "px4"
#     instance: int = 0
#     outport: int = 14550
#     startup_delay_s: float = 0.0
#     max_run_s: float = 60.0

#     # sim setup
#     vehicle: str = "gz_x500"
#     frame: str = "gazebo-iris"
#     model: str = "JSON"
#     world: str = "iris_runway"
#     location: str = "Purdue"
#     scenario_name: str = "turn3pts"


@dataclass
class ProcHandle:
    """Container holding a running process and its log file info."""
    name: str
    proc: subprocess.Popen
    log_path: Path
    log_file: launcher.TextIO  # keep handle so it doesn't get GC'ed early


def run_px4_simulation(run_id, px4_dir, max_runtime):
    print(f"\n{'='*40}")
    print(f"🚀 [RUN {run_id}] PX4 시뮬레이션 시작")
    print(f"{'='*40}")
    
    px4_gz_proc = None

    try:
        # 1) PX4 + Gazebo 실행 (통합 런치 방식)
        # 보통 PX4는 make 명령어로 SITL과 Gazebo를 한 번에 띄우는 것이 가장 안정적입니다.
        px4_cmd = [
            "make", 
            "px4_sitl", 
            "gz_x500"  # x500 기체 모델 (필요에 따라 gz_iris 등으로 변경)
        ]
        
        print("1. PX4 SITL 및 Gazebo 런치 중...")
        # 주의: 이 스크립트는 PX4-Autopilot 소스코드 최상위 폴더에서 실행하거나,
        # cwd 파라미터로 해당 경로를 지정해주어야 합니다.
        px4_gz_proc = subprocess.Popen(
            px4_cmd, 
            cwd=px4_dir,               # PX4-Autopilot 폴더 경로 지정
            stdout=subprocess.DEVNULL, # 터미널 출력이 너무 길면 무시
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid       # [핵심] 독립적인 프로세스 그룹으로 묶음
        )
        
        # 2) PX4 부팅 및 Gazebo 로딩 대기
        # PX4가 초기화되고 uORB 메시지가 안정화될 시간을 충분히 줍니다.
        print("2. PX4 및 Gazebo 초기화 대기 중 (15초)...")
        time.sleep(15)

        # ---------------------------------------------------------
        # 💡 여기에 PX4용 웨이포인트 스크립트(MAVSDK 등)를 실행하는 코드를 넣습니다.
        # print("3. PX4 미션 스크립트 실행...")
        # mission_cmd = ["python3", "px4_waypoint_mission.py"]
        # mission_proc = subprocess.Popen(mission_cmd)
        # ---------------------------------------------------------

        # 3) 프로세스 모니터링
        print(f"3. 모니터링 시작 (최대 {max_runtime}초)...")
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_runtime:
                print(f"\n⏳ 최대 실행 시간({max_runtime}초) 도달. 루틴을 강제 종료합니다.")
                break
            
            # 프로세스가 예기치 않게 죽었는지 확인
            if px4_gz_proc.poll() is not None:
                print("\n⚠️ PX4/Gazebo 프로세스가 예기치 않게 종료되었습니다.")
                break
                
            # if mission_proc and mission_proc.poll() is not None:
            #     print("\n✅ 비행 미션이 완료되었습니다.")
            #     break

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 사용자가 강제로 중단했습니다 (Ctrl+C).")
        raise 

    finally:
        # 4) 깔끔한 뒷정리
        print("\n🧹 정리 작업 시작 (PX4 데몬 및 Gazebo 킬)...")
        kill_process_group(px4_gz_proc, "PX4_Gazebo")
        print(f"✅ [RUN {run_id}] 종료 완료")

def main():
    ap = launcher.argparse.ArgumentParser()
    ap.add_argument("--outport", type=int, default=14550)
    
    ap.add_argument("--run-root", type=Path, default=Path("../scenario/data/"), help="Root folder containing run_xxx/scenario.yaml and run_xxx/px4_logs")
    # ap.add_argument("--px4-dir", type=Path, required=True, help="PX4-Autopilot repo path")
    ap.add_argument("--force", action="store_true", help="Re-run even if run_xxx/px4_logs exists")
    ap.add_argument("--world", default="default", help="Gazebo world name (e.g., default, windy, baylands, ...)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--outdir", type=Path, default=Path("./px4_runs"))
    ap.add_argument("--startup-delay-s", type=float, default=5.0)
    ap.add_argument("--max-run-s", type=float, default=60.0)
    
    args = ap.parse_args()

    # Access to the resolved run root path
    data_root = args.run_root.resolve()

    stop = {"flag": False}
    current: dict[str, launcher.Optional["ProcHandle"]] = {"gz": None, "mavproxy": None, "sitl": None, "commander": None}  

    # Mark stop flag and terminate running processes on SIGINT/SIGTERM
    def _sig(_signum, _frame):
        stop["flag"] = True
        # immediately kill running processes (so ArduPilot doesn't linger)
        launcher._finalize_proc(current.get("gz"))
        launcher._finalize_proc(current.get("mavproxy"))
        launcher._finalize_proc(current.get("sitl"))
        # launcher._finalize_proc(current.get("commander"))
        current["gz"] = None
        current["mavproxy"] = None
        current["sitl"] = None
        # current["commander"] = None

    # Terminate on Ctrl+C or SIGTERM
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Iterations for run directories
    run_dirs = launcher._iter_run_dirs(data_root)
    if not run_dirs:
        print(f"No run_xxx directories found under: {data_root}")
        return 2
    
    overall_rc = 0

    for run_dir in run_dirs:
        if stop["flag"]:
            print("Interrupted. Exiting.")
            return 130

        if launcher._should_skip_run_dir(run_dir, force=args.force):
            print(f"[SKIP] {run_dir} (px4_logs exists)")
            continue

        # Load scenario.yaml
        try:
            cfg = launcher._load_scenario_yaml(run_dir)
        except Exception as e:
            print(f"[ERROR] {run_dir}: failed to load scenario.yaml: {e}")
            overall_rc = 2
            continue

        # Create output directory for this run
        logs_root = run_dir / "px4_logs"
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
        rc = launcher.run_once(
            firmware=cfg.firmware,
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
            location=cfg.location,
            px4_dir=Path(cfg.px4_dir) if cfg.px4_dir else None
        )

    launcher._finalize_proc(current.get("gz"))
    current["gz"] = None
    launcher._finalize_proc(current.get("mavproxy"))
    current["mavproxy"] = None

if __name__ == "__main__":
    raise SystemExit(main())
