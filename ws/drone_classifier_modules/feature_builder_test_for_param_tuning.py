import data_extractor
import kinematic_processor
import flight_segmenter
from pathlib import Path

# For visualization, import the legacy processor and the plotting script
import kinematic_processor_legacy
import generate_visualizations

test_files = [
    "./data/sitl_logs/run_000/px4_logs/raw/02_34_26.ulg",
    "./data/sitl_logs/run_003/px4_logs/raw/02_39_52.ulg",
    "./data/sitl_logs/run_001/ardu_logs/raw/logs/00000001.BIN",
    "./data/sitl_logs/run_004/ardu_logs/raw/logs/00000001.BIN",
    "./data/sitl_logs/run_248/ardu_logs/raw/logs/00000001.BIN",
    "./data/260424_flight_logs_1/run_000/ardu_logs/processed/trajectory.csv",
    "./data/260527_flight_logs_1/run_001/ardu_logs/processed/ardu_run2_trajectory.csv",
    "./data/260527_flight_logs_1/run_003/px4_logs/processed/px4_traj1_trajectory.csv"
]

output_dir = Path("results/feature_builder_test_viz")
output_dir.mkdir(parents=True, exist_ok=True)

for file_path_str in test_files:
    file_path = Path(file_path_str)
    print(f"\n{'='*70}\n[Processing] {file_path.name}\n{'='*70}")
    
    # 1. Raw 데이터 파싱 (1번 모듈)
    # 확장자에 따라 적절한 파서를 자동으로 호출합니다.
    ext = file_path.suffix.lower()
    if ext == '.ulg':
        raw_flight_data = data_extractor.parse_px4_ulog(str(file_path))
    elif ext == '.bin':
        raw_flight_data = data_extractor.parse_ardu_bin(str(file_path))
    elif ext == '.csv':
        raw_flight_data = data_extractor.parse_real_csv(str(file_path), measurement_type='mocap')
    else:
        print(f"[Skip] 지원하지 않는 확장자입니다: {ext}")
        continue

    if raw_flight_data is None:
        print(f"[Error] 데이터 파싱에 실패했습니다: {file_path.name}")
        continue

    # 2. 50Hz 보간 및 PCA 특징 추출 (2번 모듈 통합본)
    kinematic_features = kinematic_processor.compute_kinematics(raw_flight_data)
    if kinematic_features is None:
        print(f"[Error] 특징 추출(Kinematics)에 실패했습니다: {file_path.name}")
        continue

    # 3. HMM 기반 시퀀스 스무딩 및 구간 분할 (3번 모듈 통합본)
    segments, spans = flight_segmenter.extract_segments(kinematic_features)
    print(f"[Info] Segmentation complete. Found spans for: {[k for k, v in spans.items() if v]}")

    # 4. Visualize the segmentation results
    print("[Info] Generating visualization of flight segments...")
    t_full, feat_full = kinematic_processor_legacy.compute_kinematics(raw_flight_data)

    # 파일별로 이미지가 덮어씌워지지 않도록 상위 경로를 포함하여 고유한 파일명 지정
    unique_prefix = "_".join(file_path.parts[-4:-1])
    save_path = output_dir / f"{unique_prefix}_{file_path.stem}_segmentation.png"

    spans_for_plot = spans.copy()
    if 'turn_left' in spans_for_plot or 'turn_right' in spans_for_plot:
        spans_for_plot['turn'] = spans_for_plot.pop('turn_left', []) + spans_for_plot.pop('turn_right', [])

    if t_full is not None and feat_full is not None:
        generate_visualizations.plot_full_trajectory_with_spans(
            t=t_full,
            features=feat_full,
            spans=spans_for_plot,
            title=f"Flight Segmentation: {unique_prefix}_{file_path.stem}",
            save_path=str(save_path),
            line_color='tab:blue'
        )
        print(f"[Success] Visualization saved to: {save_path}")
    else:
        print("[Warning] Could not compute legacy features for visualization.")

    # 5. 후속 AI 특징 생성기(4번 모듈)로 전달
    turn_segments = segments.get('turn_left', []) + segments.get('turn_right', [])
    if turn_segments:
        print(f"[Info] Extracted {len(turn_segments)} turn segments for feature building.")
        for turn_seg in turn_segments:
            # 여기서 DWT(이산 웨이블릿 변환) 및 AI 학습용 1D-CNN 피처 빌딩 진행
            pass