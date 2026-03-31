from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pyulog import ULog
from pymavlink import mavutil
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')  

TARGET_HZ = 50.0  
DT = 1.0 / TARGET_HZ #20ms

def extract_turn_segments(t, features, dt=0.02, min_turn_duration=0.5, 
                          yaw_rate_threshold=0.2, straight_duration=1.0, heading_margin=0.2):
    spans = {'takeoff': None, 'landing': None, 'straight': [], 'turn': []}
    segments = []
    
    if features is None or len(features) < 2:
        return segments, spans
        
    N = len(features)
    alt = features[:, 0]
    vz_norm = np.abs(features[:, 2])
    az_norm = np.abs(features[:, 4])
    
    alt_95 = np.percentile(alt, 95)
    target_alt = alt_95 - 0.1
    
    # [Step 1] 이착륙 구간 탐색
    flight_start_idx = 0
    state = 0
    for i in range(N):
        if state == 0 and alt[i] >= target_alt: state = 1
        elif state == 1 and vz_norm[i] <= 0.1: state = 2
        elif state == 2 and az_norm[i] <= 0.1:
            flight_start_idx = i
            break
                
    flight_end_idx = N
    state = 0
    for i in range(N - 1, flight_start_idx, -1):
        if state == 0 and alt[i] >= target_alt: state = 1
        elif state == 1 and vz_norm[i] <= 0.1: state = 2
        elif state == 2 and az_norm[i] <= 0.1:
            flight_end_idx = i
            break

    if flight_start_idx > 0:
        spans['takeoff'] = (t[0], t[flight_start_idx])
    if flight_end_idx < N - 1:
        spans['landing'] = (t[flight_end_idx], t[-1])
                
    flight_t = t[flight_start_idx:flight_end_idx]
    flight_features = features[flight_start_idx:flight_end_idx]
    
    if len(flight_features) == 0:
        return segments, spans
        
    # [Step 2] 직선 구간 찾기
    is_straight = np.zeros(len(flight_features), dtype=bool)
    window_size = int(straight_duration / dt) 
    
    if len(flight_features) >= window_size:
        for i in range(len(flight_features) - window_size + 1):
            window_heading = flight_features[i : i + window_size, 1]
            window_yaw_rate_mag = np.abs(flight_features[i : i + window_size, 9])
            
            if np.max(window_yaw_rate_mag) <= yaw_rate_threshold:
                heading_diff = np.max(window_heading) - np.min(window_heading)
                if heading_diff <= heading_margin:
                    is_straight[i : i + window_size] = True

    edges = np.diff(is_straight.astype(int))
    st_starts = np.where(edges == 1)[0] + 1
    st_ends = np.where(edges == -1)[0] + 1
    if is_straight[0]: st_starts = np.insert(st_starts, 0, 0)
    if is_straight[-1]: st_ends = np.append(st_ends, len(is_straight))
    
    for start, end in zip(st_starts, st_ends):
        safe_end = min(end, len(flight_t) - 1)
        spans['straight'].append((flight_t[start], flight_t[safe_end]))

    straight_indices = np.where(is_straight)[0]
    if len(straight_indices) == 0:
        return segments, spans 
        
    first_straight_idx = straight_indices[0]

    # [Step 3] Turn 구간 추출
    is_turning = ~is_straight 
    is_turning[:first_straight_idx] = False
    
    edges = np.diff(is_turning.astype(int))
    turn_starts = np.where(edges == 1)[0] + 1
    turn_ends = np.where(edges == -1)[0] + 1
    
    if is_turning[0]: turn_starts = np.insert(turn_starts, 0, 0)
    if is_turning[-1]: turn_ends = np.append(turn_ends, len(is_turning))
        
    min_length = int(min_turn_duration / dt) 
    
    for start, end in zip(turn_starts, turn_ends):
        if (end - start) < min_length:
            continue
            
        safe_end = min(end, len(flight_t) - 1)
        spans['turn'].append((flight_t[start], flight_t[safe_end]))
        
        first_turn_t = flight_t[start:safe_end]
        first_turn_segment = flight_features[start:safe_end, :]
        segments.append((first_turn_t, first_turn_segment))
        break 
        
    return segments, spans

def extract_wavelet_features(t, x, y, z, vx, vy, vz, window_len=200, poly_order=3):
    t, unique_indices = np.unique(t, return_index=True)
    x, y, z = x[unique_indices], y[unique_indices], z[unique_indices]
    vx, vy, vz = vx[unique_indices], vy[unique_indices], vz[unique_indices]
    
    if len(t) < 2: return None

    t_start, t_end = t[0], t[-1]
    DT = 0.02 
    t_new = np.arange(t_start, t_end, DT) 
    
    x_new = interp1d(t, x, bounds_error=False, fill_value="extrapolate")(t_new)
    y_new = interp1d(t, y, bounds_error=False, fill_value="extrapolate")(t_new)
    z_new = interp1d(t, z, bounds_error=False, fill_value="extrapolate")(t_new)
    vx_new = interp1d(t, vx, bounds_error=False, fill_value="extrapolate")(t_new)
    vy_new = interp1d(t, vy, bounds_error=False, fill_value="extrapolate")(t_new)
    vz_new = interp1d(t, vz, bounds_error=False, fill_value="extrapolate")(t_new)
    
    def smooth(signal):
        wl = window_len
        if len(signal) < wl:
            wl = len(signal) if len(signal) % 2 != 0 else len(signal) - 1
            if wl <= poly_order: return signal 
        return savgol_filter(signal, window_length=wl, polyorder=poly_order)

    z_smooth, vx_smooth, vy_smooth, vz_smooth = smooth(z_new), smooth(vx_new), smooth(vy_new), smooth(vz_new)

    altitude = -z_smooth
    v_alt = -vz_smooth
    v_xy = np.vstack((vx_smooth, vy_smooth)).T
    speed_xy = np.linalg.norm(v_xy, axis=1)
    
    ax, ay, az = np.gradient(vx_smooth, DT), np.gradient(vy_smooth, DT), np.gradient(vz_smooth, DT)
    a_alt = -az
    a_xy = np.vstack((ax, ay)).T
    acc_norm_xy = np.linalg.norm(a_xy, axis=1)
    
    jx, jy, jz = np.gradient(smooth(ax), DT), np.gradient(smooth(ay), DT), np.gradient(smooth(az), DT)
    j_alt = -jz
    j_xy = np.vstack((jx, jy)).T
    jerk_norm_xy = np.linalg.norm(j_xy, axis=1)
    
    heading = np.unwrap(np.arctan2(vy_smooth, vx_smooth))
    raw_yaw_rate = np.gradient(heading, DT)
    yaw_rate = smooth(raw_yaw_rate)
    
    v_vec_3d = np.vstack((vx_smooth, vy_smooth, vz_smooth)).T
    a_vec_3d = np.vstack((ax, ay, az)).T
    speed_3d = np.linalg.norm(v_vec_3d, axis=1)
    cross_va = np.cross(v_vec_3d, a_vec_3d)
    cross_mag = np.linalg.norm(cross_va, axis=1)
    raw_curvature = cross_mag / (speed_3d**3 + 1e-6)
    curvature = smooth(raw_curvature)
    
    features = np.vstack((
        altitude, heading, v_alt, speed_xy, a_alt, acc_norm_xy, 
        j_alt, jerk_norm_xy, curvature, yaw_rate
    )).T
    
    return t_new, features

# =====================================================================
# 통합된 원본 X-Y 궤적 플롯 함수
# =====================================================================
def plot_combined_xy_trajectory(x_px4, y_px4, x_ardu, y_ardu, title="Combined X-Y Trajectory", save_path="xy_trajectory.png"):
    if (x_px4 is None or len(x_px4) == 0) and (x_ardu is None or len(x_ardu) == 0):
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # PX4 (Green)
    if x_px4 is not None and len(x_px4) > 0:
        ax.plot(y_px4, x_px4, color='tab:green', linewidth=1.5, alpha=0.8, label='PX4 Path')
        ax.plot(y_px4[0], x_px4[0], marker='o', color='darkgreen', markersize=6, label='PX4 Start')
        ax.plot(y_px4[-1], x_px4[-1], marker='X', color='darkgreen', markersize=6, label='PX4 End')

    # ArduPilot (Orange)
    if x_ardu is not None and len(x_ardu) > 0:
        ax.plot(y_ardu, x_ardu, color='tab:orange', linewidth=1.5, alpha=0.8, label='ArduPilot Path')
        ax.plot(y_ardu[0], x_ardu[0], marker='o', color='darkorange', markersize=6, label='ArduPilot Start')
        ax.plot(y_ardu[-1], x_ardu[-1], marker='X', color='darkorange', markersize=6, label='ArduPilot End')
    
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('East (Y) [m]', fontsize=12)
    ax.set_ylabel('North (X) [m]', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def process_px4_for_wavelet(ulog_path):
    try:
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset('vehicle_local_position').data
        t_loc = loc_data['timestamp'] / 1e6
        x, y, z = loc_data['x'], loc_data['y'], loc_data['z']
        vx, vy, vz = loc_data['vx'], loc_data['vy'], loc_data['vz']
            
        extracted = extract_wavelet_features(t_loc, x, y, z, vx, vy, vz)
        if extracted is None: return None, None, None, None, None, None
        
        t_full, feat_full = extracted
        turn_segments, spans = extract_turn_segments(t_full, feat_full)
        
        # X, Y 원본 데이터도 함께 반환
        return x, y, t_full, feat_full, turn_segments, spans
        
    except Exception as e:
        print(f"[PX4 Extract Error] {ulog_path}: {e}")
        return None, None, None, None, None, None

def process_ardu_for_wavelet(bin_path):
    try:
        mlog = mavutil.mavlink_connection(bin_path)
        t_loc, x, y, z, vx, vy, vz = [], [], [], [], [], [], []
        
        while True:
            msg = mlog.recv_match(type=['XKF1', 'NKF1'], blocking=False)
            if not msg: break
            t_loc.append(msg.TimeUS / 1e6)
            x.append(msg.PN); y.append(msg.PE); z.append(msg.PD)
            vx.append(msg.VN); vy.append(msg.VE); vz.append(msg.VD)
                
        if len(x) < 50: return None, None, None, None, None, None
            
        extracted = extract_wavelet_features(np.array(t_loc), np.array(x), np.array(y), np.array(z), np.array(vx), np.array(vy), np.array(vz))
        if extracted is None: return None, None, None, None, None, None
        
        t_full, feat_full = extracted
        turn_segments, spans = extract_turn_segments(t_full, feat_full)
        
        # X, Y 원본 데이터도 함께 반환
        return x, y, t_full, feat_full, turn_segments, spans
        
    except Exception as e:
        print(f"[ArduPilot Extract Error] {bin_path}: {e}")
        return None, None, None, None, None, None

# =====================================================================
# 플롯 1: 전체 궤적 + 색칠된 구역 (라인 색상 지정 추가)
# =====================================================================
def plot_full_trajectory_with_spans(t, features, spans, title, save_path, line_color):
    if features is None or len(features) == 0: return

    feature_names = ['Altitude (m)', 'Heading (rad)', 'Z-Axis Velocity (m/s)', 'XY-Plane Speed (m/s)', 'Z-Axis Acceleration (m/s²)', 'XY-Plane Accel Norm (m/s²)', 'Z-Axis Jerk (m/s³)', 'XY-Plane Jerk Norm (m/s³)', 'Curvature (1/m)', 'Yaw rate (rad/s)']
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    axes_flat = axes.flatten()
    
    span_colors = {'takeoff': 'orange', 'straight': 'mediumseagreen', 'turn': 'crimson', 'landing': 'mediumpurple'}

    for i in range(10):
        ax = axes_flat[i]
        # 라인 색상을 지정받은 색(Green or Orange)으로 그립니다.
        ax.plot(t, features[:, i], color=line_color, linewidth=1.5, zorder=2)
        ax.set_title(feature_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
        
        if i == 6: ax.set_ylim(np.percentile(features[:, i], 2), np.percentile(features[:, i], 98))
        elif i == 7: ax.set_ylim(-1, np.percentile(features[:, i], 98))
        elif i == 8: ax.set_ylim(-0.1, np.percentile(features[:, i], 98)) 

        if spans:
            if spans.get('takeoff'): ax.axvspan(*spans['takeoff'], color=span_colors['takeoff'], alpha=0.2, zorder=0)
            if spans.get('landing'): ax.axvspan(*spans['landing'], color=span_colors['landing'], alpha=0.2, zorder=0)
            for s, e in spans.get('straight', []): ax.axvspan(s, e, color=span_colors['straight'], alpha=0.2, zorder=0)
            for s, e in spans.get('turn', []): ax.axvspan(s, e, color=span_colors['turn'], alpha=0.4, zorder=0) 
    
    if spans:
        legend_patches = [
            mpatches.Patch(color=span_colors['takeoff'], alpha=0.2, label='Take-off'),
            mpatches.Patch(color=span_colors['straight'], alpha=0.2, label='Straight'),
            mpatches.Patch(color=span_colors['turn'], alpha=0.4, label='Turn (Target)'),
            mpatches.Patch(color=span_colors['landing'], alpha=0.2, label='Landing')
        ]
        fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.95, 0.98), ncol=4, fontsize=12)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# =====================================================================
# 플롯 2: 잘라낸 Turn 구간 전용 플롯 (라인 색상 지정 추가)
# =====================================================================
def plot_turn_segment_features(t, features, title, save_path, line_color):
    if features is None or len(features) == 0: return

    feature_names = ['Altitude (m)', 'Heading (rad)', 'Z-Axis Velocity (m/s)', 'XY-Plane Speed (m/s)', 'Z-Axis Acceleration (m/s²)', 'XY-Plane Accel Norm (m/s²)', 'Z-Axis Jerk (m/s³)', 'XY-Plane Jerk Norm (m/s³)', 'Curvature (1/m)', 'Yaw rate (rad/s)']
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    axes_flat = axes.flatten()
    
    for i in range(10):
        ax = axes_flat[i]
        # 라인 색상을 지정받은 색(Green or Orange)으로 그립니다.
        ax.plot(t, features[:, i], color=line_color, linewidth=2.0)
        ax.set_title(feature_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Absolute Time (s)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 6: ax.set_ylim(np.percentile(features[:, i], 2), np.percentile(features[:, i], 98))
        elif i == 7: ax.set_ylim(-1, np.percentile(features[:, i], 98))
        elif i == 8: ax.set_ylim(-0.1, np.percentile(features[:, i], 98)) 
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    BASE_DATA_DIR = Path("data") 
    print("[Info] Starting Combined Pipeline: XY / Full Highlights / Segments...\n")

    for i in range(100):
        run_folder = f"run_{i:03d}"
        run_dir = BASE_DATA_DIR / run_folder
        if not run_dir.exists(): continue

        px4_dir = run_dir / "px4_logs" / "raw"
        ardu_dir = run_dir / "ardu_logs" / "raw" / "logs"
        
        # 데이터 초기화
        x_px4, y_px4, t_px4, feat_px4, turn_px4, spans_px4 = (None,) * 6
        x_ardu, y_ardu, t_ardu, feat_ardu, turn_ardu, spans_ardu = (None,) * 6
        
        # 1. PX4 데이터 로드
        if px4_dir.exists():
            for file in os.listdir(px4_dir):
                if file.lower().endswith('.ulg'):
                    px4_result = process_px4_for_wavelet(str(px4_dir / file))
                    x_px4, y_px4, t_px4, feat_px4, turn_px4, spans_px4 = px4_result
                    break 

        # 2. ArduPilot 데이터 로드
        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith('.bin'):
                    ardu_result = process_ardu_for_wavelet(str(ardu_dir / file))
                    x_ardu, y_ardu, t_ardu, feat_ardu, turn_ardu, spans_ardu = ardu_result
                    break

        # =============================================================
        # 3. 플롯 1 - 통합 X-Y 궤적 그리기 (PX4 & ArduPilot 한 번에)
        # =============================================================
        if (x_px4 is not None) or (x_ardu is not None):
            plot_combined_xy_trajectory(
                x_px4, y_px4, x_ardu, y_ardu, 
                title=f"Combined X-Y Trajectory ({run_folder})", 
                save_path=str(run_dir / f"trajectory_xy_combined_{run_folder}.png")
            )

        # =============================================================
        # 4. 플롯 2 & 3 - PX4 궤적 및 세그먼트 플롯 (초록색)
        # =============================================================
        if feat_px4 is not None and len(feat_px4) > 0:
            print(f"[{run_folder}] Generating PX4 Trajectory plot (Visualizing Segments)...")
            plot_full_trajectory_with_spans(
                t=t_px4, features=feat_px4, spans=spans_px4, 
                title=f"Trajectory [Segment Check]: PX4 ({run_folder})", 
                save_path=str(run_dir / f"features_px4_seg_check_{run_folder}.png"),
                line_color='tab:green' # 초록색
            )
            
            if turn_px4 and len(turn_px4) > 0:
                t_turn, feat_turn = turn_px4[0]
                plot_turn_segment_features(
                    t=t_turn, features=feat_turn, 
                    title=f"Trajectory [Isolated Turn]: PX4 ({run_folder})", 
                    save_path=str(run_dir / f"features_px4_turn_seg_{run_folder}.png"),
                    line_color='tab:green' # 초록색
                )

        # =============================================================
        # 5. 플롯 2 & 3 - ArduPilot 궤적 및 세그먼트 플롯 (주황색)
        # =============================================================
        if feat_ardu is not None and len(feat_ardu) > 0:
            print(f"[{run_folder}] Generating ArduPilot Trajectory plot (Visualizing Segments)...")
            plot_full_trajectory_with_spans(
                t=t_ardu, features=feat_ardu, spans=spans_ardu, 
                title=f"Trajectory [Segment Check]: ArduPilot ({run_folder})", 
                save_path=str(run_dir / f"features_ardupilot_seg_check_{run_folder}.png"),
                line_color='tab:orange' # 주황색
            )
            
            if turn_ardu and len(turn_ardu) > 0:
                t_turn, feat_turn = turn_ardu[0]
                plot_turn_segment_features(
                    t=t_turn, features=feat_turn, 
                    title=f"Trajectory [Isolated Turn]: ArduPilot ({run_folder})", 
                    save_path=str(run_dir / f"features_ardupilot_turn_seg_{run_folder}.png"),
                    line_color='tab:orange' # 주황색
                )

    print("\n[Info] All combined XY and segmentation plots generated successfully!")