import numpy as np
from scipy.interpolate import interp1d
from pyulog import ULog
from pymavlink import mavutil
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  

TARGET_HZ = 50.0  
DT = 1.0 / TARGET_HZ #20ms


PX4_BASE_DIR = Path("../../data/px4_logs/")

def resample_and_extract_features(t, x, y, z, vx, vy, vz, t_att, yaw_att):
    t_start = max(t[0], t_att[0])
    t_end = min(t[-1], t_att[-1])
    
    t_new = np.arange(t_start, t_end, DT)
    # print(f"New time axis: {t_new}") 

    x_new = interp1d(t, x)(t_new)
    y_new = interp1d(t, y)(t_new)
    z_new = interp1d(t, z)(t_new)
    vx_new = interp1d(t, vx)(t_new)
    vy_new = interp1d(t, vy)(t_new)
    vz_new = interp1d(t, vz)(t_new)
    
    # unwrap even 
    yaw_unwrapped = np.unwrap(yaw_att)
    yaw_new = interp1d(t_att, yaw_unwrapped)(t_new)
    
    
    #Extract 7 features:
    v_vec = np.vstack((vx_new, vy_new, vz_new)).T
    speed = np.linalg.norm(v_vec, axis=1)
    
    # Acceleration 
    # np.diff reduces one data length because it computes differences, so we pad the result with the first value.
    ax = np.insert(np.diff(vx_new) / DT, 0, 0)
    ay = np.insert(np.diff(vy_new) / DT, 0, 0)
    az = np.insert(np.diff(vz_new) / DT, 0, 0)
    a_vec = np.vstack((ax, ay, az)).T
    accel_mag = np.linalg.norm(a_vec, axis=1)
    
    # Jerk (difference of acceleration)
    jx = np.insert(np.diff(ax) / DT, 0, 0)
    jy = np.insert(np.diff(ay) / DT, 0, 0)
    jz = np.insert(np.diff(az) / DT, 0, 0)
    jerk_mag = np.linalg.norm(np.vstack((jx, jy, jz)).T, axis=1)
    
    # Curvature (using cross product of velocity and acceleration)
    cross_va = np.cross(v_vec, a_vec)
    cross_mag = np.linalg.norm(cross_va, axis=1)
    curvature = cross_mag / (speed**3 + 1e-6)
    
    # Yaw Rate & Slip
    heading_traj = np.unwrap(np.arctan2(vy_new, vx_new))
    yaw_rate_traj = np.insert(np.diff(heading_traj) / DT, 0, 0)
    yaw_rate_att = np.insert(np.diff(yaw_new) / DT, 0, 0)
    slip_rate = yaw_rate_traj - yaw_rate_att
    
    #(N, 7) 2D Array 
    features = np.vstack((speed, accel_mag, jerk_mag, curvature, yaw_rate_traj, yaw_rate_att, slip_rate)).T

    ####EXCLUDE non-flight data (e.g. on the ground, takeoff/landing hover)###############
    FLIGHT_THRESHOLD = 4
    flight_mask = (-z_new) > FLIGHT_THRESHOLD
    filtered_features = features[flight_mask]
    # print(f"Total data points: {len(features)}, Flight data points: {len(filtered_features)}")

    return filtered_features, t_new, x_new, y_new, z_new, vx_new, vy_new, vz_new

def process_px4_ulog(ulog_path):
    try:
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset('vehicle_local_position').data
        t_loc = loc_data['timestamp'] / 1e6
        x, y, z = loc_data['x'], loc_data['y'], loc_data['z']
        vx, vy, vz = loc_data['vx'], loc_data['vy'], loc_data['vz']
        
        att_data = ulog.get_dataset('vehicle_attitude').data
        t_att = att_data['timestamp'] / 1e6
        q0, q1, q2, q3 = att_data['q[0]'], att_data['q[1]'], att_data['q[2]'], att_data['q[3]']
        yaw_att = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
        
        return resample_and_extract_features(t_loc, x, y, z, vx, vy, vz, t_att, yaw_att)
    except Exception as e:
        print(f"[PX4 Data Process Error] {ulog_path}: {e}")
        return None

def process_ardu_bin(bin_path):
    try:
        mlog = mavutil.mavlink_connection(bin_path)
        t_loc, x, y, z, vx, vy, vz, t_att, yaw_att = [], [], [], [], [], [], [], [], []
        
        while True:
            msg = mlog.recv_match(type=['XKF1', 'NKF1', 'ATT'], blocking=False)
            if not msg: break
            if msg.get_type() in ['XKF1', 'NKF1']:
                t_loc.append(msg.TimeUS / 1e6)
                x.append(msg.PN); y.append(msg.PE); z.append(msg.PD)
                vx.append(msg.VN); vy.append(msg.VE); vz.append(msg.VD)
            elif msg.get_type() == 'ATT':
                t_att.append(msg.TimeUS / 1e6)
                yaw_att.append(np.radians(msg.Yaw))
                
        if len(x) < 50 or len(yaw_att) < 50: return None
        return resample_and_extract_features(
            np.array(t_loc), np.array(x), np.array(y), np.array(z), 
            np.array(vx), np.array(vy), np.array(vz), 
            np.array(t_att), np.array(yaw_att)
        )
    except Exception as e:
        print(f"[ArduPilot Data Process Error] {bin_path}: {e}")
        return None


def plot_comparison(ulog_path, output_dir="comparison_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        ulog = ULog(ulog_path)
        loc_data = ulog.get_dataset('vehicle_local_position').data
        t_loc = loc_data['timestamp'] / 1e6
        x_orig, y_orig, z_orig = loc_data['x'], loc_data['y'], loc_data['z']
        vx_orig, vy_orig, vz_orig = loc_data['vx'], loc_data['vy'], loc_data['vz']
        
        att_data = ulog.get_dataset('vehicle_attitude').data
        t_att = att_data['timestamp'] / 1e6
        q0, q1, q2, q3 = att_data['q[0]'], att_data['q[1]'], att_data['q[2]'], att_data['q[3]']
        yaw_att = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
        
        result = resample_and_extract_features(t_loc, x_orig, y_orig, z_orig, vx_orig, vy_orig, vz_orig, t_att, yaw_att)
        features, t_new, x_new, y_new, z_new, vx_new, vy_new, vz_new = result
        

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'Original vs Resampling (50Hz)\n{os.path.basename(ulog_path)}', fontsize=14, fontweight='bold')
        
        # X Axis
        axes[0, 0].plot(t_loc, x_orig, 'b-', alpha=0.5, linewidth=1, label='원본 데이터', marker='.')
        axes[0, 0].plot(t_new, x_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[0, 0].set_ylabel('X Position (m)', fontsize=10)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('X Position')
        
        # Y Axis
        axes[0, 1].plot(t_loc, y_orig, 'b-', alpha=0.5, linewidth=1, label='원본 데이터', marker='.')
        axes[0, 1].plot(t_new, y_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[0, 1].set_ylabel('Y Position (m)', fontsize=10)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Y Position')
        
        # Z Axis
        axes[1, 0].plot(t_loc, z_orig, 'b-', alpha=0.5, linewidth=1, label='원본 데이터', marker='.')
        axes[1, 0].plot(t_new, z_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[1, 0].set_ylabel('Z Position (m)', fontsize=10)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Z Position')
        
        # VX Axis
        axes[1, 1].plot(t_loc, vx_orig, 'b-', alpha=0.5, linewidth=1, label='원본 데이터', marker='.')
        axes[1, 1].plot(t_new, vx_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[1, 1].set_ylabel('VX Velocity (m/s)', fontsize=10)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('VX Velocity')
        
        # VY Axis
        axes[2, 0].plot(t_loc, vy_orig, 'b-', alpha=0.5, linewidth=1, label='원본 데이터', marker='.')
        axes[2, 0].plot(t_new, vy_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[2, 0].set_xlabel('Time (s)', fontsize=10)
        axes[2, 0].set_ylabel('VY Velocity (m/s)', fontsize=10)
        axes[2, 0].legend(fontsize=9)
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_title('VY Velocity')
        
        # VZ Axis
        axes[2, 1].plot(t_loc, vz_orig, 'b-', alpha=0.5, linewidth=1, label='원본 데이터', marker='.')
        axes[2, 1].plot(t_new, vz_new, 'r-', alpha=0.7, linewidth=1.5, label='Resampled (50Hz)')
        axes[2, 1].set_xlabel('Time (s)', fontsize=10)
        axes[2, 1].set_ylabel('VZ Velocity (m/s)', fontsize=10)
        axes[2, 1].legend(fontsize=9)
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_title('VZ Velocity')
        
        plt.tight_layout()
        
        # Save PNG 
        png_filename = os.path.basename(ulog_path).replace('.ulg', '_comparison.png')
        png_path = os.path.join(output_dir, png_filename)
        plt.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Save Comparison Graph: {png_path}")
        return png_path
        
    except Exception as e:
        print(f"[Visualization Error] {ulog_path}: {e}")
        return None
    
def save_resampled_data_to_csv(output_dir="resampled_data", plot_dir="comparison_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.exists(PX4_BASE_DIR):
        for file in os.listdir(PX4_BASE_DIR):
            if file.endswith('.ulg'):
                print(f"[Processing] {file}")
                ulog_path = os.path.join(PX4_BASE_DIR, file)
                try:
                    ulog = ULog(ulog_path)
                    loc_data = ulog.get_dataset('vehicle_local_position').data
                    t_loc = loc_data['timestamp'] / 1e6
                    x, y, z = loc_data['x'], loc_data['y'], loc_data['z']
                    vx, vy, vz = loc_data['vx'], loc_data['vy'], loc_data['vz']
                    
                    att_data = ulog.get_dataset('vehicle_attitude').data
                    t_att = att_data['timestamp'] / 1e6
                    q0, q1, q2, q3 = att_data['q[0]'], att_data['q[1]'], att_data['q[2]'], att_data['q[3]']
                    yaw_att = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
                    
                    result = resample_and_extract_features(t_loc, x, y, z, vx, vy, vz, t_att, yaw_att)
                    features, t_new, x_new, y_new, z_new, vx_new, vy_new, vz_new = result
                    
                    # Translate into DataFrame
                    df = pd.DataFrame({
                        'time': t_new,
                        'x': x_new,
                        'y': y_new,
                        'z': z_new,
                        'vx': vx_new,
                        'vy': vy_new,
                        'vz': vz_new,
                        'speed': features[:, 0],
                        'accel_mag': features[:, 1],
                        'jerk_mag': features[:, 2],
                        'curvature': features[:, 3],
                        'yaw_rate_traj': features[:, 4],
                        'yaw_rate_att': features[:, 5],
                        'slip_rate': features[:, 6]
                    })
                    
                    # Save CSV 
                    csv_filename = file.replace('.ulg', '_resampled.csv')
                    csv_path = os.path.join(output_dir, csv_filename)
                    df.to_csv(csv_path, index=False)
                    print(f"Save Resampled Data: {csv_path}")
                    print(f"    - Data Points: {len(df)}")
                    print(f"    - Sampling Interval: {DT} sec (50Hz)")
                    
                    # Generate comparison plot
                    plot_comparison(ulog_path, plot_dir)
                    
                except Exception as e:
                    print(f"Error: {e}")
                    
###############DELETE IN THE FUTURE###################
if __name__ == "__main__":
    # 예시 경로에 있는 실제 파일을 하나 넣어서 테스트해보세요!
    # px4_features = process_px4_ulog("your_px4_log.ulg")
    # print("추출된 특징 배열 모양:", px4_features.shape) # 예: (5000, 7)
        
    print("=" * 70)
    print(" Validate Resampling : ULog to CSV Conversion and Comparison Plot Generation...")
    print("=" * 70)
    save_resampled_data_to_csv()
    print("=" * 70)
    print("Complete! Resampled CSV files and comparison graphs have been saved.")
    print("CSV files location: resampled_data/ folder")
    print("Comparison graphs location: comparison_plots/ folder")
    print("Click PNG  -> 'Open Preview'.")
    print("=" * 70)