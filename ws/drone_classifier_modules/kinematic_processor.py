import numpy as np
from scipy.interpolate import interp1d

def compute_kinematics(raw_data, target_hz=50, window_sec=1.0, overlap_sec=0.8):
    """
    Processes raw flight data to extract kinematic features using linear regression and PCA.
    Returns a dictionary of features for each sliding window.

    Parameters:
    - raw_data: Dictionary containing raw flight data with keys 't', 'x', 'y', 'z'
    - target_hz: Target sampling frequency (default: 50Hz)
    - window_sec: Sliding window duration in seconds (default: 1.0s)
    - overlap_sec: Overlap duration in seconds (default: 0.8s)

    Returns:
    - Dictionary of extracted features for each sliding window
    """
    if raw_data is None or len(raw_data['t']) < 10:
        return None

    # [1] linear interpolation to target_hz (e.g., 50Hz)
    t_raw = raw_data['t']
    x_raw, y_raw, z_raw = raw_data['x'], raw_data['y'], raw_data['z']
    
    # remove duplicate timestamps if any (can cause issues with interpolation)
    t_raw, unique_indices = np.unique(t_raw, return_index=True)
    x_raw, y_raw, z_raw = x_raw[unique_indices], y_raw[unique_indices], z_raw[unique_indices]
    
    if len(t_raw) < 10:
        return None

    t_start, t_end = t_raw[0], t_raw[-1]
    t_interp = np.arange(t_start, t_end, 1.0 / target_hz)
    
    interp_x = interp1d(t_raw, x_raw, kind='linear', fill_value="extrapolate")
    interp_y = interp1d(t_raw, y_raw, kind='linear', fill_value="extrapolate")
    interp_z = interp1d(t_raw, z_raw, kind='linear', fill_value="extrapolate")
    
    x_interp = interp_x(t_interp)
    y_interp = interp_y(t_interp)
    z_interp = interp_z(t_interp)

    # [2] parameters for sliding window
    window_size = int(target_hz * window_sec)         # 50Hz * 1.0s = 50 samples
    overlap_size = int(target_hz * overlap_sec)       # 50Hz * 0.8s = 40 samples
    stride = window_size - overlap_size               # Stride: 10 samples (0.2s)
    
    features = {
        't_window': [], 'vx_reg': [], 'vy_reg': [], 'vz_reg': [],
        'heading_deg': [], 'pitch_deg': [], 'turn_sharpness': [], 'turn_direction': []
    }

    # [3] extract features using sliding window
    for i in range(0, len(t_interp) - window_size + 1, stride):
        t_win = t_interp[i : i + window_size]
        x_win = x_interp[i : i + window_size]
        y_win = y_interp[i : i + window_size]
        z_win = z_interp[i : i + window_size]
        
        # A. estimate velocity using linear regression
        vx = np.polyfit(t_win, x_win, 1)[0]
        vy = np.polyfit(t_win, y_win, 1)[0]
        vz = np.polyfit(t_win, z_win, 1)[0]
        
        # B. PCA to find the main direction of motion and calculate geometric features
        points_3d = np.vstack((x_win, y_win, z_win)).T
        points_centered = points_3d - np.mean(points_3d, axis=0)
        
        cov_matrix = np.cov(points_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues and eigenvectors in descending order
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        e1 = eigenvectors[:, 0]  # Principal direction of motion
        l1 = eigenvalues[0]      # Largest variance along the principal direction (indicates straightness)
        l2 = eigenvalues[1]      # Second largest variance (indicates turn sharpness)
        
        # Ensure e1 points in the direction of motion (from start to end of the window)
        dp = points_3d[-1] - points_3d[0]
        if np.dot(e1, dp) < 0:
            e1 = -e1
            
        # C. Attitude (Heading & Pitch)
        pitch_rad = np.arcsin(np.clip(e1[2], -1.0, 1.0))
        pitch_deg = np.degrees(pitch_rad)
        
        heading_rad = np.arctan2(e1[1], e1[0])
        heading_deg = np.degrees(heading_rad)
        
        # Turn Sharpness (λ2 / λ1)
        sharpness = l2 / (l1 + 1e-6)
        
        # Turn Direction
        mid_idx = window_size // 2
        v1 = np.array([x_win[mid_idx] - x_win[0], y_win[mid_idx] - y_win[0]])
        v2 = np.array([x_win[-1] - x_win[mid_idx], y_win[-1] - y_win[mid_idx]])
        cross_z = v1[0]*v2[1] - v1[1]*v2[0]
        turn_dir = 1.0 if cross_z > 0 else -1.0  # +1 for left turn, -1 for right turn
        
        # Append features for this window
        features['t_window'].append(t_win[mid_idx])
        features['vx_reg'].append(vx)
        features['vy_reg'].append(vy)
        features['vz_reg'].append(vz)
        features['heading_deg'].append(heading_deg)
        features['pitch_deg'].append(pitch_deg)
        features['turn_sharpness'].append(sharpness)
        features['turn_direction'].append(turn_dir)

    return {k: np.array(v) for k, v in features.items()}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import data_extractor

    # 1. Parse raw data
    log_path = "./data/sitl_logs/run_012/ardu_logs/raw/logs/00000001.BIN"
    print(f"[Info] Parsing raw flight data from:\n       {log_path}")
    raw_flight_data = data_extractor.parse_ardu_bin(log_path)

    if raw_flight_data is None:
        print("[Error] Failed to parse the specified ArduPilot BIN file.")
    else:
        print("[Info] Data successfully extracted. Processing kinematics...")
        
        # 2. Extract features
        features = compute_kinematics(raw_flight_data)

        if features is None:
            print("[Error] Could not process kinematics (insufficient data).")
        else:
            print("[Info] Kinematics processed successfully. Generating plots...")
            
            t = features['t_window']

            # 3. Create plots
            fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
            fig.suptitle('Kinematic Features over Time', fontsize=16, fontweight='bold')

            # Plot Velocities
            axes[0].plot(t, features['vx_reg'], label='VX', color='r')
            axes[0].plot(t, features['vy_reg'], label='VY', color='g')
            axes[0].plot(t, features['vz_reg'], label='VZ', color='b')
            axes[0].set_title('Regressed Velocities')
            axes[0].set_ylabel('Speed (m/s)')
            axes[0].grid(True, linestyle='--', alpha=0.7)
            axes[0].legend(loc='upper right')

            # Plot Attitude (Heading & Pitch)
            axes[1].plot(t, features['heading_deg'], label='Heading', color='purple')
            axes[1].plot(t, features['pitch_deg'], label='Pitch', color='brown')
            axes[1].set_title('Attitude Characteristics')
            axes[1].set_ylabel('Angle (degrees)')
            axes[1].grid(True, linestyle='--', alpha=0.7)
            axes[1].legend(loc='upper right')

            # Plot Turn Sharpness
            axes[2].plot(t, features['turn_sharpness'], label='Sharpness', color='darkorange')
            axes[2].set_title('Turn Sharpness (PCA λ2 / λ1)')
            axes[2].set_ylabel('Ratio')
            axes[2].grid(True, linestyle='--', alpha=0.7)
            axes[2].legend(loc='upper right')

            # Plot Turn Direction
            axes[3].plot(t, features['turn_direction'], label='Turn Direction', color='teal', drawstyle='steps-post')
            axes[3].set_title('Turn Direction (+1=Left, -1=Right)')
            axes[3].set_xlabel('Time (s)')
            axes[3].set_ylabel('Direction')
            axes[3].set_yticks([-1, 0, 1])
            axes[3].grid(True, linestyle='--', alpha=0.7)
            axes[3].legend(loc='upper right')

            plt.tight_layout(rect=[0, 0.03, 1, 0.98])
            
            # Save and optionally display
            save_file = "kinematics_test_result.png"
            plt.savefig(save_file, dpi=300)
            print(f"[Success] Plot saved to '{save_file}'")
            
            # plt.show() # Uncomment if working in a GUI environment