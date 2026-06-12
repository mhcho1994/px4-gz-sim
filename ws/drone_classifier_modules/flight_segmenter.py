import numpy as np

# ==============================================================================
# [Module 1] Motion Primitive Probability & HMM Smoothing
# ==============================================================================

def _calculate_emission_probs(features, hover_thresh=0.5, pitch_takeoff=45.0, 
                              pitch_landing=-45.0, vz_takeoff=0.5, vz_landing=-0.5, 
                              turn_sharpness=0.1):
    """
    Computes a soft probability distribution over 6 motion primitives.
    States: 0:Hovering, 1:Take-off, 2:Landing, 3:Straight, 4:Left_Turn, 5:Right_Turn
    """
    n_frames = len(features['t_window'])
    prob_matrix = np.zeros((n_frames, 6))

    for i in range(n_frames):
        vx, vy, vz = features['vx_reg'][i], features['vy_reg'][i], features['vz_reg'][i]
        pitch = features['pitch_deg'][i]
        sharpness = features['turn_sharpness'][i]
        direction = features['turn_direction'][i]

        speed_3d = np.sqrt(vx**2 + vy**2 + vz**2)
        scores = np.zeros(6)

        # 1. Hovering Score
        scores[0] = max(0.0, 1 - (speed_3d / hover_thresh))

        # 2 & 3. Take-off and Landing Scores
        if pitch > 0 and vz > 0:
            scores[1] = min(1.0, pitch / pitch_takeoff) * min(1.0, vz / vz_takeoff)
        elif pitch < 0 and vz < 0:
            scores[2] = min(1.0, pitch / pitch_landing) * min(1.0, vz / vz_landing)

        # 4 & 5. Turn Left / Right Scores
        turn_score = min(1.0, sharpness / (turn_sharpness * 2.0))
        if direction > 0:
            scores[4] = turn_score
        else:
            scores[5] = turn_score

        # 6. Straight Score (Fallback)
        max_other_score = np.max([scores[0], scores[1], scores[2], scores[4], scores[5]])
        scores[3] = max(0.0, 1.0 - max_other_score)

        # Normalization
        scores += 1e-6
        prob_matrix[i, :] = scores / np.sum(scores)

    return prob_matrix

def _smooth_with_viterbi(emission_probs):
    """
    Decodes the most likely sequence of primitives using Viterbi algorithm.
    """
    n_frames, n_states = emission_probs.shape
    pi = np.array([0.7, 0.3, 0.0, 0.0, 0.0, 0.0]) # Initial probs
    
    # Transition Matrix A (Physical Rules Engine)
    # A = np.array([
    #     # Hov,  Tkf,  Lnd,  Str,   LT,   RT
    #     [0.80, 0.10, 0.10, 0.00, 0.00, 0.00], # Hover
    #     [0.10, 0.70, 0.00, 0.20, 0.00, 0.00], # Take-off
    #     [0.10, 0.00, 0.80, 0.10, 0.00, 0.00], # Landing
    #     [0.00, 0.00, 0.05, 0.65, 0.15, 0.15], # Straight
    #     [0.00, 0.00, 0.00, 0.20, 0.70, 0.10], # Left_Turn
    #     [0.00, 0.00, 0.00, 0.20, 0.10, 0.70]  # Right_Turn
    # ])
    A = np.array([
        # Hov,  Tkf,  Lnd,  Str,   LT,   RT
        [0.25, 0.15, 0.15, 0.15, 0.15, 0.15], # Hover
        [0.15, 0.25, 0.15, 0.15, 0.15, 0.15], # Take-off
        [0.15, 0.15, 0.25, 0.15, 0.15, 0.15], # Landing
        [0.15, 0.15, 0.15, 0.25, 0.15, 0.15], # Straight
        [0.15, 0.15, 0.15, 0.15, 0.25, 0.15], # Left_Turn
        [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]  # Right_Turn
    ])
    # A = np.array([
    #     # Hov,  Tkf,  Lnd,  Str,   LT,   RT
    #     [0.80, 0.20, 0.00, 0.00, 0.00, 0.00], # Hover
    #     [0.00, 0.70, 0.00, 0.30, 0.00, 0.00], # Take-off
    #     [0.30, 0.00, 0.70, 0.00, 0.00, 0.00], # Landing
    #     [0.00, 0.00, 0.05, 0.65, 0.15, 0.15], # Straight
    #     [0.00, 0.00, 0.00, 0.25, 0.65, 0.10], # Left_Turn
    #     [0.00, 0.00, 0.00, 0.25, 0.10, 0.65]  # Right_Turn
    # ])
    
    eps = 1e-10
    log_pi, log_A, log_B = np.log(pi + eps), np.log(A + eps), np.log(emission_probs + eps)
    
    V = np.zeros((n_frames, n_states))
    ptr = np.zeros((n_frames, n_states), dtype=int)
    
    V[0, :] = log_pi + log_B[0, :]
    for t in range(1, n_frames):
        for j in range(n_states):
            seq_probs = V[t-1, :] + log_A[:, j]
            best_prev_state = np.argmax(seq_probs)
            ptr[t, j] = best_prev_state
            V[t, j] = seq_probs[best_prev_state] + log_B[t, j]
            
    best_path = np.zeros(n_frames, dtype=int)
    best_path[-1] = np.argmax(V[-1, :])
    for t in range(n_frames - 2, -1, -1):
        best_path[t] = ptr[t+1, best_path[t+1]]
        
    state_names = ["hovering", "takeoff", "landing", "straight", "turn_left", "turn_right"]
    return np.array([state_names[idx] for idx in best_path])


# ==============================================================================
# [Module 2] Legacy Interface Adapter (Plug-and-Play Output)
# ==============================================================================

def extract_segments(features_dict, dt=None, min_turn_duration=None, 
                     yaw_rate_threshold=None, straight_duration=None, heading_margin=None):
    """
    Analyzes geometric features (from kinematic_processor.py) to split the flight.
    Returns the exact same `segments` and `spans` dictionary structure as the legacy code.
    
    Note: Legacy arguments (dt, yaw_rate_threshold, etc.) are ignored as the new
    pipeline relies on PCA, Regression, and HMM parameters internally.
    """
    # Output structure identical to legacy code
    spans = {'takeoff': None, 'landing': None, 'straight': [], 'turn_left': [], 'turn_right': []}
    segments = {'takeoff': None, 'landing': None, 'straight': [], 'turn_left': [], 'turn_right': []}

    if features_dict is None or len(features_dict['t_window']) < 2:
        return segments, spans

    t_window = features_dict['t_window']
    n_frames = len(t_window)

    # 1. Run the Probabilistic & HMM pipeline
    emission_probs = _calculate_emission_probs(features_dict)
    smoothed_labels = _smooth_with_viterbi(emission_probs)

    # 2. Parse the contiguous blocks of labels and populate the legacy dictionary
    # We find the start and end indices of identical consecutive labels.
    current_label = smoothed_labels[0]
    start_idx = 0

    for i in range(1, n_frames + 1):
        # Trigger block extraction if label changes OR end of array is reached
        if i == n_frames or smoothed_labels[i] != current_label:
            end_idx = i - 1
            span = (t_window[start_idx], t_window[end_idx])
            
            # Reconstruct a subset of the feature dictionary for this block
            # (Matches the legacy expectation of providing subset features)
            subset_features = {k: v[start_idx:end_idx+1] for k, v in features_dict.items()}
            
            segment_data = {
                'label': current_label,
                'time': t_window[start_idx:end_idx+1],
                'features': subset_features,
                'span': span
            }

            # Map to dictionary keys
            if current_label in ['takeoff', 'landing']:
                # The legacy code assumed only one takeoff/landing per flight
                # We overwrite with the latest occurrence, or you could append if needed.
                spans[current_label] = span
                segments[current_label] = segment_data
            elif current_label in ['straight', 'turn_left', 'turn_right']:
                spans[current_label].append(span)
                segments[current_label].append(segment_data)
            # 'hovering' blocks are intentionally omitted from output to match legacy,
            # unless you want to add a 'hovering' key to the dictionary.

            # Reset for the next block
            if i < n_frames:
                current_label = smoothed_labels[i]
                start_idx = i

    return segments, spans


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import data_extractor
    import kinematic_processor

    # 1. Parse raw data
    log_path = "./data/sitl_logs/run_012/ardu_logs/raw/logs/00000001.BIN"
    print(f"[Info] Parsing raw flight data from:\n       {log_path}")
    raw_flight_data = data_extractor.parse_ardu_bin(log_path)

    if raw_flight_data is None:
        print("[Error] Failed to parse the specified ArduPilot BIN file.")
    else:
        print("[Info] Data successfully extracted. Processing kinematics...")
        
        # 2. Extract features
        features = kinematic_processor.compute_kinematics(raw_flight_data)

        if features is None:
            print("[Error] Could not process kinematics (insufficient data).")
        else:
            print("[Info] Kinematics processed successfully. Calculating emission probabilities...")
            
            # 3. Calculate emission probabilities
            probs = _calculate_emission_probs(features)
            
            t = features['t_window']
            state_names = ["Hovering", "Take-off", "Landing", "Straight", "Left Turn", "Right Turn"]
            colors = ['tab:gray', 'tab:orange', 'tab:brown', 'tab:green', 'tab:blue', 'tab:purple']

            # 4. Create plot using a stacked area chart to see the probability distribution
            plt.figure(figsize=(14, 8))
            plt.stackplot(t, probs.T, labels=state_names, colors=colors, alpha=0.8)
            plt.title('HMM Emission Probabilities over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Time (s)')
            plt.ylabel('Probability')
            plt.ylim(0, 1.0)
            plt.xlim(t[0], t[-1])
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.4)
            
            plt.tight_layout()
            
            # Save and optionally display
            save_file = "emission_probs_test_result.png"
            plt.savefig(save_file, dpi=300)
            print(f"[Success] Plot saved to '{save_file}'")