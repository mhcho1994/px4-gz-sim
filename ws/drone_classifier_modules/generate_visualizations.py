import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pywt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Use 'Agg' backend for non-interactive environments (WSL, Servers)
import matplotlib
matplotlib.use('Agg')


from data_extractor import parse_px4_ulog, parse_ardu_bin, parse_real_csv
from kinematic_processor_legacy import compute_kinematics, FEATURE_MAP     # Import the old signal_processor
import kinematic_processor
from flight_segmenter import extract_segments, _calculate_emission_probs, _smooth_with_viterbi

# =====================================================================
# 1. Helper Functions for New Plot Types (Unchanged)
# =====================================================================

def plot_3d_trajectory_comparison(x1, y1, z1, x2, y2, z2, labels, title, save_path, colors=None):
    """Generates a 3D comparison of two trajectories, or a single one if x2 is None."""
    if colors is None:
        colors = ['tab:green', 'tab:orange']
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot first trajectory
    if x1 is not None:
        ax.plot(y1, x1, z1, color=colors[0], label=labels[0], linewidth=1.5)
        ax.scatter(y1[0], x1[0], z1[0], color=colors[0], marker='o', s=80, edgecolor='k') # Start point
    
    # Plot second trajectory 
    if x2 is not None:
        ax.plot(y2, x2, z2, color=colors[1], label=labels[1], linewidth=1.5)
        ax.scatter(y2[0], x2[0], z2[0], color=colors[1], marker='o', s=80, edgecolor='k') # Start point

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('East (Y) [m]')
    ax.set_ylabel('North (X) [m]')
    ax.set_zlabel('Altitude (Z) [m]')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_state_variables(t_list, states_list, labels, colors, title, save_path):
    """Plots x, y, z, vx, vy, vz for comparison or single flight."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    var_names = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 
                 'VX Velocity (m/s)', 'VY Velocity (m/s)', 'VZ Velocity (m/s)']
    
    for i in range(6):
        ax = axes[i//2, i%2]
        for t, states, label, color in zip(t_list, states_list, labels, colors):
            ax.plot(t, states[:, i], color=color, label=label, alpha=0.8)
        
        ax.set_title(var_names[i], fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        if i == 0: ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_full_trajectory_with_spans(t, features, spans, title, save_path, line_color):
    """Visualizes the full flight with colored spans indicating different segments."""
    if features is None or len(features) == 0: return

    feature_names = ['Altitude (m)', 'Heading (rad)', 'Z-Axis Velocity (m/s)', 'XY-Plane Speed (m/s)', 'Z-Axis Acceleration (m/s²)', 'XY-Plane Accel Norm (m/s²)', 'Z-Axis Jerk (m/s³)', 'XY-Plane Jerk Norm (m/s³)', 'Curvature (1/m)', 'Yaw rate (rad/s)']
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    axes_flat = axes.flatten()
    
    # span_colors = {'ascending': 'tab:orange', 'straight': 'tab:green', 'turn': 'tab:red', 'descending': 'tab:brown', 'hovering': 'tab:gray'}
    # span_colors = {'ascending': 'orange', 'straight': 'mediumseagreen', 'turn': 'crimson', 'descending': 'mediumpurple', 'hovering': 'dodgerblue'}
    span_colors = {'ascending': 'tab:orange', 'straight': 'tab:green', 'turn': 'tab:red', 'descending': 'tab:blue', 'hovering': 'tab:gray'}

    for i in range(10):
        ax = axes_flat[i]
        ax.plot(t, features[:, i], color=line_color, linewidth=1.5, zorder=2)
        ax.set_title(feature_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
        
        if i == 6: ax.set_ylim(np.nanpercentile(features[:, i], 2), np.nanpercentile(features[:, i], 98))
        elif i == 7: ax.set_ylim(-1, np.nanpercentile(features[:, i], 98))
        elif i == 8: ax.set_ylim(-0.1, np.nanpercentile(features[:, i], 98))

        if spans:
            for s, e in spans.get('ascending', []): ax.axvspan(s, e, color=span_colors['ascending'], alpha=0.2, zorder=0)
            for s, e in spans.get('descending', []): ax.axvspan(s, e, color=span_colors['descending'], alpha=0.2, zorder=0)
            for s, e in spans.get('straight', []): ax.axvspan(s, e, color=span_colors['straight'], alpha=0.2, zorder=0)
            for s, e in spans.get('turn', []): ax.axvspan(s, e, color=span_colors['turn'], alpha=0.4, zorder=0) 
            for s, e in spans.get('hovering', []): ax.axvspan(s, e, color=span_colors['hovering'], alpha=0.2, zorder=0) 
    
    if spans:
        legend_patches = [
            mpatches.Patch(color=span_colors['ascending'], alpha=0.2, label='Ascending'),
            mpatches.Patch(color=span_colors['straight'], alpha=0.2, label='Straight'),
            mpatches.Patch(color=span_colors['turn'], alpha=0.4, label='Turn (Target)'),
            mpatches.Patch(color=span_colors['descending'], alpha=0.2, label='Descending'),
            mpatches.Patch(color=span_colors['hovering'], alpha=0.2, label='Hovering')
        ]
        fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.95, 0.98), ncol=5, fontsize=12)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_turn_segment_features(t, features, title, save_path, line_color):
    """Zooms in and plots features only for a specific isolated turn segment."""
    if features is None or len(features) == 0: return

    feature_names = ['Altitude (m)', 'Heading (rad)', 'Z-Axis Velocity (m/s)', 'XY-Plane Speed (m/s)', 'Z-Axis Acceleration (m/s²)', 'XY-Plane Accel Norm (m/s²)', 'Z-Axis Jerk (m/s³)', 'XY-Plane Jerk Norm (m/s³)', 'Curvature (1/m)', 'Yaw rate (rad/s)']
    
    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    axes_flat = axes.flatten()
    
    for i in range(10):
        ax = axes_flat[i]
        ax.plot(t, features[:, i], color=line_color, linewidth=2.0)
        ax.set_title(feature_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Absolute Time (s)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 6: ax.set_ylim(np.nanpercentile(features[:, i], 2), np.nanpercentile(features[:, i], 98))
        elif i == 7: ax.set_ylim(-1, np.nanpercentile(features[:, i], 98))
        elif i == 8: ax.set_ylim(-0.1, np.nanpercentile(features[:, i], 98))
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_dwt_features(t, data, target_indices, feature_names, title, save_path, color, wavelet='db4', level=3):
    """Visualizes DWT decomposition (Signal and Details) for selected features."""
    n_feat = len(target_indices)
    fig, axes = plt.subplots(n_feat, level + 1, figsize=(4 * (level+1), 3 * n_feat))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    t_rel = t - t[0]
    
    for i, f_idx in enumerate(target_indices):
        sig = data[:, f_idx]
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        
        # Original Signal
        axes[i, 0].plot(t_rel, sig, color=color)
        axes[i, 0].set_ylabel(feature_names[f_idx], fontweight='bold')
        if i == 0: axes[i, 0].set_title("Original")
        
        # Details (cD)
        for l in range(level):
            ax = axes[i, l+1]
            # Rescale time for coefficients
            t_coeff = np.linspace(0, t_rel[-1], len(coeffs[level-l]))
            ax.plot(t_coeff, coeffs[level-l], color=color, alpha=0.7)
            if i == 0: ax.set_title(f"Detail Level {l+1}")
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_hmm_viterbi_states(t, probs, viterbi_labels, title, save_path):
    """Plots HMM Emission Probabilities and Viterbi Decoding over Time."""
    state_config = {
        0: {'key': 'hovering',   'name': 'Hovering',   'color': 'tab:gray'},
        1: {'key': 'ascending',  'name': 'Ascending',  'color': 'tab:orange'},
        2: {'key': 'descending', 'name': 'Descending', 'color': 'tab:blue'},
        3: {'key': 'straight',   'name': 'Straight',   'color': 'tab:green'},
        4: {'key': 'turn_left',  'name': 'Left Turn',  'color': 'tab:purple'},
        5: {'key': 'turn_right', 'name': 'Right Turn', 'color': 'tab:brown'}
    }


    state_names = [state_config[i]['name'] for i in range(6)]
    colors = [state_config[i]['color'] for i in range(6)]
    label_map = {state_config[i]['key']: i for i in range(6)}
    numeric_labels = [label_map[l] for l in viterbi_labels]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1.2]}, sharex=True)
    
    # Subplot 1: Stacked area chart
    ax1.stackplot(t, probs.T, labels=state_names, colors=colors, alpha=0.8)
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Probability')
    ax1.set_ylim(0, 1.0)
    ax1.set_xlim(t[0], t[-1])
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Subplot 2: Viterbi Decoded State
    ax2.step(t, numeric_labels, where='post', color='black', linewidth=1.5, zorder=2)
    
    start_idx = 0
    for i in range(1, len(t)):
        if numeric_labels[i] != numeric_labels[i-1] or i == len(t)-1:
            ax2.axvspan(t[start_idx], t[i], color=colors[numeric_labels[start_idx]], alpha=0.4, lw=0, zorder=1)
            start_idx = i
            
    ax2.set_yticks(range(6))
    ax2.set_yticklabels(state_names)
    ax2.set_ylabel('Viterbi State')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, linestyle='--', alpha=0.4, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

# =====================================================================
# 2. Main Visualization Engine
# =====================================================================

def run_visualization_pipeline(base_folder_name, is_sitl=True):
    BASE_DATA_DIR = Path("data") / base_folder_name
    SAVE_BASE_DIR = Path("results") / f"{base_folder_name}_viz"
    
    if not BASE_DATA_DIR.exists():
        print(f"[Error] Folder not found: {BASE_DATA_DIR}")
        return

    run_folders = sorted([f for f in os.listdir(BASE_DATA_DIR) 
                         if f.startswith("run_") and (BASE_DATA_DIR / f).is_dir()])

    # All feature names for reference
    all_feature_names = ['Alt', 'Heading', 'VZ', 'XY-Speed', 'AZ', 'XY-Accel', 'JZ', 'XY-Jerk', 'Curvature', 'YawRate', 'Slip']
    target_features_dwt = ['XY-Accel', 'XY-Jerk', 'Curvature']
    selected_indices = [FEATURE_MAP[f] for f in target_features_dwt]

    for run_folder in run_folders:
        print(f"\n[Processing] {run_folder} in {base_folder_name}...")
        run_path = BASE_DATA_DIR / run_folder
        save_path_run = SAVE_BASE_DIR / run_folder
        save_path_run.mkdir(parents=True, exist_ok=True)

        # 1. Load Data (PX4, ArduPilot, Cogni)
        data_store = {}
        
        # Firmware search configuration
        configs = [
            ('px4', run_path / "px4_logs", '.ulg' if is_sitl else '.csv', 'tab:green'),
            ('ardu', run_path / "ardu_logs", '.bin' if is_sitl else '.csv', 'tab:orange')
        ]

        for fw, fw_path, ext, color in configs:
            if is_sitl and fw == 'px4':
                sub_dir = fw_path
            elif is_sitl and fw == 'ardu':
                sub_dir = fw_path / "logs"
            elif not is_sitl and fw in ['px4', 'ardu']:
                sub_dir = fw_path / "processed"
            else:
                raise ValueError(f"Unsupported configuration: fw={fw}, is_sitl={is_sitl}")
            
            if sub_dir.exists():
                for file in os.listdir(sub_dir):
                    if file.lower().endswith(ext):
                        path = str(sub_dir / file)
                        
                        # Step 1: Extract Raw Data
                        if fw == 'px4' and is_sitl: raw_data = parse_px4_ulog(path)
                        elif fw == 'ardu' and is_sitl: raw_data = parse_ardu_bin(path)
                        else: raw_data = parse_real_csv(path, measurement_type='mocap')
                        
                        # Step 2 & 3: Kinematics and Segmentation
                        if raw_data is not None:
                            t_full, feat_full = compute_kinematics(raw_data)
                            kinematic_features = kinematic_processor.compute_kinematics(raw_data)
                            segs, spans = extract_segments(kinematic_features)
                            
                            probs = _calculate_emission_probs(kinematic_features)
                            viterbi_labels = _smooth_with_viterbi(probs)

                            states = np.vstack((raw_data['x'], raw_data['y'], raw_data['z'], 
                                                raw_data['vx'], raw_data['vy'], raw_data['vz'])).T
                            
                            data_store[fw] = {
                                't_raw': raw_data['t'], 't_full': t_full, 'states': states, 'feat': feat_full, 
                                'segs': segs, 'spans': spans, 'color': color,
                                'raw_xy': (raw_data['x'], raw_data['y'], raw_data['z']),
                                'probs': probs, 'viterbi_labels': viterbi_labels, 'kinematic_t': kinematic_features['t_window']
                            }
                        break

        # =============================================================
        # Type 0: Trajectory 3D Plot (Combined or Individual)
        # =============================================================
        if is_sitl and 'px4' in data_store and 'ardu' in data_store:
            # SITL: Overlap 3D
            p = data_store['px4']['raw_xy']
            a = data_store['ardu']['raw_xy']
            plot_3d_trajectory_comparison(
                p[0], p[1], p[2], a[0], a[1], a[2], ['PX4', 'ArduPilot'],
                f"3D Trajectory Comparison ({run_folder})",
                str(save_path_run / f"{run_folder}_0_combined_traj3d.png")
            )
        else:
            # Real Flight (or incomplete SITL): Separate 3D per firmware
            for fw, d in data_store.items():
                x, y, z = d['raw_xy']
                plot_3d_trajectory_comparison(
                    x1=x, y1=y, z1=z, 
                    x2=None, y2=None, z2=None, 
                    labels=[fw.upper(), ''], 
                    title=f"3D Trajectory: {fw.upper()} ({run_folder})",
                    save_path=str(save_path_run / f"{run_folder}_0_{fw}_traj3d.png"),
                    colors=[d['color'], 'k'] 
                )

        # =============================================================
        # Type 1: State Variables Comparison (x,y,z,vx,vy,vz)
        # =============================================================
        if is_sitl and 'px4' in data_store and 'ardu' in data_store:
            plot_state_variables(
                [data_store['px4']['t_raw'], data_store['ardu']['t_raw']],
                [data_store['px4']['states'], data_store['ardu']['states']],
                ['PX4', 'ArduPilot'], ['tab:green', 'tab:orange'],
                f"State Variables Comparison ({run_folder})",
                str(save_path_run / f"{run_folder}_1_combined_states.png")
            )
        else:
            for fw, d in data_store.items():
                plot_state_variables(
                    [d['t_raw']], [d['states']], [fw.upper()], [d['color']],
                    f"State Variables: {fw.upper()} ({run_folder})",
                    str(save_path_run / f"{run_folder}_1_{fw}_states.png")
                )

        # =============================================================
        # Type 2 & 3 & 4 & 5: HMM, Segments, Features, and DWT
        # =============================================================
        for fw, d in data_store.items():
            # Type 2: HMM Segmentation
            plot_hmm_viterbi_states(
                t=d['kinematic_t'], probs=d['probs'], viterbi_labels=d['viterbi_labels'],
                title=f"HMM Emission Probabilities & Viterbi Decoding: {fw.upper()} ({run_folder})",
                save_path=str(save_path_run / f"{run_folder}_2_{fw}_hmm_viterbi.png")
            )

            # Type 3: Segment Check (Full Highlight)
            
            # plot_full_trajectory_with_spans(
            #     t=d['t_full'], features=d['feat'], spans=d['spans'],
            #     title=f"Segment Check: {fw.upper()} ({run_folder})",
            #     save_path=str(save_path_run / f"{run_folder}_3_{fw}_seg_check.png"),
            #     line_color=d['color']
            # )
            spans_for_plot = d['spans'].copy()
            if 'turn_left' in spans_for_plot or 'turn_right' in spans_for_plot:
                spans_for_plot['turn'] = spans_for_plot.pop('turn_left', []) + spans_for_plot.pop('turn_right', [])
            plot_full_trajectory_with_spans(
                t=d['t_full'], features=d['feat'], spans=spans_for_plot,
                title=f"Segment Check: {fw.upper()} ({run_folder})",
                save_path=str(save_path_run / f"{run_folder}_3_{fw}_seg_check.png"),
                line_color=d['color']
            )

            # Type 4 & 5: Turn Segments
            if d['segs'] and d['segs'].get('turn'):
                for i, seg in enumerate(d['segs']['turn']):
                    seg_id = f"seg{i+1}"
                    # Type 4: Isolated Segment Features
                    plot_turn_segment_features(
                        t=seg['time'], features=seg['features'],
                        title=f"Turn Segment Features: {fw.upper()} {seg_id}",
                        save_path=str(save_path_run / f"{run_folder}_4_{fw}_{seg_id}_features.png"),
                        line_color=d['color']
                    )
                    
                    # Type 5: DWT Result
                    plot_dwt_features(
                        t=seg['time'], data=seg['features'],
                        target_indices=selected_indices, feature_names=all_feature_names,
                        title=f"DWT Decomposition: {fw.upper()} {seg_id}",
                        save_path=str(save_path_run / f"{run_folder}_5_{fw}_{seg_id}_dwt.png"),
                        color=d['color']
                    )

def main():
    parser = argparse.ArgumentParser(description="Generate flight visualizations from logs.")
    parser.add_argument("log_folder", type=str, nargs='?', default="260615_sitl_logs", help="Name of the log folder (e.g., '260615_sitl_logs')")
    parser.add_argument("--sitl", action="store_true", help="Set this flag if the logs are SITL logs")
    parser.add_argument("--real", dest="sitl", action="store_false", help="Set this flag if the logs are real flight logs")
    parser.set_defaults(sitl=True)
    
    args = parser.parse_args()
    
    print(f"[Info] Starting visualization pipeline for folder: {args.log_folder} (SITL={args.sitl})")
    run_visualization_pipeline(args.log_folder, is_sitl=args.sitl)

    print("\n[Success] All visualizations generated successfully.")

if __name__ == "__main__":
    main()