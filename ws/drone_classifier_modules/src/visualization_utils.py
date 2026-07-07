import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pywt

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
        ax.plot(t, features[:, i], color=line_color, linewidth=2.0, marker='o', markersize=4)
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
