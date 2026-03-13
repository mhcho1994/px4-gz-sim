import matplotlib.pyplot as plt
import matplotlib

from signal_processor import process_ardu_for_wavelet
# Use 'Agg' backend for environments without a display (like WSL)
matplotlib.use('Agg') 
import numpy as np
import pywt

def plot_dwt_coefficients(signal, dt=0.02, waveletname='db4', level=3, title="DWT Analysis", save_path="dwt_analysis.png"):
    """
    Plots the original signal and its DWT coefficients (Approximation and Details).
    Saves the plot as an image file instead of displaying it interactively (WSL-friendly).
    
    Parameters:
    - signal: 1D numpy array (e.g., Z-axis Velocity for a single flight)
    - dt: Sampling interval (default 0.02s)
    - waveletname: Name of the mother wavelet (default 'db4')
    - level: Decomposition level (default 3)
    - title: Plot title
    - save_path: File path to save the generated plot
    """
    # Time axis for the original signal
    t = np.arange(len(signal)) * dt
    
    # Perform DWT (Returns: [cA_n, cD_n, cD_n-1, ..., cD_1])
    coeffs = pywt.wavedec(signal, waveletname, level=level)
    
    # Create subplots (1 for original signal + (level+1) for coefficients)
    fig, axes = plt.subplots(level + 2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Plot Original Signal
    axes[0].plot(t, signal, color='black', linewidth=1.5)
    axes[0].set_title("Original Signal", fontsize=12)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # 2. Plot Approximation Coefficient (cA_level)
    # Adjust time axis for the downsampled approximation coefficient
    cA = coeffs[0]
    t_cA = np.linspace(0, t[-1], len(cA))
    axes[1].plot(t_cA, cA, color='tab:blue', linewidth=1.5)
    axes[1].set_title(f"Approximation (cA{level}) - The overall low-frequency trend", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # 3. Plot Detail Coefficients (cD_level down to cD1)
    colors = ['tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for i in range(level):
        cD = coeffs[i + 1]
        t_cD = np.linspace(0, t[-1], len(cD))
        ax = axes[i + 2]
        
        # cD_level indices represent higher frequencies as they go down to 1
        current_level = level - i
        ax.plot(t_cD, cD, color=colors[i % len(colors)], linewidth=1)
        ax.set_title(f"Detail (cD{current_level}) - High-frequency components", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save the plot and close the figure to free memory
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"The plot has been saved as {save_path}")

# =====================================================================
# Example Usage
# =====================================================================
if __name__ == "__main__":
    import os
    from signal_processor import process_px4_for_wavelet
    
    # Example 1: "data/px4_logs/05_43_10.ulg"
    # Example 2: "data/ardu_logs/00000005.BIN"
    log_file_path = "data/ardu_logs/00000005.BIN"

    # Determine platform
    base_name = os.path.basename(log_file_path)        # e.g., "05_43_10.ulg"
    file_id, ext = os.path.splitext(base_name)         # e.g., "05_43_10", ".ulg"
    if ext.lower() == '.ulg':
        platform = "PX4"
        sample_data = process_px4_for_wavelet(log_file_path)
    elif ext.lower() == '.bin':
        platform = "ArduPilot"
        sample_data = process_ardu_for_wavelet(log_file_path)
    else:
        print(f"[Error] Unsupported file format: {ext}. Please use .ulg or .BIN.")
        sample_data = None


    # Proceed if data extraction is successful
    if sample_data is not None and len(sample_data) > 0:
        # Feature Indices:
        feature_names = [
            "Altitude", "Heading", "Z-Axis Velocity", "XY-Plane Speed", 
            "Z-Axis Acceleration", "XY-Plane Accel Norm", 
            "Z-Axis Jerk", "XY-Plane Jerk Norm", "Curvature"
        ]
        
        # Example: Let's extract and plot the Z-Axis Velocity (Index 2)
        feature_idx = 2 
        feature_name = feature_names[feature_idx]
        signal_1d = sample_data[:, feature_idx]

        # Generate the Title and Save Path
        # Example Title: "DWT Decomposition: PX4 [01_21_59] - Z-Axis Velocity"
        auto_title = f"DWT Decomposition: {platform} [{file_id}] - {feature_name}"

        # Example Save Path: "dwt_px4_01_21_59_Z-Axis_Velocity.png"
        safe_feature_name = feature_name.replace(" ", "_") # Replace spaces for safe filenames
        auto_save_path = f"dwt_{platform.lower()}_{file_id}_{safe_feature_name}.png"

        print(f"Generating DWT plot for {platform} data: {base_name} ({feature_name})...")
        
        plot_dwt_coefficients(
            signal=signal_1d, 
            title=auto_title, 
            save_path=auto_save_path
        )
    else:
        print("[Error] Failed to load data or the flight was too short.")