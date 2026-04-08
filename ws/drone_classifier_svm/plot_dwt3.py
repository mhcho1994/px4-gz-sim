import os
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

from signal_processor import process_ardu_flight_data, process_px4_flight_data
import numpy as np
import pywt

matplotlib.use('Agg') 

def plot_dwt_comparison_grid(t1, px4_data, t2, ardu_data, target_features, all_feature_names, dt=0.02, waveletname='db4', level=3, title="DWT Comparison", save_path="dwt_comparison.png"):
    nrows = len(target_features)
    ncols = level + 2 
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    fig.suptitle(title, fontsize=22, fontweight='bold', y=0.98)
    
    # [수정 포인트] 두 구간의 시작 시간이 다를 수 있으므로, 겹쳐서 비교하기 위해 t=0으로 상대시간 동기화
    t1_rel = t1 - t1[0]
    t2_rel = t2 - t2[0]
    
    for row_idx, feature_name in enumerate(target_features):
        feat_idx = all_feature_names.index(feature_name)
        
        sig1 = px4_data[:, feat_idx]
        sig2 = ardu_data[:, feat_idx]
        
        coeffs1 = pywt.wavedec(sig1, waveletname, level=level)
        coeffs2 = pywt.wavedec(sig2, waveletname, level=level)
        
        # [열 0] Original Signal (상대 시간 t1_rel, t2_rel 적용 및 색상 동기화)
        ax_orig = axes[row_idx, 0]
        ax_orig.plot(t1_rel, sig1, label="PX4", color='tab:green', linewidth=1.5, alpha=0.8)
        ax_orig.plot(t2_rel, sig2, label="ArduPilot", color='tab:orange', linewidth=1.5, alpha=0.8)
        ax_orig.set_ylabel(f"{feature_name}\nAmplitude", fontweight='bold', fontsize=12)
        ax_orig.grid(True, linestyle='--', alpha=0.7)
        
        if row_idx == 0:
            ax_orig.set_title("Signal", fontsize=14, fontweight='bold')
            ax_orig.legend(loc='upper right', fontsize=10)
            
        # [열 1] Approximation (cA)
        cA1, cA2 = coeffs1[0], coeffs2[0]
        t_cA1 = np.linspace(t1_rel[0], t1_rel[-1], len(cA1))
        t_cA2 = np.linspace(t2_rel[0], t2_rel[-1], len(cA2))
        
        ax_ca = axes[row_idx, 1]
        ax_ca.plot(t_cA1, cA1, color='tab:green', linewidth=1.5, alpha=0.8)
        ax_ca.plot(t_cA2, cA2, color='tab:orange', linewidth=1.5, alpha=0.8)
        ax_ca.grid(True, linestyle='--', alpha=0.7)
        if row_idx == 0: ax_ca.set_title(f"Approximation (cA{level})", fontsize=14, fontweight='bold')
            
        # [열 2 ~ N] Detail (cD)
        for i in range(level):
            col_idx = i + 2
            cD1, cD2 = coeffs1[i + 1], coeffs2[i + 1]
            t_cD1 = np.linspace(t1_rel[0], t1_rel[-1], len(cD1))
            t_cD2 = np.linspace(t2_rel[0], t2_rel[-1], len(cD2))
            
            ax_cd = axes[row_idx, col_idx]
            ax_cd.plot(t_cD1, cD1, color='tab:green', linewidth=1, alpha=0.8)
            ax_cd.plot(t_cD2, cD2, color='tab:orange', linewidth=1, alpha=0.8)
            ax_cd.grid(True, linestyle='--', alpha=0.7)
            
            if row_idx == 0:
                ax_cd.set_title(f"Detail (cD{level - i})", fontsize=14, fontweight='bold')
                
    for col_idx in range(ncols):
        axes[nrows - 1, col_idx].set_xlabel("Relative Turn Time (s)", fontsize=12, fontweight='bold')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    BASE_DATA_DIR = Path("data") 
    
    all_feature_names = [
        "Altitude", "Heading", "Z-Axis Velocity", "XY-Plane Speed", 
        "Z-Axis Acceleration", "XY-Plane Accel Norm", 
        "Z-Axis Jerk", "XY-Plane Jerk Norm", "Curvature", "Yaw rate", "Slip rate"
    ]
    target_features = [
        "XY-Plane Speed", "Heading", "XY-Plane Accel Norm", 
        "XY-Plane Jerk Norm", "Curvature", "Altitude", "Yaw rate"
    ]

    print("[Info] Starting Segmented DWT Grid comparison plot generation...")

    for i in range(100):
        run_folder = f"run_{i:03d}"
        px4_dir = BASE_DATA_DIR / "sitl_logs" / run_folder / "px4_logs" / "raw"
        ardu_dir = BASE_DATA_DIR / "sitl_logs" / run_folder / "ardu_logs" / "raw" / "logs"
        output_dir = BASE_DATA_DIR / "sitl_logs" / run_folder
        
        if not output_dir.exists(): 
            continue

        sample_data_px4, t_px4 = None, None
        sample_data_ardu, t_ardu = None, None
        
        # Process PX4
        if px4_dir.exists():
            for file in os.listdir(px4_dir):
                if file.lower().endswith('.ulg'):
                    px4_result = process_px4_flight_data(str(px4_dir / file))
                    if px4_result[0] is not None:
                        _, _, _, _, segments_px4, _ = px4_result
                        if segments_px4['turn'] and len(segments_px4['turn']) > 0:
                            turn_seg = segments_px4['turn'][0]
                            t_px4 = turn_seg['time']
                            sample_data_px4 = turn_seg['features']
                    break
                    
        # Process ArduPilot
        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith('.bin'):
                    ardu_result = process_ardu_flight_data(str(ardu_dir / file))
                    if ardu_result[0] is not None:
                        _, _, _, _, segments_ardu, _ = ardu_result
                        if segments_ardu['turn'] and len(segments_ardu['turn']) > 0:
                            turn_seg = segments_ardu['turn'][0]
                            t_ardu = turn_seg['time']
                            sample_data_ardu = turn_seg['features']
                    break
                    
        if sample_data_px4 is not None and len(sample_data_px4) > 0 and \
           sample_data_ardu is not None and len(sample_data_ardu) > 0:
           
            print(f"[{run_folder}] Generating DWT grid plot for Turn Segment...")
            auto_title = f"DWT Comparison [Turn Segment]: PX4 vs ArduPilot ({run_folder})"
            save_path = output_dir / f"dwt_compare_turn_seg_{run_folder}.png"
            
            plot_dwt_comparison_grid(
                t1=t_px4, px4_data=sample_data_px4, 
                t2=t_ardu, ardu_data=sample_data_ardu, 
                target_features=target_features,
                all_feature_names=all_feature_names,
                title=auto_title, save_path=str(save_path)
            )
        else:
            print(f"[{run_folder}] Missing valid turn data for comparison. Skipping...")

    print("[Info] All DWT grid plots generated successfully!")