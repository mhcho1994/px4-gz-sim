import os
import numpy as np
import pywt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import matplotlib

# WSL 환경 등 디스플레이가 없는 경우를 위한 백엔드 설정
matplotlib.use('Agg')

from signal_processor import process_camera_recovered_flight_data, process_px4_flight_data, process_ardu_flight_data, process_rosbag_flight_data

def load_timeseries_dataset():
    BASE_DATA_DIR = Path("data") 

    X_timeseries = []
    y = []
    
    # 추출하려는 타겟 Feature의 기존 인덱스
    # 5: 'XY-Plane Accel Norm (m/s²)'
    # 7: 'XY-Plane Jerk Norm (m/s³)'
    target_indices = [5, 7]
    
    print("[Info] Loading PX4 timeseries data (Turn Segments)...")
    for i in range(250): 
        run_folder = f"run_{i:03d}" 
        px4_dir = BASE_DATA_DIR / "260408_sitl_logs_250" / run_folder / "px4_logs" / "raw"
        
        if px4_dir.exists():
            for file in os.listdir(px4_dir):
                if file.lower().endswith('.ulg'):
                    px4_result = process_px4_flight_data(str(px4_dir / file))
                    if px4_result[0] is not None:
                        _, _, _, _, segments_px4, _ = px4_result
                        if segments_px4['turn'] and len(segments_px4['turn']) > 0:
                            for turn_segment in segments_px4['turn']:
                                feat_turn = turn_segment['features']
                                selected_features = feat_turn[:, target_indices]
                                X_timeseries.append(selected_features) 
                                y.append(0)  # Class 0: PX4
                    break 

    print("[Info] Loading ArduPilot timeseries data (Turn Segments)...")
    for i in range(250):
        run_folder = f"run_{i:03d}"
        ardu_dir = BASE_DATA_DIR / "260408_sitl_logs_250" / run_folder / "ardu_logs" / "raw" / "logs"
        
        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith('.bin'):
                    ardu_result = process_ardu_flight_data(str(ardu_dir / file))
                    if ardu_result[0] is not None:
                        _, _, _, _, segments_ardu, _ = ardu_result
                        if segments_ardu['turn'] and len(segments_ardu['turn']) > 0:
                            for turn_segment in segments_ardu['turn']:
                                feat_turn = turn_segment['features']
                                selected_features = feat_turn[:, target_indices]
                                X_timeseries.append(selected_features) 
                                y.append(1)  # Class 1: ArduPilot
                    break
                    
    print("[Info] Loading real ArduPilot timeseries data (Turn Segments)...")
    for i in range(3):
        run_folder = f"run_{i:03d}"
        ardu_dir = BASE_DATA_DIR / "260417_flight_logs" / run_folder / "ardu_logs" / "processed"
        
        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith('.csv'):
                    ardu_result = process_camera_recovered_flight_data(str(ardu_dir / file))
                    if ardu_result[0] is not None:
                        _, _, _, _, segments_rosbag, _ = ardu_result
                        if segments_rosbag['turn'] and len(segments_rosbag['turn']) > 0:
                            for turn_segment in segments_rosbag['turn']:
                                feat_turn = turn_segment['features']
                                selected_features = feat_turn[:, target_indices]
                                X_timeseries.append(selected_features) 
                                y.append(1)  # Class 1: ArduPilot
                    break

    for i in range(3):
        run_folder = f"run_{i:03d}"
        ardu_dir = BASE_DATA_DIR / "260417_flight_logs" / run_folder / "cogni_logs" / "processed"
        
        if ardu_dir.exists():
            for file in os.listdir(ardu_dir):
                if file.lower().endswith('.csv'):
                    ardu_result = process_camera_recovered_flight_data(str(ardu_dir / file))
                    if ardu_result[0] is not None:
                        _, _, _, _, segments_rosbag, _ = ardu_result
                        if segments_rosbag['turn'] and len(segments_rosbag['turn']) > 0:
                            for turn_segment in segments_rosbag['turn']:
                                feat_turn = turn_segment['features']
                                selected_features = feat_turn[:, target_indices]
                                X_timeseries.append(selected_features) 
                                y.append(0)  # Class 0: CogniPilot
                    break

    return X_timeseries, np.array(y)

def extract_dwt_features(timeseries_matrix, waveletname='db4', level=3):
    flight_features = []
    min_len = pywt.Wavelet(waveletname).dec_len * (2 ** level)
    
    if timeseries_matrix.shape[0] < min_len:
        pad_width = min_len - timeseries_matrix.shape[0]
        timeseries_matrix = np.pad(timeseries_matrix, ((0, pad_width), (0, 0)), mode='edge')
    
    for i in range(timeseries_matrix.shape[1]):
        signal = timeseries_matrix[:, i]
        coeffs = pywt.wavedec(signal, waveletname, level=level)
        
        # 저주파 3개(cA3, cD3, cD2)만 사용
        coeffs_to_use = coeffs[:3] 
        
        for coeff in coeffs_to_use:
            mean_val = np.mean(coeff)
            std_val = np.std(coeff)
            energy_val = np.sum(np.square(coeff)) 
            max_val = np.max(coeff)
            min_val = np.min(coeff)
            kurt_val = kurtosis(coeff)
            peak_loc_max = np.argmax(coeff) / len(coeff)
            peak_loc_min = np.argmin(coeff) / len(coeff)
            
            flight_features.extend([
                mean_val, std_val, energy_val, max_val, min_val, 
                kurt_val, peak_loc_max, peak_loc_min
            ])
            
    return np.array(flight_features)

# =====================================================================
# [신규 추가] SVM 결과를 시각화하는 함수
# =====================================================================
def plot_classification_results(X_train_scaled, X_test_scaled, y_train, y_test, y_pred, target_names):
    # 1. Confusion Matrix 플롯
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    # fig, ax = plt.subplots(figsize=(6, 5))
    # disp.plot(cmap=plt.cm.Blues, ax=ax)
    # plt.title("SVM Classification Confusion Matrix", fontweight='bold')
    # plt.tight_layout()
    # plt.savefig("../../data/figure/svm_confusion_matrix.png", dpi=150)
    # plt.close()
    # print("[Info] Saved Confusion Matrix plot as '../../data/figure/svm_confusion_matrix.png'")

    # 2. PCA 2D Scatter 플롯 (48차원 -> 2차원 축소)
    pca = PCA(n_components=2)

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))

    # 핵심: vmin, vmax 고정
    scatter_train = ax.scatter(
        X_train_pca[:, 0], X_train_pca[:, 1],
        c=y_train, cmap='coolwarm', vmin=0, vmax=1,
        alpha=0.3, marker='o', label='Train Data'
    )

    scatter_test = ax.scatter(
        X_test_pca[:, 0], X_test_pca[:, 1],
        c=y_test, cmap='coolwarm', vmin=0, vmax=1,
        alpha=1.0, marker='X', s=100, edgecolor='k', label='Test Data'
    )

    ax.set_title("PCA 2D Projection of DWT Features", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(0.0),
            markersize=10, label=target_names[0]),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(1.0),
            markersize=10, label=target_names[1]),
        Line2D([0], [0], marker='X', color='k', markerfacecolor='gray',
            markersize=10, linestyle='None', label='Test Data')
    ]

    ax.legend(handles=custom_lines, loc='best')

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("./results/svm_figs/svm_pca_scatter.png", dpi=150)
    plt.close()

    print("[Info] Saved PCA Scatter plot as './results/svm_figs/svm_pca_scatter.png'")



    # Train 데이터 기준으로 PCA 학습 및 변환
    # X_train_pca = pca.fit_transform(X_train_scaled)
    # X_test_pca = pca.transform(X_test_scaled)

    # fig, ax = plt.subplots(figsize=(8, 6))
    
    # # Train 데이터 시각화 (투명하게)
    # scatter_train = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
    #                            cmap='coolwarm', alpha=0.3, marker='o', label='Train Data')
    
    # # Test 데이터 시각화 (진하게, 테두리 추가)
    # scatter_test = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
    #                           cmap='coolwarm', alpha=1.0, marker='X', s=100, edgecolor='k', label='Test Data')

    # ax.set_title("PCA 2D Projection of DWT Features", fontsize=14, fontweight='bold')
    # ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    # ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    # # 범례 커스텀 생성 (색상 기반)
    # from matplotlib.lines import Line2D
    # custom_lines = [
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter_train.cmap(0.0), markersize=10, label=target_names[0]),
    #     Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter_train.cmap(1.0), markersize=10, label=target_names[1])
    # ]
    # ax.legend(handles=custom_lines, loc='best')

    # ax.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.savefig("./results/svm_figs/svm_pca_scatter.png", dpi=150)
    # plt.close()
    # print("[Info] Saved PCA Scatter plot as '../../data/figure/svm_pca_scatter.png'")


def main():
    X_timeseries, y = load_timeseries_dataset()

    if len(X_timeseries) < 5:
        print("\n[Error] Not enough data. Please check the folder paths and log files.")
        return
        
    print(f"\n[Success] Loaded {len(X_timeseries)} segmented flight data logs in total.")
    
    print("[Info] Extracting features using Discrete Wavelet Transform (DWT)...")
    X_dwt = []
    for ts_matrix in X_timeseries:
        dwt_vector = extract_dwt_features(ts_matrix, waveletname='db4', level=3)
        X_dwt.append(dwt_vector)
        
    X_dwt = np.array(X_dwt)
    print(f"[Success] Extraction complete! DWT Feature Matrix Shape: {X_dwt.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_dwt[0:1065], y[0:1065], test_size=0.2, random_state=42)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n[Info] Starting SVM model training...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)

    X_valid_scaled = scaler.transform(X_dwt[1065:])
    y_valid_pred = model.predict(X_valid_scaled)
    print(y_valid_pred)
    
    print("\n================ Classification Results ================")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    target_names = ['PX4', 'ArduPilot']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # [신규 추가] 시각화 함수 호출
    plot_classification_results(X_train_scaled, X_valid_scaled[0:], y_train, y[1065:], y_valid_pred[0:], target_names)

if __name__ == "__main__":
    main()