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
from matplotlib.lines import Line2D
import matplotlib

# WSL 환경 등 디스플레이가 없는 경우를 위한 백엔드 설정
matplotlib.use('Agg')

from data_loader import load_px4_dataset, load_ardu_dataset, load_cogni_dataset

# =====================================================================
# 3. DWT 특징 추출 함수
# =====================================================================
def extract_dwt_features(timeseries_matrix, waveletname='db4', level=3):
    flight_features = []
    min_len = pywt.Wavelet(waveletname).dec_len * (2 ** level)
    
    if timeseries_matrix.shape[0] < min_len:
        pad_width = min_len - timeseries_matrix.shape[0]
        timeseries_matrix = np.pad(timeseries_matrix, ((0, pad_width), (0, 0)), mode='edge')
    
    for i in range(timeseries_matrix.shape[1]):
        signal = timeseries_matrix[:, i]
        coeffs = pywt.wavedec(signal, waveletname, level=level)
        
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
# 4. Visualization Function
# =====================================================================
def plot_classification_results(X_train_scaled, X_test_scaled, y_train, y_test, y_pred, target_names, X_real_scaled=None, y_real=None, y_real_pred=None):
    save_dir = "data/figure"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Plot Confusion Matrix (Uncommented to fix NameError)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("SVM Classification Confusion Matrix", fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/svm_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"[Info] Saved Confusion Matrix plot to '{save_dir}/svm_confusion_matrix.png'")

    # 2. PCA 2D Scatter Plot (Reduce 48 dimensions to 2)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot SITL Train Data (Circles)
    scatter_train = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                               cmap='coolwarm', alpha=0.3, marker='o', label='Train Data (SITL)')
    
    # Plot SITL Test Data (Circles with thick edges)
    scatter_test = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
                              cmap='coolwarm', alpha=1.0, marker='o', s=100, edgecolor='k', label='Test Data (SITL)')

    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter_train.cmap(0.0), markersize=10, label=f"SITL: {target_names[0]}"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter_train.cmap(1.0), markersize=10, label=f"SITL: {target_names[1]}")
    ]

    # Visualize Real Flight Data (Mark correct/incorrect predictions)
    if X_real_scaled is not None and len(X_real_scaled) > 0 and y_real is not None and y_real_pred is not None:
        X_real_pca = pca.transform(X_real_scaled)
        
        y_real_array = np.array(y_real)
        correct_mask = (y_real_array == y_real_pred)
        incorrect_mask = (y_real_array != y_real_pred)

        if np.any(correct_mask):
            ax.scatter(X_real_pca[correct_mask, 0], X_real_pca[correct_mask, 1], 
                       c='gold', marker='*', s=300, edgecolor='k', linewidths=1.5, zorder=5)
            custom_lines.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='k', markersize=15, label='Real Flight (Correct)'))

        if np.any(incorrect_mask):
            ax.scatter(X_real_pca[incorrect_mask, 0], X_real_pca[incorrect_mask, 1], 
                       c='red', marker='X', s=200, edgecolor='white', linewidths=1.5, zorder=5)
            custom_lines.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markeredgecolor='w', markersize=12, label='Real Flight (Incorrect)'))

    ax.set_title("PCA 2D Projection of DWT Features", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")

    ax.legend(handles=custom_lines, loc='best')

    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/svm_pca_scatter.png", dpi=150)
    plt.close()
    print(f"[Info] Saved PCA Scatter plot to '{save_dir}/svm_pca_scatter.png'")


def main():
    cache_dir = Path("ws/drone_classifier_svm/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load SITL Data and Extract DWT Features (with caching)
    sitl_cache_file = cache_dir / "sitl_dwt_cache.npz"
    if sitl_cache_file.exists():
        print(f"\n[Info] Loading cached SITL DWT features from '{sitl_cache_file}'...")
        cache = np.load(sitl_cache_file)
        X_sitl_dwt = cache['X']
        y_sitl = cache['y']
        print(f"[Success] SITL DWT Feature Matrix Shape: {X_sitl_dwt.shape}")
    else:
        # 1. Load SITL Data (SITL uses 'raw' paths by default)
        X_px4_sitl, y_px4_sitl = load_px4_dataset("sitl_logs")
        X_ardu_sitl, y_ardu_sitl = load_ardu_dataset("sitl_logs", data_type='raw')
        
        X_sitl_ts = X_px4_sitl + X_ardu_sitl
        y_sitl = np.concatenate((y_px4_sitl, y_ardu_sitl))

        if len(X_sitl_ts) < 5:
            print("\n[Error] Not enough SITL data to train the model.")
            return 
            
        print(f"\n[Success] Loaded {len(X_sitl_ts)} SITL flight data segments.")
        print("[Info] Extracting DWT features for SITL data (This may take a while)...")
        
        X_sitl_dwt = np.array([extract_dwt_features(ts, waveletname='db4', level=3) for ts in X_sitl_ts])
        
        np.savez(sitl_cache_file, X=X_sitl_dwt, y=y_sitl)
        print(f"[Info] Saved extracted SITL features to cache.")
        print(f"[Success] SITL DWT Feature Matrix Shape: {X_sitl_dwt.shape}")

    # 1-2. Train / Test 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X_sitl_dwt, y_sitl, test_size=0.1, random_state=42)
    
    # 1-3. 데이터 정규화 (스케일링)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1-4. SVM 모델 학습
    print("\n[Info] Starting SVM model training on SITL data...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    target_names = ['PX4', 'ArduPilot']

    print("\n================ SITL Classification Results ================")
    print(f"Accuracy on SITL Test Data: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, labels=[0, 1], target_names=target_names, zero_division=0))

    # =====================================================================
    # --- Step 2: Load and Validate Real Flight Data ---
    # =====================================================================
    print("\n" + "="*30)
    print("  REAL FLIGHT DATA LOADING")
    print("="*30)
    
    test_folders = ["260424_flight_logs"]
    X_real_ts = []
    y_real = []

    for folder in test_folders:
        print(f"\n[Processing Folder] {folder}")
        
        # 0. Load PX4 real data (labeled as Class 0)
        print(f" Loading PX4 real flight data from '{folder}'...")
        X_px4_real, y_px4_real = load_px4_dataset(folder, data_type='processed', measurement_type='mocap')
        print(f"  --> PX4 Subtotal for {folder}: {len(X_px4_real)} segments.")
        if len(X_px4_real) > 0:
            X_real_ts.extend(X_px4_real)
            y_real.extend(y_px4_real)
            
        # 1. Load ArduPilot real data (labeled as Class 1)
        print(f" Loading ArduPilot real flight data from '{folder}'...")
        X_ardu_real, y_ardu_real = load_ardu_dataset(folder, data_type='processed', measurement_type='mocap')
        print(f"  --> ArduPilot Subtotal for {folder}: {len(X_ardu_real)} segments.")
        if len(X_ardu_real) > 0:
            X_real_ts.extend(X_ardu_real)
            y_real.extend(y_ardu_real)
            
        # 2. Load Cogni real data (labeled as Class 2)
        print(f" Loading Cogni real flight data from '{folder}'...")
        X_cogni_real, y_cogni_real = load_cogni_dataset(folder, data_type='processed', measurement_type='mocap')
        print(f"  --> Cogni Subtotal for {folder}: {len(X_cogni_real)} segments.")
        if len(X_cogni_real) > 0:
            X_real_ts.extend(X_cogni_real)
            y_real.extend(y_cogni_real)

        # 3. 폴더별 총합 출력
        folder_total = len(X_px4_real) + len(X_ardu_real) + len(X_cogni_real)
        print(f"  ==> Total segments for {folder}: {folder_total}")

    print("\n" + "="*30)
    print("  REAL FLIGHT TEST RESULTS")
    print("="*30)

    X_real_scaled = None
    y_real_pred = None

    if len(X_real_ts) > 0:
        # Extract DWT features for real flight segments
        X_real_dwt = np.array([extract_dwt_features(ts, waveletname='db4', level=3) for ts in X_real_ts])
        
        # Filter out any rows containing NaN values before scaling and prediction
        valid_indices = ~np.isnan(X_real_dwt).any(axis=1)
        removed_count = len(X_real_dwt) - np.sum(valid_indices)
        
        if removed_count > 0:
            print(f"\n[Warning] Removed {removed_count} corrupted segments containing NaN values.")
            
        # Keep only valid data without NaNs
        X_real_dwt = X_real_dwt[valid_indices]
        y_real = np.array(y_real)[valid_indices]
        
        if len(X_real_dwt) == 0:
            print("[Error] No valid real flight data left after removing NaNs.")
            return

        # Scale features and perform prediction
        X_real_scaled = scaler.transform(X_real_dwt)
        y_real_pred = model.predict(X_real_scaled)
        
        correct_count = np.sum(y_real_pred == np.array(y_real))
        print(f"Final Accuracy: {(correct_count / len(y_real)) * 100:.2f}% ({correct_count}/{len(y_real)})")
        
        # Print detailed per-segment matching report
        print("\n[Detailed Prediction Map]")
        print("-" * 65)
        print(f"{'No.':<4} | {'True Label':<12} | {'Prediction':<12} | {'Status'}")
        print("-" * 65)
        
        label_map = {0: "PX4", 1: "ArduPilot", 2: "Cogni"}
        for i in range(len(y_real)):
            true_name = label_map.get(y_real[i], "Unknown")
            pred_name = label_map.get(y_real_pred[i], "Unknown")
            status = "Match" if true_name == pred_name else "Fail"
            print(f"{i+1:<4} | {true_name:<12} | {pred_name:<12} | {status}")
        print("-" * 65)
        
        # Standard classification report
        print("\n[Classification Report]")
        print(classification_report(y_real, y_real_pred, labels=[0, 1, 2], target_names=['PX4', 'ArduPilot', 'Cogni'], zero_division=0))
            
    else:
        print("[Warning] No real flight data segments were loaded for testing.")

    # --- 단계 3: 시각화 ---
    plot_classification_results(X_train_scaled, X_test_scaled, y_train, y_test, y_pred, target_names, X_real_scaled, y_real, y_real_pred)


    # Notes for Debugging
    #print("Debugging Notes: only 70 SITL sets are used for debugging")

if __name__ == "__main__":
    main()