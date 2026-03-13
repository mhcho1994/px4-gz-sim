import os
import numpy as np
import pywt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import kurtosis

from signal_processor import process_px4_for_wavelet, process_ardu_for_wavelet

PX4_BASE_DIR = Path("data/px4_logs/")
ARDU_DIR = Path("data/ardu_logs/")

def load_timeseries_dataset():
    """
    Loads all log files from the specified directories and returns 
    the timeseries data list (X) and corresponding labels (y).
    """
    X_timeseries = []
    y = []
    
    print("[Info] Loading PX4 timeseries data...")
    if os.path.exists(PX4_BASE_DIR):
        for file in os.listdir(PX4_BASE_DIR):
            if file.endswith('.ulg'):
                feat_matrix = process_px4_for_wavelet(os.path.join(PX4_BASE_DIR, file))
                if feat_matrix is not None and len(feat_matrix) > 0:
                    X_timeseries.append(feat_matrix) # Shape: (N, 9) 
                    y.append(0)  # Class 0: PX4
                    
    print("[Info] Loading ArduPilot timeseries data...")
    if os.path.exists(ARDU_DIR):
        for file in os.listdir(ARDU_DIR):
            if file.endswith('.BIN'):
                feat_matrix = process_ardu_for_wavelet(os.path.join(ARDU_DIR, file))
                if feat_matrix is not None and len(feat_matrix) > 0:
                    X_timeseries.append(feat_matrix) # Shape: (N, 9) 
                    y.append(1)  # Class 1: ArduPilot
                    
    return X_timeseries, np.array(y)

def extract_dwt_features(timeseries_matrix, waveletname='db4', level=3):
    """
    Performs DWT with a fixed level (level=3) and extracts statistical values 
    along with kurtosis and relative peak locations from each coefficient.
    """
    flight_features = []
    
    # Calculate the minimum length required for the given wavelet and level
    min_len = pywt.Wavelet(waveletname).dec_len * (2 ** level)
    
    # Pad the timeseries matrix with edge values if it is too short
    if timeseries_matrix.shape[0] < min_len:
        pad_width = min_len - timeseries_matrix.shape[0]
        timeseries_matrix = np.pad(timeseries_matrix, ((0, pad_width), (0, 0)), mode='edge')
    
    # Perform DWT for each of the 9 signals (columns)
    for i in range(timeseries_matrix.shape[1]):
        signal = timeseries_matrix[:, i]
        
        # Decompose the signal into coefficients: [cA_n, cD_n, cD_n-1, ..., cD_1]
        coeffs = pywt.wavedec(signal, waveletname, level=level)
        
        # Extract features from each coefficient array
        for coeff in coeffs:
            # 1. Basic statistical features
            mean_val = np.mean(coeff)
            std_val = np.std(coeff)
            energy_val = np.sum(np.square(coeff)) # Energy of the signal
            max_val = np.max(coeff)
            min_val = np.min(coeff)
            
            # 2. Advanced statistical feature: Kurtosis (measures sharp spikes/outliers)
            kurt_val = kurtosis(coeff)
            
            # 3. Temporal peak locations (Relative position: 0.0 to 1.0)
            # Divide the index of the max/min by the length of the coefficient array
            peak_loc_max = np.argmax(coeff) / len(coeff)
            peak_loc_min = np.argmin(coeff) / len(coeff)
            
            # Append all 8 features for this specific coefficient
            flight_features.extend([
                mean_val, std_val, energy_val, max_val, min_val, 
                kurt_val, peak_loc_max, peak_loc_min
            ])
            
    return np.array(flight_features)

def main():
    # 1. Load original timeseries data
    X_timeseries, y = load_timeseries_dataset()
    
    if len(X_timeseries) < 5:
        print("\n[Error] Not enough data. Please check the folder paths and log files.")
        return
        
    print(f"\n[Success] Loaded {len(X_timeseries)} flight data logs in total.")
    
    # 2. Extract features using DWT
    print("[Info] Extracting features using Discrete Wavelet Transform (DWT)...")
    X_dwt = []
    for ts_matrix in X_timeseries:
        dwt_vector = extract_dwt_features(ts_matrix, waveletname='db4', level=3)
        X_dwt.append(dwt_vector)
        
    X_dwt = np.array(X_dwt)
    print(f"[Success] Extraction complete! DWT Feature Matrix Shape: {X_dwt.shape}")
    
    # 3. Data splitting and scaling
    X_train, X_test, y_train, y_test = train_test_split(X_dwt, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train and evaluate the SVM model
    print("\n[Info] Starting SVM model training...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    print("\n================ Classification Results ================")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['PX4', 'ArduPilot']))

if __name__ == "__main__":
    main()