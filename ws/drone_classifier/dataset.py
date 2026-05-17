import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import random
from trajectory_processor import (
    fit_robust_feature_scaler,
    process_ardu_flight_data,
    process_px4_flight_data,
    save_robust_feature_scaler,
    transform_robust_features,
)

# =====================================================================
# TIME-SERIES DATASET  (CNN / CNN-LSTM)
# =====================================================================
class DroneTrajectoryDataset(Dataset):
    def __init__(self, data_list, labels_list, window_size, step_size):
        self.windows = []
        self.labels = []

        for data, label in zip(data_list, labels_list):
            num_steps = len(data)

            for i in range(0, num_steps - window_size + 1, step_size):
                window = data[i : i + window_size]
                self.windows.append(window.T)
                self.labels.append(label)

        self.windows = torch.tensor(np.array(self.windows), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.windows[idx], self.labels[idx]


# =====================================================================
# PER-WINDOW RELATIVE FEATURES  (MLP)
# =====================================================================
# Feature index mapping in the 7-column time-series from trajectory_processor:
#   0: vh (vertical velocity)     1: speed_xy       2: acc_v
#   3: curvature                  4: yaw_rate        5: yaw_angular_accel
#   6: speed × curvature

def compute_relative_window_features(window: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Collapse a (window_size, 7) time-series window into 7 dimensionless scalars.

    All features are ratios or coefficients of variation so their values are
    independent of the absolute flight speed and spatial scale of the mission.
    This makes them transferable between SITL (fast/large) and real flights
    (slow/small).
    """
    vh      = window[:, 0]
    spd     = window[:, 1]
    acc_v   = window[:, 2]
    yaw_r   = window[:, 4]
    yaw_aa  = window[:, 5]
    spd_c   = window[:, 6]

    return np.array([
        np.std(spd)   / (np.mean(spd)           + eps),   # 0: CV horiz speed
        np.std(vh)    / (np.mean(np.abs(vh))     + eps),   # 1: CV vert velocity
        np.mean(np.abs(acc_v)) / (np.mean(np.abs(vh))  + eps),  # 2: |accel_v| / |vel_v|
        np.std(yaw_r) / (np.mean(np.abs(yaw_r)) + eps),   # 3: CV yaw rate
        np.mean(np.abs(yaw_aa)) / (np.mean(np.abs(yaw_r)) + eps), # 4: yaw accel / yaw rate
        np.mean(np.abs(spd_c)),                            # 5: mean turning rate (rad/s proxy)
        np.std(spd_c) / (np.mean(np.abs(spd_c)) + eps),   # 6: CV turning rate
    ], dtype=np.float32)


def compute_relative_features_for_trajectory(feat_series, window_size, step_size):
    """Slide windows over a feature time-series and return per-window scalar array."""
    n = len(feat_series)
    out = []
    for i in range(0, n - window_size + 1, step_size):
        out.append(compute_relative_window_features(feat_series[i:i + window_size]))
    return np.array(out, dtype=np.float32) if out else None


class DroneTrajectoryRelativeDataset(Dataset):
    """Dataset of per-window dimensionless scalar features for the MLP."""
    def __init__(self, window_features_list, labels_list):
        all_feats = np.vstack(window_features_list)
        all_labels = np.concatenate([
            np.full(len(wf), lbl, dtype=np.int64)
            for wf, lbl in zip(window_features_list, labels_list)
        ])
        self.features = torch.tensor(all_feats, dtype=torch.float32)
        self.labels   = torch.tensor(all_labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# =====================================================================
# SHARED HELPERS
# =====================================================================
def extract_flight_signatures(file_list, pos_noise_std=0.0):
    data_list = []
    labels_list = []

    for file_path, label in file_list:
        # Per-file noise level: sample uniformly in [0, pos_noise_std] for augmentation variety
        noise = float(np.random.uniform(0.0, pos_noise_std)) if pos_noise_std > 0 else 0.0

        if label == 0:
            result = process_px4_flight_data(str(file_path), pos_noise_std=noise)
        else:
            result = process_ardu_flight_data(str(file_path), pos_noise_std=noise)

        if result is not None:
            features = result[4]
            if features is not None and len(features) > 0:
                data_list.append(features)
                labels_list.append(label)

    return data_list, labels_list


def build_training_pipeline(px4_dir, ardu_dir, batch_size, test_ratio,
                             window_size, step_size, robust_stats_path=None,
                             use_relative_features=False, pos_noise_std=0.0):
    px4_files = list(Path(px4_dir).glob("*.ulg"))
    ardu_files = list(Path(ardu_dir).glob("*.bin"))

    if not px4_files and not ardu_files:
        raise ValueError("No log files found in the specified directories.")

    random.shuffle(px4_files)
    random.shuffle(ardu_files)

    px4_test_cnt  = max(1, int(len(px4_files)  * test_ratio)) if len(px4_files)  > 1 else 0
    ardu_test_cnt = max(1, int(len(ardu_files) * test_ratio)) if len(ardu_files) > 1 else 0

    train_files = [(f, 0) for f in px4_files[px4_test_cnt:]]  + [(f, 1) for f in ardu_files[ardu_test_cnt:]]
    test_files  = [(f, 0) for f in px4_files[:px4_test_cnt]]  + [(f, 1) for f in ardu_files[:ardu_test_cnt]]

    print(f"Files divided - Training: {len(train_files)} | Test: {len(test_files)}")
    print("⏳ Loading Train Data...")
    train_data, train_labels = extract_flight_signatures(train_files, pos_noise_std=pos_noise_std)
    print("⏳ Loading Test Data...")
    test_data, test_labels = extract_flight_signatures(test_files, pos_noise_std=pos_noise_std)

    if use_relative_features:
        # --- MLP path: compute per-window scalars, then robust-scale them ---
        print("📐 Computing per-window relative features...")
        train_wf = [compute_relative_features_for_trajectory(f, window_size, step_size)
                    for f in train_data]
        test_wf  = [compute_relative_features_for_trajectory(f, window_size, step_size)
                    for f in test_data]

        train_wf = [w for w in train_wf if w is not None and len(w) > 0]
        test_wf  = [w for w in test_wf  if w is not None and len(w) > 0]

        print("📏 Fitting robust scaler on per-window training features...")
        robust_stats = fit_robust_feature_scaler(train_wf)
        train_wf = [transform_robust_features(w, robust_stats) for w in train_wf]
        test_wf  = [transform_robust_features(w, robust_stats) for w in test_wf]

        if robust_stats_path is not None:
            save_robust_feature_scaler(robust_stats, robust_stats_path)
            print(f"✅ Saved robust scaler stats: {robust_stats_path}")

        train_dataset = DroneTrajectoryRelativeDataset(train_wf, train_labels)
        test_dataset  = DroneTrajectoryRelativeDataset(test_wf,  test_labels)

    else:
        # --- CNN / CNN-LSTM path: robust-scale time series, then window ---
        print("📏 Fitting robust scaler on training features only...")
        robust_stats = fit_robust_feature_scaler(train_data)
        train_data = [transform_robust_features(f, robust_stats) for f in train_data]
        test_data  = [transform_robust_features(f, robust_stats) for f in test_data]

        if robust_stats_path is not None:
            save_robust_feature_scaler(robust_stats, robust_stats_path)
            print(f"✅ Saved robust scaler stats: {robust_stats_path}")

        train_dataset = DroneTrajectoryDataset(train_data, train_labels, window_size, step_size)
        test_dataset  = DroneTrajectoryDataset(test_data,  test_labels,  window_size, step_size)

    print(f"✅ Tensor Loading Complete - Training Tensors: {len(train_dataset)} | Test Tensors: {len(test_dataset)}\n")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
