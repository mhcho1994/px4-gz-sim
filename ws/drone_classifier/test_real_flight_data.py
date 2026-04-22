"""
Real Flight Data Classification Test
Testing our trained PX4/ArduPilot classifier on actual _odom CSV log files
"""

import pandas as pd
import numpy as np
import torch
import glob
import json
from pathlib import Path
from model import DroneTrajectoryCNN, DroneTrajectoryCNNLSTM
from trajectory_processor import process_rosbag_flight_data

print("="*90)
print("REAL FLIGHT DATA CLASSIFICATION TEST")
print("="*90)

# ==========================================
# 1. Find available model files
# ==========================================
print("\nSearching for model files...")
cnn_files = sorted(glob.glob('drone_cnn_*.pth'))
cnn_lstm_files = sorted(glob.glob('drone_cnn_lstm_*.pth'))

print(f"\nAvailable CNN models ({len(cnn_files)}):")
for i, f in enumerate(cnn_files):
    print(f"  [{i}] {f}")

print(f"\nAvailable CNN-LSTM models ({len(cnn_lstm_files)}):")
for i, f in enumerate(cnn_lstm_files):
    print(f"  [{i}] {f}")

# ==========================================
# 2. Load trained models
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# User selects CNN model
while True:
    try:
        cnn_idx = int(input(f"\nSelect CNN model [0-{len(cnn_files)-1}]: "))
        if 0 <= cnn_idx < len(cnn_files):
            cnn_path = cnn_files[cnn_idx]
            break
        print("Invalid selection. Try again.")
    except ValueError:
        print("Please enter a valid number.")

# User selects CNN-LSTM model
while True:
    try:
        cnn_lstm_idx = int(input(f"Select CNN-LSTM model [0-{len(cnn_lstm_files)-1}]: "))
        if 0 <= cnn_lstm_idx < len(cnn_lstm_files):
            cnn_lstm_path = cnn_lstm_files[cnn_lstm_idx]
            break
        print("Invalid selection. Try again.")
    except ValueError:
        print("Please enter a valid number.")

model_cnn = DroneTrajectoryCNN(num_features=10).to(device)
model_cnn.load_state_dict(torch.load(cnn_path, map_location=device))
model_cnn.eval()
print(f"\n[OK] Loaded: CNN model ({cnn_path})")

model_cnn_lstm = DroneTrajectoryCNNLSTM(
    num_features=10, 
    lstm_hidden_size=32, 
    num_lstm_layers=1
).to(device)
model_cnn_lstm.load_state_dict(torch.load(cnn_lstm_path, map_location=device))
model_cnn_lstm.eval()
print(f"[OK] Loaded: CNN-LSTM model ({cnn_lstm_path})")

# ==========================================
# 3. Create windows from features
# ==========================================
def create_windows(features, kernel_size=100, step_size=50):
    """Create windowed tensors from normalized features"""
    if features is None or len(features) < kernel_size:
        return None
    
    windows = []
    for start_idx in range(0, len(features) - kernel_size, step_size):
        end_idx = start_idx + kernel_size
        windows.append(features[start_idx:end_idx])
    
    return np.array(windows) if windows else None

# ==========================================
# 4. Classification function
# ==========================================
def classify_windows(windows, models, device):
    """Classify windowed features using all 3 models"""
    if windows is None or len(windows) == 0:
        return None
    
    X = torch.from_numpy(windows).float().to(device)
    # Convert from (batch, time_steps, features) to (batch, features, time_steps)
    X = X.permute(0, 2, 1)
    results = {}
    
    with torch.no_grad():
        for model_name, model in models.items():
            out = model(X)
            prob = torch.softmax(out, dim=1)
            ardu_prob = prob[:, 1].mean().item()
            
            results[model_name] = {
                'prob_ardupilot': ardu_prob,
                'prob_px4': 1 - ardu_prob,
                'prediction': 'ArduPilot' if ardu_prob > 0.5 else 'PX4'
            }
    
    # Ensemble
    avg_ardu = np.mean([r['prob_ardupilot'] for r in results.values()])
    results['ensemble'] = {
        'prob_ardupilot': avg_ardu,
        'prob_px4': 1 - avg_ardu,
        'prediction': 'ArduPilot' if avg_ardu > 0.5 else 'PX4'
    }
    
    return results

# ==========================================
# 5. Process all CSV odom files
# ==========================================
print("\n" + "="*90)
print("FINDING AND PROCESSING CSV ODOM FILES")
print("="*90)

csv_files = sorted(glob.glob('/home/gayeonslee/FIRE/flightstack_sim/data/realflight/*.csv'))

print(f"\nFound {len(csv_files)} CSV odom files:")
for f in csv_files[:10]:
    print(f"   {Path(f).name}")
if len(csv_files) > 10:
    print(f"   ... and {len(csv_files)-10} more")

# Dict of models to test
models = {
    'cnn': model_cnn,
    'cnn_lstm': model_cnn_lstm,
}

classification_results = []

for csv_file in csv_files:  # Test all files
    print(f"\n{'='*90}")
    print(f"Processing: {Path(csv_file).name}")
    print(f"{'='*90}")
    
    # Load and process CSV file
    result = process_rosbag_flight_data(csv_file)
    if result is None:
        print("[FAIL] Failed to process CSV file")
        continue
    
    t_loc, traj_raw, t_resampled, traj_resampled, features, segments, spans = result
    print(f"[OK] Extracted {len(features)} feature samples")
    
    # Print feature statistics (especially jerk)
    feature_names = ['Altitude', 'Heading', 'Vertical Velocity', 'Horizontal Speed', 'Vertical Acceleration', 'Horizontal Acceleration', 'Vertical Jerk', 'Horizontal Jerk', 'Curvature', 'Yaw Rate']
    print(f"\nFeature Statistics:")
    for i, name in enumerate(feature_names):
        print(f"  {name:20s}: mean={features[:, i].mean():8.4f}, std={features[:, i].std():8.4f}, "
              f"min={features[:, i].min():8.4f}, max={features[:, i].max():8.4f}")
    
    # Create windows
    windows = create_windows(features)
    if windows is None:
        print("[FAIL] Failed to create windows")
        continue
    
    print(f"[OK] Created {len(windows)} windows")
    
    # Classify
    print(f"Running inference...")
    results = classify_windows(windows, models, device)
    
    if results is None:
        print("[FAIL] Classification failed")
        continue
    
    # Display results
    print(f"\nCLASSIFICATION RESULTS:")
    print(f"{'='*90}")
    
    for model_name, pred in results.items():
        if model_name == 'ensemble':
            print(f"\n[FINAL] {model_name.upper():15s}")
        else:
            print(f"\n{model_name.upper():15s}")
        print(f"   PX4:       {pred['prob_px4']*100:6.2f}%")
        print(f"   ArduPilot: {pred['prob_ardupilot']*100:6.2f}%")
        print(f"   → {pred['prediction']}")
    
    # Store result
    classification_results.append({
        'filename': Path(csv_file).name,
        'n_windows': len(windows),
        'ensemble_prediction': results['ensemble']['prediction'],
        'ensemble_ardupilot_prob': results['ensemble']['prob_ardupilot'],
        'cnn_prediction': results['cnn']['prediction'],
        'lstm_prediction': results['cnn_lstm']['prediction'],
    })

# ==========================================
# 6. Summary
# ==========================================
print("\n" + "="*90)
print("SUMMARY")
print("="*90)

if len(classification_results) > 0:
    df = pd.DataFrame(classification_results)
    print("\nResults:")
    print(df.to_string(index=False))
    
    ardu_count = (df['ensemble_prediction'] == 'ArduPilot').sum()
    px4_count = (df['ensemble_prediction'] == 'PX4').sum()
    
    print(f"\nEnsemble Predictions:")
    print(f"   ArduPilot: {ardu_count} files")
    print(f"   PX4:       {px4_count} files")
    
    # Save
    with open('real_flight_test_results.json', 'w') as f:
        json.dump({
            'test_date': '2026-03-30',
            'n_files': len(df),
            'results': classification_results,
            'summary': {
                'ardupilot': int(ardu_count),
                'px4': int(px4_count)
            }
        }, f, indent=2)
    print(f"\n[OK] Saved: real_flight_test_results.json")
else:
    print("\n[FAIL] No results")

print("\n" + "="*90)
