"""
SVM-pipeline DWT features + LightGBM classifier.

Pipeline (same feature extraction as drone_classifier_svm, classifier replaced):
  SITL logs
  → signal_processor.process_{px4,ardu}_flight_data   (11-col kinematic features)
  → extract_flight_segments                            (turn / straight / takeoff / landing)
  → z-score per channel within each segment            (scale-invariant)
  → train_dwt_svm.extract_dwt_features                 (DWT db4 level=3)
  → LightGBM

Feature dimension: 11 channels × 3 DWT coefficients × 8 stats = 264

Channel indices (signal_processor.FEATURE_MAP):
  0:Altitude  1:Heading  2:VZ  3:XY-Speed  4:AZ  5:XY-Accel
  6:JZ  7:XY-Jerk  8:Curvature  9:YawRate  10:SlipRate
"""

import sys
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

# ── svm-pipeline imports ──────────────────────────────────────────────────────
SVM_DIR = Path(__file__).parent.parent / "drone_classifier_svm"
sys.path.insert(0, str(SVM_DIR))

from signal_processor import process_px4_flight_data, process_ardu_flight_data
from train_dwt_svm import extract_dwt_features as svm_extract_dwt

# ── constants ─────────────────────────────────────────────────────────────────
PX4_FOLDER  = "../../data/px4_logs"
ARDU_FOLDER = "../../data/ardu_logs"
TEST_RATIO  = 0.2
WAVELET     = "db4"
LEVEL       = 3
FEAT_DIM    = 264   # 11 ch × 3 coeff × 8 stats
SEG_TYPES   = ("turn",)   # which segment types to use; add "straight" etc. if desired
MIN_SEG_LEN = 25          # minimum samples per segment


def zscore_matrix(mat, eps=1e-8):
    """Z-score each column of mat independently."""
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)
    return (mat - mean) / (std + eps)


def extract_features(segment_features):
    """
    Z-score all 11 channels, then call svm_extract_dwt.
    Returns float32 vector of length 264.
    """
    normed = zscore_matrix(segment_features.astype(np.float64))
    return svm_extract_dwt(normed, waveletname=WAVELET, level=LEVEL).astype(np.float32)


def load_data(file_list, seg_types=SEG_TYPES):
    """Extract DWT features from SITL logs; return (X, y)."""
    X, y = [], []
    for path, label in file_list:
        fn = process_px4_flight_data if label == 0 else process_ardu_flight_data
        result = fn(str(path))
        if result is None or result[2] is None:
            continue
        _, _, feat_full, segments, _ = result
        if feat_full is None or len(feat_full) < MIN_SEG_LEN:
            continue

        for stype in seg_types:
            seg_list = segments.get(stype) or []
            if not isinstance(seg_list, list):
                seg_list = [seg_list]
            for seg in seg_list:
                if seg is None:
                    continue
                feats = seg.get("features")
                if feats is None or len(feats) < MIN_SEG_LEN:
                    continue
                X.append(extract_features(feats))
                y.append(label)

    return (np.array(X, dtype=np.float32) if X else np.empty((0, FEAT_DIM))), np.array(y)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    model_out = f"svmpipe_lgbm_{ts}.pkl"

    print(f"\n{'='*70}")
    print(f"  SVM-pipeline DWT + LightGBM  ({ts})")
    print(f"  Features: signal_processor (11-col) → z-score → svm DWT (264-dim)")
    print(f"  Segments: {SEG_TYPES}")
    print(f"{'='*70}\n")

    px4_files  = [(p, 0) for p in sorted(Path(PX4_FOLDER).glob("*.ulg"))]
    ardu_files = [(p, 1) for p in sorted(Path(ARDU_FOLDER).glob("*.bin"))]
    random.shuffle(px4_files); random.shuffle(ardu_files)

    px4_test  = max(1, int(len(px4_files)  * TEST_RATIO)) if len(px4_files)  > 1 else 0
    ardu_test = max(1, int(len(ardu_files) * TEST_RATIO)) if len(ardu_files) > 1 else 0

    train_files = px4_files[px4_test:]  + ardu_files[ardu_test:]
    test_files  = px4_files[:px4_test]  + ardu_files[:ardu_test]

    print(f"Files — train: {len(train_files)}  test: {len(test_files)}")
    print("⏳ Extracting features (train)...")
    X_train, y_train = load_data(train_files)
    print("⏳ Extracting features (test)...")
    X_test,  y_test  = load_data(test_files)

    print(f"\nSegments — train: {len(X_train)}  test: {len(X_test)}")
    for lbl, name in [(0, "PX4"), (1, "ArduPilot")]:
        cnt = int((y_train == lbl).sum())
        print(f"  {name}: {cnt} ({100*cnt/max(len(y_train),1):.1f}%)")

    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=5,
        reg_alpha=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ],
    )

    y_pred = clf.predict(X_test)
    print(f"\nTest accuracy (SITL): {accuracy_score(y_test, y_pred)*100:.1f}%")
    print(classification_report(y_test, y_pred, target_names=["PX4", "ArduPilot"]))

    with open(model_out, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved: {model_out}")

    feat_names = [
        f"ch{c}_{['cA3','cD3','cD2'][k]}_{s}"
        for c in range(11)
        for k in range(3)
        for s in ["mean","std","energy","max","min","kurt","peak_max","peak_min"]
    ]
    ch_map = {i: n for i, n in enumerate(
        ["Altitude","Heading","VZ","XY-Speed","AZ","XY-Accel","JZ","XY-Jerk","Curvature","YawRate","SlipRate"]
    )}
    importances = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nTop-15 features:")
    for name, imp in importances[:15]:
        ch_idx = int(name[2:].split("_")[0])
        print(f"  {imp:5.0f}  {ch_map[ch_idx]}__{name.split('_',2)[2]}")


if __name__ == "__main__":
    main()
