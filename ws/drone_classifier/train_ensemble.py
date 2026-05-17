"""
Ensemble trainer: DWT+LightGBM + Segment+LightGBM.

DWT feature extraction uses train_dwt_svm.extract_dwt_features (via train_dwt_lgbm wrapper).
Segment feature extraction uses train_segment_lgbm.segment_flight.

Both models are trained sequentially on the same SITL log split and saved as:
  dwt_lgbm_<ts>.pkl
  segment_lgbm_hover_<ts>.pkl  +  segment_features_hover_<ts>.json
"""

import random
import pickle
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report

from train_dwt_lgbm import (
    extract_turns, extract_dwt_features,
    MIN_SEG_LEN, PX4_FOLDER, ARDU_FOLDER, TEST_RATIO,
)
from train_segment_lgbm import (
    build_10col, segment_flight, segments_to_rows,
)

# ── shared train/test split ───────────────────────────────────────────────────
def make_split():
    px4_files  = [(p, 0) for p in sorted(Path(PX4_FOLDER).glob("*.ulg"))]
    ardu_files = [(p, 1) for p in sorted(Path(ARDU_FOLDER).glob("*.bin"))]
    random.shuffle(px4_files); random.shuffle(ardu_files)
    px4_test  = max(1, int(len(px4_files)  * TEST_RATIO)) if len(px4_files)  > 1 else 0
    ardu_test = max(1, int(len(ardu_files) * TEST_RATIO)) if len(ardu_files) > 1 else 0
    train = px4_files[px4_test:]  + ardu_files[ardu_test:]
    test  = px4_files[:px4_test]  + ardu_files[:ardu_test]
    return train, test


# ── DWT data loader ───────────────────────────────────────────────────────────
def load_dwt_data(file_list):
    from trajectory_processor import process_px4_flight_data, process_ardu_flight_data
    X, y = [], []
    for path, label in file_list:
        fn = process_px4_flight_data if label == 0 else process_ardu_flight_data
        result = fn(str(path))
        if result is None or result[4] is None:
            continue
        _, _, t_res, traj_res, feat7, _, _ = result
        if len(feat7) < MIN_SEG_LEN:
            continue
        for turn in extract_turns(t_res, feat7, traj_res, dt=1/50):
            if turn["n"] < MIN_SEG_LEN:
                continue
            X.append(extract_dwt_features(turn))
            y.append(label)
    return (np.array(X, dtype=np.float32) if X else np.empty((0, 72))), np.array(y)


# ── Segment data loader ───────────────────────────────────────────────────────
def load_segment_data(file_list):
    from trajectory_processor import process_px4_flight_data, process_ardu_flight_data
    rows = []
    for path, label in file_list:
        fn = process_px4_flight_data if label == 0 else process_ardu_flight_data
        result = fn(str(path))
        if result is None or result[4] is None:
            continue
        _, _, t_res, traj_res, feat7, _, _ = result
        if len(feat7) < 50:
            continue
        feat10 = build_10col(traj_res, feat7)
        segs   = segment_flight(t_res, feat10, dt=1/50)
        rows.extend(segments_to_rows(segs, label))
    return rows


# ── LightGBM trainer ──────────────────────────────────────────────────────────
def train_lgbm(X_train, y_train, X_test, y_test, tag):
    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=15,
        min_child_samples=5, reg_alpha=0.1, class_weight="balanced",
        random_state=42, n_jobs=4, verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    acc = accuracy_score(y_test, clf.predict(X_test)) * 100
    print(f"\n[{tag}] Test accuracy (SITL): {acc:.1f}%")
    print(classification_report(y_test, clf.predict(X_test), target_names=["PX4", "ArduPilot"]))
    return clf


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n{'='*70}")
    print(f"  Ensemble Trainer (DWT-LightGBM + Segment-LightGBM)  ({ts})")
    print(f"  DWT: train_dwt_svm.extract_dwt_features + z-score")
    print(f"{'='*70}\n")

    train_files, test_files = make_split()
    print(f"Files — train: {len(train_files)}  test: {len(test_files)}\n")

    # ── 1. DWT model ─────────────────────────────────────────────────────────
    print("── DWT + LightGBM ──────────────────────────────────────────────────")
    print("⏳ Extracting DWT features (train)...")
    X_dwt_tr, y_dwt_tr = load_dwt_data(train_files)
    print("⏳ Extracting DWT features (test)...")
    X_dwt_te, y_dwt_te = load_dwt_data(test_files)
    print(f"Turn segments — train: {len(X_dwt_tr)}  test: {len(X_dwt_te)}")

    clf_dwt = train_lgbm(X_dwt_tr, y_dwt_tr, X_dwt_te, y_dwt_te, "DWT")
    dwt_model_path = f"dwt_lgbm_{ts}.pkl"
    with open(dwt_model_path, "wb") as f:
        pickle.dump(clf_dwt, f)
    print(f"Saved: {dwt_model_path}")

    # ── 2. Segment model ─────────────────────────────────────────────────────
    print("\n── Segment + LightGBM (+hover) ─────────────────────────────────────")
    print("⏳ Segmenting train flights...")
    train_rows = load_segment_data(train_files)
    print("⏳ Segmenting test flights...")
    test_rows  = load_segment_data(test_files)

    df_tr = pd.DataFrame(train_rows).drop(columns=["seg_type"])
    df_te = pd.DataFrame(test_rows).drop(columns=["seg_type"])

    label_col    = "label"
    feature_cols = [c for c in df_tr.columns if c != label_col]
    df_tr = df_tr.dropna(axis=1, how="all")
    feature_cols = [c for c in feature_cols if c in df_tr.columns]
    df_tr[feature_cols] = df_tr[feature_cols].fillna(0.0)
    df_te[feature_cols] = df_te[feature_cols].reindex(columns=feature_cols).fillna(0.0)

    print(f"Segments — train: {len(df_tr)}  test: {len(df_te)}")
    for lbl, name in [(0, "PX4"), (1, "ArduPilot")]:
        cnt = int((df_tr[label_col] == lbl).sum())
        print(f"  {name}: {cnt} ({100*cnt/len(df_tr):.1f}%)")

    clf_seg = train_lgbm(
        df_tr[feature_cols].values, df_tr[label_col].values,
        df_te[feature_cols].values, df_te[label_col].values,
        "Segment",
    )
    seg_model_path = f"segment_lgbm_hover_{ts}.pkl"
    seg_feat_path  = f"segment_features_hover_{ts}.json"
    with open(seg_model_path, "wb") as f:
        pickle.dump(clf_seg, f)
    with open(seg_feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved: {seg_model_path}")
    print(f"Saved: {seg_feat_path}")

    print(f"\n{'='*70}")
    print(f"  Done.  Run test_ensemble_realflight.py to evaluate on real flights.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
