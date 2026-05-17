"""
Real flight classification using SVM-pipeline DWT + LightGBM.

Uses signal_processor.extract_kinematic_features + extract_flight_segments
(same pipeline as SITL training) for consistent feature extraction.

Usage:
    python test_svmpipe_realflight.py [model.pkl]
    python test_svmpipe_realflight.py          # auto-selects latest svmpipe_lgbm_*.pkl
"""

import sys
import glob
import json
import pickle
from pathlib import Path

import numpy as np

SVM_DIR = Path(__file__).parent.parent / "drone_classifier_svm"
sys.path.insert(0, str(SVM_DIR))

from signal_processor import (
    extract_kinematic_features,
    extract_flight_segments,
)
from train_svmpipe_lgbm import extract_features, SEG_TYPES, MIN_SEG_LEN

REALFLIGHT_DIR = "/home/gayeonslee/FIRE/flightstack_sim/data/realflight"


# ── flexible CSV reader ───────────────────────────────────────────────────────
def load_csv_flight(csv_path):
    """
    Read a real-flight CSV and return (t, x, y, z, vx, vy, vz).
    Handles the various column naming conventions in our dataset.
    Velocity is either read directly (ROS odom) or derived from position.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = set(df.columns)

    # ── time ──────────────────────────────────────────────────────────────────
    if "bag_time_ns" in cols:
        t = df["bag_time_ns"].values / 1e9
    elif "mocap_time_s" in cols:
        t = df["mocap_time_s"].values
    elif "time_s" in cols:
        t = df["time_s"].values
    elif "timestamp" in cols:
        t = df["timestamp"].values
    else:
        raise ValueError(f"No time column found in {csv_path}")

    # ── position ──────────────────────────────────────────────────────────────
    if "x" in cols and "y" in cols and "z" in cols:
        x, y, z = df["x"].values, df["y"].values, df["z"].values
    elif "gt_x" in cols:
        x, y, z = df["gt_x"].values, df["gt_y"].values, df["gt_z"].values
    elif "gtx" in cols:
        x, y, z = df["gtx"].values, df["gty"].values, df["gtz"].values
    elif "x_smooth" in cols:
        x, y, z = df["x_smooth"].values, df["y_smooth"].values, df["z_smooth"].values
    elif "xsmooth" in cols:
        x, y, z = df["xsmooth"].values, df["ysmooth"].values, df["zsmooth"].values
    else:
        raise ValueError(f"No position columns found in {csv_path}")

    # ── velocity: direct or derived ───────────────────────────────────────────
    if "vx" in cols and "vy" in cols and "vz" in cols:
        vx, vy, vz = df["vx"].values, df["vy"].values, df["vz"].values
    else:
        vx = np.gradient(x, t)
        vy = np.gradient(y, t)
        vz = np.gradient(z, t)

    # ENU convention (z > 0 when flying up) — signal_processor expects z positive-up
    # If median z < 0, flip (NED → ENU)
    if np.median(z) < 0:
        z = -z; vz = -vz

    return t, x, y, z, vx, vy, vz


# ── split by missing-frame gaps ───────────────────────────────────────────────
def split_by_gaps(t, x, y, z, vx, vy, vz, max_gap_s=1.0, min_len=50):
    """Split into continuous segments where time gap < max_gap_s."""
    diffs = np.diff(t)
    splits = list(np.where(diffs > max_gap_s)[0] + 1)
    indices = [0] + splits + [len(t)]
    segs = []
    for a, b in zip(indices[:-1], indices[1:]):
        if b - a >= min_len:
            segs.append((t[a:b], x[a:b], y[a:b], z[a:b],
                         vx[a:b], vy[a:b], vz[a:b]))
    return segs


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else (
        sorted(glob.glob("svmpipe_lgbm_*.pkl")) or [None])[-1]
    if model_path is None:
        raise FileNotFoundError("No svmpipe_lgbm_*.pkl found. Run train_svmpipe_lgbm.py first.")

    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    print(f"\n[OK] Model: {model_path}")
    print(f"     Segments used during training: {SEG_TYPES}\n")

    csv_files = sorted(glob.glob(f"{REALFLIGHT_DIR}/*.csv"))
    print(f"Found {len(csv_files)} CSV files\n")

    all_results = []

    for csv_file in csv_files:
        fname = Path(csv_file).name
        try:
            t, x, y, z, vx, vy, vz = load_csv_flight(csv_file)
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")
            continue

        cont_segs = split_by_gaps(t, x, y, z, vx, vy, vz)
        if not cont_segs:
            print(f"[SKIP] {fname}: no valid continuous segments")
            continue

        for seg_idx, (ts, xs, ys, zs, vxs, vys, vzs) in enumerate(cont_segs):
            result = extract_kinematic_features(ts, xs, ys, zs, vxs, vys, vzs)
            if result is None:
                continue
            t_feat, feat_full = result

            if feat_full is None or len(feat_full) < MIN_SEG_LEN:
                continue

            segments, _ = extract_flight_segments(t_feat, feat_full)

            seg_features = []
            for stype in SEG_TYPES:
                seg_list = segments.get(stype) or []
                if not isinstance(seg_list, list):
                    seg_list = [seg_list]
                for seg in seg_list:
                    if seg is None or len(seg.get("features", [])) < MIN_SEG_LEN:
                        continue
                    seg_features.append(extract_features(seg["features"]))

            if not seg_features:
                print(f"  {fname} seg{seg_idx}: no {SEG_TYPES} segments found")
                continue

            feats = np.array(seg_features, dtype=np.float32)
            probs = clf.predict_proba(feats)
            ardu_prob = float(probs[:, 1].mean())
            prediction = "ArduPilot" if ardu_prob > 0.5 else "PX4"
            breakdown = ", ".join(f"{p:.2f}" for p in probs[:, 1])

            print(f"  {fname} seg{seg_idx}  "
                  f"ArduPilot={ardu_prob*100:.1f}%  → {prediction}  "
                  f"({len(seg_features)} segs: [{breakdown}])")

            all_results.append({
                "file": fname, "segment": seg_idx,
                "n_segs": len(seg_features),
                "ardu_prob": round(ardu_prob, 3),
                "prediction": prediction,
                "per_seg_ardu_prob": [round(float(p), 3) for p in probs[:, 1]],
            })

    print(f"\n{'='*70}")
    ardu = sum(1 for r in all_results if r["prediction"] == "ArduPilot")
    px4  = sum(1 for r in all_results if r["prediction"] == "PX4")
    print(f"Total: {len(all_results)} file-segments")
    print(f"  ArduPilot: {ardu}")
    print(f"  PX4:       {px4}")

    out = "svmpipe_realflight_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Saved: {out}")


if __name__ == "__main__":
    main()
