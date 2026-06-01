"""
Evaluate a trained DiversifyFlight checkpoint on real flight CSVs.
Usage:
  python eval_diversify.py                           # auto-picks latest diversify_feat7_*.pt
  python eval_diversify.py path/to/model.pt
  python eval_diversify.py model.pt --threshold 15.0
  python eval_diversify.py model.pt --cal-files 20   # SITL files per class for centroids + threshold
"""
import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import torch
import wandb

sys.path.insert(0, str(Path(__file__).parent))
import train_diversify as _td
from train_diversify import (
    DiversifyFlight, DEVICE,
    build_knn_bank, calibrate_threshold, evaluate_realflight,
    load_sitl_windows, _make_loader,
    PX4_FOLDER, ARDU_FOLDER,
    WANDB_PROJECT,
)
from sweep_diversify import _git_sha

REALFLIGHT_DIR = Path(__file__).parent.parent.parent / "data/realflight"


def main():
    parser = argparse.ArgumentParser(description="Evaluate DiversifyFlight checkpoint on real flights")
    parser.add_argument("model", nargs="?", default=None,
                        help=".pt checkpoint path (default: latest diversify_feat7_*.pt)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Centroid-distance OOD threshold (auto-calibrated from SITL val set if omitted)")
    parser.add_argument("--cal-files", type=int, default=10,
                        help="SITL files per class used for centroids + auto-calibration (default: 10)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip Weights & Biases logging")
    args = parser.parse_args()

    # ── find checkpoint ──────────────────────────────────────────────────────
    if args.model is None:
        pts = sorted(Path(".").glob("diversify_feat7_*.pt"))
        if not pts:
            pts = sorted(Path(".").glob("diversify_*.pt"))
        if not pts:
            raise FileNotFoundError("No .pt checkpoint found. Pass path explicitly.")

        print(f"\nAvailable checkpoints ({len(pts)}):")
        for i, p in enumerate(pts):
            print(f"  [{i}] {p.name}")
        default_idx = len(pts) - 1
        while True:
            try:
                raw = input(f"Select checkpoint [0-{len(pts)-1}, default={default_idx}]: ").strip()
                if raw == "":
                    idx = default_idx
                else:
                    idx = int(raw)
                if 0 <= idx < len(pts):
                    model_path = pts[idx]
                    break
                print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a valid number.")
    else:
        model_path = Path(args.model)
    print(f"Model: {model_path}")

    # ── wandb init ───────────────────────────────────────────────────────────
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"eval_{model_path.stem}",
        job_type="eval",
        mode="disabled" if args.no_wandb else None,
        config={
            "model_path": str(model_path),
            "cal_files":  args.cal_files,
            "threshold_override": args.threshold,
            "git_sha":    _git_sha(),
        },
    )
    if not args.no_wandb:
        # Snapshot all .py in this dir so the run is reproducible even with
        # uncommitted changes in the working tree.
        run.log_code(str(Path(__file__).parent))

    # ── load model ───────────────────────────────────────────────────────────
    sd = torch.load(model_path, map_location="cpu")
    _td.LATENT_DOMAIN_N = sd["dclassifier.fc.weight"].shape[0]
    print(f"Checkpoint LATENT_DOMAIN_N={_td.LATENT_DOMAIN_N}")
    model = DiversifyFlight().to(DEVICE)
    model.load_state_dict(sd)
    model.eval()

    # ── build calibration set (SITL) ─────────────────────────────────────────
    n = args.cal_files
    px4_files  = [(p, 1) for p in sorted(Path(PX4_FOLDER).glob("*.ulg"))[:n]]
    ardu_files = [(p, 0) for p in sorted(Path(ARDU_FOLDER).glob("*.bin"))[:n]]
    print(f"Loading {len(px4_files)} PX4 + {len(ardu_files)} Ardu SITL files for centroids/threshold ...",
          flush=True)
    cal_ds = load_sitl_windows(px4_files + ardu_files)
    cal_ld = _make_loader(cal_ds, shuffle=False)

    # ── kNN feature bank ─────────────────────────────────────────────────────
    banks, _ = build_knn_bank(model, cal_ld)

    # ── threshold ────────────────────────────────────────────────────────────
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Threshold: {threshold:.3f} (provided)")
    else:
        threshold = calibrate_threshold(model, cal_ld, banks)

    # ── evaluate real flights (suppress verbose processor logs) ──────────────
    csv_files = [p for p in sorted(REALFLIGHT_DIR.glob("*.csv"))
                 if "_raw" not in p.name]
    print(f"\nEvaluating {len(csv_files)} real flight files  (threshold={threshold:.3f}) ...", flush=True)
    with open(os.devnull, "w") as devnull, \
         contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        results = evaluate_realflight(model, csv_files, banks, threshold)

    # ── results table ────────────────────────────────────────────────────────
    COL = {"ArduPilot": "\033[94m", "PX4": "\033[91m", "Unknown": "\033[93m",
           "RESET": "\033[0m", "HEADER": "\033[1m"}

    hdr = f"{'File':<52} {'Pred':>10}  {'PX4%':>6}  {'kNN':>7}  {'Rej':>10}  {'Win':>5}"
    sep = "─" * len(hdr)
    print(f"\n{COL['HEADER']}{hdr}{COL['RESET']}")
    print(sep)
    for r in results:
        pred  = r["prediction"]
        px4p  = f"{r['px4_prob']*100:.1f}%" if r["px4_prob"] is not None else "  n/a"
        rej   = f"{r['n_rejected']}/{r['n_windows']}({r['reject_rate']*100:.0f}%)"
        color = COL[pred]
        print(f"  {r['file']:<50}  {color}{pred:>10}{COL['RESET']}"
              f"  {px4p:>6}  {r['knn_dist']:>7.3f}  {rej:>10}  {r['n_windows']:>5}")
    print(sep)

    ardu    = sum(1 for r in results if r["prediction"] == "ArduPilot")
    px4     = sum(1 for r in results if r["prediction"] == "PX4")
    unknown = sum(1 for r in results if r["prediction"] == "Unknown")
    print(f"  {len(results)} files   "
          f"{COL['ArduPilot']}ArduPilot:{ardu}{COL['RESET']}  "
          f"{COL['PX4']}PX4:{px4}{COL['RESET']}  "
          f"{COL['Unknown']}Unknown:{unknown}{COL['RESET']}"
          f"   (threshold={threshold:.3f})")

    out = Path(model_path.stem + "_eval.json")
    with open(out, "w") as f:
        json.dump({"threshold": threshold, "results": results}, f, indent=2)
    print(f"\nResults → {out}")

    # ── wandb logging ────────────────────────────────────────────────────────
    wandb.run.summary.update({
        "ood/threshold":        threshold,
        "realflight/total":     len(results),
        "realflight/ardupilot": ardu,
        "realflight/px4":       px4,
        "realflight/unknown":   unknown,
    })
    realflight_table = wandb.Table(
        columns=["file", "prediction", "px4_prob", "knn_dist",
                 "n_windows", "n_accepted", "n_rejected", "n_px4",
                 "n_ardu", "reject_rate"],
        data=[[r["file"], r["prediction"], r.get("px4_prob"), r["knn_dist"],
               r["n_windows"], r["n_accepted"], r["n_rejected"],
               r["n_px4"], r["n_ardu"], r["reject_rate"]] for r in results],
    )
    wandb.log({"realflight/results": realflight_table})

    artifact = wandb.Artifact(
        name=f"eval_{model_path.stem}",
        type="evaluation",
        metadata={"model_path": str(model_path), "threshold": threshold,
                  "git_sha": _git_sha()},
    )
    artifact.add_file(str(out))
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
