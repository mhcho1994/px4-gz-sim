"""
Sweep launcher for DIVERSIFY hyperparameter search.

Modes
-----
Training sweep  (default)
  Bayes search over invariance/discriminability knobs + OOD params.
  Registers a new sweep and starts one agent:
    python3 sweep_diversify.py
  Attach an additional parallel agent to an existing sweep:
    python3 sweep_diversify.py --sweep-id <ID>
  Limit trials per agent process:
    python3 sweep_diversify.py --count 10

OOD-only sweep  (--ood)
  Grid search over ood_pctile × knn_k on a fixed best-model checkpoint.
  No retraining — just varies calibration parameters and evaluates real flights.
    python3 sweep_diversify.py --ood
    python3 sweep_diversify.py --ood path/to/model.pt
    python3 sweep_diversify.py --ood --sweep-id <ID>
    python3 sweep_diversify.py --ood --cal-files 15
"""
import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path

import torch
import wandb

sys.path.insert(0, str(Path(__file__).parent))
import train_diversify as td
from train_diversify import (
    DiversifyFlight, DEVICE,
    build_knn_bank, calibrate_threshold, evaluate_realflight,
    compute_rejection_rate, load_sitl_windows, _make_loader,
    PX4_FOLDER, ARDU_FOLDER, _filename_label,
)

REALFLIGHT_DIR = Path(__file__).parent.parent.parent / "data/realflight"
OOD_CAL_FILES  = 10   # SITL files per class used for kNN bank + threshold


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


# ── training sweep config ─────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "bayes",
    "run_cap": 100,
    "metric": {
        "name": "realflight/accuracy",
        "goal": "maximize",
        # optimizes real-flight ArduPilot/PX4 accuracy directly (catches PX4 collapse
        # that acc/val misses). Cross-check ood/sitl_false_reject on the dashboard.
    },
    "parameters": {
        # invariance knobs — ranges lowered to fix PX4 collapse (ArduPilot signal
        # was being erased by over-strong GRL). Weaker invariance = more discriminable.
        "alpha": {
            "distribution": "log_uniform_values",
            "min": 0.02, "max": 1.0,
        },
        "alpha1": {
            "distribution": "log_uniform_values",
            "min": 0.02, "max": 1.0,
        },
        "lam": {
            "values": [0.0, 0.1, 0.3, 0.5, 1.0],
        },
        # domain structure (K=7,10 worked best; drop K=3 which collapsed hardest)
        "latent_domain_n": {
            "values": [5, 7, 10],
        },
        # discriminability / featurizer knobs
        "lr_decay1": {
            # biased low: weak front-end eraser preserves class-discriminative features
            "values": [0.005, 0.01, 0.02, 0.05, 0.1],
        },
        "lr": {
            "distribution": "log_uniform_values",
            "min": 0.0001, "max": 0.005,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 0.00001, "max": 0.001,
        },
        # OOD detection knobs
        "ood_pctile": {
            "values": [90, 95, 99],
        },
        "knn_k": {
            "values": [3, 5, 10, 20],
        },
        # attention pooling
        "attn_hidden": {
            "values": [32, 64, 128],
        },
    },
}

# ── OOD-only sweep config ─────────────────────────────────────────────────────
OOD_SWEEP_CONFIG = {
    "method": "random",
    "run_cap": 100,
    "metric": {
        "name": "realflight/accuracy",
        "goal": "maximize",
    },
    "parameters": {
        "ood_pctile": {
            "distribution": "int_uniform",
            "min": 80, "max": 99,
        },
        "knn_k": {
            "distribution": "int_uniform",
            "min": 1, "max": 30,
        },
        # count-based MIL quorum
        "mil_min_valid": {        # small absolute statistical floor
            "distribution": "int_uniform",
            "min": 1, "max": 5,
        },
        "mil_min_frac": {         # length-elastic quorum (main criterion)
            "distribution": "uniform",
            "min": 0.05, "max": 0.35,
        },
    },
}


# ── training sweep ────────────────────────────────────────────────────────────

def _apply_sweep_config():
    """Override train_diversify module globals from wandb.config before training."""
    cfg = wandb.config
    td.ALPHA           = cfg.get("alpha",           td.ALPHA)
    td.ALPHA1          = cfg.get("alpha1",          td.ALPHA1)
    td.LAM             = cfg.get("lam",             td.LAM)
    td.LR              = cfg.get("lr",              td.LR)
    td.LATENT_DOMAIN_N = cfg.get("latent_domain_n", td.LATENT_DOMAIN_N)
    td.LR_DECAY1       = cfg.get("lr_decay1",       td.LR_DECAY1)
    td.LR_DECAY2       = cfg.get("lr_decay2",       td.LR_DECAY2)
    td.WEIGHT_DECAY    = cfg.get("weight_decay",    td.WEIGHT_DECAY)
    td.OOD_PCTILE      = cfg.get("ood_pctile",      td.OOD_PCTILE)
    td.KNN_K           = cfg.get("knn_k",           td.KNN_K)
    td.ATTN_HIDDEN     = cfg.get("attn_hidden",     td.ATTN_HIDDEN)
    td.MIL_MIN_VALID   = cfg.get("mil_min_valid",   td.MIL_MIN_VALID)
    td.MIL_MIN_FRAC    = cfg.get("mil_min_frac",    td.MIL_MIN_FRAC)
    td.GIT_SHA         = _git_sha()


def sweep_train():
    """Entry point called by the wandb agent for each training trial."""
    wandb.init()          # must come first so wandb.config is readable
    _apply_sweep_config() # overrides td globals from sweep config
    td.main()             # td.main() sees run already active and skips re-init


# ── OOD-only sweep ────────────────────────────────────────────────────────────

_ood_model_path: Path | None = None
_ood_model: DiversifyFlight | None = None


def _load_model_cached(model_path: Path) -> DiversifyFlight:
    """Load checkpoint once per process; return cached instance on repeat calls."""
    global _ood_model_path, _ood_model
    if _ood_model_path == model_path and _ood_model is not None:
        return _ood_model
    sd = torch.load(model_path, map_location="cpu")
    td.LATENT_DOMAIN_N = sd["dclassifier.fc.weight"].shape[0]
    print(f"  LATENT_DOMAIN_N={td.LATENT_DOMAIN_N} (from checkpoint)")
    model = DiversifyFlight().to(DEVICE)
    model.load_state_dict(sd)
    model.eval()
    _ood_model_path = model_path
    _ood_model = model
    return model


def fetch_best_model() -> Path:
    """
    Query W&B for the best finished training run and download its model artifact.

    Ranking: fewest realflight/unknown (best classification coverage).
    Fallback: highest best/val_acc if no run has real-flight metrics.
    Last resort: most recent local diversify_feat7_*.pt file.
    """
    api    = wandb.Api()
    entity = api.viewer.entity
    print(f"Querying W&B  {entity}/{td.WANDB_PROJECT} ...")

    runs = list(api.runs(
        f"{entity}/{td.WANDB_PROJECT}",
        filters={"jobType": "train", "state": "finished"},
    ))
    if not runs:
        raise RuntimeError(f"No finished training runs in {td.WANDB_PROJECT}")

    rf_runs = [r for r in runs if r.summary.get("realflight/accuracy") is not None]
    if rf_runs:
        best      = max(rf_runs, key=lambda r: r.summary["realflight/accuracy"])
        score_str = f"realflight/accuracy={best.summary['realflight/accuracy']:.3f}"
    else:
        best      = max(runs, key=lambda r: r.summary.get("best/val_acc", 0.0))
        score_str = f"best/val_acc={best.summary.get('best/val_acc', 'n/a')}"

    print(f"Best run  : {best.name}  ({score_str})")

    artifacts = [a for a in best.logged_artifacts() if a.type == "model"]
    if artifacts:
        art = artifacts[-1]
        print(f"Artifact  : {art.name} v{art.version} — downloading ...")
        dl_dir = Path(art.download())
        pts = sorted(dl_dir.glob("*.pt"))
        if pts:
            return pts[0]
        print("  (no .pt in artifact — falling back to local)")

    local = sorted(Path(".").glob("diversify_feat7_*.pt"))
    if not local:
        local = sorted(Path(".").glob("diversify_*.pt"))
    if local:
        print(f"Fallback  : {local[-1].name}")
        return local[-1]

    raise FileNotFoundError(
        "No checkpoint found. Pass one explicitly: python3 sweep_diversify.py --ood path/to/model.pt"
    )


def ood_trial(model_path: Path):
    """Single OOD sweep trial: vary ood_pctile + knn_k, evaluate real flights."""
    wandb.init()
    cfg = wandb.config

    td.OOD_PCTILE    = int(cfg.ood_pctile)
    td.KNN_K         = int(cfg.knn_k)
    td.MIL_MIN_VALID = int(cfg.get("mil_min_valid", td.MIL_MIN_VALID))
    td.MIL_MIN_FRAC  = float(cfg.get("mil_min_frac", td.MIL_MIN_FRAC))

    model = _load_model_cached(model_path)

    px4_files  = [(p, 1) for p in sorted(Path(PX4_FOLDER).glob("*.ulg"))[:OOD_CAL_FILES]]
    ardu_files = [(p, 0) for p in sorted(Path(ARDU_FOLDER).glob("*.bin"))[:OOD_CAL_FILES]]
    cal_ds = load_sitl_windows(px4_files + ardu_files)
    cal_ld = _make_loader(cal_ds, shuffle=False)

    banks     = build_knn_bank(model, cal_ld)[0]
    threshold = calibrate_threshold(model, cal_ld, banks)
    sitl_fr   = compute_rejection_rate(model, cal_ld, banks, threshold)

    csv_files = [p for p in sorted(REALFLIGHT_DIR.glob("*.csv")) if "_raw" not in p.name]
    with open(os.devnull, "w") as _null, \
         contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
        results = evaluate_realflight(model, csv_files, banks, threshold)

    ardu    = sum(1 for r in results if r["prediction"] == "ArduPilot")
    px4     = sum(1 for r in results if r["prediction"] == "PX4")
    unknown = sum(1 for r in results if r["prediction"] == "Unknown")
    total   = len(results)

    labeled  = [(r, _filename_label(r["file"])) for r in results
                if _filename_label(r["file"]) is not None]
    correct  = sum(1 for r, gt in labeled if r["prediction"] == gt)
    accuracy = correct / len(labeled) if labeled else 0.0

    print(f"  pctile={td.OOD_PCTILE}  k={td.KNN_K}  "
          f"mil={td.MIL_MIN_VALID}/{td.MIL_MIN_FRAC:.2f}  "
          f"acc={accuracy*100:.1f}% ({correct}/{len(labeled)})  "
          f"Ardu={ardu}  PX4={px4}  Unk={unknown}  "
          f"sitl_fr={sitl_fr*100:.1f}%  thr={threshold:.3f}")

    wandb.log({
        "realflight/accuracy":     accuracy,
        "realflight/correct":      correct,
        "realflight/labeled":      len(labeled),
        "realflight/ardupilot":    ardu,
        "realflight/px4":          px4,
        "realflight/unknown":      unknown,
        "realflight/total":        total,
        "ood/threshold":           threshold,
        "ood/sitl_false_reject":   sitl_fr,
        "mil/min_valid":           td.MIL_MIN_VALID,
        "mil/min_frac":            td.MIL_MIN_FRAC,
    })
    wandb.run.summary.update({
        "realflight/accuracy":     accuracy,
        "ood/sitl_false_reject":   sitl_fr,
    })
    wandb.log({
        "realflight/results": wandb.Table(
            columns=["file", "ground_truth", "prediction", "correct",
                     "knn_dist", "n_windows", "reject_rate"],
            data=[[r["file"], _filename_label(r["file"]) or "Unknown",
                   r["prediction"], _filename_label(r["file"]) == r["prediction"],
                   r["knn_dist"], r["n_windows"], r["reject_rate"]] for r in results],
        )
    })
    wandb.finish()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    global OOD_CAL_FILES
    parser = argparse.ArgumentParser(description="DIVERSIFY W&B sweep launcher")
    parser.add_argument("model", nargs="?", default=None,
                        help="[--ood only] .pt checkpoint (default: auto-fetch best from W&B)")
    parser.add_argument("--ood", action="store_true",
                        help="Run OOD-only grid sweep (ood_pctile × knn_k) on a fixed model")
    parser.add_argument("--sweep-id", default=None,
                        help="Existing sweep ID to attach an agent to")
    parser.add_argument("--count", type=int, default=None,
                        help="Max trials for this agent process (default: run until sweep cap)")
    parser.add_argument("--cal-files", type=int, default=OOD_CAL_FILES,
                        help=f"[--ood only] SITL files per class for kNN calibration (default: {OOD_CAL_FILES})")
    args = parser.parse_args()

    if args.ood:
        # ── OOD-only sweep ────────────────────────────────────────────────────
        OOD_CAL_FILES = args.cal_files

        if args.model:
            model_path = Path(args.model)
            if not model_path.exists():
                raise FileNotFoundError(model_path)
        else:
            model_path = fetch_best_model()
        print(f"Checkpoint : {model_path}\n")
        _load_model_cached(model_path)   # pre-load; shared across all trials

        sweep_fn = lambda: ood_trial(model_path)
        config   = OOD_SWEEP_CONFIG
        label    = "OOD sweep"
    else:
        # ── training sweep ────────────────────────────────────────────────────
        sweep_fn = sweep_train
        config   = SWEEP_CONFIG
        label    = "training sweep"

    # ── one-time check before agent loop starts ───────────────────────────────
    if wandb.run is not None and sys.stdin.isatty():
        print(f"\nActive W&B run detected: {wandb.run.name}  ({wandb.run.url})")
        print("  [1] Stop existing run and start a new one  (default)")
        print("  [2] Add a parallel agent — open another terminal and run:")
        print(f"        python3 sweep_diversify.py --sweep-id {wandb.run.sweep_id or '<sweep-id>'}")
        if input("Choice [1/2]: ").strip() == "2":
            print("Exiting — start a new agent process manually.")
            return
        wandb.finish()

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(config, project=td.WANDB_PROJECT)
        print(f"Registered {label}: {sweep_id}")

    wandb.agent(sweep_id, function=sweep_fn,
                project=td.WANDB_PROJECT, count=args.count)


if __name__ == "__main__":
    main()
