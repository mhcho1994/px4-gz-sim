"""
Sweep launcher for DIVERSIFY invariance-discriminability trade-off search.

Usage:
  # Register a new sweep and immediately start one agent:
  python3 sweep_diversify.py

  # Run agent against an existing sweep ID:
  python3 sweep_diversify.py --sweep-id <ID>

  # Limit number of trials for this agent process:
  python3 sweep_diversify.py --count 10
"""
import argparse
import subprocess
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).parent))
import train_diversify as td


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None

# ── sweep search space ────────────────────────────────────────────────────────
SWEEP_CONFIG = {
    "method": "bayes",
    "run_cap": 30,
    "metric": {
        "name": "acc/val",
        "goal": "maximize",
        # W&B dashboard: filter manually on (acc/val high) + (ood/sitl_false_reject low)
    },
    "parameters": {
        # invariance knobs
        "alpha": {
            "distribution": "log_uniform_values",
            "min": 0.1, "max": 3.0,
        },
        "alpha1": {
            "distribution": "log_uniform_values",
            "min": 0.1, "max": 3.0,
        },
        "lam": {
            "values": [0.0, 0.1, 0.3, 0.5, 1.0],
        },
        # domain structure
        "latent_domain_n": {
            "values": [3, 5, 7, 10],
        },
        # discriminability / featurizer knobs
        "lr_decay1": {
            # 0.005–0.01 range emphasized: weakens the front-end eraser
            "values": [0.005, 0.01, 0.05, 0.1, 0.3, 1.0],
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
    },
}


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
    td.GIT_SHA         = _git_sha()


def sweep_train():
    """Entry point called by the wandb agent for each trial."""
    _apply_sweep_config()
    td.main()


def main():
    parser = argparse.ArgumentParser(description="DIVERSIFY W&B sweep launcher")
    parser.add_argument("--sweep-id", default=None,
                        help="Existing sweep ID to attach an agent to")
    parser.add_argument("--count", type=int, default=None,
                        help="Max trials for this agent process (default: run until sweep cap)")
    args = parser.parse_args()

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(SWEEP_CONFIG, project=td.WANDB_PROJECT)
        print(f"Registered sweep: {sweep_id}")

    wandb.agent(sweep_id, function=sweep_train,
                project=td.WANDB_PROJECT, count=args.count)


if __name__ == "__main__":
    main()
