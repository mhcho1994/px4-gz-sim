"""
DIVERSIFY-based drone autopilot classifier  (PX4 vs ArduPilot)

Input : 2-second sliding windows of 7 kinematic features resampled to 50 Hz
        fixed shape (N_FEAT=7, WIN_LEN=100)
        Features: vh, speed_xy, ah, curvature, yaw_rate, yaw_angular_accel, speed×curvature

Class y : 0 = ArduPilot,  1 = PX4
Latent domain d' : auto-discovered (K=5) by DIVERSIFY cosine k-means
Strategy A : Sim2Real barrier — GRL adversarial domain confusion
Strategy B : time-scale / agility invariance — kinematic features already handle it
"""

import sys
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent))
from train_dwt_lgbm import PX4_FOLDER, ARDU_FOLDER, TEST_RATIO
from trajectory_processor import (
    process_px4_flight_data, process_ardu_flight_data, process_rosbag_flight_data)

# ── hyper-parameters ──────────────────────────────────────────────────────────
FEAT_HZ         = 50
WIN_SEC         = 2.0
WIN_LEN         = int(WIN_SEC * FEAT_HZ)   # 100 samples
HOP_LEN         = WIN_LEN // 2             # 50 samples = 50 % overlap
N_FEAT          = 7
NUM_CLASSES     = 2                        # 0=ArduPilot, 1=PX4
LATENT_DOMAIN_N = 5
BOTTLENECK_DIM  = 32
DIS_HIDDEN      = 64
CNN_CH          = 128                      # final CNN channel count
ALPHA           = 1.0
ALPHA1          = 1.0
LAM             = 0.0
LOCAL_EPOCH     = 3
MAX_EPOCH       = 50
LR              = 1e-3
LR_DECAY1       = 0.1
LR_DECAY2       = 1.0
WEIGHT_DECAY    = 5e-4
BETA1           = 0.9
BATCH_SIZE      = 128
SEED            = 42
MIN_WIN         = 30
REALFLIGHT_DIR  = Path("/home/gayeonslee/FIRE/flightstack_sim/data/realflight")

DEVICE = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Network modules
# ══════════════════════════════════════════════════════════════════════════════

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class FlightFeaturizer(nn.Module):
    """
    1D-CNN on fixed (B, N_FEAT, WIN_LEN) = (B, 7, 100) input.
      Conv1(7→32,  k=7) + MaxPool(2) → (B, 32, 50)
      Conv2(32→64, k=5) + MaxPool(2) → (B, 64, 25)
      Conv3(64→128,k=3)              → (B, 128, 25)
      AdaptiveAvgPool1d(1)           → (B, 128)
    """
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(N_FEAT, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, CNN_CH, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(CNN_CH), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.in_features = CNN_CH

    def forward(self, x):
        # x: (B, N_FEAT, WIN_LEN)
        x = self.block3(self.block2(self.block1(x)))  # (B, CNN_CH, L')
        return self.pool(x).squeeze(-1)               # (B, CNN_CH)


class FeatBottleneck(nn.Module):
    def __init__(self, in_dim, out_dim=BOTTLENECK_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return self.bn(self.fc(x))


class FeatClassifier(nn.Module):
    def __init__(self, n_classes, in_dim=BOTTLENECK_DIM):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class PrototypeClassifier(nn.Module):
    """
    Distance-based classifier: logit = -dist(x, prototype_k).
    Low max-logit (= large min-distance) → OOD signal.
    """
    def __init__(self, n_classes, in_dim=BOTTLENECK_DIM):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, in_dim))

    def forward(self, x):
        dist = torch.cdist(x, self.prototypes)  # (B, n_classes)
        return -dist


class Discriminator(nn.Module):
    def __init__(self, in_dim=BOTTLENECK_DIM, hidden=DIS_HIDDEN, n_out=LATENT_DOMAIN_N):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class FlightDataset:
    """
    Each item: (x, ctarget, dtarget, pctarget, pdtarget, index)
      x       : (N_FEAT, WIN_LEN) float32 tensor
      ctarget : class label  0=ArduPilot  1=PX4
      dtarget : original domain id (per-file, unused by algorithm but kept for parity)
      pctarget: -1 (unused)
      pdtarget: pseudo-domain, initialised 0, updated by DiversifyFlight.set_dlabel
      index   : integer index (needed by set_dlabel)
    """
    def __init__(self, x: list, labels: np.ndarray, dlabels: np.ndarray):
        self.x        = x                              # list of (N_FEAT, WIN_LEN) tensors
        self.labels   = labels.astype(np.int64)
        self.dlabels  = dlabels.astype(np.int64)
        self.pclabels = np.full(len(labels), -1, np.int64)
        self.pdlabels = np.zeros(len(labels), np.int64)

    def set_labels_by_index(self, tlabels, tindex, label_type):
        if label_type == 'pdlabel':
            self.pdlabels[tindex] = tlabels

    def __getitem__(self, idx):
        return (self.x[idx],
                int(self.labels[idx]),
                int(self.dlabels[idx]),
                int(self.pclabels[idx]),
                int(self.pdlabels[idx]),
                idx)

    def __len__(self):
        return len(self.labels)

    @classmethod
    def subset(cls, ds, indices):
        d = cls.__new__(cls)
        d.x        = [ds.x[i] for i in indices]
        d.labels   = ds.labels[indices]
        d.dlabels  = ds.dlabels[indices]
        d.pclabels = ds.pclabels[indices]
        d.pdlabels = ds.pdlabels[indices]
        return d


def _collate(batch):
    xs, cls_l, dom_l, pc_l, pd_l, idxs = zip(*batch)
    return (torch.stack(xs),
            torch.tensor(cls_l, dtype=torch.long),
            torch.tensor(dom_l, dtype=torch.long),
            torch.tensor(pc_l,  dtype=torch.long),
            torch.tensor(pd_l,  dtype=torch.long),
            np.array(idxs))


def _make_loader(ds, shuffle, batch_size=BATCH_SIZE):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, num_workers=0, collate_fn=_collate)


# ══════════════════════════════════════════════════════════════════════════════
# DIVERSIFY algorithm (self-contained, CPU-friendly)
# ══════════════════════════════════════════════════════════════════════════════

def _entropy_logits(logits):
    p = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-5), dim=1))


class DiversifyFlight(nn.Module):
    def __init__(self):
        super().__init__()
        fea_dim = FlightFeaturizer().in_features   # CNN_CH = 128

        self.featurizer     = FlightFeaturizer()
        # d-branch: domain-discovery (adversarial class confusion + pseudo-domain clf)
        self.dbottleneck    = FeatBottleneck(fea_dim)
        self.ddiscriminator = Discriminator(BOTTLENECK_DIM, DIS_HIDDEN, NUM_CLASSES)
        self.dclassifier    = FeatClassifier(LATENT_DOMAIN_N)
        # main branch: prototype class clf + GRL pseudo-domain adversary
        self.bottleneck     = FeatBottleneck(fea_dim)
        self.classifier     = PrototypeClassifier(NUM_CLASSES)
        self.discriminator  = Discriminator(BOTTLENECK_DIM, DIS_HIDDEN, LATENT_DOMAIN_N)
        # auxiliary branch: joint (pseudo-domain × class) classifier
        self.abottleneck    = FeatBottleneck(fea_dim)
        self.aclassifier    = FeatClassifier(NUM_CLASSES * LATENT_DOMAIN_N)

    # Phase 1 — auxiliary branch
    def update_a(self, batch, opt):
        x  = batch[0].to(DEVICE).float()
        c  = batch[1].to(DEVICE).long()   # class label
        pd = batch[4].to(DEVICE).long()   # pseudo-domain
        y  = pd * NUM_CLASSES + c
        z  = self.abottleneck(self.featurizer(x))
        loss = F.cross_entropy(self.aclassifier(z), y)
        opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

    # Phase 2 — d-branch: adversarial class-confusion + pseudo-domain classification
    def update_d(self, batch, opt):
        x  = batch[0].to(DEVICE).float()
        c  = batch[1].to(DEVICE).long()   # class label  → GRL adversary target
        pd = batch[4].to(DEVICE).long()   # pseudo-domain → dclassifier target
        z  = self.dbottleneck(self.featurizer(x))
        rev       = ReverseLayerF.apply(z, ALPHA1)
        disc_loss = F.cross_entropy(self.ddiscriminator(rev), c)
        cd        = self.dclassifier(z)
        cls_loss  = F.cross_entropy(cd, pd) + LAM * _entropy_logits(cd)
        loss = disc_loss + cls_loss
        opt.zero_grad(); loss.backward(); opt.step()
        return {'total': loss.item(), 'dis': disc_loss.item(), 'cls': cls_loss.item()}

    # Pseudo-domain label update via cosine k-means
    @torch.no_grad()
    def set_dlabel(self, loader):
        self.eval()
        feats, outputs, indices = [], [], []
        for batch in loader:
            x   = batch[0].to(DEVICE).float()
            idx = batch[5]
            z   = self.dbottleneck(self.featurizer(x))
            o   = self.dclassifier(z)
            feats.append(z.cpu())
            outputs.append(o.cpu())
            indices.append(idx if isinstance(idx, np.ndarray) else np.array(idx))

        all_fea = torch.cat(feats, 0)
        all_out = torch.cat(outputs, 0)
        all_idx = np.hstack(indices)

        all_fea = torch.cat([all_fea, torch.ones(len(all_fea), 1)], 1)
        all_fea = (all_fea.T / all_fea.norm(p=2, dim=1)).T.numpy()
        aff     = F.softmax(all_out, dim=1).numpy()
        K       = aff.shape[1]
        initc   = aff.T @ all_fea / (1e-8 + aff.sum(0)[:, None])
        pred    = cdist(all_fea, initc, 'cosine').argmin(1)
        aff2    = np.eye(K)[pred]
        initc   = aff2.T @ all_fea / (1e-8 + aff2.sum(0)[:, None])
        pred    = cdist(all_fea, initc, 'cosine').argmin(1)

        loader.dataset.set_labels_by_index(pred, all_idx, 'pdlabel')
        print(f"    Pseudo-domain dist: {dict(Counter(pred.tolist()))}")
        self.train()

    # Phase 3 — main branch: class clf + GRL pseudo-domain adversary
    def update(self, batch, opt):
        x  = batch[0].to(DEVICE).float()
        y  = batch[1].to(DEVICE).long()   # class label
        pd = batch[4].to(DEVICE).long()   # pseudo-domain
        z  = self.bottleneck(self.featurizer(x))
        cls_loss  = F.cross_entropy(self.classifier(z), y)
        rev       = ReverseLayerF.apply(z, ALPHA)
        disc_loss = F.cross_entropy(self.discriminator(rev), pd)
        loss = cls_loss + disc_loss
        opt.zero_grad(); loss.backward(); opt.step()
        return {'total': loss.item(), 'cls': cls_loss.item(), 'dis': disc_loss.item()}

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))

    def ood_score(self, x):
        """Min distance to nearest prototype. High value = likely OOD."""
        logits = self.predict(x)   # (B, n_classes) = -dist
        return (-logits).min(dim=1).values  # (B,) minimum distance


def _make_optimizers(model):
    opta = torch.optim.Adam([
        {'params': model.featurizer.parameters(),  'lr': LR_DECAY1 * LR},
        {'params': model.abottleneck.parameters(), 'lr': LR_DECAY2 * LR},
        {'params': model.aclassifier.parameters(), 'lr': LR_DECAY2 * LR},
    ], lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.9))
    optd = torch.optim.Adam([
        {'params': model.dbottleneck.parameters(),    'lr': LR_DECAY2 * LR},
        {'params': model.dclassifier.parameters(),    'lr': LR_DECAY2 * LR},
        {'params': model.ddiscriminator.parameters(), 'lr': LR_DECAY2 * LR},
    ], lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.9))
    opt = torch.optim.Adam([
        {'params': model.bottleneck.parameters(),    'lr': LR_DECAY2 * LR},
        {'params': model.classifier.parameters(),    'lr': LR_DECAY2 * LR},
        {'params': model.discriminator.parameters(), 'lr': LR_DECAY2 * LR},
    ], lr=LR, weight_decay=WEIGHT_DECAY, betas=(BETA1, 0.9))
    return opta, optd, opt


# ══════════════════════════════════════════════════════════════════════════════
# Data utilities
# ══════════════════════════════════════════════════════════════════════════════

def _zscore(feat):
    """Z-score each channel. feat: (N_FEAT, WIN_LEN)"""
    mean = feat.mean(axis=1, keepdims=True)
    std  = feat.std(axis=1,  keepdims=True)
    return (feat - mean) / (std + 1e-8)


def _slide_windows(feat_extracted):
    """
    Slide WIN_LEN-sample windows with HOP_LEN hop over feat_extracted (T, N_FEAT).
    Returns list of (N_FEAT, WIN_LEN) float32 tensors.
    """
    T = len(feat_extracted)
    if T < MIN_WIN:
        return []
    wins = []
    for s in range(0, T - WIN_LEN + 1, HOP_LEN):
        seg = feat_extracted[s:s + WIN_LEN].T.copy().astype(np.float32)  # (N_FEAT, WIN_LEN)
        seg = _zscore(seg)
        if np.any(~np.isfinite(seg)):
            continue
        wins.append(torch.tensor(seg, dtype=torch.float32))
    return wins


def load_sitl_windows(file_list):
    """
    Load SITL logs, slide windows over feat_extracted (already 50Hz, 7 features).
    file_list : list of (path, label)  label: 0=ArduPilot, 1=PX4
    Returns FlightDataset.
    """
    all_x, all_y, all_d = [], [], []
    for domain_id, (path, label) in enumerate(file_list):
        fn = process_px4_flight_data if label == 1 else process_ardu_flight_data
        result = fn(str(path))
        if result is None:
            continue
        feat_extracted = result[4]   # (T, 7) at 50Hz
        if feat_extracted is None or len(feat_extracted) < MIN_WIN:
            continue
        wins = _slide_windows(feat_extracted)
        for w in wins:
            all_x.append(w)
            all_y.append(label)
            all_d.append(domain_id)

    return FlightDataset(all_x,
                         np.array(all_y, dtype=np.int64),
                         np.array(all_d, dtype=np.int64))


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_accuracy(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        x = batch[0].to(DEVICE).float()
        y = batch[1]
        p = model.predict(x).argmax(1).cpu()
        y_t = y if isinstance(y, torch.Tensor) else torch.tensor(y)
        correct += (p == y_t).sum().item()
        total   += len(y_t)
    model.train()
    return correct / max(total, 1)


def evaluate_realflight(model, csv_files):
    """
    Classify each real-flight CSV using process_rosbag_flight_data (→ 50Hz feat7).
    Majority vote over windows in each file.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for csv_path in sorted(csv_files):
            fname = Path(csv_path).name
            segments = process_rosbag_flight_data(str(csv_path))
            if not segments:
                print(f"  [SKIP] {fname}  (no valid segments)")
                continue

            all_wins = []
            for seg_info in segments:
                feat = seg_info['data'][4]   # (T, 7) at 50Hz
                if feat is None or len(feat) < MIN_WIN:
                    continue
                all_wins.extend(_slide_windows(feat))

            if not all_wins:
                print(f"  [SKIP] {fname}  (no windows)")
                continue

            X      = torch.stack(all_wins).to(DEVICE)        # (N, N_FEAT, WIN_LEN)
            logits = model.predict(X)
            probs  = F.softmax(logits, dim=1).cpu().numpy()  # (N, 2)
            ood    = model.ood_score(X).cpu().numpy()        # (N,) min-dist
            p_px4  = float(probs[:, 1].mean())
            ood_mean = float(ood.mean())
            pred   = "PX4" if p_px4 > 0.5 else "ArduPilot"
            per_win = ", ".join(f"{p:.2f}" for p in probs[:, 1])
            print(f"  {fname:<50s}  PX4={p_px4*100:.1f}%  OOD={ood_mean:.3f}  → {pred}  "
                  f"[{len(all_wins)}w: {per_win}]")
            results.append({"file": fname,
                            "px4_prob": round(p_px4, 3),
                            "ood_score": round(ood_mean, 4),
                            "prediction": pred,
                            "n_windows": len(all_wins)})
    model.train()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"\n{'='*70}")
    print(f"  DIVERSIFY  feat7@50Hz  2s-windows  ({ts})")
    print(f"  WIN_LEN={WIN_LEN}  N_FEAT={N_FEAT}  BOTTLENECK={BOTTLENECK_DIM}")
    print(f"  LATENT_K={LATENT_DOMAIN_N}  epochs={MAX_EPOCH}×{LOCAL_EPOCH}  lr={LR}")
    print(f"  class: 0=ArduPilot  1=PX4")
    print(f"{'='*70}\n")

    # ── train/test split ──────────────────────────────────────────────────────
    px4_files  = [(p, 1) for p in sorted(Path(PX4_FOLDER).glob("*.ulg"))]
    ardu_files = [(p, 0) for p in sorted(Path(ARDU_FOLDER).glob("*.bin"))]
    random.shuffle(px4_files); random.shuffle(ardu_files)
    px4_te  = max(1, int(len(px4_files)  * TEST_RATIO)) if len(px4_files)  > 1 else 0
    ardu_te = max(1, int(len(ardu_files) * TEST_RATIO)) if len(ardu_files) > 1 else 0
    train_files = px4_files[px4_te:]  + ardu_files[ardu_te:]
    test_files  = px4_files[:px4_te]  + ardu_files[:ardu_te]
    print(f"Files — train PX4:{len(px4_files)-px4_te}  Ardu:{len(ardu_files)-ardu_te}"
          f"  test PX4:{px4_te}  Ardu:{ardu_te}")

    # ── feature extraction ────────────────────────────────────────────────────
    print("Loading train windows...")
    train_ds_full = load_sitl_windows(train_files)
    print("Loading test windows...")
    test_ds = load_sitl_windows(test_files)

    N   = len(train_ds_full)
    idx = np.random.permutation(N)
    val_n    = max(1, int(N * 0.15))
    val_ds   = FlightDataset.subset(train_ds_full, idx[:val_n])
    train_ds = FlightDataset.subset(train_ds_full, idx[val_n:])

    n_px4  = int((train_ds.labels == 1).sum())
    n_ardu = int((train_ds.labels == 0).sum())
    print(f"Windows — train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")
    print(f"  train  PX4:{n_px4} ({100*n_px4/max(len(train_ds),1):.1f}%)"
          f"  Ardu:{n_ardu} ({100*n_ardu/max(len(train_ds),1):.1f}%)")

    train_ld    = _make_loader(train_ds, shuffle=True)
    train_ns_ld = _make_loader(train_ds, shuffle=False)
    val_ld      = _make_loader(val_ds,   shuffle=False)
    test_ld     = _make_loader(test_ds,  shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = DiversifyFlight().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    model.train()
    opta, optd, opt = _make_optimizers(model)

    best_val_acc  = 0.0
    best_test_acc = 0.0
    model_path    = f"diversify_feat7_{ts}.pt"

    # ── training loop ─────────────────────────────────────────────────────────
    for rnd in range(MAX_EPOCH):
        print(f"\n──── Round {rnd+1:02d}/{MAX_EPOCH} ─────────────────────────────")

        # Phase 1 — auxiliary branch
        for _ in range(LOCAL_EPOCH):
            for b in train_ld: model.update_a(b, opta)

        # Phase 2 — d-branch
        d_tot = d_dis = d_cls = nb = 0
        for _ in range(LOCAL_EPOCH):
            for b in train_ld:
                s = model.update_d(b, optd)
                d_tot += s['total']; d_dis += s['dis']; d_cls += s['cls']; nb += 1
        print(f"  update_d  total={d_tot/nb:.4f}  dis={d_dis/nb:.4f}  cls={d_cls/nb:.4f}")

        # Pseudo-domain reassignment
        model.set_dlabel(train_ns_ld)

        # Phase 3 — main branch
        u_tot = u_cls = u_dis = nb = 0
        for _ in range(LOCAL_EPOCH):
            for b in train_ld:
                s = model.update(b, opt)
                u_tot += s['total']; u_cls += s['cls']; u_dis += s['dis']; nb += 1
        print(f"  update    total={u_tot/nb:.4f}  cls={u_cls/nb:.4f}  dis={u_dis/nb:.4f}")

        tr_acc  = eval_accuracy(model, train_ns_ld)
        val_acc = eval_accuracy(model, val_ld)
        te_acc  = eval_accuracy(model, test_ld)
        print(f"  Acc  train={tr_acc*100:.1f}%  val={val_acc*100:.1f}%"
              f"  SITL-test={te_acc*100:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = te_acc
            torch.save(model.state_dict(), model_path)
            print(f"  ★ saved  (val={val_acc*100:.1f}%)")

    print(f"\n  Best val={best_val_acc*100:.1f}%  → SITL test={best_test_acc*100:.1f}%")

    # ── reload best checkpoint ────────────────────────────────────────────────
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # ── SITL test report ──────────────────────────────────────────────────────
    print(f"\n{'='*70}\n  SITL Test Set")
    y_true, y_pred = [], []
    with torch.no_grad():
        for b in test_ld:
            x = b[0].to(DEVICE).float()
            p = model.predict(x).argmax(1).cpu().tolist()
            y_t = b[1].tolist() if isinstance(b[1], torch.Tensor) else list(b[1])
            y_true.extend(y_t); y_pred.extend(p)
    print(classification_report(y_true, y_pred, target_names=["ArduPilot", "PX4"]))

    # ── real flight evaluation ────────────────────────────────────────────────
    csv_files = [p for p in sorted(REALFLIGHT_DIR.glob("*.csv"))
                 if "_raw" not in p.name]
    print(f"\n{'='*70}\n  Real Flight Evaluation ({len(csv_files)} files)\n{'='*70}\n")
    results = evaluate_realflight(model, csv_files)

    ardu = sum(1 for r in results if r["prediction"] == "ArduPilot")
    px4  = sum(1 for r in results if r["prediction"] == "PX4")
    print(f"\n  Total:{len(results)}  ArduPilot:{ardu}  PX4:{px4}")

    out = f"diversify_feat7_realflight_{ts}.json"
    with open(out, "w") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"  Results: {out}\n  Model:   {model_path}")


if __name__ == "__main__":
    main()
