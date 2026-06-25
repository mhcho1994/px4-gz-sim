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

import os
import sys
import json
import random
from collections import Counter
from contextlib import contextmanager
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
import wandb

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
ATTN_HIDDEN     = 64                       # temporal attention hidden dim
ALPHA           = 1.0
ALPHA1          = 1.0
LAM             = 0.0
LOCAL_EPOCH     = 3
MAX_EPOCH       = 50
# CKPT_INTERVAL   = 10   # save a checkpoint every N rounds (for double-descent curve)
LR              = 1e-3
LR_DECAY1       = 0.1
LR_DECAY2       = 1.0
WEIGHT_DECAY    = 5e-4
BETA1           = 0.9
BATCH_SIZE      = 128
SEED            = 42
MIN_WIN         = 30
REALFLIGHT_DIR  = Path(__file__).parent.parent.parent / "data/realflight"
OOD_PCTILE = 95   # 95th-percentile of SITL-val kNN distances → threshold
KNN_K      = 5    # k-th nearest neighbor for OOD scoring

# ── Multiple Instance Learning (count-based MIL, Weidmann 2003; Foulds&Frank 2010)
# A flight (bag) is classified only if enough windows (instances) pass the OOD gate
# (= in-distribution / "valid features"); otherwise → Unknown.
# Two AND-ed conditions:
#   MIL_MIN_VALID : small absolute floor — need ≥N windows for any statistical decision
#   MIL_MIN_FRAC  : elastic quorum that scales with flight length (the main criterion;
#                   prevents short flights from being unfairly rejected by a fixed count)
MIL_MIN_VALID = 3     # statistical floor (keep small; fraction does the real gating)
MIL_MIN_FRAC  = 0.15  # require n_valid/n_total ≥ this (length-elastic quorum)

# ── Dual-path BN (AdaBN, Li et al. 2017) ──────────────────────────────────────
# OOD path always uses original SITL BatchNorm stats (eval mode) so the kNN-OOD
# gate keeps its SITL reference. Classification path can adapt BN to the real
# domain (unsupervised, label-free) to bridge the Sim2Real shift before classifying.
#   "off"        — classification also uses SITL BN (baseline)
#   "offline"    — BN stats estimated ONCE from a MIXED-class pool of real windows
#                  (works: pool contains both firmwares so class shift is preserved)
#   "per_flight" — BN stats from each flight's own windows (FAILS here: a flight is
#                  single-class, so BN centers away the class signal — kept for study)
ADABN_MODE    = "off"   # experiments showed AdaBN doesn't recover ArduPilot here;
                        # "offline"/"per_flight" kept available for study
ADABN_MIN_WIN = 4       # need ≥ this many windows for stable batch BN stats, else fall back

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


WANDB_PROJECT = "drone-firmware-classifier"
GIT_SHA       = None   # set by sweep_diversify._apply_sweep_config before each trial


def _filename_label(fname: str) -> str | None:
    """Derive ground-truth autopilot label from CSV filename, or None if unknown."""
    fl = fname.lower()
    if "px4" in fl:
        return "PX4"
    if "ardu" in fl:
        return "ArduPilot"
    return None


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

class TemporalAttentionPooling(nn.Module):
    """
    Learnable attention pooling over the temporal dimension.
    Input: (B, C, T) → Output: (B, C)
    """
    def __init__(self, in_channels, hidden_dim= ATTN_HIDDEN):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C)
        attn_weights = self.attention(x)  # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (B, T, 1)

        out = (x * attn_weights).sum(dim=1)  # (B, C)
        
        return out, attn_weights
    
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
        self.attn_pool = TemporalAttentionPooling(CNN_CH)
        self.in_features = CNN_CH

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))  # (B, CNN_CH, L')
        h, _ = self.attn_pool(x)                      # Attention Applied Pooling to get (B, CNN_CH)
        return h                                      # (B, CNN_CH)

    def forward_features(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        h, attn_weights = self.attn_pool(f3)          
        return f1, f2, h


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
        
        self.l1_bottleneck    = FeatBottleneck(32)
        self.l1_classifier    = FeatClassifier(NUM_CLASSES)

    # Phase 1 — auxiliary branch
    def update_a(self, batch, opt):
        x  = batch[0].to(DEVICE).float()
        c  = batch[1].to(DEVICE).long()   # class label
        pd = batch[4].to(DEVICE).long()   # pseudo-domain
        y  = pd * NUM_CLASSES + c
        # z  = self.abottleneck(self.featurizer(x))
        # loss = F.cross_entropy(self.aclassifier(z), y)
        # opt.zero_grad(); loss.backward(); opt.step()

        z_full = self.abottleneck(self.featurizer(x))
        loss_main = F.cross_entropy(self.aclassifier(z_full), y)
        
        # 2. 보조 학습: L1 층에서만 기종 차이를 억지로 학습하도록 멱살 잡기
        f1, _, _ = self.featurizer.forward_features(x)
        z_l1 = f1.mean(dim=-1)
        z_l1_out = self.l1_bottleneck(z_l1)
        loss_l1 = F.cross_entropy(self.l1_classifier(z_l1_out), c)
        
        # 3. 두 Loss를 더해서 역전파 (네트워크 전체 + L1 집중 학습)
        loss = loss_main + loss_l1
        
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

    def extract_z(self, x):
        """Bottleneck features (B, BOTTLENECK_DIM) — used by classifier."""
        return self.bottleneck(self.featurizer(x))

    def extract_ood_features(self, x):
        """
        Single-pass extraction of two kNN OOD levels:
          [0] block1 avg-pooled (B, 32) — pre-GRL, retains firmware micro-noise → Near-OOD (Cognipilot)
          [1] bottleneck        (B, 32) — post-GRL → Far-OOD + classification
        """
        f1, f2, h = self.featurizer.forward_features(x)
        z_l1 = f1.mean(dim=-1)       # (B, 32)
        z_bn  = self.bottleneck(h)   # (B, 32)
        return [z_l1, z_bn]
    
    def extract_multi_z(self, x):
        """
        Multi-level features for Mahalanobis OOD (Lee et al. 2018, NeurIPS).
          L0 block1     (B, 32)   shallow — least affected by GRL,
                                   retains firmware-specific micro-noise → Near-OOD
          L1 block2     (B, 64)   mid
          L2 bottleneck (B, 32)   deep, post-GRL → Far-OOD (physical abnormality)
        Spatial dim is globally-average-pooled (Lee 2018 convention).
        """
        f1, f2, h = self.featurizer.forward_features(x)
        z0 = f1.mean(dim=-1)               # (B, 32)
        z1 = f2.mean(dim=-1)               # (B, 64)
        z2 = self.bottleneck(h)            # (B, 32)
        return [z0, z1, z2]


def _make_optimizers(model):
    opta = torch.optim.Adam([
        {'params': model.featurizer.parameters(),  'lr': LR_DECAY1 * LR},
        {'params': model.abottleneck.parameters(), 'lr': LR_DECAY2 * LR},
        {'params': model.aclassifier.parameters(), 'lr': LR_DECAY2 * LR},
        {'params': model.l1_bottleneck.parameters(), 'lr': LR_DECAY2 * LR},
        {'params': model.l1_classifier.parameters(), 'lr': LR_DECAY2 * LR},
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


@torch.no_grad()
def build_knn_bank(model, loader):
    """
    Build two kNN feature banks from training data:
      banks[0]: block1 avg-pooled (Near-OOD — firmware micro-noise, pre-GRL)
      banks[1]: bottleneck        (Far-OOD — physical anomaly, post-GRL)
    """
    model.eval()
    feats_l1, feats_bn, labels = [], [], []
    for batch in loader:
        x = batch[0].to(DEVICE).float()
        y = batch[1]
        z_l1, z_bn = model.extract_ood_features(x)
        feats_l1.append(z_l1.cpu())
        feats_bn.append(z_bn.cpu())
        labels.append(y if isinstance(y, torch.Tensor) else torch.tensor(y))
    bank_l1     = torch.cat(feats_l1, dim=0)   # (N, 32)
    bank_bn     = torch.cat(feats_bn, dim=0)   # (N, 32)
    bank_labels = torch.cat(labels,   dim=0)   # (N,)
    print(f"  kNN banks: {len(bank_l1):,} vectors  L1(D={bank_l1.size(1)}) + bottleneck(D={bank_bn.size(1)})  k={KNN_K}")
    return [bank_l1, bank_bn], bank_labels


def _knn_score(z_list, banks):
    """
    오염된 L2(Bottleneck) 거리는 완전히 배제하고, 
    오직 GRL의 영향을 받지 않은 L1(z_list[0]) 특징 공간의 거리만 사용하여 OOD 점수를 계산합니다.
    """
    z_l1 = z_list[0]
    bank_l1 = banks[0]
    
    dists = torch.cdist(z_l1.cpu().float(), bank_l1.float())
    knn_d, _ = dists.topk(KNN_K, dim=1, largest=False)
    
    return knn_d[:, -1]

@torch.no_grad()
def calibrate_threshold(model, loader, banks):
    """95th-percentile of two-level kNN distances on SITL val set."""
    model.eval()
    dists = []
    for batch in loader:
        x = batch[0].to(DEVICE).float()
        d = _knn_score(model.extract_ood_features(x), banks)
        dists.extend(d.tolist())
    dists = np.array(dists)
    thr   = float(np.percentile(dists, OOD_PCTILE))
    print(f"  kNN-OOD  mean={dists.mean():.3f}  std={dists.std():.3f}"
          f"  {OOD_PCTILE}th-pct={thr:.3f}")
    return thr


@torch.no_grad()
def compute_rejection_rate(model, loader, banks, threshold):
    """Fraction of windows with two-level kNN distance > threshold (SITL false-rejection)."""
    model.eval()
    rejected = total = 0
    for batch in loader:
        x = batch[0].to(DEVICE).float()
        d = _knn_score(model.extract_ood_features(x), banks)
        rejected += (d > threshold).sum().item()
        total    += len(x)
    return rejected / max(total, 1)


@contextmanager
def _adabn_classification(model):
    """
    Per-flight AdaBN for the classification path (featurizer + bottleneck only).

    Sets those BatchNorm1d layers to train mode so a forward pass normalizes with
    the CURRENT batch (= this flight's windows) statistics instead of the stored
    SITL running stats. Original running_mean/var/num_batches are saved and
    restored on exit, so the OOD path (which calls extract_ood_features in eval
    mode) keeps its untouched SITL BN reference.
    """
    bn = [m for mod in (model.featurizer, model.bottleneck)
          for m in mod.modules() if isinstance(m, nn.BatchNorm1d)]
    saved = [(m.training,
              None if m.running_mean is None else m.running_mean.clone(),
              None if m.running_var  is None else m.running_var.clone(),
              None if m.num_batches_tracked is None else m.num_batches_tracked.clone())
             for m in bn]
    try:
        for m in bn:
            m.train()                # use batch statistics for this forward
        yield
    finally:
        for m, (tr, rm, rv, nb) in zip(bn, saved):
            m.train(tr)
            if rm is not None: m.running_mean.copy_(rm)
            if rv is not None: m.running_var.copy_(rv)
            if nb is not None: m.num_batches_tracked.copy_(nb)


def _classification_bn(model):
    return [m for mod in (model.featurizer, model.bottleneck)
            for m in mod.modules() if isinstance(m, nn.BatchNorm1d)]


@torch.no_grad()
def compute_adabn_stats(model, pool_x, bs=256):
    """
    Offline AdaBN: estimate classification-path BN stats from a MIXED-class pool of
    real windows (cumulative mean/var over the pool). Returns a list of (mean, var)
    aligned with _classification_bn(model); the model is left unmodified.
    """
    bn = _classification_bn(model)
    saved = [(m.running_mean.clone(), m.running_var.clone(),
              m.num_batches_tracked.clone(), m.momentum, m.training) for m in bn]
    for m in bn:
        m.reset_running_stats(); m.momentum = None; m.train()   # momentum=None → cumulative avg
    for i in range(0, len(pool_x), bs):
        model.bottleneck(model.featurizer(pool_x[i:i + bs].to(DEVICE)))
    adapted = [(m.running_mean.clone(), m.running_var.clone()) for m in bn]
    for m, (rm, rv, nb, mom, tr) in zip(bn, saved):
        m.running_mean.copy_(rm); m.running_var.copy_(rv)
        m.num_batches_tracked.copy_(nb); m.momentum = mom; m.train(tr)
    return adapted


@contextmanager
def _apply_adabn_stats(model, adapted):
    """Temporarily swap in pre-computed adapted BN stats (eval mode) for the
    classification path; restore the SITL stats on exit (OOD path unaffected)."""
    bn = _classification_bn(model)
    saved = [(m.running_mean.clone(), m.running_var.clone(), m.training) for m in bn]
    try:
        for m, (am, av) in zip(bn, adapted):
            m.running_mean.copy_(am); m.running_var.copy_(av); m.eval()
        yield
    finally:
        for m, (rm, rv, tr) in zip(bn, saved):
            m.running_mean.copy_(rm); m.running_var.copy_(rv); m.train(tr)


@contextmanager
def _adabn_var_only(model, x):
    """
    Per-flight VARIANCE-only AdaBN: adapt BN running_var to THIS flight's window
    statistics but keep the SITL running_mean. Since a flight is single-class, the
    mean carries the class signal — keeping SITL mean avoids the per-flight collapse
    while still rescaling to the real domain. (Two-pass approximation: the per-layer
    variance is captured under the flight's own mean, then applied with the SITL mean.)
    """
    bn = _classification_bn(model)
    saved = [(m.running_mean.clone(), m.running_var.clone(),
              m.num_batches_tracked.clone(), m.momentum, m.training) for m in bn]
    # pass 1: capture this flight's per-layer variance (cumulative over the batch)
    for m in bn:
        m.reset_running_stats(); m.momentum = None; m.train()
    with torch.no_grad():
        model.bottleneck(model.featurizer(x))
    flight_var = [m.running_var.clone() for m in bn]
    try:
        # pass 2: SITL mean + this-flight variance, eval mode
        for m, (rm, rv, nb, mom, tr), fv in zip(bn, saved, flight_var):
            m.running_mean.copy_(rm)        # keep SITL mean (preserves class signal)
            m.running_var.copy_(fv)         # adapt variance to this flight
            m.num_batches_tracked.copy_(nb); m.momentum = mom; m.eval()
        yield
    finally:
        for m, (rm, rv, nb, mom, tr) in zip(bn, saved):
            m.running_mean.copy_(rm); m.running_var.copy_(rv)
            m.num_batches_tracked.copy_(nb); m.momentum = mom; m.train(tr)


def evaluate_realflight(model, csv_files, banks, threshold):
    """
    Count-based Multiple Instance Learning (Weidmann 2003; Foulds & Frank 2010).
    Each flight is a bag; each window is an instance.

    Dual-path BN (ADABN_MODE):
      * OOD path        — always original SITL BN (eval) → extract_ood_features → kNN
      * Classification  — SITL BN ("off") / offline pooled AdaBN / per-flight AdaBN

    For each real-flight CSV:
      1. window → bottleneck z   (OOD path, SITL BN)
      2. kNN OOD score = distance to k-th nearest training neighbor
      3. Instance is VALID (in-distribution) if score <= threshold
      4. MIL quorum: assign a class only if n_valid >= MIL_MIN_VALID
         (and n_valid/n_total >= MIL_MIN_FRAC); else → Unknown.
      5. Among valid instances only, decide PX4 vs ArduPilot by majority.
    """
    model.eval()

    # ── pass 1: load every flight's windows; accumulate a mixed-class real pool ──
    flights = []   # (fname, X)
    for csv_path in sorted(csv_files):
        fname = Path(csv_path).name
        segments = process_rosbag_flight_data(str(csv_path))
        if not segments:
            print(f"  [SKIP] {fname}  (no valid segments)")
            continue
        all_wins = []
        for seg_info in segments:
            feat = seg_info['data'][4]
            if feat is None or len(feat) < MIN_WIN:
                continue
            all_wins.extend(_slide_windows(feat))
        if not all_wins:
            print(f"  [SKIP] {fname}  (no windows)")
            continue
        flights.append((fname, torch.stack(all_wins).to(DEVICE)))

    # ── offline AdaBN: estimate classification BN stats once from the mixed pool ─
    adapted = None
    with torch.no_grad():
        if ADABN_MODE == "offline":
            pool = torch.cat([X for _, X in flights], dim=0)
            if len(pool) >= ADABN_MIN_WIN:
                adapted = compute_adabn_stats(model, pool)

    # ── pass 2: per flight — OOD gate (SITL BN) + MIL + classification ───────────
    results = []
    with torch.no_grad():
        for fname, X in flights:
            # OOD path: original SITL BN (eval)
            z_list = model.extract_ood_features(X)      # [z_l1, z_bn] SITL-BN
            d      = _knn_score(z_list, banks)
            mask   = (d <= threshold)                   # valid (in-distribution) instances
            n_rej  = int((~mask).sum())
            n_acc  = int(mask.sum())
            d_mean = float(d.mean())
            n_tot  = len(X)

            quorum_ok = (n_acc >= MIL_MIN_VALID) and (n_acc / n_tot >= MIL_MIN_FRAC)
            if not quorum_ok:
                verdict, p_px4 = "Unknown", float("nan")
                n_px4 = n_ardu = 0
            else:
                # classification path BN per ADABN_MODE
                if ADABN_MODE == "offline" and adapted is not None:
                    with _apply_adabn_stats(model, adapted):
                        logits = model.classifier(model.bottleneck(model.featurizer(X)))
                elif ADABN_MODE == "per_flight" and n_tot >= ADABN_MIN_WIN:
                    with _adabn_classification(model):
                        logits = model.classifier(model.bottleneck(model.featurizer(X)))
                elif ADABN_MODE == "per_flight_var" and n_tot >= ADABN_MIN_WIN:
                    with _adabn_var_only(model, X):
                        logits = model.classifier(model.bottleneck(model.featurizer(X)))
                else:   # "off" or fallback
                    logits = model.classifier(z_list[1])
                probs_in = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
                p_px4    = float(probs_in.mean())
                n_px4    = int((probs_in > 0.5).sum())
                n_ardu   = n_acc - n_px4
                verdict  = "PX4" if n_px4 >= n_ardu else "ArduPilot"

            print(f"  {fname:<50s}  kNN={d_mean:.3f}  valid={n_acc}/{n_tot}"
                  f"({100*n_acc/n_tot:.0f}%)  → {verdict}")
            results.append({"file": fname, "prediction": verdict,
                "px4_prob":    round(p_px4, 3) if not np.isnan(p_px4) else None,
                "knn_dist":    round(d_mean, 4),
                "n_windows":   n_tot,
                "n_accepted":  n_acc,
                "n_rejected":  n_rej,
                "n_px4":       n_px4,
                "n_ardu":      n_ardu,
                "reject_rate": round(n_rej / n_tot, 4)})
    model.train()
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

    ts = datetime.now().strftime("%Y%m%d_%H%M")

    _cfg = {
        "feat_hz": FEAT_HZ, "win_sec": WIN_SEC, "win_len": WIN_LEN,
        "hop_len": HOP_LEN, "n_feat": N_FEAT, "num_classes": NUM_CLASSES,
        "latent_domain_n": LATENT_DOMAIN_N, "bottleneck_dim": BOTTLENECK_DIM,
        "dis_hidden": DIS_HIDDEN, "cnn_ch": CNN_CH, "attn_hidden": ATTN_HIDDEN,
        "alpha": ALPHA, "alpha1": ALPHA1, "lam": LAM,
        "local_epoch": LOCAL_EPOCH, "max_epoch": MAX_EPOCH,
        "lr": LR, "lr_decay1": LR_DECAY1, "lr_decay2": LR_DECAY2,
        "weight_decay": WEIGHT_DECAY, "beta1": BETA1,
        "batch_size": BATCH_SIZE, "seed": SEED, "min_win": MIN_WIN,
        "ood_pctile": OOD_PCTILE, "knn_k": KNN_K,
        "mil_min_valid": MIL_MIN_VALID, "mil_min_frac": MIL_MIN_FRAC,
        "adabn_mode": ADABN_MODE,
        "git_sha": GIT_SHA,
    }
    if wandb.run is None:
        run = wandb.init(
            project=WANDB_PROJECT,
            name=f"diversify_feat7_{ts}",
            job_type="train",
            config=_cfg,
        )
    else:
        # called from sweep agent — run already initialised, just sync config
        run = wandb.run
        wandb.config.update(_cfg, allow_val_change=True)
    run.log_code(str(Path(__file__).parent))

    print(f"\n{'='*70}")
    print(f"  DIVERSIFY  feat7@50Hz  2s-windows  ({ts})")
    print(f"  WIN_LEN={WIN_LEN}  N_FEAT={N_FEAT}  BOTTLENECK={BOTTLENECK_DIM}")
    print(f"  LATENT_K={LATENT_DOMAIN_N}  epochs={MAX_EPOCH}×{LOCAL_EPOCH}  lr={LR}")
    print(f"  alpha={ALPHA}  alpha1={ALPHA1}  lam={LAM}  lr_decay1={LR_DECAY1}")
    print(f"  class: 0=ArduPilot  1=PX4")
    print(f"  device: {DEVICE}")
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
    wandb.config.update({"model_params": n_params,
                         "n_train_windows": len(train_ds),
                         "n_val_windows":   len(val_ds),
                         "n_test_windows":  len(test_ds)})
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

        wandb.log({
            "round": rnd + 1,
            "loss/update_d_total": d_tot / nb,
            "loss/update_d_dis":   d_dis / nb,
            "loss/update_d_cls":   d_cls / nb,
            "loss/update_total":   u_tot / nb,
            "loss/update_cls":     u_cls / nb,
            "loss/update_dis":     u_dis / nb,
            "acc/train": tr_acc,
            "acc/val":   val_acc,
            "acc/test":  te_acc,
        }, step=rnd + 1)

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_test_acc = te_acc
            torch.save(model.state_dict(), model_path)
            print(f"  ★ saved  (val={val_acc*100:.1f}%)")
            wandb.run.summary["best/val_acc"]  = best_val_acc
            wandb.run.summary["best/test_acc"] = best_test_acc
            wandb.run.summary["best/round"]    = rnd + 1

        # if (rnd + 1) % CKPT_INTERVAL == 0:
        #     ckpt = model_path.replace(".pt", f"_rnd{rnd+1:03d}.pt")
        #     torch.save(model.state_dict(), ckpt)
        #     print(f"  [ckpt] saved {ckpt}")

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

    # ── kNN OOD calibration ───────────────────────────────────────────────────
    print(f"\n{'='*70}\n  kNN OOD Calibration  (k={KNN_K})")
    bank_feats, _ = build_knn_bank(model, train_ns_ld)
    threshold     = calibrate_threshold(model, val_ld, bank_feats)
    sitl_false_reject = compute_rejection_rate(model, test_ld, bank_feats, threshold)
    print(f"  SITL false-rejection: {sitl_false_reject*100:.1f}%")
    wandb.run.summary["ood/threshold"]         = threshold
    wandb.run.summary["ood/sitl_false_reject"] = sitl_false_reject

    # ── real flight evaluation ────────────────────────────────────────────────
    csv_files = [p for p in sorted(REALFLIGHT_DIR.glob("*.csv"))
                 if "_raw" not in p.name]
    print(f"\n{'='*70}\n  Real Flight Evaluation ({len(csv_files)} files)\n{'='*70}\n")
    results = evaluate_realflight(model, csv_files, bank_feats, threshold)

    ardu    = sum(1 for r in results if r["prediction"] == "ArduPilot")
    px4     = sum(1 for r in results if r["prediction"] == "PX4")
    unknown = sum(1 for r in results if r["prediction"] == "Unknown")

    labeled = [(r, _filename_label(r["file"])) for r in results
               if _filename_label(r["file"]) is not None]
    correct  = sum(1 for r, gt in labeled if r["prediction"] == gt)
    accuracy = correct / len(labeled) if labeled else 0.0
    print(f"\n  Total:{len(results)}  ArduPilot:{ardu}  PX4:{px4}  Unknown:{unknown}"
          f"  Accuracy:{correct}/{len(labeled)} ({accuracy*100:.1f}%)")

    out = f"diversify_feat7_realflight_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {out}\n  Model:   {model_path}")

    # log to history (so the sweep's Bayes optimizer reliably reads the metric)
    wandb.log({
        "realflight/total":     len(results),
        "realflight/ardupilot": ardu,
        "realflight/px4":       px4,
        "realflight/unknown":   unknown,
        "realflight/accuracy":  accuracy,
        "realflight/correct":   correct,
        "realflight/labeled":   len(labeled),
    })
    wandb.run.summary.update({
        "realflight/accuracy":  accuracy,
        "realflight/correct":   correct,
        "realflight/labeled":   len(labeled),
    })
    realflight_table = wandb.Table(
        columns=["file", "ground_truth", "prediction", "correct", "px4_prob", "knn_dist",
                 "n_windows", "n_accepted", "n_rejected", "n_px4",
                 "n_ardu", "reject_rate"],
        data=[[r["file"], _filename_label(r["file"]) or "Unknown",
               r["prediction"], _filename_label(r["file"]) == r["prediction"],
               r.get("px4_prob"), r["knn_dist"],
               r["n_windows"], r["n_accepted"], r["n_rejected"],
               r["n_px4"], r["n_ardu"], r["reject_rate"]] for r in results],
    )
    wandb.log({"realflight/results": realflight_table})

    artifact = wandb.Artifact(
        name="diversify_feat7",
        type="model",
        metadata={
            "timestamp":         ts,
            "best_val_acc":      best_val_acc,
            "best_test_acc":     best_test_acc,
            "ood_threshold":     threshold,
            "sitl_false_reject": sitl_false_reject,
        },
    )
    artifact.add_file(model_path)
    artifact.add_file(out)
    run.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
