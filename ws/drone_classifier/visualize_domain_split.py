"""
Domain split visualization: (a) Initial vs (b) Our method (DIVERSIFY) vs (c) Predicted Classes

(a) Raw feat7 statistics (mean+std, 14-dim) → t-SNE
    Color = source domain (SITL-PX4 / SITL-Ardu / Real-PX4 / Real-Ardu)
    Shows the original Sim2Real domain gap.

(b) Trained DIVERSIFY bottleneck features (32-dim) → t-SNE
    Color = pseudo-domain (K=5, discovered by DIVERSIFY)
    Marker = class (PX4 ● / ArduPilot ▲),  Real flight = ★
    Shows class-aware cross-domain clustering.

(c) Trained DIVERSIFY bottleneck features (32-dim) → t-SNE
    Color = Predicted class (Blue=ArduPilot, Red=PX4)
    Marker = true class (PX4 ● / ArduPilot ▲), Real flight = ★
    Shows the actual classification decision boundaries.
"""

import sys
import random
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
import train_diversify as td

SEED        = 42
MAX_SITL    = 1500        # samples per SITL class
REALFLIGHT  = td.REALFLIGHT_DIR

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ── choose checkpoint ─────────────────────────────────────────────────────────
def _select_checkpoint():
    """CLI arg, else interactive picker over local diversify_*.pt files."""
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if not p.exists():
            raise FileNotFoundError(p)
        return p

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
        raw = input(f"Select checkpoint [0-{len(pts)-1}, default={default_idx}]: ").strip()
        if raw == "":
            return pts[default_idx]
        try:
            idx = int(raw)
            if 0 <= idx < len(pts):
                return pts[idx]
        except ValueError:
            pass
        print("Invalid selection. Try again.")


MODEL_PATH = _select_checkpoint()
_ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH   = f"domain_split_{MODEL_PATH.stem}_{_ts}.png"
print(f"Model:  {MODEL_PATH}")
print(f"Output: {OUT_PATH}")

# real flight label mapping (by filename prefix)
def _real_label(fname):
    """Return 1 (PX4) or 0 (ArduPilot) based on filename."""
    n = fname.lower()
    if "px4" in n:
        return 1
    return 0   # ardupilot / cognipilot / rosbag → treat as ArduPilot


# ── 1. Load SITL windows ──────────────────────────────────────────────────────
print("Loading SITL files …")
px4_files  = [(p, 1) for p in sorted(Path(td.PX4_FOLDER).glob("*.ulg"))]
ardu_files = [(p, 0) for p in sorted(Path(td.ARDU_FOLDER).glob("*.bin"))]

def _load_sitl(file_list, max_n):
    all_x, all_y = [], []
    for path, label in file_list:
        fn = td.process_px4_flight_data if label == 1 else td.process_ardu_flight_data
        result = fn(str(path))
        if result is None:
            continue
        feat = result[4]
        if feat is None or len(feat) < td.MIN_WIN:
            continue
        wins = td._slide_windows(feat)
        for w in wins:
            all_x.append(w); all_y.append(label)
        if len(all_x) >= max_n:
            break
    idx = np.random.choice(len(all_x), min(max_n, len(all_x)), replace=False)
    return [all_x[i] for i in idx], [all_y[i] for i in idx]

px4_x,  px4_y  = _load_sitl(px4_files,  MAX_SITL)
ardu_x, ardu_y = _load_sitl(ardu_files, MAX_SITL)
print(f"  SITL PX4: {len(px4_x)}  SITL Ardu: {len(ardu_x)}")

# ── 2. Load real flight windows ───────────────────────────────────────────────
print("Loading real flight files …")
real_csv = [p for p in sorted(REALFLIGHT.glob("*.csv")) if "_raw" not in p.name]

real_x, real_y, real_names = [], [], []
for csv_path in real_csv:
    label = _real_label(csv_path.name)
    segs = td.process_rosbag_flight_data(str(csv_path))
    for seg in segs:
        feat = seg["data"][4]
        if feat is None or len(feat) < td.MIN_WIN:
            continue
        wins = td._slide_windows(feat)
        for w in wins:
            real_x.append(w)
            real_y.append(label)
            real_names.append(csv_path.name)

print(f"  Real PX4: {sum(1 for y in real_y if y==1)}  Real Ardu: {sum(1 for y in real_y if y==0)}")

# ── 3. Source-domain labels (4 groups) ───────────────────────────────────────
# 0=SITL-PX4  1=SITL-Ardu  2=Real-PX4  3=Real-Ardu
all_x = px4_x + ardu_x + real_x
all_class = np.array(px4_y + ardu_y + real_y, dtype=np.int64)
all_src = np.array(
    [0]*len(px4_x) + [1]*len(ardu_x) +
    [2 if y==1 else 3 for y in real_y],
    dtype=np.int64
)
is_real = np.array([False]*len(px4_x) + [False]*len(ardu_x) + [True]*len(real_x))

N = len(all_x)
print(f"Total windows: {N}  (SITL:{len(px4_x)+len(ardu_x)}  Real:{len(real_x)})")

# ── 4. (a) Raw features: mean+std over time → 14-dim ─────────────────────────
print("\nComputing raw feat statistics (for panel a) …")
raw_feats = np.zeros((N, td.N_FEAT * 2), dtype=np.float32)
for i, w in enumerate(all_x):
    arr = w.numpy()                  # (N_FEAT, WIN_LEN)
    raw_feats[i, :td.N_FEAT]  = arr.mean(axis=1)
    raw_feats[i, td.N_FEAT:]  = arr.std(axis=1)

print("Running t-SNE on raw features …")
tsne_a = TSNE(n_components=2, perplexity=30, max_iter=1000,
              random_state=SEED, n_jobs=-1)
emb_a  = tsne_a.fit_transform(raw_feats)
print("  done.")

# ── 5. Load trained model ─────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_PATH}")
sd = torch.load(MODEL_PATH, map_location=td.DEVICE)
td.LATENT_DOMAIN_N = sd["dclassifier.fc.weight"].shape[0]
print(f"  LATENT_DOMAIN_N={td.LATENT_DOMAIN_N} (from checkpoint)")
model = td.DiversifyFlight().to(td.DEVICE)
model.load_state_dict(sd)
model.eval()

# ── 6. (b) Bottleneck features + pseudo-domain labels & Predictions ──────────
print("Extracting bottleneck features & predictions …")
BSIZE = 256
all_tens = torch.stack(all_x)       # (N, N_FEAT, WIN_LEN)
btn_feats = np.zeros((N, td.BOTTLENECK_DIM), dtype=np.float32)
preds = np.zeros(N, dtype=np.int64) # 모델의 최종 예측 클래스를 담을 배열 추가

with torch.no_grad():
    for s in range(0, N, BSIZE):
        xb = all_tens[s:s+BSIZE].to(td.DEVICE)
        z = model.bottleneck(model.featurizer(xb))
        btn_feats[s:s+len(xb)] = z.cpu().numpy()
        
        # [추가됨] 분류기(Classifier) 통과하여 예측 클래스 확률 계산
        logits = model.classifier(z)
        preds[s:s+len(xb)] = logits.argmax(1).cpu().numpy()

# pseudo-domain via d-branch
print("Running cosine k-means for pseudo-domain labels …")
btn_tens = torch.tensor(btn_feats)
# d-branch features
d_feats = np.zeros((N, td.BOTTLENECK_DIM), dtype=np.float32)
with torch.no_grad():
    for s in range(0, N, BSIZE):
        xb = all_tens[s:s+BSIZE].to(td.DEVICE)
        d_feats[s:s+len(xb)] = model.dbottleneck(model.featurizer(xb)).cpu().numpy()

d_out = np.zeros((N, td.LATENT_DOMAIN_N), dtype=np.float32)
with torch.no_grad():
    for s in range(0, N, BSIZE):
        xb = all_tens[s:s+BSIZE].to(td.DEVICE)
        z  = model.dbottleneck(model.featurizer(xb))
        d_out[s:s+len(xb)] = model.dclassifier(z).cpu().numpy()

# cosine k-means (same as set_dlabel)
from scipy.spatial.distance import cdist
all_fea = np.hstack([d_feats, np.ones((N, 1))])
norms   = np.linalg.norm(all_fea, axis=1, keepdims=True)
all_fea = all_fea / (norms + 1e-8)
aff     = torch.softmax(torch.tensor(d_out), dim=1).numpy()
K       = aff.shape[1]
initc   = aff.T @ all_fea / (1e-8 + aff.sum(0)[:, None])
pred    = cdist(all_fea, initc, "cosine").argmin(1)
aff2    = np.eye(K)[pred]
initc   = aff2.T @ all_fea / (1e-8 + aff2.sum(0)[:, None])
pseudo_dom = cdist(all_fea, initc, "cosine").argmin(1)
print(f"  Pseudo-domain dist: {dict(Counter(pseudo_dom.tolist()))}")

print("\nRunning t-SNE on bottleneck features …")
tsne_b = TSNE(n_components=2, perplexity=30, max_iter=1000,
              random_state=SEED, n_jobs=-1)
emb_b  = tsne_b.fit_transform(btn_feats)
print("  done.")

# ── 7. Plot ───────────────────────────────────────────────────────────────────
# [수정됨] 1x2 -> 1x3 배열로 확장하고 가로 길이를 늘림
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
fig.patch.set_facecolor("white")

SRC_COLORS = ["#2166ac", "#f4a582", "#1a9641", "#d73027"]
SRC_LABELS = ["SITL PX4", "SITL ArduPilot", "Real PX4", "Real ArduPilot"]
DOM_COLORS = plt.cm.tab10(np.linspace(0, 0.5, K))

S_SMALL  = 8    # SITL point size
S_REAL   = 120  # Real flight star size
ALPHA_BG = 0.35

CLASS_MARKERS = {1: "o", 0: "^"}   # PX4=circle, Ardu=triangle
cls_handles = [
    plt.scatter([], [], c="gray", s=30, marker="o", label="True PX4 ●"),
    plt.scatter([], [], c="gray", s=30, marker="^", label="True ArduPilot ▲"),
    plt.scatter([], [], c="gray", s=80, marker="*",
                edgecolors="black", linewidths=0.5, label="Real flight ★"),
]

# ─── Panel (a): Initial domain split ───
ax = axes[0]
ax.set_facecolor("#f8f8f8")
for grp in range(4):
    mask = (all_src == grp) & ~is_real
    ax.scatter(emb_a[mask, 0], emb_a[mask, 1],
               c=SRC_COLORS[grp], s=S_SMALL, alpha=ALPHA_BG,
               linewidths=0, rasterized=True)

for grp, (src_grp, label_idx) in enumerate([(2, 1), (3, 0)]):
    mask = (all_src == src_grp) & is_real
    if mask.sum() == 0:
        continue
    ax.scatter(emb_a[mask, 0], emb_a[mask, 1],
               c=SRC_COLORS[src_grp], s=S_REAL, marker="*",
               edgecolors="black", linewidths=0.5, zorder=5)

patches = [mpatches.Patch(color=SRC_COLORS[i], label=SRC_LABELS[i]) for i in range(4)]
star_h  = plt.scatter([], [], c="gray", s=80, marker="*", edgecolors="black", linewidths=0.5, label="Real flight ★")
ax.legend(handles=patches + [star_h], fontsize=8, loc="upper right", framealpha=0.9)
ax.set_title("(a) Initial domain split", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# ─── Panel (b): Our domain split (Pseudo-domains) ───
ax = axes[1]
ax.set_facecolor("#f8f8f8")

for d in range(K):
    for cls in [0, 1]:
        mask = (pseudo_dom == d) & (all_class == cls) & ~is_real
        if mask.sum() == 0:
            continue
        ax.scatter(emb_b[mask, 0], emb_b[mask, 1],
                   c=[DOM_COLORS[d]], s=S_SMALL, marker=CLASS_MARKERS[cls],
                   alpha=ALPHA_BG, linewidths=0, rasterized=True)

for i in np.where(is_real)[0]:
    d   = pseudo_dom[i]
    ax.scatter(emb_b[i, 0], emb_b[i, 1],
               c=[DOM_COLORS[d]], s=S_REAL, marker="*",
               edgecolors="black", linewidths=0.8, zorder=5)

dom_patches = [mpatches.Patch(color=DOM_COLORS[d], label=f"Pseudo-domain {d}") for d in range(K)]
ax.legend(handles=dom_patches + cls_handles, fontsize=8, loc="upper right", framealpha=0.9)
ax.set_title("(b) Our domain split (DIVERSIFY)", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# ─── Panel (c): True Classes ───
ax = axes[2]
ax.set_facecolor("#f8f8f8")

# 실제 정답(true class)용 색상 (Blue: ArduPilot, Red: PX4)
CLASS_COLORS = {0: "#2166ac", 1: "#d73027"}

for cls in [0, 1]:
    # 실제 정답(cls)으로 색상 + 마커 모양 결정
    mask = (all_class == cls) & ~is_real
    if mask.sum() == 0:
        continue
    ax.scatter(emb_b[mask, 0], emb_b[mask, 1],
               c=[CLASS_COLORS[cls]], s=S_SMALL, marker=CLASS_MARKERS[cls],
               alpha=ALPHA_BG, linewidths=0, rasterized=True)

for i in np.where(is_real)[0]:
    t_cls = all_class[i] # 실제 정답(true label)으로 색상 결정
    ax.scatter(emb_b[i, 0], emb_b[i, 1],
               c=[CLASS_COLORS[t_cls]], s=S_REAL, marker="*",
               edgecolors="black", linewidths=0.8, zorder=5)

class_patches = [
    mpatches.Patch(color=CLASS_COLORS[0], label="True: ArduPilot (Blue)"),
    mpatches.Patch(color=CLASS_COLORS[1], label="True: PX4 (Red)")
]
ax.legend(handles=class_patches + cls_handles, fontsize=8, loc="upper right", framealpha=0.9)
ax.set_title("(c) True Classes", fontsize=13, fontweight="bold")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.suptitle("DIVERSIFY Latent Space Analysis", fontsize=16, fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_PATH}")