"""
predict.py — best_model_model.pt を使って郡別収量を予測・評価する

Usage:
    python predict.py                        # test counties のみ評価
    python predict.py --all                  # train + test 全郡評価
    python predict.py --geoids 17001 17003   # 特定 geoid を指定

Output:
    OUTPUT_DIR / predictions.csv            予測値 vs 実測値テーブル
    OUTPUT_DIR / prediction_scatter.png     散布図 (R², RMSE 付き)
"""

import argparse, gc, json, re, datetime, sys, yaml
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import matplotlib.pyplot as plt
from einops import rearrange
from tqdm import tqdm

# ════════════════════════════════════════════════════════
#  PATHS
# ════════════════════════════════════════════════════════
REPO_ROOT  = Path("C:/Users/room208/mizuho")
EO_DIR     = REPO_ROOT / "Prithvi-EO-2.0-300M"
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR  / "mizuho_output"
HLS_DIR    = DATA_DIR  / "hls_counties"
CSV_PATH   = DATA_DIR  / "USDA_Soybean_County_2020.csv"

EO_CONFIG_PATH     = EO_DIR / "config.json"
EO_CHECKPOINT_PATH = EO_DIR / "Prithvi_EO_V2_300M.pt"
BEST_MODEL_PATH    = OUTPUT_DIR / "best_model_model.pt"

for p in [EO_DIR]:
    assert p.exists(), f"Not found: {p}"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ════════════════════════════════════════════════════════
#  MODEL DEFINITIONS  (train_model.py と同一)
# ════════════════════════════════════════════════════════

class PatchAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )
    def forward(self, x):
        scores  = self.attention_net(x)
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.transpose(1, 2), x).squeeze(1)


class CrossModalAttention(nn.Module):
    def __init__(self, d_eo=1024, d_met=5120, n_heads=8, dropout=0.1):
        super().__init__()
        self.kv_proj = nn.Linear(d_met, d_eo)
        self.attn    = nn.MultiheadAttention(d_eo, n_heads, dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(d_eo)
    def forward(self, q, kv):
        kv_p = self.kv_proj(kv)
        out, _ = self.attn(q, kv_p, kv_p)
        return self.norm(q + out)


class MLPRegressionHead(nn.Module):
    def __init__(self, d_in=1024, hidden=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden),      nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x): return self.mlp(x).squeeze(-1)


# ════════════════════════════════════════════════════════
#  LOAD CHECKPOINT
# ════════════════════════════════════════════════════════
print(f"\nLoading best model: {BEST_MODEL_PATH}")
assert BEST_MODEL_PATH.exists(), f"Not found: {BEST_MODEL_PATH}"

ckpt = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)

y_mean          = ckpt["y_mean"]
y_std           = ckpt["y_std"]
train_geoids_ck = ckpt["train_geoids"]
test_geoids_ck  = ckpt["test_geoids"]
unfreeze_layers = ckpt.get("unfreeze_eo_layers", 0)
best_val_loss   = ckpt.get("loss", float("nan"))

print(f"  y_mean={y_mean:.2f}  y_std={y_std:.2f} bu/acre")
print(f"  Train counties : {len(train_geoids_ck)}")
print(f"  Test  counties : {len(test_geoids_ck)}")
print(f"  UNFREEZE_EO_LAYERS : {unfreeze_layers}")
print(f"  Best val loss      : {best_val_loss:.6f}  "
      f"(RMSE≈{best_val_loss**0.5 * y_std:.2f} bu/acre)")

# ── Adapter models ────────────────────────────────────
patch_pool = PatchAttentionPooling(1024).to(device)
cross_attn = CrossModalAttention(1024, 5120, 8).to(device)
mlp_head   = MLPRegressionHead(1024).to(device)

patch_pool.load_state_dict(ckpt["patch_pool"])
cross_attn.load_state_dict(ckpt["cross_attn"])
mlp_head.load_state_dict(ckpt["mlp_head"])

patch_pool.eval(); cross_attn.eval(); mlp_head.eval()

# ════════════════════════════════════════════════════════
#  LOAD PRITHVI-EO MODEL
# ════════════════════════════════════════════════════════
print("\nLoading Prithvi-EO model...")
from prithvi_mae import PrithviMAE

with open(EO_CONFIG_PATH) as f:
    eo_config = yaml.safe_load(f)["pretrained_cfg"]

bands      = eo_config["bands"]
mean_norm  = eo_config["mean"]
std_norm   = eo_config["std"]
img_size   = eo_config["img_size"]
coords_enc = eo_config["coords_encoding"]

eo_cfg = dict(eo_config)
eo_cfg.update(coords_encoding=coords_enc, num_frames=1, in_chans=len(bands))
eo_model = PrithviMAE(**eo_cfg).to(device)

sd_base = torch.load(EO_CHECKPOINT_PATH, map_location=device, weights_only=True)
for k in list(sd_base.keys()):
    if "pos_embed" in k: del sd_base[k]
eo_model.load_state_dict(sd_base, strict=False)
del sd_base

if "eo_model" in ckpt:
    eo_model.load_state_dict(ckpt["eo_model"])
    print(f"  Loaded fine-tuned EO weights (UNFREEZE_EO_LAYERS={unfreeze_layers})")
else:
    print("  EO was frozen during training — using pretrained weights only")

eo_model.eval()
del ckpt
torch.cuda.empty_cache()

# ════════════════════════════════════════════════════════
#  HLS LOADING HELPERS
# ════════════════════════════════════════════════════════
NO_DATA       = -9999
NO_DATA_FLOAT = 0.0001
TARGET_YEAR   = 2020
BATCH_SIZE    = 2

def load_hls_windows(tif_path: str):
    with rasterio.open(tif_path) as src:
        img = src.read()
        try:    coords = src.lnglat()
        except: coords = None

    img = np.moveaxis(img, 0, -1)
    img = np.where(img == NO_DATA, NO_DATA_FLOAT,
                   (img - mean_norm) / std_norm).astype("float32")
    img = np.moveaxis(img, -1, 0)
    img = np.expand_dims(img, axis=(0, 2))

    fname = Path(tif_path).stem
    match = re.search(r"(\d{7,8}T\d{6})", fname)
    if match:
        year     = int(match.group(1)[:4])
        jday_str = match.group(1).split("T")[0][4:]
        jday     = int(jday_str) if len(jday_str) == 3 else \
                   datetime.datetime.strptime(jday_str, "%m%d").timetuple().tm_yday
    else:
        year, jday = TARGET_YEAR, 1

    oh, ow = img.shape[-2:]
    ph = img_size - (oh % img_size) if oh % img_size != 0 else 0
    pw = img_size - (ow % img_size) if ow % img_size != 0 else 0
    img = np.pad(img, ((0,0),(0,0),(0,0),(0,ph),(0,pw)), mode="reflect")

    batch   = torch.tensor(img)
    windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
    windows = rearrange(windows, "b c t h1 w1 h w -> (b h1 w1) c t h w",
                        h=img_size, w=img_size)

    tc = torch.tensor([[year, jday]], dtype=torch.float32).to(device)
    lc = torch.tensor([list(coords)], dtype=torch.float32).to(device) if coords else None
    return windows, tc, lc


def run_eo_and_pool(windows, tc, lc):
    n_patches  = windows.shape[0]
    cls_tokens = []
    with torch.no_grad():
        for i in range(0, n_patches, BATCH_SIZE):
            x     = windows[i:i+BATCH_SIZE].to(device)
            feats = eo_model.forward_features(
                x,
                tc.expand(x.shape[0], -1),
                lc.expand(x.shape[0], -1) if lc is not None else None,
            )
            cls_tokens.append(feats[-1][:, 0, :])
    patches = torch.cat(cls_tokens, dim=0).unsqueeze(0)  # [1, N_patches, 1024]
    return patch_pool(patches)                            # [1, 1024]


# ════════════════════════════════════════════════════════
#  COUNTY / MET / YIELD DATA
# ════════════════════════════════════════════════════════
with open(OUTPUT_DIR / "q_save_paths.json") as f:
    q_save_paths = json.load(f)
RESOLVED_GEOIDS = list(q_save_paths.keys())

hls_paths = {}
for geoid in RESOLVED_GEOIDS:
    p = HLS_DIR / f"{geoid}_HLS.tif"
    if p.exists():
        hls_paths[geoid] = str(p)

met_embedding = torch.load(OUTPUT_DIR / "met_embedding.pt", map_location=device)
nan_count = torch.isnan(met_embedding).sum().item()
if nan_count > 0:
    print(f"  [WARN] met_embedding has {nan_count} NaN — replacing with 0")
    met_embedding = torch.nan_to_num(met_embedding, nan=0.0)
met_emb = met_embedding[:, :len(RESOLVED_GEOIDS), :]

df_yield = pd.read_csv(CSV_PATH)
df_yield["GEOID"] = (df_yield["state_ansi"].astype(str).str.zfill(2) +
                     df_yield["county_ansi"].astype(str).str.zfill(3))
yield_map = dict(zip(df_yield["GEOID"], df_yield["YIELD, MEASURED IN BU / ACRE"]))

# ════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--all",    action="store_true",
                    help="Predict all counties (train + test)")
parser.add_argument("--geoids", nargs="+", default=None,
                    help="Specific GEOID list to predict")
args = parser.parse_args()

if args.geoids:
    predict_geoids = [g for g in args.geoids if g in hls_paths]
    missing = [g for g in args.geoids if g not in hls_paths]
    if missing:
        print(f"  [WARN] HLS not found for: {missing}")
    split_label = "custom"
elif args.all:
    predict_geoids = [g for g in (train_geoids_ck + test_geoids_ck)
                      if g in hls_paths and g in yield_map]
    split_label = "all"
else:
    predict_geoids = [g for g in test_geoids_ck
                      if g in hls_paths and g in yield_map]
    split_label = "test"

print(f"\nPredicting {len(predict_geoids)} counties  (split='{split_label}')")

# ════════════════════════════════════════════════════════
#  INFERENCE LOOP
# ════════════════════════════════════════════════════════
records = []

for geoid in tqdm(predict_geoids, desc="Predict", ncols=90):
    geoid_idx = RESOLVED_GEOIDS.index(geoid)
    met_i     = met_emb[:, geoid_idx:geoid_idx+1, :]   # [1, 1, 5120]

    windows, tc, lc = load_hls_windows(hls_paths[geoid])
    q = run_eo_and_pool(windows, tc, lc)   # [1, 1024]
    q = q.unsqueeze(1)                     # [1, 1, 1024]

    with torch.no_grad():
        fused      = cross_attn(q, met_i)
        y_hat_norm = mlp_head(fused).item()

    y_pred   = y_hat_norm * y_std + y_mean
    y_actual = yield_map.get(geoid, float("nan"))

    split = ("train" if geoid in train_geoids_ck else
             "test"  if geoid in test_geoids_ck  else "unknown")

    records.append({
        "geoid":    geoid,
        "split":    split,
        "y_actual": round(y_actual, 2),
        "y_pred":   round(y_pred,   2),
        "error":    round(y_pred - y_actual, 2) if not np.isnan(y_actual) else float("nan"),
    })

    del windows, tc, lc, q, fused

df_pred = pd.DataFrame(records)

# ════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════
def compute_metrics(df_sub, label):
    df_v = df_sub.dropna(subset=["y_actual", "y_pred"])
    if len(df_v) == 0:
        print(f"  [{label}] no valid rows")
        return {}
    y_a  = df_v["y_actual"].values
    y_p  = df_v["y_pred"].values
    rmse = np.sqrt(np.mean((y_a - y_p) ** 2))
    mae  = np.mean(np.abs(y_a - y_p))
    ss_res = np.sum((y_a - y_p) ** 2)
    ss_tot = np.sum((y_a - y_a.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  [{label:5s}]  N={len(df_v):4d}  "
          f"RMSE={rmse:.2f} bu/acre  MAE={mae:.2f} bu/acre  R²={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": len(df_v)}

print("\n── Evaluation ─────────────────────────────────────────")
m_all   = compute_metrics(df_pred,                          "all")
m_train = compute_metrics(df_pred[df_pred["split"]=="train"], "train")
m_test  = compute_metrics(df_pred[df_pred["split"]=="test"],  "test")

# ════════════════════════════════════════════════════════
#  SAVE CSV
# ════════════════════════════════════════════════════════
csv_out = OUTPUT_DIR / "predictions.csv"
df_pred.to_csv(csv_out, index=False)
print(f"\n  Saved: {csv_out}")
print(df_pred.sort_values("split").to_string(index=False, max_rows=30))

# ════════════════════════════════════════════════════════
#  SCATTER PLOT
# ════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 7))

colors = {"train": "#1D9E75", "test": "#E24B4A", "unknown": "#888888"}
for split, grp in df_pred.groupby("split"):
    grp_v = grp.dropna(subset=["y_actual", "y_pred"])
    ax.scatter(grp_v["y_actual"], grp_v["y_pred"],
               c=colors.get(split, "#888888"), alpha=0.7,
               edgecolors="white", linewidths=0.4, s=60,
               label=f"{split}  (N={len(grp_v)})")

lo = df_pred[["y_actual","y_pred"]].min().min() - 2
hi = df_pred[["y_actual","y_pred"]].max().max() + 2
ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x")

txt_lines = []
for label, m in [("All  ", m_all), ("Train", m_train), ("Test ", m_test)]:
    if m:
        txt_lines.append(f"{label}  RMSE={m['rmse']:.1f}  R²={m['r2']:.3f}")
ax.text(0.03, 0.97, "\n".join(txt_lines),
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9))

ax.set_xlabel("Actual Yield (bu/acre)",    fontsize=12)
ax.set_ylabel("Predicted Yield (bu/acre)", fontsize=12)
ax.set_title("Soybean Yield Prediction vs Actual  (2020)", fontsize=13)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.legend(fontsize=10)
ax.grid(alpha=0.25)
plt.tight_layout()

plot_out = OUTPUT_DIR / "prediction_scatter.png"
plt.savefig(plot_out, dpi=150)
print(f"  Saved: {plot_out}")
plt.show()
