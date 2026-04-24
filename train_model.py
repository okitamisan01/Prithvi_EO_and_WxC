"""
train_model.py — Foundation model の一部を unfreeze して学習するバージョン

train.py との違い:
  - Prithvi-EO モデルをロードし、最終 N 層の重みも学習する
  - HLS TIF ファイルから毎 epoch EO 推論を実行（patch キャッシュは使わない）
  - met_embedding は frozen のまま（WxC は VRAM 不足で unfreeze 不可）

メモリ目安（RTX 4090 / 25.8GB）:
  UNFREEZE_EO_LAYERS = 0   → アダプタのみ学習 (train.py と同等) ~2GB
  UNFREEZE_EO_LAYERS = 2   → EO 最終2層 + アダプタ               ~6GB
  UNFREEZE_EO_LAYERS = 6   → EO 最終6層 + アダプタ               ~10GB
  UNFREEZE_EO_LAYERS = -1  → EO 全層 unfreeze                    ~25GB (ギリギリ)

Prerequisites:
  DATA_DIR / hls_counties / {geoid}_HLS.tif    (from preprocess.py Step 2)
  OUTPUT_DIR / met_embedding.pt                (from preprocess.py Step 4)
  OUTPUT_DIR / q_save_paths.json
  DATA_DIR / USDA_Soybean_County_2020.csv
"""

import gc, json, re, datetime, sys, yaml, time
import random
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
#  SETTINGS
# ════════════════════════════════════════════════════════
REPO_ROOT  = Path("C:/Users/room208/mizuho")
EO_DIR     = REPO_ROOT / "Prithvi-EO-2.0-300M"
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR  / "mizuho_output"
HLS_DIR    = DATA_DIR  / "hls_counties"
CSV_PATH   = DATA_DIR  / "USDA_Soybean_County_2020.csv"

EO_CONFIG_PATH     = EO_DIR / "config.json"
EO_CHECKPOINT_PATH = EO_DIR / "Prithvi_EO_V2_300M.pt"

# ── unfreeze 設定 ────────────────────────────────────────
# 0  = EO 完全 frozen（アダプタのみ学習）
# 2  = EO 最終 2 transformer block を学習（推奨）
# -1 = EO 全層を学習（VRAM 注意）
UNFREEZE_EO_LAYERS = 2

N_EPOCHS    = 1
LR_ADAPTER  = 1e-4   # アダプタ・新規層の学習率
LR_EO       = 1e-5   # EO unfreeze 層の学習率（小さめ）
L2_LAMBDA   = 1e-5
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
TARGET_YEAR = 2020

EARLY_STOPPING_PATIENCE = 20

# ════════════════════════════════════════════════════════
#  PATHS & IMPORTS
# ════════════════════════════════════════════════════════
for p in [EO_DIR]:
    assert p.exists(), f"Not found: {p}"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    gc.collect(); torch.cuda.empty_cache()
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            nn.Linear(d_in, hidden),   nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x): return self.mlp(x).squeeze(-1)


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

sd = torch.load(EO_CHECKPOINT_PATH, map_location=device, weights_only=True)
for k in list(sd.keys()):
    if "pos_embed" in k: del sd[k]
eo_model.load_state_dict(sd, strict=False)

# ── Freeze / Unfreeze ──────────────────────────────────
for param in eo_model.parameters():
    param.requires_grad = False          # まず全部 freeze

eo_unfreeze_params = []

if UNFREEZE_EO_LAYERS == 0:
    print("  EO: fully frozen")

elif UNFREEZE_EO_LAYERS == -1:
    for param in eo_model.parameters():
        param.requires_grad = True
    eo_unfreeze_params = list(eo_model.parameters())
    total = sum(p.numel() for p in eo_unfreeze_params) / 1e6
    print(f"  EO: ALL layers unfrozen  ({total:.0f}M params) ⚠️ VRAM heavy")

else:
    # transformer blocks の最終 N 層だけ unfreeze
    n_blocks = len(eo_model.encoder.blocks)
    unfreeze_from = max(0, n_blocks - UNFREEZE_EO_LAYERS)
    for i, block in enumerate(eo_model.encoder.blocks):
        if i >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True
            eo_unfreeze_params += list(block.parameters())
    # norm layer も unfreeze
    if hasattr(eo_model.encoder, "norm"):
        for param in eo_model.encoder.norm.parameters():
            param.requires_grad = True
        eo_unfreeze_params += list(eo_model.encoder.norm.parameters())

    n_unfreeze = sum(p.numel() for p in eo_unfreeze_params) / 1e6
    print(f"  EO: last {UNFREEZE_EO_LAYERS} blocks unfrozen  ({n_unfreeze:.1f}M params)")
    print(f"  EO: blocks {unfreeze_from}〜{n_blocks-1} (total {n_blocks} blocks)")

# ════════════════════════════════════════════════════════
#  HLS DATA LOADING (EO 推論用)
# ════════════════════════════════════════════════════════
NO_DATA       = -9999
NO_DATA_FLOAT = 0.0001

def load_hls_windows(tif_path: str):
    """
    HLS TIF → sliding window テンソル + 座標
    戻り値: windows [N_patches, C, T, H, W], temporal_coords, location_coords
    """
    geoid = Path(tif_path).stem.replace("_HLS", "")
    if geoid in _hls_img_cache:
        img, coords = _hls_img_cache[geoid]
    else:
        with rasterio.open(tif_path) as src:
            img = src.read()   # [C, H, W]
            try:    coords = src.lnglat()
            except: coords = None
        img = np.moveaxis(img, 0, -1)
        img = np.where(img == NO_DATA, NO_DATA_FLOAT,
                       (img - mean_norm) / std_norm).astype("float32")
        img = np.moveaxis(img, -1, 0)
    img = np.expand_dims(img, axis=(0, 2))  # [1, C, 1, H, W]

    # Timestamp から temporal coords 推定
    fname = Path(tif_path).stem
    match = re.search(r"(\d{7,8}T\d{6})", fname)
    if match:
        year     = int(match.group(1)[:4])
        jday_str = match.group(1).split("T")[0][4:]
        jday     = int(jday_str) if len(jday_str) == 3 else \
                   datetime.datetime.strptime(jday_str, "%m%d").timetuple().tm_yday
    else:
        year, jday = TARGET_YEAR, 1

    # Pad & sliding window
    oh, ow = img.shape[-2:]
    ph = img_size - (oh % img_size) if oh % img_size != 0 else 0
    pw = img_size - (ow % img_size) if ow % img_size != 0 else 0
    img = np.pad(img, ((0,0),(0,0),(0,0),(0,ph),(0,pw)), mode="reflect")

    batch   = torch.tensor(img)
    windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
    windows = rearrange(windows, "b c t h1 w1 h w -> (b h1 w1) c t h w",
                        h=img_size, w=img_size)   # [N_patches, C, 1, H, W]

    tc = torch.tensor([[year, jday]], dtype=torch.float32).to(device)
    lc = torch.tensor([list(coords)], dtype=torch.float32).to(device) if coords else None

    return windows, tc, lc


def run_eo_and_pool(tif_path, patch_pool, patch_pbar=None, _preloaded=None, phase="train"):
    if _preloaded is not None:
        windows, tc, lc = _preloaded
    else:
        windows, tc, lc = load_hls_windows(tif_path)

    n_patches = windows.shape[0]
    # Larger batches = fewer kernel launches = faster GPU utilization
    # Training: limit by activation memory for backprop; val: no_grad allows much larger
    BATCH_SIZE = 32 if (phase == "train" and UNFREEZE_EO_LAYERS != 0) else 128

    cls_tokens = []
    for i in range(0, n_patches, BATCH_SIZE):
        x = windows[i:i+BATCH_SIZE].to(device)  # [B, C, T, H, W]

        if UNFREEZE_EO_LAYERS == 0:
            with torch.no_grad():
                feats = eo_model.forward_features(x, tc.expand(x.shape[0], -1),
                                                   lc.expand(x.shape[0], -1) if lc is not None else None)
        else:
            feats = eo_model.forward_features(x, tc.expand(x.shape[0], -1),
                                               lc.expand(x.shape[0], -1) if lc is not None else None)

        cls = feats[-1][:, 0, :]  # [B, 1024]
        cls_tokens.append(cls)
        if patch_pbar is not None:
            patch_pbar.update(x.shape[0])

    patches = torch.cat(cls_tokens, dim=0).unsqueeze(0)  # [1, N_patches, 1024]
    return patch_pool(patches)                    # [1, 1024]


# ════════════════════════════════════════════════════════
#  COUNTY LIST & HLS PATHS
# ════════════════════════════════════════════════════════
print("\nBuilding county list...")

with open(OUTPUT_DIR / "q_save_paths.json") as f:
    q_save_paths = json.load(f)
RESOLVED_GEOIDS = list(q_save_paths.keys())

# HLS ファイルが存在する county だけ使う
hls_paths = {}
for geoid in RESOLVED_GEOIDS:
    p = HLS_DIR / f"{geoid}_HLS.tif"
    if p.exists():
        hls_paths[geoid] = str(p)
    else:
        print(f"  [WARN] HLS not found: {geoid}")

print(f"  Counties with HLS : {len(hls_paths)} / {len(RESOLVED_GEOIDS)}")
RESOLVED_GEOIDS = list(hls_paths.keys())

# ════════════════════════════════════════════════════════
#  PRE-LOAD HLS IMAGES INTO CPU RAM (eliminates repeated disk I/O)
# ════════════════════════════════════════════════════════
print("\nPre-loading HLS images into CPU cache...")
_hls_img_cache: dict = {}  # geoid -> (img_chw float32, coords)
for _g in tqdm(RESOLVED_GEOIDS, desc="Caching HLS"):
    with rasterio.open(hls_paths[_g]) as _src:
        _img = _src.read()
        try:    _coords = _src.lnglat()
        except: _coords = None
    _img = np.moveaxis(_img, 0, -1)
    _img = np.where(_img == NO_DATA, NO_DATA_FLOAT,
                    (_img - mean_norm) / std_norm).astype("float32")
    _img = np.moveaxis(_img, -1, 0)
    _hls_img_cache[_g] = (_img, _coords)
print(f"  Cached {len(_hls_img_cache)} county images")

# ════════════════════════════════════════════════════════
#  MET EMBEDDING (frozen)
# ════════════════════════════════════════════════════════
met_embedding = torch.load(OUTPUT_DIR / "met_embedding.pt", map_location=device)
met_emb       = met_embedding[:, :len(RESOLVED_GEOIDS), :]
print(f"  met_embedding : {met_emb.shape}")

# ════════════════════════════════════════════════════════
#  YIELD LABELS & SPLIT
# ════════════════════════════════════════════════════════
df_yield = pd.read_csv(CSV_PATH)
df_yield["GEOID"] = (df_yield["state_ansi"].astype(str).str.zfill(2) +
                     df_yield["county_ansi"].astype(str).str.zfill(3))
yield_map = dict(zip(df_yield["GEOID"], df_yield["YIELD, MEASURED IN BU / ACRE"]))

valid_geoids = [g for g in RESOLVED_GEOIDS if g in yield_map]
print(f"  Counties with yield : {len(valid_geoids)} / {len(RESOLVED_GEOIDS)}")

random.seed(RANDOM_SEED)
shuffled     = valid_geoids.copy(); random.shuffle(shuffled)
n_train      = int(len(shuffled) * TRAIN_RATIO)
train_geoids = shuffled[:n_train]
test_geoids  = shuffled[n_train:]
train_idx    = [RESOLVED_GEOIDS.index(g) for g in train_geoids]
test_idx     = [RESOLVED_GEOIDS.index(g) for g in test_geoids]

train_yields = [yield_map[g] for g in train_geoids]
y_mean       = sum(train_yields) / len(train_yields)
y_std        = (sum((v-y_mean)**2 for v in train_yields)/len(train_yields))**0.5
norm         = lambda gs: [(yield_map[g]-y_mean)/y_std for g in gs]

y_train = torch.tensor(norm(train_geoids), dtype=torch.float32).unsqueeze(0).to(device)
y_test  = torch.tensor(norm(test_geoids),  dtype=torch.float32).unsqueeze(0).to(device)

met_train = met_emb[:, train_idx, :].detach()
met_test  = met_emb[:, test_idx,  :].detach()

print(f"  Train/Test : {len(train_geoids)} / {len(test_geoids)}")
print(f"  y_mean={y_mean:.2f}  y_std={y_std:.2f} bu/acre")

# ════════════════════════════════════════════════════════
#  ADAPTER MODELS
# ════════════════════════════════════════════════════════
patch_pool = PatchAttentionPooling(1024).to(device)
cross_attn = CrossModalAttention(1024, met_emb.shape[-1], 8).to(device)
mlp_head   = MLPRegressionHead(1024).to(device)

# ════════════════════════════════════════════════════════
#  OPTIMIZER
#  EO unfreeze 層は小さい LR、アダプタは通常 LR
# ════════════════════════════════════════════════════════
param_groups = [
    {"params": patch_pool.parameters(), "lr": LR_ADAPTER},
    {"params": cross_attn.parameters(), "lr": LR_ADAPTER},
    {"params": mlp_head.parameters(),   "lr": LR_ADAPTER},
]
if eo_unfreeze_params:
    param_groups.append({"params": eo_unfreeze_params, "lr": LR_EO})

optimizer = torch.optim.Adam(param_groups)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
loss_fn   = nn.MSELoss()
scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

total_trainable = sum(p.numel() for g in param_groups for p in g["params"]) / 1e6
print(f"\n  Trainable params : {total_trainable:.2f}M")

# ════════════════════════════════════════════════════════
#  FORWARD HELPER
# ════════════════════════════════════════════════════════

def build_eo_q(geoid_list: list, phase: str = "train",
               outer_pbar=None) -> torch.Tensor:
    n = len(geoid_list)
    print(f"  [{phase}] starting {n} counties...", flush=True)

    qs = []
    for i, g in enumerate(geoid_list):
        print(f"  [{phase}] {i+1}/{n}  geoid={g}", flush=True)

        windows, tc, lc = load_hls_windows(hls_paths[g])
        n_patches = windows.shape[0]
        print(f"           patches={n_patches}", flush=True)

        q = run_eo_and_pool(hls_paths[g], patch_pool, _preloaded=(windows, tc, lc), phase=phase)
        qs.append(q)

        del windows, tc, lc

        if i % 10 == 0 and torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"           VRAM={vram:.2f}GB", flush=True)

    print(f"  [{phase}] done — {n} counties", flush=True)
    return torch.stack(qs, dim=1)  # [1, N, 1024]


# ════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════
print(f"\nStarting training — {N_EPOCHS} epochs  (UNFREEZE_EO_LAYERS={UNFREEZE_EO_LAYERS})\n")

best_loss        = float("inf")
loss_history     = []
val_loss_history = []
epoch_times      = []
patience_counter = 0

pbar = tqdm(range(N_EPOCHS), desc="Epoch", ncols=110,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

for epoch in pbar:
    t0 = time.time()

    ts = datetime.datetime.now().strftime("%H:%M:%S")
    pbar.write(f"\n{'─'*70}")
    pbar.write(f"  Epoch {epoch+1}/{N_EPOCHS}  |  {ts}  |  "
               f"patience {patience_counter}/{EARLY_STOPPING_PATIENCE}")

    # ── Train ───────────────────────────────────────────
    eo_model.train() if UNFREEZE_EO_LAYERS != 0 else eo_model.eval()
    patch_pool.train(); cross_attn.train(); mlp_head.train()
    optimizer.zero_grad(set_to_none=True)

    t_eo = time.time()
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        eo_q_tr = build_eo_q(train_geoids, phase="train", outer_pbar=pbar)
        fused   = cross_attn(eo_q_tr, met_train)
        y_hat   = mlp_head(fused)
        loss    = loss_fn(y_hat, y_train)
        l2_reg  = sum(p.pow(2).sum() for p in cross_attn.parameters()) * L2_LAMBDA
    t_eo_done = time.time() - t_eo

    scaler.scale(loss + l2_reg).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for g in param_groups for p in g["params"]], max_norm=1.0
    )
    scaler.step(optimizer); scaler.update()
    scheduler.step()
    loss_history.append(loss.item())

    # ── Validation ──────────────────────────────────────
    t_val = time.time()
    with torch.no_grad():
        eo_model.eval(); patch_pool.eval(); cross_attn.eval(); mlp_head.eval()
        eo_q_te   = build_eo_q(test_geoids, phase="val", outer_pbar=pbar)
        fused_val = cross_attn(eo_q_te, met_test)
        y_hat_val = mlp_head(fused_val)
        val_loss  = loss_fn(y_hat_val, y_test).item()
        val_loss_history.append(val_loss)
    t_val_done = time.time() - t_val

    rmse_tr = (loss.item() ** 0.5) * y_std
    rmse_va = (val_loss    ** 0.5) * y_std

    t_epoch = time.time() - t0
    epoch_times.append(t_epoch)
    avg_t   = sum(epoch_times[-10:]) / len(epoch_times[-10:])
    eta_min = avg_t * (N_EPOCHS - epoch - 1) / 60

    pbar.write(f"  Train : loss={loss.item():.6f}  RMSE={rmse_tr:.2f} bu/acre"
               f"  (EO {t_eo_done:.1f}s)")
    pbar.write(f"  Val   : loss={val_loss:.6f}  RMSE={rmse_va:.2f} bu/acre"
               f"  (inf {t_val_done:.1f}s)")
    pbar.write(f"  Best  : {best_loss:.6f}  |  epoch {t_epoch:.1f}s"
               f"  avg={avg_t:.1f}s  ETA≈{eta_min:.0f}min")
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / 1e9
        vram_res   = torch.cuda.memory_reserved()  / 1e9
        pbar.write(f"  VRAM  : {vram_alloc:.2f}/{vram_res:.2f} GB  "
                   f"(alloc/reserved)  LR={scheduler.get_last_lr()[0]:.2e}")

    pbar.set_postfix(
        tr=f"{loss.item():.5f}",
        va=f"{val_loss:.5f}",
        best=f"{best_loss:.5f}",
        rmse=f"{rmse_va:.1f}",
        t=f"{t_epoch:.0f}s",
    )

    # ── Checkpoint ──────────────────────────────────────
    ckpt = {
        "epoch": epoch, "loss": loss.item(),
        "patch_pool": patch_pool.state_dict(),
        "cross_attn": cross_attn.state_dict(),
        "mlp_head":   mlp_head.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "y_mean": y_mean, "y_std": y_std,
        "train_geoids": train_geoids, "test_geoids": test_geoids,
        "unfreeze_eo_layers": UNFREEZE_EO_LAYERS,
    }
    if UNFREEZE_EO_LAYERS != 0:
        ckpt["eo_model"] = eo_model.state_dict()

    torch.save(ckpt, OUTPUT_DIR / "latest_checkpoint_model.pt")

    if val_loss < best_loss:
        best_loss        = val_loss
        patience_counter = 0
        best_ckpt = {k: v for k, v in ckpt.items()
                     if k not in ("epoch", "optimizer", "scheduler")}
        torch.save(best_ckpt, OUTPUT_DIR / "best_model_model.pt")
        pbar.write(f"  ★ New best!  epoch={epoch+1}  val={val_loss:.6f}"
                   f"  RMSE={rmse_va:.2f} bu/acre  → saved")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            pbar.write(f"\n  Early stopping  epoch={epoch+1}"
                       f"  best_val={best_loss:.6f}")
            break

print(f"\nDone.  Best val loss: {best_loss:.6f}")
print(f"Best model: {OUTPUT_DIR / 'best_model_model.pt'}")

# ════════════════════════════════════════════════════════
#  LOSS CURVE
# ════════════════════════════════════════════════════════
plt.figure(figsize=(8, 4))
plt.plot(loss_history,     color="#1D9E75", linewidth=1.5, label="Train")
plt.plot(val_loss_history, color="#E24B4A", linewidth=1.5, label="Val")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title(f"Loss Curve (UNFREEZE_EO_LAYERS={UNFREEZE_EO_LAYERS})")
plt.yscale("log"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve_model.png", dpi=150)
print(f"Loss curve: {OUTPUT_DIR / 'loss_curve_model.png'}")