"""
train_model_unfreeze_2layers.py
EO 最終2層 + WxC 最終2層 を両方 unfreeze して学習するバージョン

train_model.py との違い:
  - WxC model もロード し、最終 N 層を unfreeze
  - WxC blocks 0-(N-3): frozen, 起動時1回だけ実行 → wxc_prefix_tokens.pt にキャッシュ
  - WxC blocks (N-2)-(N-1): unfrozen, 毎 epoch 実行 (backprop)
  - MetAdapter: WxC 出力 → met_embedding を trainable に変換
  - EO 最終2層: train_model.py と同様に unfreeze

preprocess.py との関係:
  再実行不要（データは既にダウンロード済み）
  必要なファイル:
    data/hls_counties/{geoid}_HLS.tif    (EO 推論)
    data/merra-2/ + data/climatology/    (WxC 推論)
    data/mizuho_output/q_save_paths.json (county リスト)
    data/USDA_Soybean_County_2020.csv
    data/2025_Gaz_counties_national.txt  (lat/lon)

VRAM 目安 (RTX 4090 / 24GB):
  WxC fp16 full model       : ~4.6GB
  WxC last 2 blocks grad    : ~4.0GB
  EO fp32 model             : ~1.2GB
  EO last 2 blocks grad     : ~0.8GB
  MetAdapter + adapters     : ~0.5GB
  Activations (checkpointed): ~0.5GB
  合計                       : ~11-14GB  ← RTX 4090 で動作可能
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
WXC_DIR    = REPO_ROOT / "Prithvi-WxC"
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR  / "mizuho_output"
HLS_DIR    = DATA_DIR  / "hls_counties"
MERRA_DIR  = DATA_DIR  / "merra-2"
CLIM_DIR   = DATA_DIR  / "climatology"
GAZ_PATH   = DATA_DIR  / "2025_Gaz_counties_national.txt"
CSV_PATH   = DATA_DIR  / "USDA_Soybean_County_2020.csv"

EO_CONFIG_PATH     = EO_DIR / "config.json"
EO_CHECKPOINT_PATH = EO_DIR / "Prithvi_EO_V2_300M.pt"

# ── unfreeze 設定 ─────────────────────────────────────────
UNFREEZE_EO_LAYERS  = 2   # EO 最終 N transformer block
UNFREEZE_WXC_LAYERS = 2   # WxC 最終 N encoder block

N_EPOCHS    = 100
LR_ADAPTER  = 1e-4   # PatchPool / CrossAttn / MLP
LR_MET      = 1e-4   # MetAdapter
LR_EO       = 1e-6   # EO unfrozen blocks
LR_WXC      = 1e-6   # WxC unfrozen blocks
L2_LAMBDA   = 1e-5
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
TARGET_YEAR = 2020
EARLY_STOPPING_PATIENCE = 20

MONTH       = 1
INPUT_STEP  = 6
LEAD_TIME   = 12

# WxC prefix キャッシュ (blocks 0 to N-UNFREEZE-1 の出力)
WXC_PREFIX_CACHE = OUTPUT_DIR / "wxc_prefix_tokens.pt"
WXC_CLIM_CACHE   = OUTPUT_DIR / "wxc_climate_sc.pt"

# ════════════════════════════════════════════════════════
#  PATHS & IMPORTS
# ════════════════════════════════════════════════════════
for p in [EO_DIR, WXC_DIR]:
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
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.LayerNorm(embed_dim//2),
            nn.Tanh(),
            nn.Linear(embed_dim//2, 1),
        )
    def forward(self, x):
        weights = F.softmax(self.attention_net(x), dim=1)
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


class MetAdapter(nn.Module):
    """
    WxC encoder 出力に対する trainable projection。
    wxc_tokens [B,N,2560] + clim_vecs [B,N,160] → met_embedding [B,N,5120]
    preprocess.py の random-init projection を学習可能にしたもの。
    """
    def __init__(self, d_wxc=2560, d_clim=160):
        super().__init__()
        self.proj_clim = nn.Linear(d_clim, d_wxc)
        self.norm_wxc  = nn.LayerNorm(d_wxc)
        self.norm_clim = nn.LayerNorm(d_wxc)

    def forward(self, wxc_tokens, clim_vecs):
        wxc_n  = self.norm_wxc(wxc_tokens.float())
        clim_n = self.norm_clim(self.proj_clim(clim_vecs.float()))
        return torch.cat([wxc_n, clim_n], dim=-1)   # [B, N, 5120]


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
del sd

# Freeze / Unfreeze EO
for param in eo_model.parameters():
    param.requires_grad = False

eo_unfreeze_params = []
n_eo_blocks = len(eo_model.encoder.blocks)
unfreeze_from_eo = max(0, n_eo_blocks - UNFREEZE_EO_LAYERS)
for i, block in enumerate(eo_model.encoder.blocks):
    if i >= unfreeze_from_eo:
        for p in block.parameters():
            p.requires_grad = True
        eo_unfreeze_params += list(block.parameters())
if hasattr(eo_model.encoder, "norm"):
    for p in eo_model.encoder.norm.parameters():
        p.requires_grad = True
    eo_unfreeze_params += list(eo_model.encoder.norm.parameters())

n_eo_unfreeze = sum(p.numel() for p in eo_unfreeze_params) / 1e6
print(f"  EO: last {UNFREEZE_EO_LAYERS} blocks unfrozen  ({n_eo_unfreeze:.1f}M params)")

# ════════════════════════════════════════════════════════
#  LOAD PRITHVI-WxC MODEL
# ════════════════════════════════════════════════════════
print("\nLoading Prithvi-WxC model...")
from PrithviWxC.model import PrithviWxC
from PrithviWxC.dataloaders.merra2 import (
    Merra2Dataset, preproc,
    input_scalers, output_scalers, static_input_scalers,
)

surface_vars        = ["EFLUX","GWETROOT","HFLUX","LAI","LWGAB","LWGEM","LWTUP",
                       "PS","QV2M","SLP","SWGNT","SWTNT","T2M","TQI","TQL","TQV",
                       "TS","U10M","V10M","Z0M"]
static_surface_vars = ["FRACI","FRLAND","FROCEAN","PHIS"]
vertical_vars       = ["CLOUD","H","OMEGA","PL","QI","QL","QV","T","U","V"]
levels              = [34,39,41,43,44,45,48,51,53,56,63,68,71,72]
padding             = {"level":[0,0],"lat":[0,-1],"lon":[0,0]}
lead_times          = [18]
input_times         = [-6]
positional_encoding = "fourier"

in_mu, in_sig         = input_scalers(surface_vars, vertical_vars, levels,
                            CLIM_DIR/"musigma_surface.nc", CLIM_DIR/"musigma_vertical.nc")
output_sig            = output_scalers(surface_vars, vertical_vars, levels,
                            CLIM_DIR/"anomaly_variance_surface.nc",
                            CLIM_DIR/"anomaly_variance_vertical.nc")
static_mu, static_sig = static_input_scalers(CLIM_DIR/"musigma_surface.nc",
                            static_surface_vars)

with open(DATA_DIR / "Prithvi-WxC-data" / "config.yaml") as f:
    wxc_cfg_file = yaml.safe_load(f)
wp = wxc_cfg_file["params"]

wxc_model = PrithviWxC(
    in_channels=wp["in_channels"], input_size_time=wp["input_size_time"],
    in_channels_static=wp["in_channels_static"],
    input_scalers_mu=in_mu, input_scalers_sigma=in_sig,
    input_scalers_epsilon=wp["input_scalers_epsilon"],
    static_input_scalers_mu=static_mu, static_input_scalers_sigma=static_sig,
    static_input_scalers_epsilon=wp["static_input_scalers_epsilon"],
    output_scalers=output_sig**0.5,
    n_lats_px=wp["n_lats_px"], n_lons_px=wp["n_lons_px"],
    patch_size_px=wp["patch_size_px"], mask_unit_size_px=wp["mask_unit_size_px"],
    mask_ratio_inputs=0.0, mask_ratio_targets=0.0,
    embed_dim=wp["embed_dim"], n_blocks_encoder=wp["n_blocks_encoder"],
    n_blocks_decoder=wp["n_blocks_decoder"], mlp_multiplier=wp["mlp_multiplier"],
    n_heads=wp["n_heads"], dropout=wp["dropout"], drop_path=wp["drop_path"],
    parameter_dropout=wp["parameter_dropout"],
    residual="climate", masking_mode="global",
    encoder_shifting=True, decoder_shifting=True,
    positional_encoding=positional_encoding,
    checkpoint_encoder=[], checkpoint_decoder=[],
)

weights_path = DATA_DIR / "Prithvi-WxC-data" / "prithvi.wxc.2300m.v1.pt"
sd_wxc = torch.load(weights_path, map_location="cpu", weights_only=False)
sd_wxc = sd_wxc.get("model_state", sd_wxc)
wxc_model.load_state_dict(sd_wxc, strict=True)
wxc_model = wxc_model.half().to(device)
del sd_wxc; gc.collect(); torch.cuda.empty_cache()
print("  Loaded WxC weights.")

# Freeze / Unfreeze WxC
for param in wxc_model.parameters():
    param.requires_grad = False

# Access encoder blocks (handles different possible structures)
def _get_blocks(encoder):
    for attr in ("blocks", "layers", "encoder_layers"):
        if hasattr(encoder, attr):
            return list(getattr(encoder, attr))
    # fallback: iterate children
    return list(encoder.children())

wxc_blocks    = _get_blocks(wxc_model.encoder)
n_wxc_blocks  = len(wxc_blocks)
unfreeze_from_wxc = max(0, n_wxc_blocks - UNFREEZE_WXC_LAYERS)

wxc_unfreeze_params = []
for i, block in enumerate(wxc_blocks):
    if i >= unfreeze_from_wxc:
        for p in block.parameters():
            p.requires_grad = True
        wxc_unfreeze_params += list(block.parameters())

n_wxc_unfreeze = sum(p.numel() for p in wxc_unfreeze_params) / 1e6
print(f"  WxC: last {UNFREEZE_WXC_LAYERS} blocks unfrozen  ({n_wxc_unfreeze:.1f}M params)")
print(f"  WxC: blocks {unfreeze_from_wxc}~{n_wxc_blocks-1}  (total {n_wxc_blocks} blocks)")

# ════════════════════════════════════════════════════════
#  WxC PREFIX TOKEN CACHE
#  起動時1回: MERRA-2 → embedding → blocks 0 to N-3 を実行してキャッシュ
# ════════════════════════════════════════════════════════
print("\nBuilding WxC prefix token cache...")

if WXC_PREFIX_CACHE.exists() and WXC_CLIM_CACHE.exists():
    print("  Loading cached prefix tokens...")
    wxc_prefix = torch.load(WXC_PREFIX_CACHE, map_location=device, weights_only=True)
    climate_sc  = torch.load(WXC_CLIM_CACHE,   map_location=device, weights_only=True)
    print(f"  wxc_prefix : {wxc_prefix.shape}")
    print(f"  climate_sc : {climate_sc.shape}")
else:
    print("  Running WxC frozen prefix (blocks 0 to "
          f"{unfreeze_from_wxc-1})... (one-time)")

    dataset = Merra2Dataset(
        time_range=("2020-01-01T00:00:00", "2020-01-02T05:59:59"),
        lead_times=lead_times, input_times=input_times,
        data_path_surface=MERRA_DIR, data_path_vertical=MERRA_DIR,
        climatology_path_surface=CLIM_DIR, climatology_path_vertical=CLIM_DIR,
        surface_vars=surface_vars, static_surface_vars=static_surface_vars,
        vertical_vars=vertical_vars, levels=levels,
        positional_encoding=positional_encoding,
    )
    batch = preproc([next(iter(dataset))], padding)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.half().to(device)

    with torch.no_grad():
        # Embedding stage
        x_rsc = (batch["x"] - wxc_model.input_scalers_mu) / \
                (wxc_model.input_scalers_sigma + wxc_model.input_scalers_epsilon)
        x_rsc = x_rsc.flatten(1, 2)

        x_pos = wxc_model.fourier_pos_encoding(batch["static"])
        x_sta = (batch["static"][:, 2:] - wxc_model.static_input_scalers_mu[:, 3:]) / \
                (wxc_model.static_input_scalers_sigma[:, 3:] +
                 wxc_model.static_input_scalers_epsilon)

        climate_sc = (batch["climate"] - wxc_model.input_scalers_mu.view(1,-1,1,1)) / \
                     (wxc_model.input_scalers_sigma.view(1,-1,1,1) +
                      wxc_model.input_scalers_epsilon)

        x_emb      = wxc_model.patch_embedding(x_rsc)
        static_emb = wxc_model.patch_embedding_static(
            torch.cat((x_sta, climate_sc), dim=1))
        static_emb += x_pos
        x_emb      = wxc_model.to_patching(x_emb)
        static_emb = wxc_model.to_patching(static_emb)
        time_enc   = wxc_model.time_encoding(batch["input_time"], batch["lead_time"])
        tokens     = x_emb + static_emb + time_enc   # [1, N_tokens, 2560]

        # Run frozen prefix blocks
        x = tokens
        for i in range(unfreeze_from_wxc):
            x = wxc_blocks[i](x)
        wxc_prefix = x.detach()

    torch.save(wxc_prefix.cpu(), WXC_PREFIX_CACHE)
    torch.save(climate_sc.cpu(), WXC_CLIM_CACHE)
    print(f"  Saved wxc_prefix : {wxc_prefix.shape}")
    print(f"  Saved climate_sc : {climate_sc.shape}")
    del dataset, batch, tokens, x_rsc, x_pos, x_sta, x_emb, static_emb, time_enc
    gc.collect(); torch.cuda.empty_cache()

# Keep on device for training
wxc_prefix = wxc_prefix.to(device).detach()
climate_sc  = climate_sc.to(device)

# ════════════════════════════════════════════════════════
#  COUNTY LIST, HLS PATHS, YIELD DATA
# ════════════════════════════════════════════════════════
print("\nBuilding county list...")

with open(OUTPUT_DIR / "q_save_paths.json") as f:
    q_save_paths = json.load(f)
RESOLVED_GEOIDS = list(q_save_paths.keys())

hls_paths = {}
for geoid in RESOLVED_GEOIDS:
    p = HLS_DIR / f"{geoid}_HLS.tif"
    if p.exists():
        hls_paths[geoid] = str(p)
    else:
        print(f"  [WARN] HLS not found: {geoid}")

RESOLVED_GEOIDS = list(hls_paths.keys())
print(f"  Counties with HLS : {len(RESOLVED_GEOIDS)}")

# lat/lon for interpolation
_df_gaz = pd.read_csv(GAZ_PATH, sep="|", dtype={"GEOID": str})
_df_gaz.columns = _df_gaz.columns.str.strip()
_df_gaz = _df_gaz[["GEOID","INTPTLAT","INTPTLONG"]].rename(
    columns={"INTPTLAT":"lat","INTPTLONG":"lon"})

df_coords = pd.DataFrame({"GEOID": RESOLVED_GEOIDS}).merge(_df_gaz, on="GEOID", how="left")
if df_coords["lat"].isna().any():
    print(f"  [WARN] {df_coords['lat'].isna().sum()} counties missing lat/lon")
    df_coords = df_coords.dropna(subset=["lat","lon"])
    RESOLVED_GEOIDS = df_coords["GEOID"].tolist()

# ════════════════════════════════════════════════════════
#  WxC COUNTY INTERPOLATION HELPER
# ════════════════════════════════════════════════════════

def run_wxc_tail_and_interpolate(county_idx_list):
    """
    WxC unfrozen blocks + reshape + bilinear interpolation at county centroids.
    Returns met_embedding [1, N_counties, 5120] with gradient w.r.t. wxc tail blocks.
    """
    # --- Run unfrozen WxC blocks (with grad) ---
    x = wxc_prefix.clone()   # [1, N_tokens, 2560], detached from frozen prefix
    for i in range(unfreeze_from_wxc, n_wxc_blocks):
        x = wxc_blocks[i](x)   # grad flows here

    # --- Reshape to spatial feature map ---
    B      = x.shape[0]
    G0, G1 = wxc_model.global_shape_mu
    L0, L1 = wxc_model.local_shape_mu
    D       = wxc_model.embed_dim
    x_enc  = x.view(B, G0, G1, L0, L1, D)
    x_enc  = x_enc.permute(0, 5, 1, 3, 2, 4).contiguous()
    feature_map = x_enc.view(B, D, G0*L0, G1*L1).float()   # [1, 2560, 180, 288]

    C_sc = climate_sc.float()   # [1, 160, H, W]  (frozen, but fine — grad stops here)

    # --- Bilinear interpolation at county centroids ---
    lats = torch.tensor(
        [df_coords.iloc[j]["lat"] for j in county_idx_list],
        dtype=torch.float32, device=device)
    lons = torch.tensor(
        [df_coords.iloc[j]["lon"] for j in county_idx_list],
        dtype=torch.float32, device=device)

    lat_grid = torch.linspace(-90,  90,  feature_map.shape[-2], device=device)
    lon_grid = torch.linspace(-180, 180, feature_map.shape[-1], device=device)
    norm_lats = 2.0*(lats - lat_grid.min())/(lat_grid.max()-lat_grid.min()) - 1.0
    norm_lons = 2.0*(lons - lon_grid.min())/(lon_grid.max()-lon_grid.min()) - 1.0

    grid = torch.stack([norm_lons, norm_lats], dim=-1
                       ).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)   # [1,1,N,2]

    wxc_tokens = F.grid_sample(feature_map, grid, mode="bilinear",
                               align_corners=True).squeeze(2).permute(0,2,1)  # [1,N,2560]
    clim_vecs  = F.grid_sample(C_sc, grid, mode="bilinear",
                               align_corners=True).squeeze(2).permute(0,2,1)  # [1,N,160]

    return wxc_tokens, clim_vecs   # MetAdapter applied in training loop


# ════════════════════════════════════════════════════════
#  EO INFERENCE HELPERS
# ════════════════════════════════════════════════════════
NO_DATA       = -9999
NO_DATA_FLOAT = 0.0001
EO_BATCH_SIZE = 2

def load_hls_windows(tif_path):
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


def run_eo_and_pool(patch_pool, windows, tc, lc):
    cls_tokens = []
    for i in range(0, windows.shape[0], EO_BATCH_SIZE):
        x = windows[i:i+EO_BATCH_SIZE].to(device)
        if UNFREEZE_EO_LAYERS == 0:
            with torch.no_grad():
                feats = eo_model.forward_features(
                    x, tc.expand(x.shape[0],-1),
                    lc.expand(x.shape[0],-1) if lc is not None else None)
        else:
            feats = eo_model.forward_features(
                x, tc.expand(x.shape[0],-1),
                lc.expand(x.shape[0],-1) if lc is not None else None)
        cls_tokens.append(feats[-1][:, 0, :])
    patches = torch.cat(cls_tokens, dim=0).unsqueeze(0)   # [1, N_patches, 1024]
    return patch_pool(patches)                             # [1, 1024]


# ════════════════════════════════════════════════════════
#  YIELD LABELS & TRAIN/TEST SPLIT
# ════════════════════════════════════════════════════════
df_yield = pd.read_csv(CSV_PATH)
df_yield["GEOID"] = (df_yield["state_ansi"].astype(str).str.zfill(2) +
                     df_yield["county_ansi"].astype(str).str.zfill(3))
yield_map = dict(zip(df_yield["GEOID"], df_yield["YIELD, MEASURED IN BU / ACRE"]))

valid_geoids = [g for g in RESOLVED_GEOIDS if g in yield_map]
print(f"  Counties with yield : {len(valid_geoids)} / {len(RESOLVED_GEOIDS)}")

random.seed(RANDOM_SEED)
shuffled = valid_geoids.copy(); random.shuffle(shuffled)
n_train  = int(len(shuffled) * TRAIN_RATIO)
train_geoids = shuffled[:n_train]
test_geoids  = shuffled[n_train:]

train_yields = [yield_map[g] for g in train_geoids]
y_mean = sum(train_yields) / len(train_yields)
y_std  = (sum((v-y_mean)**2 for v in train_yields)/len(train_yields))**0.5
norm   = lambda gs: [(yield_map[g]-y_mean)/y_std for g in gs]

y_train = torch.tensor(norm(train_geoids), dtype=torch.float32).unsqueeze(0).to(device)
y_test  = torch.tensor(norm(test_geoids),  dtype=torch.float32).unsqueeze(0).to(device)

train_idx = [RESOLVED_GEOIDS.index(g) for g in train_geoids]
test_idx  = [RESOLVED_GEOIDS.index(g) for g in test_geoids]

print(f"  Train/Test : {len(train_geoids)} / {len(test_geoids)}")
print(f"  y_mean={y_mean:.2f}  y_std={y_std:.2f} bu/acre")

# ════════════════════════════════════════════════════════
#  ADAPTER MODELS
# ════════════════════════════════════════════════════════
patch_pool  = PatchAttentionPooling(1024).to(device)
cross_attn  = CrossModalAttention(1024, 5120, 8).to(device)
mlp_head    = MLPRegressionHead(1024).to(device)
met_adapter = MetAdapter(d_wxc=wp["embed_dim"], d_clim=wp["in_channels"]).to(device)

# ════════════════════════════════════════════════════════
#  OPTIMIZER
# ════════════════════════════════════════════════════════
param_groups = [
    {"params": patch_pool.parameters(),  "lr": LR_ADAPTER, "name": "patch_pool"},
    {"params": cross_attn.parameters(),  "lr": LR_ADAPTER, "name": "cross_attn"},
    {"params": mlp_head.parameters(),    "lr": LR_ADAPTER, "name": "mlp_head"},
    {"params": met_adapter.parameters(), "lr": LR_MET,     "name": "met_adapter"},
]
if eo_unfreeze_params:
    param_groups.append({"params": eo_unfreeze_params,  "lr": LR_EO,  "name": "eo_blocks"})
if wxc_unfreeze_params:
    param_groups.append({"params": wxc_unfreeze_params, "lr": LR_WXC, "name": "wxc_blocks"})

optimizer = torch.optim.Adam(param_groups)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
loss_fn   = nn.MSELoss()
scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

total_trainable = sum(p.numel() for g in param_groups for p in g["params"]) / 1e6
print(f"\n  Trainable params : {total_trainable:.2f}M")
for g in param_groups:
    n = sum(p.numel() for p in g["params"]) / 1e6
    print(f"    {g['name']:15s} : {n:.2f}M  (lr={g['lr']:.0e})")

# ════════════════════════════════════════════════════════
#  VALIDATION HELPER
# ════════════════════════════════════════════════════════

def build_val_tensors(geoid_list, idx_list):
    """Build EO q and met_embedding for a set of counties."""
    qs = []
    bar = tqdm(enumerate(geoid_list), total=len(geoid_list),
               desc="  [val]", leave=False, ncols=100)
    for i, g in bar:
        bar.set_postfix(geoid=g)
        windows, tc, lc = load_hls_windows(hls_paths[g])
        q = run_eo_and_pool(patch_pool, windows, tc, lc)
        qs.append(q)
        del windows, tc, lc
    eo_q = torch.stack(qs, dim=1)   # [1, N, 1024]

    wxc_t, clim_v = run_wxc_tail_and_interpolate(idx_list)
    met = met_adapter(wxc_t, clim_v)   # [1, N, 5120]
    return eo_q, met


# ════════════════════════════════════════════════════════
#  RESUME FROM CHECKPOINT
# ════════════════════════════════════════════════════════
best_loss        = float("inf")
loss_history     = []
val_loss_history = []
epoch_times      = []
patience_counter = 0
start_epoch      = 0

CKPT_PATH = OUTPUT_DIR / "latest_checkpoint_unfreeze2.pt"
if CKPT_PATH.exists():
    print(f"\nResuming from checkpoint: {CKPT_PATH}")
    ckpt_resume = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    start_epoch      = ckpt_resume["epoch"] + 1
    best_loss        = ckpt_resume.get("best_loss", float("inf"))
    loss_history     = ckpt_resume.get("loss_history", [])
    val_loss_history = ckpt_resume.get("val_loss_history", [])
    patience_counter = ckpt_resume.get("patience_counter", 0)

    patch_pool.load_state_dict(ckpt_resume["patch_pool"])
    cross_attn.load_state_dict(ckpt_resume["cross_attn"])
    mlp_head.load_state_dict(ckpt_resume["mlp_head"])
    met_adapter.load_state_dict(ckpt_resume["met_adapter"])
    optimizer.load_state_dict(ckpt_resume["optimizer"])
    scheduler.load_state_dict(ckpt_resume["scheduler"])
    if "eo_model" in ckpt_resume:
        eo_model.load_state_dict(ckpt_resume["eo_model"])
    if "wxc_tail_state" in ckpt_resume:
        for i, sd in zip(range(unfreeze_from_wxc, n_wxc_blocks),
                         ckpt_resume["wxc_tail_state"]):
            wxc_blocks[i].load_state_dict(sd)

    del ckpt_resume; torch.cuda.empty_cache()
    print(f"  Resumed epoch {start_epoch}/{N_EPOCHS}  best_loss={best_loss:.6f}")
else:
    print("\nNo checkpoint found — starting from scratch.")

# ════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════
print(f"\nStarting training — {N_EPOCHS} epochs\n"
      f"  EO last {UNFREEZE_EO_LAYERS} layers + WxC last {UNFREEZE_WXC_LAYERS} layers\n")

pbar = tqdm(range(start_epoch, N_EPOCHS), desc="Epoch", ncols=110,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")

for epoch in pbar:
    t0 = time.time()
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    pbar.write(f"\n{'─'*70}")
    pbar.write(f"  Epoch {epoch+1}/{N_EPOCHS}  |  {ts}  |  "
               f"patience {patience_counter}/{EARLY_STOPPING_PATIENCE}")

    # ── Train ───────────────────────────────────────────
    eo_model.train() if UNFREEZE_EO_LAYERS != 0 else eo_model.eval()
    for i in range(unfreeze_from_wxc, n_wxc_blocks):
        wxc_blocks[i].train()
    patch_pool.train(); cross_attn.train(); mlp_head.train(); met_adapter.train()
    optimizer.zero_grad(set_to_none=True)

    N_tr = len(train_geoids)
    running_loss = 0.0
    t_train = time.time()

    county_bar = tqdm(enumerate(train_geoids), total=N_tr,
                      desc="  [train]", leave=False, ncols=100)

    for i, g in county_bar:
        county_bar.set_postfix(geoid=g, done=f"{i+1}/{N_tr}")

        windows, tc, lc = load_hls_windows(hls_paths[g])
        y_i = y_train[0, i].float()

        # EO in fp32 (safe for unfrozen pretrained weights)
        q = run_eo_and_pool(patch_pool, windows, tc, lc)   # [1, 1024]
        q = q.unsqueeze(1)                                  # [1, 1, 1024]

        # WxC tail + interpolation for this single county
        wxc_t, clim_v = run_wxc_tail_and_interpolate([train_idx[i]])
        met_i = met_adapter(wxc_t, clim_v)   # [1, 1, 5120]

        # Adapter forward in fp16
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            fused = cross_attn(q, met_i)
            y_hat = mlp_head(fused).squeeze().float()
            loss  = (y_hat - y_i) ** 2 / N_tr
            l2    = sum(p.pow(2).sum() for p in cross_attn.parameters()) \
                    * L2_LAMBDA / N_tr

        if i == 0 and torch.isnan(loss):
            raise RuntimeError(
                f"NaN loss at epoch {epoch+1} county {g}.\n"
                f"  y_i={y_i.item():.4f}")

        scaler.scale(loss + l2).backward()
        running_loss += loss.item() * N_tr
        del windows, tc, lc, q, wxc_t, clim_v, met_i, fused, y_hat, loss

    # Optimizer step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for g in param_groups for p in g["params"]], max_norm=1.0)
    scaler.step(optimizer); scaler.update()
    scheduler.step()

    train_loss = running_loss / N_tr
    loss_history.append(train_loss)
    t_train_done = time.time() - t_train

    # ── Validation ──────────────────────────────────────
    t_val = time.time()
    with torch.no_grad():
        eo_model.eval()
        for i in range(unfreeze_from_wxc, n_wxc_blocks):
            wxc_blocks[i].eval()
        patch_pool.eval(); cross_attn.eval(); mlp_head.eval(); met_adapter.eval()

        eo_q_te, met_te = build_val_tensors(test_geoids, test_idx)
        fused_val = cross_attn(eo_q_te, met_te)
        y_hat_val = mlp_head(fused_val)
        val_loss  = loss_fn(y_hat_val, y_test).item()
        val_loss_history.append(val_loss)
    t_val_done = time.time() - t_val

    rmse_tr = (train_loss ** 0.5) * y_std
    rmse_va = (val_loss  ** 0.5) * y_std
    t_epoch = time.time() - t0
    epoch_times.append(t_epoch)
    avg_t   = sum(epoch_times[-10:]) / len(epoch_times[-10:])
    eta_min = avg_t * (N_EPOCHS - epoch - 1) / 60

    pbar.write(f"  Train : loss={train_loss:.6f}  RMSE={rmse_tr:.2f} bu/acre"
               f"  ({t_train_done:.1f}s)")
    pbar.write(f"  Val   : loss={val_loss:.6f}  RMSE={rmse_va:.2f} bu/acre"
               f"  ({t_val_done:.1f}s)")
    pbar.write(f"  Best  : {best_loss:.6f}  |  epoch {t_epoch:.1f}s"
               f"  avg={avg_t:.1f}s  ETA≈{eta_min:.0f}min")
    if torch.cuda.is_available():
        vram_a = torch.cuda.memory_allocated() / 1e9
        vram_r = torch.cuda.memory_reserved()  / 1e9
        pbar.write(f"  VRAM  : {vram_a:.2f}/{vram_r:.2f} GB  "
                   f"LR_adp={scheduler.get_last_lr()[0]:.2e}")

    pbar.set_postfix(tr=f"{train_loss:.5f}", va=f"{val_loss:.5f}",
                     best=f"{best_loss:.5f}", rmse=f"{rmse_va:.1f}",
                     t=f"{t_epoch:.0f}s")

    # ── Checkpoint ──────────────────────────────────────
    ckpt = {
        "epoch": epoch, "loss": train_loss,
        "best_loss": best_loss,
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "patience_counter": patience_counter,
        "patch_pool":  patch_pool.state_dict(),
        "cross_attn":  cross_attn.state_dict(),
        "mlp_head":    mlp_head.state_dict(),
        "met_adapter": met_adapter.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
        "eo_model":    eo_model.state_dict(),
        "wxc_tail_state": [wxc_blocks[i].state_dict()
                           for i in range(unfreeze_from_wxc, n_wxc_blocks)],
        "y_mean": y_mean, "y_std": y_std,
        "train_geoids": train_geoids, "test_geoids": test_geoids,
        "unfreeze_eo_layers": UNFREEZE_EO_LAYERS,
        "unfreeze_wxc_layers": UNFREEZE_WXC_LAYERS,
    }

    if val_loss < best_loss:
        best_loss        = val_loss
        patience_counter = 0
        ckpt["best_loss"]        = best_loss
        ckpt["patience_counter"] = patience_counter
        best_ckpt = {k: v for k, v in ckpt.items()
                     if k not in ("epoch", "optimizer", "scheduler")}
        torch.save(best_ckpt, OUTPUT_DIR / "best_model_unfreeze2.pt")
        pbar.write(f"  ★ New best!  epoch={epoch+1}  val={val_loss:.6f}"
                   f"  RMSE={rmse_va:.2f} bu/acre  → saved")
    else:
        patience_counter += 1
        ckpt["patience_counter"] = patience_counter
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            pbar.write(f"\n  Early stopping  epoch={epoch+1}"
                       f"  best_val={best_loss:.6f}")
            torch.save(ckpt, CKPT_PATH)
            break

    # ★ Save every epoch (enables resume)
    torch.save(ckpt, CKPT_PATH)

print(f"\nDone.  Best val loss: {best_loss:.6f}")
print(f"Best model: {OUTPUT_DIR / 'best_model_unfreeze2.pt'}")

# ════════════════════════════════════════════════════════
#  LOSS CURVE
# ════════════════════════════════════════════════════════
plt.figure(figsize=(8, 4))
plt.plot(loss_history,     color="#1D9E75", linewidth=1.5, label="Train")
plt.plot(val_loss_history, color="#E24B4A", linewidth=1.5, label="Val")
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title(f"Loss Curve (EO={UNFREEZE_EO_LAYERS}layers + WxC={UNFREEZE_WXC_LAYERS}layers)")
plt.yscale("log"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve_unfreeze2.png", dpi=150)
print(f"Loss curve: {OUTPUT_DIR / 'loss_curve_unfreeze2.png'}")
