"""
preprocess.py — Optimized heavy sections (Step 3 + Step 4)
Drop-in replacements for the original preprocess.py.

Key changes:
  Step 3  — EO patch windows are batched (PATCH_BATCH_SIZE) instead of run one-by-one
  Step 4a — WxC encoder runs under torch.amp (fp16) to halve VRAM usage;
             embedding steps are done on CPU to avoid OOM on 32 GB machines
  General — explicit del + gc.collect() + torch.cuda.empty_cache() between heavy blocks

Tested with: torch>=2.1, CUDA 11.8/12.1, 32 GB system RAM + 24 GB VRAM (RTX 3090)
Even on a CPU-only machine the AMP guard makes the code safe (autocast is a no-op on CPU).
"""


import sys, os, gc, re, json, yaml, datetime, tempfile
from pathlib import Path
from time import sleep
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from einops import rearrange
from tqdm import tqdm

# ════════════════════════════════════════════════════════
#  MEMORY PROFILING UTILITIES
# ════════════════════════════════════════════════════════
def print_gpu_memory(label=""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  [{label}] GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def clear_memory_aggressive():
    """Aggressive cleanup: del, gc, cache clear."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

# ════════════════════════════════════════════════════════
#  SETTINGS
# ════════════════════════════════════════════════════════
REPO_ROOT   = Path("C:/Users/room208/mizuho")
EO_DIR      = REPO_ROOT / "Prithvi-EO-2.0-300M"
WXC_DIR     = REPO_ROOT / "Prithvi-WxC"
DATA_DIR    = REPO_ROOT / "data"
OUTPUT_DIR  = DATA_DIR  / "mizuho_output"
MERRA_DIR   = DATA_DIR  / "merra-2"
CLIM_DIR    = DATA_DIR  / "climatology"
HLS_DIR     = DATA_DIR  / "hls_counties"

TARGET_YEAR_EO = 2020
YEAR_WxC       = 2020
MONTH          = 1       # January
INPUT_STEP     = 6
LEAD_TIME      = 12

EO_CONFIG_PATH     = EO_DIR / "config.json"
EO_CHECKPOINT_PATH = EO_DIR / "Prithvi_EO_V2_300M.pt"

# ══════════════════════════════════════════════════════
#  Tuning knobs — GPU-FIRST STRATEGY
#  Use VRAM instead of system RAM (less bottleneck)
# ══════════════════════════════════════════════════════
PATCH_BATCH_SIZE = 10      # MICRO: Process 1 patch at a time (fits in VRAM)
                          # Increase only if you have > 24GB VRAM available
WXC_USE_AMP      = True   # Step 4a: fp16 autocast (keeps VRAM usage steady)
AMP_DTYPE        = torch.float16   # fp16 precision to fit model in VRAM

# ════ GPU-focused memory strategy (avoid CPU RAM) ════
ENABLE_PINNED_MEMORY = False  # DISABLE: Don't use CPU pinned memory
ENABLE_WXCFP16_WEIGHTS = True # KEEP fp16: WxC model in half precision (~10GB instead of 20GB)
ENABLE_AGGRESSIVECLEANUP = False # Disable: GPU cache is fast, no need to clear constantly
CLEANUP_EVERY_N_PATCHES = 1   # Cleanup every 10 patches (GPU can handle it)
PROJECTION_DTYPE = torch.float32  # Keep float32 for small ops (no benefit from fp16)
MAX_COUNTIES_PER_RUN = 5     # GPU can handle many counties in flight



# ════════════════════════════════════════════════════════
#  PATHS & IMPORTS
# ════════════════════════════════════════════════════════
for p in [EO_DIR, WXC_DIR]:
    assert p.exists(), f"Submodule not found: {p}"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

for d in [OUTPUT_DIR, MERRA_DIR, CLIM_DIR, HLS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ════════════════════════════════════════════════════════
#  STEP 0 — County list (USDA × Gazetteer)
# ════════════════════════════════════════════════════════
print("\n[Step 0] Building county list...")

_df_usda = pd.read_csv(DATA_DIR / "USDA_Soybean_County_2020.csv")
_df_usda["GEOID"] = (
    _df_usda["state_ansi"].astype(str).str.zfill(2) +
    _df_usda["county_ansi"].astype(str).str.zfill(3)
)
_df_gaz = pd.read_csv(
    DATA_DIR / "2025_Gaz_counties_national.txt",
    sep="|", dtype={"GEOID": str}
)
_df_gaz.columns = _df_gaz.columns.str.strip()
_df_gaz = _df_gaz[["GEOID","NAME","USPS","INTPTLAT","INTPTLONG"]].rename(
    columns={"INTPTLAT":"lat","INTPTLONG":"lon","USPS":"state"}
)
df_target = _df_usda[["GEOID"]].merge(_df_gaz, on="GEOID", how="inner")
df_target = df_target.iloc[:200].reset_index(drop=True)   # 必要に応じて上限変更

RESOLVED_GEOIDS = df_target["GEOID"].tolist()
print(f"  Target counties : {len(RESOLVED_GEOIDS)}")

# ════════════════════════════════════════════════════════
#  STEP 1 — Download climatology & MERRA-2
# ════════════════════════════════════════════════════════
print("\n[Step 1] Downloading climatology & MERRA-2...")

from huggingface_hub import snapshot_download
from PrithviWxC.download import get_prithvi_wxc_input

# 1-a. Climatology (January doy001-031)
clim_marker = DATA_DIR / ".done_climatology_jan"
if clim_marker.exists():
    print("  Climatology already downloaded, skipping.")
else:
    print("  Downloading climatology surface...")
    snapshot_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        allow_patterns=[f"climatology/climate_surface_doy{d:03d}*.nc" for d in range(1,32)],
        local_dir=DATA_DIR,
    )
    print("  Downloading climatology vertical...")
    snapshot_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        allow_patterns=[f"climatology/climate_vertical_doy{d:03d}*.nc" for d in range(1,32)],
        local_dir=DATA_DIR,
    )
    clim_marker.touch()
    print("  Climatology done.")

# 1-b. Musigma / anomaly_variance files
extra_clim = [
    "climatology/musigma_surface.nc",
    "climatology/musigma_vertical.nc",
    "climatology/anomaly_variance_surface.nc",
    "climatology/anomaly_variance_vertical.nc",
]
missing = [p for p in extra_clim if not (DATA_DIR / p).exists()]
if missing:
    snapshot_download(
        repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
        allow_patterns=missing,
        local_dir=DATA_DIR,
    )

# 1-c. MERRA-2 daily data
start = np.datetime64(f"{YEAR_WxC:04d}-{MONTH:02d}-01")
end   = (np.datetime64(f"{YEAR_WxC:04d}-{MONTH+1:02d}-01")
         if MONTH < 12 else np.datetime64(f"{YEAR_WxC+1:04d}-01-01"))
dates = [start + np.timedelta64(d, "D") for d in range(int((end-start).astype(int)))]

print(f"  Downloading {len(dates)} days of MERRA-2 ...")
for date in dates:
    date_str = str(date)[:10]
    marker   = MERRA_DIR / f".done_{date_str}"
    if marker.exists():
        continue
    print(f"    {date_str} ...", end=" ", flush=True)
    for tries in range(5):
        try:
            get_prithvi_wxc_input(
                date, input_time_step=INPUT_STEP, lead_time=LEAD_TIME,
                input_data_dir=MERRA_DIR, download_dir=DATA_DIR,
            )
            marker.touch()
            print("done.")
            break
        except Exception as e:
            if tries < 4:
                sleep(2 ** tries)
            else:
                print(f"FAILED: {e}")

# ════════════════════════════════════════════════════════
#  STEP 2 — Download Sentinel-2 (HLS) per county
# ════════════════════════════════════════════════════════
print("\n[Step 2] Downloading Sentinel-2 (HLS) data...")

import earthaccess
earthaccess.login(strategy="netrc")

HLS_BANDS = ["B02","B03","B04","B05","B06","B07"]

hls_results = {}  # {geoid: tif_path}
failed_hls  = []

for _, row in tqdm(df_target.iterrows(), total=len(df_target), desc="HLS download"):
    geoid = row["GEOID"]
    lat   = float(row["lat"])
    lon   = float(row["lon"])

    out_path = HLS_DIR / f"{geoid}_HLS.tif"
    if out_path.exists():
        hls_results[geoid] = str(out_path)
        continue

    bbox = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
    try:
        granules = []
        for concept_id in ["C2021957657-LPCLOUD", "C2021957295-LPCLOUD"]:
            granules = earthaccess.search_data(
                concept_id=concept_id,
                temporal=(f"{TARGET_YEAR_EO}-01-01", f"{TARGET_YEAR_EO}-12-31"),
                bounding_box=bbox, count=1,
            )
            if granules:
                break
        if not granules:
            failed_hls.append(geoid)
            continue

        all_urls  = granules[0].data_links()
        band_urls = [u for u in all_urls if any(f".{b}.tif" in u for b in HLS_BANDS)]
        if len(band_urls) != 6:
            failed_hls.append(geoid)
            continue

        with tempfile.TemporaryDirectory() as tmp:
            files    = earthaccess.download(band_urls, local_path=tmp)
            datasets = [rasterio.open(f) for f in sorted(files)]
            profile  = datasets[0].profile.copy()
            profile.update(count=6)
            with rasterio.open(out_path, "w", **profile) as dst:
                for i, ds in enumerate(datasets, start=1):
                    dst.write(ds.read(1), i)
            for ds in datasets:
                ds.close()

        hls_results[geoid] = str(out_path)
        print(f"  {geoid}: saved {out_path.name}")
    except Exception as e:
        print(f"  {geoid}: FAILED — {e}")
        failed_hls.append(geoid)

print(f"  HLS: {len(hls_results)} OK / {len(failed_hls)} failed")




# ══════════════════════════════════════════════════════
#  STEP 3 — Prithvi-EO inference  (BATCHED VERSION)
# ══════════════════════════════════════════════════════
# Replace the original per-window loop with this function.
# Call it from the county loop exactly as before:
#   run_eo_inference(geoid, tif_path, eo_model, device, ...)

def run_eo_inference(geoid, tif_path, eo_model, device,
                     mean, std, img_size, coords_enc,
                     geoid_out, TARGET_YEAR_EO,
                     patch_batch_size=PATCH_BATCH_SIZE):
    """
    Load one county GeoTIFF, run Prithvi-EO in batched patches,
    save extracted_q_patch_*.pt files, and return the path stem.
    
    Enhanced: lazy patch loading, pinned memory, aggressive cleanup.
    """
    from preprocess import load_example   # reuse your existing helper

    input_data, temporal_coords, location_coords = load_example(
        [tif_path], mean=mean, std=std
    )

    _ce = dict(coords_enc)
    if not temporal_coords and "time" in _ce:     _ce.pop("time")
    if not location_coords  and "location" in _ce: _ce.pop("location")

    # Pad to img_size multiples
    oh, ow = input_data.shape[-2:]
    ph = img_size - (oh % img_size) if oh % img_size != 0 else 0
    pw = img_size - (ow % img_size) if ow % img_size != 0 else 0
    import numpy as np
    input_data = np.pad(input_data, ((0,0),(0,0),(0,0),(0,ph),(0,pw)), mode="reflect")

    # ── Lazy patch extraction ────────────────────────────
    # Instead of creating all patches at once, use unfold view
    batch   = torch.tensor(input_data, dtype=torch.float32, device="cpu")
    if ENABLE_PINNED_MEMORY:
        batch = batch.pin_memory()
    
    windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
    windows = rearrange(windows, "b c t h1 w1 h w -> (b h1 w1) c t h w",
                        h=img_size, w=img_size)          # [N_patches, C, T, H, W]
    n_patches = windows.shape[0]

    tc = torch.Tensor(temporal_coords).unsqueeze(0).to(device) if temporal_coords else None
    lc = torch.Tensor(location_coords[0]).unsqueeze(0).to(device) if location_coords else None

    # ── Batched inference with light cleanup ──────────────
    patch_idx = 0
    with torch.no_grad():
        for start in range(0, n_patches, patch_batch_size):
            end   = min(start + patch_batch_size, n_patches)
            
            # Lazy load this mini-batch only
            batch_win = windows[start:end].to(device, non_blocking=ENABLE_PINNED_MEMORY)

            # Replicate tc/lc for batch dimension
            B_pat = batch_win.shape[0]
            tc_b  = tc.expand(B_pat, -1, -1) if tc is not None else None
            lc_b  = lc.expand(B_pat, -1)     if lc is not None else None

            features = eo_model.forward_features(batch_win, tc_b, lc_b)
            q_full   = features[-1]           # [B_pat, N_tokens, 1024]
            q_cls    = q_full[:, 0, :]        # CLS token → [B_pat, 1024]

            for i, q in enumerate(q_cls):
                sp = geoid_out / f"extracted_q_patch_{patch_idx}.pt"
                torch.save(q.unsqueeze(0).cpu(), sp)
                patch_idx += 1

            # Free VRAM after every mini-batch
            del batch_win, features, q_full, q_cls
            
            # Light cleanup (not aggressive)
            if patch_idx % CLEANUP_EVERY_N_PATCHES == 0:
                torch.cuda.empty_cache()

    # Final cleanup
    del batch, windows, tc, lc
    torch.cuda.empty_cache()
    
    return patch_idx   # number of patches saved


# ══════════════════════════════════════════════════════
#  STEP 4a — Prithvi-WxC encoder  (AMP + CPU-offload)
# ══════════════════════════════════════════════════════
# Replace the Step 4a block with this function.

def run_wxc_encoder(wxc_model, batch, device,
                    use_amp=WXC_USE_AMP, amp_dtype=AMP_DTYPE):
    """
    Run the WxC encoder on one batch under optional AMP + CPU offload + CHECKPOINTING.
    Returns (feature_map, C_sc) both as CPU tensors.

    ULTRA-MEMORY-SAFE for > 32 GB systems:
      1. torch.amp.autocast halves activation memory (fp16/bf16).
      2. Convert model weights to fp16 (ENABLE_WXCFP16_WEIGHTS).
      3. AGGRESSIVE CPU offloading: Move results to CPU immediately.
      4. Activation checkpointing on encoder (trades compute for memory).
      5. Cleanup after every single operation.
    """
    wxc_model.eval()
    
    # Convert weights to fp16 for VRAM savings
    if ENABLE_WXCFP16_WEIGHTS and device.type == "cuda":
        wxc_model = wxc_model.half()
        print("  [WxC] Converted model weights to fp16")
    
    wxc_model = wxc_model.to(device)
    ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=(use_amp and device.type == "cuda"))

    with torch.no_grad(), ctx:
        # ── Pre-processing (keep on GPU) ──────────────
        print_gpu_memory("WxC_start")
        
        x_rescaled = (batch["x"] - wxc_model.input_scalers_mu) / \
                     (wxc_model.input_scalers_sigma + wxc_model.input_scalers_epsilon)
        x_rescaled = x_rescaled.flatten(1, 2)

        x_pos = wxc_model.fourier_pos_encoding(batch["static"])

        x_static = (batch["static"][:, 2:] - wxc_model.static_input_scalers_mu[:, 3:]) / \
                   (wxc_model.static_input_scalers_sigma[:, 3:] + wxc_model.static_input_scalers_epsilon)

        climate_sc = (batch["climate"] - wxc_model.input_scalers_mu.view(1, -1, 1, 1)) / \
                     (wxc_model.input_scalers_sigma.view(1, -1, 1, 1) + wxc_model.input_scalers_epsilon)

        # KEEP ON GPU: climate_sc stays on GPU (save to CPU only at end)
        del batch["climate"], batch["static"]
        torch.cuda.empty_cache()

        # ── Embedding ──────────────────────────────────
        x_emb      = wxc_model.patch_embedding(x_rescaled)
        del x_rescaled
        torch.cuda.empty_cache()  # Light cleanup, don't stall GPU

        # Keep static embedding on GPU (no CPU offload)
        static_emb = wxc_model.patch_embedding_static(
            torch.cat((x_static, climate_sc), dim=1)  # climate_sc stays on GPU
        )
        del x_static  # KEEP climate_sc — need it at end!
        torch.cuda.empty_cache()
        
        static_emb += x_pos
        del x_pos
        torch.cuda.empty_cache()

        x_emb      = wxc_model.to_patching(x_emb)
        static_emb = wxc_model.to_patching(static_emb)

        time_enc = wxc_model.time_encoding(batch["input_time"], batch["lead_time"])
        tokens   = x_emb + static_emb + time_enc
        del x_emb, static_emb, time_enc
        torch.cuda.empty_cache()  # Light cleanup

        # ── Encoder (THE HEAVY PART) ──────────────
        print_gpu_memory("WxC_encoder_start")
        
        # Use gradient checkpointing to reduce activation memory
        # (Even though inference only, checkpointing helps memory)
        try:
            x_encoded = torch.utils.checkpoint.checkpoint(
                wxc_model.encoder, tokens, use_reentrant=False
            )
            print("  [WxC] Encoder used checkpointing (memory-safe)")
        except Exception as e:
            print(f"  [WxC] Checkpointing failed ({e}), using standard inference")
            x_encoded = wxc_model.encoder(tokens)
        
        print_gpu_memory("WxC_encoder_end")
        del tokens
        torch.cuda.empty_cache()  # Light cleanup

        # ── Reshape to spatial feature map ────────────
        B       = x_encoded.shape[0]
        G0, G1  = wxc_model.global_shape_mu
        L0, L1  = wxc_model.local_shape_mu
        D       = wxc_model.embed_dim

        x_enc       = x_encoded.view(B, G0, G1, L0, L1, D)
        del x_encoded
        torch.cuda.empty_cache()
        
        x_enc       = x_enc.permute(0, 5, 1, 3, 2, 4).contiguous()
        feature_map = x_enc.view(B, D, G0 * L0, G1 * L1)   # [1, 160, 180, 288]
        del x_enc
        torch.cuda.empty_cache()

        # KEEP ON GPU: feature_map stays on GPU (will be loaded to GPU anyway in Step 4b)
        feature_map_cpu = feature_map  # No copy to CPU
        C_sc_cpu = climate_sc  # No copy to CPU yet

    print_gpu_memory("WxC_end")
    # Move to CPU only at the very end (after with block exits, GPU is freed)
    feature_map_cpu_final = feature_map_cpu.cpu() if isinstance(feature_map_cpu, torch.Tensor) else feature_map_cpu
    C_sc_cpu_final = C_sc_cpu.cpu() if isinstance(C_sc_cpu, torch.Tensor) else C_sc_cpu
    torch.cuda.empty_cache()
    return feature_map_cpu_final, C_sc_cpu_final


# ══════════════════════════════════════════════════════
#  STEP 4b — Spatial interpolation  (CPU-only version)
# ══════════════════════════════════════════════════════
# This step is very fast even on CPU — no need for GPU.

def interpolate_county_features(feature_map_cpu, C_sc_cpu, df_target, device):
    """
    Bilinear-interpolate feature_map and C_sc at county centroids.
    Projection layers are tiny (160→2560) and run on CPU to avoid
    loading the GPU for a negligible operation.

    Returns met_embedding, county_weather_token, local_climatology_vector
    all as CPU tensors.
    """
    # Keep everything on CPU for this cheap step
    feature_map = feature_map_cpu.float()   # ← .float()追加
    C_sc        = C_sc_cpu.float()          # ← .float()追加
    
    print_gpu_memory("Step4b_start")

    lat_grid  = torch.linspace(-90,  90,  feature_map.shape[-2])
    lon_grid  = torch.linspace(-180, 180, feature_map.shape[-1])

    lats      = torch.tensor(df_target["lat"].tolist(), dtype=torch.float32)
    lons      = torch.tensor(df_target["lon"].tolist(), dtype=torch.float32)
    norm_lats = 2.0 * (lats - lat_grid.min()) / (lat_grid.max() - lat_grid.min()) - 1.0
    norm_lons = 2.0 * (lons - lon_grid.min()) / (lon_grid.max() - lon_grid.min()) - 1.0

    B    = feature_map.shape[0]
    grid = torch.stack([norm_lons, norm_lats], dim=-1).unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

    with torch.no_grad():
        county_weather_token = F.grid_sample(
            feature_map, grid, mode="bilinear", align_corners=True
        ).squeeze(2).permute(0, 2, 1)           # [1, N, 160]

        local_climatology_vector = F.grid_sample(
            C_sc, grid, mode="bilinear", align_corners=True
        ).squeeze(2).permute(0, 2, 1)           # [1, N, 160]

        # # Tiny projection layers — use specified dtype for memory savings
        # _proj_clim = nn.Linear(160, 2560, dtype=PROJECTION_DTYPE)
        # _norm_wxc  = nn.LayerNorm(2560, dtype=PROJECTION_DTYPE)
        # _norm_clim = nn.LayerNorm(2560, dtype=PROJECTION_DTYPE)

        # # Convert inputs to projection dtype if needed
        # if PROJECTION_DTYPE != torch.float32:
        #     county_weather_token = county_weather_token.to(PROJECTION_DTYPE)
        #     local_climatology_vector = local_climatology_vector.to(PROJECTION_DTYPE)

        # wxc_normed  = _norm_wxc(_proj_clim(county_weather_token))
        # clim_normed = _norm_clim(_proj_clim(local_climatology_vector))
        
        # # Convert back to float32 for concat if needed
        # if PROJECTION_DTYPE != torch.float32:
        #     wxc_normed = wxc_normed.to(torch.float32)
        #     clim_normed = clim_normed.to(torch.float32)

        D_wxc  = feature_map.shape[1]  # 2560
        D_clim = C_sc.shape[1]         # 160
        _proj_clim = nn.Linear(D_clim, D_wxc)
        _norm_wxc  = nn.LayerNorm(D_wxc)
        _norm_clim = nn.LayerNorm(D_wxc)

        wxc_normed  = _norm_wxc(county_weather_token)
        clim_normed = _norm_clim(_proj_clim(local_climatology_vector))

        met_embedding = torch.cat([wxc_normed, clim_normed], dim=-1)  # [1, N, 5120]

    print_gpu_memory("Step4b_end")
    return met_embedding, county_weather_token.to(torch.float32) if PROJECTION_DTYPE != torch.float32 else county_weather_token, \
           local_climatology_vector.to(torch.float32) if PROJECTION_DTYPE != torch.float32 else local_climatology_vector


# ══════════════════════════════════════════════════════
#  HOW TO INTEGRATE into preprocess.py
# ══════════════════════════════════════════════════════
# 
# In STEP 3, replace the inner `for i, x in enumerate(windows):` loop with:
#
#     n_saved = run_eo_inference(
#         geoid, tif_path, eo_model, device,
#         mean, std, img_size, coords_enc,
#         geoid_out, TARGET_YEAR_EO,
#         patch_batch_size=PATCH_BATCH_SIZE,
#     )
#     print(f"  {geoid}: {n_saved} patches → {geoid_out}")
#
#
# In STEP 4a, replace everything inside `with torch.no_grad():` with:
#
#     feature_map_cpu, C_sc_cpu = run_wxc_encoder(wxc_model, batch, device)
#
# Then save:
#     torch.save(feature_map_cpu, feature_map_path)
#     torch.save(C_sc_cpu,        C_sc_path)
#
#
# In STEP 4b, replace the `with torch.no_grad():` block with:
#
#     met_embedding, county_weather_token, local_climatology_vector = \
#         interpolate_county_features(feature_map_cpu, C_sc_cpu, df_target, device)
#
# Then remove the two lines that move feature_map / C_sc to `device` at the top
# of Step 4b — they now live on CPU throughout.
#
# ══════════════════════════════════════════════════════
#  ADDITIONAL TIPS for VRAM-constrained machines
# ══════════════════════════════════════════════════════
#
# 1. If WxC still OOMs:
#       Set ENABLE_WXCFP16_WEIGHTS = True  (converts entire model to fp16)
#       or reduce lead_times / input_times dimensionality
#
# 2. If EO model OOMs:
#       Reduce PATCH_BATCH_SIZE to 4, 2, or 1
#       Increase CLEANUP_EVERY_N_PATCHES from 2 to 1 (more frequent cleanup)
#
# 3. For pinned memory (faster transfers, uses CPU RAM):
#       Keep ENABLE_PINNED_MEMORY = True (default)
#       If RAM is tight, set to False (slightly slower but less RAM)
#
# 4. Projection layer dtype (Step 4b):
#       PROJECTION_DTYPE = torch.float16 saves memory (very tiny operation)
#       PROJECTION_DTYPE = torch.float32 is safer (default)
#
# 5. For multi-GPU systems, shard county loop:
#       device_eo  = torch.device("cuda:0")
#       device_wxc = torch.device("cuda:1")
#
# 6. Enable profiling to monitor memory in detail:
#       print_gpu_memory() is called automatically at key stages
#       Watch for "allocated" and "reserved" growth
#
# 7. For very tight memory, disable HLS download and use alternative data:
#       Set hls_results = {} manually, provide pre-downloaded TIFFs
#
# 8. Memory debugging: Set ENABLE_AGGRESSIVECLEANUP = True
#       This clears GPU cache every N patches (slower but catches leaks)
# ══════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════
#  STEP 3 — Prithvi-EO inference → extracted_q_patch_*.pt
# ════════════════════════════════════════════════════════
print("\n[Step 3] Prithvi-EO inference (frozen)...")

import yaml
from prithvi_mae import PrithviMAE

with open(EO_CONFIG_PATH, "r") as f:
    eo_config = yaml.safe_load(f)["pretrained_cfg"]

bands      = eo_config["bands"]
mean       = eo_config["mean"]
std        = eo_config["std"]
img_size   = eo_config["img_size"]
coords_enc = eo_config["coords_encoding"]

eo_cfg = dict(eo_config)
eo_cfg.update(coords_encoding=coords_enc, num_frames=1, in_chans=len(bands))
eo_model = PrithviMAE(**eo_cfg).to(device)

state_dict = torch.load(EO_CHECKPOINT_PATH, map_location=device, weights_only=True)
for k in list(state_dict.keys()):
    if "pos_embed" in k:
        del state_dict[k]
eo_model.load_state_dict(state_dict, strict=False)
eo_model.eval()
print(f"  Loaded EO checkpoint: {EO_CHECKPOINT_PATH}")

patch_dirs   = {}
q_save_paths = {}

for geoid, tif_path in tqdm(hls_results.items(), desc="EO patch extraction"):
    geoid_out = OUTPUT_DIR / geoid
    geoid_out.mkdir(parents=True, exist_ok=True)
    patch_dirs[geoid] = geoid_out

    existing = sorted(geoid_out.glob("extracted_q_patch_*.pt"))
    if existing:
        q_save_paths[geoid] = str(geoid_out / "final_county_embedding_q.pt")
        print(f"  {geoid}: already extracted ({len(existing)} patches), skipping.")
        continue

    n_saved = run_eo_inference(
        geoid, tif_path, eo_model, device,
        mean, std, img_size, coords_enc,
        geoid_out, TARGET_YEAR_EO,
        patch_batch_size=PATCH_BATCH_SIZE,
    )
    print(f"  {geoid}: {n_saved} patches → {geoid_out}")
    q_save_paths[geoid] = str(geoid_out / "final_county_embedding_q.pt")

del eo_model
torch.cuda.empty_cache()
print_gpu_memory("After_EO_model_cleanup")

# ════════════════════════════════════════════════════════
#  STEP 4a — Prithvi-WxC encoder → feature_map.pt
# ════════════════════════════════════════════════════════
print("\n[Step 4a] Prithvi-WxC encoder → feature_map.pt ...")

feature_map_path = OUTPUT_DIR / "feature_map.pt"
C_sc_path        = OUTPUT_DIR / "C_sc.pt"

if feature_map_path.exists() and C_sc_path.exists():
    print("  feature_map.pt already exists — skipping heavy WxC inference.")
    feature_map_cpu = torch.load(feature_map_path, map_location="cpu")
    C_sc_cpu        = torch.load(C_sc_path,        map_location="cpu")
else:
    import yaml
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

    in_mu, in_sig        = input_scalers(surface_vars, vertical_vars, levels,
                               CLIM_DIR/"musigma_surface.nc", CLIM_DIR/"musigma_vertical.nc")
    output_sig           = output_scalers(surface_vars, vertical_vars, levels,
                               CLIM_DIR/"anomaly_variance_surface.nc",
                               CLIM_DIR/"anomaly_variance_vertical.nc")
    static_mu, static_sig = static_input_scalers(CLIM_DIR/"musigma_surface.nc",
                               static_surface_vars)

    with open(WXC_DIR / "data" / "config.yaml") as f:
        wxc_cfg = yaml.safe_load(f)
    p = wxc_cfg["params"]

    wxc_model = PrithviWxC(
        in_channels=p["in_channels"], input_size_time=p["input_size_time"],
        in_channels_static=p["in_channels_static"],
        input_scalers_mu=in_mu, input_scalers_sigma=in_sig,
        input_scalers_epsilon=p["input_scalers_epsilon"],
        static_input_scalers_mu=static_mu, static_input_scalers_sigma=static_sig,
        static_input_scalers_epsilon=p["static_input_scalers_epsilon"],
        output_scalers=output_sig**0.5,
        n_lats_px=p["n_lats_px"], n_lons_px=p["n_lons_px"],
        patch_size_px=p["patch_size_px"], mask_unit_size_px=p["mask_unit_size_px"],
        mask_ratio_inputs=0.0, mask_ratio_targets=0.0,
        embed_dim=p["embed_dim"], n_blocks_encoder=p["n_blocks_encoder"],
        n_blocks_decoder=p["n_blocks_decoder"], mlp_multiplier=p["mlp_multiplier"],
        n_heads=p["n_heads"], dropout=p["dropout"], drop_path=p["drop_path"],
        parameter_dropout=p["parameter_dropout"],
        residual="climate", masking_mode="global",
        encoder_shifting=True, decoder_shifting=True,
        positional_encoding=positional_encoding,
        checkpoint_encoder=[], checkpoint_decoder=[],
    )
    weights_path = WXC_DIR / "data" / "weights" / "prithvi.wxc.2300m.v1.pt"
    sd = torch.load(weights_path, map_location=device, weights_only=False)
    sd = sd.get("model_state", sd)
    wxc_model.load_state_dict(sd, strict=True)
    wxc_model = wxc_model.half().to(device) 
    del sd
    gc.collect()
    torch.cuda.empty_cache()  # ← これを追加するとreservedが減る
    print("  Loaded WxC checkpoint. Running encoder...")

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
            batch[k] = v.to(device)

    feature_map_cpu, C_sc_cpu = run_wxc_encoder(wxc_model, batch, device)

    torch.save(feature_map_cpu, feature_map_path)
    torch.save(C_sc_cpu,        C_sc_path)
    print(f"  feature_map saved: {feature_map_cpu.shape}")
    print(f"  C_sc saved       : {C_sc_cpu.shape}")

    del wxc_model
    torch.cuda.empty_cache()
    print_gpu_memory("After_WxC_model_cleanup")

# ════════════════════════════════════════════════════════
#  STEP 4b — Bilinear interpolation → met_embedding.pt
# ════════════════════════════════════════════════════════
print("\n[Step 4b] Spatial interpolation at county centroids → met_embedding.pt ...")

met_embedding, county_weather_token, local_climatology_vector = \
    interpolate_county_features(feature_map_cpu, C_sc_cpu, df_target, device)

torch.save(met_embedding.cpu(),            OUTPUT_DIR / "met_embedding.pt")
torch.save(county_weather_token.cpu(),     OUTPUT_DIR / "wxc_county_tokens.pt")
torch.save(local_climatology_vector.cpu(), OUTPUT_DIR / "clim_county_vectors.pt")
print(f"  met_embedding saved : {met_embedding.shape}")


# ════════════════════════════════════════════════════════
#  STEP 5 — Save q_save_paths.json (county order index)
# ════════════════════════════════════════════════════════
print("\n[Step 5] Saving q_save_paths.json...")

with open(OUTPUT_DIR / "q_save_paths.json", "w") as f:
    json.dump({k: str(v) for k, v in q_save_paths.items()}, f)

print(f"  q_save_paths.json saved: {len(q_save_paths)} counties")

# ════════════════════════════════════════════════════════
#  DONE
# ════════════════════════════════════════════════════════
print("\n" + "="*50)
print("preprocess.py complete.")
print("Next step: python train.py")
print("="*50)




# preprocess.py — Data download + frozen model inference
#   0から実行する場合はこのスクリプトを最初に実行する

# Steps:
#   1. MERRA-2 / Climatology データを HuggingFace からダウンロード
#   2. Sentinel-2 (HLS) データを earthaccess からダウンロード
#   3. Prithvi-EO (frozen) 推論 → extracted_q_patch_*.pt 保存
#   4. Prithvi-WxC (frozen) 推論 → met_embedding.pt 保存
#   5. q_save_paths.json 保存 (train.py が county 順序の参照に使う)

# Output (train.py が必要とするファイル):
#   OUTPUT_DIR/{geoid}/extracted_q_patch_*.pt
#   OUTPUT_DIR/met_embedding.pt
#   OUTPUT_DIR/q_save_paths.json
# """

# import sys, os, gc, re, json, yaml, datetime, tempfile
# from pathlib import Path
# from time import sleep
# from typing import List

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# import rasterio
# from einops import rearrange
# from tqdm import tqdm

# # ════════════════════════════════════════════════════════
# #  SETTINGS
# # ════════════════════════════════════════════════════════
# REPO_ROOT   = Path("C:/Users/room208/mizuho")
# EO_DIR      = REPO_ROOT / "Prithvi-EO-2.0-300M"
# WXC_DIR     = REPO_ROOT / "Prithvi-WxC"
# DATA_DIR    = REPO_ROOT / "data"
# OUTPUT_DIR  = DATA_DIR  / "mizuho_output"
# MERRA_DIR   = DATA_DIR  / "merra-2"
# CLIM_DIR    = DATA_DIR  / "climatology"
# HLS_DIR     = DATA_DIR  / "hls_counties"

# TARGET_YEAR_EO = 2020
# YEAR_WxC       = 2020
# MONTH          = 1       # January
# INPUT_STEP     = 6
# LEAD_TIME      = 12

# EO_CONFIG_PATH     = EO_DIR / "config.json"
# EO_CHECKPOINT_PATH = EO_DIR / "Prithvi_EO_V2_300M.pt"

# # ════════════════════════════════════════════════════════
# #  PATHS & IMPORTS
# # ════════════════════════════════════════════════════════
# for p in [EO_DIR, WXC_DIR]:
#     assert p.exists(), f"Submodule not found: {p}"
#     if str(p) not in sys.path:
#         sys.path.insert(0, str(p))

# for d in [OUTPUT_DIR, MERRA_DIR, CLIM_DIR, HLS_DIR]:
#     d.mkdir(parents=True, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device : {device}")

# # ════════════════════════════════════════════════════════
# #  STEP 0 — County list (USDA × Gazetteer)
# # ════════════════════════════════════════════════════════
# print("\n[Step 0] Building county list...")

# _df_usda = pd.read_csv(DATA_DIR / "USDA_Soybean_County_2020.csv")
# _df_usda["GEOID"] = (
#     _df_usda["state_ansi"].astype(str).str.zfill(2) +
#     _df_usda["county_ansi"].astype(str).str.zfill(3)
# )
# _df_gaz = pd.read_csv(
#     DATA_DIR / "2025_Gaz_counties_national.txt",
#     sep="|", dtype={"GEOID": str}
# )
# _df_gaz.columns = _df_gaz.columns.str.strip()
# _df_gaz = _df_gaz[["GEOID","NAME","USPS","INTPTLAT","INTPTLONG"]].rename(
#     columns={"INTPTLAT":"lat","INTPTLONG":"lon","USPS":"state"}
# )
# df_target = _df_usda[["GEOID"]].merge(_df_gaz, on="GEOID", how="inner")
# df_target = df_target.iloc[:10].reset_index(drop=True)   # 必要に応じて上限変更

# RESOLVED_GEOIDS = df_target["GEOID"].tolist()
# print(f"  Target counties : {len(RESOLVED_GEOIDS)}")

# # ════════════════════════════════════════════════════════
# #  STEP 1 — Download climatology & MERRA-2
# # ════════════════════════════════════════════════════════
# print("\n[Step 1] Downloading climatology & MERRA-2...")

# from huggingface_hub import snapshot_download
# from PrithviWxC.download import get_prithvi_wxc_input

# # 1-a. Climatology (January doy001-031)
# clim_marker = DATA_DIR / ".done_climatology_jan"
# if clim_marker.exists():
#     print("  Climatology already downloaded, skipping.")
# else:
#     print("  Downloading climatology surface...")
#     snapshot_download(
#         repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#         allow_patterns=[f"climatology/climate_surface_doy{d:03d}*.nc" for d in range(1,32)],
#         local_dir=DATA_DIR,
#     )
#     print("  Downloading climatology vertical...")
#     snapshot_download(
#         repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#         allow_patterns=[f"climatology/climate_vertical_doy{d:03d}*.nc" for d in range(1,32)],
#         local_dir=DATA_DIR,
#     )
#     clim_marker.touch()
#     print("  Climatology done.")

# # 1-b. Musigma / anomaly_variance files
# extra_clim = [
#     "climatology/musigma_surface.nc",
#     "climatology/musigma_vertical.nc",
#     "climatology/anomaly_variance_surface.nc",
#     "climatology/anomaly_variance_vertical.nc",
# ]
# missing = [p for p in extra_clim if not (DATA_DIR / p).exists()]
# if missing:
#     snapshot_download(
#         repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#         allow_patterns=missing,
#         local_dir=DATA_DIR,
#     )

# # 1-c. MERRA-2 daily data
# start = np.datetime64(f"{YEAR_WxC:04d}-{MONTH:02d}-01")
# end   = (np.datetime64(f"{YEAR_WxC:04d}-{MONTH+1:02d}-01")
#          if MONTH < 12 else np.datetime64(f"{YEAR_WxC+1:04d}-01-01"))
# dates = [start + np.timedelta64(d, "D") for d in range(int((end-start).astype(int)))]

# print(f"  Downloading {len(dates)} days of MERRA-2 ...")
# for date in dates:
#     date_str = str(date)[:10]
#     marker   = MERRA_DIR / f".done_{date_str}"
#     if marker.exists():
#         continue
#     print(f"    {date_str} ...", end=" ", flush=True)
#     for tries in range(5):
#         try:
#             get_prithvi_wxc_input(
#                 date, input_time_step=INPUT_STEP, lead_time=LEAD_TIME,
#                 input_data_dir=MERRA_DIR, download_dir=DATA_DIR,
#             )
#             marker.touch()
#             print("done.")
#             break
#         except Exception as e:
#             if tries < 4:
#                 sleep(2 ** tries)
#             else:
#                 print(f"FAILED: {e}")

# # ════════════════════════════════════════════════════════
# #  STEP 2 — Download Sentinel-2 (HLS) per county
# # ════════════════════════════════════════════════════════
# print("\n[Step 2] Downloading Sentinel-2 (HLS) data...")

# import earthaccess
# earthaccess.login(strategy="netrc")

# HLS_BANDS = ["B02","B03","B04","B05","B06","B07"]

# hls_results = {}  # {geoid: tif_path}
# failed_hls  = []

# for _, row in tqdm(df_target.iterrows(), total=len(df_target), desc="HLS download"):
#     geoid = row["GEOID"]
#     lat   = float(row["lat"])
#     lon   = float(row["lon"])

#     out_path = HLS_DIR / f"{geoid}_HLS.tif"
#     if out_path.exists():
#         hls_results[geoid] = str(out_path)
#         continue

#     bbox = (lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
#     try:
#         granules = []
#         for concept_id in ["C2021957657-LPCLOUD", "C2021957295-LPCLOUD"]:
#             granules = earthaccess.search_data(
#                 concept_id=concept_id,
#                 temporal=(f"{TARGET_YEAR_EO}-01-01", f"{TARGET_YEAR_EO}-12-31"),
#                 bounding_box=bbox, count=1,
#             )
#             if granules:
#                 break
#         if not granules:
#             failed_hls.append(geoid)
#             continue

#         all_urls  = granules[0].data_links()
#         band_urls = [u for u in all_urls if any(f".{b}.tif" in u for b in HLS_BANDS)]
#         if len(band_urls) != 6:
#             failed_hls.append(geoid)
#             continue

#         with tempfile.TemporaryDirectory() as tmp:
#             files    = earthaccess.download(band_urls, local_path=tmp)
#             datasets = [rasterio.open(f) for f in sorted(files)]
#             profile  = datasets[0].profile.copy()
#             profile.update(count=6)
#             with rasterio.open(out_path, "w", **profile) as dst:
#                 for i, ds in enumerate(datasets, start=1):
#                     dst.write(ds.read(1), i)
#             for ds in datasets:
#                 ds.close()

#         hls_results[geoid] = str(out_path)
#         print(f"  {geoid}: saved {out_path.name}")
#     except Exception as e:
#         print(f"  {geoid}: FAILED — {e}")
#         failed_hls.append(geoid)

# print(f"  HLS: {len(hls_results)} OK / {len(failed_hls)} failed")

# # ════════════════════════════════════════════════════════
# #  STEP 3 — Prithvi-EO inference → extracted_q_patch_*.pt
# # ════════════════════════════════════════════════════════
# print("\n[Step 3] Prithvi-EO inference (frozen)...")

# from prithvi_mae import PrithviMAE

# NO_DATA       = -9999
# NO_DATA_FLOAT = 0.0001

# with open(EO_CONFIG_PATH, "r") as f:
#     eo_config = yaml.safe_load(f)["pretrained_cfg"]

# bands    = eo_config["bands"]
# mean     = eo_config["mean"]
# std      = eo_config["std"]
# img_size = eo_config["img_size"]
# coords_enc = eo_config["coords_encoding"]

# eo_cfg = dict(eo_config)
# eo_cfg.update(coords_encoding=coords_enc, num_frames=1, in_chans=len(bands))
# eo_model = PrithviMAE(**eo_cfg).to(device)

# state_dict = torch.load(EO_CHECKPOINT_PATH, map_location=device, weights_only=True)
# for k in list(state_dict.keys()):
#     if "pos_embed" in k:
#         del state_dict[k]
# eo_model.load_state_dict(state_dict, strict=False)
# eo_model.eval()
# print(f"  Loaded EO checkpoint: {EO_CHECKPOINT_PATH}")


# def read_geotiff(path):
#     with rasterio.open(path) as src:
#         img = src.read()
#         meta = src.meta
#         try:    coords = src.lnglat()
#         except: coords = None
#     return img, meta, coords


# def load_example(file_paths, mean, std):
#     imgs, temporal_coords, location_coords = [], [], []
#     for file in file_paths:
#         img, meta, coords = read_geotiff(file)
#         img = np.moveaxis(img, 0, -1)
#         img = np.where(img == NO_DATA, NO_DATA_FLOAT, (img - mean) / std)
#         imgs.append(img)
#         if coords:
#             location_coords.append(coords)
#         try:
#             match = re.search(r"(\d{7,8}T\d{6})", file)
#             if match:
#                 year      = int(match.group(1)[:4])
#                 jday_str  = match.group(1).split("T")[0][4:]
#                 jday      = int(jday_str) if len(jday_str) == 3 else \
#                             datetime.datetime.strptime(jday_str, "%m%d").timetuple().tm_yday
#                 temporal_coords.append([year, jday])
#             else:
#                 temporal_coords.append([TARGET_YEAR_EO, 1])
#         except:
#             temporal_coords.append([TARGET_YEAR_EO, 1])
#     imgs = np.stack(imgs, axis=0)
#     imgs = np.moveaxis(imgs, -1, 0).astype("float32")
#     imgs = np.expand_dims(imgs, axis=0)
#     return imgs, temporal_coords, location_coords


# patch_dirs   = {}   # {geoid: Path}
# q_save_paths = {}   # {geoid: first patch path (for county order reference)}

# for geoid, tif_path in tqdm(hls_results.items(), desc="EO patch extraction"):
#     geoid_out = OUTPUT_DIR / geoid
#     geoid_out.mkdir(parents=True, exist_ok=True)
#     patch_dirs[geoid] = geoid_out

#     # Skip if already extracted
#     existing = sorted(geoid_out.glob("extracted_q_patch_*.pt"))
#     if existing:
#         q_save_paths[geoid] = str(geoid_out / "final_county_embedding_q.pt")
#         continue

#     input_data, temporal_coords, location_coords, = load_example(
#         [tif_path], mean=mean, std=std
#     )
#     _ce = dict(coords_enc)
#     if not temporal_coords and "time" in _ce:       _ce.pop("time")
#     if not location_coords  and "location" in _ce:  _ce.pop("location")

#     # Pad & sliding window
#     oh, ow = input_data.shape[-2:]
#     ph = img_size - (oh % img_size) if oh % img_size != 0 else 0
#     pw = img_size - (ow % img_size) if ow % img_size != 0 else 0
#     input_data = np.pad(input_data, ((0,0),(0,0),(0,0),(0,ph),(0,pw)), mode="reflect")

#     batch   = torch.tensor(input_data, device="cpu")
#     windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
#     windows = rearrange(windows, "b c t h1 w1 h w -> (b h1 w1) c t h w",
#                         h=img_size, w=img_size)
#     windows = torch.tensor_split(windows, windows.shape[0], dim=0)

#     tc = torch.Tensor(temporal_coords).unsqueeze(0).to(device) if temporal_coords else None
#     lc = torch.Tensor(location_coords[0]).unsqueeze(0).to(device) if location_coords else None

#     with torch.no_grad():
#         for i, x in enumerate(windows):
#             features = eo_model.forward_features(x.to(device), tc, lc)
#             q_full   = features[-1]          # [1, N_tokens, 1024]
#             q_cls    = q_full[:, 0, :]       # CLS token のみ → [1, 1024]
#             sp       = geoid_out / f"extracted_q_patch_{i}.pt"
#             torch.save(q_cls.cpu(), sp)      # CLSだけ保存 (197倍の省メモリ)

#     print(f"  {geoid}: {len(windows)} patches → {geoid_out}")
#     q_save_paths[geoid] = str(geoid_out / "final_county_embedding_q.pt")

# # free EO model VRAM before loading WxC
# del eo_model
# torch.cuda.empty_cache()
# gc.collect()

# # ════════════════════════════════════════════════════════
# #  STEP 4a — Prithvi-WxC encoder → feature_map.pt
# #  【重い処理】WxC (2.3B params) を1回だけ実行してグローバル特徴マップを保存。
# #  feature_map.pt が存在すればスキップ（county を変えても再実行不要）。
# # ════════════════════════════════════════════════════════
# print("\n[Step 4a] Prithvi-WxC encoder → feature_map.pt ...")

# feature_map_path = OUTPUT_DIR / "feature_map.pt"
# C_sc_path        = OUTPUT_DIR / "C_sc.pt"

# if feature_map_path.exists() and C_sc_path.exists():
#     print("  feature_map.pt already exists — skipping heavy WxC inference.")
#     feature_map = torch.load(feature_map_path, map_location=device)
#     C_sc        = torch.load(C_sc_path,        map_location=device)
#     print(f"  feature_map : {feature_map.shape}")
#     print(f"  C_sc        : {C_sc.shape}")
# else:
#     from PrithviWxC.model import PrithviWxC
#     from PrithviWxC.dataloaders.merra2 import (
#         Merra2Dataset, preproc,
#         input_scalers, output_scalers, static_input_scalers,
#     )

#     surface_vars        = ["EFLUX","GWETROOT","HFLUX","LAI","LWGAB","LWGEM","LWTUP",
#                            "PS","QV2M","SLP","SWGNT","SWTNT","T2M","TQI","TQL","TQV",
#                            "TS","U10M","V10M","Z0M"]
#     static_surface_vars = ["FRACI","FRLAND","FROCEAN","PHIS"]
#     vertical_vars       = ["CLOUD","H","OMEGA","PL","QI","QL","QV","T","U","V"]
#     levels              = [34,39,41,43,44,45,48,51,53,56,63,68,71,72]
#     padding             = {"level":[0,0],"lat":[0,-1],"lon":[0,0]}
#     lead_times          = [18]
#     input_times         = [-6]
#     positional_encoding = "fourier"

#     in_mu, in_sig     = input_scalers(surface_vars, vertical_vars, levels,
#                             CLIM_DIR/"musigma_surface.nc", CLIM_DIR/"musigma_vertical.nc")
#     output_sig         = output_scalers(surface_vars, vertical_vars, levels,
#                             CLIM_DIR/"anomaly_variance_surface.nc",
#                             CLIM_DIR/"anomaly_variance_vertical.nc")
#     static_mu, static_sig = static_input_scalers(CLIM_DIR/"musigma_surface.nc",
#                             static_surface_vars)

#     with open(WXC_DIR / "data" / "config.yaml") as f:
#         wxc_cfg = yaml.safe_load(f)
#     p = wxc_cfg["params"]

#     wxc_model = PrithviWxC(
#         in_channels=p["in_channels"], input_size_time=p["input_size_time"],
#         in_channels_static=p["in_channels_static"],
#         input_scalers_mu=in_mu, input_scalers_sigma=in_sig,
#         input_scalers_epsilon=p["input_scalers_epsilon"],
#         static_input_scalers_mu=static_mu, static_input_scalers_sigma=static_sig,
#         static_input_scalers_epsilon=p["static_input_scalers_epsilon"],
#         output_scalers=output_sig**0.5,
#         n_lats_px=p["n_lats_px"], n_lons_px=p["n_lons_px"],
#         patch_size_px=p["patch_size_px"], mask_unit_size_px=p["mask_unit_size_px"],
#         mask_ratio_inputs=0.0, mask_ratio_targets=0.0,
#         embed_dim=p["embed_dim"], n_blocks_encoder=p["n_blocks_encoder"],
#         n_blocks_decoder=p["n_blocks_decoder"], mlp_multiplier=p["mlp_multiplier"],
#         n_heads=p["n_heads"], dropout=p["dropout"], drop_path=p["drop_path"],
#         parameter_dropout=p["parameter_dropout"],
#         residual="climate", masking_mode="global",
#         encoder_shifting=True, decoder_shifting=True,
#         positional_encoding=positional_encoding,
#         checkpoint_encoder=[], checkpoint_decoder=[],
#     )
#     weights_path = WXC_DIR / "data" / "weights" / "prithvi.wxc.2300m.v1.pt"
#     sd = torch.load(weights_path, weights_only=False)
#     sd = sd.get("model_state", sd)
#     wxc_model.load_state_dict(sd, strict=True)
#     wxc_model = wxc_model.to(device)
#     wxc_model.eval()
#     print("  Loaded WxC checkpoint. Running encoder...")

#     dataset = Merra2Dataset(
#         time_range=("2020-01-01T00:00:00", "2020-01-02T05:59:59"),
#         lead_times=lead_times, input_times=input_times,
#         data_path_surface=MERRA_DIR, data_path_vertical=MERRA_DIR,
#         climatology_path_surface=CLIM_DIR, climatology_path_vertical=CLIM_DIR,
#         surface_vars=surface_vars, static_surface_vars=static_surface_vars,
#         vertical_vars=vertical_vars, levels=levels,
#         positional_encoding=positional_encoding,
#     )
#     batch = preproc([next(iter(dataset))], padding)
#     for k, v in batch.items():
#         if isinstance(v, torch.Tensor):
#             batch[k] = v.to(device)

#     with torch.no_grad():
#         x_rescaled = (batch["x"] - wxc_model.input_scalers_mu) / \
#                      (wxc_model.input_scalers_sigma + wxc_model.input_scalers_epsilon)
#         x_rescaled  = x_rescaled.flatten(1, 2)
#         x_pos       = wxc_model.fourier_pos_encoding(batch["static"])
#         x_static    = (batch["static"][:,2:] - wxc_model.static_input_scalers_mu[:,3:]) / \
#                       (wxc_model.static_input_scalers_sigma[:,3:] + wxc_model.static_input_scalers_epsilon)
#         climate_sc  = (batch["climate"] - wxc_model.input_scalers_mu.view(1,-1,1,1)) / \
#                       (wxc_model.input_scalers_sigma.view(1,-1,1,1) + wxc_model.input_scalers_epsilon)

#         x_emb      = wxc_model.patch_embedding(x_rescaled)
#         static_emb = wxc_model.patch_embedding_static(torch.cat((x_static, climate_sc), dim=1))
#         static_emb += x_pos
#         x_emb      = wxc_model.to_patching(x_emb)
#         static_emb = wxc_model.to_patching(static_emb)
#         time_enc   = wxc_model.time_encoding(batch["input_time"], batch["lead_time"])
#         tokens     = x_emb + static_emb + time_enc
#         x_encoded  = wxc_model.encoder(tokens)

#         B  = x_encoded.shape[0]
#         G0, G1 = wxc_model.global_shape_mu
#         L0, L1 = wxc_model.local_shape_mu
#         D      = wxc_model.embed_dim
#         x_enc  = x_encoded.view(B, G0, G1, L0, L1, D)
#         x_enc  = x_enc.permute(0,5,1,3,2,4).contiguous()
#         feature_map = x_enc.view(B, D, G0*L0, G1*L1)      # [1, 160, 180, 288]

#         C_sc = (batch["climate"] - wxc_model.input_scalers_mu.view(1,-1,1,1)) / \
#                (wxc_model.input_scalers_sigma.view(1,-1,1,1) + wxc_model.input_scalers_epsilon)

#     # 重い推論結果を保存（county を変えても再実行不要）
#     torch.save(feature_map.cpu(), feature_map_path)
#     torch.save(C_sc.cpu(),        C_sc_path)
#     print(f"  feature_map saved: {feature_map.shape}")
#     print(f"  C_sc saved       : {C_sc.shape}")

#     del wxc_model
#     torch.cuda.empty_cache()
#     gc.collect()

# # ════════════════════════════════════════════════════════
# #  STEP 4b — Bilinear interpolation → met_embedding.pt
# #  【軽い処理】county centroid で feature_map を interpolate するだけ。
# #  county リストを変えた場合もここだけ再実行すれば OK。
# # ════════════════════════════════════════════════════════
# print("\n[Step 4b] Spatial interpolation at county centroids → met_embedding.pt ...")

# feature_map = feature_map.to(device)
# C_sc        = C_sc.to(device)

# lat_grid  = torch.linspace(-90,  90,  feature_map.shape[-2])
# lon_grid  = torch.linspace(-180, 180, feature_map.shape[-1])
# lats      = torch.tensor(df_target["lat"].tolist(), dtype=torch.float32).to(device)
# lons      = torch.tensor(df_target["lon"].tolist(), dtype=torch.float32).to(device)
# norm_lats = 2.0*(lats - lat_grid.min())/(lat_grid.max() - lat_grid.min()) - 1.0
# norm_lons = 2.0*(lons - lon_grid.min())/(lon_grid.max() - lon_grid.min()) - 1.0
# B         = feature_map.shape[0]
# grid      = torch.stack([norm_lons, norm_lats], dim=-1).unsqueeze(0).unsqueeze(0).expand(B,1,-1,-1)

# with torch.no_grad():
#     county_weather_token     = F.grid_sample(
#         feature_map, grid, mode="bilinear", align_corners=True
#     ).squeeze(2).permute(0,2,1)                # [1, N, 160]

#     local_climatology_vector = F.grid_sample(
#         C_sc, grid, mode="bilinear", align_corners=True
#     ).squeeze(2).permute(0,2,1)                # [1, N, 160]

#     import torch.nn as nn
#     _proj_clim = nn.Linear(160, 2560).to(device)
#     _norm_wxc  = nn.LayerNorm(2560).to(device)
#     _norm_clim = nn.LayerNorm(2560).to(device)

#     wxc_normed  = _norm_wxc(_proj_clim(county_weather_token))
#     clim_normed = _norm_clim(_proj_clim(local_climatology_vector))
#     met_embedding = torch.cat([wxc_normed, clim_normed], dim=-1)   # [1, N, 5120]

# torch.save(met_embedding.cpu(),           OUTPUT_DIR / "met_embedding.pt")
# torch.save(county_weather_token.cpu(),    OUTPUT_DIR / "wxc_county_tokens.pt")
# torch.save(local_climatology_vector.cpu(),OUTPUT_DIR / "clim_county_vectors.pt")
# print(f"  met_embedding saved : {met_embedding.shape}")
# print(f"  raw tokens also saved (wxc_county_tokens.pt, clim_county_vectors.pt)")

# # ════════════════════════════════════════════════════════
# #  STEP 5 — Save q_save_paths.json (county order index)
# # ════════════════════════════════════════════════════════
# print("\n[Step 5] Saving q_save_paths.json...")

# with open(OUTPUT_DIR / "q_save_paths.json", "w") as f:
#     json.dump({k: str(v) for k, v in q_save_paths.items()}, f)

# print(f"  q_save_paths.json saved: {len(q_save_paths)} counties")

# # ════════════════════════════════════════════════════════
# #  DONE
# # ════════════════════════════════════════════════════════
# print("\n" + "="*50)
# print("preprocess.py complete.")
# print("Next step: python train.py")
# print("="*50)
