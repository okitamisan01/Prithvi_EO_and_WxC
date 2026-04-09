"""
train.py — Standalone crop yield training script
Run from command line: python train.py

Prerequisites — these files must exist before running:
  OUTPUT_DIR / {geoid} / final_county_embedding_q.pt   (from Cell 3)
  OUTPUT_DIR / met_embedding.pt                         (from Cell 4)
  OUTPUT_DIR / q_save_paths.json                        (from Cell 4)
  DATA_DIR   / USDA_Soybean_County_2020.csv             (teacher data)

To save the prerequisites from your notebook, run this once in Cell 4/5:
    torch.save(met_embedding.cpu(), OUTPUT_DIR / "met_embedding.pt")
    import json
    with open(OUTPUT_DIR / "q_save_paths.json", "w") as f:
        json.dump({k: str(v) for k, v in q_save_paths.items()}, f)
"""

import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ════════════════════════════════════════════════════════
#  SETTINGS — edit these to match your environment
# ════════════════════════════════════════════════════════
REPO_ROOT  = Path("C:/Users/room208/mizuho")
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "mizuho_output"
CSV_PATH   = DATA_DIR / "USDA_Soybean_County_2020.csv"

N_EPOCHS   = 500       # was 100 — model hadn't converged yet
LR         = 1e-4
L2_LAMBDA  = 1e-5

# ════════════════════════════════════════════════════════
#  MODEL DEFINITIONS (copied from Cell 5 & 6)
#  Must match exactly what the notebook used
# ════════════════════════════════════════════════════════

class CrossModalAttention(nn.Module):
    def __init__(self, d_eo: int, d_met: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.kv_proj = nn.Linear(d_met, d_eo)
        self.attn    = nn.MultiheadAttention(
            embed_dim=d_eo, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_eo)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        kv_proj     = self.kv_proj(kv)
        attn_out, _ = self.attn(q, kv_proj, kv_proj)
        return self.norm(q + attn_out)


class MLPRegressionHead(nn.Module):
    def __init__(self, d_in: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).squeeze(-1)


# ════════════════════════════════════════════════════════
#  SETUP
# ════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if torch.cuda.is_available():
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ════════════════════════════════════════════════════════
#  LOAD CHECKPOINTS
# ════════════════════════════════════════════════════════

print("\nLoading checkpoints...")

assert (OUTPUT_DIR / "met_embedding.pt").exists(),   "met_embedding.pt not found — run Cell 4 first"
assert (OUTPUT_DIR / "q_save_paths.json").exists(),  "q_save_paths.json not found — run Cell 3 first"
assert CSV_PATH.exists(),                            f"CSV not found: {CSV_PATH}"

met_embedding = torch.load(OUTPUT_DIR / "met_embedding.pt", map_location=device)
print(f"  met_embedding : {met_embedding.shape}")

with open(OUTPUT_DIR / "q_save_paths.json") as f:
    q_save_paths = {k: Path(v) for k, v in json.load(f).items()}

RESOLVED_GEOIDS = list(q_save_paths.keys())
print(f"  counties      : {len(RESOLVED_GEOIDS)}")

# ════════════════════════════════════════════════════════
#  BUILD eo_q  [1, N, 1024]
# ════════════════════════════════════════════════════════

print("\nBuilding eo_q...")
eo_q_list = []
for geoid in RESOLVED_GEOIDS:
    path = q_save_paths[geoid]
    assert path.exists(), f"Missing patch file: {path}"
    q = torch.load(path, map_location=device)   # [1, 1024]
    eo_q_list.append(q)

eo_q    = torch.stack(eo_q_list, dim=1)         # [1, N, 1024]
met_emb = met_embedding[:, :len(RESOLVED_GEOIDS), :]
print(f"  eo_q    : {eo_q.shape}")
print(f"  met_emb : {met_emb.shape}")

# ════════════════════════════════════════════════════════
#  LOAD TEACHER DATA
# ════════════════════════════════════════════════════════

print("\nLoading USDA yield labels...")
df_yield = pd.read_csv(CSV_PATH)
df_yield["GEOID"] = (
    df_yield["state_ansi"].astype(str).str.zfill(2) +
    df_yield["county_ansi"].astype(str).str.zfill(3)
)
yield_map = dict(zip(df_yield["GEOID"], df_yield["YIELD, MEASURED IN BU / ACRE"]))

valid_geoids  = [g for g in RESOLVED_GEOIDS if g in yield_map]
valid_indices = [RESOLVED_GEOIDS.index(g) for g in valid_geoids]

print(f"  Counties with yield data    : {len(valid_geoids)} / {len(RESOLVED_GEOIDS)}")
print(f"  Counties without yield data : {len(RESOLVED_GEOIDS) - len(valid_geoids)} (urban/non-soybean — skipped)")

# Normalize yields to mean=0 std=1 — MSE of ~1 means avg error ~1 std (~10 bu/acre)
yield_values  = [yield_map[g] for g in valid_geoids]
y_mean        = sum(yield_values) / len(yield_values)
y_std         = (sum((v - y_mean) ** 2 for v in yield_values) / len(yield_values)) ** 0.5
yield_norm    = [(v - y_mean) / y_std for v in yield_values]

y_true        = torch.tensor(yield_norm, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N_valid]
print(f"  y_mean={y_mean:.2f} bu/acre  y_std={y_std:.2f} bu/acre")
print(f"  (to recover bu/acre: y_pred_buacre = y_hat * y_std + y_mean)")

eo_q_train    = eo_q[:, valid_indices, :].detach()        # [1, N_valid, 1024]
met_emb_train = met_emb[:, valid_indices, :].detach()     # [1, N_valid, 5120]

print(f"  y_true (normalized) : {y_true.shape}  range {y_true.min():.3f}–{y_true.max():.3f}")
print(f"  eo_q_train          : {eo_q_train.shape}")
print(f"  met_emb_train       : {met_emb_train.shape}")

# ════════════════════════════════════════════════════════
#  SANITY CHECK — shape / dtype / device
# ════════════════════════════════════════════════════════

print("\nSanity check...")

D_eo  = eo_q_train.shape[-1]       # 1024
D_met = met_emb_train.shape[-1]    # 5120

cross_attn = CrossModalAttention(d_eo=D_eo, d_met=D_met, n_heads=8).to(device)
mlp_head   = MLPRegressionHead(d_in=D_eo).to(device)
proj_clim  = nn.Linear(160, 2560).to(device)
norm_wxc   = nn.LayerNorm(2560).to(device)
norm_clim  = nn.LayerNorm(2560).to(device)

with torch.no_grad():
    fused_test = cross_attn(eo_q_train, met_emb_train)
    y_hat_test = mlp_head(fused_test)

assert y_hat_test.shape == y_true.shape, \
    f"Shape mismatch: y_hat={y_hat_test.shape} vs y_true={y_true.shape}"
assert y_hat_test.dtype == y_true.dtype, \
    f"Dtype mismatch: y_hat={y_hat_test.dtype} vs y_true={y_true.dtype}"
assert y_hat_test.device == y_true.device, \
    f"Device mismatch: y_hat={y_hat_test.device} vs y_true={y_true.device}"

print(f"  y_hat shape  : {y_hat_test.shape}  ✅")
print(f"  y_true shape : {y_true.shape}  ✅")
print(f"  dtype        : {y_true.dtype}  ✅")
print(f"  device       : {y_true.device}  ✅")

# ════════════════════════════════════════════════════════
#  OPTIMIZER & SCHEDULER
# ════════════════════════════════════════════════════════

optimizer = torch.optim.Adam([
    {"params": cross_attn.parameters(), "lr": LR},
    {"params": mlp_head.parameters(),   "lr": LR},
    {"params": proj_clim.parameters(),  "lr": LR},
    {"params": norm_wxc.parameters(),   "lr": LR},
    {"params": norm_clim.parameters(),  "lr": LR},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
loss_fn   = nn.MSELoss()

# ════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════

print(f"\nStarting training — {N_EPOCHS} epochs...\n")

best_loss    = float("inf")
loss_history = []

for epoch in range(N_EPOCHS):
    cross_attn.train()
    mlp_head.train()
    proj_clim.train()
    norm_wxc.train()
    norm_clim.train()

    optimizer.zero_grad()

    fused = cross_attn(eo_q_train, met_emb_train)  # [1, N_valid, 1024]
    y_hat = mlp_head(fused)                         # [1, N_valid]

    loss       = loss_fn(y_hat, y_true)
    l2_reg     = sum(p.pow(2).sum() for p in cross_attn.parameters()) * L2_LAMBDA
    total_loss = loss + l2_reg

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(cross_attn.parameters()) + list(mlp_head.parameters()), max_norm=1.0
    )
    optimizer.step()
    scheduler.step()

    loss_history.append(loss.item())

    # Save checkpoint every epoch so a crash never loses more than 1 epoch
    torch.save({
        "epoch":      epoch,
        "loss":       loss.item(),
        "cross_attn": cross_attn.state_dict(),
        "mlp_head":   mlp_head.state_dict(),
        "proj_clim":  proj_clim.state_dict(),
        "norm_wxc":   norm_wxc.state_dict(),
        "norm_clim":  norm_clim.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
        "y_mean":     y_mean,
        "y_std":      y_std,
    }, OUTPUT_DIR / "latest_checkpoint.pt")

    # Save best model separately
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save({
            "cross_attn": cross_attn.state_dict(),
            "mlp_head":   mlp_head.state_dict(),
            "proj_clim":  proj_clim.state_dict(),
            "norm_wxc":   norm_wxc.state_dict(),
            "norm_clim":  norm_clim.state_dict(),
            "y_mean":     y_mean,
            "y_std":      y_std,
        }, OUTPUT_DIR / "best_model.pt")

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{N_EPOCHS} | "
              f"Loss: {loss.item():.6f} | "
              f"Best: {best_loss:.6f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

print(f"\nTraining complete!")
print(f"  Best loss  : {best_loss:.6f}")
print(f"  Best model : {OUTPUT_DIR / 'best_model.pt'}")

# ════════════════════════════════════════════════════════
#  LOSS CURVE
# ════════════════════════════════════════════════════════

plt.figure(figsize=(8, 4))
plt.plot(loss_history, color="#1D9E75", linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.yscale("log")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
print(f"  Loss curve : {OUTPUT_DIR / 'loss_curve.png'}")