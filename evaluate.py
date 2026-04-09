"""
evaluate.py — Evaluation script for the crop yield model
Run from command line: python evaluate.py

Prerequisites:
  OUTPUT_DIR / best_model.pt               (from train.py)
  OUTPUT_DIR / met_embedding.pt            (from Cell 4)
  OUTPUT_DIR / q_save_paths.json           (from Cell 3)
  DATA_DIR   / USDA_Soybean_County_2020.csv
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ════════════════════════════════════════════════════════
#  SETTINGS
# ════════════════════════════════════════════════════════
REPO_ROOT  = Path("C:/Users/room208/mizuho")
DATA_DIR   = REPO_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "mizuho_output"
CSV_PATH   = DATA_DIR / "USDA_Soybean_County_2020.csv"

# ════════════════════════════════════════════════════════
#  MODEL DEFINITIONS (must match train.py exactly)
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

# ════════════════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════════════════

print("\nLoading best model...")
assert (OUTPUT_DIR / "best_model.pt").exists(), "best_model.pt not found — run train.py first"

checkpoint = torch.load(OUTPUT_DIR / "best_model.pt", map_location=device)

y_mean = checkpoint["y_mean"]
y_std  = checkpoint["y_std"]
print(f"  y_mean = {y_mean:.2f} bu/acre")
print(f"  y_std  = {y_std:.2f} bu/acre")

cross_attn = CrossModalAttention(d_eo=1024, d_met=5120, n_heads=8).to(device)
mlp_head   = MLPRegressionHead(d_in=1024).to(device)

cross_attn.load_state_dict(checkpoint["cross_attn"])
mlp_head.load_state_dict(checkpoint["mlp_head"])

cross_attn.eval()
mlp_head.eval()
print("  Model loaded ✅")

# ════════════════════════════════════════════════════════
#  LOAD EMBEDDINGS
# ════════════════════════════════════════════════════════

print("\nLoading embeddings...")
met_embedding = torch.load(OUTPUT_DIR / "met_embedding.pt", map_location=device)

with open(OUTPUT_DIR / "q_save_paths.json") as f:
    q_save_paths = {k: Path(v) for k, v in json.load(f).items()}

RESOLVED_GEOIDS = list(q_save_paths.keys())

eo_q_list = []
for geoid in RESOLVED_GEOIDS:
    q = torch.load(q_save_paths[geoid], map_location=device)
    eo_q_list.append(q)

eo_q    = torch.stack(eo_q_list, dim=1).detach()                          # [1, N, 1024]
met_emb = met_embedding[:, :len(RESOLVED_GEOIDS), :].detach()             # [1, N, 5120]

# ════════════════════════════════════════════════════════
#  LOAD TEACHER DATA
# ════════════════════════════════════════════════════════

print("Loading USDA yield labels...")
df_yield = pd.read_csv(CSV_PATH)
df_yield["GEOID"] = (
    df_yield["state_ansi"].astype(str).str.zfill(2) +
    df_yield["county_ansi"].astype(str).str.zfill(3)
)
yield_map   = dict(zip(df_yield["GEOID"], df_yield["YIELD, MEASURED IN BU / ACRE"]))
county_name = dict(zip(df_yield["GEOID"], df_yield["county_name"]))
state_name  = dict(zip(df_yield["GEOID"], df_yield["state_name"]))

valid_geoids  = [g for g in RESOLVED_GEOIDS if g in yield_map]
valid_indices = [RESOLVED_GEOIDS.index(g) for g in valid_geoids]

eo_q_eval    = eo_q[:, valid_indices, :].detach()
met_emb_eval = met_emb[:, valid_indices, :].detach()

# ════════════════════════════════════════════════════════
#  INFERENCE
# ════════════════════════════════════════════════════════

print("Running inference...")
with torch.no_grad():
    fused = cross_attn(eo_q_eval, met_emb_eval)
    y_hat_norm = mlp_head(fused)                    # [1, N_valid]  normalized

# Convert predictions back to bu/acre
y_hat_np  = y_hat_norm[0].cpu().numpy() * y_std + y_mean   # predicted bu/acre
y_true_np = np.array([yield_map[g] for g in valid_geoids]) # actual bu/acre

# ════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════

errors    = y_hat_np - y_true_np
abs_err   = np.abs(errors)
mse       = float(np.mean(errors ** 2))
rmse      = float(np.sqrt(mse))
mae       = float(np.mean(abs_err))
mape      = float(np.mean(abs_err / y_true_np) * 100)
ss_res    = float(np.sum(errors ** 2))
ss_tot    = float(np.sum((y_true_np - y_true_np.mean()) ** 2))
r2        = 1 - ss_res / ss_tot
corr      = float(np.corrcoef(y_hat_np, y_true_np)[0, 1])

print("\n" + "=" * 55)
print("EVALUATION RESULTS")
print("=" * 55)
print(f"  Counties evaluated : {len(valid_geoids)}")
print(f"  Yield range (true) : {y_true_np.min():.1f} – {y_true_np.max():.1f} bu/acre")
print(f"  Yield range (pred) : {y_hat_np.min():.1f} – {y_hat_np.max():.1f} bu/acre")
print(f"  ---")
print(f"  RMSE  : {rmse:.2f} bu/acre   (avg prediction error)")
print(f"  MAE   : {mae:.2f} bu/acre   (avg absolute error)")
print(f"  MAPE  : {mape:.1f}%          (avg % error)")
print(f"  R²    : {r2:.4f}            (1.0 = perfect, 0 = baseline mean)")
print(f"  Pearson r : {corr:.4f}      (correlation with ground truth)")
print("=" * 55)

# Interpretation
print("\nInterpretation:")
if r2 > 0.8:
    print("  ✅ R² > 0.8 — strong fit")
elif r2 > 0.5:
    print("  🔶 R² 0.5–0.8 — moderate fit, more training or data may help")
elif r2 > 0:
    print("  ⚠️  R² 0–0.5 — weak fit, model is better than guessing the mean")
else:
    print("  ❌ R² < 0 — model is worse than predicting the mean yield")

if rmse < 5:
    print(f"  ✅ RMSE {rmse:.1f} bu/acre — excellent accuracy")
elif rmse < 10:
    print(f"  🔶 RMSE {rmse:.1f} bu/acre — reasonable for a first training run")
else:
    print(f"  ⚠️  RMSE {rmse:.1f} bu/acre — high error, model needs more training")

# ════════════════════════════════════════════════════════
#  PER-COUNTY TABLE
# ════════════════════════════════════════════════════════

print(f"\n{'GEOID':<10} {'State':<6} {'County':<22} {'True':>8} {'Pred':>8} {'Error':>8} {'Abs%':>6}")
print("-" * 72)
sorted_idx = np.argsort(np.abs(errors))[::-1]   # worst first
for i in sorted_idx:
    g    = valid_geoids[i]
    cname = county_name.get(g, "?")[:20]
    sname = state_name.get(g, "?")[:5]
    flag  = " ← worst" if i == sorted_idx[0] else (" ← best" if i == sorted_idx[-1] else "")
    print(f"{g:<10} {sname:<6} {cname:<22} {y_true_np[i]:>8.1f} {y_hat_np[i]:>8.1f} "
          f"{errors[i]:>+8.1f} {abs_err[i]/y_true_np[i]*100:>5.1f}%{flag}")

# ════════════════════════════════════════════════════════
#  PLOTS
# ════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

# -- Plot 1: Scatter predicted vs actual
ax1 = fig.add_subplot(gs[0])
ax1.scatter(y_true_np, y_hat_np, color="#1D9E75", edgecolors="white", linewidths=0.5, s=60, zorder=3)
lims = [min(y_true_np.min(), y_hat_np.min()) - 3,
        max(y_true_np.max(), y_hat_np.max()) + 3]
ax1.plot(lims, lims, "k--", linewidth=1, alpha=0.4, label="Perfect prediction")
ax1.set_xlabel("Actual yield (bu/acre)")
ax1.set_ylabel("Predicted yield (bu/acre)")
ax1.set_title(f"Predicted vs Actual\nR²={r2:.3f}  RMSE={rmse:.1f} bu/acre")
ax1.legend(fontsize=8)
ax1.set_xlim(lims); ax1.set_ylim(lims)
ax1.spines[["top", "right"]].set_visible(False)

# -- Plot 2: Residuals
ax2 = fig.add_subplot(gs[1])
ax2.bar(range(len(errors)), errors[np.argsort(errors)],
        color=["#E24B4A" if e > 0 else "#378ADD" for e in errors[np.argsort(errors)]],
        edgecolor="white", linewidth=0.4)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xlabel("County (sorted by error)")
ax2.set_ylabel("Prediction error (bu/acre)")
ax2.set_title(f"Residuals\nMAE={mae:.1f}  MAPE={mape:.1f}%")
ax2.spines[["top", "right"]].set_visible(False)

# -- Plot 3: Error distribution
ax3 = fig.add_subplot(gs[2])
ax3.hist(errors, bins=12, color="#1D9E75", edgecolor="white", linewidth=0.5)
ax3.axvline(0, color="black", linewidth=1)
ax3.axvline(errors.mean(), color="#E24B4A", linewidth=1.5, linestyle="--",
            label=f"Mean error: {errors.mean():+.1f}")
ax3.set_xlabel("Prediction error (bu/acre)")
ax3.set_ylabel("Count")
ax3.set_title("Error Distribution")
ax3.legend(fontsize=8)
ax3.spines[["top", "right"]].set_visible(False)

plt.suptitle("Crop Yield Model — Evaluation on Training Counties (2020 Soybeans)", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "evaluation.png", dpi=150, bbox_inches="tight")
print(f"\nPlots saved to: {OUTPUT_DIR / 'evaluation.png'}")