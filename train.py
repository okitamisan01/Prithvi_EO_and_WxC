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
import random
from tqdm import tqdm

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
TRAIN_RATIO = 0.7
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 30  # これも追加

# ════════════════════════════════════════════════════════
#  MODEL DEFINITIONS (copied from Cell 5 & 6)
#  Must match exactly what the notebook used
# ════════════════════════════════════════════════════════

class PatchAttentionPooling(nn.Module):
    """Prithvi-EOブロック直後: 可変長パッチ列 → county embedding Q [B, 1024]"""
    def __init__(self, embed_dim: int = 1024):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        # patch_embeddings: [B, N_patches, 1024]
        scores  = self.attention_net(patch_embeddings)          # [B, N_patches, 1]
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.transpose(1, 2), patch_embeddings).squeeze(1)  # [B, 1024]


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
#  LOAD PATCH EMBEDDINGS (PatchAttentionPooling の入力)
# ════════════════════════════════════════════════════════

print("\nLoading patch-level Q embeddings per county...")
county_patches: dict = {}   # geoid → [N_patches, 1024]
for geoid in tqdm(RESOLVED_GEOIDS, desc="Loading patches"):
    county_dir  = OUTPUT_DIR / geoid
    patch_files = sorted(county_dir.glob("extracted_q_patch_*.pt"))
    assert len(patch_files) > 0, \
        f"No patch files in {county_dir} — eo_extract_features.py を先に実行してください"
    patches = []
    for pf in patch_files:
        q = torch.load(pf, map_location="cpu")  # [1, 1024] or [1, N_tok, 1024]
        if q.dim() == 3:
            q = q[:, 0, :]                       # CLS token → [1, 1024]
        patches.append(q)
    county_patches[geoid] = torch.cat(patches, dim=0)  # [N_patches, 1024]

met_emb = met_embedding[:, :len(RESOLVED_GEOIDS), :]
print(f"  Loaded patches for {len(county_patches)} counties")
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

print(f"  y_true (normalized) : {y_true.shape}  range {y_true.min():.3f}–{y_true.max():.3f}")

# ════════════════════════════════════════════════════════
#  SPLIT TRAIN/TEST
# ════════════════════════════════════════════════════════

random.seed(RANDOM_SEED)
shuffled = valid_geoids.copy()
random.shuffle(shuffled)
 
n_train = int(len(shuffled) * TRAIN_RATIO)
train_geoids = shuffled[:n_train]   # 70%
test_geoids  = shuffled[n_train:]   # 30%
 
train_indices = [RESOLVED_GEOIDS.index(g) for g in train_geoids]
test_indices  = [RESOLVED_GEOIDS.index(g) for g in test_geoids]
 
print(f"\n  Train/Test split (seed={RANDOM_SEED})")
print(f"  Train : {len(train_geoids)} counties ({len(train_geoids)/len(valid_geoids)*100:.0f}%)")
print(f"  Test  : {len(test_geoids)}  counties ({len(test_geoids)/len(valid_geoids)*100:.0f}%)")
print(f"  Test counties : {test_geoids}")
 

# ════════════════════════════════════════════════════════
#  NORMALIZE
# ════════════════════════════════════════════════════════

train_yield_values = [yield_map[g] for g in train_geoids]
y_mean = sum(train_yield_values) / len(train_yield_values)
y_std  = (sum((v - y_mean) ** 2 for v in train_yield_values) / len(train_yield_values)) ** 0.5
 
def normalize(geoids):
    return [(yield_map[g] - y_mean) / y_std for g in geoids]
 
y_train = torch.tensor(normalize(train_geoids), dtype=torch.float32).unsqueeze(0).to(device)
y_test  = torch.tensor(normalize(test_geoids),  dtype=torch.float32).unsqueeze(0).to(device)
 
print(f"\n  y_mean={y_mean:.2f} bu/acre  y_std={y_std:.2f} bu/acre  (train統計量)")
 
met_emb_train = met_emb[:, train_indices, :].detach()   # [1, N_train, 5120]
met_emb_test  = met_emb[:, test_indices,  :].detach()   # [1, N_test,  5120]

print(f"  met_emb_train : {met_emb_train.shape}  |  met_emb_test : {met_emb_test.shape}")

# ════════════════════════════════════════════════════════
#  SANITY CHECK — shape / dtype / device
# ════════════════════════════════════════════════════════

print("\nSanity check...")

D_met = met_emb.shape[-1]    # 5120

patch_pool = PatchAttentionPooling(embed_dim=1024).to(device)
cross_attn = CrossModalAttention(d_eo=1024, d_met=D_met, n_heads=8).to(device)
mlp_head   = MLPRegressionHead(d_in=1024).to(device)


def build_eo_q(geoid_list: list) -> torch.Tensor:
    """パッチ埋め込み → PatchAttentionPooling → [1, N_counties, 1024]"""
    qs = []
    for g in geoid_list:
        patches = county_patches[g].to(device)      # [N_patches, 1024]
        q = patch_pool(patches.unsqueeze(0))         # [1, 1024]
        qs.append(q)
    return torch.stack(qs, dim=1)                    # [1, N, 1024]


with torch.no_grad():
    _q_chk         = build_eo_q(train_geoids[:2])
    _met_chk       = met_emb_train[:, :2, :]
    fused_test     = cross_attn(_q_chk, _met_chk)
    y_hat_test_chk = mlp_head(fused_test)

# assert y_hat_test_chk.shape == y_train.shape, \
#     f"Shape mismatch: y_hat={y_hat_test_chk.shape} vs y_true={y_train.shape}"

assert y_hat_test_chk.shape == (1, 2), \
    f"Shape mismatch: y_hat={y_hat_test_chk.shape}, expected (1, 2)"

print(f"  shapes OK ✅  device={y_train.device}  dtype={y_train.dtype}")


# ════════════════════════════════════════════════════════
#  OPTIMIZER & SCHEDULER
# ════════════════════════════════════════════════════════

optimizer = torch.optim.Adam([
    {"params": patch_pool.parameters(), "lr": LR},
    {"params": cross_attn.parameters(), "lr": LR},
    {"params": mlp_head.parameters(),   "lr": LR},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
loss_fn   = nn.MSELoss()
scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  # mixed precision

# ════════════════════════════════════════════════════════
#  TRAINING LOOP
# ════════════════════════════════════════════════════════

print(f"\nStarting training — {N_EPOCHS} epochs...\n")

best_loss        = float("inf")
loss_history     = []
val_loss_history = []

pbar = tqdm(range(N_EPOCHS), desc="Training", ncols=110,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
patience_counter = 0  # ← 追加

for epoch in pbar:
    # ── Train ──────────────────────────────────────────
    patch_pool.train(); cross_attn.train(); mlp_head.train()
    optimizer.zero_grad(set_to_none=True)   # set_to_none でメモリ節約

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        eo_q_tr = build_eo_q(train_geoids)              # [1, N_train, 1024]
        fused   = cross_attn(eo_q_tr, met_emb_train)    # [1, N_train, 1024]
        y_hat   = mlp_head(fused)                        # [1, N_train]
        loss       = loss_fn(y_hat, y_train)
        l2_reg     = sum(p.pow(2).sum() for p in cross_attn.parameters()) * L2_LAMBDA
        total_loss = loss + l2_reg

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        list(patch_pool.parameters()) +
        list(cross_attn.parameters()) +
        list(mlp_head.parameters()), max_norm=1.0
    )
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    loss_history.append(loss.item())

    # ── Validation ─────────────────────────────────────
    with torch.no_grad():
        patch_pool.eval(); cross_attn.eval(); mlp_head.eval()
        eo_q_te   = build_eo_q(test_geoids)
        fused_val = cross_attn(eo_q_te, met_emb_test)
        y_hat_val = mlp_head(fused_val)
        val_loss  = loss_fn(y_hat_val, y_test).item()
        val_loss_history.append(val_loss)

    # ── tqdm postfix (毎 epoch 更新) ───────────────────
    pbar.set_postfix(
        train=f"{loss.item():.5f}",
        val=f"{val_loss:.5f}",
        best=f"{best_loss:.5f}",
        lr=f"{scheduler.get_last_lr()[0]:.1e}",
    )

    # ── Checkpoint (every epoch) ───────────────────────
    torch.save({
        "epoch":        epoch,
        "loss":         loss.item(),
        "patch_pool":   patch_pool.state_dict(),
        "cross_attn":   cross_attn.state_dict(),
        "mlp_head":     mlp_head.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "y_mean":       y_mean,
        "y_std":        y_std,
        "train_geoids": train_geoids,
        "test_geoids":  test_geoids,
    }, OUTPUT_DIR / "latest_checkpoint.pt")

    # ── Best model ─────────────────────────────────────
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({
            "patch_pool":   patch_pool.state_dict(),
            "cross_attn":   cross_attn.state_dict(),
            "mlp_head":     mlp_head.state_dict(),
            "y_mean":       y_mean,
            "y_std":        y_std,
            "train_geoids": train_geoids,
            "test_geoids":  test_geoids,
        }, OUTPUT_DIR / "best_model.pt")
        pbar.write(f"  ★ Best model updated  epoch={epoch}  val={val_loss:.6f}")
    else:
        patience_counter += 1          # ← 追加
        if patience_counter >= EARLY_STOPPING_PATIENCE:   # ← 追加
            pbar.write(f"  Early stopping at epoch={epoch}  best_val={best_loss:.6f}")  # ← 追加
            break  
print(f"\nTraining complete!")
print(f"  Best val loss : {best_loss:.6f}")
print(f"  Best model    : {OUTPUT_DIR / 'best_model.pt'}")

# ════════════════════════════════════════════════════════
#  LOSS CURVE
# ════════════════════════════════════════════════════════

plt.figure(figsize=(8, 4))
plt.plot(loss_history,     color="#1D9E75", linewidth=1.5, label="Train loss")
plt.plot(val_loss_history, color="#E24B4A", linewidth=1.5, label="Test loss")  # ★ 追加
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training / Test Loss Curve")
plt.yscale("log")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
print(f"  Loss curve : {OUTPUT_DIR / 'loss_curve.png'}")