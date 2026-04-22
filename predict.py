"""
predict.py — 学習済み重みを使って新しい county の収量を予測する

使い方:
  python predict.py --geoid 17019 --patches_dir data/mizuho_output/17019 --met data/mizuho_output/met_embedding.pt --county_idx 5

Prerequisites:
  1. best_model.pt が存在すること (train.py を実行済み)
  2. 新しい county の patch embeddings が存在すること
       (eo_extract_features.py で抽出済み)
  3. met_embedding.pt が存在すること (ノートブック Cell 4 実行済み)
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ════════════════════════════════════════════════════════
#  MODEL DEFINITIONS (train.py と同一でなければならない)
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

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        scores  = self.attention_net(patch_embeddings)
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.transpose(1, 2), patch_embeddings).squeeze(1)


class CrossModalAttention(nn.Module):
    def __init__(self, d_eo: int = 1024, d_met: int = 5120,
                 n_heads: int = 8, dropout: float = 0.1):
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
    def __init__(self, d_in: int = 1024, hidden_dim: int = 256,
                 dropout: float = 0.1):
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
#  LOAD TRAINED MODEL
# ════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    patch_pool = PatchAttentionPooling(embed_dim=1024).to(device)
    cross_attn = CrossModalAttention(d_eo=1024, d_met=5120, n_heads=8).to(device)
    mlp_head   = MLPRegressionHead(d_in=1024).to(device)

    # 重みをロード
    if "patch_pool" in ckpt:
        patch_pool.load_state_dict(ckpt["patch_pool"])
    cross_attn.load_state_dict(ckpt["cross_attn"])
    mlp_head.load_state_dict(ckpt["mlp_head"])

    # 推論モードに切り替え (dropoutをオフ, batch normをeval)
    patch_pool.eval()
    cross_attn.eval()
    mlp_head.eval()

    y_mean = ckpt["y_mean"]
    y_std  = ckpt["y_std"]

    return patch_pool, cross_attn, mlp_head, y_mean, y_std


# ════════════════════════════════════════════════════════
#  LOAD PATCH EMBEDDINGS (Prithvi-EO の出力)
# ════════════════════════════════════════════════════════

def load_patches(patches_dir: str, device: torch.device) -> torch.Tensor:
    """
    patches_dir 内の extracted_q_patch_*.pt を読み込み
    [1, N_patches, 1024] を返す
    """
    patch_files = sorted(Path(patches_dir).glob("extracted_q_patch_*.pt"))
    assert len(patch_files) > 0, f"No patch files in {patches_dir}"

    patches = []
    for pf in patch_files:
        q = torch.load(pf, map_location="cpu")   # [1, 1024] or [1, N_tok, 1024]
        if q.dim() == 3:
            q = q[:, 0, :]                        # CLS token → [1, 1024]
        patches.append(q)

    return torch.cat(patches, dim=0).unsqueeze(0).to(device)  # [1, N_patches, 1024]


# ════════════════════════════════════════════════════════
#  PREDICT
# ════════════════════════════════════════════════════════

def predict(
    checkpoint_path: str,
    patches_dir: str,
    met_path: str,
    county_idx: int,          # met_embedding.pt 内での county のインデックス
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- モデルのロード ---
    patch_pool, cross_attn, mlp_head, y_mean, y_std = load_model(
        checkpoint_path, device
    )
    print(f"Loaded model from {checkpoint_path}")
    print(f"  y_mean={y_mean:.2f} bu/acre  y_std={y_std:.2f} bu/acre")

    # --- EO patch embeddings のロード ---
    patches = load_patches(patches_dir, device)      # [1, N_patches, 1024]
    print(f"Loaded {patches.shape[1]} patches from {patches_dir}")

    # --- met embedding のロード ---
    met_full = torch.load(met_path, map_location=device)  # [1, N_all_counties, 5120]
    met      = met_full[:, county_idx:county_idx+1, :]    # [1, 1, 5120]
    print(f"Met embedding: county index={county_idx}  shape={met.shape}")

    # --- 推論 ---
    with torch.no_grad():
        county_q = patch_pool(patches)          # [1, 1024]
        county_q = county_q.unsqueeze(1)        # [1, 1, 1024]  (N=1 county)

        fused    = cross_attn(county_q, met)    # [1, 1, 1024]
        y_norm   = mlp_head(fused)              # [1, 1]

        # 正規化を解除して実際の bu/acre に戻す
        y_pred   = y_norm.item() * y_std + y_mean

    print(f"\n{'='*40}")
    print(f"Predicted yield: {y_pred:.2f} bu/acre")
    print(f"{'='*40}")
    return y_pred


# ════════════════════════════════════════════════════════
#  BATCH PREDICT (複数 county を一括予測)
# ════════════════════════════════════════════════════════

def predict_batch(
    checkpoint_path: str,
    output_dir: str,          # mizuho_output/ — {geoid}/ サブディレクトリを含む
    met_path: str,
    geoid_to_idx: dict,       # {"17019": 5, "17031": 12, ...}
):
    """
    複数 county をまとめて予測し、結果を表示する。

    geoid_to_idx: geoid → met_embedding.pt 内のインデックス のマッピング
    (ノートブックで county_order.json を保存していれば自動生成できる)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_pool, cross_attn, mlp_head, y_mean, y_std = load_model(
        checkpoint_path, device
    )
    met_full = torch.load(met_path, map_location=device)

    results = {}

    with torch.no_grad():
        for geoid, idx in geoid_to_idx.items():
            patches_dir = Path(output_dir) / geoid
            if not patches_dir.exists():
                print(f"  Skip {geoid}: no patch dir")
                continue

            patches  = load_patches(str(patches_dir), device)          # [1, N_patches, 1024]
            met      = met_full[:, idx:idx+1, :]                       # [1, 1, 5120]

            county_q = patch_pool(patches).unsqueeze(1)                 # [1, 1, 1024]
            fused    = cross_attn(county_q, met)
            y_norm   = mlp_head(fused)
            y_pred   = y_norm.item() * y_std + y_mean

            results[geoid] = y_pred
            print(f"  {geoid}: {y_pred:.2f} bu/acre")

    return results


# ════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Crop yield prediction")
    parser.add_argument("--checkpoint", type=str,
                        default="data/mizuho_output/best_model.pt",
                        help="Path to best_model.pt")
    parser.add_argument("--patches_dir", type=str, required=True,
                        help="Directory containing extracted_q_patch_*.pt")
    parser.add_argument("--met", type=str,
                        default="data/mizuho_output/met_embedding.pt",
                        help="Path to met_embedding.pt")
    parser.add_argument("--county_idx", type=int, required=True,
                        help="Index of this county in met_embedding.pt")
    args = parser.parse_args()

    predict(
        checkpoint_path=args.checkpoint,
        patches_dir=args.patches_dir,
        met_path=args.met,
        county_idx=args.county_idx,
    )
