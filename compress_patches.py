"""
compress_patches.py — 既存の extracted_q_patch_*.pt を CLS token のみに圧縮
  [1, 197, 1024] → [1, 1024]  (197倍の省メモリ)

一回だけ実行すれば OK。train.py 実行前に必ず実行すること。
"""

import torch
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path("C:/Users/room208/mizuho/data/mizuho_output")

patch_files = sorted(OUTPUT_DIR.rglob("extracted_q_patch_*.pt"))
print(f"Found {len(patch_files)} patch files")

already_small = 0
converted     = 0
errors        = 0

for pf in tqdm(patch_files, desc="Compressing"):
    try:
        q = torch.load(pf, map_location="cpu", weights_only=True)

        if q.dim() == 1:                    # [1024] — すでに圧縮済み
            already_small += 1
            continue
        elif q.dim() == 2:                  # [1, 1024] — 圧縮済み
            already_small += 1
            continue
        elif q.dim() == 3:                  # [1, N_tokens, 1024] — 要圧縮
            q_cls = q[:, 0, :]             # CLS token → [1, 1024]
            torch.save(q_cls, pf)          # 上書き保存
            converted += 1

    except Exception as e:
        print(f"\n  ERROR: {pf} — {e}")
        errors += 1

print(f"\nDone.")
print(f"  Converted     : {converted}")
print(f"  Already small : {already_small}")
print(f"  Errors        : {errors}")
print(f"\nNow run: python train.py")
