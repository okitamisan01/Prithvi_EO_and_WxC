import json
import re

# Define translation pairs
translations = {
    "GPU空き": "GPU available",
    "trainable adapter layers（Cell 7でも使う）": "trainable adapter layers (also used in Cell 7)",
    "Cell 7でも使う": "also used in Cell 7",
    "対象郡数 N = ": "Number of target counties N = ",
    "Cell 3のfinal_qの統計を確認": "Check statistics of final_q from Cell 3",
    "全球気象特徴マップ": "global weather feature map",
    "2郡の気象埋め込み": "weather embedding for 2 counties",
    "郡の位置をマップ上にプロット": "plot county locations on map",
    "郡のcentroidをプロット": "plot county centroids",
    "各埋め込みの統計": "Statistics of each embedding",
    "HLS衛星画像 RGB（2郡）": "HLS satellite imagery RGB (2 counties)",
    "MERRA-2 T2M（地表気温）全球マップ": "MERRA-2 T2M (surface temperature) global map",
    "2郡の位置をプロット": "plot location of 2 counties",
    "Climatology T2M（20年平均）全球マップ": "Climatology T2M (20-year average) global map",
    "MERRA-2 - Climatology 差分（anomaly）": "MERRA-2 - Climatology difference (anomaly)",
    "T2M anomaly (MERRA-2 - Climatology) — 2郡の位置": "T2M anomaly (MERRA-2 - Climatology) — Location of 2 counties",
}

# Load notebook
with open('main.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Translate all cells
for cell in nb['cells']:
    if cell['cell_type'] == 'code' or cell['cell_type'] == 'markdown':
        source = cell.get('source', [])
        if isinstance(source, list):
            new_source = []
            for line in source:
                new_line = line
                for ja, en in translations.items():
                    new_line = new_line.replace(ja, en)
                new_source.append(new_line)
            cell['source'] = new_source

# Save translated notebook
with open('main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Translation complete!")
