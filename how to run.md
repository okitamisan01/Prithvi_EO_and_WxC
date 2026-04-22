正確な実行順序（今回）
# ① パッチファイルを圧縮（46GB → 237MB）← 必須・先にmain.ipynb
python compress_patches.py

# ② met_embedding.pt を削除
del data\mizuho_output\met_embedding.pt

# ③ preprocess.py を再実行（Step 4b だけ走る・数秒）
python preprocess.py

# ④ 学習
python train.py
次回以降は
# 通常はこの3行だけ
python preprocess.py   # データが全部揃っていればほぼスキップ
python train.py
python predict.py ...