# `swin3d_t_binary50_v1` — 含動態位置索引修正的 Swin3D-T

## 模型定位與行為揭露

- **模型名稱**：含動態位置索引修正的 Swin3D-T（`swin3d_t_binary50_dynamic_pos_v1`）。
- **定位**：影片預訓練骨幹新系統比較（非 ACA 模組消融）。
- **唯一輸入／輸出契約**：
  - 輸入：`[B, 3, 50, 120, 160]` `float32` RGB，單次 ImageNet 正規化（`mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`，與既有 50 幀管線一致，禁止重複正規化）。
  - 分類頭：`Linear(768, 2)`，類別順序 `["Miss", "Make"]`，搭配未加權 `CrossEntropyLoss`。
  - 正規化與正則化：維持原生 29 層 `LayerNorm`（0 層 `BatchNorm`）與 12 層 `StochasticDepth`（`p=0 → 0.1`）。
- **與原版 `torchvision 0.26.0` `swin3d_t` 的行為差異**：
  - 縮小視窗不一定都會造成原版索引錯位：若**僅最外層時間維度**縮小（例如 `(4, 7, 7)`），原版直接截取 `relative_position_index[:N, :N]` 仍與真實座標一致；但當**空間維度**縮小至小於 `(7, 7)` 時（例如 `ceil(H/32) < 7` 或 `ceil(W/32) < 7`，即 `H <= 192` 或 `W <= 192`；在 `50×120×160` 下 Stage 3 為 `[B, 25, 4, 5, 768]`，其中 block 0 `shift=(0,0,0)`、block 1 `shift=(4,0,0)`），直接截取會破壞內層步長 `(49, 7, 1)`，在 `(8, 4, 5)` 視窗下有 `24,980 / 25,600` 個索引與實際幾何不符。
  - 本隔離套件在不修改已安裝 `torchvision` 的前提下，於 `DynamicPosShiftedWindowAttention3d` 中保留原 `relative_position_bias_table`（`[2535, num_heads]`）、原 `relative_position_index` buffer（`[392, 392]`）與所有參數／buffer 名稱，於動態視窗縮小時統一改以實際 token 座標 `(t, h, w)` 從原 `[392, 392]` 索引表選取對應行列（`index_select(0, coords).index_select(1, coords)`）。
  - 在完整 `(8, 7, 7)` 視窗下（例如 `T >= 16, H >= 224, W >= 224`），本模型輸出與所有參數梯度皆與原版逐值一致。

## 分階段核准邊界

1. **階段一（已通過 `PASS_CPU_ONLY`）**：隔離套件 + `weights=None` 有上限 CPU 驗收。
2. **階段二（已通過 `PASS_PRETRAINED_LOAD_CPU_ONLY`）**：單次官方 `Swin3D_T_Weights.KINETICS400_V1` 權重下載至 `artifacts/weights/swin3d_t-7615ae03.pth`（核對 `7615ae03` 前綴並鎖定完整 SHA-256 `7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130`）→ 400 類 `strict=True` 載入 → 以獨立 `head_seed` 建立 `Linear(768, 2)`（分類頭結果由獨立 seed 決定，初始化前後全域 CPU RNG 狀態不變；完整模型建構仍會消耗 RNG，故續訓須於建構與載入模型／optimizer **之後**恢復 checkpoint RNG）→ 確認全部 183 個非分類頭張量與下載檔逐值一致 → 固定 `04`/`22` 樣本 CPU 前向驗收。
3. **階段三（本階段：有上限 GPU 工程驗收，不含正式訓練）**：以鎖定 SHA-256 權重、FP32 `[1, 3, 50, 120, 160]`、僅 `04`/`22` 樣本驗證 `microbatch=1`（`accum=16` 與尾批 `8`）、Adam 顯存峰值、零增強 eval、以及含 `best_model` 非空與 CPU/CUDA RNG 之續訓逐值一致（外部逾時 15 分鐘、總工程更新上限 8 次、沿用既有 GPU 排程政策）。

## 執行方式（搭配外部 OS `timeout` 控制）

```bash
timeout --signal=TERM --kill-after=10 120s env CUDA_VISIBLE_DEVICES="" \
  /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest \
  development.swin3d_t_binary50_v1.test_swin3d_t_binary50

timeout --signal=TERM --kill-after=10 180s env CUDA_VISIBLE_DEVICES="" \
  /home/tsai/miniconda3/envs/yolo11/bin/python -B -m \
  development.swin3d_t_binary50_v1.pretrained_load_acceptance

timeout --signal=TERM --kill-after=10 900s env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/tsai/miniconda3/envs/yolo11/bin/python -B -m \
  development.swin3d_t_binary50_v1.gpu_readiness
```
