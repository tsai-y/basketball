# TANet V1：作者開源模組的 50 幀二分類準備

目前階段只完成 CPU 工程準備；**沒有正式訓練、GPU 預檢或 test 評估**。
這是 TANet 的專題資料適配版，不是 ACA-Net 原版，也不是作者完整訓練配方的逐項復現。

## 開源來源與保留範圍

- 作者倉庫：<https://github.com/liu-zhy/temporal-adaptive-module>
- 固定 commit：`5fcb46fe671ead2f7958d8ffc3321a0e274d6491`
- 論文：TAM: Temporal Adaptive Module for Video Recognition，ICCV 2021，<https://arxiv.org/abs/2005.06803>。
- `upstream/temporal_module.py` 原樣保留作者的 TAM、TemporalBottleneck、make_temporal_modeling，沒有改寫核心演算法。
- `upstream/reference_models.py.txt` 原樣保存作者 TSN 包裝定義，僅供來源核對，不執行。它會隱式下載權重且依賴舊版周邊模組，所以本版本改用獨立薄介面。
- `upstream/LICENSE` 是完整 Apache-2.0；檔案 SHA256、來源網址與修改狀態見 `upstream/PROVENANCE.json`。固定版本完整 Git tree 未發現額外 NOTICE 檔。
- 本地程式只處理明確的五維輸入、現代 torchvision 建模、已存在的本機 ImageNet 權重、二分類頭、時間平均與資料防護。原作者檔不可默默修改。

MMAction2/TANet 仍可作為日後框架路線，但目前 yolo11 環境沒有 mmcv/mmengine/mmaction。
這次沒有安裝、降版或改動任何套件；採用已存在的 PyTorch/torchvision 執行作者純 PyTorch 模組。

## V1 模型與輸入

`RGB [B,3,50,192,256] → 逐幀 ResNet50（16 個 Bottleneck 使用官方 TAM）→ 空間池化 → Dropout(0.5) → Linear(2048,2) → 50 幀 logits 平均 → [B,2]`

- 寬×高 256×192；參數 24,859,490。沒有 ACA 門控或 Triplet Attention。
- TAM：官方局部分支 k3→k1、C/4；全域分支 T→2T→3，BN 正規化 2T，逐影片/通道動態卷積。
- 對外只接受 `[B,3,T,H,W]`，T 必須等於建模時設定值；不猜測維度、不插值、不靜默拆影片。
- 已存在的 ResNet50 ImageNet1K V1 權重以完整 SHA256 驗證，先 strict 載入原 ResNet50，再加入全新 TAM 與二分類頭。沒有載入其他已訓練 ACA/主線 checkpoint。
- **50 幀不能直接完整載入 8/16 幀 TANet 的時間 MLP 權重**；目前不下載或使用 Kinetics 影片預訓練。
- 模型回傳未 softmax 的兩個 logits；未來 CrossEntropyLoss 的標籤必須為 int64，Miss=0、Make=1。驗證需要分開命名 accuracy、macro-F1、CE，不能把 positive-class F1 當 macro-F1。
- Dropout、50 幀、矩形全畫面與抽幀規則是明確記錄的專題設定，不是聲稱完全複製作者配方。
- 目前只有 eval 工程檢查。正式 optimizer、epoch/patience、batch、BN 更新/凍結、augmentation 等尚未定案；不藉此宣稱已驗證 GPU 訓練可行。

## 資料與時間選段邊界

- 沿用 `scripts/audit_i3d_temporal_feasibility.py` 的既有 allowlist、唯讀 FD 和 immutable manifest/split SHA256。
- train 7193、validation 1547，來源群組不交集。不重新隨機切分、不改標籤、不排除新的樣本。
- 目前從全片取 `floor(i*(N-1)/49)`，i=0…49，含首尾並保持順序，無循環補幀、黑幀或無聲重用最後一幀。
- OpenCV 持唯讀 FD 逐幀解碼；原生 320×240 BGR → RGB → 全畫面 INTER_AREA resize 256×192 → /255 → ImageNet mean/std（僅一次）。
- 原影片與既有 features/checkpoints 全部保留。只取 50 幀不代表解碼器只需讀取 50 幀；目前仍順序檢查完整影片解碼。
- test/未知 split 在 metadata/path 讀取前拒絕；跨 split、excluded、偽造 ID 在影片路徑存取前拒絕。清單可讀到 test 的**中繼資料**，但不得開啟其影片。
- `FrameInterval` 與 `uniform_indices(..., interval=...)` 預留 `[start,end)` 介面，未連接 `temporal_selection_v1`，沒有宣稱該選段器已辨認出投籃結果。
- 日後先凍結通過人工檢查的選段/回退規則，再讓不同模型使用可追溯的同一資料版本；此版本的 Dataset 和 CLI 預設仍只用全片。

## 安全入口（在 /home/tsai/projects/basketball 執行）

只做模型與清單檢查，不解碼影片、不寫報告：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.tanet_v1.prepare
```

驗證本機權重、做 CPU 前向並產生獨立報告；不解碼影片：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.tanet_v1.prepare --prepare
```

明確允許兩支不同來源群組的 train 影片做完整 50 幀資料/CPU 前向檢查：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.tanet_v1.prepare --prepare --smoke-videos
```

CPU 測試：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.tanet_v1.tests.test_tanet -v
```

報告存於本目錄 `artifacts/cpu_prepare_*/report.json`，每次新建、不覆寫前次。
小樣本前向僅證明介面可用，不計準確率、不保存訓練模型；單元測試的 synthetic backward 不建立 optimizer。
CLI 刻意**沒有** start/train/test/resume 模式，不會排程或自動接續訓練。

## 下一步（仍需另行確認）

1. 完成時間選段小型原型與事件完整性檢查；不可靠案例回退規則定案。
2. 確認正式模型比較的前處理、初始化、訓練預算和 BN 政策。
3. 另行授權 GPU readiness 與正式訓練；本步停在正式開始之前。
4. TANet 基準穩定後，才建立官方 TAM＋Triplet Attention 的獨立 V2 對照。
