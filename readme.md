# Basketball-51 籃球影片研究專案

本專案使用 Basketball-51 研究 **full-video Make/Miss classification**。長期目標是只用**球離手前**畫面預測結果，但目前 A1、A2、B 與新建立的 C 流程都取用完整影片，可能看到球飛行、碰框與進球畫面；它們都**不是 pre-release prediction**。Experiment B 是 Basketball-51 上的 Reference C3D adaptation，並非原論文 benchmark 的完整重現。以下狀態依 **2026-10-01** 讀到的本地 run 檔案及 C 驗證結果整理。

## Research Status

| 工作 | 目前狀態 |
| --- | --- |
| Dataset inspection、清理、固定 train/validation/test split | ✅ 已完成；原始影片未修改 |
| Experiment A1 訓練與一次 test 評估 | ✅ 已完成 |
| Experiment A2 extended training | ✅ epoch 28 early stop；最佳 checkpoint 為 epoch 21 |
| Experiment A2 test | **Not evaluated yet** |
| Experiment B 架構、continuous clip pipeline、smoke／diagnostic gates | ✅ 已完成 |
| Experiment B 正式訓練 | ✅ epoch 32 early stop；最佳 checkpoint 為 epoch 25（上限 50、effective patience 7） |
| Experiment B test | **Not evaluated yet** |
| Experiment C continuous clip 實作與 smoke／dry-run | ✅ 已通過 CPU 驗證；正式訓練尚未開始 |
| Experiment C test | **Not evaluated yet** |
| Pre-release pipeline、YOLO、SPL skeleton stage | 尚未實作 |

原始 **10,311** 支影片中，排除 **6** 支結果標籤衝突影片及 **26** 支一致標籤的重複複本，留下 **10,279** 支。可用資料分為 train **7,193**、validation **1,547**、test **1,539**。split 依固定來源代碼分組，**尚未核實為真正的 game-level split**。

## Experiments Overview

**Experiment A1** 是 Modified C3D-style initial baseline；**Experiment A2** 使用**完全相同的模型架構與前處理**，由 A1 最佳模型權重 warm-start，作為 extended-training follow-up，並非新 architecture，也不是 A1 optimizer 的 true resume。**Experiment B** 改用 Reference C3D adaptation 與連續重疊 clip／fc6 影片描述子流程。**Experiment C** 保留 A1/A2 的卷積特徵骨幹，改接 B 的連續 clip 資料流與影片級特徵聚合；C 尚未正式訓練。四者使用同一固定清理後的 Basketball-51 split，任務仍是完整影片 Make/Miss 分類。

| Item | Experiment A1 | Experiment A2 | Experiment B | Experiment C |
| --- | --- | --- | --- | --- |
| Purpose | 初始 baseline | 延長 A1 訓練 | Reference C3D adaptation | 比較 continuous clips 下的 Modified backbone |
| Architecture | Modified C3D-style | 同 A1 | Reference C3D adaptation | A1/A2 卷積骨幹與 GAP，另接影片級 head |
| Video sampling | 全片 16 bins，各取 1 frame | 同 A1 | 連續 16-frame clips，stride 8、overlap 8 | 同 B |
| Input shape | 每影片 `[3,16,96,96]` | 同 A1 | 每 clip `[3,16,112,112]` | 同 B |
| Resolution | 96×96 letterbox | 同 A1 | resize 128×171、crop 112×112 | 同 B |
| Conv layers / channels | 8；`3→32→64→128→128→256→256→256→256` | 同 A1 | 8；`3→64→128→256→256→512→512→512→512` | 同 A1 |
| Conv kernel | `3×3×3`、padding 1 | 同 A1 | `3×3×3`、stride 1、padding 1 | 同 A1 |
| BatchNorm / ReLU | 每個 Conv 後 BN、ReLU | 同 A1 | 無 BN；Conv 後 ReLU | 同 A1 骨幹 |
| Pooling | 5；Pool1 `1×2×2`，Pool2–5 `2×2×2` | 同 A1 | 同 kernel/stride，`ceil_mode=True` | 同 A1，未啟用 `ceil_mode` |
| Global Average Pooling | `AdaptiveAvgPool3d(1)` | 同 A1 | 無 | 同 A1 |
| fc6 / fc7 / Dropout | 無；小 head `256→128→2`、dropout 0.3 | 同 A1 | fc6/fc7 各 4096、dropout 0.5；影片路徑在 fc6 dropout 前取特徵 | 無；A1/A2 小 head 不在 C 路徑 |
| Video aggregation | 無多 clip 聚合 | 同 A1 | 平均 fc6、L2 normalization | 平均 256-D GAP 特徵、L2 normalization |
| Classifier | `Linear(128,2)` | 同 A1 | `Linear(4096,2)` | `Linear(256,2)` |
| Parameters | 6,951,874 | 6,951,874 | reference model 78,003,970；加影片 head 共 78,012,164，實際訓練路徑 61,222,658 | 6,919,234；全數可訓練 |
| Training | 從零訓練 5 epochs | A1 權重 warm-start；到 epoch 28 | 從零訓練；epoch 32 early stop | **Not trained yet**；僅 smoke/dry-run |
| Best criterion | 最低 validation loss | 最低 validation loss | 最低 validation loss | 擬用最低 validation loss |
| Best epoch | 5 | 21 | 25 | **Not trained yet** |
| Best validation loss | 0.5254383027630617 | 0.4401281460416494 | 0.5345329800506291 | **Not trained yet** |
| Validation accuracy at best | 0.7601809954751131 | 0.7970265029088559 | 0.7504848093083387 | **Not trained yet** |
| Test status | 一次評估：accuracy **0.747888**（約 74.79%） | **Not evaluated yet** | **Not evaluated yet** | **Not evaluated yet** |

### Experiment A1 — Modified C3D-style Initial Baseline

**Goal.** 建立快速、可運行的完整影片 Make/Miss baseline。`UniformBinSampler` 將全片分成 16 個 temporal bins，每個 bin 取一 frame；16 張影格分散在整支影片，**不是連續 16-frame clip**。輸入 `[3,16,96,96]` 的 3 是 RGB channels，16 是 sampled frames，96×96 是每張 frame 的空間尺寸。影像按比例 resize 並補黑邊，RGB 值轉至 `[-1,1]`；train 有區間內 jitter 與水平翻轉，validation/test 固定取 bin 中心。

**Architecture.** [實際模型](baseline/model.py)有 8 個 `3×3×3`、padding 1 的 Conv3D，channels 為 `3→32→64→128→128→256→256→256→256`。每個 block 是 `Conv3D → BatchNorm3d → ReLU`；五個 MaxPool3d 中，Pool1 為 `1×2×2`，Pool2–5 為 `2×2×2`。之後依序是 `AdaptiveAvgPool3d(1) → Flatten → Linear(256,128) → ReLU → Dropout(0.3) → Linear(128,2)`。

```text
Full video → 16 temporal bins → 16 sparse RGB frames → [3,16,96,96]
→ Conv/BN/ReLU + Pool（重複至 8 Conv、5 Pool）→ GAP → FC → Dropout
→ Make/Miss logits
```

**Training / results.** 使用 AdamW、LR `0.001`、batch size 16，從零訓練 5 epochs，以最低 validation loss 選 checkpoint。最佳 epoch **5**，validation loss **0.5254383027630617**、accuracy **0.7601809954751131**；一次 test accuracy **0.747888**（約 **74.79%**）。其餘指標及 gate 紀錄見 [baseline_results.md](docs/baseline_results.md)。

### Experiment A2 — Modified C3D-style Extended Training

**Architecture and preprocessing are identical to A1.** A1 到 epoch 5 時 validation loss 仍持續下降、accuracy 仍上升，因此建立獨立 extended-training follow-up。A2 載入 A1 `best.pt` 的模型權重，但 A1 沒保存 optimizer／RNG 等完整續訓 state；A1→A2 屬於 **WARM_START_ONLY**，A2 新建 AdamW，不是 true resume。A2 自己的 `last.pt` 則保存可續訓 state。

原始 A2 上限是總 epoch 標籤 **15**、patience **5**；之後以 resume-only override 設上限 **50**、patience **7**。LR 持續 `0.001`，沒有 scheduler，以**最低 validation loss** 選最佳 checkpoint。實際在 **epoch 28** 停止（`last.pt` 的未改善 counter 為 7），最佳 epoch **21**，validation loss **0.4401281460416494**、accuracy **0.7970265029088559**。A2 test：**Not evaluated yet**；不得沿用 A1 的 test accuracy。

```text
Full video → 16 temporal bins → 16 sparse RGB frames → [3,16,96,96]
→ Modified C3D-style → GAP → FC → Make/Miss
```

### Experiment B — Reference C3D Adaptation

**Goal.** 建立較忠實於 C3D 論文與作者官方網路定義的 Basketball-51 adaptation。完整影片切成連續 **16-frame clips**，起點 stride **8**、相鄰 overlap **8**：clip 1 為 frames `0–15`、clip 2 為 `8–23`、clip 3 為 `16–31`。這與 A1/A2 從全片稀疏抽 16 frames 不同。每個 clip 輸入 `[3,16,112,112]`；先 resize 至高 128、寬 171，train 隨機裁 112×112 並以 `p=0.5` 水平翻轉，validation 使用 center crop。RGB `[-1,1]` 正規化是 **PROJECT_ADAPTATION**；原權重所需的精確 mean／色彩處理仍 **UNRESOLVED**。

**Architecture.** [實際模型](reference_c3d/model.py)的八個 Conv3D 為 Conv1a `3→64`、Conv2a `64→128`、Conv3a `128→256`、Conv3b `256→256`、Conv4a `256→512`、Conv4b `512→512`、Conv5a `512→512`、Conv5b `512→512`；kernel 都是 `3×3×3`、stride 1、padding 1，卷積後使用 ReLU。五個 pool 中 Pool1 是 `1×2×2`，Pool2–5 是 `2×2×2`。**沒有 BatchNorm 或 Global Average Pooling。**對 112×112 輸入，Pool5 是 `[512,1,4,4]`，Flatten 後為 8192；fc6/fc7 各 4096，兩處定義 dropout 0.5。fc6 的實際匯出點為 ReLU 後、dropout 前，屬 **PROJECT_ADAPTATION**。

目前正式影片級路徑從每個 clip 取得 fc6 4096-D，對同支影片所有 clips 取平均並 L2 normalization，再以獨立 `Linear(4096,2)` 做 Make/Miss 分類；reference model 雖定義 fc7/fc8，但它們不在此 loss 路徑，訓練時凍結。reference model 共 **78,003,970** parameters；另加影片 head **8,194**，其中 **61,222,658** 為此路徑的有效可訓練參數。這是 Reference C3D adaptation，**不是**原論文 benchmark reproduction。

```text
Full video → overlapping continuous 16-frame clips（stride 8）
→ Reference C3D → 每 clip 的 fc6 → 平均 fc6 → L2 normalize
→ Linear(4096,2) → Make/Miss
```

**Training / status.** 從零初始化、影片級 CrossEntropyLoss、AdamW（初始 LR `1e-4`）、ReduceLROnPlateau 與 validation-based early stopping；原 config 為上限 10 epochs、patience 3，後續以 resume-only 設定延長至上限 **50**、effective patience **7**。正式訓練**已完成**，在 **epoch 32** 因連續未改善而 early stop；最佳 checkpoint 為 epoch **25**（validation loss **0.5345329800506291**、accuracy **0.7504848093083387**）。`training_status.json` 與 32 筆 log 已核對。B test：**Not evaluated yet**。

## Experiment C — Modified C3D-style + Continuous Clips

**Research question.** 在連續重疊 clip 資料流下，使用 A1/A2 的 Modified C3D-style 卷積骨幹，探索 sampling／聚合與 backbone 的關係。C 沿用 B 的 [continuous sampler 與解碼](reference_c3d/data.py)：每個 16-frame clip 都是連續影格，起點間隔 8、重疊 8；不使用 A1/A2 的 `UniformBinSampler`。尾端不足 16 frames 時捨棄。每 clip 輸入 `[3,16,112,112]`，resize 128×171；train random crop／水平翻轉，validation center crop，RGB `[-1,1]` 正規化與 B 相同。

**Backbone / objective.** [C 模型](experiment_c/model.py)重用 A1/A2 的八層 Conv3D＋BatchNorm3d＋ReLU、五層 pooling、GAP，不改動其原本 `forward`。每 clip 在 GAP 後取得 **256-D** 特徵；對同支影片的所有 clip 特徵取平均、L2 normalization，再用新的 `Linear(256,2)` 產生 Make/Miss logits，計算**一個影片級 CrossEntropyLoss**。A1/A2 的 `256→128→2` head 沒有接入 C；C 也不使用 B 的 fc6/fc7。實測 C 模型 **6,919,234** 個 parameters，全部可訓練。

```text
Full video → overlapping continuous 16-frame clips（stride 8）
→ Modified C3D-style Conv/BN/ReLU → GAP → 每 clip 256-D
→ 平均 clip 特徵 → L2 normalize → Linear(256,2) → Make/Miss
```

**Status.** **Implementation / validation completed, full training not yet started.** 固定 manifest／split 的 config check、`[B,3,16,112,112]` shape audit、CPU smoke，以及一支 train＋一支 validation 影片的 dry-run 均通過；尚無正式 validation 指標或 test 結果。C 的完整規格與驗證紀錄見 [Experiment C protocol](docs/c3d_style_continuous_protocol.md)。

## Architecture Differences

Experiment A1/A2 是較輕量的 Modified C3D-style：完整影片稀疏取樣、BatchNorm、Global Average Pooling 與小型二分類 head。Experiment B 使用較大的 Reference C3D：連續重疊 clips、沒有 BatchNorm／GAP，並以 fc6 聚合形成影片描述子。Experiment C 重用 A2 的卷積骨幹，接上 B 的 continuous clip 管線，但換成影片級 256-D GAP 特徵聚合。

**A2 vs C** 可研究 sparse full-video frames 與 continuous overlapping clips；但 C 也從 96×96 letterbox 改成 112×112 crop，並改用影片級聚合與 classifier，且 A2 是 warm-start、C 擬從零訓練，**不是純粹的單變因比較**。**C vs B** 共用 16-frame、stride 8、112×112、影片級 mean＋L2 流程，但 backbone、特徵維度（256 vs 4096）和 classifier 輸入不同。任何未來結果都不能簡化解讀為只有模型大小的影響。

## 專案目錄與檔案

### 原始資料與研究背景

| 路徑 | 用途 |
| --- | --- |
| [`archive/Basketball_51 dataset/`](archive/Basketball_51%20dataset/) | 原始 Basketball-51 RGB 影片，約 6.1 GB。八個子目錄直接存放 MP4；請勿刪除、搬移、重新命名或覆寫。 |
| [`TOPIC 1.pdf`](TOPIC%201.pdf) | 教授的研究題目說明。YOLO、I3D/MixNet、RL 選影格、SpinalNet 等是後續方向；PDF 內的 `XXXX`、`XX.X%` 是占位文字，不是本專案實驗結果。 |
| [`papers/c3d_tran_2015.pdf`](papers/c3d_tran_2015.pdf) | Tran 等人 ICCV 2015 的完整 C3D 論文與補充資料，是 C3D 稽核的主要來源。 |

原始資料的類別與本地影片數：

| 資料夾 | 意義 | 影片數 |
| --- | --- | ---: |
| `2p0` | 兩分球未進（Miss=0） | 1,418 |
| `2p1` | 兩分球命中（Make=1） | 1,950 |
| `3p0` | 三分球未進 | 2,132 |
| `3p1` | 三分球命中 | 1,178 |
| `ft0` | 罰球未進 | 566 |
| `ft1` | 罰球命中 | 1,724 |
| `mp0` | 中距離未進 | 785 |
| `mp1` | 中距離命中 | 558 |

八個類別的 `0/1` 意義已與 [Basketball-51 原始論文](docs/references.md#basketball-51)核對。原始影片都是 H.264/MP4、320×240，FPS 包含 23.976、25、29.97。資料集**沒有可直接使用的球離手 frame 標註**；檔名中的時間不是球離手時間。

### 資料檢查與切分

| 路徑 | 用途 |
| --- | --- |
| [`scripts/inspect_dataset.py`](scripts/inspect_dataset.py) | 掃描影片數量、格式、FPS、解析度、片長、檔名、可解碼性與 SHA-256 完全重複組；可輸出 JSON。加 `--decode` 會逐支完整解碼。 |
| [`docs/dataset_summary.json`](docs/dataset_summary.json) | 上述檢查的機器可讀結果：各類數量、29 組完全重複影片、3 組 Make/Miss 衝突等。 |
| [`docs/dataset_analysis.md`](docs/dataset_analysis.md) | 第一輪資料分析：標籤、release annotation 缺口、未來 pre-release clip 方案與洩漏風險。這是研究分析文件，不代表 pre-release 流程已實作。 |
| [`scripts/build_manifest.py`](scripts/build_manifest.py) | 依檢查摘要建立清理紀錄及固定 split。會核對目前原始影片數量與 exact duplicate 組是否仍與摘要一致。 |
| [`data/processed/dataset_manifest.csv`](data/processed/dataset_manifest.csv) | **本輪實驗的主資料表**；每支原始影片一列，包含標籤、重複組、排除原因與 split。路徑相對於資料根目錄。 |
| [`data/processed/split_summary.json`](data/processed/split_summary.json) | 固定 seed `20260927`、來源群組的 split 指派、每個 split 的 Make/Miss 數量與排除統計。 |

`dataset_manifest.csv` 欄位：

| 欄位 | 解釋 |
| --- | --- |
| `video_id`, `video_path` | 相對於 `archive/Basketball_51 dataset/` 的影片 ID／路徑；兩欄目前相同。 |
| `original_class` | 原始八類之一，例如 `3p1`。 |
| `binary_label` | `0=Miss`，`1=Make`。 |
| `game_code`, `source_timestamp` | 從檔名擷取的 `v` 代碼與六位數時間；**不能直接當作已核實的比賽 ID 或 release frame**。 |
| `source_group` | split 使用的來源分組；已把有跨代碼相同影片的 `v160`、`v166` 合併。 |
| `duplicate_group` | SHA-256 完全相同影片的群組；非重複影片有各自的 `single:` ID。 |
| `excluded`, `exclusion_reason` | 是否排除及原因：`conflicting_outcome` 或 `duplicate_copy`；未排除時原因留空。 |
| `split` | `train`、`validation`、`test` 或 `excluded`。 |

### Experiment A1 / A2 程式與設定

| 路徑 | 用途 |
| --- | --- |
| [`configs/baseline.yaml`](configs/baseline.yaml) | Experiment A1 的原始五個 epoch 設定。 |
| [`configs/c3d_style_extended.yaml`](configs/c3d_style_extended.yaml) | Experiment A2 的原始 15／patience 5 設定；實際延長到 50／7 使用 resume-only CLI override，沒有改寫此歷史 config。 |
| [`baseline/__init__.py`](baseline/__init__.py) | 將 `baseline/` 標示為 Python package。 |
| [`baseline/data.py`](baseline/data.py) | `UniformBinSampler`、影片解碼、RGB 轉換、保持比例補邊、正規化、訓練增強及 PyTorch Dataset。validation/test 固定取各時間區間中心。 |
| [`baseline/model.py`](baseline/model.py) | 修改版 C3D-style 3D CNN；含八層 3D 卷積、BatchNorm3d、五層 pooling 及二分類 head。**不是原始 C3D 的完全相同架構。** |
| [`scripts/run_baseline.py`](scripts/run_baseline.py) | 四個執行階段：`smoke`、`overfit`、`train`、`evaluate`。只有前兩個 gate 通過且 manifest/protocol hash 相符，才能進入訓練；評估載入 validation loss 最低的 checkpoint。 |
| [`scripts/run_c3d_style_extended.py`](scripts/run_c3d_style_extended.py) | Experiment A2 的獨立訓練與 resume 腳本；只用 train/validation，保存 A2 `last.pt`、`best.pt` 與 log；沒有 test evaluation 階段。 |
| [`runs/full_video_c3d_style_extended_v1/`](runs/full_video_c3d_style_extended_v1/) | A2 config/data 快照、checkpoint、訓練紀錄與狀態；沒有 test 評估產物。 |

### Experiment B：Reference C3D adaptation

| 路徑 | 用途 |
| --- | --- |
| [`configs/c3d_reference.yaml`](configs/c3d_reference.yaml) | 獨立設定：16-frame clip、stride 8、resize/crop、尾段處理、fc6 聚合及 gate 參數；使用新 run 目錄。 |
| [`reference_c3d/data.py`](reference_c3d/data.py) | `ContinuousClipSampler` 和影片 Dataset；每個 clip 使用連續 frame，與 Experiment A1/A2 的 16-bin sampler 分開。 |
| [`reference_c3d/model.py`](reference_c3d/model.py) | C3D 參考拓撲、fc6 特徵擷取及明列的 Basketball-51 適配；目前沒有預訓練權重。 |
| [`reference_c3d/aggregate.py`](reference_c3d/aggregate.py) | 多個 clip 的 fc6 平均後 L2 normalization，產生 4096 維影片描述子。 |
| [`scripts/run_c3d_reference_checks.py`](scripts/run_c3d_reference_checks.py) | 只提供 `verify`、`smoke`、`subset`；不包含全量訓練或 test 評估階段。 |
| [`runs/full_video_c3d_reference_v1/`](runs/full_video_c3d_reference_v1/) | 獨立保存 clip 驗證、CUDA smoke、小樣本第一次失敗與修正後通過的紀錄；沒有正式 checkpoint。 |
| [`runs/full_video_c3d_reference_v1/feasibility/`](runs/full_video_c3d_reference_v1/feasibility/) | 本輪獨立保存 Caffe/PyTorch shape audit、三種訓練模式的短時 VRAM／吞吐、影片級記憶體及 DataLoader benchmark。它不是訓練結果。 |
| [`configs/c3d_reference_benchmark.yaml`](configs/c3d_reference_benchmark.yaml)、[`scripts/benchmark_c3d_reference.py`](scripts/benchmark_c3d_reference.py) | 短時 train-only GPU／資料載入可行性測試設定與程式；沒有完整 epoch。 |
| [`scripts/audit_c3d_shape.py`](scripts/audit_c3d_shape.py) | 以作者 Caffe pooling shape 公式與 PyTorch 實際輸出核對 pool5、flatten 和 fc6。 |
| [`configs/c3d_reference_train.yaml`](configs/c3d_reference_train.yaml)、[`scripts/run_c3d_reference_train.py`](scripts/run_c3d_reference_train.py) | B 的原始 10／patience 3 設定與正式影片級 fc6 訓練腳本；後續以 resume-only CLI 延長上限／patience。 |
| [`runs/full_video_c3d_reference_train_v1/`](runs/full_video_c3d_reference_train_v1/) | B 的正式 checkpoint、32-epoch log 與 COMPLETE 狀態；epoch 32 early stop，最佳 epoch 25；沒有 test 評估產物。 |

### Experiment C：Modified C3D-style + continuous clips

| 路徑 | 用途 |
| --- | --- |
| [`configs/c3d_style_continuous.yaml`](configs/c3d_style_continuous.yaml) | C 獨立設定與固定 manifest/split SHA-256；空間與時間前處理需與 B config 相符。 |
| [`experiment_c/model.py`](experiment_c/model.py)、[`experiment_c/aggregate.py`](experiment_c/aggregate.py) | 重用 A1/A2 卷積骨幹與 GAP，產生 256-D clip 特徵，再做 mean＋L2＋獨立二分類 head。 |
| [`scripts/run_c3d_style_continuous.py`](scripts/run_c3d_style_continuous.py) | `check`、`shape`、`smoke`、`dry-run` 與尚未執行的 `train`；僅建立 train/validation Dataset，不提供 test 評估。 |
| `runs/full_video_c3d_style_continuous_v1/` | C 預定的獨立正式 run 位置；目前**尚未建立**。 |

### 研究與實驗文件

| 路徑 | 用途 |
| --- | --- |
| [`docs/references.md`](docs/references.md) | Basketball-51、C3D、I3D 的文獻資訊、各篇在本專案中的角色及與論文實作的差異。I3D 尚未實作。 |
| [`docs/c3d_reproduction_spec.md`](docs/c3d_reproduction_spec.md) | C3D 實作稽核：逐項區分論文確認、作者官方實作確認、Basketball-51 適配選項與尚未確認事項，並比較目前 modified baseline。 |
| [`docs/c3d_reference_implementation.md`](docs/c3d_reference_implementation.md) | 新 reference C3D adaptation 的實作細節、PROJECT_ADAPTATION、實際 clip 索引、tensor shapes、失敗與通過的 gate 紀錄及限制。 |
| [`docs/c3d_training_protocol.md`](docs/c3d_training_protocol.md) | 初始化與訓練層級決策、影片級 loss／分類器比較、pretrained 相容性門檻、實測 VRAM／吞吐／DataLoader、下一階段正式設定與命令。 |
| [`docs/c3d_style_extended_protocol.md`](docs/c3d_style_extended_protocol.md) | A2 與 A1 的 warm-start 關係、延長訓練與 resume protocol。 |
| [`docs/c3d_style_continuous_protocol.md`](docs/c3d_style_continuous_protocol.md) | C 的比較目的、共用 clip pipeline、模型／loss 接線、訓練規劃與短測結果。 |
| [`docs/baseline_protocol.md`](docs/baseline_protocol.md) | 本輪研究問題、清理與 split 規則、前處理、模型、訓練、評估及限制。 |
| [`docs/baseline_results.md`](docs/baseline_results.md) | 實際 smoke/overfit、五個 epoch、test 指標、混淆矩陣和環境紀錄。 |
| [`docs/progress_log.md`](docs/progress_log.md) | 階段性進度與各次 audit 的歷史紀錄；較早的「尚未訓練」敘述屬當時狀態，現況以上方 run 檔案為準。 |

### C3D 稽核確認了什麼

| 來源 | 已確認內容 |
| --- | --- |
| [完整 C3D 論文](papers/c3d_tran_2015.pdf) | 最終網路有八層 `3×3×3` 卷積、五層 max-pool、fc6/fc7 各 4096 units；訓練 crop 為 `3×16×112×112`。影片描述子使用**連續** 16-frame clips、相鄰 clips 重疊 8 frames，平均 fc6 特徵後做 L2 normalization。 |
| [作者官方 C3D 程式庫](https://github.com/facebookarchive/C3D)的網路定義 | 卷積及 fc6/fc7 後的 ReLU、fc6/fc7 後 0.5 dropout、卷積的時間與空間 padding 1、bias 和初始化宣告；檢查的定義沒有 BatchNorm/LayerNorm。這些細節不能誤寫成論文正文有明確說明。 |
| 本專案 A1/A2 程式 | 從整段影片的 16 個時間區間各取一 frame，得到 `[3,16,96,96]`；採較窄通道、BatchNorm、global average pooling 及二分類 head，沒有原始 fc6/fc7 或多 clip 聚合。 |

**影片切 clip 的程式位置：**Experiment A1/A2 由 [`baseline/data.py`](baseline/data.py) 的 `UniformBinSampler.indices()` 從完整影片的 16 個區間各取一 frame。Experiment B 由 [`reference_c3d/data.py`](reference_c3d/data.py) 的 `ContinuousClipSampler.indices()` 產生連續 16-frame、stride 8 的多個 clips；`C3DVideoDataset.__getitem__()` 解碼、前處理並組成 `[clips,3,16,112,112]`。兩者使用不同 config 與 run 目錄。

待確認的 C3D 實作細節包含原實驗權重與 seed 的完整來源、測試 clip 的精確位置、與原權重相容的色彩及 mean preprocessing，以及 fc6 特徵匯出的精確位置。本專案已把尾段丟棄、pooling rounding 與 fc6 擷取點明列為 PROJECT_ADAPTATION；B 雖已有正式訓練 checkpoint，仍不能稱為重現原論文的 benchmark 成績。

先前已用作者 Caffe pooling 公式確認目前 PyTorch `ceil_mode=True` 的 **pool5 shape `[B,512,1,4,4]`、flatten 8192、fc6 輸入 8192**；這只核對 shape，未驗證 pretrained 權重的數值相容性。短時 GPU 測試的 clip batch 1/2/4/8 皆可 forward/backward，但「8 clips 可行」不等於「8 支完整影片同時保留梯度圖可行」。B 的訓練路徑採一支影片分塊處理、影片級 loss；詳細理由與風險見 [training protocol](docs/c3d_training_protocol.md)。

### Experiment A1 已完成實驗的產物

[`runs/full_video_c3d_style_v1/`](runs/full_video_c3d_style_v1/) 是已完成的 Experiment A1；其中各檔用途如下：

| 檔案 | 用途 |
| --- | --- |
| `config.yaml` | 此次執行時使用的設定副本。 |
| `dataset_manifest.csv` | 此次執行時使用的 manifest 副本，供結果核對；不需從原始資料重新猜測 split。 |
| `split_summary.json` | 此次執行的 split 統計與群組指派副本。 |
| `environment.json` | Python、PyTorch、torchvision、NumPy、OpenCV、PyYAML、CUDA 與 GPU 資訊。 |
| `smoke.json` | GPU smoke test 的輸入／輸出形狀、標籤、loss、梯度及通過狀態。 |
| `overfit.json` | 八支訓練影片的小樣本 overfit gate 紀錄；這不是正式準確率。 |
| `training_log.json` | 每個 epoch 的 train/validation loss、accuracy 與耗時。 |
| `training_status.json` | 訓練完成狀態、最佳 epoch、最低 validation loss、manifest hash 與 peak GPU memory。 |
| `best.pt` | 依最低 validation loss 儲存的 PyTorch checkpoint。 |
| `test_metrics.json` | 固定 checkpoint 的一次 test accuracy、precision、recall、F1 與混淆矩陣；Make (`1`) 為正類。 |
| `test_predictions.csv` | 1,539 支 test 影片的相對 ID、真值、預測與 Make 機率；不包含不必要的本機絕對路徑。 |

## 如何使用現有流程

目前記錄中的實驗使用既有的 **`yolo11` Conda environment**（Python 3.10.20、PyTorch 2.11.0+cu128）；未修改系統 Python 或安裝新套件。請從專案根目錄執行。以下前兩個歷史命令會**重新產生摘要與 manifest/split 檔案**；若只是查看既有結果，直接閱讀上面的文件和 `runs/` 即可。

```bash
conda run -n yolo11 python -B scripts/inspect_dataset.py --decode --output docs/dataset_summary.json
conda run -n yolo11 python -B scripts/build_manifest.py
```

Experiment A1 的原始階段命令如下，保留供理解當時流程；**不要對已完成的 A1 run 直接重新執行**：原腳本會寫入同一個 `runs/full_video_c3d_style_v1/` 目錄。

```bash
conda run -n yolo11 python -B scripts/run_baseline.py --stage smoke
conda run -n yolo11 python -B scripts/run_baseline.py --stage overfit
conda run -n yolo11 python -B scripts/run_baseline.py --stage train
conda run -n yolo11 python -B scripts/run_baseline.py --stage evaluate
```

A2 的獨立訓練／resume 指令與歷史設定見 [A2 protocol](docs/c3d_style_extended_protocol.md)；B 的影片級訓練與 resume 指令見 [B training protocol](docs/c3d_training_protocol.md)。C 的 config、check／smoke／dry-run 與下一階段正式訓練指令見 [C protocol](docs/c3d_style_continuous_protocol.md)。A1/A2/B run 已完成訓練；C 尚未訓練。**不要因 README 的指令而重跑已完成實驗或提前評估 A2/B/C test**。

## 解讀結果時的限制

- 模型看的是**完整影片**，可能直接利用球飛行、碰框、籃網與投籃後反應辨認結果；test accuracy **不能**證明模型在球離手前就能預測。
- `v` 代碼尚未由原始比賽清單核實；目前是來源代碼分組，不可宣稱已完成真正的 game-level holdout。SHA-256 也找不到轉碼後的近重複影片。
- 此 CUDA/PyTorch 組合的 `max_pool3d` backward 沒有 deterministic GPU 實作；雖固定 seed、split 與設定，重跑未必得到逐位元相同的權重。
- 未來的 pre-release-only、I3D、YOLO 或強化學習方法仍是**待規劃／待驗證**，目前沒有其訓練結果。
