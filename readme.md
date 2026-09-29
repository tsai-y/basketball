# Basketball-51 籃球影片研究專案

本專案目前完成 **Basketball-51 完整影片的 Make / Miss 二元分類 baseline**：從 RGB 影片取樣影格，送入修改版 C3D-style 3D CNN，預測投籃結果。研究的長期目標是只利用**球離手前**畫面預測結果，但目前的模型會看到完整約六秒影片，**不能稱為 pre-release prediction**，也不是原始 C3D 論文的完整重現。

## 目前完成到哪裡

1. 檢查本地 Basketball-51 的目錄、標籤、影片規格與解碼狀況。
2. 用 manifest 記錄完全重複影片、互相矛盾的標籤、排除原因與固定 split；原始影片沒有被移動或改寫。
3. 建立並執行 RGB Dataset/DataLoader、modified C3D-style 模型、smoke test、小樣本 overfit test、五個 epoch 訓練和一次 test 評估。
4. 完成原始 C3D 論文與作者官方 Caffe 網路定義的實作稽核，寫成 reference-faithful C3D reproduction specification。
5. 依規格新增獨立的 continuous-clip C3D adaptation 與 fc6 影片描述子流程；clip 索引、CUDA smoke 和兩支 train 影片的小樣本 pipeline gate 已通過。**Reference C3D 尚未進行完整資料訓練或 test 評估。**
6. 完成 Reference C3D 的短時 shape、VRAM、throughput、DataLoader feasibility 與正式訓練 protocol；正式訓練腳本只通過 config check 和單支 train／validation dry-run，**尚未跑完整 epoch**。

原始 **10,311** 支影片中，排除 **6** 支結果標籤衝突影片及 **26** 支一致標籤的重複複本，留下 **10,279** 支。可用資料分為 train **7,193**、validation **1,547**、test **1,539**。這些是固定來源代碼分組，**尚未由原始比賽資料核實為真正的 game-level split**。已完成的 modified C3D-style baseline test accuracy 為 **74.79%**；完整指標、失敗與修正紀錄見 [baseline_results.md](docs/baseline_results.md)。新 reference C3D 的小樣本 gate 不是正式研究結果。

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

### Baseline 程式與設定

| 路徑 | 用途 |
| --- | --- |
| [`configs/baseline.yaml`](configs/baseline.yaml) | 集中存放 dataset/manifest 路徑、seed、16 frames、影像大小、batch size、learning rate、epochs、裝置及 checkpoint 路徑等設定。 |
| [`baseline/__init__.py`](baseline/__init__.py) | 將 `baseline/` 標示為 Python package。 |
| [`baseline/data.py`](baseline/data.py) | `UniformBinSampler`、影片解碼、RGB 轉換、保持比例補邊、正規化、訓練增強及 PyTorch Dataset。validation/test 固定取各時間區間中心。 |
| [`baseline/model.py`](baseline/model.py) | 修改版 C3D-style 3D CNN；含八層 3D 卷積、BatchNorm3d、五層 pooling 及二分類 head。**不是原始 C3D 的完全相同架構。** |
| [`scripts/run_baseline.py`](scripts/run_baseline.py) | 四個執行階段：`smoke`、`overfit`、`train`、`evaluate`。只有前兩個 gate 通過且 manifest/protocol hash 相符，才能進入訓練；評估載入 validation loss 最低的 checkpoint。 |

### Reference C3D adaptation（尚未完整訓練）

| 路徑 | 用途 |
| --- | --- |
| [`configs/c3d_reference.yaml`](configs/c3d_reference.yaml) | 獨立設定：16-frame clip、stride 8、resize/crop、尾段處理、fc6 聚合及 gate 參數；使用新 run 目錄。 |
| [`reference_c3d/data.py`](reference_c3d/data.py) | `ContinuousClipSampler` 和影片 Dataset；每個 clip 使用連續 frame，與 Experiment A 的 16-bin sampler 分開。 |
| [`reference_c3d/model.py`](reference_c3d/model.py) | C3D 參考拓撲、fc6 特徵擷取及明列的 Basketball-51 適配；目前沒有預訓練權重。 |
| [`reference_c3d/aggregate.py`](reference_c3d/aggregate.py) | 多個 clip 的 fc6 平均後 L2 normalization，產生 4096 維影片描述子。 |
| [`scripts/run_c3d_reference_checks.py`](scripts/run_c3d_reference_checks.py) | 只提供 `verify`、`smoke`、`subset`；不包含全量訓練或 test 評估階段。 |
| [`runs/full_video_c3d_reference_v1/`](runs/full_video_c3d_reference_v1/) | 獨立保存 clip 驗證、CUDA smoke、小樣本第一次失敗與修正後通過的紀錄；沒有正式 checkpoint。 |
| [`runs/full_video_c3d_reference_v1/feasibility/`](runs/full_video_c3d_reference_v1/feasibility/) | 本輪獨立保存 Caffe/PyTorch shape audit、三種訓練模式的短時 VRAM／吞吐、影片級記憶體及 DataLoader benchmark。它不是訓練結果。 |
| [`configs/c3d_reference_benchmark.yaml`](configs/c3d_reference_benchmark.yaml)、[`scripts/benchmark_c3d_reference.py`](scripts/benchmark_c3d_reference.py) | 短時 train-only GPU／資料載入可行性測試設定與程式；沒有完整 epoch。 |
| [`scripts/audit_c3d_shape.py`](scripts/audit_c3d_shape.py) | 以作者 Caffe pooling shape 公式與 PyTorch 實際輸出核對 pool5、flatten 和 fc6。 |
| [`configs/c3d_reference_train.yaml`](configs/c3d_reference_train.yaml)、[`scripts/run_c3d_reference_train.py`](scripts/run_c3d_reference_train.py) | **尚未執行的正式訓練方案**：既有固定 split、影片級 fc6 loss、獨立 `runs/full_video_c3d_reference_train_v1/`、validation checkpoint 與 epoch-boundary resume。 |

### 研究與實驗文件

| 路徑 | 用途 |
| --- | --- |
| [`docs/references.md`](docs/references.md) | Basketball-51、C3D、I3D 的文獻資訊、各篇在本專案中的角色及與論文實作的差異。I3D 尚未實作。 |
| [`docs/c3d_reproduction_spec.md`](docs/c3d_reproduction_spec.md) | C3D 實作稽核：逐項區分論文確認、作者官方實作確認、Basketball-51 適配選項與尚未確認事項，並比較目前 modified baseline。 |
| [`docs/c3d_reference_implementation.md`](docs/c3d_reference_implementation.md) | 新 reference C3D adaptation 的實作細節、PROJECT_ADAPTATION、實際 clip 索引、tensor shapes、失敗與通過的 gate 紀錄及限制。 |
| [`docs/c3d_training_protocol.md`](docs/c3d_training_protocol.md) | 初始化與訓練層級決策、影片級 loss／分類器比較、pretrained 相容性門檻、實測 VRAM／吞吐／DataLoader、下一階段正式設定與命令。 |
| [`docs/baseline_protocol.md`](docs/baseline_protocol.md) | 本輪研究問題、清理與 split 規則、前處理、模型、訓練、評估及限制。 |
| [`docs/baseline_results.md`](docs/baseline_results.md) | 實際 smoke/overfit、五個 epoch、test 指標、混淆矩陣和環境紀錄。 |
| [`docs/progress_log.md`](docs/progress_log.md) | 階段性進度摘要；reference C3D 的 pipeline、驗證 gate 與訓練 feasibility/protocol 已完成，完整訓練／特徵擷取實驗尚未執行。 |

### C3D 稽核確認了什麼

| 來源 | 已確認內容 |
| --- | --- |
| [完整 C3D 論文](papers/c3d_tran_2015.pdf) | 最終網路有八層 `3×3×3` 卷積、五層 max-pool、fc6/fc7 各 4096 units；訓練 crop 為 `3×16×112×112`。影片描述子使用**連續** 16-frame clips、相鄰 clips 重疊 8 frames，平均 fc6 特徵後做 L2 normalization。 |
| [作者官方 C3D 程式庫](https://github.com/facebookarchive/C3D)的網路定義 | 卷積及 fc6/fc7 後的 ReLU、fc6/fc7 後 0.5 dropout、卷積的時間與空間 padding 1、bias 和初始化宣告；檢查的定義沒有 BatchNorm/LayerNorm。這些細節不能誤寫成論文正文有明確說明。 |
| 本專案程式 | 現有模型從整段影片的 16 個時間區間各取一 frame，得到 `[3,16,96,96]`；採較窄通道、BatchNorm、global average pooling 及二分類 head，沒有原始 fc6/fc7 或多 clip 聚合。 |

**影片切 clip 的程式位置：**Experiment A 由 [`baseline/data.py`](baseline/data.py) 的 `UniformBinSampler.indices()` 從完整影片的 16 個區間各取一 frame。新 reference C3D 由 [`reference_c3d/data.py`](reference_c3d/data.py) 的 `ContinuousClipSampler.indices()` 產生連續 16-frame、stride 8 的多個 clips；`C3DVideoDataset.__getitem__()` 解碼、前處理並組成 `[clips,3,16,112,112]`。兩者使用不同 config 與 run 目錄。

待確認的 C3D 實作細節包含原實驗權重與 seed 的完整來源、測試 clip 的精確位置、與原權重相容的色彩及 mean preprocessing，以及 fc6 特徵匯出的精確位置。本專案已把尾段丟棄、pooling rounding 與 fc6 擷取點明列為 PROJECT_ADAPTATION；下一步是另行規劃完整訓練／特徵擷取實驗，仍不能稱為重現原論文的 benchmark 成績。

本輪已用作者 Caffe pooling 公式確認目前 PyTorch `ceil_mode=True` 的 **pool5 shape `[B,512,1,4,4]`、flatten 8192、fc6 輸入 8192**；這只核對 shape，未驗證 pretrained 權重的數值相容性。短時 GPU 測試的 clip batch 1/2/4/8 皆可 forward/backward，但「8 clips 可行」不等於「8 支完整影片同時保留梯度圖可行」。正式訓練方案選擇一支影片分塊處理、影片級 loss；詳細理由與風險見 [training protocol](docs/c3d_training_protocol.md)。

### 已完成實驗的產物

[`runs/full_video_c3d_style_v1/`](runs/full_video_c3d_style_v1/) 是已完成的 Experiment A；其中各檔用途如下：

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

本輪實驗使用既有的 **`yolo11` Conda environment**（Python 3.10.20、PyTorch 2.11.0+cu128）；未修改系統 Python 或安裝新套件。請從專案根目錄執行。以下前兩個命令會**重新產生對應的摘要與 manifest/split 檔案**；若只是查看既有結果，直接閱讀上面的文件和 `runs/` 即可。

```bash
conda run -n yolo11 python -B scripts/inspect_dataset.py --decode --output docs/dataset_summary.json
conda run -n yolo11 python -B scripts/build_manifest.py
```

模型流程的階段名稱如下；重新執行 `train`／`evaluate` 會寫入同一個 `runs/full_video_c3d_style_v1/` 目錄，研究比較前應先保存目前產物並使用新的 run path。

```bash
conda run -n yolo11 python -B scripts/run_baseline.py --stage smoke
conda run -n yolo11 python -B scripts/run_baseline.py --stage overfit
conda run -n yolo11 python -B scripts/run_baseline.py --stage train
conda run -n yolo11 python -B scripts/run_baseline.py --stage evaluate
```

Reference C3D **下一階段**的正式命令已寫入 [training protocol](docs/c3d_training_protocol.md)。本輪只執行其 `--check` 與 `--dry-run`；目前沒有建立 `runs/full_video_c3d_reference_train_v1/` 或開始完整訓練。

## 解讀結果時的限制

- 模型看的是**完整影片**，可能直接利用球飛行、碰框、籃網與投籃後反應辨認結果；test accuracy **不能**證明模型在球離手前就能預測。
- `v` 代碼尚未由原始比賽清單核實；目前是來源代碼分組，不可宣稱已完成真正的 game-level holdout。SHA-256 也找不到轉碼後的近重複影片。
- 此 CUDA/PyTorch 組合的 `max_pool3d` backward 沒有 deterministic GPU 實作；雖固定 seed、split 與設定，重跑未必得到逐位元相同的權重。
- 未來的 pre-release-only、I3D、YOLO 或強化學習方法仍是**待規劃／待驗證**，目前沒有其訓練結果。
