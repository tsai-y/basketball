# Basketball-51 籃球影片研究

本專案目前研究 **full-video Make/Miss classification**（完整影片投籃命中／未進分類）。A1/A2/B/C 已完成訓練與一次性正式 test，均為 **frozen baseline**。下一階段優先規劃自動品質篩檢與影像增強，再研究特徵擷取及多模型互補／融合；增強策略、模型組合與融合方式尚未決定。

**本輪只整理 repository、路徑與文件；沒有開始資料清洗、模型實作或訓練。** Gate 1A／1B completed；Gate 1C 與原 Experiment D paused。Pre-release 不在目前範圍；SPL 指 Sport Performance Lab 的 SPL Open Data，是最後才考慮的資料集延伸，不是模型模組。

## 組員從哪裡開始

1. [目前研究方向](docs/current_research_direction.md)：當前範圍、優先順序與暫停項目。
2. [Repository map](docs/repository_map.md)：目錄責任、保留原位的原因及完整入口。
3. [Baseline 說明](baseline/README.md)：A1/A2/B/C 的程式、設定及歷史產物。
4. [Development 規則](development/README.md) → [自動品質篩檢](development/data_quality/README.md)：新工作放置方式。
5. [文件索引](docs/README.md)、[進度紀錄](docs/progress_log.md)、[本次整理紀錄](docs/reorganization_log.md)：研究依據與變更查核。

## 結構與責任

```text
basketball/
├── README.md                    # 目前團隊入口
├── baseline/                    # frozen；A1/A2 程式與基準索引
│   ├── README.md
│   ├── data.py / model.py
│   └── scripts/                 # B 的歷史 shape audit、benchmark 入口
├── development/                 # 新研究；本輪僅文件
│   ├── README.md
│   └── data_quality/README.md
├── docs/                        # 目前方向、研究文件、歷史與稽核證據
├── reference_c3d/               # frozen B；B/C 共用 continuous clip 程式，原位
├── experiment_c/                # frozen C 程式，原位
├── configs/                     # 六份 frozen 基準設定，原位
├── scripts/                     # frozen 訓練／評估、共用資料工具、Gate 1A，原位
├── data/processed/              # 固定 manifest / split，原位
├── archive/Basketball_51 dataset/ # 原始影片，原位、本機 ignored
├── runs/                        # 全部歷史產物，原位
└── papers/、TOPIC 1.pdf          # 本機研究文獻，ignored
```

`baseline/` 是已完成基準的責任入口，不代表所有 frozen 檔案都已搬入該目錄。核心程式與設定存在 source／config／protocol hash 相依，因此保留原位並建立索引，不複製第二套程式。新方法應放在 `development/`，不可直接修改 baseline 或原位保留的 frozen 元件。

## 已完成成果與 frozen 範圍

| 實驗 | 定位 | 正式產物 |
| --- | --- | --- |
| A1 | Modified C3D-style 初始 full-video 基準 | [A1 run](runs/full_video_c3d_style_v1/) |
| A2 | A1 權重 warm-start 的延長訓練；架構與前處理相同 | [A2 run](runs/full_video_c3d_style_extended_v1/) |
| B | Reference C3D adaptation；continuous clips、fc6 聚合 | [B run](runs/full_video_c3d_reference_train_v1/) |
| C | Modified backbone；continuous clips、GAP 特徵聚合 | [C run](runs/full_video_c3d_style_continuous_v1/) |

結果及限制以 [Research Baseline v1.3](docs/research_baseline_v1.3.md)、[正式 test protocol](docs/formal_test_evaluation_protocol.md) 和各 run 的 `test_metrics.json`／`test_predictions.csv` 為準。模型、sampling／preprocessing、超參數、seed、split、指標計算、checkpoint 選擇與正式產物均 frozen。不要根據 test 結果回頭調整或重跑。這些是完整流程比較，不能解讀為單變因結論，也不能宣稱具有 pre-release 預測能力。

## 共用資料與歷史證據

- [原始影片](archive/Basketball_51%20dataset/)：保持原內容、檔名及位置；不以複製資料的方式建立新階段。
- [固定 manifest](data/processed/dataset_manifest.csv)、[固定 split](data/processed/split_summary.json)：唯一共用輸入；manifest 的影片路徑相對於資料根目錄。run 內原有快照是歷史證據，保留但不再複製新主資料表。
- [歷史 runs](runs/)：checkpoints、training logs、環境、config／split 快照、formal test metrics／predictions 及 gates 均不可覆寫或搬移。新研究輸出使用 `development/outputs/<實驗識別碼>/` 等獨立位置，執行前記錄；本輪未建立產物。
- [Gate 1A](docs/i3d_temporal_feasibility_audit.md)／[JSON](docs/i3d_temporal_feasibility_audit.json)、[Gate 1B](docs/i3d_pretrained_artifact_provenance_audit.md)／[JSON](docs/i3d_pretrained_artifact_provenance_audit.json)：已完成證據，原文原位保留，不表示已選定 I3D 或啟動 Gate 1C。

原始影片、PDF、checkpoint 多為 Git ignored；clone 不會自動取得它們。請依團隊資料供應方式放回 map 所列既有位置，不要為補檔重建 split 或重跑歷史實驗。歷史數量為 train 7,193、validation 1,547、test 1,539；來源代碼分組尚未核實為真正 game-level split。

## 從 repository root 使用既有工具

既有實驗環境為 Conda `yolo11`。以下命令只顯示說明；最後一項只讀 manifest／split metadata，沒有影片 probe、解碼或 checkpoint 載入：

```bash
conda run -n yolo11 python -B scripts/run_baseline.py --help
conda run -n yolo11 python -B baseline/scripts/audit_c3d_shape.py --help
conda run -n yolo11 python -B baseline/scripts/benchmark_c3d_reference.py --help
python3 -B scripts/inspect_dataset.py --help
python3 -B scripts/build_manifest.py --help
python3 -B scripts/audit_i3d_temporal_feasibility.py --mode check
```

更多入口見 [baseline 說明](baseline/README.md)。保留既有路徑慣例：Python 入口由 `__file__` 找 repository root，設定的資源路徑相對於 root；明確傳入的相對 CLI 路徑請從 root 執行。不需要新增 `PYTHONPATH` 或本機絕對路徑。

`inspect_dataset.py` 正常模式會掃描／probe 影片；`build_manifest.py` 會重建 manifest／split；shape audit 與 benchmark 正常模式會寫入歷史 feasibility 目錄。不要將這些正常模式當作整理驗證。訓練與評估入口僅保留供追溯，**evaluator 的 `--check` 仍會載入 checkpoint**，本輪也不執行。舊 protocol 內的訓練、resume、evaluation 命令屬歷史，不是現在的開始步驟。
