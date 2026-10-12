# Frozen full-video baseline

> - **給誰看：** 要查 A1/A2/B/C 定義、程式或歷史產物的組員。
> - **回答什麼：** 四個 full-video baseline 的差異、入口與 frozen 邊界。
> - **不包含：** 資料品質技術、重跑教學或新方法開發規格。
> - **上一層入口：** [專案 README](../README.md)。
> - **建議下一份：** [Research Baseline v1.3](../docs/research_baseline_v1.3.md)。

A1/A2/B/C 的訓練與一次性正式 test 已完成。此目錄是統一查詢入口；部分核心程式、設定與產物因歷史路徑及 hash 相依而保留原位。這不代表可以在原位開發新方法。

## A1/A2/B/C 定義與入口

| 實驗 | 定義 | 主要程式／設定 | 研究文件 |
| --- | --- | --- | --- |
| A1 | Modified C3D-style 初始 full-video baseline；稀疏 16-bin 取樣 | [data.py](data.py)、[model.py](model.py)、[baseline.yaml](../configs/baseline.yaml)、[run_baseline.py](../scripts/run_baseline.py) | [protocol](../docs/baseline_protocol.md)、[results](../docs/baseline_results.md) |
| A2 | A1 權重 warm-start 的延長訓練；不是 optimizer true resume | A1 data/model、[c3d_style_extended.yaml](../configs/c3d_style_extended.yaml)、[runner](../scripts/run_c3d_style_extended.py) | [protocol](../docs/c3d_style_extended_protocol.md) |
| B | Reference C3D adaptation；continuous clips、fc6 mean＋L2 聚合 | [reference_c3d](../reference_c3d/)、[training config](../configs/c3d_reference_train.yaml)、[runner](../scripts/run_c3d_reference_train.py) | [implementation](../docs/c3d_reference_implementation.md)、[training protocol](../docs/c3d_training_protocol.md) |
| C | Modified backbone；continuous clips、GAP 特徵 mean＋L2 聚合 | [experiment_c](../experiment_c/)、[config](../configs/c3d_style_continuous.yaml)、[runner](../scripts/run_c3d_style_continuous.py) | [protocol](../docs/c3d_style_continuous_protocol.md) |

四個實驗都是 full-video 流程比較，不是單變因結論，也不證明 pre-release 能力。完整結果與限制只讀 [Research Baseline](../docs/research_baseline_v1.3.md) 及[正式 test protocol](../docs/formal_test_evaluation_protocol.md)。

## Frozen 邊界

- 不修改 `baseline/`、`reference_c3d/`、`experiment_c/`、對應 `configs/`、訓練／評估入口或既有 `runs/`。
- 不改模型架構、sampling／preprocessing、超參數、seed、split、指標、checkpoint 選擇或正式產物。
- `reference_c3d/data.py` 由 B/C 共用，C 匯入 `baseline.model`；維持單一來源。
- 新研究放在 [development](../development/README.md)，使用獨立設定、程式與輸出位置。

## 主要歷史入口

| 用途 | 入口 |
| --- | --- |
| A1/A2/B/C 完整研究統整 | [Research Baseline v1.3](../docs/research_baseline_v1.3.md) |
| A2/B/C 一次性正式 test | [正式 test protocol](../docs/formal_test_evaluation_protocol.md) |
| B shape／VRAM／throughput gates | [run_c3d_reference_checks.py](../scripts/run_c3d_reference_checks.py)、[baseline/scripts](scripts/) |
| A2/B/C evaluator | [evaluate_full_video_experiments.py](../scripts/evaluate_full_video_experiments.py) |
| 歷史執行與決策 | [progress log](../docs/progress_log.md) |

歷史入口只供追溯；名稱含 check、smoke 或 dry-run 不保證不載入模型或寫入產物，不應把它們當作本次文件整理的驗證命令。
