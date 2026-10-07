# Frozen full-video baseline

A1/A2/B/C 的訓練及一次性正式 test 已完成。此目錄是基準的統一閱讀入口；核心程式、設定與入口有雜湊及歷史路徑相依，部分仍在 root 的既有位置。**本輪目錄整理不代表重新訓練或重新驗證結果。不要在此開發新方法。**

## 實驗、程式與設定

| 實驗 | 用途與程式 | 設定與執行入口 | 歷史產物 |
| --- | --- | --- | --- |
| A1 | 初始 Modified C3D-style；[data.py](data.py) 稀疏 16-bin 取樣、[model.py](model.py) 分類 | [baseline.yaml](../configs/baseline.yaml)、[run_baseline.py](../scripts/run_baseline.py) | [full_video_c3d_style_v1](../runs/full_video_c3d_style_v1/) |
| A2 | A1 最佳權重 warm-start，延長訓練；重用同一份 A1 data/model，非 A1 optimizer true resume | [c3d_style_extended.yaml](../configs/c3d_style_extended.yaml)、[run_c3d_style_extended.py](../scripts/run_c3d_style_extended.py) | [full_video_c3d_style_extended_v1](../runs/full_video_c3d_style_extended_v1/) |
| B | Reference C3D adaptation；[模型](../reference_c3d/model.py)、[continuous clip 資料流](../reference_c3d/data.py)、[fc6 mean＋L2](../reference_c3d/aggregate.py) | [clip 設定](../configs/c3d_reference.yaml)、[訓練設定](../configs/c3d_reference_train.yaml)、[訓練入口](../scripts/run_c3d_reference_train.py) | [full_video_c3d_reference_train_v1](../runs/full_video_c3d_reference_train_v1/) |
| C | [Modified backbone](../experiment_c/model.py) 重用 A1 features/GAP、B 的 data；[256-D mean＋L2](../experiment_c/aggregate.py) | [c3d_style_continuous.yaml](../configs/c3d_style_continuous.yaml)、[run_c3d_style_continuous.py](../scripts/run_c3d_style_continuous.py) | [full_video_c3d_style_continuous_v1](../runs/full_video_c3d_style_continuous_v1/) |

A2 設定仍保留原始 15／patience 5，B 設定保留原始 10／patience 3；實際訓練曾使用 resume-only override。請讀 [A2 protocol](../docs/c3d_style_extended_protocol.md)、[B protocol](../docs/c3d_training_protocol.md) 及 run logs，勿為配合最終 epoch 改寫原 config。

## 歷史支援工具

| 入口 | 用途與對應資料 |
| --- | --- |
| [run_c3d_reference_checks.py](../scripts/run_c3d_reference_checks.py) | B 的 verify／smoke／subset；[歷史 gate 產物](../runs/full_video_c3d_reference_v1/) |
| [scripts/audit_c3d_shape.py](scripts/audit_c3d_shape.py) | B Caffe pooling shape 比對；使用 [benchmark 設定](../configs/c3d_reference_benchmark.yaml) |
| [scripts/benchmark_c3d_reference.py](scripts/benchmark_c3d_reference.py) | B 訓練模式、DataLoader、VRAM／吞吐 benchmark；使用同一份 benchmark 設定 |
| [evaluate_full_video_experiments.py](../scripts/evaluate_full_video_experiments.py) | A2/B/C 一次性正式 test evaluator；結果已存在，不重跑 |

兩個搬入的輔助腳本仍指向 [原 feasibility 目錄](../runs/full_video_c3d_reference_v1/feasibility/)，正常執行可能覆寫歷史證據，僅供追溯；本輪只驗證 import／`--help`。未新增轉接腳本或複製共用程式。

## Frozen 邊界

- 範圍包括本目錄 Python、`reference_c3d/`、`experiment_c/`、六份 `configs/*.yaml`、對應訓練／評估／gate／benchmark 入口，以及全部既有 `runs/`。放在 root 不代表可用來開發新方法。
- 不修改模型架構、sampling／preprocessing、超參數、seed、split、指標計算、checkpoint 選擇。原始資料、固定 manifest／split 與歷史產物保持原位、原內容。
- `reference_c3d/data.py` 確實由 B/C 共用；C 確實匯入 `baseline.model`。繼續維持單一來源，後續研究若需改變行為應在 [development](../development/README.md) 建立獨立研究元件及版本紀錄，不修改 frozen 實作。
- 結果讀 [Research Baseline](../docs/research_baseline_v1.3.md)、[A1 結果](../docs/baseline_results.md)、[正式 test protocol](../docs/formal_test_evaluation_protocol.md)。本輪未載入 checkpoint、未驗證模型數值等價。

## 查看 CLI

在 repository root、既有 `yolo11` 環境執行，將下表路徑代入 `conda run -n yolo11 python -B <入口> --help`：

| 實驗／用途 | 入口 |
| --- | --- |
| A1 | `scripts/run_baseline.py` |
| A2 | `scripts/run_c3d_style_extended.py` |
| B 訓練／gate | `scripts/run_c3d_reference_train.py`、`scripts/run_c3d_reference_checks.py` |
| C | `scripts/run_c3d_style_continuous.py` |
| A2/B/C evaluator | `scripts/evaluate_full_video_experiments.py` |
| B shape／benchmark | `baseline/scripts/audit_c3d_shape.py`、`baseline/scripts/benchmark_c3d_reference.py` |

`--help` 會在資料與模型執行前退出；其他名稱含 check、smoke、dry-run 的模式不一定只是路徑驗證。尤其 evaluator `--check` 會載入 checkpoint，不屬本輪驗證範圍。
