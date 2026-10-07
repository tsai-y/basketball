# 文件目錄

本目錄收錄 Basketball-51 專案的研究統整、資料稽核、實驗 protocol、結果與參考文獻。若是第一次閱讀，請從下方的「建議閱讀順序」開始；若要查特定實驗，則直接使用分類索引。

> 根目錄的 [`README.md`](../README.md) 負責專案概覽、目前狀態與程式入口；本頁只負責文件導航。各實驗的詳細設定與限制應以其專屬 protocol、實作紀錄及凍結產物為準。

## 2026-10-07 團隊入口

先讀 [目前研究方向](current_research_direction.md)、[repository map](repository_map.md) 與 [整理紀錄](reorganization_log.md)。目前優先規劃 full-video 自動品質篩檢與影像增強；A1/A2/B/C frozen，Gate 1A／1B completed，Gate 1C 與原 Experiment D paused。以下既有文件的 D／pre-release 規劃保留其歷史語境，不代表目前工作排程。Research Baseline 與 Gate 報告內容及路徑均未改動。

## 建議閱讀順序

1. [`research_baseline_v1.3.md`](research_baseline_v1.3.md)：目前最完整、已凍結的研究統整入口。
2. [`dataset_analysis.md`](dataset_analysis.md)：了解資料內容、標籤、重複影片與 pre-release 缺口。
3. [`formal_test_evaluation_protocol.md`](formal_test_evaluation_protocol.md)：了解 A2／B／C 的正式測試規則與不可覆寫原則。
4. 依關注的實驗閱讀下方專屬 protocol 與實作紀錄。
5. 若準備 Experiment D，再閱讀兩份 I3D 前置稽核。

## 狀態標記

| 標記 | 意義 |
| --- | --- |
| **Frozen** | 已完成審閱或正式執行，不應因後續 test 表現回頭修改。 |
| **Current** | 目前仍適用的 protocol、分析或實作說明。 |
| **Historical** | 保留研究與執行歷程；不一定代表目前的最終設定。 |
| **Generated** | 由腳本產生的機器可讀資料；通常應搭配同名 Markdown 摘要閱讀。 |

## 研究總覽與共用規範

| 文件 | 狀態 | 用途 |
| --- | --- | --- |
| [`research_baseline_v1.3.md`](research_baseline_v1.3.md) | **Frozen** | A1／A2／B／C 結果、研究限制、C3D／I3D 證據與 Experiment D 前置研究的整體統整；為目前研究脈絡的主要入口。 |
| [`formal_test_evaluation_protocol.md`](formal_test_evaluation_protocol.md) | **Frozen** | A2／B／C 的 frozen inputs、正式 test 流程、產物與一次性評估紀錄。 |
| [`references.md`](references.md) | **Current** | Basketball-51、C3D、I3D 與教授題目文件的來源及其在本專案中的角色。 |
| [`progress_log.md`](progress_log.md) | **Historical** | 按時間保留專案進度、決策與執行紀錄；查最新結論時優先使用研究統整及專屬文件。 |

## 資料集與前處理

| 文件 | 狀態 | 用途 |
| --- | --- | --- |
| [`dataset_analysis.md`](dataset_analysis.md) | **Current** | 資料結構、標籤、異常與重複影片、split 風險，以及 pre-release annotation 缺口。 |
| [`dataset_summary.json`](dataset_summary.json) | **Generated** | 資料掃描的機器可讀摘要，包括影片格式、數量與完全重複群組。 |

## Experiment A1／A2：Modified C3D-style

| 文件 | 狀態 | 用途 |
| --- | --- | --- |
| [`baseline_protocol.md`](baseline_protocol.md) | **Frozen** | Experiment A1 的研究問題、資料切分、取樣、模型、訓練與評估規格。 |
| [`baseline_results.md`](baseline_results.md) | **Frozen** | Experiment A1 的 gate、訓練與一次性 test 結果。 |
| [`c3d_style_extended_protocol.md`](c3d_style_extended_protocol.md) | **Frozen** | Experiment A2 的 warm-start、延長訓練、early stopping、resume 與重現資訊。 |

## Experiment B：Reference C3D adaptation

建議依序閱讀規格、實作，再閱讀正式訓練 protocol。

| 文件 | 狀態 | 用途 |
| --- | --- | --- |
| [`c3d_reproduction_spec.md`](c3d_reproduction_spec.md) | **Current** | 區分原論文、作者官方實作、未決細節及 Basketball-51 adaptation；也比較目前 modified baseline。 |
| [`c3d_reference_implementation.md`](c3d_reference_implementation.md) | **Frozen** | 實際 Reference C3D adaptation、clip 索引、tensor shape、PROJECT_ADAPTATION 與 gates。 |
| [`c3d_training_protocol.md`](c3d_training_protocol.md) | **Frozen** | Experiment B 的初始化、fine-tuning、影片級 objective、記憶體／吞吐 gate 與訓練決策。 |

## Experiment C：Modified C3D-style + continuous clips

| 文件 | 狀態 | 用途 |
| --- | --- | --- |
| [`c3d_style_continuous_protocol.md`](c3d_style_continuous_protocol.md) | **Frozen** | Experiment C 的 continuous clip pipeline、影片級聚合、訓練流程、檢查與 run-directory lifecycle。 |

## Experiment D：I3D 前置研究

Experiment D 尚未完成正式 protocol、實作或訓練。以下文件是 protocol 凍結前的證據與可行性稽核，不應解讀為已決定的訓練設定。

| 文件 | 狀態 | 用途 |
| --- | --- | --- |
| [`i3d_temporal_feasibility_audit.md`](i3d_temporal_feasibility_audit.md) | **Current** | Gate 1A：只使用 train／validation，稽核 FPS、frame count、duration、短片比例及 looping exposure。 |
| [`i3d_temporal_feasibility_audit.json`](i3d_temporal_feasibility_audit.json) | **Generated** | Gate 1A 的完整機器可讀統計；一般閱讀以同名 Markdown 摘要為主。 |
| [`i3d_pretrained_artifact_provenance_audit.md`](i3d_pretrained_artifact_provenance_audit.md) | **Current** | Gate 1B：候選 pretrained artifacts 的來源、可用性、license、介面與風險稽核。 |
| [`i3d_pretrained_artifact_provenance_audit.json`](i3d_pretrained_artifact_provenance_audit.json) | **Generated** | Gate 1B 的完整候選與 provenance 紀錄；一般閱讀以同名 Markdown 摘要為主。 |

## Source of truth

- 專案現況與跨實驗比較：[`research_baseline_v1.3.md`](research_baseline_v1.3.md)。
- 各實驗的模型、訓練與限制：對應的 experiment protocol／implementation 文件。
- A2／B／C 正式 test 規則：[`formal_test_evaluation_protocol.md`](formal_test_evaluation_protocol.md)。
- 資料清理與 split 的實際輸入：根目錄 README 所列的 manifest、split summary 與 run snapshots；本目錄文件是其說明，不取代凍結產物。
- I3D 稽核的完整數值：同名 JSON；Markdown 提供可讀摘要與結論。

## 維護原則

- 新增文件時，同步加入本目錄並標示狀態與用途。
- 不在本頁複製完整結果表或 protocol，避免與來源文件產生版本分歧。
- 自動產生的大型 JSON 應保留對應的人類可讀 Markdown 摘要。
- 歷史決策保留在 `progress_log.md`；目前結論寫入對應的 current／frozen 文件。
- 正式 test 產物與 frozen protocol 不因觀察 test 表現而回頭修改。
