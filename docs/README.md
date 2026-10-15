# 研究文件索引

> - **給誰看：** 要尋找研究方向、baseline、資料品質、I3D 或歷史文件的組員。
> - **回答什麼：** 每類研究文件的權威入口與狀態。
> - **不包含：** 完整研究內容、執行命令或重複的結果表。
> - **上一層入口：** [專案 README](../README.md)。
> - **建議下一份：** [目前研究方向](current_research_direction.md)。

本頁只負責分類導航。查目前排程時以[目前研究方向](current_research_direction.md)為準；查研究數值與結論時使用對應的 frozen report、method 或 protocol；查歷史變更時使用 [progress log](progress_log.md)。

## 目前狀態與 repository 治理

| 文件 | 用途 |
| --- | --- |
| [目前研究方向](current_research_direction.md) | 唯一跨階段 current 狀態、優先順序、暫停與範圍外項目 |
| [目前待討論事項](current_discussion_items.md) | 下一輪研究與治理決策清單；保留 2026-10-12 的討論語境 |
| [Repository map](repository_map.md) | 目錄責任、原位保留理由與主要入口 |
| [Development 規則](../development/README.md) | 新研究元件的放置與紀錄要求 |
| [整理紀錄](reorganization_log.md) | 先前目錄與連結整理的歷史查核紀錄 |

## Research Baseline 與正式測試

| 文件 | 狀態與用途 |
| --- | --- |
| [Research Baseline v1.3](research_baseline_v1.3.md) | **Frozen**；A1/A2/B/C 結果、限制與研究脈絡的權威統整 |
| [Baseline 入口](../baseline/README.md) | A1/A2/B/C 定義、程式、設定與歷史產物導航 |
| [正式 test protocol](formal_test_evaluation_protocol.md) | **Frozen**；A2/B/C 一次性正式 test 規則與產物 |
| [資料集分析](dataset_analysis.md) | 資料結構、標籤、重複影片、split 風險與 pre-release 缺口 |
| [參考文獻](references.md) | Basketball-51、C3D、I3D 與題目文件來源 |

## A1/A2/B/C 專屬文件

| 實驗 | 文件 |
| --- | --- |
| A1 | [Baseline protocol](baseline_protocol.md)、[Baseline results](baseline_results.md) |
| A2 | [C3D-style extended protocol](c3d_style_extended_protocol.md) |
| B | [C3D reproduction spec](c3d_reproduction_spec.md)、[reference implementation](c3d_reference_implementation.md)、[training protocol](c3d_training_protocol.md) |
| C | [C3D-style continuous protocol](c3d_style_continuous_protocol.md) |

## 資料品質研究

- [資料品質研究中文索引](data_quality/README.md)：一般組員從此理解 v1、v2、v3/v3.1、Sampler v1 與 Visual Preprocessing Pilot。
- [資料品質技術入口](../development/data_quality/README.md)：實作者查 configs、scripts、samplers、transforms、tests、sidecars、CLI 與 runbooks。
- 各版 method、pilot report、smoke report 與結構化 evidence 由上述兩個入口導覽；本頁不重複完整清單。

## I3D 前置文件

Gate 1A／1B 已完成，但 Gate 1C 與原 Experiment D 仍 paused；這些稽核不代表已選定 I3D。

| 文件 | 用途 |
| --- | --- |
| [Temporal feasibility audit](i3d_temporal_feasibility_audit.md) | Gate 1A 的時間可行性摘要；完整統計在同名 JSON |
| [Pretrained artifact provenance audit](i3d_pretrained_artifact_provenance_audit.md) | Gate 1B 的來源、license、介面與風險摘要；完整候選在同名 JSON |

## 歷史紀錄

| 文件 | 用途 |
| --- | --- |
| [Progress log](progress_log.md) | 按時間保留研究執行與決策；不是目前狀態入口 |
| [Reorganization log](reorganization_log.md) | 目錄移動、相容路徑與先前驗證紀錄 |

## 維護原則

- 同一資訊只保留一個權威來源；其他 README 放摘要與連結。
- 新研究文件加入對應分類並標示 current、frozen、historical 或 generated。
- 技術命令放 `development/` 技術入口或 runbook，研究結果放 method／report，不新增 `README_v3.md`、`README_v4.md`。
- 歷史決策保留在 `progress_log.md`；目前狀態只更新 `current_research_direction.md`。
- 正式 test 產物與 frozen protocol 不因觀察 test 表現而回頭修改。
