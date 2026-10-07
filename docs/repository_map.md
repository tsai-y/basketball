# Repository map

更新日期：2026-10-07。路徑均相對於 repository root；**檔案歸屬依用途與實際引用判定，不依目錄名稱猜測**。先讀 [root README](../README.md) 與 [目前研究方向](current_research_direction.md)。

## 目錄、用途與組員使用方式

| 目錄／檔案 | 用途 | 狀態 | 組員使用方式 |
| --- | --- | --- | --- |
| [README.md](../README.md) | 團隊入口、root 指令與現況 | Current | 新組員第一份文件；取代舊 root `readme.md` |
| [readme.md](../readme.md) | 指向 `README.md` 的相容 symlink | Alias | 保留 Gate 1B 的小寫路徑連結，內容只有一份；跨平台限制見整理紀錄 |
| [baseline/](../baseline/README.md) | A1/A2 資料與模型；A1/A2/B/C 統一索引 | Frozen | 查基準程式與產物，不開發新方法 |
| [baseline/scripts/](../baseline/scripts/) | B 歷史 shape audit、benchmark | Frozen | 僅供追溯；新位置可用 `--help`，正常模式仍會寫歷史 feasibility |
| [reference_c3d/](../reference_c3d/) | B 的 model／aggregate；B/C 共用 continuous clip data | Frozen、原位 | 被訓練／評估入口及 C 使用，維持單一來源，不複製或改寫 |
| [experiment_c/](../experiment_c/) | C 的模型及 GAP 特徵聚合 | Frozen、原位 | 實際匯入 `baseline.model` 與自身 aggregate；不是新研究目錄 |
| [configs/](../configs/) | 六份 A1/A2/B/C 與 B gate／benchmark 設定 | Frozen、原位 | 含固定 hash、原始 epoch／patience 及 root-relative 路徑；不以最終訓練狀態回寫 |
| [scripts/](../scripts/) | 訓練／評估入口、共用資料工具及 Gate 1A | 混合、原位 | 依下表區分責任；不是任意新增研究的預設位置 |
| [development/](../development/README.md) | 新方法的研究工作區 | Planned | 新主題放此處，記錄實驗輸入、版本、設定與獨立輸出位置 |
| [development/data_quality/](../development/data_quality/README.md) | 自動品質篩檢 | Planned、僅 README | 先規劃品質分數、候選排除清單、報告；尚無演算法或門檻 |
| [archive/Basketball_51 dataset/](../archive/Basketball_51%20dataset/) | 原始 RGB 影片 | Protected、ignored | 原位原內容保留；本輪未開啟影片，沒有掃描／probe／解碼 |
| [data/processed/](../data/processed/) | `dataset_manifest.csv`、`split_summary.json` | Fixed、共用 | 唯一主輸入；影片路徑相對於資料根目錄，不另複製或重抽 split |
| [runs/](../runs/) | config／manifest／split 快照、checkpoint、training logs、metrics／predictions、gates | Historical、protected | 唯讀追溯；tracked／ignored 產物皆保留，不作為新研究輸出位置 |
| [docs/](README.md) | 研究方向、protocol、結果、歷史及稽核 | 依文件標記 | 目前排程讀 current_research_direction；歷史證據讀原文 |
| [papers/](../papers/)、[TOPIC 1.pdf](../TOPIC%201.pdf) | 本機論文與研究背景 | 原位、ignored | 按 [references](references.md) 使用；未因本輪下載或改動 |
| [.gitignore](../.gitignore) | 既有資料、checkpoint、PDF、cache 排除規則 | 原位、未改 | ignored 不代表可刪除；新產物尚未新增忽略規則 |

## `scripts/` 責任分類

| 路徑 | 歸屬及理由 | 現在如何使用 |
| --- | --- | --- |
| [run_baseline.py](../scripts/run_baseline.py) | Frozen A1；本身加入 protocol hash | `--help`；結果讀 A1 run |
| [run_c3d_style_extended.py](../scripts/run_c3d_style_extended.py) | Frozen A2；本身加入 protocol hash，檢查 model/data hash | `--help`；不 resume 或重跑 |
| [run_c3d_reference_train.py](../scripts/run_c3d_reference_train.py) | Frozen B；固定 source/config hash、legacy protocol 與 resume 相容分支 | `--help`；不修改相容機制 |
| [run_c3d_reference_checks.py](../scripts/run_c3d_reference_checks.py) | Frozen B gates；含來源 fingerprint 與歷史 run 路徑 | `--help`；既有 gates 讀 `runs/full_video_c3d_reference_v1/` |
| [run_c3d_style_continuous.py](../scripts/run_c3d_style_continuous.py) | Frozen C；source hash／config hash 與 run lifecycle 檢查 | `--help`；不寫入完成的 C run |
| [evaluate_full_video_experiments.py](../scripts/evaluate_full_video_experiments.py) | Frozen A2/B/C evaluator；匯入 B 的固定 hash 表 | `--help`；其他模式會載入 checkpoint，正式 test 已完成 |
| [inspect_dataset.py](../scripts/inspect_dataset.py) | 共用資料檢查；其 LABELS／exact_duplicate_groups 被 build_manifest 使用 | 只查看說明或既有摘要；正常執行會遍歷並 probe 影片 |
| [build_manifest.py](../scripts/build_manifest.py) | 共用 manifest／split 建置，直接匯入同目錄 inspect_dataset | `--help`；固定輸入已存在，不重新建置 |
| [audit_i3d_temporal_feasibility.py](../scripts/audit_i3d_temporal_feasibility.py) | Gate 1A 歷史證據工具；報告記錄其腳本 hash 與路徑 | `--help` 或 `--mode check` 只讀 metadata；不重跑 audit |

資料工具目前由既有基準資料準備流程及後續研究共同參考；不將 baseline 專用訓練入口誤稱通用工具。未發現 repository 中的 shell scripts 或額外專案設定檔（除六份 YAML 與 `.gitignore`）。

## 歷史 run 對照

| 原位目錄 | 內容與用途 |
| --- | --- |
| [full_video_c3d_style_v1](../runs/full_video_c3d_style_v1/) | A1 smoke／overfit、五 epoch training、best checkpoint、一次 test |
| [full_video_c3d_style_extended_v1](../runs/full_video_c3d_style_extended_v1/) | A2 extended training、best／last、logs、正式 metrics／predictions |
| [full_video_c3d_reference_v1](../runs/full_video_c3d_reference_v1/) | B clip gates、失敗及修正後小樣本診斷；`feasibility/` 保存 shape／VRAM／吞吐等歷史證據，非正式 test |
| [full_video_c3d_reference_train_v1](../runs/full_video_c3d_reference_train_v1/) | B 正式 training、best／last、logs、正式 metrics／predictions |
| [full_video_c3d_style_continuous_v1](../runs/full_video_c3d_style_continuous_v1/) | C 正式 training、best／last、logs、正式 metrics／predictions；本機 manifest 快照為 ignored |

原位還包括各 run 的 `.test_evaluation_claim`、C 的 `.training_started` 與所有既有環境／設定／split 快照。這些狀態檔不可因「看似暫存」而刪除。`.gitignore` 不影響已 tracked 的 manifests；clone 仍不含 ignored 的影片／checkpoint／PDF／C manifest 快照，取得方式待團隊確認。

## 研究證據入口

- [Research Baseline v1.3](research_baseline_v1.3.md)：已完成結果與研究限制，原文原位 frozen。
- [Gate 1A Markdown](i3d_temporal_feasibility_audit.md)／[JSON](i3d_temporal_feasibility_audit.json)：completed，train／validation 時間 metadata 稽核。
- [Gate 1B Markdown](i3d_pretrained_artifact_provenance_audit.md)／[JSON](i3d_pretrained_artifact_provenance_audit.json)：completed，預訓練候選來源及相容性未決事項。
- [dataset_analysis](dataset_analysis.md)、[dataset_summary](dataset_summary.json)：既有資料證據；不在本輪重新產生。
- [文件索引](README.md)：其餘 A1/A2/B/C protocols、結果及文獻。
- [progress log](progress_log.md)：既有內容完整保留，尾端追加本次整理；頂部舊 Current Status 是 2026-10-01 的歷史快照。

歷史文中的舊路徑及當時規劃不批次改寫；搬移對照與未處理事項見 [reorganization log](reorganization_log.md)。
