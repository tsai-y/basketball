# 目前研究方向

更新日期：2026-10-13。本頁是跨階段「目前狀態與排程」的唯一權威來源；歷史報告保留當時結論，不回寫既有研究證據。

## 研究範圍

本階段仍是 **full-video Make/Miss classification**。完整影片可能包含球離手後的球路、碰框、進球與反應，因此現有成果不能視為 pre-release prediction 能力。

目前工作的順序是：

1. 凍結第一版正式視覺前處理政策，並定義下一個可比較、與 frozen baseline 分離的受控實驗。
2. 先完成資料管線整合與驗收，再決定 I3D、YOLO 特徵／ROI 或其他特徵路線。
3. 至少有兩個可比較的單模型 validation 證據後，才討論多模型互補／融合。

SPL 指 Sport Performance Lab 的 SPL Open Data，是最後才考慮的資料集延伸，不是模型模組；本階段沒有啟動 SPL 整合。

## 完成、待決與暫停狀態

| 項目 | 狀態 | 權威入口 |
| --- | --- | --- |
| A1/A2/B/C | **Frozen**；full-video baseline 與一次性正式 test 已完成 | [Baseline 入口](../baseline/README.md)、[Research Baseline](research_baseline_v1.3.md) |
| 完整 train 品質掃描 v3/v3.1 | **Completed**；候選不是刪除名單 | [資料品質研究索引](data_quality/README.md) |
| Quality-aware Clip Sampler v1 | **Completed**；train-only sidecar 與 fallback 契約已建立 | [Sampler 方法](data_quality/quality_aware_clip_sampler_v1.md) |
| Visual Preprocessing Pilot v1 | **Pilot completed**；正式政策未定案 | [方法](data_quality/visual_preprocessing_method_v1.md)、[報告](data_quality/visual_preprocessing_pilot_v1.md) |
| 下一個受控實驗 | **待定義** | [目前待討論事項](current_discussion_items.md) |
| Gate 1A／1B | **Completed** | [Gate 1A](i3d_temporal_feasibility_audit.md)、[Gate 1B](i3d_pretrained_artifact_provenance_audit.md) |
| Gate 1C／原 Experiment D | **Paused** | 尚未啟動權重相容性實測、模型實作或訓練 |
| I3D／YOLO／融合 | **Not started** | 不得把既有 Gate 稽核視為模型選型 |
| Pre-release／SPL | **Out of scope** | 歷史文件中的長期構想不是目前排程 |

## 使用邊界

- A1/A2/B/C 的模型、sampling／preprocessing、超參數、seed、split、指標、checkpoint 選擇與正式產物維持 frozen。
- 固定 manifest／split 是共用依據；不得依 test 表現回頭調整。validation/test 影片保持封存。
- 品質候選只供 train 品質研究與 clip 選擇，不代表確認模糊、排除名單或 cleaned dataset。
- 新實驗必須記錄 manifest、split、前處理版本、設定、停止規則及獨立輸出位置；不得寫入歷史 `runs/`。

下一輪需決定的政策、比較組與驗收條件見[目前待討論與決策事項](current_discussion_items.md)。檔案位置見 [repository map](repository_map.md)，歷史變更見 [progress log](progress_log.md)。
