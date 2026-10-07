# 目前研究方向

更新日期：2026-10-07。本頁記錄目前工作安排；歷史報告保留當時結論，不回寫既有研究證據。

## 研究範圍與順序

本階段為 **full-video Make/Miss classification**。完整影片可能包含球離手後的球路、碰框、進球與反應；本階段成果不能視為 pre-release prediction 能力。

1. 優先研究**自動品質篩檢與影像增強**。原始資料保持不變，品質分數及候選排除名單另存；品質方法與門檻尚未決定。影像增強的具體策略仍待確認。
2. 後續研究**特徵擷取與多模型互補／融合**。尚未決定模型組合及融合方式，不能將既有 I3D 稽核當成模型選型。
3. **SPL** 指 **Sport Performance Lab 的 SPL Open Data**，是最後才考慮的資料集延伸，**不是模型模組**；本階段沒有啟動 SPL 整合。

本輪只有 repository 結構、必要路徑與團隊文件，尚未開始資料清洗、影像增強、模型實作或訓練。新工作依 [development 規則](../development/README.md) 放置。

## 完成、凍結與暫停狀態

| 項目 | 狀態 | 意義與依據 |
| --- | --- | --- |
| A1/A2/B/C | **Frozen** | full-video 基準已完成；[基準索引](../baseline/README.md)、[Research Baseline v1.3](research_baseline_v1.3.md) |
| Gate 1A | **Completed** | [時間可行性報告](i3d_temporal_feasibility_audit.md)／[JSON](i3d_temporal_feasibility_audit.json)，保留原位原文 |
| Gate 1B | **Completed** | [Pretrained artifact provenance 報告](i3d_pretrained_artifact_provenance_audit.md)／[JSON](i3d_pretrained_artifact_provenance_audit.json)，完成來源稽核不代表數值等價或選型完成 |
| Gate 1C | **Paused** | 不啟動權重下載、轉換或相容性實測 |
| 原 Experiment D | **Paused** | 原 I3D 工作暫停；本次沒有新增 D protocol、模型實作或訓練 |
| Pre-release | **Out of scope** | 不納入目前範圍；既有文件中的長期構想不是當前任務 |

## 證據與使用邊界

A1/A2/B/C 的模型、sampling／preprocessing、超參數、seed、split、指標計算、checkpoint 選擇及產物維持 frozen。固定 manifest／split 是共用依據，不依 test 表現回頭調整；test 影片保持封存。

[Research Baseline](research_baseline_v1.3.md) 與 Gate 1A／1B 原文、JSON 及路徑完全保留。它們對 D、pre-release 或 SPL 的歷史敘述應在當時脈絡閱讀；**目前範圍與排程以本頁為準，歷史結果與稽核事實仍以原報告與原產物為準**。本頁不重新解釋或改算其數值。

新實驗應記錄 manifest、split、前處理版本、設定及獨立輸出位置。組員找檔案時使用 [repository map](repository_map.md)，追溯整理前位置時使用 [reorganization log](reorganization_log.md)。
