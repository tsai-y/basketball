# 新研究工作區

本階段仍是 **full-video Make/Miss classification**。優先工作為 [自動品質篩檢](data_quality/README.md) 與影像增強；再研究特徵擷取、多模型互補／融合。影像增強策略、模型組合與融合方式尚未定案。本輪只建立目錄和文件，沒有新演算法、模型或訓練入口。

## 放置規則

- 新研究放在 `development/<研究主題>/`，每個主題先有 README 記錄問題、範圍、輸入與輸出；目前只建立 `data_quality/`，其他目錄等工作定案再建。
- 不修改 [baseline](../baseline/README.md)，也不修改原位保留的 `reference_c3d/`、`experiment_c/`、`configs/` 及 frozen 執行入口。不能藉由修改其共用函式改變既有基準行為。
- 直接引用唯一的 [固定 manifest](../data/processed/dataset_manifest.csv)、[split](../data/processed/split_summary.json) 與 [原始資料](../archive/Basketball_51%20dataset/)，不再複製一套。需要品質註記時以 video ID 關聯衍生表，不覆寫原 manifest／split，不更改 split 指派。
- 確認已有共用功能時直接引用；不要複製 decoder、sampler 或其他共用程式。需要不同前處理行為時另立具名、可追溯的研究版本。
- 新輸出不得寫入 `baseline/` 或任何歷史 `runs/`。預計採 `development/outputs/<實驗識別碼>/` 等獨立路徑，建立前先在實驗紀錄中指定；此路徑目前尚未建立，也未設為 ignored，未來加入大型產物前應先審查版本管理方式。

## 每個實驗的必要紀錄

| 項目 | 必須記錄 |
| --- | --- |
| 問題與狀態 | 研究目的、負責人、日期、planned／running／completed 等狀態 |
| 資料 manifest | 原 manifest 路徑及版本／hash；衍生品質表或候選清單的路徑與版本 |
| Split | 固定 split 路徑及版本／hash、使用的 split 範圍；不重抽 split |
| 前處理 | 版本、具體步驟、程式來源；若有影像增強，另記其策略與參數 |
| 設定 | 設定檔、seed、環境與程式版本、實際執行命令及 override |
| 輸出 | 唯一輸出目錄、scores／report／模型等檔案位置與產生狀態 |

品質政策、門檻及模型設計只能依可用 train／validation 研究證據制定，test 保持封存。候選清單不是已核准的排除結果。Gate 1C 與原 Experiment D paused，不因建立此工作區自動恢復；Pre-release 與 SPL 資料延伸不在目前實作範圍。完整決策見 [目前研究方向](../docs/current_research_direction.md)。
