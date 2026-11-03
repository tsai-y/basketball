# 原低解析度三模型融合：隔離單次 Test

使用者已指定：**三模型等權融合為主，單模型只作附表**。不依 test 選模型、調權重或門檻；不訓練、不使用高解析度改版、不修改任何既有程式與 checkpoint。

主模型是已由 validation 凍結的 dense ACA 第 8 輪、DW3D 第 4 輪、MixNet-I3D＋Transformer 第 0 輪。三個 Make 機率各乘 1/3 再相加，機率嚴格大於 0.5 判 Make。主線單模型保留原本 logit >= 0 判 Make 的 tie 規則。

新 test scope 只允許原始固定 test 的全部 1539 影片（Miss 731、Make 808）；沿用原標籤與來源群組隔離。既有 train/validation guard 不變。測試前先用固定四部 validation 比對輸入張量完全相同、模型預測一致及機率容差；重播全部 1547 部既有 validation 融合輸出驗證融合算式。通過才建立持久 exclusive claim 並讀 test。

此專案舊模型曾使用相同 test 切分。因此不能稱為全專案完全未見的最終 holdout；本次只稱「凍結三模型融合的首次 test」。人工 validation 複查不改寫原始標籤，不剔除困難影片。ACA 為候選架構，不是已驗證作者原版，不保證 90%。

命令須在原宿主 GPU 環境，以指定 Python 直接執行檔案（不可用 -m，以保持原主線套件載入順序）。

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B /home/tsai/.codex/worktrees/aca-net-reproduction/basketball/development/aca_test_v1/test_evaluate.py
# 唯讀 protocol 核對，不開 test 影片
/home/tsai/miniconda3/envs/yolo11/bin/python -B /home/tsai/.codex/worktrees/aca-net-reproduction/basketball/development/aca_test_v1/evaluate.py
# 僅使用者授權後執行一次；claim 存在則拒絕，不自動重試或刪除 claim
/home/tsai/miniconda3/envs/yolo11/bin/python -B /home/tsai/.codex/worktrees/aca-net-reproduction/basketball/development/aca_test_v1/evaluate.py --execute-once
```

產物獨立保存在本目錄 artifacts/：事前 protocol、模型候選與評估程式快照、validation 一致性報告、各模型及融合逐影片 JSONL、狀態/完整成績 JSON、繁體中文 HTML 報告。只有完整 cohort、模型狀態與保護檔案 SHA 全部核對成功才產生 completed 報告；中途失敗保留 claim 與 failure.json，不自動續跑。
