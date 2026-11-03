# TANet 固定設定三 seed 對照

2026-10-29 15:22（台灣時間）已依使用者「執行下一步」啟動另外兩個 seed。

- 既有 TANet：seed 20261025，正式訓練已完成；沿用最低 validation CE 所選 epoch 10。
- 新增順序：seed 20261024 → seed 20261026，GPU 依序執行。
- MixNet-I3D＋Transformer 主線三個相同 seed 已完成，這次不重跑。
- 每組仍最多 15 輪，patience 5、min_delta 1e-4，以最低完整 validation CE 選模。
- train 7193、validation 1547；不執行 test、不改門檻、標籤、資料切分或既有權重。

## 本次佇列與入口

[固定計畫](artifacts/multiseed_c0qf_w7l/plan.json) · [佇列狀態](artifacts/multiseed_c0qf_w7l/state.json) · [啟動程序身分](artifacts/multiseed_c0qf_w7l/launcher.json) · [啟動檢查](artifacts/multiseed_c0qf_w7l/prelaunch_checks.json)

目前 seed 20261024 的 [正式訓練日誌](../tanet_seed20261024_v1/artifacts/formal_train_3auu0x0k/progress.log)、[run 狀態](../tanet_seed20261024_v1/artifacts/formal_train_3auu0x0k/run_state.json)；[GPU 預檢](../tanet_seed20261024_v1/artifacts/gpu_readiness_3uo_1oc6/report.json) 已通過。第一輪完成以前尚無新 validation 成績。這段是啟動時紀錄，最新進度請讀日誌。

各階段 stdout 在佇列資料夾內 `seed<seed>_readiness.log`／`seed<seed>_training.log`；其中 `OUTPUT:` 指向該階段真正的 run。第二個 seed 及後續 run 路徑由 state.json 的 jobs 記錄，不預先捏造。

`repeat_seeds_v1.py` 是佇列入口。原本訓練程式和預檢報告綁定來源／設定雜湊，故未直接改原 seed 或放寬驗證，而是建立兩個同層獨立套件：

- `development/tanet_seed20261024_v1/`
- `development/tanet_seed20261026_v1/`

每套 15 個程式／設定／上游授權檔中，程式與作者 TAM 全部逐檔位元一致；只有 config.json、train_config.json 的 seed 欄位改變。各套件 REPEAT_PROVENANCE.json 記錄原契約、重跑契約及逐檔 SHA256。相同 ImageNet 權重、全新 TAM 與分類頭，不從既有 TANet best.pt、last.pt 或工程權重開始。

每個 seed 各做一次新的完整 GPU 工程預檢，通過後才正式啟動。預檢包含三次工程更新和精確續跑驗證，這些小樣本結果不列入效能成果。正式訓練引擎保持不變；每輪完整 train 與 validation，仍有 checkpoint RNG／Adam／早停狀態。

## 完成後的產物與限制

兩組訓練完成後，佇列會讀取三個 TANet best.pt 的既有 validation logits，按 ID 與各自相同 seed 的主線 validation_predictions.csv 對齊。自動產生本次資料夾內的 summary.json、index.html 與三份 paired_seed<seed>.csv，涵蓋 accuracy、macro-F1、Miss／Make recall、各 seed 最佳 epoch 和配對錯誤數。此彙整不做模型 forward。

報告所有三個 seed 的平均與**樣本標準差**，不挑最高分 seed。跨 seed 標準差不是對影片／來源的信賴區間；三次訓練也未消除兩系統輸入與訓練設定不同的限制。validation 曾參與選模，不能稱為 test 成果。

任一階段失敗，佇列停止並保存狀態，不自動重試、續訓或增加輪數。不要再次啟動同一佇列；要先核對程序是否存活。斷電後只能在確認程序終止後，由明確續訓指示處理完整 epoch 邊界，不能宣稱中途 microstep 已保存。

## 驗證與監控

啟動前 35 項測試通過，包括原 28 項以及新增的 seed 隔離、設定差異限制、既有主線設定一致、失敗停止／禁止重啟、完整 1547 筆 ID 對齊與三 seed 彙整測試。兩套正式入口 `--dry-run` 均通過，零影片開啟、零 CUDA context、零 optimizer 更新。原模型與分析所保護的檔案 SHA256 均維持一致。

背景 supervisor 與各子程序的 PID、process_start_ticks、boot_id、命令列保存在 launcher.json／state.json。讀日誌時間不足以認定卡住，須交叉核對程序及 validation 階段。沿用既有 `tanet` 每小時 heartbeat 排程，已更新為追蹤本次佇列；兩個新 seed 和彙整皆完成，或確認失敗後，回報一次並暫停排程。
