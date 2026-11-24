# ACA 單 seed 消融正式執行計畫

後續更新：2026-11-19 LSTA已完成，使用者已指示啟動TSCI一組，每小時監看ACTIVE；見[TSCI啟動紀錄](TSCI_FORMAL_LAUNCH.md)。下文保留準備時快照。

日期：2026-11-18。狀態：**計畫與入口準備完成，正式訓練未啟動**。本輪接續只準備單seed執行，不新增架構或再次跑真實GPU預檢。這是明列假設的ACA重建，尚非作者精確複刻。

## 執行順序與沿用決定

原八類baseline可沿用：重新建立的共同計畫與原run計畫逐欄完全相同，canonical SHA仍為 `7934894ec6d8252b1b03ebd918ee2cc973e92676af3a66c7e0a8f4f7fca39318`。包含完整順序清單、來源與tensor驗收參考、資料inventory、原數值程式、初始化、GPU及軟體runtime。12輪history、5400更新、最低CE選第7輪與summary一致。配對初始化／baseline算子及BN相容證據見既有工程報告。新TSCI BN僅存在新增模組，不影響baseline。

| 順序 | 組別 | seed | 處理 |
| --- | --- | --- | --- |
| 0 | baseline | 20261024 | 沿用已完成run，禁止由新入口重跑 |
| 1 | LSTA | 20261024 | 下一個啟動對象；從固定ImageNet V1＋fresh head初始化 |
| 2 | TSCI | 20261024 | LSTA完成後明確啟動；不沿用LSTA權重 |
| 3 | ACA | 20261024 | TSCI完成後明確啟動；不沿用前組權重 |

入口會核對前一組完整history與最低CE選模結果。沒有自動佇列；結果有無增益都保留，不依中途分數改架構、挑seed或延長輪數。先完成這三個缺少的run，再彙整同seed四組；另外兩個seed及SpaceJam不在本次啟動範圍。

## 固定共同協定

- 輸入：完整片段均勻50幀RGB、160×120、ImageNet正規化；不增強、不裁切、不改內容旗標或標籤。
- train7192／validation1547，原source-group切分與八類順序不變；不開test。
- seed20261024；backbone／head配對初始化，新模組使用隔離RNG；禁止載入舊任務或工程預檢權重。
- Adam lr1e-4，micro1／effective16，尾批8，每輪450次更新；FP32、AMP／TF32關閉；clip norm1。
- 骨幹BN affine與buffers凍結；TSCI BN只在train更新，validation使用已存統計；LSTA IN沿用固定定義。
- 每組最多30輪、patience5；最低完整八類validation CE選模，同分保留較早輪。
- 八類accuracy、macro-F1、每類recall與混淆矩陣為主要報告；八類argmax映射Make／Miss僅作診斷，不當另一個二類模型成績。

## 已準備的計畫

| 組別 | 計畫 | canonical SHA256 |
| --- | --- | --- |
| LSTA | [plan.json](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_lsta_plan_seed20261024_v1/plan.json) | `076399e5668f6df57124e84de9027a2f8c8cbf7ad8a044c9f4e060530bad2e12` |
| TSCI | [plan.json](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_tsci_plan_seed20261024_v1/plan.json) | `210a7ab1d1c05794b2fdd32c76f824735afe268f5f816e94ff308d9c97049424` |
| ACA | [plan.json](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_aca_plan_seed20261024_v1/plan.json) | `94d447a702270b2be9aabac9813164bb45d8ff2d899d26e5130fafe1d8c23913` |

三份plan已通過重新建立一致性檢查；prepare只檢查metadata、來源fstat及宿主runtime，模型建立／影片解碼／optimizer更新／test存取均為0。完整source hash包含新入口與凍結相依程式；修改程式或資料後會拒絕舊plan啟動，必須重新說明與準備，不可繞過。

## 正式入口與恢復

程式為 [formal_train.py](../../development/aca_reconstruction_v2/formal_train.py)；數值迴圈直接引用已通過GPU預檢的EngineeringLoop._run，正式checkpoint使用不同schema。正式adapter的六項CPU測試通過，涵蓋完整狀態恢復、工程／其他run權重拒絕、固定範圍、plan變更阻擋、先後順序及baseline選模一致性。這些測試使用微小合成模型，沒有再次訓練真實影片。

正式執行前仍會核對GPU占用與剩餘資源；不終止其他程序，不降解析度或改精度。既有ACA小型GPU預檢約31秒，峰值allocated3.69 GiB／reserved3.94 GiB；不能直接保證全程速度或峰值，正式首輪日誌才用於估算完成時間。其他組目前具有CPU／meta證據，真實GPU證據僅ACA。

啟動LSTA的精確命令如下，僅供後續明確啟動時使用；本次沒有執行：

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
env CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  /home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_reconstruction_v2.formal_train start \
  --plan /home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_lsta_plan_seed20261024_v1/plan.json \
  --plan-sha256 076399e5668f6df57124e84de9027a2f8c8cbf7ad8a044c9f4e060530bad2e12 \
  --run-dir /home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_lsta_seed20261024_v1
```

- start只接受該arm固定新目錄；已存在則拒絕覆寫。GPU共用鎖與run鎖防止重複寫入。
- 每25次更新與train輪末原子保存last.pt，包含model／Adam／CPU及CUDA RNG／history／best_model。
- stop只寫自身STOP.json；程序於optimizer邊界保存並停下。validation中斷不提交半輪成績。
- resume必須明確執行，核對plan SHA、run_id、checkpoint schema、狀態及Adam；重播未保存的train部分或整輪未完成validation。
- 沒有checkpoint時停止並回報，不猜測重開。失敗保留failure.json與最後有效checkpoint，不自動重試。
- launcher記錄PID／process_start_ticks／boot_id／完整命令；progress.log、run_state.json、history.json、各attempt與summary供監看。已完成組拒絕resume。

## 完成條件與下一步

本階段完成條件已達成：baseline沿用已核定、三份plan可重建、新入口與恢復邊界驗證完成。下一步是啟動**LSTA seed20261024一組**；完成後再照同協定進入TSCI與ACA。當前無正式訓練程序或新排程，原每小時監看仍暫停。

[準備核對紀錄](results/aca_reconstruction_formal_preparation_v2.json) · [工程驗收](ACA_ENGINEERING_STATUS.md) · [研究總入口](ACA_CURRENT_STATUS.md)
