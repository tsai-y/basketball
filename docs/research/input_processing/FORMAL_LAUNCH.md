# 單一八類 baseline：正式啟動與一般續訓入口

更新：2026-11-18。**正式run已於2026-11-17 16:54台灣時間正常完成：12輪、5400次更新、早停5／5，最低CE選第7輪（CE1.1907208，八類accuracy59.6639%，macro-F1 0.4425495）。** 每小時排程已PAUSED；完成run不續訓、不延長。

[正式結果](/home/tsai/projects/basketball/outputs/01_current/baseline8_formal_seed20261024_v1/summary.json) · [最新研究狀態](../ACA_CURRENT_STATUS.md)

以下為2026-11-17「準備入口」時的歷史快照；其中「本輪未啟動」描述準備階段，不代表目前狀態。原plan與驗收JSON保留當時內容，不回寫改造歷史。

---

日期：2026-11-17。**入口已實作，7項新增工程檢查通過，完整執行計畫已準備；正式訓練尚未開始。下一步是依這份固定計畫啟動唯一的baseline。**

目前不再增加模型、裁切候選或多seed佇列。已完成資料驗收、八類介面、CPU初始化／迴圈、有限真實影片GPU接線，以及本輪正式啟動／恢復介面。

## 已準備的唯一計畫

| 項目 | 固定值 |
| --- | --- |
| 模型 | ResNet50 baseline＋八類頭，固定ImageNet V1全新初始化 |
| 輸入 | 50幀RGB、完整片段均勻取樣、160×120、ImageNet正規化，無裁切或增強 |
| 資料 | train7192、validation1547；不讀test |
| seed | 20261024，只有一組 |
| 優化 | Adam lr1e-4、micro1／effective16、尾批8、每輪450次更新、梯度裁剪1 |
| 停止 | 最多30輪，patience5；只以最低完整八類validation CE選模，同分保留較早輪 |
| BN／數值 | BN affine及統計凍結；CUDA FP32、AMP／TF32關閉、deterministic |
| 身分 | 綁定程式、規格、模型初始化權重、8739支有序清單、來源檔案狀態、全量既有audit與執行環境 |
| 資源 | 沿用既有GPU桌面例外，未知compute阻擋；至少6144MiB可用顯存及8GiB磁碟空間 |

[完整plan.json](/home/tsai/projects/basketball/outputs/01_current/baseline8_launch_plan_20261117_v1/plan.json) · [準備結果review.json](/home/tsai/projects/basketball/outputs/01_current/baseline8_launch_plan_20261117_v1/review.json)

plan的canonical SHA256為 `7934894ec6d8252b1b03ebd918ee2cc973e92676af3a66c7e0a8f4f7fca39318`。這是內容canonical hash，與檔案排版bytes的SHA概念不同。

本輪prepare核對8739個允許的檔案描述符與stat（來源總計5,489,738,312 bytes），未解碼、未讀影片內容。未來正式loader每次讀片仍會核對原始影片SHA及張量SHA，必須與先前8739支全量驗收一致；沒有為本輪重跑全量解碼。

## 本輪新增實作

- [training_runtime.py](../../../development/aca_eight_class_data_v1/training_runtime.py)：沿用已驗收的`FixtureLoop._run`，接入任意輪次／optimizer邊界的CPU與CUDA RNG、Adam、歷史與最佳權重恢復，以及跨輪次的兩worker資料串流。
- [formal_train.py](../../../development/aca_eight_class_data_v1/formal_train.py)：分開`prepare`、`start`、`resume`、`stop`；沒有test或替換seed／輪數／輸入設定的命令參數。
- [test_formal_train.py](../../../development/aca_eight_class_data_v1/test_formal_train.py)：只用極小合成模型核對新增介面。

同run鎖及共享GPU啟動鎖防止本入口重複寫入或同時啟動兩個run。開始與恢復前重建計畫，若程式、資料、runtime或固定設定與已審閱計畫不符就拒絕。每次start／resume保留獨立attempt目錄及PID、process_start_ticks、boot_id、完整命令列，舊attempt的failure不會被當成新attempt結果。

## 中斷與日誌

`last.pt`是唯一權威checkpoint，內含當前模型、Adam、CPU／CUDA RNG、epoch、cursor、updates、指標、history、best_epoch／CE、bad_epochs及best_model。每25次更新、train輪末、完整validation提交及訓練中要求停止的更新邊界保存。

正常暫停會等正在累積的有效批次完成。validation途中暫停不提交部分成績，恢復時整輪validation從第一筆重做。強制關機後只承認已提交checkpoint，不加總未保存的更新。尚無last.pt的初始化失敗不會被當成可以續訓；完整完成的run也不會被重跑。

`history.json`與`run_state.json`可由checkpoint恢復；即使checkpoint提交後日誌寫入失敗，也不會把舊history當成模型狀態。`run_state.json`區分目前更新數與checkpoint更新數；`epoch_index`從0起算，訓練／validation中展示第幾輪時加1，完成狀態則以completed_epochs為準。

預定正式run目錄為 `/home/tsai/projects/basketball/outputs/01_current/baseline8_formal_seed20261024_v1/`，**本輪未建立或啟動該run**。啟動後主要檔案：

| 檔案 | 用途 |
| --- | --- |
| plan.json、run.json | 固定計畫與唯一run_id |
| launcher.json | 最新程序身分與attempt |
| progress.log | 追加式台灣時間進度；train每25次更新、validation每25片更新 |
| run_state.json | 目前階段、train cursor、validation進度及已保存更新數 |
| history.json | 已完成完整validation的逐輪指標 |
| last.pt | 權威checkpoint，最佳權重位於其best_model欄位 |
| attempt_*/failure.json／outcome.json | 每次執行的錯誤或結束結果 |
| summary.json | 正式完成後的最低CE選模結果；不含test成績 |

## 操作入口（本輪只執行prepare）

下列start／resume命令是後續操作方式，本輪沒有執行。從原ACA工作樹運行；不更改已凍結的規格JSON。

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball

env CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_eight_class_data_v1.formal_train start \
  --plan /home/tsai/projects/basketball/outputs/01_current/baseline8_launch_plan_20261117_v1/plan.json \
  --plan-sha256 7934894ec6d8252b1b03ebd918ee2cc973e92676af3a66c7e0a8f4f7fca39318 \
  --run-dir /home/tsai/projects/basketball/outputs/01_current/baseline8_formal_seed20261024_v1
```

正常暫停只寫入該run自己的STOP marker，不向其他程序發送訊號：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_eight_class_data_v1.formal_train stop \
  --run-dir /home/tsai/projects/basketball/outputs/01_current/baseline8_formal_seed20261024_v1
```

已明確決定續訓時，恢復同一run；其餘規格與30輪上限不變：

```bash
env CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_eight_class_data_v1.formal_train resume \
  --run-dir /home/tsai/projects/basketball/outputs/01_current/baseline8_formal_seed20261024_v1 \
  --plan-sha256 7934894ec6d8252b1b03ebd918ee2cc973e92676af3a66c7e0a8f4f7fca39318
```

以上入口以前景程序執行；需要背景長時間執行時，應由受控啟動程序保存stdout／stderr及PID身分。沒有自動重啟、延長輪數或每小時排程的隱藏行為。

## 本輪驗收與限制

7項檢查全部通過，耗時約7.04秒：CUDA合成模型的validation中斷／第二輪train中斷恢復與連續執行完全一致；錯run、預檢schema、壞history／Adam等在修改現有狀態前拒絕；兩spawn workers跨輪順序正確；STOP與signal只要求安全邊界暫停；checkpoint成功但日誌失敗可恢復；同run鎖拒絕第二個寫入者；完整資料metadata綁定與test拒絕通過。

本輪沒有建立ResNet、沒有真實影片解碼或真實資料參數更新。CUDA檢查僅更新極小合成模型，暫存checkpoint已清除；全部115個受保護檔案SHA一致。先前24＋8支真實影片GPU預檢保持原報告，不重跑、不覆寫。

本輪證據支持正式入口與恢復介面已完成，尚無正式八類validation結果，不能推論90%或作者精確ACA複刻。完整epoch耗時與模型成績須從後續唯一baseline正式run取得，不由小樣本預檢分數外推。原規格中的implementation_pending等文字是定稿快照，後續實作進度由本頁補充。

[機器可讀驗收紀錄](eight_class_formal_launch_qa_v1.json) · [上一輪GPU預檢](GPU_READINESS.md) · [資料驗收](BASELINE_ACCEPTANCE.md) · [研究工作台](../../../workspace/README.md)
