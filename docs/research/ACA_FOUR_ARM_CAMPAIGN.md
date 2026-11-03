# ACA 四組 × 三 seed 正式訓練（2026-11-03）

已依使用者「下一步」指示啟動十二組全新正式訓練。這是共同協定下的專案消融比較；尚未核實為作者精確復刻。既有 ACA／TANet 結果與權重不作為這十二組的初始化或對照成績。

啟動前重新核對十二組 GPU readiness、CPU plan、資料與來源雜湊、環境及 ImageNet 權重；共用 trainer／protocol 未更改。獨立監督器的九項 CPU 檢查通過。新增監督器首次啟動在正式 run 建立前，因 JSON list/tuple 直接比對誤判而被攔下；改為與預檢相同的 canonical JSON SHA 後才開始本次訓練，並未產生額外正式更新或重跑。

## 已固定的執行方式

- 順序：seed20261024、20261025、20261026；各 seed 依 baseline → LSTA → TSCI → dense ACA，一次一組，共十二組。
- 每組從固定 ImageNet V1 與配對 head／新增元件全新初始化。50幀、120×160、全圖 RGB，train7193／validation1547。
- max30輪、patience5、Adam1e-4、micro1／effective16、尾批9，全部 BN 凍結。只按最低完整 validation CE 選模，平手保留較早輪。
- 單組早停後繼續下一組；非零退出、缺少完整 summary 或驗證失敗則停止佇列。沒有自動重跑／續訓／延長，也不執行 test。

## 執行紀錄

- 本次 campaign：`development/aca_four_arm_campaign_v1/artifacts/campaign_g14srnja/`。
- 活動狀態：上述目錄的 `state.json`；完整啟動指紋為 `plan.json`／`launcher.json`，監督日誌為 `supervisor.log`。
- 第一組正式 run：`development/aca_four_arm_v1/artifacts/formal_baseline_seed20261024_mfeniv_1/`。
- 第一組 stdout：campaign 目錄的 `baseline_seed20261024.log`；之後每組都有獨立 log，唯一 `OUTPUT:` 行指向正式 run。
- [啟動時觀測快照與指紋](results/aca_four_arm_launch_v1.json)。快照不是即時進度，當下首輪尚無完整 validation 成績。

監督器在獨立 session 執行，關閉啟動終端不會自動中止；主機關機仍會中斷。檢查存活時必須同時核對 PID、process_start_ticks、boot_id 與完整命令列。`state.json` 每十秒更新，但訓練進度取最後已記錄 checkpoint（每25次更新），不是即時精確影片數；validation 中途沒有分數，不以日誌時間單獨判定卡住。

十二組完成後才產生 campaign `summary.json`，保存每組所選 validation、各 arm 三 seed 平均／樣本標準差與 Miss／Make recall。結果屬 validation；專案已有歷史 test 接觸，本輪不開啟 test、不下載 SpaceJam。下一個研究決策應等這個固定比較完成。

監督器與操作說明：[獨立佇列](../../development/aca_four_arm_campaign_v1/README.md)。本次程式與文件保存在本機研究分支，尚未推送 GitHub；大型權重與完整執行產物維持 ignored。
