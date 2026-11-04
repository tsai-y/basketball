# ACA 四組 × 三 seed 正式訓練（更新：2026-11-04）

已依使用者「下一步」指示啟動十二組全新正式訓練。這是共同協定下的專案消融比較；尚未核實為作者精確復刻。既有 ACA／TANet 結果與權重不作為這十二組的初始化或對照成績。

啟動前重新核對十二組 GPU readiness、CPU plan、資料與來源雜湊、環境及 ImageNet 權重；共用 trainer／protocol 未更改。獨立監督器的九項 CPU 檢查通過。新增監督器首次啟動在正式 run 建立前，因 JSON list/tuple 直接比對誤判而被攔下；改為與預檢相同的 canonical JSON SHA 後才開始本次訓練，並未產生額外正式更新或重跑。

## 2026-11-04 明確授權續接

使用者於本對話明確要求持續執行後，已保留完成的 baseline／LSTA（seed20261024），從尚未開始的 TSCI／seed20261024 接續十組。原佇列在 11/04 03:44（台灣時間）完成第二組後，因 `papers`、`gnome-control-center` 未在桌面程式清單中而停止；兩組本身均正常早停，沒有中途模型需要續訓。

**目前讀取入口是 `development/aca_four_arm_campaign_v1/artifacts/campaign_g14srnja/continuation_6k8sd4fk/state.json`。**
同目錄 `launcher.json`、`plan.json`、`supervisor.log` 保存新程序身分、續接計畫及日誌；原 campaign 的 `latest_attempt.json` 指向此目錄。原 `state.json`／`failure.json` 仍保存先前停止紀錄，不能當作最新訓練狀態。

- Baseline seed24：12輪早停、最佳7輪，validation CE0.580804／accuracy74.79%／macro-F1 0.747646。
- LSTA seed24：16輪早停、最佳11輪，validation CE0.411338／accuracy81.77%／macro-F1 0.817369。
- 兩組正式 plan／history／summary／best.pt／last.pt 均保存 SHA256；不重跑、不改動。
- 原共同模型／資料／trainer／protocol 與十二組 readiness 指紋完全不變，仍採同一更新與選模迴圈。
- 新增獨立啟動 adapter，**只替換執行前的 GPU 排程檢查綁定**；此更動有獨立來源快照及 policy SHA，不把原 GPU 檢查說成未改變。
- 額外桌面例外只接受 `/usr/bin/papers`、`/usr/bin/gnome-control-center`：核對 root 保護權限、dpkg 套件歸屬及執行檔 SHA。其他未知 compute 程序仍攔截，並檢查至少6144MiB可用顯存；不關閉其他程式。
- 續接與原佇列合計17項 CPU 檢查通過，包含十組順序、跳過前兩組、完整十二組彙總、未知程序拒絕及原 trainer 數值函式未替換。
- 第一個新正式 run：`development/aca_four_arm_v1/artifacts/formal_tsci_seed20261024_viwjqsar/`。

[續接啟動時快照與保留檔案 SHA](results/aca_four_arm_continuation_v1.json)。這是時點快照，實際進度以新 attempt 的 state／每組日誌為準。後續維持 max30、patience5，不執行 test；完成全部十二組後才產生新的 campaign summary。

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
