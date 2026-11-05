# ACA 四組 × 三 seed 正式訓練（更新：2026-11-05）

已依使用者「下一步」指示啟動十二組全新正式訓練。這是共同協定下的專案消融比較；尚未核實為作者精確復刻。既有 ACA／TANet 結果與權重不作為這十二組的初始化或對照成績。

啟動前重新核對十二組 GPU readiness、CPU plan、資料與來源雜湊、環境及 ImageNet 權重；共用 trainer／protocol 未更改。獨立監督器的九項 CPU 檢查通過。新增監督器首次啟動在正式 run 建立前，因 JSON list/tuple 直接比對誤判而被攔下；改為與預檢相同的 canonical JSON SHA 後才開始本次訓練，並未產生額外正式更新或重跑。

## 2026-11-05 GPU 環境修復與剩餘八組續接

使用者明確要求「修正並持續」後，已從 Baseline／seed20261025 接續。**最新 state／launcher／plan／supervisor.log 位於 `development/aca_four_arm_campaign_v1/artifacts/campaign_g14srnja/continuation_6k8sd4fk/continuation_runtime_4vljx69c/`。** 原 campaign 的 `latest_attempt.json` 已更新，前兩層失敗 state 保留作為歷史。

seed20261024 的四組均已完成並保留：Baseline 12輪／最佳7，LSTA 16輪／最佳11，TSCI 11輪／最佳6，完整ACA 14輪／最佳9。完整ACA所選validation CE0.418008／accuracy81.71%／macro-F1 0.816977。尚未開始的seed25、26共八組接續，不重跑這四組。

停止原因已核實：系統11/05 06:33自動更新NVIDIA使用者端套件至595.99.02，已載入核心仍595.91.07；完成ACA後，08:01在下一組的NVML啟動檢查失敗。Baseline seed25當時沒有子程序、stdout或正式run；新plan記錄明確授權的prelaunch reset，舊失敗state沒有改寫。

修復採獨立程序環境：從 [Ubuntu官方建置33482603](https://api.launchpad.net/1.0/~canonical-x/+archive/ubuntu/prerelease-drivers/+build/33482603) 下載原版595.91.07的compute與utils套件，依官方.changes核對SHA256，只用dpkg-deb解壓，不執行安裝腳本、不變動系統套件、不重啟主機。原版CUDA／NVML與nvidia-smi保存於 ignored `development/aca_four_arm_campaign_v1/artifacts/runtime_5959107_g80cyzp_/`；兩個套件合計約86MB下載，執行所需11個檔案與11個連結另保存指紋。

- GPU查詢、CUDA小型forward/backward精確梯度及實際程序libcuda／NVML載入映射已通過。這是環境驗證，正式optimizer更新0，不是模型成績。
- 原共同campaign SHA `12085d4b6a7a430e5faed853e028b0f7b7d997db4a05147371f054e82b949d83` 完全恢復一致；十二組原readiness仍有效。沒有更改數值trainer／資料／規則或偽造GPU查詢輸出。
- 固定OpenCV套件會在LD_LIBRARY_PATH前置自己的lib64；只允許該已知目錄，且必須不存在可覆蓋CUDA/NVIDIA函式庫的檔案。其他未知優先路徑仍拒絕。
- 恢復入口與原佇列合計21項CPU檢查通過。四組成果與best/last權重SHA逐一核對，保存在新plan的preserved_files。
- 新正式run為 `development/aca_four_arm_v1/artifacts/formal_baseline_seed20261025_0nv14783/`；已觀察實際更新，後面七組排隊。此時點不代表即時進度。

監測GPU必須讀取最新attempt `launch_request.json` 的 `runtime_environment`，以同樣的PATH／LD_LIBRARY_PATH執行真實nvidia-smi。系統預設環境仍可能因更新後版本失配而查詢失敗；本次修復限訓練程序。主機若重啟載入不同版本核心，必須重新核對，不能假設這份局部環境仍適用。排程已恢復每小時回報，不自動修環境或重跑。

[修復來源、環境驗證、保留成果SHA與啟動快照](results/aca_four_arm_runtime_recovery_v1.json)。模型上限30輪、patience5、最低完整validation CE選模及不執行test均維持不變。

## 2026-11-04 明確授權續接

使用者於本對話明確要求持續執行後，已保留完成的 baseline／LSTA（seed20261024），從尚未開始的 TSCI／seed20261024 接續十組。原佇列在 11/04 03:44（台灣時間）完成第二組後，因 `papers`、`gnome-control-center` 未在桌面程式清單中而停止；兩組本身均正常早停，沒有中途模型需要續訓。

該次11/04 attempt 的入口為 `development/aca_four_arm_campaign_v1/artifacts/campaign_g14srnja/continuation_6k8sd4fk/state.json`；其後已停止，最新入口見11/05章節。
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
