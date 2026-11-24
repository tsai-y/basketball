# LSTA seed20261024 啟動紀錄

台灣時間2026-11-19 03:44啟動唯一LSTA八類正式run，run_id=`e8c0acd4fc5c4237a7bee0de476a63bc`。使用者已授權啟動、每小時回報與完成心得。原baseline沿用；未啟動TSCI、ACA或其他seed。

固定50幀RGB／160×120、train7192／validation1547、每輪450 updates、最多30輪、patience5，最低完整validation八類CE選模。原計畫與數值程式SHA核對通過。這是明列假設的單seed重建消融，不是作者精確版。

## 排程例外與身分

首個資源檢查被GNOME桌面服務阻擋後已停下詢問。使用者核准後，新建獨立排程adapter，只額外允許`/usr/libexec/xdg-desktop-portal-gnome`，每次驗證root權限、套件歸屬與SHA。原未知compute阻擋、6144 MiB空間及8 GiB磁碟門檻保留；未停止其他程序。

原formal plan canonical SHA仍為 `076399e5668f6df57124e84de9027a2f8c8cbf7ad8a044c9f4e060530bad2e12`。adapter只替換入口的GPU檢查函式，不改原程式、模型、資料或權重。原plan的gpu_policy是基礎規則，新增的已授權例外由獨立[scheduling_policy.json](/home/tsai/projects/basketball/outputs/00_admin/lsta_launch_20261119_v1/scheduling_policy.json)記錄，不能只看原plan推測完整啟動排程。

- 首個trainer：PID129784／start_ticks5250144；supervisor：PID129783／start_ticks5250142。
- boot_id：88ae0970-f2af-45aa-a7a6-7f43c107a2bb；實際存活以最新launcher與宿主交叉核對。
- 首個attempt：attempt_fcb706ca40eb；已保存初始checkpoint，GPU實際開始運算。
- supervisor只啟動一次、不自動重啟或接跑下一組。保留[啟動核對證據](/home/tsai/projects/basketball/outputs/00_admin/lsta_launch_20261119_v1/start_verification.json)。

## 每小時回報與心得

既有排程`tanet`已改為「LSTA 每小時進度與完成心得」，ACTIVE，每小時在目前對話回報，即使正常也回報。內容包括階段、輪數、train／validation進度、累積與已保存更新、最新完整validation CE／八類accuracy／macro-F1、最佳輪與早停計數。

完成後核對summary與完整history，依最低CE模型寫入run內`final_review.md`及`final_review.json`，比較同seed baseline的CE、accuracy、macro-F1、每類recall／混淆矩陣、Make／Miss診斷與改善退步，說明曲線、早停、限制及下一步。單seed不能推論統計顯著性；所有數值均為validation，不執行test或重新評估權重。回報後將排程PAUSED。

[進度日誌](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_lsta_seed20261024_v1/progress.log) · [目前狀態](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_lsta_seed20261024_v1/run_state.json) · [背景stdout](/home/tsai/projects/basketball/outputs/00_admin/lsta_launch_20261119_v1/stdout.log) · [完整固定計畫](ACA_SINGLE_SEED_LAUNCH.md)
