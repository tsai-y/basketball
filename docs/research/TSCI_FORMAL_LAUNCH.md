# TSCI seed20261024 啟動紀錄

台灣時間2026-11-19 09:37啟動唯一TSCI八類正式run。run_id=`58b57208b6b8450a856271043199421a`，首個attempt=`attempt_d6a4f7015875`。使用者要求接續TSCI；既有baseline與LSTA已完成並保留，完整ACA尚未啟動。

TSCI從固定ImageNet V1與配對fresh head初始化，不載入LSTA已訓練權重。50幀RGB／160×120整段、train7192／validation1547、micro1/effective16、尾批8、450更新/輪、最多30輪、patience5、最低完整八類validation CE選模均維持原定協定。骨幹BN凍結；TSCI新增BN在train更新，validation使用已存統計。

原plan canonical SHA：`210a7ab1d1c05794b2fdd32c76f824735afe268f5f816e94ff308d9c97049424`。啟動前核對原程式／資料／runtime及LSTA完整history與選模結果通過，目標run原不存在。沿用先前使用者已核准的GNOME精確例外；重新驗證root保護、套件及SHA。與LSTA排程guard的AST一致，未擴大例外或改動數值程式。排程例外的完整證據見[scheduling_policy.json](/home/tsai/projects/basketball/outputs/00_admin/tsci_launch_20261119_v1/scheduling_policy.json)。

背景supervisor只啟動一次、不自動重啟。trainer PID3514483、supervisor PID3514482；實際身分與存活應以最新launcher／supervisor.json及宿主/proc核對，不只看PID。初始及25更新checkpoint已保存，GPU實際運算正常。

既有排程tanet改為「TSCI 每小時進度與完成心得」，每小時在目前對話回報，即使正常也回報。完成後核對summary與完整history，寫入run內final_review.md與final_review.json，比較baseline及LSTA同seed各自最低CE結果，分析類別取捨與限制，回報後將排程PAUSED。不自動啟動ACA、其他seed或test；所有成績為validation，不宣稱作者精確複刻或單seed顯著性。

[進度日誌](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_tsci_seed20261024_v1/progress.log) · [狀態](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_tsci_seed20261024_v1/run_state.json) · [啟動核對](/home/tsai/projects/basketball/outputs/00_admin/tsci_launch_20261119_v1/start_verification.json) · [固定計畫](ACA_SINGLE_SEED_LAUNCH.md)
