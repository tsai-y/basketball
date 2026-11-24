# ACA 重建工程

唯一明列假設候選的工程實作，不是作者原碼；正式CLI已備妥但未啟動。資料保持50幀RGB／160×120，詳見[提案](../../docs/research/ACA_RECONSTRUCTION_PROPOSAL.md)及[目前驗收](../../docs/research/ACA_ENGINEERING_STATUS.md)。

- model.py：直接建立dense1³＋depthwise3³ LSTA，重用既有forward及TSCI；配對初始化與精確BN策略。
- engine.py：隔離工程checkpoint schema；凍結數值迴圈逐字沿用，由AST檢查防漂移，僅改BN模式綁定。
- test_engineering.py：8項CPU／合成工程檢查；不開影片。
- gpu_readiness.py：固定24／8支ACA工程預檢，共4次暫存更新後清除權重；GPU鎖及未知compute阻擋規則不變。

CPU檢查及使用者接續後的有限ACA真實GPU預檢已通過；固定24／8支，共4次暫存更新，完整中斷恢復逐值一致，暫存權重已清除。初次GPU資源阻擋紀錄保留。此結果不代表分類效果或其他組的真實GPU驗收，正式入口仍未啟動，詳見[單seed計畫](../../docs/research/ACA_SINGLE_SEED_LAUNCH.md)。

formal_train.py提供prepare／start／resume／stop，training_runtime.py使用隔離正式schema；test_formal.py驗證身份、恢復及組別邊界。工程checkpoint不能用來初始化正式run。僅seed24的LSTA→TSCI→ACA，baseline沿用；沒有test、其他seed、自動重啟或延長輪數選項。
