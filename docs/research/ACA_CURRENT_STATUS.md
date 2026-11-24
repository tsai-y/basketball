# ACA 複刻目前狀態與執行順序

更新：2026-11-19。此頁是目前研究入口；歷史報告保留各自時點。資料基礎及單一八類 baseline 已完成；唯一候選的工程驗收與單seed正式計畫／入口已完成，LSTA已完成，TSCI seed20261024已於2026-11-19 09:37啟動。目前只有TSCI一組正式訓練，每小時監看ACTIVE；[啟動紀錄](TSCI_FORMAL_LAUNCH.md)。

## 已完成與採用的內容

| 工作 | 目前成果 | 適用範圍 |
| --- | --- | --- |
| 資料流程 | train7192、validation1547；test僅本輪metadata1538；隔離34 | 原source-group切分；全專案有歷史test接觸 |
| 輸入 | 完整片段原生幀索引均勻50幀RGB，寬160×高120，ImageNet正規化 | 無ROI、事件窗口或augmentation；50幀不是全部原生幀 |
| 驗收 | train＋validation8739支技術檢查通過，固定16支內容對照結案 | 保留5支疑義；未全面人工核定標籤 |
| 八類工程 | class_id、CE、8×8混淆矩陣、macro-F1與中斷恢復已接通 | 沿用已通過的CPU／GPU驗收，不重跑 |
| 八類baseline | 2026-11-17 16:54台灣時間完成12輪、5400次已保存更新，早停5／5 | seed20261024，最低完整validation CE選模 |
| 歷史研究 | 二類ACA四組seed24、TANet及其他支線保留 | 不併入目前八類公平消融，不自動續跑 |
| 未啟用／否決 | 80/10/10與虛擬25Hz候選未啟用；固定5%裁切否決 | 不因舊文件的下一步而重新啟用 |

[輸入驗收](input_processing/BASELINE_ACCEPTANCE.md) · [歷史四組結果](ACA_REPRODUCTION_CHECKPOINT.md) · [產物分類入口](/home/tsai/projects/basketball/outputs/index.html)

## 八類 baseline 最終結果

按最低完整 validation CE 選到第7輪：CE **1.1907208**、八類accuracy **59.6639%**、macro-F1 **0.4425495**。八類argmax映射為Make／Miss的accuracy為**81.5126%**，不是另一個二類訓練結果。全部是validation，本run未執行test。

各類recall：2p0 47.9532%、2p1 60.1504%、3p0 78.3708%、3p1 58.9595%、ft0 10.0000%、ft1 98.1949%、mp0 14.6341%、mp1 1.0989%。類別不均的結果不足以單獨判定輸入、標籤或架構為原因。

[summary.json](/home/tsai/projects/basketball/outputs/01_current/baseline8_formal_seed20261024_v1/summary.json) · [history.json](/home/tsai/projects/basketball/outputs/01_current/baseline8_formal_seed20261024_v1/history.json) · [progress.log](/home/tsai/projects/basketball/outputs/01_current/baseline8_formal_seed20261024_v1/progress.log)

固定ImageNet V1初始化，Adam lr1e-4、micro1/effective16、尾批8、每輪450更新、最多30輪、patience5，BN affine/buffers凍結。權威checkpoint為同run的last.pt，最佳權重在best_model欄位。完成run不續訓、不延長。

## 完整執行順序與關卡

| 階段 | 交付與完成條件 | 現況 |
| --- | --- | --- |
| 1 現況收斂 | 本頁、正式結果核對及入口同步 | 完成 |
| 2 ACA架構定稿 | 已知／未知／專案假設逐項分開，參數預算可解釋；決定作者證據或假設重建路線 | 已選明列假設重建並接續工程；候選按提案實作 |
| 3 共同協定 | 一次固定split、輸入、初始化、BN、優化、停止與指標定義；核定baseline能否沿用 | 共同協定與三份正式plan已固定；baseline核定沿用 |
| 4 有限工程檢查 | 只驗證新模組shape、梯度、參數、BN與必要端到端／恢復；通過即停止 | CPU8項及ACA有限真實GPU預檢通過，工程檢查結束 |
| 5 單seed消融 | 同協定baseline、LSTA、TSCI、ACA依序各一組；結果無增益也照實保存 | baseline及LSTA完成；TSCI已啟動，ACA尚未啟動 |
| 6 三seed重複 | 相同四組補足預先固定三seed；逐seed及平均／樣本標準差 | 不預先啟動佇列，不依分數挑seed |
| 7 鎖定後評估 | 事先鎖定選模與指標；核准後才做test，揭露歷史接觸 | 未執行；不能稱全新未見holdout |
| 8 SpaceJam | 獨立資料契約與分類頭，先釐清原文16／8幀差異，再沿用定稿方法 | 後續範圍，不直接套50幀輸入 |
| 9 結案 | 程式／環境／清單／設定／權重／指標及論文差異表可追溯 | 不以達90%才結案 |

Basketball-51四組×三seed共12個正式run；目前baseline已確認相容並沿用，剩11個。單seed四組完成且協定有可解釋性後才安排其餘8個。若共同BN、split或訓練方法變更，先說明baseline為何需重跑；不偷偷替換其成果。

完整論文所有比較表的其他方法重跑屬另行核定的擴充範圍；本主線先完成ACA架構、核心消融與雙資料集證據。正式ACA時數必須由定稿模型的有限資源量測估計，不能把baseline速度直接套用。

## 目前工程進度與待處理事項

使用者已於2026-11-18指示接續唯一候選，已完成[隔離實作與CPU工程驗收](ACA_ENGINEERING_STATUS.md)。原[提案](ACA_RECONSTRUCTION_PROPOSAL.md)及JSON保留審閱時快照，實際接續授權與進度記於工程紀錄。

使用者釋放GPU並指示接續後，ACA固定24 train＋8 validation有限預檢通過；4次暫存更新，連續與恢復完整狀態相同，權重已清除。舊資源阻擋紀錄保留。[單seed正式計畫與入口](ACA_SINGLE_SEED_LAUNCH.md)已完成，三份plan重新驗證通過；LSTA已於2026-11-19明確授權啟動，詳見[啟動紀錄](LSTA_FORMAL_LAUNCH.md)。

現有baseline的初始化、合成輸出、共同數值迴圈及資料契約相容核對通過，保留原結果；ACA真實CUDA接線已通過；正式沿用決定已記入launch計畫，不重跑baseline。

[工程證據](results/aca_reconstruction_engineering_v2.json) · [歷史缺口](ACA_ARCHITECTURE_GAPS.md)

最新接續：LSTA完成11輪、4950更新，最低CE選第6輪；[完成心得](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_lsta_seed20261024_v1/final_review.md)。TSCI已啟動，見[TSCI啟動紀錄](TSCI_FORMAL_LAUNCH.md)，不自動接跑ACA。
