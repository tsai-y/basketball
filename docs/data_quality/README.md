# Basketball-51 Data Quality 研究文件索引

組員的唯一技術入口是 [development/data_quality/README.md](../../development/data_quality/README.md)；本頁說明研究脈絡與證據，不另建立執行流程。v1／v2／v3 均已完成並提交；v3 的完成範圍是 scanner 與固定 smoke 驗證，全量執行仍由組員在 Ubuntu／tmux 啟動。

## 建議閱讀順序

1. 本研究索引：了解三版各自回答的問題及限制。
2. [v3 method](blur_screening_method_v3.md)：凍結規則、train 防護、輸出與恢復契約。
3. [v3 smoke report](blur_screening_smoke_v3.md)／[JSON](blur_screening_smoke_v3.json)：已完成的工程驗證、容量外推及其限制。
4. [v3 tmux runbook](../../development/data_quality/runbooks/blur_train_v3_tmux.md)：主機 checklist、正式命令與完成後處理。
5. 需要研究背景時，再讀下列 v1／v2 方法與報告；重現命令不代表需要再次執行。

## 三版研究問題、輸入與結果

| 版本 | 研究問題 | 輸入 | 方法 | 產出 | 結論與適用範圍 |
| --- | --- | --- | --- | --- | --- |
| v1：絕對清晰度探索 | 兩種全畫面清晰度指標的分布、相關性與候選影響為何？ | 固定 manifest 的 30 支 train，30 來源，Make/Miss 各 15 | 原始解析度 Laplacian variance／Tenengrad；Pilot P10、雙低與平滑／持續時間對照 | 4,617 幀分數、候選區段、窗口、contact sheets、Pilot 報告 | 流程可行，但來源／FPS／類別混雜，分數不是真實模糊標籤；不可制定刪除門檻 |
| v2：合成模糊與時間校準 | 指標能否感知合成模糊？如何凍結可供候選掃描的時間規則？ | 48 支 train、12 來源；來源互斥的 calibration／verification 各 24 支、6 來源 | Gaussian／motion 合成、paired relative／log drop、中心 3 秒 median/MAD local z；只在 calibration 選門檻 | 240 anchors、合成敏感度與混雜分析、凍結門檻、時間偵測報告 | 提供 v3 Laplacian 候選規則；verification F1 約 0.8617，僅限此合成測試，不是實際清理成效 |
| v3：完整 train 候選掃描器 | 能否以凍結規則串流處理完整 train，並保留可稽核、可恢復的結果？ | 固定 train allowlist 共 7,193 支；既有工程驗證限固定 20 支 smoke | 沿用 v2 規則、單影片處理、immutable attempts、completion hashes、resume／retry、derived aggregate 與獨立 verifier | 已有 20 支／3,055 幀 smoke 報告；正式掃描將產生逐幀 candidate flags、區段、診斷與 ledgers | Scanner complete，可於主機執行候選掃描；smoke 不代表全量盛行率，candidate 不是確認模糊或排除名單 |

表中數字均摘自既有報告；本次整理沒有重算或修改歷史結果。報告中的「本輪」「尚未提交」等敘述保留原交付時間點，不代表目前 Git 狀態。

## v1 為何不能直接制定門檻

v1 的 P10 是目的性小樣本內的相對低分位置，沒有人工模糊真值，也不是完整 train 的代表分布。低紋理可能低分；字幕、高紋理背景與鏡頭尺度可能掩蓋球或投籃者的模糊。每來源僅一支，來源、FPS 與類別互有混雜，不能將群組差異解讀成品質因果。分數亦受解析度、灰階與導數實作影響，不能跨條件套用絕對數值。Contact sheets 只檢查演算法反應，不構成標註資料。

## v2 如何得到凍結規則

v2 在每支 train 影片固定位置取得 anchors，使用可重現的 Gaussian／motion 工程尺度，並在時間串流注入已知區段。Calibration 與 verification 依來源分開；這裡的 verification 是 **train 內部的研究子集，不是資料集 validation/test**。

每條觀測串流各自計算 log-score 與中心 3 秒 rolling median/MAD，不使用乾淨原片分數作 oracle baseline。每策略搜尋 7 個 z 門檻 × 3 個最短持續時間，只以 calibration 在 control flag rate ≤5% 約束下最大化 source-macro frame F1；同分依較低 control flag rate、較嚴 z、較長 minimum duration 決定。先保存 frozen thresholds，才載入 verification 時間分數，verification 不重新調參。

Laplacian 凍結為 local z ≤ −2.0、minimum duration 0.16 秒；中心窗 3 秒、MAD scale floor 0.05、log floor 1e-12。兩指標皆感知本次合成模糊，AND／OR 沒有提升 overall F1，故 v3 以 Laplacian 為 primary，Tenengrad 保留診斷。Control 本身可能模糊，其旗標率不是真實世界 FPR；合成全畫面、瞬間邊界也不等於自然模糊。

## v3 如何沿用規則

v3 核對已提交 v2 config、JSON、方法與評分程式的 frozen hashes，重用評分與 local normalization，不用 smoke 或 full train 重新擬合門檻。Primary candidate 只依上述 Laplacian 規則；Tenengrad、低絕對分數與 16/32/64 非候選窗口只提供診斷資訊。中心窗使用前後畫面，屬離線篩檢。

只有固定 manifest 的 train 可進入影片 opener；validation/test 影片不得存取。原始影片、label、manifest、split、baseline 及歷史 runs 保持不變。完整掃描依 [runbook](../../development/data_quality/runbooks/blur_train_v3_tmux.md) 在 Ubuntu／tmux 執行，輸出至獨立 ignored artifacts，失敗與候選分開記錄；resume／retry 不覆寫舊 attempts。

全量完成後應 verify 並保存 run 與 runtime logs。現有 publisher 只支援固定 smoke，**full-train Markdown／JSON publication 尚未實作**，不可將 smoke publisher 當全量 publish 命令；詳見 runbook 的 publish 限制。

## 尚未完成的正式清理政策

- 真實模糊的品質定義、人工真值與複核程序，以及球／籃框／投籃者等主體範圍。
- 跨來源、FPS、紋理、字幕、鏡頭尺度的泛化；局部正規化可能漏掉整片持續低品質。
- 誤判成本、短片與長期低品質的處理，以及保留／排除／刪格／重編碼的正式審核規則。
- 清理或保留策略對 Make/Miss 分類的效益證據與後續研究 protocol。

目前沒有正式排除名單或 cleaned dataset。Candidate 不等於確認模糊，未標記不等於清晰，**不得直接刪除 candidate 資料**。本文件不授權影片掃描、清理、訓練或評估。

## 方法、報告與重現索引

| 版本 | 方法文件 | Markdown 報告 | JSON | 重現與設定 |
| --- | --- | --- | --- | --- |
| v1 | [method v1](blur_screening_method_v1.md) | [pilot v1](blur_screening_pilot_v1.md) | [pilot v1 JSON](blur_screening_pilot_v1.json) | [統一入口內歷史重現](../../development/data_quality/README.md)、[config](../../development/data_quality/configs/blur_pilot_v1.yaml)、[manifest](../../development/data_quality/pilot_manifest_v1.csv) |
| v2 | [method v2](blur_screening_method_v2.md) | [pilot v2](blur_screening_pilot_v2.md) | [pilot v2 JSON](blur_screening_pilot_v2.json) | [README_v2](../../development/data_quality/README_v2.md)、[config](../../development/data_quality/configs/blur_pilot_v2.yaml)、[manifest](../../development/data_quality/pilot_manifest_v2.csv)、[來源庫存](../../development/data_quality/source_inventory_v2.csv) |
| v3 | [method v3](blur_screening_method_v3.md) | [smoke v3](blur_screening_smoke_v3.md) | [smoke v3 JSON](blur_screening_smoke_v3.json) | [tmux runbook](../../development/data_quality/runbooks/blur_train_v3_tmux.md)、[config](../../development/data_quality/configs/blur_train_v3.yaml)、[smoke manifest](../../development/data_quality/smoke_manifest_v3.csv) |

詳細分數、圖像與 runtime evidence 留在本機 ignored artifacts；clone 不會取得這些產物。歷史紀錄見 [progress log](../progress_log.md)，目前操作一律從 [唯一技術入口](../../development/data_quality/README.md) 開始。
