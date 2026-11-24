# ACA 重建工程驗收結果

日期：2026-11-18。唯一候選的隔離實作、8項CPU／合成驗收及有限ACA真實GPU預檢均已完成。使用者關閉占用kernel並指示「接續」後，以新attempt保存結果；正式訓練尚未啟動。

## 已完成的工程

新程式集中在[aca_reconstruction_v2](../../development/aca_reconstruction_v2/README.md)，舊模型、baseline入口、資料契約及權重保留。

| 檢查 | 已驗證結果 |
| --- | --- |
| 四組完整尺寸meta shape與參數 | 16個插入點尺寸、8類輸出均正確；baseline 23,524,424、LSTA 25,471,780、TSCI 23,529,224、ACA 25,476,580 |
| 配對初始化 | 四組骨幹／head相同，共有LSTA／TSCI模組逐張量相同；新增模組不消耗共同CPU RNG |
| baseline相容性 | 同初始化、BN模式與合成輸入logits逐值相同；資料契約保持原SHA |
| 動態時間核 | 與獨立逐索引參考一致，跨影片梯度隔離 |
| 新模組梯度與BN | LSTA／TSCI梯度有限且模組總量非零；骨幹BN凍結、新BN訓練、eval不改buffers |
| 共同數值迴圈 | 新隔離迴圈_run與凍結核心AST相同，僅全域training_mode綁定改為新BN規則 |
| 中斷恢復 | CPU小模型的train及validation中斷後，模型／BN／Adam／RNG／history／best選模與連續執行完全相同 |
| 尾批與範圍 | 尾批8支與獨立Adam累積參考相同；正式plan、其他seed及超出工程範圍被拒絕 |

8項unittest通過，約2.34秒，CUDA未初始化。包含極小合成模型更新，不產生分類效果證據。測試細項同時涵蓋上表內容，不是新增8組訓練。

## GPU 有限預檢結果

新attempt為 `gpu_attempt2`，耗時30.96秒。固定24支train＋8支validation，保持50幀RGB／160×120；連續路徑2次更新、中斷路徑1次、恢復路徑1次，共4次工程更新、96次解碼。沒有開啟test。

| 檢查 | 結果 |
| --- | --- |
| 真實輸入 | 來源SHA及輸入tensor SHA與凍結全資料驗收一致；兩個worker與serial一致 |
| 梯度與更新 | backbone、head、LSTA、TSCI梯度有限且各模組總量非零；骨幹及head確實更新 |
| BN | 骨幹BN完全不變；48個TSCI BN訓練計數符合16支中斷／24支完成；eval不改buffers |
| 中斷恢復 | model、Adam、CPU/CUDA RNG、state/history及best_model與連續執行逐值完全相同 |
| 權重清理 | 暫存checkpoint已刪除，不能沿用為正式初始化 |
| 資源 | PyTorch峰值allocated約3.69 GiB、reserved約3.94 GiB；這是小型預檢量測，不是正式全程上限 |
| 保護範圍 | 28份受保護檔案及新程式SHA重新核對未變；原影片未變 |

首個 `gpu/` attempt仍保留：當時PID3722836的Jupyter kernel占用約1206 MiB，於資源門檻停止，沒有解碼或更新。本次確認該占用已釋放後才接續；未停止其他程序、未放寬GPU規則。新attempt成功不覆寫舊紀錄。

## 範圍與下一步

此輪工程驗收完成，停止追加預檢。真實CUDA只涵蓋ACA組；其餘組具有CPU／meta證據。小樣本更新不是分類效果驗證，不能據此宣稱準確率提升或作者精確複刻。

後續更新：[單seed正式執行計畫與入口](ACA_SINGLE_SEED_LAUNCH.md)已備妥，baseline已核定沿用。該計畫固定：明列baseline沿用核對、LSTA→TSCI→ACA順序、共同資料／BN／優化／選模協定、資源及中斷處理。既有baseline結果保留；三份正式plan已驗證，下一步為LSTA一組。本輪未啟動正式訓練、其他seed或test。

[CPU報告](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_engineering_20261118_v1/cpu_report.json) · [GPU通過報告](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_engineering_20261118_v1/gpu_attempt2/report.json) · [歷史GPU阻擋](/home/tsai/projects/basketball/outputs/01_current/aca_reconstruction_v2_engineering_20261118_v1/gpu/failure.json) · [來源SHA與驗收紀錄](results/aca_reconstruction_engineering_v2.json) · [原提案](ACA_RECONSTRUCTION_PROPOSAL.md)
