# 單一八類 baseline：訓練規格定稿

日期：2026-11-17。**規格已定稿；實作與訓練啟動尚未完成，也未授權自動開跑。**

這是以已凍結資料建立一個可解釋的八類基準。沿用現有ResNet50 baseline，不加入LSTA、TSCI、新裁切或多seed佇列。名稱：`frozen_native50_rgb_baseline8_seed20261024_v1`。

[完整機器可讀規格](../specs/eight_class_baseline_train_v1.json) · [本輪靜態核對紀錄](eight_class_baseline_train_spec_qa_v1.json)

## 固定設定

| 項目 | 定稿值 |
| --- | --- |
| 資料 | 既有source-group split，train7192／validation1547；test不開啟；疑義旗標保留 |
| 輸入 | 50幀RGB、160×120、完整片段均勻取樣，沿用FoundationDataset，無augmentation |
| 目標 | 八類class_id；二類只作另外標明的衍生指標 |
| 模型 | 現有VideoResNet50，variant=baseline，所有ACA位置為Identity；時間與空間平均後接Linear(2048,8) |
| 初始化 | 已固定ImageNet V1檔案，完整SHA核實；移除來源1000類頭，建立新8類頭；不用任何舊訓練權重 |
| Seed | 僅20261024，沿用既有研究seed序列第一個；在任何八類分數前固定，不搜尋seed |
| 可訓練層 | 骨幹卷積與8類頭都訓練；所有BN affine及running buffers凍結，train模式後重新固定BN為eval |
| 參數預期 | 既有盤點：總23,524,424，可訓練23,471,304，凍結BN affine53,120；接線時須實際複核 |
| Optimizer | Adam，lr1e-4，betas=(0.9,0.999)，eps1e-8，wd0；amsgrad／foreach／fused關閉；無scheduler |
| 梯度 | 每有效批次歸零一次、累積完成後clip L2 norm1，再更新一次；非有限值直接停止 |
| Batch | 實體1片、有效16片；每輪449個16片批次＋最後8片，共450次更新；不丟尾批 |
| Loss | 八類CE，無class weights／label smoothing／額外二類loss；每個microbatch的sum CE除以實際有效批次影片數 |
| 輪數／選模 | 上限30輪；完整1547支validation的八類CE嚴格下降才換best；平手留早輪；連續5輪未改善停止 |
| Validation | eval、no_grad、每片一次；報八類CE／accuracy／macro-F1／混淆矩陣／recall；二類使用八類argmax後映射 |
| 數值與執行 | 單GPU FP32，關閉AMP／TF32，固定deterministic；GPU資源可用性尚未在本輪驗證 |
| 中斷保存 | 每25次更新及輪末保存；使用者停止時到optimizer邊界保存；續訓需明確要求與相同版本身分 |

最後一批必須除以8，**不能沿用舊7193支的尾批9，也不能一律除以16**。最多30×450＝13,500次更新；早停可能提前結束。這是更新數上限，不是工時估計。

BN凍結沿用專案既有做法，骨幹其餘部分仍會更新。實體1＋累積16不宣稱等同作者實體batch／BN行為；本輪沒有以這個選擇宣稱作者精確複刻。

## 已確認、尚需接通的地方

1. 舊 `weights.initialized_baseline` 寫死 `num_classes=2`，不得直接呼叫。下一步須建立隔離八類initializer，使用現有嚴格ImageNet轉移邏輯；53個來源缺少的BN計數器明確設0並記錄。新頭保留constructor初始化，seed在建模前只設一次。
2. 新adapter的batch使用 `targets` 與 `records`，舊trainer使用 `binary_label` 及頂層video_id，不能只替換模型頭就啟動。
3. 新loss／metrics介面目前只驗收CPU。未來GPU訓練須明確以CUDA logits與八類targets計CE；指標以detach後的CPU資料累積。GPU橋接尚未驗收，不偷偷把資料搬回CPU當成完成接線。
4. 新metrics沒有舊二類 `Metrics.state` 結構；未來checkpoint須新增並驗證八類狀態序列化，不能沿用舊resume schema。
5. 中斷validation不得更新best或早停計數；續接時重跑完整validation。best權重與history須有一致的提交／復原規則，避免斷電後配錯epoch。

## 本輪實際驗證範圍

只核對程式文字、既有盤點及metadata；驗證本機ImageNet檔案102,530,333 bytes與完整SHA，但未反序列化權重或建立模型。64個基準檔案與51個歷史檔案內容指紋全數一致。資料數、尾批與更新數算術通過；前輪10項介面測試結果沿用，沒有為文件變更重跑。

[論文差異表](PAPER_INPUT_GAP.md)列出的split、原生FPS、裁切與作者細節缺證仍有效。BN、patience、seed、梯度裁剪及確切初始化等是明示專案選擇，不是作者原始設定。規格定稿沒有解決ACA約26.8M的架構差距。

## 下一個單一任務

**實作隔離八類initializer及CPU模型接線檢查**：核對預訓練轉移、8類頭、參數數量、BN凍結、單一合成輸入的8維輸出與loss接線；不做optimizer更新、不開真實影片、不啟動訓練。先完成這個有限任務，再處理GPU／trainer／resume，不直接排多組訓練。

後續更新（2026-11-17）：[隔離八類初始化與CPU單次合成forward驗收已完成](CPU_MODEL_WIRING.md)。規格JSON維持原定稿快照；trainer／GPU／resume尚未驗收，沒有啟動訓練。
