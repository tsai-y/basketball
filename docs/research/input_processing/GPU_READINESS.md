# 八類 baseline：真實輸入與 GPU 端到端預檢

日期：2026-11-17。**有限預檢通過：24支train、8支validation，總計4次工程參數更新。正式訓練尚未啟動。**

目前完成鏈：基準資料驗收 → 八類介面 → 固定單一baseline規格 → CPU初始化 → CPU迴圈／恢復 → **真實影片、兩個worker、CUDA及中斷恢復接線**。

本輪唯一新增程式為 [gpu_readiness.py](../../../development/aca_eight_class_data_v1/gpu_readiness.py)。它繼承既有 `FixtureLoop` 的 `_run` 狀態機，新增受限的 CUDA 張量傳送與checkpoint介面；沒有修改已驗收的 CPU 核心或舊二類trainer，沒有新增模型或重新比較架構。

## 固定執行範圍

| 項目 | 本輪範圍 |
| --- | --- |
| 輸入 | 維持50幀RGB、整段均勻取樣、160×120、ImageNet常數正規化；無裁切／增強 |
| 模型 | 固定ImageNet V1初始化的baseline ResNet50＋新八類頭；全部BN affine及統計凍結 |
| 選片 | 既定train各類依ID排序取前3支，共24支；validation各類取前1支，共8支；未依模型成績挑片 |
| 資料讀取 | 2個spawn workers、micro1、prefetch2、獨立loader generator；逐片核對八類／二類對映及tensor SHA |
| 連續路徑 | 24支train＝16＋尾批8，2次更新；之後完整讀取本預檢8支validation |
| 中斷路徑 | 先16支、1次更新後存檔暫停；新建模型及Adam，恢復後做剩下8支、1次更新，再讀8支validation |
| 總量 | 32支不同真實影片、96次有限解碼、48次train forward/backward、16次eval forward、4次工程更新 |
| 執行上限 | 固定1個預檢輪次、4次工程更新、外部900秒timeout；不接受擴大成正式訓練的參數 |
| 權重 | 全部暫存於TemporaryDirectory，預檢結束已刪除；禁止作為正式初始化 |
| GPU檢查 | 沿用既有桌面程序例外與未知compute阻擋規則；未放寬規則或停止其他程序 |

所稱8支validation是**工程預檢子集**，並非完整1547支研究validation。不以小樣本accuracy／CE或training loss宣稱模型改善。

## 實際結果

| 檢查 | 結果 |
| --- | --- |
| 真實輸入與worker一致性 | 所有經worker輸出的tensor SHA、ID、split、class_id及binary_label都與單程序參考一致 |
| loader亂數隔離 | 建立worker iterator前後，模型使用的Python／NumPy／torch CPU RNG未被消耗 |
| GPU梯度與更新 | 骨幹及八類頭均取得有限、非零梯度；stem、末段卷積及head均有更新；所有凍結BN權重與buffers不變 |
| CUDA RNG | 保存狀態、產生亂數、恢復狀態後再次產生的CUDA亂數完全相同 |
| 中斷恢復 | 連續與恢復路徑的模型、Adam、CPU RNG、CUDA RNG、cursor／updates／指標／history，以及最佳權重逐項完全相同 |
| 選模提交 | 當前模型、最佳權重與完整預檢history同存於一份checkpoint，磁碟內容與已提交記憶體狀態一致 |
| 原始資料與程式 | 32支來源影片SHA前後一致；凍結規格核對通過；64個基準及51個歷史受保護檔案SHA一致 |
| 執行後清理 | 程序正常退出、工程權重已刪除；GPU compute清單僅剩原有桌面Discord程序 |

環境：RTX5070Ti 16GiB、PyTorch2.11.0+cu128、CUDA12.8、FP32、AMP／TF32關閉。整個有限預檢約22.8秒；最高allocated約2.04GiB、reserved約2.28GiB。已完成輪次的工程checkpoint約359MiB。這些是小樣本工程量測，不能直接當成全資料訓練時間預估。

[完整執行計畫](/home/tsai/projects/basketball/outputs/01_current/baseline8_gpu_readiness_20261117_v1/plan.json) · [原始預檢報告](/home/tsai/projects/basketball/outputs/01_current/baseline8_gpu_readiness_20261117_v1/report.json) · [整理後驗收紀錄](eight_class_gpu_readiness_qa_v1.json)

## 下一步與尚未完成範圍

下一步集中補齊**單一baseline的正式啟動及一般續訓入口**：綁定全量train7192／validation1547的清單與順序、正式run身分、停止／恢復、日誌及資源檢查。維持原定seed20261024、30輪上限、patience5，不新增模型或多seed實驗。

目前GPU adapter刻意只接受本預檢epoch0／cursor16的恢復點；跨輪次及validation途中中斷的通用CUDA恢復入口尚未接入。CPU核心已驗收這些狀態機行為，但不能把CPU驗收直接宣稱為GPU正式續訓就緒。因此報告明確保留 `formal_training_ready=false`；這是尚待接線，並非本輪預檢失敗。

本輪沒有讀取test、重啟舊實驗、開啟每小時排程或commit／push。原凍結規格是定稿快照，保持其SHA不變；進度以本頁及QA紀錄補充。沒有90%或作者精確複刻的結論。

[上一輪CPU迴圈驗收](CPU_TRAIN_LOOP.md) · [固定訓練規格](BASELINE_TRAIN_SPEC.md) · [資料驗收](BASELINE_ACCEPTANCE.md) · [工作台](../../../workspace/README.md)

2026-11-17後續：正式啟動及一般續訓入口已補齊，合成CUDA驗收通過、單一baseline完整計畫已準備，見[正式啟動入口](FORMAL_LAUNCH.md)。本頁保留當時預檢範圍，尚未執行正式訓練。
