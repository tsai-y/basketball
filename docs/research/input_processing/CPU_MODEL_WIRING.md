# 八類 baseline：CPU初始化與模型接線驗收

日期：2026-11-17。**已完成獨立八類初始化，以及單一完整尺寸合成輸入到八類loss的接線；4項檢查全部通過。訓練仍未啟動。**

## 本輪實作

在原有資料模組內新增 [model_adapter.py](../../../development/aca_eight_class_data_v1/model_adapter.py) 與 [test_model_adapter.py](../../../development/aca_eight_class_data_v1/test_model_adapter.py)。未修改凍結FoundationDataset、舊VideoResNet50、二類initializer或舊trainer。

`initialized_baseline8()` 固定seed20261024，驗證已定稿規格與依賴SHA，讀取固定本機ImageNet檔案的bytes並驗證完整SHA後，才從同一份bytes反序列化。模型使用baseline、8類頭；沒有下載或載入任何舊任務checkpoint。

`baseline_training_mode()` 讓骨幹卷積及新頭維持可訓練，同時重新固定所有BN為eval、關閉其affine梯度。這是模式設定函式，不是訓練啟動器；後續trainer每次切入train模式都必須使用它。

## 實際結果

| 核對 | 結果 |
| --- | --- |
| ImageNet轉移 | 來源共267個張量；其中265個骨幹張量與模型逐一完全一致；1000類來源頭不轉移 |
| 新任務頭 | `Linear(2048,8)`；轉移前後新頭完全一致，未被來源頭覆寫 |
| BN | 53個BN模組；53個來源缺少的計數器明確設0；affine凍結且running統計未更新 |
| 參數 | 總23,524,424；可訓練23,471,304；凍結affine53,120，符合定稿規格 |
| 模組範圍 | 所有ACA插入位置均為Identity；沒有LSTA／TSCI |
| 唯一forward | CPU合成張量`[1,3,50,120,160]` → `[1,8]`有限logits |
| 目標與loss | 合成目標為class_id6（其binary_label是0），使用八類targets；CE與獨立logsumexp計算相符 |
| 狀態完整性 | forward前後所有模型權重與buffers的SHA完全相同；所有參數grad均為None |
| GPU／更新 | CUDA未初始化；沒有模型backward、optimizer、參數更新或checkpoint輸出 |

另外通過固定seed拒絕、權重缺失／大小／SHA錯誤拒絕、規格遭改動拒絕的檢查。既有64個基準檔案與51個歷史檔案SHA仍全部一致。

這次只建立一個模型、做一次full-size forward，使用no_grad。合成資料不是原始影片，沒有透過真實影片decoder取得輸入。此結果不表示模型accuracy提高，也不是validation／test成績。

[驗收明細、執行方式與來源SHA](eight_class_cpu_model_qa_v1.json)

## 尚未完成的接線

GPU、模型反向傳播、真實資料串流、optimizer loop、早停與checkpoint／resume仍待驗證。當前CPU-only指標契約與已定稿訓練規格均保留，沒有為了讓本次測試通過而修改輸入或擴大模型。

下一個有限任務是隔離的八類訓練迴圈及保存／恢復狀態：先以極小CPU合成fixture核對累積、尾批8、完整validation後選模及中斷復原；不直接跑完整資料集或啟動多seed佇列。本輪未實作或執行該任務。

[單一訓練規格](BASELINE_TRAIN_SPEC.md) · [八類資料／指標介面](EIGHT_CLASS_ADAPTER.md) · [基準資料驗收](BASELINE_ACCEPTANCE.md)

2026-11-17後續進度：上述下一個有限任務已完成CPU合成部分，見[訓練迴圈與恢復驗收](CPU_TRAIN_LOOP.md)。本頁保留原輪次驗收快照；GPU及真實資料端到端接線仍待完成。
