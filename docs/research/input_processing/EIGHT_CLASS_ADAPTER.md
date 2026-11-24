# 八類資料與指標介面驗收

日期：2026-11-16。**隔離介面已實作，10項CPU工程檢查通過；沒有啟用八類訓練。**

## 已完成

在既有 `development/aca_eight_class_data_v1` 內新增 [adapter.py](../../../development/aca_eight_class_data_v1/adapter.py) 與 [test_adapter.py](../../../development/aca_eight_class_data_v1/test_adapter.py)，沒有建立新模型專案或大型產物資料夾。

- `EightClassDataset` 包裝凍結的FoundationDataset，委派原有像素處理，保留來源資訊；只接受train／validation。
- `collate_eight_class` 將原張量堆成 `[N,3,50,120,160]`，明確從 `class_id` 產生int64 `targets`；`binary_targets`另列。核對原類別、二類映射、契約、形狀及有限值，拒絕混合split或同批重複影片。
- `eight_class_ce` 為無權重、無label smoothing的八類平均CE，保留梯度。這版僅接受CPU float32／float64 logits，尚未驗證GPU訓練接線。
- `EightClassMetrics` 以樣本數加權累積CE、輸出8×8混淆矩陣（列真值、欄預測）、accuracy、各類recall／F1及macro-F1。macro固定平均全部8類，分母為0時該類計0；空結果拒絕。指標累積使用detach與float64，不保留梯度圖。
- 衍生二類指標僅採「八類argmax後映射成Miss／Make」，不使用兩組機率加總；平手選類別順序較前者。

類別順序固定：`2p0, 2p1, 3p0, 3p1, ft0, ft1, mp0, mp1`；二類對映 `0,1,0,1,0,1,0,1`。Standalone loss／metrics無法從值為0或1的targets猜出呼叫者是否誤傳二類標籤，所以接入時必須使用經collate核對的 `targets`，不可改用 `binary_targets`。

## 驗收證據

6項新介面測試加4項既有資料邊界測試全部通過。新增測試全程禁止呼叫影片decoder；資料筆數以固定metadata確認7192／1547。張量委派與批次測試使用合成張量，不重做真實影片解碼。

手算案例涵蓋：已知機率的CE與解析梯度、八類混淆矩陣與macro-F1、同為Miss但動作類別錯誤、不同批次大小及尾批、二類機率加總與八類argmax不一致。另確認錯誤標籤／dtype／形狀／NaN／Inf被拒絕，拒絕後指標狀態不變。

這些是人工構造的工程案例，**不是訓練、validation或test成績**。既有64個基準檔案及51個歷史檔案內容指紋全部保持一致。

[機器可讀驗收、執行命令與來源SHA](eight_class_adapter_qa_v1.json)

## 目前界線與下一步

凍結的50幀RGB、160×120、整片取樣、原split與品質旗標保持不變。舊二類trainer未修改；新增介面尚未接入任何訓練排程。本輪沒有開影片、建立模型、做模型forward、更新optimizer或執行test。

下一個有限任務是**定稿單一八類baseline的訓練契約**：確認標籤入口、輸出／loss／metrics、初始化、可訓練層與BN、實體batch及累積方式、選模／停止條件，再決定是否進入獨立接線驗收。先沿用既有架構證據，不增加模組或多seed佇列；本文件不授權訓練。

[資料驗收結論](BASELINE_ACCEPTANCE.md) · [論文差異表](PAPER_INPUT_GAP.md)

後續更新（2026-11-17）：[單一八類baseline訓練規格已定稿](BASELINE_TRAIN_SPEC.md)，尚未完成模型／trainer接線，訓練維持暫停。
