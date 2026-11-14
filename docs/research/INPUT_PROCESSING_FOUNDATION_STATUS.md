# 資料處理基礎流程：第2步暫停待決策

日期：2026-11-15。依使用者「一路執行，有問題停止提問」要求，目前停在單支影片視覺核對；資料處理尚未完成。

## 已完成

- 第1步：保留原來源分組與split，套用既定八類歧義隔離規則；唯讀audit清單為7192 train／1547 validation／1538 test／34 excluded。原manifest、原標籤與split assignment未覆寫。
- 16支代表樣本在看片前固定，每類2支train，16個不同來源群組，以固定metadata hash順序選取；尚未展開16支檢視。
- 第一支為train的 `2p0/2p0_v182_002546_x264.mp4`：146幀、23.976FPS，完整按原生索引均勻取50幀；原畫面重播與實際已訓練baseline decoder輸出逐位元一致。
- 正規化輸入為contiguous CPU float32 `[3,50,120,160]`，有限值、RGB與縮放流程正確。標籤只確認資料欄位及檔名映射，未宣稱人工重新核定真值。

## 發現的問題與停止點

球員動作及場地背景仍可見，但160×120輸入的球與籃框細節難以肉眼確認。這是單支來源的視覺觀察，不能推論整體資料都不可用、模型瓶頸已定位或提高解析度一定有效。

依使用者指示暫停於第2步：未做16支、未做全量、未調解析度／ROI／取幀，也未啟用D3虛擬25Hz或80/10/10候選。下一步需使用者選擇：先維持基準完成固定16支檢視（建議），或先做原320×240畫面的影像對照。兩者都不訓練。

## 具體對照與紀錄

- [單支原畫面與輸入並排](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/single_comparison.png)
- [關鍵畫面：原320×240與輸入160×120放大2倍](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/frame_24_comparison.png)
- [全部50幀實際輸入](../../development/aca_eight_class_data_v1/artifacts/input_foundation_20261115_v1/single_contact_input.png)
- [完整機器紀錄](results/input_processing_foundation_progress_v1.json)

圖片放大只供檢視，不增加模型輸入資訊。完整metadata與圖片在ignored artifacts；本次只開1支train影片，因擷取原圖及核對精確baseline而解碼2次，validation／test影片均為0。

模型建立／推論／optimizer更新／訓練均為0，51個歷史受保護檔案未變，原權重與排程不變；每小時排程仍PAUSED。這是等待輸入處理決策，不是程式崩潰、正式成績或七步資料流程完成。
