# ACA clip 一致輕量空間增強：CPU 階段

此隔離套件是 aca_binary_temporal50_seed20261024_labelrev1_v1 的單一增強組合候選，尚未接入正式訓練入口。

## 固定規格

- train：原時間採樣及 resize 後，對整段 50 幀同步水平翻轉 p=0.5；brightness / contrast / saturation 各抽一個 [0.95, 1.05] 因子，無 hue。
- 固定順序：flip → brightness → contrast → saturation → clamp [0,1] → 一次 ImageNet 正規化。
- contrast 共用 brightness 後的整段灰階平均值；NumPy float64 固定 reduction 再轉 float32，避免 Torch 工作程序執行緒數造成不同結果。
- RGB 灰階係數 [0.2989,0.5870,0.1140]。這是明確固定順序的實作，不宣稱等同 torchvision 隨機順序 ColorJitter。
- SHA256 命名抽樣依版本、seed、zero-based epoch、video_id、欄位決定；不使用 Python、NumPy 或 Torch 全域 RNG。
- validation 直接委派原 TemporalDataset；沒有空間增強參數生成。
- 此版限定既有 seed20261024 與 30 輪範圍。資料、標籤、split、時間採樣、模型與最佳輪選擇規則未改。

## 介面

augment.py：確定性參數、RGB 變換及正規化。
data.py：SpatialDataset / SpatialCollator / EpochStream；正式計畫未建立，不提供訓練入口。
acceptance.py：四個已保存 train RGB fixture 的 CPU 驗收與圖片輸出。

## 有上限驗收

僅 04、16、22、27 的既存 RGB 陣列及合成輸入；不開啟影片、不接觸 validation/test 影片、不建立模型、不執行 GPU 或 optimizer。validation 證據限於合成像素逐值相同與程式直接委派檢查，並非真實 validation 影片重讀。

確認同輸入重現、worker 0/2 相同、全域 RNG 不變、整段無額外跨幀參數漂移、identity 逐值相同、錯誤 metadata 拒收，以及第15支原始/有效標籤沿用既有更正。

影像對照包括原圖、epoch0 實抽參數、全部0.95與全部1.05邊界示例；並非涵蓋所有參數組合或所有影片。水平翻轉會鏡射字幕/計分板，使用者已接受此取捨。

本階段只驗證接線，不證明準確率提升、事件必定可判或 GPU 記憶體足夠。未授權 GPU 與正式訓練。

## 第二版範圍收窄

政策版本 clip_flip_color_light_v2；抽樣命名空間仍為 clip_flip_color_light_v1，保留相同翻轉決定與色彩抽樣分位數，僅將各色彩偏離1.0的程度減半。CPU驗收額外確認這一點。舊報告與完整舊套件快照已保留。
