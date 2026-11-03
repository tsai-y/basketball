# TANet L 當片統計 v2：三 seed 正式訓練

此分支在每輪完整 validation 使用同一個固定規則，再以最低 validation CE 選擇
該 seed 的 best.pt。TAM 的 L BatchNorm1d 用單支影片的時間觀測計算統計，
`training=True, track_running_stats=False`，不更新 mean、variance 或 counter；
G、預訓練 BN2d 使用已存統計，dropout 關閉。推論同樣必須 batch=1。
這是評估規則的修正，不是新增線條、拉普拉斯或影片裁切演算法。

訓練階段沿用 v1：7193 train／1547 validation、全片均勻50幀、192×256 RGB、
ImageNet normalization、micro batch=1、累積16（尾組9）、Adam 1e-4、weight decay
1e-4、BF16、15輪上限、patience=5、min_delta=1e-4。預訓練 BN2d 統計固定、
affine 可訓練，兩個 TAM 分支照常訓練。每輪450 optimizer updates。

20261024 → 20261025 → 20261026 依序執行。每個 seed 都先跑兩支 train 影片的 GPU
工程預檢，再從同一份 SHA 固定的 ImageNet ResNet50 初始化、新 TAM 與分類頭正式訓練。
預檢權重不作正式初始化。不得沿用舊 best／last checkpoint 或挑選較好的 seed。
原版程式、作者原碼、三 seed 權重、主線及診斷結果以 SHA 固定並保持原狀。

`campaign.py --prepare` 建立不可覆寫的 seed24／26 複本、plan.json 與 prepared state；
`--run <campaign>` 明確啟動序列監督程序。state.json 保留每個子程序 PID、
process_start_ticks、boot_id、命令列、日誌和輸出目錄；launcher.json 記錄監督程序。
任一預檢或訓練失敗便停止，不自動重試、續訓或延長輪數。正式訓練可用原有入口
明確續訓，但監督程序不自動做這件事。

程式檢查驗證與前一輪 L 診斷逐值一致、模型所有參數與 buffer 不變、RNG 不變、
旗標還原，以及工程 checkpoint 的 Adam／RNG／模型連續與續訓結果逐值一致。
CPU 測試另外覆蓋 batch1 契約、例外還原、訓練更新與 v1 完全一致、epoch 選模、
中斷續訓、三 seed 順序與失敗後不繼續。

完成三組後 summary.json／index.html 從各 seed 的已存 validation logits 重算指標，
報告逐 seed、平均、樣本標準差、Miss／Make recall 與主線及原 TANet 對照。
本規則是在檢視同一 validation 後提出；結果屬探索性比較，並非未見資料估計。
只使用 train／validation；不執行 test、不改門檻、標籤、輸入或資料分割。

作者 TAM 原碼來自 Apache-2.0 的
https://github.com/liu-zhy/temporal-adaptive-module ，固定 commit
`5fcb46fe671ead2f7958d8ffc3321a0e274d6491`；新評估包裝、訓練入口與監督程序為本專案程式。
這不是作者完整實驗配方或 ACA-Net 的重現。
