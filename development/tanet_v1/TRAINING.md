# TANet 50幀基準：正式訓練入口與GPU預檢

更新：2026-10-29（台灣時間）。**本輪準備與GPU預檢已完成，正式訓練尚未啟動。**
原 `README.md` 與 `prepare.py` 保留先前CPU準備階段的紀錄；目前階段以本文件與下列GPU報告為準。

- [本輪預檢總覽](artifacts/gpu_readiness_c6pfj7gh/index.html)
- [GPU原始報告](artifacts/gpu_readiness_c6pfj7gh/report.json)
- [固定設定](train_config.json)
- [正式入口](train.py)

## 研究問題與固定輸入

先建立使用作者原版TAM模組的TANet二分類基準，不加入Triplet Attention、ACA門控、YOLO、RL或融合。
TAM原作者程式、commit、授權與既有CPU資料適配器均未改動。
這是專題適配版本，不宣稱重現作者完整訓練配方或ACA-Net原版。

從完整影片取 `floor(i*(N-1)/49)`，i=0…49，含首尾、50個不同影格、保持順序。
全畫面320×240 BGR轉RGB、縮放為寬256×高192、除255、ImageNet mean/std一次。
輸入 `[B,3,50,192,256]`，兩個原始logits `[B,2]`，Miss=0、Make=1；用CrossEntropyLoss。
不接時間選段器、不重複幀、不補黑幀、不做隨機增強，影片不足50幀則拒絕。
目前解碼器仍順序核對原片全部影格，不是只解碼50幀。

本版從本機SHA固定的ResNet50 ImageNet V1權重初始化骨幹，TAM及二分類頭全新初始化；
不使用8/16幀的影片預訓練權重，也不使用任何工程預檢、ACA或主線訓練權重。
所有模型參數均可訓練，共24,859,490個；骨幹權重並未凍結。

原主線取三個連續64幀窗口，TANet取全片等距50幀，因此兩者比較包含架構及時間抽樣差異，
不是只改骨幹的單變因消融。第一輪單一seed是建立基準，不代表最佳幀數、最佳模型或穩定優勢。

## 正式訓練規則

|項目|固定設定|
|---|---|
|Train／validation|7193／1547支，沿用來源分離的固定manifest與split|
|Seed|20261025|
|Micro-batch／有效batch|1／16，梯度累積16支後更新一次|
|尾批|每輪450次更新，最後9支以實際9支的summed CE取平均，不除16|
|抽樣|train每輪seed+epoch洗牌，無放回、全部看一次；不依類別平衡或品質降權|
|Optimizer|Adam，lr=1e-4、weight decay=1e-4、無scheduler|
|Gradient clipping|全模型norm上限5.0，非有限值拒絕更新|
|BN|預訓練BatchNorm2d固定running stats、affine仍訓練；新TAM BatchNorm1d正常更新|
|Dropout|0.5|
|數值政策|CUDA BF16 autocast、參數FP32、CE轉FP32；無GradScaler、無TF32|
|可重現性|deterministic algorithms、CUBLAS `:4096:8`、cuDNN benchmark關閉|
|上限／早停|15輪；完整validation CE，patience=5、min_delta=1e-4|
|最佳模型|實際最低完整validation CE；微小改善可更新best但不一定重置早停|
|Validation|每輪全1547支固定順序，只前向，不更新權重或BN|
|指標|CE、accuracy、兩類macro-F1、各類F1與混淆矩陣；不把Make F1當macro-F1|
|CPU／解碼|4個torch threads、OpenCV 1 thread、DataLoader workers=0|
|Test|無test入口；原片存取前拒絕test／未知split|

這些是事前固定的第一版專題設定；不是依本輪兩支影片的loss選出來的超參數。
兩支預檢影片不是效能評估集，工程loss下降不可當成泛化改善。

## 已完成的驗證

22項CPU／合成資料測試全部通過，含先前12項契約測試及新增10項：
實際9支尾批的梯度權重、macro-F1、非有限值拒絕、BN政策、最低loss／早停分離、
Adam／Dropout RNG精確續跑、工程權重拒絕、入口模式／dry-run零更新，以及正式入口的合成資料完整兩輪與中斷續訓。

GPU使用兩支既有train影片、不同來源、一支Miss一支Make：

- 固定做三次工程optimizer更新，每次16個microsteps，共48個microsteps；
  是第一步、連續第二步、以及重載第一步後重播第二步，邏輯上只走兩步。
- 骨幹、TAM、分類頭皆有有限梯度並更新；預訓練BN2d buffers逐值不變。
- 續訓第二步的logits、loss、全部模型／BN state及Adam state均與連續路徑逐值相同。
- 評估前向不改任何模型參數或buffers；未計兩支影片的分類準確率。
- 只開啟兩支train原片；validation/test原片開啟次數皆為0。
- RTX5070Ti、PyTorch2.11.0+cu128；peak allocated約3.07GiB、peak reserved約3.47GiB。
  數字不含其他程式占用或所有CUDA額外配置。
- GPU訓練microstep中位約0.077秒；兩支原片首次解碼約0.081／0.057秒。

從這兩支影片外推：每輪約20.2分鐘，15輪約5.05小時。
這不是全資料的一輪實測；解碼快取、不同影片長度、存檔、完整validation彙總及其他工作會改變工時，
正式啟動時應預留更久；早停也可能提早結束。

工程產物 `engineering_step1.pt` 只供續訓驗收。其kind是`engineering_only`，
正式入口拒絕載入這種權重；全新正式run會重新從固定ImageNet初始化。
第一次工程報告`gpu_readiness_ad7gb9b5`仍保留，最終來源雜湊覆蓋補齊後以`gpu_readiness_c6pfj7gh`為準。

## 可執行命令

以下在 `/home/tsai/projects/basketball`、既有yolo11環境執行，不需安裝套件。

只檢查固定split／程式／設定，不開影片、不建立CUDA context或optimizer、不寫run：

```bash
python -B -m development.tanet_v1.train --dry-run
```

重新做短程GPU工程預檢（另建獨立目錄，仍不啟動正式訓練）：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B -m development.tanet_v1.readiness --run
```

本輪沒有執行下列正式啟動命令。之後明確決定啟動時才執行：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B -m development.tanet_v1.train --start --readiness-report development/tanet_v1/artifacts/gpu_readiness_c6pfj7gh/report.json
```

正式入口必須同時提供`--start`及通過的預檢報告，核對來源、資料、設定、torch/CUDA與GPU。
不帶模式會拒絕，不會自動接續CPU準備或排程啟動。
改動已固定的訓練來源或設定後，舊GPU報告會失效，需要先重做預檢。

正式run獨立存於`artifacts/formal_train_*/`：
`contract.json`、`progress.log`、`history.json`、`run_state.json`、`best.pt`、`last.pt`。
checkpoint含optimizer、Python／NumPy／CPU／CUDA RNG、epoch、global step及早停狀態，採原子存檔。

中斷後，先確認原訓練程序已結束，再將下例中的run目錄換成實際路徑：

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B -m development.tanet_v1.train --start --readiness-report development/tanet_v1/artifacts/gpu_readiness_c6pfj7gh/report.json --resume development/tanet_v1/artifacts/formal_train_實際目錄/last.pt
```

只支援完整epoch邊界續跑；中途未完成的一輪從上一輪last.pt重新跑，並非microstep接續。
只接受原formal run的last.pt、相同程式／設定／環境；已跑滿或早停的run不會偷偷延長。
正常例外與Ctrl-C會清除active.lock；若是強制終止／斷電留下鎖檔，需先核對程序確實結束，再處理殘留鎖檔。

## 技術依據

- [作者TAM固定來源](https://github.com/liu-zhy/temporal-adaptive-module/tree/5fcb46fe671ead2f7958d8ffc3321a0e274d6491)。
- [PyTorch2.11 AMP](https://docs.pytorch.org/docs/2.11/amp.html)：本版用autocast，不將整個模型轉成BF16。
- [PyTorch可重現性](https://docs.pytorch.org/docs/2.11/notes/randomness.html)：本輪同環境驗證逐值續跑，不保證跨硬體或跨版本相同。
- [PyTorch序列化](https://docs.pytorch.org/docs/2.11/notes/serialization.html)：以state_dict與weights_only載入自有checkpoint。
