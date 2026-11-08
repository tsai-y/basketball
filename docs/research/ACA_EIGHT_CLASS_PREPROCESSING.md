# ACA八類 D3：單一前處理契約

日期：2026-11-08。識別：`aca8_pts25_uniform50_identitycrop_v1`。

**D3已定稿為一套明示假設的重建規則，並通過17項合成工程測試。尚未驗證真實影片解碼，沒有啟用候選或訓練。**

- [機器可讀固定規格](specs/aca_eight_class_preprocessing_v1.json)
- [測試、來源SHA與時間映射例子](results/aca_eight_class_preprocessing_qa_v1.json)
- [參考實作](../../development/aca_eight_class_data_v1/preprocessing.py)、[合成工程測試](../../development/aca_eight_class_data_v1/test_preprocessing.py)
- [D1/D2品質與切分候選](ACA_EIGHT_CLASS_SPLIT_CANDIDATE.md)、[研究決策總覽](ACA_REPRODUCTION_CHECKPOINT.md)

## 1. 哪些是論文依據，哪些是重建選擇

[ACA論文§3.1與§3.3](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1471327/full)提供25FPS資料描述、T50及crop後120×160；未交代完整時間抽樣／轉碼、crop區域或增強實作。保存的補充檔亦沒有補足這些程式。本輪不以找不到細節為由持續開新候選，也不假稱下列選擇就是作者算法。

| 項目 | 本版固定決定 | 證據界線 |
| --- | --- | --- |
| 時基 | 原始顯示時間戳→25Hz虛擬格點 | 25FPS有論文描述；虛擬格點與最近幀映射是重建選擇 |
| T50 | 全片時間範圍均勻選50個格點 | T50有依據；全片端點與抽樣公式是重建選擇 |
| Crop | identity crop：完整320×240畫面 | 作者crop算法未取得；不發明依球員／籃框裁切去冒充原版 |
| 空間 | INTER_AREA縮成寬160、高120 | 尺寸有依據；插值與保留全圖為專案選擇 |
| 輸入 | RGB、float32、固定ImageNet V1 mean/std | 重用已固定權重的正規化常數；作者完整transform未核實 |
| Train / validation | 相同deterministic單一view，無augmentation | 專案選擇；不開多套crop或FPS搜尋挑分數 |

這不代表裁切缺證已由作者證實解決，而是退出缺證、固定一個可驗收版本。保留完整畫面可避免在沒有ROI依據時自行切掉投籃者或籃框；本輪沒有量測其準確率優劣。

## 2. FPS與時間抽樣的確切定義

輸入為每個解碼畫面的**顯示時間戳**，以精確有理數秒表示，按畫面顯示順序排列。未來decoder須提供並驗證frame-to-timestamp綁定；不能只由平均FPS猜逐幀時間。

FFmpeg的best_effort_timestamp是stream time_base單位的估計時間戳，見[官方AVFrame來源](https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/libavutil/frame.h)。本版使用其顯示時間語義，**沒有在本輪驗證本機decoder／ffprobe的畫面綁定**。D4須以實際解碼確認，不因合成測試通過就跳過。

令顯示時間為t_j：

```text
u_j = t_j - t_0                         # 相對原生顯示時間，嚴格遞增
M   = floor(25 * u_last) + 1            # 不晚於最後畫面的25Hz虛擬格點數
v_i = floor(i * (M-1) / 49), i=0..49    # 全片格點均勻取50個位置
q_i = v_i / 25                         # 目標時間（秒）
j_i = argmin_j |u_j - q_i|              # 最近原生畫面；同距離取較早幀
```

- 虛擬25Hz格點只是抽樣時基，不編碼新MP4、不插值合成畫面，也不把原始23.976／29.97檔案改寫成25FPS。
- 50張仍覆蓋完整約6秒片段，**不是連續50張25FPS畫面約2秒**；模型實際觀察的時間間隔由50個q_i決定，不能稱為模型輸入每秒25張。
- 精確25FPS來源會得到與舊版端點均勻取樣相同的索引；其他FPS可能改選鄰近幀。這是時基定義變更，不是提升accuracy的證據。
- PTS缺失、重複、倒序、少於50個原生幀、虛擬格點不足50或選到重複原生索引時拒絕，不循環補幀、不依label決定跳過、不偷偷換成平均FPS算法。
- 字串有理數／小數轉Fraction；不接受二進位float時間戳，避免邊界平手因浮點改變。decoder可使用integer timestamp×rational time_base精確轉換。

固定三個合成CFR例子涵蓋本地舊檢查的23.976、25與29.97。完整50個索引與時間誤差保存在QA；它們不是實際影片或模型成績。

## 3. 空間、色彩與數值

參考函式只接受已選好的50張uint8 BGR畫面，形狀為`50×240×320×3`，不接受影片路徑。依序：

1. 全畫面identity crop `[0,0,320,240)`；不做ROI、Laplacian或內容挑幀。
2. BGR→RGB，再以OpenCV INTER_AREA縮放至寬160、高120。原生與輸出都是4:3，不加letterbox、不額外伸縮長寬比。
3. `float32/255`，逐RGB通道減mean `[0.485,0.456,0.406]`、除std `[0.229,0.224,0.225]`。
4. 輸出contiguous float32 array `[3,50,120,160]`；未來adapter才轉torch並加batch維度。標籤、路徑、source group及比分不成為影像特徵。

[TorchVision ResNet50官方文件](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)可核對上述mean/std。本版**只重用常數**；沒有套用官方ImageNet推論的resize256／center crop224，也不宣稱二者完整transform相同。來源固定權重metadata仍為既有ImageNet V1，不下載或載入新權重。

train與validation使用同一函式；不做random crop、flip、色彩擾動、多view平均或自動品質篩選。未來test使用相同規則的定義已寫入契約，但這一步沒有授權decoder開test。

## 4. 拒絕條件與資料邊界

畫面大小、dtype、幀數或時間戳不符時，回報工程失敗；不靜默drop、不替換成另一支、也不改D1/D2的8230／1027／1020與34支排除定義。若真實資料有新問題，另記資料版本與原因後再決定，不在看分數後臨時改規則。

目前的參考程式不開影片、不讀manifest、不建立模型、不使用GPU，也沒有訓練入口。來源授權、test／excluded拒絕、檔案身份與畫面／PTS綁定均是D4 adapter必須補上的邊界，不能把這個純轉換函式單獨稱為完整安全loader。

本版綁定已驗收候選manifest SHA與split政策SHA；既有二類data.py、trainer、原manifest、checkpoint與每小時排程不變。候選Test來自舊Train的界線仍有效，舊checkpoint不可拿來宣稱該候選的泛化。

## 5. 驗收與下一步

17項合成工程測試通過：三種原生FPS、精確25FPS與舊索引一致、VFR顯示時間、時間原點平移、平手取較早幀、虛擬端點、短片／缺失／倒序／重複時間戳與缺幀拒絕，以及BGR/RGB、2×2區塊平均縮放、黑白正規化、輸出shape/dtype、可重做與輸入不被修改。固定環境為Python3.10.20、NumPy2.1.3、OpenCV4.13.0。

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.aca_eight_class_data_v1.test_preprocessing -v
```

這次沒有開真實影片、載入torch／模型／權重、做推論或optimizer更新，沒有test成績，也沒有宣稱前處理已提高accuracy。

**下一步D4：建立隔離的八類資料與metrics adapter，先驗證候選Train／Validation授權與實際顯示時間綁定，再驗證8類CE、8×8混淆矩陣和top-1二類映射。** 在進入正式訓練前，G1作者精確架構／BN／約26.8M的差距仍須明示，不能由D3定稿推論已取得作者原版。訓練與排程繼續暫停，不自動續跑舊八組佇列。
