# TANet validation 基準分析 V1

完成日期：2026-10-29（台灣時間）。

本輪整理既有正式訓練成果、按影片 ID 對齊主線預測，並完成固定 16 例的有限 AI 視覺初審。完整可互動材料見 [分析首頁](artifacts/analysis_v1_cfh3n8r2/index.html)、[初審與結論](artifacts/analysis_v1_cfh3n8r2/review.html)、[完整數據](artifacts/analysis_v1_cfh3n8r2/report.json) 與 [逐片對照](artifacts/analysis_v1_cfh3n8r2/paired_predictions.csv)。

## 完整 validation 結果

兩個既有系統皆使用 seed 20261025；比較同一批 1,547 支 validation（Miss 740、Make 807）。TANet 選最低 validation CE 的 epoch 10，主線選最低 validation loss 的 epoch 4。未用較高 accuracy 的其他 epoch 重新選模。

| 指標 | MixNet-I3D＋Transformer 主線 | TANet |
| --- | ---: | ---: |
| Accuracy | 75.50%（1168／1547） | 76.66%（1186／1547） |
| Macro-F1 | 0.7550 | 0.7637 |
| Miss recall | 77.57%（574／740） | 68.38%（506／740） |
| Make recall | 73.61%（594／807） | 84.26%（680／807） |
| 已保存 validation loss | BCE 0.512035 | CE 0.510853 |

混淆矩陣以資料集標籤為列、預測為欄，順序均為 Miss／Make：主線 `[[574,166],[213,594]]`；TANet `[[506,234],[127,680]]`。

| 同片對照 | Miss | Make | 合計 |
| --- | ---: | ---: | ---: |
| 兩模型皆正確 | 431 | 538 | 969 |
| 只有 TANet 正確 | 75 | 142 | 217 |
| 只有主線正確 | 143 | 56 | 199 |
| 兩模型皆錯誤 | 91 | 71 | 162 |

TANet 淨多判對 18 支（accuracy 增加約 1.16 個百分點），其中 Make 多對 86 支、Miss 少對 68 支。TANet 預測 Make 914 支，主線 760 支；這是預測行為差異，不能直接推論訓練失衡或原因。來源 v113、v149、v168 的 TANet accuracy 低於主線，其他五個來源較高；v149 僅 31 支。完整來源與出手類型分項保存在 report.json。

TANet 訓練 CE 從 0.6764 降至 0.2974，但第 10 輪後 validation CE 未再改善；第 15 輪 validation CE 為 0.5543。曲線有過擬合跡象，本輪保留 epoch 10，沒有延長訓練。

## 固定 16 例與實際初審範圍

先保存 sampling_plan.json，才讀影格。依資料集 Make／Miss × 四種兩模型判對組合，各取 2 例；固定 seed 20261029，以來源分散及 SHA256 排序選片，不依信心分數或畫面挑例，也不在看過畫面後換片。實際涵蓋八個 validation 來源各 2 例。這是分層案例分析，不能估計整體錯誤或影像現象的比例。

AI 實際看過 16 張各含 12 個固定等距原生影格的總覽、4 張各含 24 個連續原生影格的補看板、A005／A013 的全部 T50 縮放輸入板，以及 3 張數值圖。並未逐幀看完全部原片或播放全部影片；審查時已知資料標籤和模型分數，並非盲審。每例觀察及看過材料的 SHA256 在 review.json。

可觀察到小球、籃下遮擋、鏡頭跟移、近景切鏡與短暫亮幀。例如 A005 的 f70、A016 的 f39 較亮且相鄰幀一般曝光；A011 開頭由球員近景切到罰球遠景。A005／A013 的接續籃下動作仍保留標註目標未獨立確認的狀態。這些觀察沒有證明模型使用了哪一線索或造成錯誤，沒有新增人工真值、改標籤、刪片或調整裁切規則。

原片 PNG、預覽 MP4、TANet 全部 50 幀輸入與主線三個 64 幀窗口均保留供回看。T50 用同一套固定抽幀、resize、normalization 函式重建，記錄張量雜湊；未獨立重讀完整訓練 loader 或做新模型 forward。

## 範圍與後續

TANet 使用全片均勻 50 幀並訓練骨幹；主線使用三個連續 64 幀窗口與凍結特徵。兩者 optimizer、可訓練參數與預算也不同。因此這是單 seed 的兩套系統描述比較，不能把差異全歸因於 TANet 架構，也尚不能宣稱穩定優勢。validation 已參與選模，並非新的 held-out 測試。

建議下一輪固定設定補兩個 seed，再檢查均值、波動與類別召回率取捨。本輪未啟動新訓練、融合、門檻調整或 test。不要依這 16 例直接修改標籤或宣稱裁切有效。

數值分析與材料產生器：`analyze_v1.py`；測試：`tests/test_analysis_v1.py`。數值預檢可執行：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.tanet_v1.analyze_v1 --dry-run
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest discover -s development/tanet_v1/tests -v
```

不帶 `--dry-run` 會建立另一個獨立分析資料夾；AI 視覺觀察需實際檢視後另外保存，產生材料本身不等同完成初審。

本輪只開啟 16 支 validation 原片，train／test 原片開啟數皆為 0，模型 forward／optimizer steps 皆為 0。原模型、訓練程式、設定、歷史、manifest／split 以 SHA256 核對；既有特徵快取比對名稱、大小與修改時間清冊，未重新雜湊全部快取內容。
