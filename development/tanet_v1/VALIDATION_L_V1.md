# TAM L分支完整 validation 對照 V1

2026-10-30 完成。固定 batch1，只有 TAM L BN 改用當片統計；G／預訓練BN2d維持保存統計、Dropout關閉，六份checkpoint的參數與buffers均不變。

[報告與圖表](artifacts/validation_L_v1_40caaq7i/index.html) · [固定計畫](artifacts/validation_L_v1_40caaq7i/plan.json) · [完整數據](artifacts/validation_L_v1_40caaq7i/report.json) · [精簡解讀](artifacts/validation_L_v1_40caaq7i/interpretation.json)

## 預先固定的範圍

本輪策略來自前一輪固定train16的BN行為診斷。只評估保存統計與L當片統計兩種條件；沒有測G／GL、更多batch sizes、temperature、分類門檻或新超參數。三個seed各自原best與last皆完整評估，原best仍取原validation CE所選的epoch1／10／8，不因新分數挑另一個epoch或seed。last只用於輔助診斷。

沿用原完整影片50幀、RGB256×192、相同正規化與BF16／FP32數值規則，完整1547支validation按相同ID順序、每片僅解碼一次，同一張量供六份權重兩種條件使用。總18564次forward；六份原策略共9282筆logits向量全部與保存結果逐值相同。若任何一筆不吻合，入口會停止而不把差異解釋為新策略效果。

TAM L以`training=True, track_running_stats=False`在記憶體副本上取得當支影片的統計，離開條件即恢復flags；沒有更新running mean／variance／counter。固定batch1，統計不跨影片聚合，不使用validation標籤建立統計、擬合校正或改參數。模型全部參數requires_grad=False，沒有optimizer。

## 固定快照結果

| Seed／快照 | 原epoch | 原CE→L CE | 原Accuracy→L Accuracy | 原Macro-F1→L Macro-F1 | 淨多判對 |
| --- | ---: | --- | --- | --- | ---: |
| 20261024 best | 1 | 0.7668→0.5986 | 66.58%→68.33% | 0.6620→0.6595 | +27 |
| 20261024 last | 6 | 1.0326→0.3868 | 72.01%→82.42% | 0.7135→0.8237 | +161 |
| 20261025 best | 10 | 0.5109→0.4526 | 76.66%→79.70% | 0.7637→0.7945 | +47 |
| 20261025 last | 15 | 0.5543→0.5030 | 70.01%→77.25% | 0.6800→0.7666 | +112 |
| 20261026 best | 8 | 0.5489→0.5046 | 71.75%→76.08% | 0.7175→0.7593 | +67 |
| 20261026 last | 13 | 0.5869→0.5191 | 70.01%→76.02% | 0.6984→0.7602 | +93 |

## 解讀

原策略共9282筆兩logits向量全部逐值重現checkpoint保存結果；每支影片只解碼一次、固定batch1。L條件只改BN統計來源，其他設定相同。

保留原best epoch後，三seed accuracy平均從71.67% ± 5.04pp變為74.70% ± 5.81pp（平均差+3.04pp）；macro-F1從0.7144 ± 0.0509變為0.7377 ± 0.0700。±為跨seed樣本標準差，不是信賴區間。

三個原best中，accuracy改善3/3、macro-F1改善2/3、CE下降3/3；平均CE從0.6088變為0.5186。

三seed平均Miss recall：74.28%→62.21%；Make recall：69.27%→86.16%。完整類別、來源、出手類型和信心分數取捨均保留在report.json。

歷史主線三seed accuracy平均為76.84% ± 1.24pp；本輪沒有重跑主線，也沒有把兩系統設定差異當成純架構效果。

last權重的結果僅用於診斷；沒有事後重選best、挑最高分seed或宣稱last是新最佳epoch。新策略若要成為正式基準，應在每一輪選模時事先使用同一套推論規則。

選模不一致在20261024很明顯：原best epoch1使用L後為68.33%／CE0.5986；原last epoch6使用L後為82.42%／CE0.3868。這支持原選模與新推論規則需要一起檢驗，不能直接從僅有的兩份快照選出新最佳模型。

穩定性尚未改善：原best的accuracy跨seed樣本標準差由5.04增為5.81個百分點，macro-F1標準差由0.0509增為0.0700；20261024的macro-F1略降，Miss recall由80.68%降至43.78%。不能只因平均accuracy上升便宣稱問題已解決。

這是新的推論策略在既有validation上的對照。batch1的L統計只來自當支影片時間維度，未跨validation影片聚合，未更新running buffers或擬合參數／門檻／temperature。

共1547支不同影片、18564筆含seed／快照／條件重複的輸出；重複輸出不能當成18564個獨立樣本。原validation已參與過選模與研究，這不是test或新的獨立holdout。

六份記憶體模型的完整參數與buffers雜湊均不變，flags已恢復、grad皆None；原權重／程式／資料清單SHA256也未改變。train和test原片開啟數皆0。

保留此策略為候選。下一步建議事先固定L當片統計的validation規則，在同一組三seed中每輪都用此規則驗證／選模，重新檢驗早停、類別取捨與跨seed波動；訓練forward本來就使用L當批統計，其餘訓練設定可保持一致。本輪不替換原模型、不從兩份快照重選，也未啟動新訓練或追加BN策略搜尋。

## 保存產物與驗證

`per_video_outputs.csv`含18564列logits、機率、NLL與來源；六份`paired_seed<seed>_<best|last>.csv`各1547列，保存判對組合與淨改善。`decoded_inputs.json`保留1547次guard解碼、抽幀座標與normalized input SHA256。CE以原FP32定義計算，逐片NLL以FP64 log_softmax計算，避免機率飽和；accuracy、macro-F1、兩類recall、混淆矩陣、ECE10、Brier與來源／出手類型皆保留。

53項CPU／合成測試通過，含完整報告圖表資料流程、逐ID配對、嚴格baseline replay、固定L／batch1／checkpoint範圍拒絕，以及副本參數與buffers不變。原checkpoint、訓練來源、設定與資料清單以SHA256核對不變；沒有新權重檔、重訓或test。

入口為`validate_L_bn_v1.py --prepare`（僅metadata計畫），接著`--run <plan.json>`（有上限GPU前向）；每份計畫僅接受一次執行，沒有自動重啟、續訓或擴大驗證範圍。
