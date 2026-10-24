# 八類平衡抽樣實驗 v1

目的：測試 `shot type × Make/Miss` 的訓練頻率是否影響罰球 Miss 等弱項。
沿用 frozen MixNet-I3D 特徵與原本 WindowTemporalTransformer，不加入 YOLO、RL 或品質降權。

## 唯一訓練變因

- 原版：每輪 shuffle 全部 7,193 支 train，各看一次。
- 本版：每支影片抽樣權重 `1 / 該 original_class 的 train 數量`。
- 八類為 `2p0, 2p1, 3p0, 3p1, ft0, ft1, mp0, mp1`，每類預期抽中機率 12.5%。
- 每輪仍抽 7,193 次，採有放回抽樣；部分影片重複、部分當輪未抽中。並非每輪或每個 batch 強制等量。
- 不做 loss class weighting／pos_weight，避免同時更動兩種平衡機制。
- validation 保留原始分布，全部 1,547 支每輪評估一次；不讀 test 影片或特徵。
- 原版模型、學習率、batch size、CPU threads、50 輪上限、早停及選模規則不變。

固定 seeds：20261024、20261025、20261026。以最低整體 validation loss 選 checkpoint，
不是挑最高 accuracy 或最高罰球 recall；同時報告三次結果，不挑最好的一次。

這是依 validation 弱項提出的探索性實驗，不能當作獨立 test 效能或因果證明。
若弱類 recall 上升但整體準確率下降，應如實報告取捨，不直接取代原版。

## 執行（專案根目錄，yolo11 環境）

```bash
python -B -m unittest discover -s development/temporal_balanced_v1/tests -p 'test_*.py' -v
python -B -m development.temporal_balanced_v1.train_balanced --dry-run
python -B -m development.temporal_balanced_v1.train_balanced --epochs 50 --seed 20261024
python -B -m development.temporal_balanced_v1.train_balanced --epochs 50 --seed 20261025
python -B -m development.temporal_balanced_v1.train_balanced --epochs 50 --seed 20261026
python -B -m development.temporal_balanced_v1.compare
```

dry-run 讀完整 train/validation 快取、檢查身分與形狀、做 forward 並預覽抽樣，
不建立 optimizer、不寫 run、不更新權重。預覽會還原抽樣 RNG，不影響正式第一輪。

## 輸出

獨立 `artifacts/balanced_seed<seed>_<random>/`，不覆蓋原版：

- `config.json`：設定、原始八類數量、manifest 與程式 SHA-256。
- `history.csv`：每輪 sampled train loss 與原分布 validation 指標。
- `sampling_counts.csv`：每輪八類實際抽取次數、總次數、不同影片數。
- `best.pt`：最低 validation loss 的分類頭，不包含或更新 frozen backbone。
- `summary.json`：整體指標與 2p/3p/ft/mp 診斷。
- `validation_predictions.csv`：逐影片機率與預測，閾值固定 0.5。

`compare` 核對三組 seed 的設定／原版來源雜湊、選模、validation ID／標籤與
sampling audit，產出獨立 comparison JSON。若有重跑造成多組候選，必須明確指定
`--balanced-runs <run24> <run25> <run26>`，不自動挑分數最高者。

注意：sampled train loss 的分布與原版不同，不宜直接以兩版 train loss 比較優劣。

## 首輪結果（2026-10-24）

三個固定 seed、各自最低 validation loss checkpoint 的平均 ± 樣本標準差；
下表單位為百分比，標準差不是信賴區間。

| 指標 | 原版 shuffle | 八類平衡抽樣 |
| --- | --- | --- |
| 整體 accuracy | 76.84 ± 1.24 | 73.65 ± 2.59 |
| 整體 macro-F1 | 76.81 ± 1.22 | 73.33 ± 2.53 |
| 罰球 Miss recall | 30.00 ± 5.77 | 39.63 ± 9.71 |
| 罰球 Make recall | 85.44 ± 4.41 | 77.14 ± 7.40 |
| 罰球 accuracy | 71.84 ± 2.05 | 67.94 ± 3.33 |

平均 validation loss：原版 0.51125，平衡版 0.55449。
平均整體 accuracy 下降約 3.19 個百分點；三個 seed 中兩次下降、一次上升。
罰球 Miss recall 的上升也不是三次一致。這輪不以平衡版取代原版；
不能據此斷言其他平衡方法都無效，或已證明瓶頸一定在 backbone。

本輪八類抽樣、快取、設定一致性、最低 loss 選模及逐影片預測核對通過，
完整結果：[comparison.json](artifacts/comparison_m_btc99y/comparison.json)。
原版所有結果保留，test 未使用。
