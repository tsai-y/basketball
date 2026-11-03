# TAM BatchNorm 行為對照 V1

2026-10-30 完成。使用固定16支 train 影片與三seed各自 best／last，共6份 checkpoint，直接比較保存統計與當批統計的輸出；沒有重訓或修改原權重。

[報告與圖表](artifacts/bn_behavior_v1_2q0xo24h/index.html) · [固定實驗計畫](artifacts/bn_behavior_v1_2q0xo24h/plan.json) · [精簡解讀](artifacts/bn_behavior_v1_2q0xo24h/interpretation.json) · [完整報告](artifacts/bn_behavior_v1_2q0xo24h/report.json)

## 固定範圍

從7193支 train 的metadata，以seed20261030、來源分散與SHA排序，為原始8類各抽2支，共16支、16個不同來源。計畫先存檔才讀像素，不看信心、模型預測或先前錯誤挑片；不在看到結果後換片。batch2的固定配對為同原始類別、不同來源的兩支片。

沿用原正式輸入：完整影片等距50幀、RGB、256×192、ImageNet normalization。每片只用既有guard解碼一次，RAM中保留normalized tensor，六份權重與所有條件共用相同tensor；前後SHA256核對不變。沒有讀 validation／test 原片或既有特徵快取。

所有條件均關閉 Dropout、凍結全部參數、預訓練BN2d保持eval。僅在完整載入checkpoint的記憶體模型上，暫時令指定TAM BN1d `training=True, track_running_stats=False`，使用當批統計而不更新任何running buffers；離開條件時恢復flags。使用與正式訓練相同的BF16 autocast、FP32參數與deterministic設定。

| 條件 | 實際batch | TAM BN統計來源 |
| --- | ---: | --- |
| saved_b1 | 1 | G、L皆保存統計，為同片比較基準 |
| G_batch_b1 | 1 | 只將G改為當批統計 |
| L_batch_b1 | 1 | 只將L改為當批統計 |
| GL_batch_b1 | 1 | G、L皆改為當批統計 |
| saved_b2 | 2 | 皆保存統計，用作batch-size數值控制 |
| GL_batch_b2 | 2 | G、L皆當批統計，量測固定配對的context影響 |

每份權重最後再重播第一支saved_b1，要求logits逐值相同。總計486次模型forward、582個含重播的影片forward instances；只有16支不同train影片，不能把重複比較當成582個獨立樣本。未計算準確率或CE，避免把小樣本當成果。

## 結果

下面差異皆為同片相對saved_b1的平均絕對P(Make)差，單位為百分點；翻轉是相對基準的分類改變，不代表變好或變差。

| Seed／權重 | 只切G | 只切L | 同時切G＋L | G＋L翻轉數 |
| --- | ---: | ---: | ---: | ---: |
| 20261024 best（epoch1） | 0.85 | 14.63 | 14.91 | 5／16 |
| 20261024 last（epoch6） | 1.89 | 24.17 | 24.35 | 4／16 |
| 20261025 best（epoch10） | 3.57 | 18.04 | 19.70 | 3／16 |
| 20261025 last（epoch15） | 7.06 | 21.52 | 22.45 | 3／16 |
| 20261026 best（epoch8） | 4.04 | 13.25 | 15.47 | 3／16 |
| 20261026 last（epoch13） | 9.24 | 16.80 | 19.44 | 5／16 |

六份權重均為只切L的變動大於只切G；全部TAM切換帶來14.91～24.35個百分點平均差、3～5支翻轉。這直接顯示BN統計來源在本控制條件中足以改變輸出，但未證明哪一種較準，亦未證明是validation不穩定的唯一原因。

保持保存統計時，batch1→batch2皆沒有判斷翻轉，最大P(Make)差只有0.0686個百分點；部分權重完全相同。六份權重的saved_b1重播也全部逐值相同。大幅變化遠超此數值控制所示的誤差，且Dropout並未參與。

第一個TAM的L BN，在同樣上游輸入下切統計，輸出相對L2的跨影片中位數約0.64～0.71；當片平均與保存平均的差異，以保存標準差正規化後，中位數約0.40～0.72。第一個G BN的平均差則約0.022～0.039個保存標準差。後續層會同時受到上游變化影響，不把下游大變動直接當作獨立故障定位。相對L2也受基準norm影響，CSV同時保留norm和絕對RMSE。

當批統計本身有context影響：GL batch1→batch2的平均P(Make)差為8.11～14.67個百分點。GL batch2相對saved_b1的差距在五份權重縮小、一份擴大；不足以保證加大正式micro-batch後的訓練或泛化會改善。這只是固定同類別配對下的前向診斷。

三個seed的last皆比各自best有更大GL平均差，但快照少且只有16支train片，不能推成所有epoch／來源的單調現象。

## 保存產物與驗證

- `per_video_outputs.csv`：576列條件輸出，原始兩logits、P(Make)、margin差與翻轉。
- `BN_activation_comparisons.csv`：18624列，含每次forward的32個TAM BN輸出與輸入moment差；batch2兩支片共享該batch輸入統計，不能當成兩次獨立統計觀察。
- `GL_batch_context_<seed>_<best|last>.csv`：每份權重16列batch1／batch2 context對照。
- `decoded_inputs.json`：固定16支片的guard解碼紀錄、時間座標與輸入SHA256。
- `plan.json`：樣本、配對、六份權重、條件、預算與原檔雜湊；`report.json`：結果、環境與限制。

六份記憶體模型在所有條件前後，完整參數／buffers的雜湊均不變，所有grad皆為None、flags恢復；沒有optimizer或新權重檔。原checkpoint、訓練設定、作者程式、資料清單SHA256亦未改變。

48項測試通過，含固定樣本順序／split拒絕、分支模式對照PyTorch functional BN、buffer不變、異常時flags恢復、in-place ReLU前的activation複製、近門檻翻轉與報告圖表流程。已視覺檢查兩張科學圖，沒有看新影片畫面。

入口：`probe_bn_behavior_v1.py --prepare`只寫metadata計畫；`--run <plan.json>`才執行有上限GPU前向。每個計畫只接受一次執行，不會自動放大樣本、重試OOM、重訓、續訓或讀test。

## 下一步建議

預先固定batch1與L分支統計策略，於獨立副本做完整validation診斷對照，再決定訓練設定是否需要改動。必須清楚標示這是新的推論策略候選，不是原基準成績或test結果；原模型繼續保留。這項完整validation對照尚未執行。
