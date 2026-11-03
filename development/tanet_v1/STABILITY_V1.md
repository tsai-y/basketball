# TANet 訓練穩定性診斷 V1

2026-10-30 完成。這輪只讀現有 checkpoint／預測，不重新推論、重訓、校正 BN 或改動權重。

[診斷報告與圖表](artifacts/stability_v1_km0d6q26/index.html) · [完整數據](artifacts/stability_v1_km0d6q26/report.json) · [結論與逐片分解](artifacts/stability_v1_km0d6q26/interpretation.json)

## 證據範圍

三個 seed 各自 best.pt 與 last.pt，共 6 份正式 checkpoint；每份完整 1547 支 validation logits、資料集標籤及影片 ID。另加入相同 seed 的主線既有 CSV 預測。共有 13923 筆預測列，但始終是同一批 1547 支影片重複比較，不能當成 13923 個獨立樣本。

先保存診斷規則：信心 90%／95% 門檻、ECE10 的十個等寬箱、NLL 最高 1% 的貢獻、每份預測的最高 20 筆 NLL、BN 方差 1e-8／1e-5 描述門檻。沒有依結果調分類門檻、挑 seed、擬合 temperature、刪除影片或更動標籤。

TANet 使用保存的 FP32 logits 計算 FP64 log_softmax，逐片 NLL 不受 softmax 機率飽和為 0／1 影響；另重算原 FP32 CE 並核對保存值。主線未保存 logits，逐片 NLL 只能從保存機率近似，裁切至 `[1e-15,1-1e-15]`；保留原 BCE 為正式 loss，不宣稱近似值完全等價。

只有 best／last 保留逐片資料與 BN 狀態，中間輪次只有整體 history。seed 20261024 第 4 輪 CE=1.6773 的逐片原因，不能由本輪直接重建。

## 已確認的現象

| Seed | best→last epoch | CE | 信心≥90%中的錯誤／總數 | ECE10 |
| --- | --- | --- | --- | --- |
| 20261024 | 1→6 | 0.7668→1.0326 | 41／194→167／798 | 0.0583→0.1447 |
| 20261025 | 10→15 | 0.5109→0.5543 | 8／131→18／241 | 0.0804→0.0412 |
| 20261026 | 8→13 | 0.5489→0.5869 | 27／302→46／462 | 0.0255→0.0526 |

三組後期高信心錯誤支數都增加，但總高信心預測數也增加，不能只用錯誤數宣稱條件錯誤率惡化。例如 seed 20261024 高信心組錯誤率約 21.13%→20.93%，基本相近；它的所有錯判平均信心則從 63.97%升至80.76%。

同一 seed 20261024 的 326 支影片由錯轉對、242 支由對轉錯，accuracy 淨多對84支（66.58%→72.01%）。然而後者最後一輪平均 NLL=3.836，佔最後總 NLL 約58.11%，足以抵銷其他影片的改善。這解釋保存快照中 accuracy 上升、CE 卻上升的數值組成，並非對訓練原因的證明。

seed 20261024 最後一輪的 v113／v168 合計佔49.34%的總 NLL，僅佔27.73%的影片。它們是優先查看的來源現象，不構成資料錯誤、重標或排除來源的理由；v149 只有31支，不能由其高 accuracy 推廣至整體。

三個 best 的預測 Make 數為576／914／758，真實資料集 Make 數為807。best→last 分別變成974／1127／625，方向不一致。來源、類別及出手類型的完整分項保留在 report.json。

ECE 並不能解釋所有 seed 的能力下降：20261025 的 ECE10 變小，但 accuracy 由76.66%降到70.01%。不應把校正信心當成分類能力必然改善，也不能直接為了提高 accuracy 改用另一套事後選模規則。

三個 best：全部判對768支、兩個seed判對385支、一個seed判對252支、全部判錯142支。這些對錯仍依既有資料集標籤，沒有新增人工真值或視覺初審。

## BatchNorm 檢查

每份 checkpoint 有32個 TAM BN1d（G／L 各16）；每份共2544個 running variance 元素。全部均為有限值、非負方差，num_batches_tracked 精確等於 epoch×7193。原 ResNet50 的53個 BN2d、159個 running buffers 均與初始化逐值一致，沒有發現原本固定 running stats 的政策失效。

固定 ImageNet 檔是267個 tensor 的舊版權重，不包含53個 num_batches_tracked；torchvision 標準載入會把缺少的 counters 初始化為0。本輪比較106個原始 mean／variance tensors 加53個初始化零 counters，沒有將缺少 counters 誤報為權重損毀。

| Seed | best→last：variance ≤1e-8 元素數 | best→last：variance ≤1e-5 元素數 |
| --- | ---: | ---: |
| 20261024 | 0→0 | 0→0 |
| 20261025 | 7→21 | 50→94 |
| 20261026 | 0→0 | 1→3 |

較多接近零方差出現在表現最好的20261025，最差20261024反而沒有。這不足以證明小方差無害，也不足以證明 BN 是根因。報告另保存 `abs(gamma)/sqrt(running_var+1e-5)` 的分布與同 seed best→last 統計變化；這些量不包含實際 activation，不能證明輸出爆炸或 train/eval 落差。

作者 TAM 的 G 分支 BN 從 `[B*C,2*T]` 取樣，L 分支從 `[B,C/4,T]` 取樣。因此 micro-batch1 不代表 BN 僅有一個 scalar。另一方面，梯度累積16也不會將16支影片的 BN 統計合併成一個 batch。不同 seed 的隨機 TAM 通道並未對齊，本輪僅比較分布摘要，沒有逐通道聲稱對應相同特徵。

## 下一步與驗證

目前有高信心錯誤、類別偏向與來源差異的直接證據；**BN train/eval 行為落差仍是待驗證假說**。下一步建議固定 train 小樣本，在記憶體副本上關閉 Dropout，做 BN 行為對照；確認差距後再決定是否改實際 micro-batch 或其他訓練設定。這輪沒有啟動該對照，也沒有直接加入 Triplet Attention。

產生器 `diagnose_stability_v1.py` 與 `tests/test_diagnose_stability_v1.py`；42項測試通過，含數值飽和時的穩定 NLL、ID／split 拒絕、固定分箱邊界、BN 計數／非法數值、舊版初始化 counter 與圖表資料流程。

```bash
MPLCONFIGDIR=/tmp/basketball-tanet-diagnose-mpl /home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.tanet_v1.diagnose_stability_v1 --run
```

每次建立獨立資料夾；中文解讀與 best→last 分解是本輪結果整理，另存 interpretation.json。檢查過三張圖表，沒有看新影格。train／validation／test 原片開啟皆0、CUDA context未初始化、模型forward／optimizer steps皆0。原 checkpoint／設定／訓練來源／資料清單以 SHA256 核對保持不變。

首個產物 stability_v1_75y5pmli 保留為圖表欄位錯誤的失敗紀錄；修正後的完整結果是 stability_v1_km0d6q26，不使用失敗目錄作為交付。
