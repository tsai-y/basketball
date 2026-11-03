# ACA 架構核對：接續入口

本分支從已保存既有候選實驗的 `codex/aca-net-reproduction` 建立。
本次只新增接續範圍與 Git 分工，沒有修改模型、重訓、下載 SpaceJam 或執行 test。
既有候選與完整執行狀態見 [ACA 實驗索引](aca_experiments.md)。

## 核對目標

建立「論文規格／既有實作／外部 TAM→ACA 提案」三方對照，區分確定結構、重建假設與改良版。
以 [ACA-Net 原論文](https://doi.org/10.3389/fnbot.2024.1471327) §2、Figures 1–3 及實驗設定為依據；尚無已核實作者程式的部分不得標為精確復刻。

| 待核對事項 | 現有證據與界線 |
| --- | --- |
| Dense／depthwise 3D gating | 既有 dense 二類 93,163,870、DW3D 二類 25,464,286，均未核實為論文約 26.8M 的設定 |
| Gating Sigmoid、IN、bias、padding | 分別記錄圖、公式與候選選擇，不能只依參數量反推唯一結構 |
| Short branch 的 r／kernel／BN | 分清 TANet 作者原碼與 ACA 候選選擇，不能自動等同 |
| Global branch 的張量與 BN | TANet 官方先 reshape `[N*C,T]`，BN 位於 Linear 後的 hidden features；外部提案的通道 BN 不等價 |
| TSCI 三分支與融合位置 | 核對逐幀 C/H/W 互動、逆向 permute、三路平均，以及 LSTA/TSCI 並聯相加 |
| 訓練與評估協定 | 二類與論文八類分開；TANet L 當片統計為探索性改良，不自動套用為 ACA 原版設定 |

## 完成核對後才決定的實驗

1. 明確選定有證據支持的重建候選，公開尚未解決的差異。
2. 完成維度、數值、梯度、初始化、BN 與完整輸入 GPU 工程驗收。
3. 事前固定相同資料與訓練協定，規劃 ResNet50、LSTA-only、TSCI-only、完整 ACA 四組與三 seed。
4. SpaceJam 另立資料版本與十類分類頭；先核對資料筆數及 16／8 幀差異。

以上是後續工作範圍，不代表已授權自動啟動訓練或再次 test。
既有 test 使用歷史保留，不得稱為全專案未見 holdout；此次整理不根據 test 成績修改設定。

## Git 操作界線

- 原 YOLO、TANet 與 ACA 候選的程式已在各自分支保存，這裡只接續 ACA 架構核對。
- 不在架構核對時順便合併其他研究分支、重寫歷史、移動舊 artifact 或重構 seed 快照。
- 新設定與修正另作 commit；保留舊候選可重現的來源及結果。
