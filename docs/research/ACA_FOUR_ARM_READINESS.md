# ACA 四組共同協定與工程預檢（2026-11-03）

**已完成三個 seed × 四組共十二組 readiness；正式訓練與佇列尚未啟動。**
本輪依 [架構核對結論](ACA_ARCHITECTURE_AUDIT.md) 建立共同 trainer，沒有更改既有 ACA／TANet 的來源、權重或歷史結果。
這是 Make/Miss 專案上的受控消融準備，作者精確設定仍未核實。

## 已固定的比較

- baseline、LSTA-only、TSCI-only、完整 dense ACA；seed20261024／20261025／20261026。
- 共用一個更新與選模迴圈、相同 train7193／validation1547、50幀120×160 RGB完整畫面與 ImageNet normalization。
- 相同固定 ImageNet V1 骨幹與配對新分類頭；每個 block 的共用 LSTA／TSCI 也以獨立具名 RNG 初始化，三個 seed 均逐位驗證相同。
- 固定 dense gating、r16、短期k3/k3、K3、β0.25。所有 BN affine/buffers 凍結，包含新 TSCI；不加入另一支線的 L 當片 BN。
- FP32、micro1／effective16，尾批9按實際數平均；Adam lr1e-4、wd0、clip1，無 augmentation／scheduler。
- 新實驗最多30輪、patience5，最低完整 validation CE 選模，平手保留較早輪。這不是延長任何舊 run。

只用相同全域 seed 時，LSTA-only 與 ACA 的模組建立順序會造成亂數消耗不同。本輪具名 RNG 修正的是新的對照初始化方法；不覆寫舊來源，也不把舊訓練結果當成此對照的一組。

## 實際 GPU 驗收

| 組別 | 二類總參數 | 可訓練參數 | 三 seed 最高 allocated GiB | 狀態 |
| --- | ---: | ---: | ---: | --- |
| ResNet50 | 23,512,130 | 23,459,010 | 1.86 | 三 seed 全通過 |
| LSTA-only | 93,159,070 | 93,105,950 | 3.92 | 三 seed 全通過 |
| TSCI-only | 23,516,930 | 23,463,714 | 2.09 | 三 seed 全通過 |
| 完整 ACA 候選 | 93,163,870 | 93,110,654 | 4.01 | 三 seed 全通過 |

顯存為 PyTorch allocated、包含 Adam，並非整張 GPU 的總用量。每組只做一次工程更新，不能外推完整 epoch 工時或模型泛化表現。

每 seed 使用 metadata 先選定的相同4支train＋4支validation，經原本guarded decoder、RGB/normalization及雙worker正式loader。
每組以4支train各重複4次湊16次工程曝光，所有組用同順序，不是16支獨立影片或資料增強。
validation 僅測有限輸出及重載logits，沒有計算新的真實影片accuracy／CE。

十二組全部通過：

1. 與 CPU 準備的骨幹、分類頭及新增元件初始化指紋一致。
2. 可訓練參數梯度有限，各骨幹／head／LSTA／TSCI分支均有非零梯度；工程更新後對應元件有改變。
3. 全部凍結 BN 參數與 buffers 不變。
4. 真實完整尺寸工程checkpoint模型與Adam可重載，logits逐位一致。
5. 含實際小型模組與dropout的合成模型，連續與中斷續跑的模型／Adam／RNG／history逐位一致。

合成續跑驗證不是完整尺寸、完整資料集的續訓效能實驗；真實工程checkpoint與正式checkpoint使用不同schema，不能沿用工程權重起跑。

## 計數與保存

- 16項新CPU測試通過，包含尾批9對完整平均的FP64對照、最低CE／平手規則、validation中斷不重做train、跨seed/arm及工程checkpoint拒絕。
- 最終指紋下12次真實工程optimizer更新、96次合成更新；補齊環境版本前的預檢另有12／96次，保留為舊指紋歷史。因此本輪合計24次真實工程更新、192次合成更新；**正式訓練更新0次**。
- 唯一原片為4支train＋4支validation；最終指紋解碼train12／validation12／test0次，連同舊指紋歷史，本輪合計train24／validation24／test0次。
- 選定原片與ImageNet權重SHA、資料契約、共同程式指紋前後相同；既有93個ACA相關來源檔SHA不變。
- 工程checkpoint、大型來源快照與完整報告位於ignored `development/aca_four_arm_v1/artifacts/`，不納入Git。
- [可攜的十二組結果與來源SHA](results/aca_four_arm_readiness_v1.json)。沒有新validation/test分數，沒有SpaceJam下載。

## 下一步與啟動界線

十二個arm/seed已各有對應GPU報告與共同plan，可安排全新正式訓練。先前五輪pilot、延長、高解析度或TANet的結果不能混入這十二組當受控成績。
正式入口預設只列plan，必須指定arm/seed、匹配readiness及明示啟動；每個run重新載入固定ImageNet，新分類頭／模組依共同配對方式建立。
程序不會由prepare或preflight自動接著訓練。本輪沒有建立／啟動正式佇列、沒有重啟舊工作、不讀test。
後續若新增佇列監督器，應與已驗證common trainer分開保存，並核對命令與plan SHA；更動共同模型／設定／trainer來源時，既有readiness將自動失效。
