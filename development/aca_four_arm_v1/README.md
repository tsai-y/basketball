# ACA 四組共同協定 v1

本模組只新增共同實驗設定、配對初始化、四組共用訓練迴圈與有界 readiness。
不更動既有 ACA/TANet 來源或 run，不自動建立正式訓練佇列，也沒有 test 介面。
這是專案 Make/Miss 二分類的受控消融候選，不是已核實的作者 ACA-Net。

## 固定設定

- 四組：`baseline / lsta / tsci / aca`，三 seed：20261024、20261025、20261026。
- 固定 train7193／validation1547，同 source-group split；全片均勻50幀、RGB120×160、ImageNet normalization。
- ImageNet V1 骨幹、全新配對分類頭；每個 block 的 LSTA／TSCI 使用獨立具名 RNG，所以重疊模組逐位相同。
- dense gating、r16、短期k3/k3、K3、β0.25，沿用明列的依圖候選，不以參數量湊作者設定。
- 所有 BN affine／running buffers 凍結（包含新 TSCI）；IN 維持既有無 running stats 設定，不加入 TANet L 當片 BN。
- FP32，micro1／effective16，尾批9按實際數平均；Adam lr1e-4、wd0、clip1；無 scheduler 或 augmentation。
- 最多30輪、patience5，最低完整 validation CE 選模，平手保留較早輪。30輪來自已保存共同協定草案，並非延長歷史run。

不同 arm 的新增模組原本會消耗不同亂數。只把全域 seed 設一樣不足以配對 LSTA／TSCI；本模組的具名 block/component RNG 是明列的新受控初始化，不能把舊候選成績直接拿來充當這次消融。

## 操作與範圍

以下在 ACA 工作目錄執行；不帶動作旗標只列計畫，不訓練、不解碼。

```bash
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.aca_four_arm_v1.test_four_arm
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_four_arm_v1.train --source-root /home/tsai/projects/basketball --prepare
```

CPU prepare 核對三 seed 的十二組初始化，保存 source/data/weight/runtime 指紋、plan 與來源快照；不解碼原片、不開 CUDA。
`--preflight --preparation /ABSOLUTE/cpu_prepare_RUN --seed SEED` 對指定 seed 的四組依序做 GPU readiness。不同 seed 需自己的報告，舊版或其他 seed 的 readiness 不能解鎖正式入口。

GPU readiness：固定 metadata 選出的4 train＋4 validation，一次循序解碼並使用正式雙 worker loader；四組共用相同快取。每組只做一次16次影片曝光的工程更新（4支train各重複4次），不是擴增獨立訓練資料。validation 只驗有限輸出與checkpoint logits，不計真實影片準確率或新validation CE。
另在含實際小型模組與 dropout 的合成模型上測中斷續跑、Adam／RNG／history 逐位一致；此合成更新與真實工程更新分開計數。

工程 checkpoint 有獨立 schema，正式入口拒絕沿用；正式訓練必須重新從固定 ImageNet 初始化。
CPU/meta、未通過報告、跨 seed／arm／source 的報告不能解鎖。GPU 有其他非桌面工作時拒絕，不終止其他程序；失敗保留failure.json，不自動重試。

正式入口為 `--start-training --variant ARM --seed SEED --readiness-report /ABSOLUTE/REPORT --accept-candidate-architecture`。
這只是入口說明，本輪沒有執行；不會由prepare/preflight自動接續。checkpoint只在optimizer邊界保存，`--resume`要求同一formal run、schema與plan指紋。

## 驗證與限制

16項CPU測試涵蓋四組共享元件初始化、尾批9對完整平均的數值對照、四種實際小模組的合成續跑、跨seed/arm/工程checkpoint拒絕、有限梯度、CE平手與中斷validation不重做train更新。
真實工程更新與checkpoint重載不等於完整資料訓練成功；小樣本時間也不是epoch工時估計。
來源、權重、資料及完整readiness在ignored `artifacts/`；可攜的完成狀態見 [四組預檢報告](../../docs/research/ACA_FOUR_ARM_READINESS.md)。
