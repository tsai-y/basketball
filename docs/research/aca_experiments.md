# ACA 候選實驗保存狀態（更新：2026-11-08）

**2026-11-16：固定16支[時間／裁切檢查](INPUT_CROP_RETENTION_DECISION.md)已完成，決定不採用固定5%裁切；維持已凍結50幀RGB基準，未訓練或啟用新候選。**

**2026-11-15：[七步基準資料處理](INPUT_PROCESSING_FOUNDATION_STATUS.md)已完成並凍結；保留整片與160×120，全8739支train／validation解碼通過。未啟用新切分、D3虛擬25Hz或模型訓練，原片內容問題只記註記。下列為既有歷史紀錄。**

**目前訓練與每小時排程均暫停。** 四組seed20261024已完成，Baseline seed20261025保存於第1輪4928支／308次更新，另外七組未開始。G1重建規格、G2盤點、D1/D2候選及[D3前處理規格](ACA_EIGHT_CLASS_PREPROCESSING.md)已完成。D3通過17項合成測試，候選未啟用，作者精確版未核實；下一步D4真實解碼與隔離八類adapter工程驗證，不直接續跑八組；見 [研究決策](ACA_REPRODUCTION_CHECKPOINT.md)。

11/03架構核對與十二組工程readiness完成；後續正式訓練/環境修復與11/05使用者主動暫停，見 [執行歷史](ACA_FOUR_ARM_CAMPAIGN.md)。這些工程驗收不代表已核實作者精確實作。

以下歷史候選與Git驗收保存前輪整理結果；目前研究決策以上方核對表為準。
`aca_net_v1/README.md` 的「正式訓練尚未開始」描述原型階段，不涵蓋後來獨立目錄的實驗。
原型與實驗來源檔按原貌保留，不修改已被歷史 run 記錄的程式指紋。

## 已完成的候選訓練

| 實驗 | 完成 epoch | 最佳 epoch | Validation accuracy | 定位 |
| --- | ---: | ---: | ---: | --- |
| Dense 五輪 pilot | 5 | 5 | 79.12% | 事前固定五輪篩選 |
| Dense 延長至總十五輪 | 13，早停 | 8 | 80.61% | 看過五輪結果後授權延長，不能算獨立 seed |
| DW3D | 9，早停 | 4 | 80.16% | 未核實作者設定的 depthwise 候選 |
| 高解析度 Dense | 12，早停 | 7 | 78.67% | 全新 ImageNet 初始化，192×256 |

上述候選主要為 seed20261025，不能當成三 seed 平均或作者精確復刻。
原 dense 二類模型 93,163,870 參數，DW3D 二類 25,464,286；論文約 26.8M 的設定仍待核實。
不得直接用參數接近宣稱已還原作者模型，也不以本次 Git 整理重選模型。

## Validation 與 Test 必須分開

已完成 dense／DW3D 及固定主線融合的 validation 比較；屬重複使用開發 validation 的探索。
[Validation 兩模型摘要](results/aca_validation_models.json)、[主線融合摘要](results/aca_validation_fusion.json)。

另有 `three_equal_test_g62sv35q` 的 completed 報告：固定低解析度 Dense、DW3D、MixNet-I3D＋Transformer 三模型等權融合，1539 支 test。
這是已有的歷史 test，不是本次重新執行；不依該結果調架構、權重、門檻或重跑。
來源報告明確標記 `globally_unseen_test=false`，不能稱為整個專案從未觸及的 holdout。
[凍結融合 Test 摘要](results/aca_frozen_ensemble_test.json)。TANet 自己未執行 test 不會改變此歷史。

## 可攜的訓練結果快照

- [Dense 五輪](results/aca_dense_pilot.json)
- [Dense 延長](results/aca_dense_extension.json)
- [DW3D](results/aca_dw3d.json)
- [高解析度](results/aca_high_resolution.json)

小型 JSON 保存來源 artifact 路徑、完整原報告 SHA 與結果；部分診斷或 runtime 欄位省略，來源檔 SHA 仍對應完整原報告。
沒有重新推論或更改舊摘要。大型 checkpoint、原片、逐幀圖與逐片預測留在原 ignored artifacts。

## 分支與下一步

`codex/aca-net-reproduction` 保存既有候選程式、設定、測試與摘要；由此建立
`codex/aca-architecture-audit` 進行下一步結構核對。
TANet 位於 [獨立研究分支](https://github.com/tsai-y/basketball/tree/codex/tanet-experiments)，本次不整支合併。
共同預檢與單seed四組已完成；作者規格仍有缺口，依最新決策先收斂複刻核對，暫不續跑多seed或下載SpaceJam。

## 初次 Git 整理驗收（歷史）

既有 ACA 原型與後續套件的 177 項 Python 測試通過，Node 融合測試檔檢查通過。
未執行任何 `--execute-validation`、`--execute-once`、訓練或續訓入口。
所有既有來源程式、config 與 artifacts 內容保留；改動僅為 Git 忽略規則及新增狀態文件／摘要。
Git push 只保存已追蹤內容，不包含被忽略的權重、資料、依賴 symlink 與完整產物，不能取代模型備份。
