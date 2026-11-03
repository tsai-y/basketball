# ACA 候選實驗保存狀態（2026-11-03）

此文件依現有 run 摘要核對；本次只整理 Git，沒有新訓練、影片推論或 test。
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
先釐清架構證據，再決定是否建立新的固定協定對照；SpaceJam 暫不下載或訓練。

## 本次 Git 驗收

既有 ACA 原型與後續套件的 177 項 Python 測試通過，Node 融合測試檔檢查通過。
未執行任何 `--execute-validation`、`--execute-once`、訓練或續訓入口。
所有既有來源程式、config 與 artifacts 內容保留；改動僅為 Git 忽略規則及新增狀態文件／摘要。
Git push 只保存已追蹤內容，不包含被忽略的權重、資料、依賴 symlink 與完整產物，不能取代模型備份。
