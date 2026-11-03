# 研究分支與實驗狀態（2026-11-03）

這是 Git 整理時依現有產物核對的狀態快照；沒有啟動訓練、重新推論或重新執行 test。
各模組 README 可能包含歷史階段說明，請搭配具體 run 的完成摘要閱讀。

## 分支分工

| 分支 | 範圍 |
| --- | --- |
| `codex/yolo-isolated-prototype` | 球標註、資料版本與 YOLO 診斷；既有來源隔離規則不變 |
| `codex/tanet-experiments` | 時間取樣、固定作者 TAM 原碼、TANet 三 seed、BN 診斷與 L v2 |
| `codex/aca-net-reproduction` | ACA dense／DW3D 候選、訓練延長、融合、解析度與評估紀錄 |
| `codex/aca-architecture-audit` | 從已保存的 ACA 候選分支接續架構對照；尚未執行新架構實驗 |

目前工作目錄保留在 `/home/tsai/projects/basketball`，避免改變既有 run、資料契約與報告使用的路徑。
ACA 使用原有 `/home/tsai/.codex/worktrees/aca-net-reproduction/basketball` 工作目錄。
分支共用 Git 物件，但各自保存所屬程式；這次不合併、不 rebase `main`。

## TANet 已完成範圍

- 原 TANet 三 seed 與 L 當片統計 v2 三 seed 都已完成。
- L v2 最佳 epoch：seed20261024=8、20261025=12、20261026=15。
- L v2 validation accuracy 平均 81.1463%，樣本標準差 2.2637 個百分點；macro-F1 0.8098413，樣本標準差 0.0233793。
- L v2 的規則由同一 validation 的探索提出，不能視為獨立未見資料估計或作者 ACA-Net 設定。
- 固定 24 支 validation 影片已完成素材與 AI 初審；專案記錄人工確認 0 件，不代表人工已確認標籤或事件時間點。
- 本 TANet 實驗未執行 test，每小時排程已暫停。

機器摘要：

- [原 TANet 三 seed](results/tanet_original_three_seed.json)
- [L v2 三 seed](results/tanet_l_v2_three_seed.json)
- [24 支影片診斷狀態](results/tanet_l_v2_error_audit_status.json)

這些摘要從現有 JSON 保存並記錄來源 SHA，沒有重算模型預測；影片、權重、逐片預測與大型中間資料留在原本 ignored artifacts。

## 與 ACA 支線的界線

ACA 另一工作目錄已有完成的 dense、DW3D、延長及高解析度候選訓練；不能再概括為只有未訓練原型。
原低解析度固定三模型融合也已有一次 completed test 報告；因此「TANet 本輪未使用 test」不代表整個專案的 test 從未使用。
ACA 候選仍非已核實的作者精確復刻；詳見
[ACA 分支狀態](https://github.com/tsai-y/basketball/blob/codex/aca-net-reproduction/docs/research/aca_experiments.md)。

## 後續順序

先做 ACA 論文／既有原型／外部提案的結構對照，釐清 dense／depthwise gating、BN、超參數及約 93.18M／26.8M 參數差異。
再決定候選、工程驗收與固定協定的四組比較；SpaceJam 放在架構確定之後。本次 Git 整理不授權新訓練或 test。

## Git 與資料保存

- Git 保存程式、設定、測試、作者 LICENSE／PROVENANCE，以及小型結果摘要。
- `artifacts/`、權重、影片、`runs/detect/`、本機 `CURRENT_*.json` 與依賴 symlink 不納入提交。
- 目前固定路徑與 seed 複本按實驗原貌保存，不在整理 Git 時重構或刪除。
- 舊根目錄 `modelbook.md` 已移至 [歷史規劃](../history/modelbook_v0.1.md)，原始正文保留。
- commit／push 不等於備份 ignored 模型與資料；原產物保持原地，仍需另行安排備份。

本次提交前 CPU 檢查：YOLO 115、時間取樣 88、TANet 53、L v2 20 項測試通過。
ACA 工作目錄另有 177 項 Python 測試通過，Node 融合測試檔檢查通過；沒有啟動實驗評估入口。
