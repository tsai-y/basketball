# Project Progress Log

## Current Status

Current phase:
Reference C3D training protocol and short feasibility tests completed; full training pending

Current model:
Modified C3D-style Experiment A (trained); reference C3D adaptation (implemented, untrained)

Latest test accuracy:
74.79%

Current reference:
C3D - Tran et al., ICCV 2015

Next planned task:
Full Reference C3D training in terminal/tmux

Last updated:
2026-09-30


## 2026-09-27 - Dataset Inspection

### Goal
檢查 Basketball-51 資料集結構與品質。

### Completed
- 檢查 10,311 支影片
- 確認 8 個原始類別
- 確認 Make / Miss mapping
- 發現 exact duplicates
- 發現 Make / Miss label conflicts

### Key Findings
- Basketball-51 沒有 release frame annotation
- 所有影片均可正常解碼
- exact duplicate 不應跨 split

### Next
建立資料清理、split 與 baseline。


## 2026-09-27 - C3D-style Baseline

### Goal
建立 Basketball-51 Make/Miss full-video classification baseline。

### References
- Basketball-51 paper
- Tran et al., C3D, ICCV 2015

### Completed
- 建立 dataset manifest
- 清理 duplicates / conflicts
- 建立 train / validation / test split
- 建立 PyTorch Dataset/DataLoader
- 建立 modified C3D-style model
- Smoke test PASS
- Small-subset overfit test PASS
- 完成 5 epochs training
- 完成 test evaluation

### Results
Validation Accuracy:
76.02%

Test Accuracy:
74.79%

Precision:
71.17%

Recall:
87.38%

F1:
78.44%

### Important Notes
目前不是原始 C3D 的完整 reproduction。

目前使用完整影片，因此不能稱為 pre-release prediction。

### Next
進一步做更忠實的 C3D reproduction。


## 2026-09-29 - C3D Implementation Audit / Reproduction Specification

### Goal
依完整論文與作者官方 Caffe 網路定義，建立可追溯的 C3D reproduction specification；本輪只更新研究文件。

### Paper Reviewed
- `papers/c3d_tran_2015.pdf`: Tran et al., *Learning Spatiotemporal Features with 3D Convolutional Networks*, ICCV 2015，含補充資料。主要核對印刷頁 4491–4493 的 §3.1、§3.3、Fig. 3、§4。
- 作者官方 `facebookarchive/C3D` 的 Sports-1M video feature extractor 與 UCF101 fine-tuning train prototxt；來源連結與逐項證據記於 `docs/c3d_reproduction_spec.md`。

### Files Inspected
- `readme.md`、`docs/progress_log.md`、`docs/references.md`、`docs/baseline_protocol.md`、`docs/baseline_results.md`
- `configs/baseline.yaml`、`baseline/data.py`、`baseline/model.py`、`scripts/run_baseline.py`
- `runs/full_video_c3d_style_v1/` 的既有狀態與結果（僅讀取，未修改）

### Confirmed Implementation Details
- 論文：8 個 `3×3×3` conv、通道 64/128/256/256/512/512/512/512、5 個 max-pool、fc6/fc7 各 4096、16-frame RGB crop、影片描述子使用 8-frame overlap，平均 fc6 後 L2 normalization。論文另有 Sports-1M 的十個隨機 clip prediction averaging，與 fc6 描述子不同。
- 官方實作：每個 conv 與 fc6/fc7 後 ReLU；fc6/fc7 後 dropout 0.5；conv 的 spatial/temporal padding 1；conv/FC bias 與 prototxt 初始化 filler；檢查的網路定義無 BatchNorm/LayerNorm。

### Unresolved Implementation Details
- 原實驗權重的完整來源／seed、pretrained checkpoint 與發布論文結果的精確對應。
- Sports-1M 十個測試 clips 的抽樣 seed/位置、影片尾端與不足 16 frames 的處理、Caffe crop 細節。
- 官方 feature export 所指 fc6 blob 的精確 activation point、原模型參數總數及 Caffe/PyTorch pooling shape 對應。
- 論文稱 RGB；已檢查的 mean file 宣告不足以確認原權重所需的完整色彩順序與數值正規化。

### Important Decisions
- 只將論文明寫的內容標成 PAPER_CONFIRMED；ReLU、dropout、padding 等由官方 prototxt 確認的內容另列 OFFICIAL_IMPLEMENTATION_CONFIRMED。
- 保留完成的 Experiment A 與固定 split；未修改模型、未進行新訓練，也未覆寫任何 run artifact。
- 未來成果應稱為 Basketball-51 上的 reference-faithful C3D adaptation，而不是原論文 benchmark 的完整 reproduction。

### Files Changed
- 新增 `docs/c3d_reproduction_spec.md`。
- 僅補充 `docs/references.md` 的 C3D 小節。
- 更新本檔的 Current Status 並追加本輪紀錄；保留既有歷史。

### Next Step
Reference-faithful C3D implementation（須先定案未解決的 preprocessing、aggregation、checkpoint 來源；本輪未開始）。


## 2026-09-30 - Basketball-51 Reference C3D Adaptation Pipeline

### Goal
實作連續 16-frame clips、原始 C3D 架構對應、fc6 影片描述子，以及有限的 clip／CUDA smoke／小樣本 pipeline gates；不進行全量訓練或 test 評估。

### Completed
- 沿用既有 cleaned manifest 與固定 split，沒有重抽或改寫；Experiment A 的模型與 run 未修改。
- 新增獨立 `reference_c3d/`、`configs/c3d_reference.yaml`、`scripts/run_c3d_reference_checks.py` 與 `docs/c3d_reference_implementation.md`。
- 尾段規則選擇只保留完整 clip，標記為 PROJECT_ADAPTATION；兩支 146-frame train 影片皆產生 17 個連續 clips，第一、二個零起算索引為 `[0..15]` 與 `[8..23]`。
- CUDA smoke PASS：`[2,3,16,112,112]` → pool5 `[2,512,1,4,4]` → fc6 `[2,4096]` → descriptor `[4096]`；梯度有限且非零；模型參數 `78,003,970`。
- 小樣本第一次 FAIL：原始隨機 fc6 描述子差距很小，診斷分類器 loss 停在約 0.693；失敗紀錄保留。只用兩支 train 影片做診斷特徵縮放後，小樣本 pipeline gate PASS。這只訓練線性診斷 head，C3D backbone 沒有訓練。

### Important Decisions and Unresolved
- fc6 輸出取在 ReLU 後、dropout 前；`ceil_mode: true`、RGB `[-1,1]`、尾段丟棄與二類 fc8 都是明列的 PROJECT_ADAPTATION。
- 原 checkpoint／mean-file 相容性、精確 Caffe pooling 與 fc6 匯出位置仍需確認。小樣本結果不是研究準確率，也不能證明 backbone 已學會分類。
- `runs/full_video_c3d_reference_v1/` 只有 config、環境與 gate 紀錄；無正式 checkpoint 或 test metrics。詳情見 `docs/c3d_reference_implementation.md`。

### Next Step
Full reference C3D training / feature extraction experiment（本輪未開始）。


## 2026-09-30 - Reference C3D Training Protocol / Feasibility

### Goal
決定初始化、訓練層級、影片級 loss 與 binary classifier；在 train 資料上做短時 shape、VRAM、throughput、DataLoader 和正式腳本 dry-run。**沒有跑完整 epoch、完整資料訓練或 test 評估。**

### Modes Benchmarked
- Full fc6 gradient path、frozen C3D + linear head、partial conv5a/conv5b/fc6 + head。
- 三種模式均以 clip batch 1/2/4/8 各做 3 warmup + 5 timed iterations；全部 forward/backward PASS、無 OOM。
- 對一支 17-clip train 影片各測一次：全 clips 同批、兩 clip 分塊但保留完整梯度圖、兩 clip 分塊且 backbone frozen。記憶體與時間詳見 `docs/c3d_training_protocol.md` 及 `runs/full_video_c3d_reference_v1/feasibility/`。

### VRAM / Throughput Results
- Full fc6 path batch 1/2/4/8：peak allocated 約 1258/1360/1683/2334 MiB，平均 17.54/26.01/42.26/73.56 ms/step；8 clip batch 約 108.8 clips/s。完整 17-clip 影片分塊、保留梯度圖時 peak allocated 3537 MiB、reserved 3840 MiB。
- Frozen batch 8：約 1139 MiB allocated、354.4 clips/s；partial batch 8：約 1503 MiB allocated、274.5 clips/s。這些只是短 benchmark，非正式 epoch 速度。
- DataLoader 六支 train 影片、每支 17 clips：0/2/4 workers 的平均 data wait 為 0.0908/0.0156/0.0189 秒／影片，三者皆穩定；選 2 workers。沒有直接量測 GPU utilization。

### Initialization / Protocol Decision
- 目前沒有可靠相容的原作者 Sports-1M checkpoint 或 mean file，因此**主要可執行方案**選官方 prototxt filler 從零初始化，訓練 conv1a→fc6 全部活躍路徑加 4096→2 線性 head；影片所有 clip 先平均 fc6、L2 normalize，再計算一次 video-level CE。fc7/fc8 不在此 loss 路徑，將凍結。
- **次要條件比較**：若日後驗證作者權重、channel/mean、pool5/fc6 與輸出一致，再做 frozen descriptors + binary linear SVM，之後才比較 partial/full fine-tuning。這些是研究建議，不是本輪已完成實驗。
- 官方 Caffe pool3d `ceil` shape 公式與 PyTorch `ceil_mode=True` 的最小 audit PASS：pool5 `[B,512,1,4,4]`、flatten 8192、fc6 `8192→4096`。

### Training Script Check
- 新正式腳本只允許既定 train/validation split，另存新 run 目錄，支援 epoch-boundary checkpoint/resume；本輪只執行 `--check` 與單支 train／validation 的 `--dry-run`。
- 第一次 dry-run 的運算完成後，NumPy 布林值 JSON serialization 報錯；修正型別後重跑 PASS。train 與 validation 各處理 17 clips，loss 均為有限值，沒有寫正式 run 檔。CUDA max-pool3d backward 發出非 bitwise-deterministic 警告。

### Files Changed
- 新增 `configs/c3d_reference_benchmark.yaml`、`configs/c3d_reference_train.yaml`、`scripts/audit_c3d_shape.py`、`scripts/benchmark_c3d_reference.py`、`scripts/run_c3d_reference_train.py`、`docs/c3d_training_protocol.md`。
- 僅在 `runs/full_video_c3d_reference_v1/feasibility/` 寫入本輪短 benchmark JSON；既有 Experiment A 與 3B gate 檔未覆寫。
- 追加本紀錄並補充 README。

### Remaining Risks
從零訓練過擬合風險、原權重的 mean/color 相容性、fc6 精確匯出點、短測無法代表完整 epoch、CUDA max-pool3d backward 非 bitwise deterministic、真實 game identity 與 near-duplicate 未核實。

### Next Step
Full Reference C3D training in terminal/tmux（待確認，尚未執行）。


## 2026-09-30 - Reference C3D Early-Stopping / Scheduler Audit

### Goal
稽核已完成的 Experiment B epoch 13 early stop 與降 LR 的交互作用；只做唯讀檢查和 resume 專用的小修正，不執行 epoch 14 或 test 評估。

### Confirmed State
- `training_log.json` 有 13 筆；`last.pt` 記錄 epoch 13、best epoch 10、best validation loss `0.6015637946479461`、未改善 counter 3、optimizer LR `5e-5`，scheduler `last_epoch=13`。
- 程式於 epoch 12 validation 後降 LR；log 記的是 `scheduler.step` 後的 LR，所以低 LR 實際只用於 epoch 13 訓練一次。epoch 11–13 均未達原有 best-loss/min-delta 條件，counter 因而為 3，符合 early-stopping 停止條件。
- `best.pt` 仍是 epoch 10；`training_status.json` 為 13 epochs、目標上限 20、`test_evaluation_performed=false`。

### Decision and Files Changed
- 研究判斷：原 patience 3 可能未給降 LR 後足夠觀察期；保守建議 resume-only patience 6、max epochs 20，不修改原 config、不清零 counter，也不變更 optimizer/scheduler/best checkpoint。這不是 C3D 論文設定。
- `scripts/run_c3d_reference_train.py` 新增 resume-only patience override、唯讀 preflight 顯示當前 LR/counter/epoch，並保存後續 checkpoint 的有效 patience；`docs/c3d_training_protocol.md` 新增 audit 與指令。本紀錄只追加，不改寫歷史。
- 唯讀 `--resume --check --max-epochs 20 --early-stopping-patience 6` 通過：next epoch 14、LR `5e-5`、best epoch 10、原 counter 3。未執行正式訓練，未讀取 test set。

### Next Step
待研究者確認後，才從 epoch 13 的 `last.pt` 續跑；test set 仍保持封存。


## 2026-09-30 - Experiment A1 Resume Audit / A2 Protocol

### Audit
A1 `best.pt` 僅保存模型權重、epoch 5、validation loss、seed、config、manifest/protocol hash；沒有 optimizer、scheduler、RNG、training history 或 `last.pt`。因此 A1→A2 分類為 **WARM_START_ONLY**，不能稱為真正 resume。A1 最後五個 validation loss 連續下降；以 validation 趨勢支持有界的延長訓練，但不預測 A2 效果。

### A2 Prepared
新增獨立 `configs/c3d_style_extended.yaml`、`scripts/run_c3d_style_extended.py` 和 `docs/c3d_style_extended_protocol.md`。A2 從 A1 epoch-5 最佳權重初始化、新建 AdamW，最多跑總 epoch 標籤 6–15，early-stopping patience 5，以最低 validation loss 選 checkpoint；不新增 scheduler。正式 A2 run 目錄尚未建立。

### Checks
唯讀 config/checkpoint compatibility PASS；CPU 單批 train/validation dry-run PASS，輸入 `[2,3,16,96,96]`、輸出 `[2,2]`、loss 與梯度有限。沒有執行 A2 正式訓練，也沒有讀取 test 影片或重新評估 A1 test。A1 四個受保護檔案的 SHA-256 於本輪前後一致。

### Next Step
待確認後執行獨立 A2 training command；A2 固定完成前不做 test 評估。


## 2026-09-30 - Experiment A2 Epoch-15 Resume Audit

### Confirmed State
A2 `last.pt` 已到 epoch 15，保存模型、AdamW state、四種 RNG state、10 筆 A2 歷史、best epoch 15／validation loss `0.48392225430138125`、未改善 counter 0；optimizer LR `0.001`。`best.pt` 與此最佳值一致；`training_status.json` 記錄 `test_evaluation_performed=false`。這是 **A2 內部 TRUE_RESUME**，不改變先前 A1→A2 為 warm-start 的判定。

### Decision / Changes
只新增 A2 resume 專用上限 50、patience 7；保留原 config 的歷史 15／5、固定 LR `0.001`、既有 optimizer/RNG/counter/history/best。`scripts/run_c3d_style_extended.py` 加入唯讀 resume preflight 和從 `last.pt` 恢復的分支；`docs/c3d_style_extended_protocol.md` 記錄延長決策與指令。沒有執行 epoch 16 或 test 評估。

### Checks
`--resume --check` PASS：next epoch 16、best epoch 15、best loss `0.48392225430138125`、LR `0.001`、history 10 筆。從 epoch-15 權重與 optimizer 載入的單批 CPU train/validation dry-run PASS，沒有寫入 run 檔。A2 `last.pt`、`best.pt`、`training_log.json` 的 SHA-256 前後一致。

### Next Step
待確認後在終端執行 A2 resume command；test 仍封存。
