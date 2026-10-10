# Project Progress Log

## Current Status

Current phase:
A1/A2/B/C full-video training and one-time formal test evaluation completed

Current model:
Modified C3D-style A1/A2, Reference C3D B, and continuous-clip Modified C3D-style C (all trained)

Formal test accuracy (A1 / A2 / B / C):
0.7478882391163093 / 0.7738791423001949 / 0.7342430149447693 / 0.6049382716049383

Current reference:
C3D - Tran et al., ICCV 2015

Next planned task:
整理完整影片階段結論；後續另定 pre-release protocol

Last updated:
2026-10-01


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


## 2026-09-30 - Experiment B Early-Stopping Extension Audit

### Confirmed State
Experiment B 仍停在 epoch 13：`last.pt` 有 model/head、AdamW、ReduceLROnPlateau、四種 RNG state、13 筆歷史，best epoch 10／validation loss `0.6015637946479461`、counter 3、optimizer LR `5e-5`。`best.pt` 仍是 epoch 10，test 尚未使用。程式於 epoch 12 validation 後才降 LR，故只在 epoch 13 以 `5e-5` 訓練一輪；counter 在 epoch 10–13 為 0、1、2、3，符合原 patience 3 的 early stop。

### Decision and Changes
研究判斷：原 patience 3 可能讓降 LR 後的觀察期過短，無法保證延長必有改善。以 resume-only CLI 建議 max epochs **30**、patience **7**；保留 epoch-13 optimizer/scheduler/LR、best、RNG、歷史 counter 3 和原 config。先前 20/6 建議未執行，本次 30/7 取代它。發現舊程式將 `min_delta=0.001` 同時用於 best checkpoint；已小幅修正為 `best.pt` 依嚴格較低的 validation loss 更新，early-stopping counter 仍以原 `0.001` 門檻判斷，新增獨立 reference loss 供後續 checkpoint 保存。這是專案訓練策略澄清，不是 C3D 論文設定。

### Files / Checks
更新 `scripts/run_c3d_reference_train.py` 與 `docs/c3d_training_protocol.md`，本紀錄只追加。唯讀 `--resume --check --max-epochs 30 --early-stopping-patience 7` 確認 next epoch 14、LR `5e-5`、best epoch 10、counter 3；未跑 epoch 14，未讀取 test。`last.pt`、`best.pt`、`training_log.json` 前後 SHA-256 一致。

### Next Step
待確認後續跑 Experiment B；test set 保持封存。


## 2026-10-01 - Experiment B Training Finalized / Experiment C Pipeline Validation

### Goal
依最新正式 run 檔案更新 B 狀態；建立 Experiment C 的 Modified C3D-style backbone＋B continuous clip 管線，只做 config、接線、shape、smoke 與極小 train/validation dry-run，不做 C 正式訓練或任何新 test 評估。

### Experiment B Confirmed Final State
- `runs/full_video_c3d_reference_train_v1/training_status.json` 為 `COMPLETE`，`epochs_completed=32`、`target_max_epochs=50`、effective early-stopping patience 7，因未改善而在 epoch 32 停止。
- 32 筆 `training_log.json` 中最佳 epoch 25：validation loss `0.5345329800506291`、validation accuracy `0.7504848093083387`；status 的最佳 epoch/loss 與 log 相符。
- `test_evaluation_performed=false`。A2/B/C test 都未因本輪被讀取或評估；歷史段落反映當時狀態，目前狀態以本段及 run 檔案為準。

### Experiment C Design and Files
- 研究 C vs A2 的稀疏抽樣與連續 clips、C vs B 的 Modified/Reference backbone；C vs A2 同時有 96→112 解析度、crop/head、初始化差異，不是單變因 ablation。
- 新增 `configs/c3d_style_continuous.yaml`、`experiment_c/__init__.py`、`experiment_c/model.py`、`experiment_c/aggregate.py`、`scripts/run_c3d_style_continuous.py`、`docs/c3d_style_continuous_protocol.md`；更新 `readme.md` 與本 progress log。A1/A2/B 模型、config、checkpoint 與正式 run 未修改。
- 直接重用 B 的 `ContinuousClipSampler`、`C3DVideoDataset` 和前處理；重用 A1/A2 模型的 `features`/GAP，不引入原分類 head。256-D clip 特徵 mean→L2→新 `Linear(256,2)`，只計算影片級 CrossEntropyLoss。
- C 擬從零訓練；初始 PROJECT_ADAPTATION 設定為 AdamW LR `1e-4`、ReduceLROnPlateau、最多 50 epochs、patience 7，以最低 validation loss 選 checkpoint。尚未執行完整 epoch。

### Checks and Remaining Limits
- Config/hash 與 B preprocessing equality check **PASS**；C run 目錄尚不存在。40-frame 索引檢查得到 `[0..15]`、`[8..23]`、`[16..31]`、`[24..39]`，stride/overlap 均 8。
- CPU shape audit **PASS**：`[2,3,16,112,112]`→feature map `[2,256,1,3,3]`→GAP `[2,256,1,1,1]`→clip `[2,256]`→影片 `[256]`→logits `[2]`；總／可訓練參數均為 **6,919,234**。
- CPU synthetic smoke **PASS**：有限 loss、有限非零梯度；CPU 真實資料 dry-run **PASS**：一支 train 與一支 validation 影片各 146 frames／17 clips，train 完成一次 optimizer step，validation loss 有限。這些不是正式準確率；沒有寫入 C run 或讀取 test 影片。
- 本執行環境的 PyTorch 選到 CPU，C 的 CUDA VRAM/throughput、完整 epoch 穩定性尚未確認。A2 vs C 存在 spatial/head/initialization 混雜因素，game ID 與 near duplicate 問題仍未解決。

### Next Step
Experiment C full training（需在正式環境監控 VRAM/throughput，test 仍封存）。


## 2026-10-01 - Experiment C Formal Training Completed

### Goal
依現有 C run 的正式狀態與 log，同步 README 和目前進度；不重新訓練或評估 test。上一節描述的是 C 實作與 dry-run 當時的狀態。

### Confirmed Training State
- `runs/full_video_c3d_style_continuous_v1/training_status.json` 為 `EARLY_STOPPED`，完成 **11 epochs**，最佳 epoch **4**，最佳 validation loss **0.6593953967287067**，停止時未改善 counter **7**。
- `training_log.json` 有連續的 epoch 1–11 共 11 筆；epoch 4 validation accuracy **0.6153846153846154**。`last.pt` 記錄 epoch 11／best epoch 4；`best.pt` 為 epoch 4，與 status 和 log 相符。
- `test_evaluation_performed=false`。本次只讀取 C 的 status、log 與 checkpoint metadata；沒有讀取或評估 test 影片。

### Files Changed
僅更新 `readme.md` 與本進度紀錄，將 C 標示為訓練完成、epoch 11 early stop、最佳 epoch 4；未修改模型、config、checkpoint 或 run 輸出。

### Next Step
待研究者確認下一步；A2/B/C test 仍未評估。


## 2026-10-01 - Formal A2/B/C Test Evaluator Prepared

### Goal
建立只讀取 validation 選出的 `best.pt` 的一次性正式 test evaluator，先完成唯讀稽核；本輪不執行 A2/B/C test。

### Frozen Training State and Dataset Audit
- A2/B/C 均已完成訓練；預期 `best.pt` epoch 分別為 **21／25／4**。三個 checkpoint 的 metadata 與 training status、最佳 validation log 相符，模型 state 嚴格載入成功。B checkpoint 未直接保存 manifest hash，另核對其 run manifest/split 快照、config 及 status。
- 只使用 manifest／Dataset row metadata 比對三組 test IDs；每組都是相同的 **1,539** 個唯一影片（Miss **731**、Make **808**），與 train/validation ID 不重疊。排序後 test ID SHA-256 為 `2c6c37cc846179aea27e1b2a0726a4210c50a184357792202791ddd9752d3eba`。

### Implementation and Checks
- 新增 `scripts/evaluate_full_video_experiments.py` 與 `docs/formal_test_evaluation_protocol.md`。A2 使用固定 bin center/96×96；B/C 使用固定 16-frame、stride 8、112×112 center crop，各自載入 frozen best model/head。正式路徑使用 `eval()` 與 `inference_mode()`，以 argmax 產生預測，計算影片級 test loss、accuracy、Make precision/recall/F1、混淆矩陣與類別數量；CSV 只含相對 video ID 與預測欄位。
- 三個指定的 `--experiment a2|b|c --check` 均 **PASS**，確認 source/config/manifest/split、test 身份、checkpoint epoch/state 與輸出路徑可用。Check 不呼叫 test Dataset `__getitem__`，不解碼影片、不計算 test metrics、不寫 test output。
- 一次性輸出保護拒絕既有 `test_metrics.json`、`test_predictions.csv` 或 `.test_evaluation_claim`；正式路徑才會獨占建立 claim 與結果檔，沒有 `--force`。中斷後須人工 audit，不能自動重跑。

### Test Status / Next Step
**Formal test executed: NO.** A2/B/C test metrics 仍未知。下一步是人工審閱 evaluator 與凍結 protocol，確認後才各執行一次正式 A2/B/C test。


## 2026-10-01 - Full-Video Formal Test Stage Finalized

### Goal and authoritative sources
依已完成的一次性正式 test，將 A1/A2/B/C 全片分類結果同步到 README 與 `docs/formal_test_evaluation_protocol.md`。本次文件整理只讀取四個 run 既有 `test_metrics.json`，未執行訓練或評估，也未修改任何 checkpoint、split、`test_metrics.json` 或 `test_predictions.csv`。上一節「尚未評估」記錄的是正式 test 前的歷史狀態。

### Final Results
正類為 Make=1；混淆矩陣列是真值 `[Miss, Make]`、欄是預測 `[Miss, Make]`。

| Experiment | Best epoch | Test loss | Accuracy | Precision Make | Recall Make | F1 Make | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1 | 5 | 未記錄 | 0.7478882391163093 | 0.7116935483870968 | 0.8737623762376238 | 0.7844444444444445 | `[[445,286],[102,706]]` |
| A2 | 21 | 0.5014848665729446 | 0.7738791423001949 | 0.7572706935123042 | 0.8378712871287128 | 0.7955346650998824 | `[[514,217],[131,677]]` |
| B | 25 | 0.5671578797525675 | 0.7342430149447693 | 0.7484433374844334 | 0.7438118811881188 | 0.7461204220980757 | `[[529,202],[207,601]]` |
| C | 4 | 0.6629814902747738 | 0.6049382716049383 | 0.6265822784810127 | 0.6126237623762376 | 0.6195244055068836 | `[[436,295],[313,495]]` |

A1 的舊版正式 metrics 沒有 test loss 欄位；不能將 A1 validation loss 當成 test loss。

### Frozen Test Identity and Decisions
- A2/B/C 的正式結果均記錄同一 **1,539** 支 test 影片（Miss **731**、Make **808**），相同 manifest SHA-256 `e7bba1acd6028cf67cdf28251e69852e82cff201e13519ed05970ac483cce6b6`、split-summary SHA-256 `953da31313976763c8f5d597da5f0fe5a6a45e10f5d5edcd42fb0c120af2f7c5`、排序後 test-ID SHA-256 `2c6c37cc846179aea27e1b2a0726a4210c50a184357792202791ddd9752d3eba`。A1 舊版 metrics 僅保存同一 manifest hash 與 test count，未保存後兩項 hash。
- A2/B/C 是**完整實驗流程**比較，不是單變因 ablation。A2/C 的解析度、crop、head、初始化也不同；B/C 的 backbone、特徵維度及訓練歷程不同。不能據此斷言稀疏取樣本質上優於連續取樣，或 Modified C3D 本質上優於 Reference C3D。
- 四個模型均使用**完整影片**，可能看到球離手後球路、碰框、籃網與後續反應；本階段結果不證明 pre-release prediction。A2/B/C 不得根據 test 表現重新調整模型、訓練或閾值。既有 `test_metrics.json` 和 `test_predictions.csv` 視為不可變的正式結果產物，不重新評估。

### Files Changed / Next Step
本次只更新 `readme.md`、`docs/formal_test_evaluation_protocol.md` 與本進度紀錄。完整影片階段結論已封存；後續需另行制定 pre-release 資料與評估 protocol。


## 2026-10-07 - Repository Structure and Team Documentation

### Scope and current direction

本輪只整理 repository、必要路徑與團隊文件，沒有開始資料清洗、模型實作或訓練。目前仍為 full-video Make/Miss classification，優先規劃自動品質篩檢與影像增強，之後研究特徵擷取、多模型互補／融合；具體策略、組合及融合方式尚未決定。A1/A2/B/C frozen；Gate 1A／1B completed，Gate 1C 與原 Experiment D paused。Pre-release 不在目前範圍；SPL 指 Sport Performance Lab 的 SPL Open Data，是最後才考慮的資料集延伸，不是模型模組。本檔頂部 Current Status 保留 2026-10-01 的歷史內容，目前排程見 [current_research_direction.md](current_research_direction.md)。

### Structure and preservation

- 根目錄 README 改為繁體中文團隊入口；小寫 `readme.md` 為指向 `README.md` 的相容 symlink，維持 Gate 1B 既有連結。
- `scripts/audit_c3d_shape.py`、`scripts/benchmark_c3d_reference.py` 搬至 `baseline/scripts/`，各只將 ROOT 的 `parents[1]` 改為 `parents[2]`，仍定位相同 root 及歷史輸出。
- 建立 baseline、development、data_quality 說明，以及目前研究方向、repository map、reorganization log；docs 索引補充目前工作入口。
- 核心模型／資料程式、六份 configs、訓練／評估入口因 frozen hash／protocol 相依保留原位；共用資料工具不複製。原始資料、固定 manifest／split、全部 runs／checkpoints／logs／test 產物與 Research Baseline、Gate 報告保留原文原位。
- 開始時 tracked 工作樹乾淨，無非 ignored untracked；沒有使用者未提交變更衝突。本紀錄只追加，舊段落及其中舊路徑未改寫。

### Lightweight validation and limits

21 份 Python syntax、12 個模組 import、11 個 CLI `--help`、六份 YAML 及 29 個 root-relative 資源路徑檢查通過。Gate 1A `--mode check` 僅讀固定 manifest／split metadata，回報 PASS、video opens／probes／test video accesses 均為 0。最終本機連結與保護檔案比對結果見 [reorganization_log.md](reorganization_log.md)。沒有訓練、evaluation、benchmark、checkpoint 載入、模型下載或影片掃描／probe／解碼；不宣稱模型數值等價已驗證。未 git add、commit 或 push。

### Team entry and follow-up

先讀 root README，再讀 [repository_map.md](repository_map.md) 與 development 規則。後續品質篩檢尚無演算法或門檻。保留原位元件的全面搬移、ignored 資料分發、symlink／大小寫跨平台 checkout 與疑似過時產物審核，均列於整理紀錄；未刪除任何歷史產物。


## 2026-10-08 - Basketball-51 Train-only Blur Screening Pilot v1

### Scope and input state

實作並完成約 30 支 train 的自動品質評分、candidate flags、時間區段與影響報告，不是正式資料清理。開始時 `main` / `cafe04b62bb6ac4719a6c05131ded13b3d467024`，`git status --short` 為空；repository 整理結果已存在，沒有既有未提交變更。閱讀 README、baseline/development/data_quality 說明、目前研究方向、repository map/reorganization/progress log、Gate 1A/1B 及固定 manifest/split；適用 repository/上層未找到 AGENTS.md。

重用 Gate 1A 的 manifest/split hash、allowlist 與唯讀 opener。原 per-video `/tmp` cache 已不存在，僅對先選定的 30 支 train probe；沒有重新掃描資料集或計算 validation/test metadata。Gate 1A train 彙總提供 FPS 覆蓋對照。

### Pilot and findings

固定 seed **20261008**，交替 Make/Miss、優先來源群組分散、SHA-256(seed:video_id) 破同分；Make/Miss 各 **15** 支、**30** 個 source groups，23.976 FPS **23** 支、29.97 FPS **7** 支。全片原始 320×240、BGR→uint8 gray→float64 計算 Laplacian variance 與 Sobel gradient energy，共 **4,617** 幀，30 支 frame count 與 metadata 一致，decode failures **0**。

Pilot train P10：Laplacian **2052.339911**、Tenengrad **26513.657885**；Pearson **0.836563**、Spearman **0.840420**。Laplacian/ Tenengrad 單低各 **462** 幀（10.0065%），分別 **52/54** 區段、**15/16** 影片；雙低 **327** 幀（7.0825%）、**47** 區段、**14** 影片；0.25s median 平滑加至少 0.20s 持續條件後 **314** 幀（6.8010%）、**15** 區段、**11** 影片。全部是探索性候選，沒有選定正式清理門檻。

雙低比例 Make **6.11%**、Miss **8.01%**；23.976 FPS **9.20%**、29.97 FPS **1.43%**；各來源 **0–58.90%**。每來源僅一支，類別/來源/FPS 互有混雜，不能外推完整 train 或作因果解釋。兩張 contact sheets、18 tiles 只是檢查演算法反應，不是 ground-truth labels。

### Files and verification

新增 `development/data_quality/scripts/audit_blur_pilot.py`、`verify_blur_pilot.py`、`configs/blur_pilot_v1.yaml`、`tests/test_blur_pilot.py`、`pilot_manifest_v1.csv`、`docs/data_quality/blur_screening_method_v1.md`、`blur_screening_pilot_v1.md/.json`；更新 data_quality README 與 `.gitignore`，本進度紀錄只追加。大量 CSV/圖片/metadata/logs 位於 ignored `development/data_quality/artifacts/blur_pilot_v1/`。

Python syntax/import、CLI help/dry-run、全選樣 train 正向 guard、所有 validation/test/excluded ID 和對應錯配 path 拒絕、未知/absolute/traversal path 拒絕、same-seed/輸入順序重現、解碼失敗分離測試通過。原始解碼記錄、逐幀 CSV、JSON/Markdown 數字、四策略 flags、區段、窗口與分組摘要均重算一致；拒絕既有輸出覆寫。

初次 sandbox ptrace 受限，程式尚未啟動；獲准後以 strace 執行 Pilot。獨立 trace 核對 **120** 個唯讀 dataset open syscalls（**60** guarded opens＋**60** descriptor reopens），唯一影片均為那 **30** 支 train；**30** probes、**30** decoder sessions；validation/test opens/probes/decodes 全部 **0**。未執行 ffmpeg 影片輸出。受保護 tracked 文字檔與開始時 SHA-256 比對不變，沒有影片/checkpoint hash 或全資料集掃描；最終 `git diff --check` 通過。

### Boundary and next step

沒有修改原始影片、固定 manifest/label/split、frozen artifacts、Research Baseline 或 Gate 報告；沒有刪格、拼接、排除影片、清理後影片、增強、特徵擷取、融合、訓練、評估或 checkpoint loading。A1/A2/B/C frozen；Gate 1C/原 Experiment D paused，Pre-release/SPL 延伸仍不在範圍。

Pilot 證明此小樣本評分/防護/影響報告流程可運作；完整 train 尚需分批/記憶體容量規劃與另行授權。正式品質定義、主體範圍、來源校準、門檻/持續時間、錯誤成本與任何保留/排除政策仍待決定。本輪完成後停止，不擴展完整 train；未 git add、commit 或 push。


## 2026-10-09 - Basketball-51 Blur Train Scanner v3 and Bounded Smoke

### Scope and provenance

Pilot v2 已由使用者提交至 `dab6c46`；本輪開始 `git status --short` 為空，讀取既有規範、v1/v2 configs/scripts/tests/manifests/報告、Gate 1A 與固定 manifest/split。125 個初始 tracked 檔案建立 SHA-256 snapshot；僅本進度紀錄追加與 data_quality README 統一入口為預期修改，v1/v2、baseline、原始 manifest/split 及 frozen evidence 保留。沒有 git add、commit、push。

### Scanner and frozen rule

新增 v3 scanner、independent verifier、exclusive publisher、config、固定 smoke CSV、unit tests、方法／smoke 報告與 Ubuntu/tmux runbook。一次一支、一張解碼畫面、單影片分數，10,000-frame 工程上限；temporary file＋fsync＋atomic rename、flock、immutable attempts、completion/config/code/output hashes、input stat identity、append-only access/error/decode-failure ledgers。支援 resume、明確 retry failures、derived-only aggregate 重建；可疑或部分輸出不當成完成，舊檔不覆寫。Full-train CLI 必須確認旗標，smoke 固定最多 20 IDs。

已提交 v2 config SHA-256 `80fe9ad128d284777eb51a428ec9eabc9a6c3d4c0256637a522b1aa44d607473`，JSON SHA-256 `95f95463ccf1cbb0e5b24e73dca60e375c71be7fcc0261bdd15cd40cf96e341f`。凍結 primary Laplacian local z ≤ −2.0、中心 3 秒 log-score median/MAD、MAD scale floor 0.05、log floor 1e-12、minimum 0.16 秒；Prompt 與 repository 證據一致，沒有重新調參。Tenengrad 僅診斷，不加 AND/OR。中心窗屬離線篩檢；16/32/64 非候選連續窗未選定模型輸入長度。

### Smoke and recovery

seed=20261009，固定 20 支 train、20 來源、Make/Miss 各 10；依 label/source 與既有 train FPS metadata 選樣，沒有品質分數導向選樣或失敗替換。實際 23.976 FPS×16、29.97 FPS×4，共 3,055 幀。54 候選幀（1.7676%）、5 區段、涉及 5 支；實際 decode failures=0。這不是 full train 盛行率或清理結果。初次 scanner wall time 5.784127 秒，scanner peak RSS 131,768 KiB；RSS 父／子各自峰值相加只提供保守總上界，不宣稱同步 process-tree 峰值。

普通 resume 跳過 20/20、無新增影片存取；相同設定另目錄重跑相同 20 支，frames CSV bytes 完全一致，summary 排除耗時後一致。恢復 fixture 複製 18 支完成輸出、模擬 1 支 partial 與 1 支空 decoder failure；fixture 建置不開影片，普通 resume exit 2，retry 僅重算原 smoke 中 2 支、保留舊 attempts／獨立 failure ledger，最終 20/20。實際不同 config bytes hash 拒絕續跑且 ledger 不增加影片存取。Aggregate 獨立重建、local z／segments/windows 重算與原始影片 stat identity 比對通過。

15 個 v3、10 個 v2、6 個 v1 tests 通過；涵蓋 train 正向 guard、全部 held-out ID/path 字串拒絕、seed 重現、凍結 hashes、KeyboardInterrupt、corruption、atomic write failure、input identity、concurrent lock、單支 failure 後繼續／retry。Syntax/import/CLI help、smoke dry-run 通過。JSON／Markdown／CSV 一致性與最終保護檔／diff 檢查見 smoke 報告及 ignored validation logs。

### Access evidence and remaining work

每次實際評分皆有 durable ledger 與獨立 strace：初次 20、同樣本重跑 20、恢復重試 2，共 42 probes、42 decodes，含 FD reopen 共 168 次唯讀 dataset open syscall，唯一 IDs 仍為固定 20 個。validation/test opens/probes/decodes 全為 0，無 encoder 或原資料寫入。最初 ptrace capability test 受 sandbox 限制，取得允許後成功；第一個 smoke 嘗試因 ledger 參數命名錯誤在開影片前失敗，已修正、補測試，原 partial 目錄／零影片存取 trace 保留。

[Smoke 報告](data_quality/blur_screening_smoke_v3.md)／[JSON](data_quality/blur_screening_smoke_v3.json) 提供逐支耗時、容量、候選、分布、resume overhead 與 7,193 支 smoke extrapolation；[tmux runbook](../development/data_quality/runbooks/blur_train_v3_tmux.md) 提供主機正式命令、進度、detach/attach、中斷／resume／retry／rebuild／verify 及版本記錄。全量前需由使用者 review、commit/push v3；本輪未使用 `--confirm-full-train`，未執行 full-train dry-run 或掃描。

Scanner implementation complete；完整 train 執行等待使用者在主機啟動。沒有資料刪除、排除、cleaned dataset、增強、特徵擷取、訓練、評估或 checkpoint loading。持續低品質只是診斷；人工真值、來源混雜、主體品質、錯誤成本與清理政策未決。A1/A2/B/C frozen，Gate 1C／原 Experiment D paused。


## 2026-10-09 - Data Quality v1／v2／v3 Documentation and Host Workflow

### Scope and entry points

本輪開始於 `main` / `76500256cd424b766db49116acebeae03c5a92cc`，工作樹乾淨；本機 `origin/main` 指向相同 commit（未連線更新遠端）。v1／v2／v3 已提交，只整理文件、索引、連結及主機說明，不修改演算法、設定、manifest、報告數值或歷史結果。

- [唯一技術入口](../development/data_quality/README.md)：統一三版定位、研究證據、train-only 邊界及 v3 執行流程；v1 原重現段落保留。
- [研究文件索引](data_quality/README.md)：新增三版研究問題、輸入／方法／產出／結論、凍結規則來源、未決政策及閱讀順序；docs 總索引補上入口。
- README_v2 原位保留，頂端標示歷史重現並連回統一入口。[v3 runbook](../development/data_quality/runbooks/blur_train_v3_tmux.md) 增加主機 checklist、CLI 對照、config hash、dry-run 邊界、verify exit 紀錄與 publish 限制；正式 scan／resume／retry／rebuild 命令參數與原版相同。

### Allowed verification and preservation

執行三個 v3 CLI 的 `--help`，解析全部三份 YAML，核對固定 manifest／split 路徑存在、output／runtime 由既有 `.gitignore` 排除且位於獨立 artifacts。Full-train dry-run 使用固定 config／manifest、明示全量 output、`--scope full-train --confirm-full-train --seed 20261009 --dry-run`，exit 0、`DRY_RUN_PASS`、selected=7,193；video opens/probes/decodes 全為 0，validation/test opens/probes/decodes 亦為 0，未建立 full-train output。依據為 CLI 回報及 dry-run 在 output／context／run 前返回、guard 只讀 metadata 的靜態核對；本輪未產生獨立 syscall trace，也未檢查原影片存在性或 identity。

本機 Markdown 連結與三版文件可達性通過；runbook 的 9 條 v3 CLI 命令與 argparse 選項核對通過。正式 scan／resume／retry／rebuild 的 script 路徑、config、manifest、output、seed 及旗標與 HEAD 原文一致。沒有執行 tests、smoke、掃描、verify 重算、publish、訓練、評估、清理或資料刪除。

開始時全部 135 個 tracked files（含 31 個 Data Quality 檔案）記錄 SHA-256，本機快照為 `/tmp/basketball-dq-before-sha256.json`。最終只允許 5 份既有文件變更：development/data_quality/README.md、README_v2.md、runbooks/blur_train_v3_tmux.md、docs/README.md、本進度紀錄；其餘 130 個 tracked files／28 個 Data Quality 檔案 hash 不變。新增 docs/data_quality/README.md；三版方法、Markdown 報告、JSON、scripts、configs、manifests 及 frozen evidence 均未改。進度舊內容保持前綴不變，`git diff --check` 通過。Dry-run JSON 保存在 `/tmp/basketball-dq-full-train-dry-run.json`；未 git add、commit 或 push。

### Host readiness and publish gap

Scanner 可進入主機全量候選掃描準備；操作者仍須完成 runbook checklist，確認已 push 的版本、乾淨工作樹、config hash、主機 dry-run 與新 output／有效 resume。Dry-run 不檢查影片可讀性、環境工具齊備或 output/resume 可用性，不能取代正式主機 preflight。

實際 publisher 只接受固定 20 支 smoke 並拒絕覆寫既有 smoke 報告，沒有 full-train publish CLI。本輪不改程式，故 checklist 的全量 publish 明列待辦；全量完成先 verify、保存 run／runtime logs，發表流程另行實作。此缺口不阻擋候選掃描，但「scan → verify → publish」尚未全部具備。Candidate 不等於確認模糊或排除名單，不直接刪除任何 candidate 資料；正式清理政策仍未完成。

## 2026-10-09 — Blur Screening v3.1 publisher／verification／runbook

- 擴充 `publish_blur_train_v3.py` 為 smoke/full-train 共用 publisher：精確 ID 集合核對、四種完整度狀態、artifact hashes、bounded-memory aggregate rebuild、candidate 分布、同源 Markdown／JSON 與 exclusive-create 輸出。
- `verify_blur_train_v3.py` 改為共用同一份 fail-closed verification core；full train 預設要求 `COMPLETE`，`COMPLETE_WITH_FAILURES`／`PARTIAL` 非零，`INVALID` 永不發表。
- 新增 13 個 derived-only publisher fixtures，涵蓋 complete、missing completion、duplicate、unexpected、held-out、config mismatch、corrupt hash、partial、unresolved failure、aggregate mismatch、拒絕覆寫、Markdown/JSON 一致性與既有 smoke 相容性。
- 既有 smoke 核心值保持 20 支、3,055 幀、54 candidate frames、1.7676%、5 candidate segments、0 decode failure；沒有重新開啟或 decode 影片。
- 更新 Ubuntu/tmux runbook 為 dry-run → scan → resume → retry → rebuild → verify COMPLETE → ignored runtime publish → Markdown/JSON/hash 核對的實際命令。
- 本輪未執行 smoke/full-train scanner、未開啟影片、未更改 v2/v3 偵測參數、未修改資料或 A1/A2/B/C，亦未 git add/commit/push。

## 2026-10-10 — Quality-aware Clip Sampler v1

### Sidecar 與 provenance

由 v3.1 `COMPLETE` full-train scanner metadata 匯出精簡 tracked sidecar；沒有讀取影片。Candidate segment CSV 共 1,798 列／18,390 candidate frames，video summary 共 7,193 列，ID 集合精確等於固定 train、validation/test ID 為 0。1,105,146 source frames、1,698 affected、5,495 unaffected；16/32/64 zero-overlap availability 獨立重算為 7,193／7,193／7,188。

Metadata JSON 記錄來源 Markdown/full JSON/verification/config/manifest SHA-256、scanner/publisher commit `f9989c0`、run identity `19372422e554779a09d54ca93e4927bf91d5665af5e6c127904727d4b5edc02a`、seed 與 CSV hashes；tracked sidecar 不含本機絕對路徑。Exporter 從 ignored runtime artifacts 重生並以 `--check-only` byte-for-byte 通過；5.3 MB source/verification JSON 未加入 Git。

### Sampler 與 smoke

新增 train-only `quality_aware_sampler_v1`：列舉合法連續 `[start,start+clip_length)`，優先 cost=0；沒有乾淨窗口時只從 minimum candidate-overlap starts 選擇並記錄 fallback/reason/overlap。SHA-256 seed material 包含 global seed、video ID、epoch、clip index、policy version 與 start；不使用 Python `hash()`。多 clip 優先 unique start 並最大化時間距離，candidate margin v1 固定為 0；越界、非法／重疊 segments 與 clip 長於影片均 fail closed。

全 7,193 支、64-frame、epoch 0 audit：7,188 candidate-free clips、5 fallback videos/clips，fallback overlap 分別 1、8、5、5、1，合計 20 frames。固定 20 支 smoke 為 Make/Miss 各 10、20 sources，含 8 affected-clean、7 unaffected、全部 5 fallback；15 clips overlap=0，smoke fallback overlap 合計同為 20。Tracked smoke manifest、JSON、Markdown/timeline 由 publisher 重生核對一致；timeline 僅由 frame index 生成。

### 驗證與邊界

Canonical `yolo11` 環境的全部 data-quality tests 為 65/65 PASS，其中新 sampler 21 tests 涵蓋需求的 20 類 edge/safety cases。Python syntax/import、CLI help/dry-run、export/publish check-only、sidecar coverage/totals、16/32/64 availability、完整 selection overlap 獨立重算、smoke 重現、CSV/JSON/Markdown、tracked absolute-path、Markdown links 與 `git diff --check` 均通過。

Full metadata dry-run 的獨立 strace 位於 ignored `development/data_quality/artifacts/quality_aware_sampler_v1/syscalls-dry-run.log`，SHA-256 `3d8e7053b2dffe5b9ffa7072150ea84068346839c8265275b7d847fbacce1088`；122 行 syscall 中 dataset/archive/`.mp4` path matches=0，ffprobe/ffmpeg/cv2/VideoCapture matches=0，唯一 exec 為 Python audit CLI。因此 video opens/probes/decodes=0，validation/test video access=0。

沒有修改原始影片、label、manifest、split、v2/v3 門檻或 frozen baseline；沒有刪格、拼接、augmentation、I3D/YOLO、訓練或評估。下一輪 I3D DataLoader 可依 `video_id` 查 sidecar、呼叫 sampler 取得連續 `[start,end)`，再由既有 decoder 載入並傳遞 fallback/overlap metadata；本輪到此停止。沒有 git add、commit 或 push。
