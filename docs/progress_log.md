# Project Progress Log

## Current Status

Current phase:
C3D implementation audit / reproduction specification completed

Current model:
Modified C3D-style

Latest test accuracy:
74.79%

Current reference:
C3D - Tran et al., ICCV 2015

Next planned task:
Reference-faithful C3D reproduction

Last updated:
2026-09-29


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
