# Basketball-51 影片理解研究現況與證據統整報告 (Research Baseline v1.3)
——供 Experiment D 與後續 Pre-Release 階段參考之專案綜合基線文件

**Research Baseline v1.3**  
**Status**: Documentation Baseline Frozen  
**Source state**: 2026-10-01 project results  
**Reviewed**: 2026-10-07  

---

## 一頁式 Executive Summary

本報告為 Basketball-51 籃球影片 Make/Miss 分類專案之全方位研究現況與證據統整文件（正式入庫 Research Baseline v1.3 版）。本報告旨在建立客觀、嚴謹且完全基於專案原始檔案（`readme.md`、`progress_log.md`）、奠基論文（C3D、I3D）與教授研究構想 PDF（`TOPIC 1.pdf`）之事實基準 `COMPLETED-PROJECT-FACT`。報告嚴格排除未經證實之推測、第三方未知程式碼與未授權之外部假設，對所有無法由來源證實之細節標示「來源未指定」`KNOWN-PROJECT-LIMITATION`，並對論文推導項目標示 `DERIVED-FROM-PAPER`。

### 核心執行摘要重點

1. **已完成階段之邊界限制**：專案目前已完整完成 A1、A2、B、C 四組模型之訓練與一次性封存 Test 集評估 `COMPLETED-PROJECT-FACT`。所有模型均輸入完整影片（Full-Video），可能包含球離手後的飛行、碰框、進球與球員反應，**絕不能宣稱具備球離手前（Pre-Release）預測能力** `KNOWN-PROJECT-LIMITATION`。
2. **正式 Test 表現與數據稽核邊界**：A2、B、C 已驗證使用完全相同的 1,539 支 test IDs（`2c6c37cc...`）；A1 舊版結果具有相同 manifest hash（`e7bba1ac...`）與 1,539 支 test count，但未保存 test-ID 及 split-summary hash，因此無法以同等強度重新稽核其 test identity `KNOWN-PROJECT-LIMITATION`。正式 Test 報告準確率依序為 **A2 (77.39%) > A1 (74.79%) > B (73.42%) > C (60.49%)** `PROJECT-RESULT`。A2、B、C 為完整流程比較而非單變因對照，**絕不能簡化歸因宣稱稀疏取樣優於連續取樣，或 Modified C3D 優於 Reference C3D** `KNOWN-PROJECT-LIMITATION`。
3. **I3D 論文轉移學習與架構細節澄清**：
   - **Stream 與 Fusion 關係**：分別獨立訓練 RGB-I3D stream 與 Flow-I3D stream，Two-Stream I3D 係於測試時融合兩個 streams 之 predictions `PAPER-SPECIFIED`。
   - **Kinetics 訓練路徑**：論文研究兩種 Kinetics 訓練路徑：(1) 由 inflated ImageNet weights 初始化後訓練 Kinetics；(2) 不使用 ImageNet initialization、直接於 Kinetics from scratch 訓練 `PAPER-SPECIFIED`。
   - **數據 protocol 嚴格分開**：Table 3 Kinetics Top-1 達到 74.2%（ImageNet→Kinetics）與 71.6%（Kinetics from scratch）；Table 4 Split 1 Full-FT 於 UCF-101 達 98.0/97.6%、HMDB-51 達 81.2/81.3%（with/without ImageNet init）；Table 5 3-Split 平均於 ImageNet+Kinetics 達 UCF-101 98.0%、HMDB-51 80.7% `PAPER-RESULT`。
   - **時間維度推導與 Padding**：64-frame 主幹時間維度 `64 → 32 → 16 → 8` 屬於在 length-preserving padding 假設下之條件式推導 `DERIVED-FROM-PAPER`；Figure 3 未標示 stride 者依 caption 均為 1，故 final `2×7×7 Avg-Pool` 時間 stride 為 1，Valid padding 下時間軸由 8 降至 7，Same padding 下則維持 8 `DERIVED-FROM-PAPER`。Prediction averaging 層級（logits 或 probabilities）在論文中「來源未指定」`KNOWN-PROJECT-LIMITATION`。
4. **教授研究構想與專案延伸之區分**：`TOPIC 1.pdf` 提出結合 YOLO 玩家偵測、MixNetI3D、RL 影格選擇與 SpinalNet 之策略預測架構 `PROFESSOR-DIRECTION`，其內文 PDF 之 `XXXX` 與 `XX.X%` 為佔位符號而非實驗數據 `KNOWN-PROJECT-LIMITATION`；球離手前預測（Pre-Release Pipeline）為專案長期目標 `PROJECT-RESEARCH-GOAL`，非教授 PDF 之明文要求。
5. **Experiment D 範疇與兩階段前置研究 Gate**：Experiment D 預定研究 I3D on Basketball-51；其是否延續 full-video classification，以及與未來 pre-release 階段的邊界，仍須在 protocol 中正式凍結 `UNRESOLVED-PROJECT-DECISION`。進入 Experiment D 前，須嚴格依循兩階段 Gate（Gate 1: 通用時間維度/Provenance/RGB 可行性稽核；Gate 2: 光流候選時之 benchmark 與儲存估算），且稽核範圍僅限 8,740 支 train/validation 影片，保持 1,539 支 test 影片絕對封存 `UNRESOLVED-PROJECT-DECISION`。

---

## 1. 研究問題與範圍

### 1.1 已完成階段：Full-Video Make/Miss Classification
專案目前已完整完成 Experiment A1、A2、B、C 四組實驗之訓練與一次性正式 Test 評估 `COMPLETED-PROJECT-FACT`。上述四組實驗之核心任務均為對 **完整影片（Full-Video）** 進行二分類（Make=1 命中 / Miss=0 未進）`COMPLETED-PROJECT-FACT`。

### 1.2 長期目標：Pre-Release Prediction
專案之長期研究目標為僅利用 **球離手前（Pre-Release）** 之影格畫面進行投籃結果預測 `PROJECT-RESEARCH-GOAL`。然而，教授研究題目 PDF（`TOPIC 1.pdf` p. 1）提出之研究構想為基於策略預測、玩家偵測（YOLO）、特徵擷取（MixNetI3D）、強化學習選框（3D-XXXXNet）與 SpinalNet 之混合深度學習架構 `PROFESSOR-DIRECTION`；`TOPIC 1.pdf` 內文並未明文要求 Pre-Release Prediction，故 Pre-Release Prediction 屬專案團隊自身設定之長期目標，不應自動歸類為教授 PDF 之明確要求 `KNOWN-PROJECT-LIMITATION`。

### 1.3 證據範圍與絕不混用之邊界
1. **完整影片資訊洩漏風險**：A1、A2、B、C 模型於輸入完整影片時，均可能觀察到球離手後的飛行軌跡、碰框、籃網變化以及球員與觀眾之投籃後反應 `KNOWN-PROJECT-LIMITATION`。
2. **預測能力之不可外推性**：現有 A1/A2/B/C 之任何 Validation 或 Test 準確率，**均不能稱為 Pre-Release Prediction 結果，亦不能作為模型具備球離手前預測能力之證據** `KNOWN-PROJECT-LIMITATION`。
3. **標註缺口（Release-Frame Annotation Gap）**：Basketball-51 資料集目前 **完全沒有** 可直接使用之球離手影格（Release Frame）時間點標註；影片檔名中之時間戳記僅為擷取時間，非球離手時間 `KNOWN-PROJECT-LIMITATION`。
4. **凍結狀態（Frozen Status）**：Experiment A1、A2、B、C 之 Checkpoint、前處理流程、資料切分與訓練超參數均已完全凍結，不可根據 Test 結果回頭調整或重新訓練 `COMPLETED-PROJECT-FACT`。

---

## 2. Basketball-51 資料集狀態

### 2.1 原始影片與清理紀錄
- **原始影片總數**：10,311 支 RGB MP4 影片（解析度 320×240，檔案總大小約 6.1 GB）`COMPLETED-PROJECT-FACT`（依據 `readme.md` §2.1; `progress_log.md` 2026-09-27）。
- **排除影片數**：
  - **Outcome-Conflicting Videos**：排除 6 支標籤與內容衝突之影片 `COMPLETED-PROJECT-FACT`。
  - **Exact Duplicate Copies**：排除 26 支 SHA-256 完全相同且標籤一致之重複複本 `COMPLETED-PROJECT-FACT`。
- **最終可用影片總數**：10,279 支 `COMPLETED-PROJECT-FACT`。

### 2.2 Train / Validation / Test 切分數量與證據強度說明
固定切分種子（Seed 20260927），依據來源代碼分組（Source Grouping）進行切分 `COMPLETED-PROJECT-FACT`（依據 `readme.md` §2.2）：
- **Train Set**：7,193 支 `COMPLETED-PROJECT-FACT`
- **Validation Set**：1,547 支 `COMPLETED-PROJECT-FACT`
- **Train + Validation 合計**：8,740 支（為 Experiment D 前置稽核與設計所使用的唯一影片範圍）`COMPLETED-PROJECT-FACT`
- **Test Set**：1,539 支（包含 Miss 731 支、Make 808 支；完全封存，不參與 Experiment D 設計）`COMPLETED-PROJECT-FACT`
- **Test Identity 凍結狀態與驗證雜湊值**：
  - `dataset_manifest.csv` SHA-256: `e7bba1acd6028cf67cdf28251e69852e82cff201e13519ed05970ac483cce6b6` `COMPLETED-PROJECT-FACT`
  - `split_summary.json` SHA-256: `953da31313976763c8f5d597da5f0fe5a6a45e10f5d5edcd42fb0c120af2f7c5` `COMPLETED-PROJECT-FACT`
  - 排序後 Test-ID SHA-256: `2c6c37cc846179aea27e1b2a0726a4210c50a184357792202791ddd9752d3eba` `COMPLETED-PROJECT-FACT`
- **Test ID 稽核強度差異**：A2、B、C 已驗證使用完全相同的 1,539 支 test IDs（`2c6c37cc...`）；A1 舊版結果具有相同 manifest hash 與 test count，但未保存 test-ID 及 split-summary hash，因此無法以同等強度重新稽核其 test identity `KNOWN-PROJECT-LIMITATION`。

### 2.3 資料集已知限制
1. **Game-Level Holdout 未核實**：目前切分係依據影片檔名之來源代碼（v 代碼）進行群組隔離，但代碼尚未經原始比賽官方清單逐一核實，故 **尚未能宣稱已達成真正的 Game-Level Split** `KNOWN-PROJECT-LIMITATION`。
2. **Duplicates 處理差異**：專案已透過 SHA-256 精確排除 26 支 Exact Duplicates；然而對於經過重新編碼、轉碼或微小剪輯之 **Near Duplicates**，SHA-256 無法識別，目前尚未進行定量排除 `KNOWN-PROJECT-LIMITATION`。

---

## 3. A1 / A2 / B / C 實驗矩陣與比較

### 3.1 完整實驗矩陣對照表

| 比較項目 | Experiment A1 `COMPLETED-PROJECT-FACT` | Experiment A2 `COMPLETED-PROJECT-FACT` | Experiment B `COMPLETED-PROJECT-FACT` | Experiment C `COMPLETED-PROJECT-FACT` |
| :--- | :--- | :--- | :--- | :--- |
| **研究目的** | 初始 Modified C3D 基線 | A1 之 Extended-Training 延伸 | Reference C3D 於本專案之 Adaptation | Continuous Clip 下驗證 Modified Backbone |
| **Backbone 網絡** | Modified C3D-style (8 Conv3D) | 同 A1 (Modified C3D-style) | Reference C3D (8 Conv3D) | Modified C3D-style (同 A1/A2 卷積骨幹) |
| **初始化方式** | 從零初始化 (Seed 20260927) | 由 A1 Best Checkpoint Warm-Start | 從零初始化 (Prototxt Filler 風格) | 從零初始化 |
| **Temporal Sampling** | 全片 16 bins，各採 1 frame (稀疏) | 同 A1 (全片 16 bins 稀疏採樣) | 連續 16-frame clips (Stride 8, Overlap 8) | 連續 16-frame clips (Stride 8, Overlap 8) |
| **Input Shape** | 每影片 `[3, 16, 96, 96]` | 同 A1 (`[3, 16, 96, 96]`) | 每 Clip `[3, 16, 112, 112]` | 每 Clip `[3, 16, 112, 112]` |
| **Spatial Preprocessing** | Resize 比例補黑邊 96×96 (Letterbox) | 同 A1 (96×96 Letterbox) | Resize 128×171, Crop 112×112 | Resize 128×171, Crop 112×112 |
| **Normalization** | RGB 正規化至 `[-1, 1]` | 同 A1 (`[-1, 1]`) | RGB 正規化至 `[-1, 1]` (Project Adaptation) | RGB 正規化至 `[-1, 1]` |
| **BN / Pooling / GAP** | 每 Conv 後加 BN3d；5 Pool；加 GAP | 同 A1 (含 BN3d、5 Pool、GAP) | **無 BN**；5 Pool (ceil_mode=True)；**無 GAP** | 每 Conv 後加 BN3d；5 Pool；加 GAP (256-D) |
| **FC / Head 結構** | Linear(256,128) -> ReLU -> Drop(0.3) -> FC(128,2) | 同 A1 | fc6 位於有效可訓練路徑；fc7 及原 fc8 凍結且在影片級 Loss 外；接新 Linear(4096,2) | A1 head 與 B fc6/fc7 均不用；新設 Linear(256,2) |
| **Clip 特徵提取** | 無 (整支影片單次前向傳遞) | 無 (整支影片單次前向傳遞) | 提取 Clip fc6 4096-D (ReLU後/Dropout前) | 提取 Clip GAP 256-D 特徵 |
| **Video Aggregation** | 無 (單一影片特徵) | 無 (單一影片特徵) | 平均同影片所有 Clips 之 fc6 + L2 正規化 | 平均同影片所有 Clips 之 GAP + L2 正規化 |
| **Classifier & Loss** | Linear(128,2) + 影片級 CrossEntropy | Linear(128,2) + 影片級 CrossEntropy | Linear(4096,2) + 影片級 CrossEntropy | Linear(256,2) + 影片級 CrossEntropy |
| **Parameter Count** | 6,951,874 (全數可訓練) | 6,951,874 (全數可訓練) | Reference 78,003,970 + Head 8,194 (有效訓練 61,222,658) | 6,919,234 (全數可訓練) |
| **Training Strategy** | AdamW (LR 1e-3), Batch 16, 5 Epochs | AdamW (LR 1e-3), Warm-start, Early stop @ Ep 28 | AdamW (LR 1e-4), ReduceLROnPlateau, Early stop @ Ep 32 | AdamW (LR 1e-4), ReduceLROnPlateau, Early stop @ Ep 11 |
| **Best Epoch** | 5 `PROJECT-RESULT` | 21 `PROJECT-RESULT` | 25 `PROJECT-RESULT` | 4 `PROJECT-RESULT` |
| **Best Val Loss / Acc** | 0.525438 / 0.760181 `PROJECT-RESULT` | 0.440128 / 0.797027 `PROJECT-RESULT` | 0.534533 / 0.750485 `PROJECT-RESULT` | 0.659395 / 0.615385 `PROJECT-RESULT` |
| **Test Loss** | 未記錄 (舊版 Metric 未存) `PROJECT-RESULT` | 0.501485 `PROJECT-RESULT` | 0.567158 `PROJECT-RESULT` | 0.662981 `PROJECT-RESULT` |
| **Formal Test Accuracy** | 0.747888 (74.79%) `PROJECT-RESULT` | **0.773879 (77.39%)** `PROJECT-RESULT` | 0.734243 (73.42%) `PROJECT-RESULT` | 0.604938 (60.49%) `PROJECT-RESULT` |
| **Test Precision (Make)**| 0.711694 `PROJECT-RESULT` | 0.757271 `PROJECT-RESULT` | 0.748443 `PROJECT-RESULT` | 0.626582 `PROJECT-RESULT` |
| **Test Recall (Make)** | 0.873762 `PROJECT-RESULT` | 0.837871 `PROJECT-RESULT` | 0.743812 `PROJECT-RESULT` | 0.612624 `PROJECT-RESULT` |
| **Test F1 Score (Make)**| 0.784444 `PROJECT-RESULT` | 0.795535 `PROJECT-RESULT` | 0.746120 `PROJECT-RESULT` | 0.619524 `PROJECT-RESULT` |
| **Confusion Matrix** | `[[445, 286], [102, 706]]` `PROJECT-RESULT` | `[[514, 217], [131, 677]]` `PROJECT-RESULT` | `[[529, 202], [207, 601]]` `PROJECT-RESULT` | `[[436, 295], [313, 495]]` `PROJECT-RESULT` |
| **凍結狀態與稽核強度**| 已凍結 `COMPLETED-PROJECT-FACT`；缺少 test-ID 及 split-summary hash `KNOWN-PROJECT-LIMITATION` | 已凍結（具完整 hash 驗證）`COMPLETED-PROJECT-FACT` | 已凍結（具完整 hash 驗證）`COMPLETED-PROJECT-FACT` | 已凍結（具完整 hash 驗證）`COMPLETED-PROJECT-FACT` |

### 3.2 實驗關鍵細節說明
1. **A1 至 A2 之 Warm-Start 性質**：A2 載入 A1 第 5 Epoch 之權重進行續訓，但因 A1 當時未儲存 Optimizer 與 RNG 狀態，故 A1→A2 為 **WARM_START_ONLY**，非 True Resume `COMPLETED-PROJECT-FACT`（依據 `readme.md` §3.2; `progress_log.md` 2026-09-30）。
2. **Experiment B 之模型定位**：Experiment B 係 Reference C3D 於 Basketball-51 之 Adaptation，**並非對原論文 Benchmark 之重現（Reproduction）** `COMPLETED-PROJECT-FACT`。
3. **Experiment B 之有效 Loss 路徑**：fc6 位於有效且可訓練路徑；fc7 及原 fc8 凍結且位於影片級 Loss 路徑之外。在對影片所有 Clips 之 fc6 (4096-D) 進行平均與 L2 正規化後，直接連接新的 `Linear(4096, 2)` 分類器計算 CrossEntropyLoss `COMPLETED-PROJECT-FACT`。
4. **Experiment C 之架構組合**：Experiment C 結合 A1/A2 之 Modified C3D Backbone（含 BN3d 與 GAP 256-D）與 Experiment B 之連續重疊 Clip 處理管線 `COMPLETED-PROJECT-FACT`。
5. **正式 Test 表現最高者與因果解讀禁忌**：在四組實驗中，正式 Test Accuracy 最高者為 **Experiment A2 (77.39%)** `PROJECT-RESULT`。然而，**嚴禁將此排名轉化為單變因因果結論** `KNOWN-PROJECT-LIMITATION`：
   - A2 與 C 之間除採樣方式（Sparse vs Continuous）外，尚存在空間解析度（96×96 Letterbox vs 112×112 Crop）、分類 Head 結構、影片級聚合方式與初始化（Warm-start vs Scratch）等多重混雜因素（Confounders）`KNOWN-PROJECT-LIMITATION`。
   - C 與 B 之間則存在 Backbone 結構（Modified vs Reference）、通道數、BN 之有無、特徵維度（256-D vs 4096-D）與訓練歷程等差異 `KNOWN-PROJECT-LIMITATION`。
   - 因此，**絕不能僅憑此結果宣稱「Sparse Sampling 本質上優於 Continuous Sampling」或「Modified C3D 本質上優於 Reference C3D」** `KNOWN-PROJECT-LIMITATION`。

---

## 4. C3D 原論文理解 (Tran et al., ICCV 2015)

以下內容嚴格依據 `c3d_tran_2015.pdf` 原始論文整理：

### 4.1 核心架構與原論文設定
1. **3D Convolution 研究動機**：2D 卷積套用於多影格時會在首層觀察到時間特徵被立即壓縮；3D 卷積與 3D Pooling 則能在時空雙軸同時維持階層式特徵建構（p. 4489, Abstract; p. 4491, §3.1, Fig. 1）`PAPER-SPECIFIED`。
2. **16-Frame Clips 輸入**：論文設定輸入長度為 16 影格，解像度 resize 至 128×171 後隨機裁切為 16×112×112（p. 4491, §3.1）`PAPER-SPECIFIED`。
3. **`3×3×3` Convolution 最佳化**：在論文所進行的 architecture search 範圍內，均勻採用 `3×3×3` 卷積核（d=3）取得最佳實驗表現與合理效率（p. 4491–4492, §3.1, §3.2, Fig. 2）`PAPER-SPECIFIED`。
4. **Temporal Pooling 設計**：Pool1 使用 `1×2×2` kernel（stride `1×2×2`）以保留早期時間訊號；Pool2 至 Pool5 均採用 `2×2×2` kernel（stride `2×2×2`）（p. 4492, §3.3, Fig. 3）`PAPER-SPECIFIED`。
5. **Sports-1M Pretraining**：C3D 在包含 1.1M 支體育影片之 Sports-1M 資料集上進行監督式預訓練，最佳模型達 85.2% Video Top-5 Accuracy（p. 4492–4493, §3.3, Table 2）`PAPER-RESULT`。
6. **fc6 Feature Extraction & Overlapping Clips**：論文在**影片描述子抽取階段**採用連續重疊 16-frame clips（重疊 8 frames / Stride 8），提取 fc6 的 4096 維特徵；C3D 的原始訓練並非所有情況都固定以 Stride 8 滑窗取樣（p. 4493, §3.3）`PAPER-SPECIFIED`。
7. **Video-Level Feature Aggregation**：將一支影片中所有 Clips 的 fc6 特徵進行平均（Averaging），隨後執行 L2 Normalization，形成 4096 維影片描述子（Video Descriptor），並搭配 Linear SVM 進行分類（p. 4493, §3.3, §4）`PAPER-SPECIFIED`。

### 4.2 C3D 在 Basketball-51 專案中之對照與未重現細節
- **被 Experiment B 採用之內容**：連續 16-frame clips（Stride 8, Overlap 8）、Reference C3D 8 Conv3D + 5 MaxPool3d 拓撲、fc6 4096-D 提取、同影片 Clips 平均 + L2 正規化 `COMPLETED-PROJECT-FACT`。
- **Basketball-51 Project Adaptation**：RGB 正規化至 `[-1, 1]`；fc6 特徵匯出點設於 ReLU 後、Dropout 前；使用 PyTorch `ceil_mode=True` 處理 Pool5；尾段不足 16 frames 時捨棄 `COMPLETED-PROJECT-FACT`。
- **未完整重現之原論文細節**：原始 Sports-1M 預訓練 Checkpoint 及其精確 Mean File / 色彩順序未在專案中載入；原論文之 10 隨機 Crop 測試策略未採納 `KNOWN-PROJECT-LIMITATION`。

---

## 5. I3D 原論文理解 (Carreira & Zisserman, CVPR 2017)

以下內容嚴格依據 `1705.07750v3.pdf` 原始論文整理：

### 5.1 核心技術與論文設定
1. **Inflated Inception-v1 架構**：將 2D Inception-v1 之 2D 卷積核與 Pooling 核展開（Inflate）為 3D（例如 $N \times N \rightarrow N \times N \times N$），使其能無縫擷取時空特徵（p. 2, §2.4, Fig. 3）`PAPER-SPECIFIED`。
2. **Boring-Video Fixed Point**：為利用 ImageNet 2D 預訓練權重，將 2D 權重沿時間軸重複 $N$ 次並除以 $N$。此舉確保當輸入為靜止重複影格（Boring Video）時，3D 卷積之輸出與 2D 卷積完全等價（p. 3, §2.4）`PAPER-SPECIFIED`。
3. **Kinetics 訓練路徑研究**：論文研究兩種 Kinetics 訓練路徑：(1) 由 inflated ImageNet weights 初始化後訓練 Kinetics；(2) 不使用 ImageNet initialization、直接於 Kinetics from scratch 訓練（p. 1–3, §1, §2.4, §3）`PAPER-SPECIFIED`。
4. **RGB-I3D stream, Flow-I3D stream 與 Two-Stream I3D 融合**：
   - 分別獨立訓練 **RGB-I3D stream**（輸入 RGB 影格，例如 64 張連續 RGB frames，每張 224×224）與 **Flow-I3D stream**（輸入 TV-L1 光流影格）`PAPER-SPECIFIED`（p. 2, §2.4; p. 3, §2.5）。
   - **Two-Stream I3D** 於測試階段將兩個獨立 streams 之 predictions 進行融合平均（Averaging predictions），於 UCF-101 達 98.0%、HMDB-51 達 80.9%（p. 2, §2.4; p. 6-7, §5.1, Table 5）`PAPER-RESULT`。
5. **64-Frame Snippets & 25 FPS**：訓練時採用 25 FPS 下之 64-frame continuous snippets（時間足跡 2.56 秒）；RGB-I3D 輸入為 64 張連續 RGB frames (每張 224×224)，Flow-I3D 採用對應之光流輸入 `PAPER-SPECIFIED`。
6. **Temporal Receptive Field Pacing**：前兩層 Max-Pool 使用 `1×3×3` kernel（stride 1 in time），避免過早融合時間維度；後續 Pool 採用對稱 `2×3×3` / `2×2×2`（stride 2 in time）（p. 3, §2.4, Fig. 3）`PAPER-SPECIFIED`。
7. **Whole-Video Convolutional Testing & Averaging**：測試時將模型以全卷積方式套用於整支影片（250 frames, ~10 秒），並對時間維度的預測進行平均（p. 3–4, §2.5, Table 1）`PAPER-SPECIFIED`。
8. **Shorter-Video Looping**：若影片長度不足模型輸入影格數，論文採取重複循環（Looping）影片至滿足長度為止（p. 3, §2.5）`PAPER-SPECIFIED`。
9. **TV-L1 Optical Flow**：光流串流採用 TV-L1 演算法計算 Optical Flow（p. 3, §2.5）`PAPER-SPECIFIED`。

### 5.2 明確保留之推導與論文未完備指定事項
1. **主幹時間維度推導標記 (`DERIVED-FROM-PAPER`)**：在 Length-Preserving Padding 假設前提下，64-frame 輸入經過 Inception-v1 主幹層級時，時間維度可條件式推導為 `64 → 32 → 16 → 8` `DERIVED-FROM-PAPER`。此為由卷積/Pooling 時間 Stride 設定導出之結果，非論文明列之 Tensor Shape `DERIVED-FROM-PAPER`。
2. **Figure 3 Caption 宣告與 Stride/Padding 評估**：
   - Figure 3 caption 明確指出未標示 Stride 之 Convolution 與 Pooling 其 Stride 均為 1 `PAPER-SPECIFIED`。
   - 因此，末端 **Final `2×7×7 Avg-Pool`** 之 Temporal Stride 明確為 1 `DERIVED-FROM-PAPER`。
   - 尚存未確定性者為 Temporal Padding：
     - 若採用 **Temporal Valid Padding**，輸入 8 個時間影格經 `2×7×7` pool 後，時間維度變為 **7**（即 $8 - 2 + 1 = 7$）`DERIVED-FROM-PAPER`。
     - 若採用 **Temporal Same Padding**，時間維度則維持為 **8** `DERIVED-FROM-PAPER`。
3. **Prediction Averaging 計算層級不確定性**：論文內文僅提及「averaging predictions」，**未充分指定** 係平均 Pre-Softmax Logits 或 Post-Softmax Probabilities，標示為「來源未指定」`KNOWN-PROJECT-LIMITATION`。
4. **Normalization 之未確定性**：I3D 論文未完整指定可直接移植的精確 Normalization。候選 pretrained checkpoint 所需之像素尺度、色彩順序、mean／std 或其他 normalization 尚待 provenance audit，不得僅憑 I3D 論文推定 `UNRESOLVED-PROJECT-DECISION`。

---

## 6. ImageNet × Kinetics × Downstream Adaptation 證據矩陣

依據 I3D 原論文（`1705.07750v3.pdf`）Table 3、Table 4 與 Table 5，整理多維度轉移學習實驗數據：

### 6.1 Downstream Protocols 與比較維度說明
- **Original**：完全無 Kinetics Pretraining，直接在目標資料集（UCF-101 / HMDB-51）上進行訓練 `PAPER-SPECIFIED`。
- **Fixed**：Kinetics Pretrained Network 權重完全凍結，僅訓練目標資料集之最後一層分類器 Head `PAPER-SPECIFIED`。
- **Full-FT**：Kinetics Pretrained Network 在目標資料集上進行端到端全網路 Fine-Tuning `PAPER-SPECIFIED`。
- **ImageNet Initialization 效益分析**：ImageNet initialization 的效益依 architecture、dataset 及 protocol 而異；多數比較有改善，但並非所有設定都嚴格較高（例如 Table 4 中 HMDB Full-FT without ImageNet 81.3% 略高於 with ImageNet 81.2%，Table 5 中 HMDB Kinetics-only 80.9% 略高於 ImageNet+Kinetics 80.7%），因此不能宣稱它在所有 downstream settings 下都必然優勝 `PAPER-RESULT`。

### 6.2 Table 3、Table 4 與 Table 5 獨立數據矩陣

#### A. Table 3: Kinetics Pretraining 表現 (Top-1 Accuracy %)
> **說明**：Table 3 評估於 Kinetics Held-out Test Set 之 Top-1 準確率 `PAPER-RESULT`。

- **Two-Stream I3D (ImageNet + Kinetics Pretrained)**: **74.2%** Top-1 (RGB: 71.1%, Flow: 63.4%) `PAPER-RESULT`。
- **Two-Stream I3D (Kinetics-only Pretrained / From Scratch)**: **71.6%** Top-1 (RGB: 68.4%, Flow: 61.5%) `PAPER-RESULT`。

#### B. Table 4: Downstream Adaptation 精確數據矩陣 (UCF-101 & HMDB-51 Split 1)
> **格式與權威說明**： Table 4 每格數值嚴格表示為 `with ImageNet init / without ImageNet init`。Table 4 為 Split 1 結果，不可當作 Table 5 之 3-Split 平均結果 `PAPER-SPECIFIED`。

| 模型架構 (Architecture) | Downstream Protocol | UCF-101 (Split 1 Accuracy %) `PAPER-RESULT` | HMDB-51 (Split 1 Accuracy %) `PAPER-RESULT` |
| :--- | :--- | :--- | :--- |
| **(e) Two-Stream I3D** | Original | `93.4 / 88.8` | `66.4 / 62.2` |
| | Fixed | `97.7 / 97.4` | `79.7 / 78.6` |
| | Full-FT | **`98.0 / 97.6`** | **`81.2 / 81.3`** |
| **(c) Two-Stream (2D ConvNet)** | Original | `91.2 / 83.6` | `58.3 / 47.1` |
| | Fixed | `93.9 / 93.3` | `66.6 / 65.9` |
| | Full-FT | `94.2 / 93.8` | `66.6 / 64.3` |
| **(b) 3D-ConvNet (C3D-like)** | Original | `- / 51.6` (無 ImageNet 版本) | `- / 24.3` (無 ImageNet 版本) |
| | Fixed | `- / 76.0` | `- / 47.0` |
| | Full-FT | `- / 79.9` | `- / 49.4` |

#### C. Table 5: 最終 3-Split 平均結果與 C3D 比較
- **Two-Stream I3D (ImageNet + Kinetics Pretrained)**: UCF-101 **98.0%**, HMDB-51 **80.7%** `PAPER-RESULT`。
- **Two-Stream I3D (Kinetics-only Pretrained)**: UCF-101 **97.8%**, HMDB-51 **80.9%** `PAPER-RESULT`。
- **C3D single network (Sports-1M Pretrained, Tran et al.)**: UCF-101 **82.3%** `PAPER-RESULT`。
- **C3D ensemble + IDT (Sports-1M Pretrained, Tran et al.)**: UCF-101 **90.1%** `PAPER-RESULT`。

### 6.3 逐列說明「不能推論」的事項 (Non-Derivable Inferences)
1. **不能將 Table 4 (Split 1) 數據等同於 Table 5 (3-Split 平均)**：Table 4 僅反映單一切分點表現，Table 5 係 3 個 Splits 之平均值，兩者不可混用 `KNOWN-PROJECT-LIMITATION`。
2. **不能推論 Full-FT 高準確率即代表 Fixed 具備相同相對排名**：在某些架構（如 Two-Stream 2D）中，Fixed 與 Full-FT 報告數值非常接近（93.9% vs 94.2%），但在 3D-ConvNet 中 Full-FT 的報告數值高於 Fixed（79.9% vs 76.0%）`PAPER-RESULT`。
3. **不能將 UCF-101 / HMDB-51 動作分類結果外推至 Basketball-51**：UCF-101 為 101 類粗粒度動作，Basketball-51 為細粒度 Make/Miss 投籃結果分類，二者難度與時空語義完全不同 `KNOWN-PROJECT-LIMITATION`。
4. **不能推論 ImageNet 初始化在所有下游設定中均必然優於 Without ImageNet**：數據顯示大部分情況改善，但 HMDB-51 部分 Full-FT 與 3-Split 實驗中 Without ImageNet 或 Kinetics-only 反而持平或微幅偏高，效益隨架構與資料集改變 `KNOWN-PROJECT-LIMITATION`。

---

## 7. 教授方向與專案延伸

本章節建立三分法對照表，嚴格區分教授 PDF（`TOPIC 1.pdf`）明文內容與 Basketball-51 專案文件（`readme.md`、`progress_log.md`）之延伸，並標示實作狀態：

| 研究項目 / 技術構想 | TOPIC 1.pdf 明確提出 `PROFESSOR-DIRECTION` | Basketball-51 專案文件提出 `PROJECT-RESEARCH-GOAL` / `KNOWN-PROJECT-LIMITATION` | 目前實作狀態 `COMPLETED-PROJECT-FACT` |
| :--- | :--- | :--- | :--- |
| **YOLO 玩家偵測與追蹤** | **有**（明文：「利用『你只看一次』（YOLO）模型進行目標追蹤的玩家偵測」） | 有（列為未實作後續方向） | **尚未實作** |
| **MixNet / I3D 特徵擷取** | **有**（明文：「利用建議中的 MixNet 膨脹三維對集合網路 (MixNetI3D) ... 修改 MixNetCNN 利用 I3D」） | 有（列為未實作延伸，`docs/references.md` 記載） | **尚未實作** |
| **RL Frame Selection** | **有**（明文：「強化學習（RL）被用來從影片序列中選擇最相關的影格」） | 有（`dataset_analysis.md` 討論作為選框方案） | **尚未實作** |
| **SpinalNet 預測** | **有**（明文：「根據這些選定的框架，透過 SpinalNet 模型進行預測」） | 無（專案文件未包含 SpinalNet 規劃） | **尚未實作** |
| **球偵測 (Ball Detection)** | **無**（PDF 專注於玩家與策略預測） | 無（專案文件目前未具體規劃球偵測） | **尚未實作** |
| **SPL / 骨架姿態 (Skeleton)** | **無** | **有**（`progress_log.md` 標示 SPL skeleton stage 為未實作後續） | **尚未實作** |
| **Pre-Release Pipeline** | **無**（PDF 專注於全片策略與遊戲層級洞察） | **有**（`readme.md` 列為專案長期研究目標） | **尚未實作**（A1-C 均為 Full-Video） |
| **PDF Placeholders (`XXXX`, `XX.X%`)** | **有**（明文：「XXXX 神經網路」、「XX.X% 準確率」） | 有（`readme.md` 說明其為佔位文字） | **非正式結果**（僅為佔位符號） |

---

## 8. Experiment D 前的已知事項與未決事項

### 8.1 已知事項 (Knowns)
已由論文（C3D, I3D）或 Basketball-51 專案文件嚴格支持之事實：

1. **Experiment D 範疇預定**：Experiment D 預定研究 I3D on Basketball-51；其是否延續 full-video classification，以及與未來 pre-release 階段的邊界，仍須在 protocol 中正式凍結 `UNRESOLVED-PROJECT-DECISION`。
2. **I3D 尚未在 Basketball-51 實作**：Basketball-51 目前僅有 A1/A2/B/C 四組 C3D-style 實驗，I3D 管線尚未寫入任何程式碼 `COMPLETED-PROJECT-FACT`。
3. **A1/A2/B/C 已完全凍結**：四組實驗之 Checkpoint、前處理、Split 與正式評估結果已完全封存 `COMPLETED-PROJECT-FACT`。
4. **Basketball-51 沒有 Release-Frame 標註**：資料集目前缺乏球離手影格時間點標註 `KNOWN-PROJECT-LIMITATION`。
5. **I3D 原論文關鍵參數規格**：原論文採用 25 FPS 下之 64-frame snippets（2.56 秒）、Whole-Video Convolutional Testing 以及 Temporal Prediction Averaging `PAPER-SPECIFIED`。
6. **I3D 論文留存之未指定事項**：論文未標註部分 Padding，且未明言 Prediction Averaging 係指 Pre-Softmax 或 Post-Softmax `KNOWN-PROJECT-LIMITATION`。

### 8.2 未決事項 (Unresolved Items)
Experiment D 或後續階段 **尚未決定** 之選項（本報告嚴禁代為決定）`UNRESOLVED-PROJECT-DECISION`：

| 未決維度 | 尚未決定的具體選項與不確定性 `UNRESOLVED-PROJECT-DECISION` |
| :--- | :--- |
| **1. Temporal Scope 凍結** | 是否延續 Full-Video 分類，或在 Protocol 中明確定義與 Pre-Release 之邊界。 |
| **2. Modality 選擇** | 採 RGB-only、Flow-only 或 Two-Stream (RGB + Optical Flow)。 |
| **3. 初始化與預訓練路徑** | 採 Scratch 從零初始化、ImageNet 2D 初始化、Kinetics 3D 預訓練或其他路徑。 |
| **4. Pretrained Checkpoint 來源與 Normalization** | 候選 Checkpoint 來源血統、Kinetics-400/600 版本、像素尺度、色彩順序與 Mean/Std。 |
| **5. Temporal Length** | 採 16-frame, 32-frame, 64-frame 或全片連續影格。 |
| **6. FPS Resampling** | 是否統一重採樣至 25 FPS，或保留原始 FPS (23.976 / 25 / 29.97)。 |
| **7. 短影片處理政策** | 長度短於 64 frames 時採 Looping 重複、Padding 補零、Truncation 或捨棄。 |
| **8. Spatial Resolution** | 採 96×96 Letterbox, 112×112 Crop, 224×224 Crop 或 256×256。 |
| **9. Prediction Aggregation** | 採 Clip 特徵平均、Pre-Softmax Logits 平均或 Post-Softmax 概率平均。 |
| **10. Classifier Adaptation** | 採 Linear Layer、Multi-layer FC、Linear SVM 或端到端分類器。 |
| **11. Fine-Tuning 範圍** | 採 Frozen Backbone (Fixed)、Partial Fine-Tuning 或 Full Fine-Tuning。 |
| **12. Training Hyperparameters**| Learning Rate、Optimizer (AdamW / SGD)、Weight Decay、Batch Size、Patience。 |
| **13. Compute & Storage 需求**| VRAM 記憶體上限、GPU 吞吐量與 TV-L1 Optical Flow 磁碟儲存可行性。 |
| **14. Validation & Test Protocol**| Early Stopping 門檻、Validation 觸發頻率與一次性 Test 評估保護機制。 |

---

## 9. 下一步研究清單 (Pre-Experiment D Research Checklist)

本清單僅包含「研究、稽核與可行性分析工作」，分層次為兩個關鍵門檻（Gate），**嚴禁產生 Implementation Plan 或決定超參數** `UNRESOLVED-PROJECT-DECISION`：

### Gate 1：所有方案共同需要的前置稽核與決策 (Universal Feasibility & Audit Gate)
1. **Basketball-51 Train / Validation Temporal Feasibility Audit**：針對 8,740 支 train／validation 影片開展時間維度可行性稽核；不使用 1,539 支 test 影片的內容或分布制定 Experiment D `UNRESOLVED-PROJECT-DECISION`。
2. **FPS、Frame Count 與 Duration Distribution**：精確統計 8,740 支 train／validation 影片之原始 FPS（23.976, 25, 29.97）、總影格數與持續秒數分佈 `UNRESOLVED-PROJECT-DECISION`。
3. **短於 64 Frames 影片比例統計**：分別計算 8,740 支 train／validation 影片之：(1) 原始影片 raw frame count 少於 64 的比例；(2) 依 duration 換算／重採樣至 25 FPS 後，等效 frame count 少於 64 的比例，評估對 64-frame I3D 輸入之影響 `UNRESOLVED-PROJECT-DECISION`。
4. **I3D Pretrained Artifact Provenance Audit**：對候選 I3D 預訓練 Checkpoint 進行血統與規格稽核（含 Modality、Kinetics 類別數、輸入像素尺度、色彩順序、Mean/Std 正規化公式、Tensor 欄位佈局、開源授權與 Checksum）`UNRESOLVED-PROJECT-DECISION`。
5. **RGB-I3D 最小 Shape、VRAM 與 Throughput Feasibility**：實測最小配置下 RGB-I3D 在現有 GPU 上之 Peak VRAM 佔用與 FPS 吞吐量 `UNRESOLVED-PROJECT-DECISION`。
6. **Temporal Scope、Pretraining 與 Modality 初步研究決策**：在完成 Gate 1 稽核後，凍結 temporal scope（Full-Video vs Pre-Release 邊界）、預訓練路徑與是否將 Optical Flow 納入候選 `UNRESOLVED-PROJECT-DECISION`。

### Gate 2：Flow 仍是候選或選用時的特定稽核 (Optical Flow & Two-Stream Gate)
7. **TV-L1 Optical Flow 樣本 Benchmark 與 Storage 估算**：若 Optical Flow 仍為候選，於代表性樣本集實測 TV-L1 光流計算耗時，並估算 8,740 支 train/validation 影片之預計算光流硬碟容量需求 `UNRESOLVED-PROJECT-DECISION`。
8. **Flow-I3D / Two-Stream VRAM 與 Throughput 實測**：實測 Flow-I3D 與 Two-Stream 併行計算時之 Peak VRAM 與推論/訓練吞吐量 `UNRESOLVED-PROJECT-DECISION`。
9. **RGB 與 Flow 時間對齊與處理政策**：制定 RGB 影格與 Optical Flow 影格之時間步對齊、採樣與短影片 Looping/Padding 政策 `UNRESOLVED-PROJECT-DECISION`。

### Protocol 凍結與實作階段 (Post-Gate Execution)
10. **Protocol 制定與凍結**：通過 Gate 1（及 Gate 2，若適用）後，正式制定並凍結 Experiment D Protocol `UNRESOLVED-PROJECT-DECISION`。
11. **Implementation Prompt 產生**：Protocol 凍結後方可產生 Codex / Implementation Prompt `UNRESOLVED-PROJECT-DECISION`。
12. **實作、訓練與一次性 Formal Test**：最後方能進行程式碼實作、模型訓練與一次性 Formal Test 執行 `UNRESOLVED-PROJECT-DECISION`。

---

## 十、 綜合附錄與對照表

### A. C3D vs I3D vs Basketball-51 (A1/A2/B/C) 全方位對照表

| 對照維度 | C3D 原論文 (Tran et al.) `PAPER-SPECIFIED` | I3D 原論文 (Carreira & Zisserman) `PAPER-SPECIFIED` | Basketball-51 A1/A2 `COMPLETED-PROJECT-FACT` | Basketball-51 B `COMPLETED-PROJECT-FACT` | Basketball-51 C `COMPLETED-PROJECT-FACT` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Backbone 拓撲** | 8 Conv3D, 5 MaxPool | Inflated Inception-v1 (3D) | Modified C3D-style (8 Conv3D) | Reference C3D (8 Conv3D) | Modified C3D-style (8 Conv3D) |
| **參數量** | ~78 M | ~25 M (Two-Stream) | 6,951,874 | 78,012,164 (有效 61.2M) | 6,919,234 |
| **BatchNorm / GAP** | 無 BN / 無 GAP | 有 BN／固定 2×7×7 average pooling；其後取得時間軸 predictions 並進行 temporal averaging | 每 Conv 後加 BN / 有 GAP | 無 BN / 無 GAP | 每 Conv 後加 BN / 有 GAP |
| **訓練輸入 Shape** | `[3, 16, 112, 112]` | RGB-I3D：64 張連續 RGB frames，每張 224×224；實際 tensor layout 由候選 implementation／checkpoint 決定 | `[3, 16, 96, 96]` | `[3, 16, 112, 112]` | `[3, 16, 112, 112]` |
| **Temporal Sampling**| 連續 16-frame, Stride 8 (僅限描述子抽取) | 連續 64-frame, 25 FPS | 全片 16 Bins 稀疏抽樣 | 連續 16-frame, Stride 8 | 連續 16-frame, Stride 8 |
| **預訓練資料** | Sports-1M (1.1M videos) | Kinetics-400；論文比較 with／without ImageNet initialization | 無 (從零 / Warm-start) | 無 (從零初始化) | 無 (從零初始化) |
| **UCF-101 表現** | 82.3% (Single Net) `PAPER-RESULT`| **98.0%** (Two-Stream) `PAPER-RESULT`| N/A | N/A | N/A |
| **B51 Test Acc** | N/A | N/A | **77.39% (A2)** `PROJECT-RESULT`| 73.42% `PROJECT-RESULT` | 60.49% `PROJECT-RESULT` |

### B. 已知限制清單 (Known Limitations List)

1. **Full-Video 資訊洩漏**：現有 A1/A2/B/C 均使用完整影片，可能包含碰框與進球畫面，無法代表 Pre-Release 預測能力 `KNOWN-PROJECT-LIMITATION`。
2. **Release-Frame 標註缺失**：Basketball-51 目前無球離手影格標註 `KNOWN-PROJECT-LIMITATION`。
3. **Game-Level Split 未核實**：來源代碼隔離尚未經官方比賽清單核實 `KNOWN-PROJECT-LIMITATION`。
4. **Near Duplicates 未排除**：非 Exact Duplicates 之相似影片尚未過濾 `KNOWN-PROJECT-LIMITATION`。
5. **混雜因素無法宣稱單變因因果**：A2/B/C 差異包含解析度、Head、BN 與採樣，不能宣稱稀疏採樣或 Modified 骨幹本質上較優 `KNOWN-PROJECT-LIMITATION`。
6. **A1 Test Identity 稽核強度差異**：A1 舊版結果缺乏 test-ID hash 與 split-summary hash，無法以同等強度重新驗證其 test identity `KNOWN-PROJECT-LIMITATION`。

### C. Research Baseline 正式版修正紀錄 (Revision Log)

1. **更正 A1 凍結狀態與稽核限制標記**：將 A1 標記拆分為 `已凍結 COMPLETED-PROJECT-FACT；缺少 test-ID 及 split-summary hash KNOWN-PROJECT-LIMITATION`，明確分離客觀狀態與證據強度。
2. **更正 I3D Pooling 操作標記**：在附錄 A 將 I3D 標註更正為 `有 BN／固定 2×7×7 average pooling；其後取得時間軸 predictions 並進行 temporal averaging`，不與 GAP 混淆。
3. **新增 Repository 入庫標頭元資料**：標明 `Research Baseline v1.3`、`Status: Documentation Baseline Frozen`、`Source state: 2026-10-01 project results`、`Reviewed: 2026-10-07`。
4. **更正附錄 I3D 預訓練資料**：更正為 `Kinetics-400；論文比較 with／without ImageNet initialization`，排除單一路徑寫法。
5. **更正 I3D 輸入 Shape 表示**：更正為 `RGB-I3D：64 張連續 RGB frames，每張 224×224；實際 tensor layout 由候選 implementation／checkpoint 決定`，避免宣稱特定框架之 tensor layout。
6. **規範修辭與客觀用語**：移除未經統計檢定之「顯著」或普遍性用語，將 C3D `3×3×3` 改為「在論文所進行的 architecture search 範圍內取得最佳實驗表現與合理效率」，將 3D-ConvNet 描述改為「報告數值高於 Fixed（79.9% vs 76.0%）」。
7. **分層優化研究清單 (Gate 1 & Gate 2)**：將第 9 章優化為 Gate 1（通用時間/Provenance/RGB可行性稽核）與 Gate 2（Flow候選時之Benchmark與儲存估算），提升研究資源效益。

### D. 來源仍未指定或無法解決的問題清單 (Unspecified / Unresolved Items List)

1. **A1 Test Loss 缺失**：A1 舊版 Metric 未儲存 Test Loss，標示為「來源未指定」`PROJECT-RESULT`。
2. **C3D 官方權重色彩與 Mean 處理**：原始 C3D Checkpoint 所需之精確 Mean File 與色彩順序在專案檔案中「來源未指定」`KNOWN-PROJECT-LIMITATION`。
3. **I3D Figure 3 Padding 細節**：I3D 論文 Figure 3 未逐層指定 Padding，標示為「來源未指定」`KNOWN-PROJECT-LIMITATION`。
4. **I3D Prediction Averaging 計算式**：I3D 論文未明言 Averaging 係指 Pre-Softmax 或 Post-Softmax，標示為「來源未指定」`KNOWN-PROJECT-LIMITATION`。
5. **TOPIC 1.pdf 數據與架構佔位符**：`TOPIC 1.pdf` 中之 `XXXXNet` 與 `XX.X%` 為佔位符號，具體數據「來源未指定」`PROFESSOR-DIRECTION`。

### E. 必須由實際 Workspace 資料分析完成之項目清單 (Workspace Analysis Required List)

以下項目**嚴禁由 NotebookLM 推測或虛構**，必須於後續階段透過實際執行指令與 Python 腳本對 Workspace 內之 8,740 支 train/validation 影片檔與系統環境進行物理稽核：

1. **8,740 支 train/validation 影片之精確 FPS、Frame Count 與 Duration 統計分佈**（需執行 OpenCV / ffprobe 腳本解碼統計）。
2. **8,740 支 train/validation 影片在 raw frame count 與 25 FPS 重採樣下短於 64 Frames 之精確數量與比例**（需逐支統計）。
3. **短影片採 Looping 策略時重複影格之精確曝光比例**（需依據真實短影片長度分佈計算）。
4. **候選 I3D Checkpoint 之 SHA-256 Checksum、Tensor Layout、像素尺度與 Mean/Std**（需載入 Checkpoint 檢驗）。
5. **64-frame 3D 卷積在當前 GPU 上的實際 Peak VRAM 佔用與 FPS 吞吐量**（需執行 CUDA benchmark 腳本）。
6. **TV-L1 Optical Flow 預計算所需的實際硬碟容量與計算耗時**（需在真實影片樣本上測試計算時間並推算全集容量）。
