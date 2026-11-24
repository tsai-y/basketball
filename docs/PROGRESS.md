# Basketball-51 投籃進球二元分類：目前進度與研究路線總覽

> 最後更新：2026-11-24
> 對應活躍分支：`research/video-action-baseline`（舊 `origin/codex/*` 實驗分支已全數打上 `archive/codex/*` tag 封存，GitHub 僅維持 `main` 與單一活躍研究分支）

## 1. 任務定義與評估原則

- **任務**：Basketball-51 單片段（50 幀、`120×160` RGB）直接二元分類（`Miss` vs `Make`）。
- **評估與選模**：於 8 場未見比賽組成的驗證集（1,547 clips；`Miss=740`、`Make=807`）進行評估，統一以**最低 validation Cross-Entropy (CE)** 作為選模準則與 early stopping 指標。
- **評估揭露與限制**：
  - 下表皆為**有限 validation 證據**；近期所有實驗皆未接觸 `test` split（`test_run: false`）。
  - 本專案早期已有歷史 `test` 接觸紀錄，因此既有 `test` 不宜宣稱為從未使用的獨立最終測試。
  - 兩組 random seed 的平均僅為描述性摘要，不代表統計顯著，亦尚未宣稱達成目標泛化或 90% 準確率。

---

## 2. 近期正式驗證結果（Validation, N = 1,547）

| 模型／實驗組 | Seed | Best Epoch / Total | Val CE ↓ | Accuracy | Macro-F1 | Miss Recall | Make Recall | 備註／結論 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `MixNet-I3D` (50-frame binary, 49 BN3d frozen) | `20261024` | 3 / 8 | 0.440891 | 80.16% | 0.8003 | 75.54% | 84.39% | 較早過擬合、FP32 `microbatch=1` 峰值顯存 9.58 GB reserved，**已關閉此線** |
| `ACA Temporal50`（時間增強基準） | `20261024` | 7 / 12 | 0.417718 | 81.64% | 0.8159 | 79.86% | 83.27% | 單 seed 基準 |
| `ACA Spatial + Temporal50`（加空間增強） | `20261024` | 10 / 15 | **0.409121** | 80.87% | 0.8050 | 70.14% | 90.71% | CE 改善、ROC-AUC `0.8992 → 0.9074`，但 Miss recall 明顯下降、Brier/ECE 變差、8 場比賽 4 升 4 降 |
| `ACA Temporal50`（時間增強基準） | `20261025` | 5 / 10 | 0.385870 | 82.48% | 0.8246 | 82.84% | 82.16% | 跨 seed 對照組 |
| `ACA Spatial + Temporal50`（加空間增強） | `20261025` | 7 / 12 | **0.374929** | **82.93%** | **0.8293** | 84.59% | 81.41% | CE、Accuracy、Macro-F1 同步改善且雙類 recall 平衡 |
| *描述性兩 seed 平均（ACA Temporal50）* | *2 seeds* | — | *0.401794* | *82.06%* | *0.8203* | *81.35%* | *82.71%* | — |
| *描述性兩 seed 平均（ACA Spatial + Temporal50）* | *2 seeds* | — | ***0.392025*** | *81.90%* | *0.8171* | *77.36%* | *86.06%* | CE 跨 seed 方向一致（`-0.0086` / `-0.0109`）；Acc / F1 方向分歧，保留為候選設定 |

> 空間增強設定（`clip_flip_color_light_v2`）：整段 clip 一致性隨機水平翻轉（`p=0.5`）與亮度／對比／飽和度微調（`[0.95, 1.05]`）。

---

## 3. 已完成盤點：Stage B（`swin3d_t` 影片預訓練骨幹可行性）

為確認換用影片預訓練骨幹（新系統比較，而非 ACA 模組消融）是否可行，已完成有上限的唯讀盤點（未下載權重、未跑 GPU、未修改程式碼）：

- **本機相容性（`torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`）**：
  - `swin3d_t(weights=None)` 改為 2 類分類頭（`27,852,008` 參數；29 層 `LayerNorm`、0 層 `BatchNorm`、12 層 `StochasticDepth(0 → 0.1)`），CPU 推論 `[1, 3, 50, 120, 160]` 與訓練反向 `[2, 3, 8, 64, 64]` 皆可通過且梯度有限。
- **盤點發現的兩項關鍵缺口／風險**：
  1. **本機權重缺口**：官方 `Swin3D_T_Weights.KINETICS400_V1`（`swin3d_t-7615ae03.pth`，約 121.5 MB）目前不在本機快取，尚未下載。
  2. **輸入規格與 Stage 3 位置偏置差異**：官方權重於 `32 幀、224×224` 評估；若直接輸入現有 `50 幀、120×160`，Stage 3 特徵圖縮至 `[1, 25, 4, 5, 768]`（`H=4, W=5 < window 7`），上游 `torchvision` 實作會直接對 `(8, 7, 7)` 展平的 `relative_position_index[:160, :160]` 切片，造成最後 2 個 block 的相對位置索引步長不對齊。
- **備註**：`LayerNorm` 雖消除了 BN 的 batch 統計相依，但不保證 `microbatch=1 + grad_accum=16` 與實體 `batch=16` 逐值等價（仍受 loss 縮放、梯度裁剪時點、浮點加總順序與 `StochasticDepth` 隨機性影響）。

---

## 4. 目前決策與下一步計畫

1. **優先（Stage B — `swin3d_t` 新系統比較）**：
   - 依唯讀盤點結果拍板唯一輸入／正規化／分類頭契約（決定 `120×160` 直接輸入或補齊/縮放至 `224×224`，並核准下載本機權重）。
   - 完成有上限的 CPU 驗收與顯存／續訓 GPU readiness 後，再決定是否啟動正式訓練；暫不納入 `LSTA/TSCI` 或並行展開 `mvit_v2_s`。
2. **備案（Stage A2 — ACA `CosineAnnealingLR`）**：
   - 僅在 Stage B 不可行時啟動：固定其餘超參數與資料設定，只更動 LR schedule，並補齊 scheduler checkpoint 與續訓一致性驗證。
3. **已關閉路線（不再投入）**：
   - 純 `MixNet-I3D` 50 幀二元分類；固定降 LR `5e-5`；重現論文未公開 8 類高分設定。

---

## 5. 主要目錄導覽

- `development/adjudicated_input_v1/`、`development/aca_eight_class_data_v1/`：資料裁決清單、50 幀載入與基礎檢查。
- `development/aca_binary_v1/`、`development/aca_temporal_v1/`、`development/aca_temporal_pair_v1/`：ACA 二元基準與時間增強（`seed20261024` / `seed20261025`）。
- `development/mixnet_binary50_v1/`、`development/mixnet_binary50_formal_v1/`：MixNet-I3D 50 幀二元封裝與正式訓練腳本（已結案）。
- `development/aca_spatial_v1/`、`development/aca_spatial_pair_v1/`：ACA 空間增強（`clip_flip_color_light_v2`）與雙 seed 正式訓練／readiness 腳本。
- `docs/research/`：各階段工程規格、研究筆記與歷史審計紀錄。
- `outputs/` 與 `development/*/artifacts/`：訓練產物、checkpoint 與預訓練權重（皆由 `.gitignore` 排除，不上傳 GitHub）。
