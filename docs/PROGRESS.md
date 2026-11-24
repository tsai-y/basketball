# Basketball-51 投籃進球二元分類：目前進度與研究路線總覽

> 最後更新：2026-11-25
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

## 3. Stage B 進展：含動態位置索引修正的 Swin3D-T（已完成全關工程驗收，準備啟動正式訓練）

為進行影片預訓練骨幹的**系統比較**（同時改變架構、Kinetics-400 預訓練來源、LayerNorm／StochasticDepth，非純骨幹消融），已完成全部四道隔離工程驗收（未修改已安裝 `torchvision`、未動用受保護基準檔）：

1. **隔離套件與 `weights=None` CPU 驗收（`PASS_CPU_ONLY`）**：
   - 於 `development/swin3d_t_binary50_v1/` 實作 `DynamicPosShiftedWindowAttention3d`，保留原 `relative_position_bias_table`（`[2535, num_heads]`）、原 `relative_position_index` buffer（`[392, 392]`）與所有參數／buffer 名稱，於空間視窗縮小時以實際 `(t, h, w)` 座標選取對應行列（修正 `(8, 4, 5)` 下原版截取 `24,980 / 25,600` 錯位；完整 `(8, 7, 7)` 視窗輸出與 173 組梯度與原版逐值一致）。
2. **官方預訓練權重下載與 CPU 載入驗收（`PASS_PRETRAINED_LOAD_CPU_ONLY`）**：
   - 單次下載官方 `Swin3D_T_Weights.KINETICS400_V1` 至本地 `.gitignore` 路徑，鎖定完整 SHA-256 `7615ae035996b65eb38dad437ae533d2dfcd36f9f89d28c0f0fa7bfb8e6b3130`；驗證 400 類 `strict=True` 載入、以獨立 `head_seed` 替換 `Linear(768, 2)` 並確認全部 183 個非分類頭張量與下載檔逐值一致。
3. **正式入口整合與 CPU／GPU 接線驗收（`PREPARED_NOT_STARTED` → 核准啟動）**：
   - 於 `development/swin3d_t_binary50_formal_v1/` 整合與 `ACA temporal50`（`seed=20261024`）1:1 對齊之正式訓練契約（`plan_sha256: 6bd908ced780140ebf88fae132a566f338eb2b04e6ce76b5a9c44d5d6ff2787c`）：train seed `20261024`、head seed `20264024`、endpoint-stratified 50 幀時間採樣、關閉空間／色彩增強、固定均勻 50 幀 validation（1,547 支）、Adam 固定 LR `1e-4`、`weight_decay=0`、梯度裁切 `1.0`、嚴格 FP32、`microbatch=1` / `effective=16` / 尾批 `8`、最多 30 輪、`patience=5`、最低完整二類 validation CE（同分保留較早輪）。
   - 正式迴圈實測單模型峰值顯存 **`3.3891 GiB` allocated / `4.3555 GiB` reserved**，並驗證輪中 `cursor=16`（`best_model` 非空）恢復與未改善時保留先前 `best_model` 分支皆 6 / 6 逐值一致。

---

## 4. 目前決策與下一步計畫

1. **進行中（Stage B — `swin3d_t_binary50_seed20261024_labelrev1_v1` 正式單 seed 訓練）**：
   - 執行唯一一組 `seed20261024` 正式比較（對照 `ACA temporal50`），依 1,547 支 validation 結果決定後續方向；不並行追加其他模型或 LR 搜尋分支。
2. **備案（Stage A2 — ACA `CosineAnnealingLR`）**：
   - 僅在 Stage B 結果不理想時啟動：固定其餘超參數與資料設定，只更動 LR schedule，並補齊 scheduler checkpoint 與續訓一致性驗證。
3. **已關閉路線（不再投入）**：
   - 純 `MixNet-I3D` 50 幀二元分類；固定降 LR `5e-5`；重現論文未公開 8 類高分設定。

---

## 5. 主要目錄導覽

- `development/adjudicated_input_v1/`、`development/aca_eight_class_data_v1/`：資料裁決清單、50 幀載入與基礎檢查。
- `development/aca_binary_v1/`、`development/aca_temporal_v1/`、`development/aca_temporal_pair_v1/`：ACA 二元基準與時間增強（`seed20261024` / `seed20261025`）。
- `development/mixnet_binary50_v1/`、`development/mixnet_binary50_formal_v1/`：MixNet-I3D 50 幀二元封裝與正式訓練腳本（已結案）。
- `development/aca_spatial_v1/`、`development/aca_spatial_pair_v1/`：ACA 空間增強（`clip_flip_color_light_v2`）與雙 seed 正式訓練／readiness 腳本。
- `development/swin3d_t_binary50_v1/`、`development/swin3d_t_binary50_formal_v1/`：含動態位置索引修正的 Swin3D-T 隔離套件、預訓練載入驗收與 `seed20261024` 正式訓練入口。
- `docs/research/`：各階段工程規格、研究筆記與歷史審計紀錄。
- `outputs/` 與 `development/*/artifacts/`：訓練產物、checkpoint 與預訓練權重（皆由 `.gitignore` 排除，不上傳 GitHub）。
