# Basketball-51 投籃進球二元分類：目前進度與研究路線總覽

> 最後更新：2026-11-26（對稱 `4×2` 主線＋3 組 1-seed 探索實驗：`AdamW(wd=1e-2)`、`224×224`、`Gated Temporal Attention`；全部七組皆通過 Codex 獨立唯讀複查並修訂表述）
> 對應活躍分支：`research/video-action-baseline`（舊 `origin/codex/*` 實驗分支已全數打上 `archive/codex/*` tag 封存，GitHub 僅維持 `main` 與單一活躍研究分支）

## 1. 任務定義與評估原則

- **任務**：Basketball-51 單片段（50 幀、`120×160` RGB）直接二元分類（`Miss` vs `Make`）。
- **評估與選模**：於 8 場未見比賽組成的驗證集（1,547 clips；`Miss=740`、`Make=807`）進行評估，統一以**最低 validation Cross-Entropy (CE)** 作為選模準則與 early stopping 指標。
- **評估揭露與限制**：
  - 下表皆為**反覆用於選模與研究決策之有限 validation 證據**；近期所有實驗皆未接觸 `test` split（`test_run: false`）。
  - 本專案早期已有歷史 `test` 接觸紀錄，因此既有 `test` 不宜宣稱為從未使用的獨立最終測試。
  - **兩組 random seed 共享同一份資料而非兩份獨立資料**；其平均僅為描述性摘要，不代表統計顯著，亦尚未宣稱達成目標泛化或 90% 準確率。
  - 表中 ECE 均為以正類 Make 機率 $P(\text{Make})$ 等寬切分 10 區間的校準誤差（Make-probability 10-bin ECE），非最高類別信心 ECE。

---

## 2. 近期正式驗證結果（Validation, N = 1,547）

| 模型／實驗組 | Seed | Best Epoch / Total | Val CE ↓ | Accuracy | Macro-F1 | Miss Recall | Make Recall | 備註／結論 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `MixNet-I3D` (50-frame binary, 49 BN3d frozen) | `20261024` | 3 / 8 | 0.440891 | 80.16% | 0.8003 | 75.54% | 84.39% | 較早過擬合、FP32 `microbatch=1` 峰值顯存 9.58 GB reserved，**已關閉此線** |
| `ACA Temporal50`（時間增強基準） | `20261024` | 7 / 12 | 0.417718 | 81.64% | 0.8159 | 79.86% | 83.27% | ROC-AUC `0.8992`、Brier `0.1322`、ECE `0.0549` |
| `ACA Spatial + Temporal50`（加空間增強） | `20261024` | 10 / 15 | 0.409121 | 80.87% | 0.8050 | 70.14% | 90.71% | CE 改善、ROC-AUC `0.9074`，但 Miss recall 明顯下降、Brier/ECE 變差、8 場 4 升 4 降 |
| `ACA Temporal50`（時間增強基準） | `20261025` | 5 / 10 | 0.385870 | 82.48% | 0.8246 | 82.84% | 82.16% | ROC-AUC `0.9116`、Brier `0.1237`、ECE `0.0508` |
| `ACA Spatial + Temporal50`（加空間增強） | `20261025` | 7 / 12 | 0.374929 | 82.93% | 0.8293 | 84.59% | 81.41% | CE、Accuracy、Macro-F1 同步改善且雙類 recall 平衡 |
| `Swin3D-T Temporal50`（修正版＋K400，無空間增強） | `20261024` | 5 / 10 | 0.411293 | 84.62% | 0.8455 | 81.89% | 87.11% | ROC-AUC `0.9181`（+0.0189）、Brier `0.1189`、ECE `0.0732`；6/8 場 AUC 改善 |
| `Swin3D-T Temporal50`（修正版＋K400，無空間增強） | `20261025` | 3 / 8 | 0.397231 | 84.29% | 0.8416 | 78.51% | 89.59% | ROC-AUC `0.9231`（+0.0115）、Brier `0.1204`、ECE `0.0717`；6/8 場 AUC 改善 |
| **`Swin3D-T Spatial + Temporal50`（單因子＋空間增強）** | `20261024` | 6 / 11 | 0.355902 | 85.07% | 0.8506 | **85.95%** | 84.26% | 整體 CE `-0.0554`、ROC-AUC **`0.9389`**、Brier `0.1066`、ECE `0.0722`；7/8 場 CE/AUC 改善，但 Make recall `-2.85 pp`、8 場 Acc 3 升 1 平 4 降 |
| **`Swin3D-T Spatial + Temporal50`（單因子＋空間增強）** | `20261025` | 3 / 8 | **0.342713** | 85.84% | 0.8576 | 81.76% | 89.59% | **主線最佳**：CE `-0.0545`、ROC-AUC `0.9308`、Brier **`0.1054`**、ECE **`0.0392`**；8/8 場 CE 改善、7/8 場 Acc 改善 |
| *探索性：`Swin3D-T Spatial + AdamW (wd=1e-2)`* | `20261024` | 6 / 11 | 0.385291 | 86.36% | 0.8623 | 80.27% | **91.95%** | Acc/F1 提升（`+20` 支）、ECE `0.0635`，但 CE `+0.0294`、AUC `0.9319`（`-0.0070`）、Miss recall 退步、8 場 4 升 4 降；**尚未取代主線基準** |
| *探索性：`Swin3D-T Spatial + 224×224 (INTER_AREA)`* | `20261024` | 5 / 10 | 0.358751 | **86.94%** | **0.8686** | 82.43% | 91.08% | Acc/F1 創單輪新高、Brier(`0.1041`) 偏低（淨 `+29` 支 = `ft1 +59`、`ft0 -24`、運動戰 `-6`；8 場 CE 3 降改善 5 升惡化）、ECE `0.0627`，但 CE `+0.0028`、AUC `0.9385` 未優於原規格，`ft0` 降至 `33.33%`；**保留為探索候選、未取代主線** |
| *探索性：`Swin3D-T Spatial + Gated Temporal Attention`* | `20261024` | 6 / 11 | 0.351489 | 85.91% | 0.8578 | 79.86% | 91.45% | 整體指標皆小幅優於 `s24` 基底（CE `-0.0044`、ROC-AUC **`0.9394`**／AP **`0.9430`** 新高、Brier **`0.1033`** 新低、ECE `0.0571` 仍高於 `s25` 之 `0.0392`）；惟淨 `+13` 支為 `ft1 +47` / `ft0 -21` / 運動戰 `-13`、MissR 與 `ft0`（`36.67%`）退步，群體平均權重偏重末段 token `[20..24]` 且分布仍分散（有效位置 `21.35/25`）；**保留平均池化主線，列為可做 `s25` 確認之有限候選** |
| *描述性兩 seed 平均（ACA Temporal50）* | *2 seeds* | — | *0.401794* | *82.06%* | *0.8203* | *81.35%* | *82.71%* | *Mean AUC `0.9054` / Brier `0.1279` / ECE `0.0528`* |
| *描述性兩 seed 平均（ACA Spatial + Temporal50）* | *2 seeds* | — | *0.392025* | *81.90%* | *0.8171* | *77.36%* | *86.06%* | *CE 跨 seed 方向一致（`-0.0086` / `-0.0109`）；Acc / F1 方向分歧* |
| *描述性兩 seed 平均（Swin3D-T Temporal50）* | *2 seeds* | — | *0.404262* | *84.45%* | *0.8436* | *80.20%* | *88.35%* | *Mean AUC `0.9206` / Brier `0.1196` / ECE `0.0725`* |
| ***描述性兩 seed 平均（Swin3D-T Spatial + Temporal50）*** | ***2 seeds*** | — | ***0.349307*** | ***85.46%*** | ***0.8541*** | ***83.85%*** | ***86.93%*** | ***Mean AUC `0.9349` / Brier `0.1060` / ECE `0.0557`（兩 seed 整體彙總指標皆改善；類別與場次仍有取捨，平均 ECE 仍略高於 ACA）*** |

> 空間增強設定（`clip_flip_color_light_v2`）：整段 clip 一致性隨機水平翻轉（`p=0.5`）與亮度／對比／飽和度微調（`[0.95, 1.05]`）。

---

## 3. Stage B 成果：含動態位置索引修正的 Swin3D-T（已完成對稱 4×2 雙種子驗證）

為進行影片預訓練骨幹的**系統比較**（同時改變架構、Kinetics-400 預訓練來源、LayerNorm／StochasticDepth，非純骨幹消融），已完成全部工程關卡與七組正式訓練（對稱 4×2 主線四組＋3 組 1-seed 探索實驗：`AdamW(wd=1e-2)`、`224×224` 整幀 `INTER_AREA`、`Gated Temporal Attention Pooling`，全數保存逐片 logits 並以 ties-aware 公式重算 ROC-AUC／AP／Brier／Make-prob ECE）：

1. **動態位置索引修正（`development/swin3d_t_binary50_v1/`）**：保留原 `relative_position_bias_table`、`relative_position_index` 與全部鍵名，於空間視窗縮小時依實際 `(t, h, w)` 座標選取對應行列（修正 `(8, 4, 5)` 下原版截取 `24,980 / 25,600` 錯位；完整 `(8, 7, 7)` 視窗輸出與 173 組梯度與原版逐值一致），`120×160` 單模型 FP32 峰值顯存為 **`3.39 GiB` allocated / `4.36 GiB` reserved**。
2. **雙 seed 系統比較（`Swin3D-T Temporal50` vs `ACA Temporal50`）**：在關閉空間增強、固定 `Adam(lr=1e-4, wd=0)` 的 1:1 條件下，`Swin3D-T` 兩組 seed 皆優於 ACA（`84.62%` / `84.29%` vs `81.64%` / `82.48%`，平均 **`+2.39 pp`**；兩 seed 皆在 6/8 場提升 ROC-AUC），但第 3–5 輪後出現明顯後期過擬合（Train CE 降、Val CE 升），平均 Make-prob ECE 為 `0.0725`。
3. **雙 seed 單因子空間增強消融（`Swin3D-T + clip_flip_color_light_v2`, `seed20261024` & `seed20261025`）**：僅加入已驗證之空間增強，**兩個 seed 的整體彙總指標（CE、Accuracy、Macro-F1、ROC-AUC、Brier、Make-prob ECE）皆改善**——描述性平均 **Val CE `0.349307`（`-0.0550` vs Swin Temp50 / `-0.0525` vs ACA）、Accuracy `85.46%`（`+3.39 pp` vs ACA）、Macro-F1 `0.8541`、ROC-AUC `0.9349`、Brier `0.1060`、Make-prob ECE `0.0557`**；惟類別與場次仍有取捨（`s24` Make recall `-2.85 pp`、8 場 Acc 3 升 1 平 4 降），且平均 Make-prob ECE 仍略高於 `ACA Temporal50`（`0.0528`）、跨種子校準波動明顯（`0.0722` vs `0.0392`）。

---

## 4. 目前決策與下一步計畫

1. **優先 1**：`Gated Temporal Attention` 為三組探索中唯一小幅改善主指標 Val CE（`0.3515`）之有限候選（含類別取捨），可補跑 `seed20261025` 確認跨種子穩定性；或在平均池化主線（`120×160 Spatial+Temp50 + Adam wd=0`）進行純單因子 `CosineAnnealingLR` 對照（鎖定 `T_max`、`eta_min`、步進單位與 scheduler 續訓驗收）。
2. **優先 2（與教授討論後評估針對 `ft`／`ft0` 之待驗證假說）**：三組探索實驗（`AdamW`、`224×224`、`TAttn`）均未改善 `ft0`（`33–39%`），目前證據不支持直接採用 `TAttn` 或據此跳至 RL 選幀／`SpinalNet`；建議會中討論 8 類／家族內均衡監督、籃框 ROI 聚焦與整體後續優先序。
3. **優先 3**：若會後決議推進高解析度路線，再補跑 `224×224 seed20261025` 檢驗跨種子穩定性。
4. **已關閉路線（不再投入）**：純 `MixNet-I3D` 50 幀二元分類；固定降 LR `5e-5`；重現論文未公開 8 類高分設定。

---

## 5. 主要目錄導覽

- `development/adjudicated_input_v1/`、`development/aca_eight_class_data_v1/`：資料裁決清單、50 幀載入與基礎檢查。
- `development/aca_binary_v1/`、`development/aca_temporal_v1/`、`development/aca_temporal_pair_v1/`：ACA 二元基準與時間增強（`seed20261024` / `seed20261025`）。
- `development/mixnet_binary50_v1/`、`development/mixnet_binary50_formal_v1/`：MixNet-I3D 50 幀二元封裝與正式訓練腳本（已結案）。
- `development/aca_spatial_v1/`、`development/aca_spatial_pair_v1/`：ACA 空間增強（`clip_flip_color_light_v2`）與雙 seed 正式訓練／readiness 腳本。
- `development/swin3d_t_binary50_v1/`、`development/swin3d_t_binary50_formal_v1/`、`development/swin3d_t_binary50_pair_v1/`、`development/swin3d_t_binary50_spatial_v1/`、`development/swin3d_t_binary50_spatial_pair_v1/`、`development/swin3d_t_binary50_spatial_adamw_v1/`、`development/swin3d_t_binary50_spatial_224_v1/`、`development/swin3d_t_binary50_spatial_tattn_v1/`：含動態位置索引修正的 Swin3D-T 隔離套件、雙 seed 主線、雙 seed 空間增強與三組探索性單因子入口（`AdamW`、`224×224`、`Gated Temporal Attention`）。
- `docs/research/`：各階段工程規格、研究筆記與歷史審計紀錄。
- `outputs/` 與 `development/*/artifacts/`：訓練產物、checkpoint 與預訓練權重（皆由 `.gitignore` 排除，不上傳 GitHub）。
