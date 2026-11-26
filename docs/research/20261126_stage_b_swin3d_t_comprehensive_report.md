# Basketball-51 投籃進球二元分類：Stage B（含動態位置索引修正的 Swin3D-T）完整研究與實驗總結報告

- **報告日期**：2026-11-26（對稱 `4×2` 主線＋3 組 1-seed 探索實驗：`AdamW(wd=1e-2)`、`224×224` 整幀 `INTER_AREA`、`Gated Temporal Attention Pooling`；**全部七組皆已通過 Codex 獨立唯讀複查並依意見修訂表述**）
- **活躍研究分支**：`research/video-action-baseline`（GitHub 僅維持 `main` 與此單一研究分支；舊 `origin/codex/*` 分支全數封存於 `archive/codex/*` tag）
- **驗證與資料邊界**：
  - 所有量化結果皆來自 **8 場未見比賽組成的驗證集（Validation, $N = 1{,}547$；Miss = 740, Make = 807）**，統一以**最低 validation Cross-Entropy (CE)** 選模（`direct_two_logit_argmax`，零門檻調參）；**本驗證集反覆用於選模與研究決策，兩組 random seed 共享同一份資料而非兩份獨立資料**。
  - 本系列全部實驗 `test_run: false`、**未接觸 `test` split**；同時維持揭露「專案早期已有歷史 `test` 接觸紀錄」，故既有 `test` 不宜宣稱為從未接觸的最終獨立測試集。
  - 保護檔案完整性：[verify_handoff.py](file:///home/tsai/projects/basketball/docs/handoffs/20261123_aca_to_mixnet_i3d/verify_handoff.py) 維持 **256 / 256 PASS**，已安裝 `torchvision`（`0.26.0+cu128`）未修改。

---

## 1. 核心摘要（Executive Summary）

Stage B 將骨幹與預訓練來源從 ImageNet-1k ResNet50+ACA 切換為 **Kinetics-400 預訓練之 3D Video Swin Transformer（`swin3d_t`）**，並在隔離套件內修正了上游 `torchvision` 於小空間特徵圖下的相對位置索引錯位。經四道嚴格工程關卡（`weights=None` CPU 驗收 → 官方權重下載與 400 類載入驗收 → 獨立 GPU 工程驗收 → 正式迴圈 CPU/GPU 接線驗收）後，已完成完整的**對稱 `4 設定 × 2 Seeds`（`20261024`、`20261025`）正式驗證矩陣**、3 組 1-seed 探索實驗（`AdamW(wd=1e-2)`、`224×224` 整幀 `INTER_AREA`、`Gated Temporal Attention Pooling`）與對應七組唯讀逐片診斷（全數保存逐片 logits 並以 ties-aware 公式獨立重算 ROC-AUC／AP／Brier／Make 機率 10-bin ECE）：

1. **系統比較跨兩組 seed 一致領先 `ACA Temporal50`**（在完全關閉空間增強、固定 `Adam(lr=1e-4, wd=0)`、嚴格 FP32 的 1:1 條件下）：
   - `Swin3D-T Temporal50` 兩組 seed 描述性平均：Accuracy 由 `82.06%` 升至 **`84.45%`（`+2.39 pp`）**、Macro-F1 由 `0.8203` 升至 **`0.8436`（`+0.0233`）**、ROC-AUC 由 `0.9054` 升至 **`0.9206`（`+0.0152`）**、Brier 由 `0.1279` 降至 **`0.1196`（`-0.0083`）**。
   - 但未加空間增強時於第 3–5 輪後即呈現後期過擬合現象（Train Acc 升至 95.6–96.1%、Val CE 反彈至 0.80–0.90），整體 Val CE（平均 `0.4043`）與 Make-prob ECE（平均 `0.0725`）未同步壓低。
2. **單因子加上空間增強（`clip_flip_color_light_v2`）在兩組 seed 的整體彙總指標全面改善（類別與場次仍有取捨）**：
   - `seed20261024`（Best Ep 6/11）：整體指標均改善——Val CE **`0.355902`（`-0.0554`）**、Accuracy **`85.07%`（`+0.45 pp`）**、Macro-F1 **`0.8506`**、ROC-AUC **`0.9389`（`+0.0208`）**、Brier **`0.1066`**、Make-prob ECE `0.0722`（`-0.0010`）；8 場未見比賽中 **7/8 場 CE 與 ROC-AUC 改善**，但類別與場次準確率存在取捨（Make Recall 由 `87.11%` 降至 `84.26%`［`-2.85 pp`］；8 場 Accuracy 為 **3 升、1 平、4 降**）。
   - `seed20261025`（Best Ep 3/8）：Val CE **`0.342713`（`-0.0545`，專案全線最低）**、Accuracy **`85.84%`（`+1.55 pp`，Ep 4 達 `85.97%`）**、Macro-F1 **`0.8576`**、ROC-AUC **`0.9308`（`+0.0077`）**、Brier **`0.1054`**、Make-prob ECE 降至 **`0.0392`（`-0.0325`，低於同種子 ACA 的 `0.0508`）**；8 場未見比賽中 **8/8 場 CE 改善、7/8 場 Accuracy 改善**，單場 `v152` 達 `89.82%` / AUC `0.9606`。
   - **對稱兩組 seed 描述性平均總覽（`Swin3D-T Spatial+Temp50` vs `ACA Temporal50`）**：
     - **Val CE**：`0.401794 → 0.349307`（**`-0.0525`**）
     - **Accuracy**：`82.06% → 85.46%`（**`+3.39 pp`**，兩組 seed 分別淨多答對 **`+53`** 與 **`+52`** 支片段）
     - **Macro-F1**：`0.8203 → 0.8541`（**`+0.0338`**）
     - **ROC-AUC**：`0.9054 → 0.9349`（**`+0.0295`**）
     - **Brier Score**：`0.1279 → 0.1060`（**`-0.0219`**）
     - **Make-prob ECE (10-bin)**：`0.0528 → 0.0557`（較無空間增強組的 `0.0725` 明顯縮小，但平均仍略高於 ACA 的 `0.0528`，且 `s24` `0.0722` 與 `s25` `0.0392` 具明顯跨種子差異）

> [!NOTE]
> `Swin3D-T` 相比 `ACA Temporal50` 同時改變了骨幹架構、Kinetics-400 影片預訓練來源與正規化／正則化（LayerNorm + StochasticDepth），因此定位為**系統比較**；而 `Swin3D-T Spatial+Temporal50` 相對同種子 `Swin3D-T Temporal50`（同初始權重、同優化器、同時間採樣）則為**嚴格雙種子單因子空間增強消融**。

---

## 2. 工程方法與四道關卡驗收摘要

### 2.1 動態位置索引修正（`DynamicPosShiftedWindowAttention3d`）
- **上游問題定位**：`torchvision 0.26.0` `swin3d_t` 預設視窗為 `(8, 7, 7)`（392 tokens，row-major 步長 `(49, 7, 1)`）。當空間特徵圖縮小至 `ceil(H/32) < 7` 或 `ceil(W/32) < 7`（即 `H <= 192` 或 `W <= 192`；在 `50×120×160` 下 Stage 3 特徵為 `[B, 25, 4, 5, 768]`，其中 `features.6.0` `shift=(0,0,0)`、`features.6.1` `shift=(4,0,0)`）時，上游 `_get_relative_position_bias` 直接切片 `relative_position_index[:160, :160]`，經獨立座標公式驗證在 `(8, 4, 5)` 下有 **`24,980 / 25,600`** 個相對位置索引與實際幾何不符（註：若僅最外層時間維度縮小如 `(4, 7, 7)` 則不發生錯位）。
- **隔離修正策略**（[model.py](file:///home/tsai/.codex/worktrees/antigravity-mixnet-binary50/basketball/development/swin3d_t_binary50_v1/model.py)）：不修改已安裝 `torchvision`，完整保留 `relative_position_bias_table`（`[2535, num_heads]`）、`relative_position_index` buffer（`[392, 392]`）與全部 185 個 `state_dict` 鍵名／形狀（僅二類 `head` 不同），於動態視窗縮小時依實際 `(t, h, w)` 座標選取 `relative_position_index.index_select(0, coords).index_select(1, coords)`。

### 2.2 四道前置工程驗收紀錄

| 關卡 | 證據目錄 | 結果 | 關鍵量化確認 |
| :--- | :--- | :---: | :--- |
| ① 隔離套件＋`weights=None` CPU | [REPORT.md](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_cpu_acceptance_20261124_v1/REPORT.md) | **PASS** | 全部 5 種實際遇到視窗 `(4,2,2)`～`(8,7,7)` 100% 吻合獨立公式；完整 `(8,7,7)` 視窗下 eval 輸出與 173 組梯度與原版 `max_abs_diff = 0.0`；固定 `04`/`22` 單次正規化誤差 `0.0` |
| ② 單次官方權重下載＋CPU 載入 | [REPORT.md](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_pretrained_load_20261124_v1/REPORT.md) | **PASS** | 鎖定 `swin3d_t-7615ae03.pth` 完整 SHA-256 `7615ae03...b3130`；400 類 `strict=True` 載入與原版 `(8,7,7)` 輸出逐值一致；獨立 `head_seed` 替換二類頭後全部 183 個非分類頭張量與下載檔逐值一致 |
| ③ 獨立 GPU 工程驗收 | [REPORT.md](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_gpu_readiness_20261125_v1/REPORT.md) | **PASS** | 嚴格 FP32（`allow_tf32=False`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`）、`micro1/accum16/tail8` 全部有限；峰值 `3.2730 GiB` alloc / `4.3613 GiB` reserved（已補註跨重試累計 10 次更新之紀錄） |
| ④ 四條正式入口接線驗收（`s24`/`s25`/`sp24`/`sp25`） | [REPORT.md](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_formal_readiness_20261125_v1/REPORT.md) | **PASS** | 全部四條正式入口皆通過輪中 `cursor=16` 且 `best_model` 非空續訓（6/6 逐值一致）與第二輪未改善保留 Epoch 1 `best_model` 實測；正式迴圈單模型峰值 **`3.3891 GiB` alloc / `4.3555 GiB` reserved**；`s25`、`sp24`、`sp25` 全採每步原子更新帳本（各一次過關、累計 `5 / 8` 次更新） |

---

## 3. 對稱 4 × 2 全線正式驗證結果總表（Validation, N = 1,547）

### 3.1 單次實驗完整明細

| 模型／實驗組 | Seed | Best / Total Epoch | Val CE ↓ | Accuracy ↑ | Macro-F1 ↑ | Miss Recall | Make Recall | ROC-AUC ↑ | AP Make ↑ | Brier ↓ | ECE ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MixNet-I3D`（50 幀二元，已關閉） | `20261024` | 3 / 8 | 0.440891 | 80.16% | 0.8003 | 75.54% | 84.39% | — | — | — | — |
| `ACA Temporal50`（基準對照） | `20261024` | 7 / 12 | 0.417718 | 81.64% | 0.8159 | 79.86% | 83.27% | 0.8992 | 0.9010 | 0.1322 | 0.0549 |
| `ACA Spatial + Temporal50` | `20261024` | 10 / 15 | 0.409121 | 80.87% | 0.8050 | 70.14% | 90.71% | 0.9074 | 0.9130 | 0.1327 | 0.0718 |
| `ACA Temporal50`（基準對照） | `20261025` | 5 / 10 | 0.385870 | 82.48% | 0.8246 | 82.84% | 82.16% | 0.9116 | 0.9155 | 0.1237 | 0.0508 |
| `ACA Spatial + Temporal50` | `20261025` | 7 / 12 | 0.374929 | 82.93% | 0.8293 | 84.59% | 81.41% | — | — | — | — |
| `Swin3D-T Temporal50`（無空間增強） | `20261024` | 5 / 10 | 0.411293 | 84.62% | 0.8455 | 81.89% | 87.11% | 0.9181 | 0.9126 | 0.1189 | 0.0732 |
| `Swin3D-T Temporal50`（無空間增強） | `20261025` | 3 / 8 | 0.397231 | 84.29% | 0.8416 | 78.51% | 89.59% | 0.9231 | 0.9222 | 0.1204 | 0.0717 |
| **`Swin3D-T Spatial + Temporal50`** | `20261024` | 6 / 11 | 0.355902 | 85.07% | 0.8506 | **85.95%** | 84.26% | **0.9389** | **0.9420** | 0.1066 | 0.0722 |
| **`Swin3D-T Spatial + Temporal50`** | `20261025` | 3 / 8 | **0.342713** | 85.84% | 0.8576 | 81.76% | 89.59% | 0.9308 | 0.9315 | **0.1054** | **0.0392** |
| *探索性單因子：`Swin3D-T Spatial + AdamW (wd=1e-2)`* | `20261024` | 6 / 11 | 0.385291 | 86.36% | 0.8623 | 80.27% | **91.95%** | 0.9319 | 0.9340 | 0.1103 | 0.0635 |
| *探索性規格比較：`Swin3D-T Spatial + 224×224 (INTER_AREA)`* | `20261024` | 5 / 10 | 0.358751 | **86.94%** | **0.8686** | 82.43% | 91.08% | 0.9385 | 0.9413 | 0.1041 | 0.0627 |
| *探索性池化替換：`Swin3D-T Spatial + Gated Temporal Attention`* | `20261024` | 6 / 11 | 0.351489 | 85.91% | 0.8578 | 79.86% | 91.45% | **0.9394** | **0.9430** | **0.1033** | 0.0571 |

### 3.2 對稱兩組 Seed 描述性平均（`20261024` & `20261025`）

| 設定（2-seed 描述性平均） | Mean Val CE ↓ | Mean Accuracy ↑ | Mean Macro-F1 ↑ | Mean MissR / MakeR | Mean ROC-AUC ↑ | Mean AP Make ↑ | Mean Brier ↓ | Mean ECE ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ACA Temporal50` | 0.401794 | 82.06% | 0.8203 | 81.35% / 82.71% | 0.9054 | 0.9082 | 0.1279 | **0.0528** |
| `ACA Spatial + Temporal50` | 0.392025 | 81.90% | 0.8171 | 77.36% / 86.06% | — | — | — | — |
| `Swin3D-T Temporal50` | 0.404262 | 84.45% | 0.8436 | 80.20% / 88.35% | 0.9206 | 0.9174 | 0.1196 | 0.0725 |
| **`Swin3D-T Spatial + Temporal50`** | **`0.349307`** | **`85.46%`** | **`0.8541`** | **`83.85% / 86.93%`** | **`0.9349`** | **`0.9367`** | **`0.1060`** | `0.0557` |

> **註**：表中 `ACA Temporal50` 與全部七組 `Swin3D-T` 之 ROC-AUC／AP／Brier／ECE 均由逐片 logits 檔以 ties-aware 公式獨立重算，且與 `summary.json` 的 CE 誤差皆為 **`0.0`**、混淆矩陣完全吻合，重算數值皆在指定數值精度內吻合（註：沿用基底 AP Miss 與由保存機率重算存在約 `8.82×10⁻⁸` 之 float32／float64 補數同分捨入差，不影響結論；`ACA Spatial seed24` 的 ROC-AUC/AP 來自先前 Stage 0 診斷）。**ECE 定義說明**：本報告全部 ECE 數值皆為**以正類 Make 預測機率 $P(\text{Make})$ 等寬切分 10 個區間之校準誤差（Make-probability 10-bin ECE）**，而非以最高類別信心 $\max_c P(c)$ 分箱的 ECE，跨實驗皆採同一公式可完整重現。

---

## 4. 8 場未見比賽逐場細部表現對照（雙種子演進）

### 4.1 `seed20261024` 逐場演進（`ACA` → `Swin3D-T Temp50` → `Swin3D-T Spatial+Temp50`）

| 比賽 ID | 片段數 | `ACA Temporal50` (CE / Acc / AUC) | `Swin3D-T Temporal50` (CE / Acc / AUC) | `Swin3D-T Spatial+Temp50` (CE / Acc / AUC) | `Swin Spatial` vs `ACA` 淨變化 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `v113` | 211 | `0.4469` / `79.62%` / `0.8889` | `0.4512` / `83.89%` / `0.9014` | **`0.3983`** / `83.41%` / **`0.9286`** | CE `-0.0485`, Acc `+3.79 pp`, AUC `+0.0396` |
| `v123` | 221 | `0.4023` / `81.45%` / `0.9060` | **`0.3219`** / **`86.43%`** / **`0.9495`** | `0.3231` / **`86.43%`** / `0.9466` | CE `-0.0792`, Acc `+4.98 pp`, AUC `+0.0406` |
| `v149` | 31 | **`0.1667`** / **`96.77%`** / **`0.9952`** | `0.4064` / `93.55%` / `0.9619` | `0.4030` / `90.32%` / `0.9857` | 小樣本場次（31 支）唯一略退場次 |
| `v152` | 226 | `0.4337` / `79.20%` / `0.9143` | `0.3327` / **`86.73%`** / `0.9532` | **`0.3123`** / `84.51%` / **`0.9563`** | CE `-0.1214`, Acc `+5.31 pp`, AUC `+0.0420` |
| `v153` | 215 | `0.4118` / `84.19%` / `0.9086` | `0.4018` / `83.26%` / `0.9288` | **`0.3748`** / **`85.12%`** / **`0.9388`** | CE `-0.0371`, Acc `+0.93 pp`, AUC `+0.0302` |
| `v154` | 226 | `0.3076` / **`88.50%`** / `0.9443` | `0.3652` / **`88.50%`** / `0.9407` | **`0.2846`** / `88.05%` / **`0.9589`** | CE `-0.0230`, Acc `-0.44 pp`, AUC `+0.0146` |
| `v165` | 199 | `0.5165` / `77.89%` / `0.8534` | `0.5434` / `80.40%` / `0.8776` | **`0.4215`** / **`83.92%`** / **`0.9128`** | **最難場次**：CE `-0.0951`, Acc **`+6.03 pp`**, AUC **`+0.0594`** |
| `v168` | 218 | `0.4540` / `77.98%` / `0.8723` | `0.4822` / `81.19%` / `0.8915` | **`0.3820`** / **`83.03%`** / **`0.9338`** | **次難場次**：CE `-0.0720`, Acc **`+5.05 pp`**, AUC **`+0.0614`** |
| **合計** | **1,547** | **`0.4177` / `81.64%` / `0.8992`** | **`0.4113` / `84.62%` / `0.9181`** | **`0.3559` / `85.07%` / `0.9389`** | **7/8 場 CE 與 AUC 優於 ACA 及 Swin Temp50（vs Swin Temp50 之 8 場 Acc 為 3 升 1 平 4 降）** |

### 4.2 `seed20261025` 逐場演進（`ACA` → `Swin3D-T Temp50` → `Swin3D-T Spatial+Temp50`）

| 比賽 ID | 片段數 | `ACA Temporal50` (CE / Acc / AUC) | `Swin3D-T Temporal50` (CE / Acc / AUC) | `Swin3D-T Spatial+Temp50` (CE / Acc / AUC) | `Swin Spatial` vs `Swin Temp50`／vs `ACA` |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `v113` | 211 | `0.4033` / `80.57%` / `0.9031` | `0.4228` / `81.99%` / **`0.9167`** | **`0.4011`** / **`82.46%`** / `0.9113` | vs ACA: CE `-0.0022`, Acc `+1.90 pp`, AUC `+0.0082` |
| `v123` | 221 | `0.3896` / `84.62%` / `0.9141` | `0.3355` / **`87.33%`** / **`0.9366`** | **`0.3321`** / `85.97%` / `0.9324` | vs ACA: CE `-0.0575`, Acc `+1.36 pp`, AUC `+0.0183` |
| `v149` | 31 | **`0.2965`** / **`90.32%`** / **`0.9714`** | `0.5165` / `83.87%` / `0.9429` | `0.4131` / `87.10%` / `0.9381` | 小樣本（31 支）；加空間增強後 CE 降 `-0.1033`、Acc `+3.23 pp` |
| `v152` | 226 | `0.3565` / `81.42%` / `0.9306` | `0.2750` / `88.50%` / `0.9551` | **`0.2482`** / **`89.82%`** / **`0.9606`** | vs ACA: CE **`-0.1083`**, Acc **`+8.41 pp`**, AUC **`+0.0300`** |
| `v153` | 215 | `0.3876` / `81.86%` / `0.9087` | `0.3869` / `85.12%` / `0.9229` | **`0.3405`** / **`86.05%`** / **`0.9267`** | vs ACA: CE `-0.0471`, Acc `+4.19 pp`, AUC `+0.0180` |
| `v154` | 226 | `0.3447` / `84.96%` / `0.9321` | `0.4212` / `85.84%` / `0.9277` | **`0.2881`** / **`88.50%`** / **`0.9584`** | vs ACA: CE `-0.0566`, Acc `+3.54 pp`, AUC **`+0.0263`** |
| `v165` | 199 | `0.4585` / `80.90%` / `0.8730` | `0.4610` / `80.40%` / `0.9049` | **`0.4015`** / **`84.42%`** / **`0.9087`** | **最難場次**：vs ACA Acc **`+3.52 pp`**, AUC **`+0.0357`** |
| `v168` | 218 | **`0.3830`** / `81.65%` / `0.9089` | `0.4719` / `80.28%` / `0.9105` | `0.3901` / **`83.03%`** / **`0.9216`** | **次難場次**：加空間增強後 CE 降 `-0.0819`, Acc `+2.75 pp` |
| **合計** | **1,547** | **`0.3859` / `82.48%` / `0.9116`** | **`0.3972` / `84.29%` / `0.9231`** | **`0.3427` / `85.84%` / `0.9308`** | **vs Swin Temp50：8/8 場 CE 改善、7/8 場 Acc 改善；vs ACA：7/8 場 Acc 與 AUC 改善** |

### 關鍵現象觀察與待驗證假說
1. **已觀察到的跨種子一致性（事實）**：在 `Swin3D-T` 上加入 `clip_flip_color_light_v2` 後，兩組 seed 的整體 Val CE 降幅高度接近（`-0.0554` 與 `-0.0545`），Brier 同步降至 `0.1066` / `0.1054`，最困難的 `v165`（`83.92%` / `84.42%`）與 `v168`（`83.03%` / `83.03%`）在兩組 seed 皆達 `83–84%`；同時也觀察到 `s24` 存在類別與場次準確率取捨（Make Recall `-2.85 pp`、8 場 Acc 3 升 1 平 4 降）。
2. **已觀察到的後期過擬合曲線（事實）與可能原因（假說，尚未因果證實）**：
   - **觀察事實**：從 `history.json` 可見無空間增強組在後期 `Train Acc` 升至 `95.6–96.1%`、Val CE 反彈至 `0.80–0.90`；加入空間增強後，後期訓練擬合幅度變緩（如 `s25` Ep 8 Train Acc `95.6% → 92.4%`、`s24` Ep 10 Train CE `0.1329 → 0.2003`），但第 6–8 輪後仍出現 Train CE 持續下降、Val CE 上升的過擬合趨勢。
   - **待驗證假說**：空間增強之所以降低後期過擬合，一個可能假說是抑制模型過度依賴特定視角方向或光影／背景外觀，但目前尚未進行特徵歸因實驗直接證實模型記憶了場館或球衣；同樣地，後期 Val CE 反彈是否由固定 `LR=1e-4` 所主導，仍須透過下一階段單因子學習率排程實驗驗證。

---

## 5. 配對同意矩陣（McNemar 視角）

| 比較組（`N = 1,547`） | 皆答對 | 皆答錯 | 僅對照答對 | 僅候選答對 | 淨多答對 | 探索性雙側 exact McNemar $p$ 值 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Swin3D-T Temp50 (s24)` vs `ACA Temp50 (s24)` | 1,165 | 140 | 98 | **144** | **`+46`** | `0.00373` |
| `Swin3D-T Temp50 (s25)` vs `ACA Temp50 (s25)` | 1,182 | 149 | 94 | **122** | **`+28`** | `0.06594` |
| `Swin3D-T Spatial+Temp50 (s24)` vs `Swin3D-T Temp50 (s24)` | 1,225 | 147 | 84 | **91** | **`+7`** | `0.65027` |
| `Swin3D-T Spatial+Temp50 (s25)` vs `Swin3D-T Temp50 (s25)` | 1,253 | 168 | 51 | **75** | **`+24`** | `0.04004` |
| **`Swin3D-T Spatial+Temp50 (s24)` vs `ACA Temp50 (s24)`** | 1,166 | 134 | 97 | **150** | **`+53`** | `0.00090` |
| **`Swin3D-T Spatial+Temp50 (s25)` vs `ACA Temp50 (s25)`** | 1,207 | 150 | 69 | **121** | **`+52`** | `0.00020` |
| *探索性：`Swin3D-T Spatial + AdamW (s24)` vs `Swin3D-T Spatial + Adam (s24)`* | 1,244 | 139 | 72 | **92** | **`+20`** | `0.13766` |
| *探索性：`Swin3D-T Spatial 224×224 (s24)` vs `Swin3D-T Spatial 120×160 (s24)`* | 1,229 | 115 | 87 | **116** | **`+29`** | `0.04912` |
| *探索性：`Swin3D-T Spatial + TAttn (s24)` vs `Swin3D-T Spatial MeanPool (s24)`* | 1,232 | 134 | 84 | **97** | **`+13`** | `0.37246` |

> **統計限定說明**：上述雙側 exact McNemar $p$ 值**僅供探索性描述**。由於 1,547 支片段聚集於 8 場比賽（非獨立同分布抽樣）、模型係依同一 validation 集選出，且本專案歷經多次比較，這些 $p$ 值**不可直接作為跨比賽泛化之統計顯著性證明**。

### 5.1 探索性單因子消融解讀：`AdamW (weight_decay=1e-2)`（1-seed, `seed20261024`）

為檢驗「在固定 `LR=1e-4` 下僅加入 `AdamW(weight_decay=1e-2)` 能否壓低 `s24` 的 Make-prob ECE 與後期過擬合」，在與 `Swin3D-T Spatial+Temp50 (s24)` 100% 同初始化、同增強條件下完成單因子對照（同樣通過 CPU/GPU 接線驗收，原子帳本 `5/8`、6/6 續訓逐值一致）：

- **正向訊號**：Accuracy 達專案單輪新高 **`86.36%`（`+1.29 pp`、淨多答對 `+20` 支，單場 `v123` 達 `90.95%`、`v154` 達 `89.82%`）**、Macro-F1 達 **`0.8623`（`+0.0118`）**、Make-prob ECE 由 `0.0722` 降至 **`0.0635`（`-0.0087`）**。
- **明確取捨（尚未取代主線基準）**：依專案鎖定主指標 **Val CE** 由 `0.3559` 升至 **`0.3853`（`+0.0294`）**，無門檻排序指標 **ROC-AUC**（`0.9389 → 0.9319`，僅 `1/8` 場改善）、AP Make（`0.9420 → 0.9340`）、Brier（`0.1066 → 0.1103`）與 Miss Recall（`85.95% → 80.27%`）均退步；預測傾向偏向 Make（`57.40%` vs 真實 `52.17%`），使 `v113`（CE `0.3983 → 0.5275`）與最難兩場 `v165`/`v168` 略降，8 場呈 **4 升 4 降**。
- **實驗觀察**：在固定 `LR=1e-4` 下僅改用 `AdamW(wd=1e-2)` 並未改變後期 Val CE 上升的早停走勢（同於 Ep 6/11 停止，Ep 11 Train Acc 達 `94.40%`、Val CE 升至 `0.6631`），因此尚未取代主線設定。

### 5.2 探索性輸入規格比較：`224×224` 整幀 `cv2.INTER_AREA` 縮放（1-seed, `seed20261024`）

基於 8 類錯誤結構解剖（發現運動戰 `2p+3p+mp` 已達 `87.3–89.2%`，錯誤集中於罰球沒進 `ft0`——僅佔 5.8% 樣本卻佔 25.7% 跨種子共錯、平均 CE 高達 `1.11`），在沿用 `cv2.INTER_AREA` 插值、直接將原片 `320×240` 整幀縮放至 `224×224`（不裁切、無 `120×160` 中介；明示解析度、`4:3 → 1:1` 幾何比例變形與 Stage 3 視窗配置同時改變，屬**輸入規格比較**）、相同權重初始化與 `Adam(lr=1e-4, wd=0)` 條件下完成本組實驗。驗收歷程揭露：CPU 首次因 `list`／`tuple` 型別比對失敗（紀錄保留於 `cpu_report_attempt1_list_vs_tuple.json`，0 GPU 更新），修正後通過；GPU 驗收單次通過（原子帳本 `5/8`、6/6 續訓逐值一致，Stage 3 實測 `[25, 7, 7]` / 視窗 `[8, 7, 7]` / 第二 block shift `[4, 0, 0]`，峰值顯存 `5.3539 GiB` alloc / `7.1035 GiB` reserved；GPU 探針之模擬 validation 係以 train fixture 測試且零解碼真實 validation，正式零增強 validation 路徑另由程式查核與診斷重現確認）。另備註：已鎖定之 `plan.json` 沿用父計畫 `temporal_sampling` 欄位文字中仍保留舊 `160x120` 描述，實際解碼與縮放嚴格依 `input_contract_224`（`320×240 → 224×224` `cv2.INTER_AREA`）執行，為維持計畫不可變性不回改舊檔。

- **整體彙總表現（Best Ep 5/10）**：
  - **Accuracy 達 `86.94%`**（`1,345 / 1,547`，較 `120×160 Spatial s24` **`+1.87 pp`** / 淨多答對 **`+29` 支**，探索性雙側 exact McNemar `p = 0.04912`；8 場 Accuracy 為 **5 場上升改善、2 場持平、1 場下降**，其中 `v152` `91.15%`、`v123` `90.50%`、`v149` `90.32%`）、**Macro-F1 `0.8686`（`+0.0180`，創專案單輪新高）**、**Brier `0.1041`（`-0.0025`，創專案單輪新低）**、**Make-prob ECE 降至 `0.0627`（`-0.0095`）**。
  - **主指標 Val CE 與無門檻排序未優於原規格（存在明確類別取捨）**：鎖定主指標 **Val CE 為 `0.358751`**（較 `120×160` 之 `0.355902` 略高 `+0.002849`；**8 場 CE 為 3 場下降改善、5 場上升惡化**）、**ROC-AUC `0.9385`**（`-0.0004`，基本持平）、AP Make `0.9413`（`-0.0007`）；Miss Recall 由 `85.95%` 降至 `82.43%`，Make Recall 由 `84.26%` 升至 `91.08%`。
- **預先指定之 `ft`／`ft0` 瓶頸檢驗與運動戰配對變化（僅作診斷，不用於選模）**：
  - **淨多答對 `+29` 支的結構來源**：`ft1` 增加 **`+59` 支**、`ft0` 減少 **`-24` 支**、運動戰（`2p+3p+mp`）減少 **`-6` 支**。
  - **罰球家族（`ft`, $N=367$）**：Acc `71.93% → 81.47%`、CE **`0.5520 → 0.4602`**、無門檻 ROC-AUC `0.7821 → 0.7947`；惟可觀察到 `ft` 預測 Make 比例由 `67.03%` 升至 `89.65%`、`ft1` 達 `97.11%`（`+21.30 pp`）、**`ft0` 則由 `60.00%` 降至 `33.33%`（`-26.67 pp`，CE `0.9856 → 1.4164`）**，罰球 balanced accuracy 反由 `67.91%` 降至 `65.22%`（呈現明確類別取捨，尚無因果證據將此偏移單獨歸因於訓練集先驗）。
  - **運動戰配對退步（`2p+3p+mp`, $N=1{,}180$）**：合計 Acc `89.15% → 88.64%`、CE `0.2949 → 0.3272`、ROC-AUC `0.9568 → 0.9524`（其中 `2p` Acc `87.41% → 85.81%`／AUC `0.9363 → 0.9235`、`3p` Acc `91.12% → 90.74%`／AUC `0.9711 → 0.9650` 皆較 `120×160` 略退，僅 `mp` Acc `87.85% → 89.25%`／AUC `0.9519 → 0.9616` 改善）。
  - **嚴謹結論**：本 seed 所選 `224×224` 模型在整體 Accuracy、Macro-F1、Brier 與 ECE 改善，但最低 CE 與 AUC 未優於 `120×160`，且 **未改善 `ft0`（明顯退步）**；可保留為探索候選，尚不足以取代既有主線或宣稱已改善籃框細節辨識。籃框 ROI 或家族均衡監督皆列為待驗證假說。

### 5.3 探索性單一池化替換：`Gated Temporal Attention Pooling`（1-seed, `seed20261024`）

為檢驗「以可學習時間注意力取代 25 步等權重平均能否改善整體 CE 並減輕短暫結果訊號被稀釋之問題」（經 validation 診斷引導之探索，採 Ilse et al. 2018 式 9 形式，本專案設定為 `hidden=128`、選項 A 初始化 `w=0`、`V/U` 固定截斷 `[-0.04, 0.04]`、獨立 `attn_seed=20265024` 全域 RNG 隔離，共新增 196,992 參數並納入全域 `clip_grad_norm_(1.0)` 與原 `Adam(lr=1e-4, wd=0)`），於 `120×160 Spatial+Temp50` 基底完成單一池化替換實驗。驗收與實作範圍揭露：
- 400 類骨幹載入係以 `load_state_dict(..., strict=False)` 執行並嚴格核對僅缺少 5 個新增 attention 鍵、無 unexpected keys，隨後逐位元驗證 183 個非 head 張量（metadata 摘要欄位之舊字樣未回改以保持已鎖定計畫不可變）。
- CPU 驗收（10.25 秒）：185 基底張量逐位元一致、初始 logits 誤差 `1.49e-8`、**合成 pooling 輸入特徵 `h` 之梯度差 `5.82e-11`**（非影片像素經完整 Swin 反向梯度差）、2 步合成更新 `V/U/w` 梯度流通過、`0` 次 Swin3D-T 完整模型更新；另註 `validation_zero_aug_fixture_check_passed` 僅檢查既有 fixture 經 `BinaryCollator` 後無增強 metadata，`SpatialDataset` 之 validation 分支委派原載入器則由程式查核確認。
- GPU 驗收（14.28 秒）：原子帳本 `5/8`、6/6 續訓逐值一致且 5 個新張量均已更新並於 model／`best_model`／Adam 狀態逐位元一致，峰值 `3.3928 / 4.3574 GiB`。

- **整體彙總指標全數改善（Best Ep 6/11）**：
  - **主指標 Val CE：`0.355902 → 0.351489`（`-0.004413`）**，為 `seed20261024` 已列模型最低 CE（8 場 CE 為 **4 場下降改善、4 場上升惡化**）。
  - **Accuracy：`85.07% → 85.91%`（`+0.84 pp`、淨多答對 `+13` 支**，探索性雙側 exact McNemar `p = 0.37246`；8 場 Acc 為 5 升 1 平 2 降，單場 `v123` 達 `91.40%`、`v153` 達 `89.30%`）、**Macro-F1：`0.8506 → 0.8578`（`+0.0072`）**。
  - **排序與校準指標**：**ROC-AUC `0.9394`（`+0.0005`）**、**AP Make `0.9430`（`+0.0010`）** 與 **Brier `0.1033`（`-0.0033`）** 刷新已列模型單輪紀錄；**Make-prob ECE 降至 `0.0571`（`-0.0151`）**，但仍高於 `Spatial s25` 之 `0.0392`。
- **明確類別取捨（依預先登錄規則，不自動採用）**：
  - **淨多答對 `+13` 支來源**：`ft1 +47`、`ft0 -21`、運動戰 `-13`。整體 Miss Recall 由 `85.95%` 降至 `79.86%`（`-6.08 pp`），Make Recall 由 `84.26%` 升至 `91.45%`（`+7.19 pp`）。
  - **罰球家族（`ft`, $N=367$）**：CE `0.5520 → 0.4737`（`-0.0784`）、Acc `71.93% → 79.02%`、ROC-AUC `0.7821 → 0.7875`（`+0.0054`）；惟 `ft` 預測 Make 比例由 `67.03%` 升至 `85.56%`，`ft0` Recall 由 `60.00%` 降至 `36.67%`（CE `0.9856 → 1.2226`），罰球 balanced accuracy 由 `67.91%` 降至 `64.72%`。
  - **運動戰（`2p+3p+mp`, $N=1{,}180$）**：無門檻 ROC-AUC 維持 `0.9568 → 0.9574`（其中 `mp` 提升 `87.85% → 90.19%` / AUC `0.9519 → 0.9680`），但 `2p`（`87.41% → 84.67%`）與 `3p`（`91.12% → 89.98%`）略退，合計 Acc `89.15% → 88.05%`、CE `0.2949 → 0.3135`。
- **25 個具脈絡時間 token 之注意力權重診斷（純描述性診斷，不作選模條件）**：
  - 25 個輸出時間 token 已經過 Swin 多層時間交互，指標統計為：全體 `mean_alpha_max = 0.0968`、正規化 entropy `0.9500`、有效位置數 `21.35 / 25`（高 entropy 與約 21–23 個有效位置顯示整體權重仍相當分散）。
  - **群體平均權重偏重末段**：全體、運動戰與罰球之群體平均權重前 5 名皆為末段 token `[24, 23, 22, 21, 20]`，最後 5 個 token 平均權重總和為全體 `36.87%`、運動戰 `38.22%`、罰球 `32.53%`；單片最大權重落於 token 24 之比例為全體 `57.47%`、`ft` `41.69%`。
  - `ft` 群體平均分布較平緩（`mean_alpha_max = 0.0728`、有效位置數 `22.73 / 25`，對比運動戰 `0.1043` / `20.92 / 25`；未做顯著性檢定，亦不推論模型依賴原片尾端動作或已定位碰框時刻）。
  - **保存精度揭露**：JSONL 檔中 `alpha_25` 保存至小數第 6 位（每列總和最大捨入誤差約 `5×10⁻⁶`），重新正規化後可於報告顯示精度內重現 entropy 與有效位置數，但不宣稱保存之 `alpha_25` 可逐位元重現原始統計；未來診斷宜改存完整 float32 可往返精度。
- **判定結論**：全部整體彙總指標（含主指標 Val CE）均小幅優於同種子平均池化基準，但原本針對 `ft0` 之動機尚未得到支持且存在明確 Miss recall 與 `ft0` 退步取捨；保留平均池化主線，將本組列為**有類別取捨、可做 `seed20261025` 確認的有限候選**（目前證據不支持直接採用，亦不支持據此跳至 RL 選幀或 `SpinalNet`）。

---

## 6. 產物清單與 SHA-256 稽核表

### 6.1 正式訓練產物（對稱 4×2 主線四組＋3 組 1-seed 探索實驗）
| 實驗名稱 | `plan_sha256` | `summary.json` SHA-256 | `last.pt` SHA-256 |
| :--- | :--- | :--- | :--- |
| [swin3d_t_binary50_seed20261024_labelrev1_v1](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_seed20261024_labelrev1_v1/summary.json) | `6bd908ced780140ebf88fae132a566f338eb2b04e6ce76b5a9c44d5d6ff2787c` | `e9ebea3034dff3fdc4d0ced0a76dfba9e41c087eb8246613b044d6e71b35e3bf` | `d96f40c6d0fe317184c6a3c8d44fed0fc393d8d666053f39bf017990187a3523` |
| [swin3d_t_binary50_seed20261025_labelrev1_v1](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_seed20261025_labelrev1_v1/summary.json) | `7737737be27b526a8e474fbc16d87bae6f2e187b4936ae8a80d7a32ac9840f54` | `0c887332b3724336e6a9cc19fc721dda68f3a310e7f3a9c18f9a1b04e3ae3803` | `9ef1d405a53f6e399d138e798c1ffff7a9de2685b503c50343ae0b87bc41106f` |
| [swin3d_t_binary50_spatial_seed20261024_labelrev1_v1](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_spatial_seed20261024_labelrev1_v1/summary.json) | `2318e391357792bd4ba69d9a614e64c227b2b2e7a9b46d5491451468c5a53490` | `1b402def9b55781eba9654b722877ffc5024a11082ebb1f7cfed43f9a5ea698a` | `0441e314e2a79bc77da4736d0fcbfbf7c5c7a3081048d604e7cf2cb199e66279` |
| [swin3d_t_binary50_spatial_seed20261025_labelrev1_v1](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_spatial_seed20261025_labelrev1_v1/summary.json) | `7fd4d55761135606daec62d8e3ce5099f5edce036c594486eb278a5bea4f9b1a` | `bbed0726180a278ef717eae80f2086ff6c002497d241d4fb69cc4455e5192630` | `7bdf4da94bbea06ea75e0dac756a6ec83cd027aab2a659669bc5522296a1218e` |
| [swin3d_t_binary50_spatial_adamw1e2_seed20261024_labelrev1_v1](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_spatial_adamw1e2_seed20261024_labelrev1_v1/summary.json) | `1bf0587e1e08519b6868a3383b995bc0ca70dac10f0226baab4fdd4a582534e0` | `01fbb18699353543f9f4137d6827ffb54390967d05f63e794a8c9334f4d103d7` | `93796aa9e67d20dfdbaa0456531da40c734716a58bbe1f3c5bc631e8bb143d07` |
| [swin3d_t_binary50_spatial_224_seed20261024_labelrev1_v1](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_spatial_224_seed20261024_labelrev1_v1/summary.json) | `51fb8762036b47f5b0963cf166ed0976d6b5a04a9b20fd4b2b0877c1e38549d2` | `cfcad9e3c192efff8fe99c1fa5fcbfa47ebd4c764902e30f773622a32ad46968` | `2a51284dbe4240e8ceb01a4af89d5ddf8a574dbd33107e2100d77fdbcff7d74b` |
| [swin3d_t_binary50_spatial_tattn_seed20261024_labelrev1_v1](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_binary50_spatial_tattn_seed20261024_labelrev1_v1/summary.json) | `9311d0da5de4bac42afe21e3c50996515eb9c314dec0218c7f398773c486f359` | `30265c9a61ab164123d91066155365aff5b05e826979a26f0f3116390a112f2f` | `907d6ff311d75a8656e54773309cde7eadb4b33b807804456dc83ce891a000d5` |

### 6.2 唯讀驗證診斷與逐片 Logits（全部七組，`0` 更新、`0` test 接觸）
- `seed20261024`（Swin3D-T Temp50 vs ACA）：[diagnostic.json](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_seed20261024_val_diagnostic_20261125_v1/diagnostic.json)｜`per_clip_predictions.jsonl` SHA-256：`c5b8f40b0cbabd1421ce4fa23da068a7ad17da0f4d8435f771886ff4a2603780`
- `seed20261025`（Swin3D-T Temp50 vs ACA）：[diagnostic.json](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_seed20261025_val_diagnostic_20261125_v1/diagnostic.json)｜`per_clip_predictions.jsonl` SHA-256：`612e2a8e1cfdee8061a34386fe4f1fa070c4f104cdaf6defd7c96b6f5a30cbc4`
- `seed20261024`（Swin3D-T Spatial vs Temp50）：[diagnostic.json](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_spatial_seed20261024_val_diagnostic_20261125_v1/diagnostic.json)｜`per_clip_predictions.jsonl` SHA-256：`c60a096b109859ef214623967cdf496ef335aec750cbaa72704ac23e252650ba`
- `seed20261025`（Swin3D-T Spatial vs Temp50）：[diagnostic.json](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_spatial_seed20261025_val_diagnostic_20261125_v1/diagnostic.json)｜`per_clip_predictions.jsonl` SHA-256：`ced664b8b8b779c48fb03a26afd0702467f146c8a64c0a4d02da8a7520afc145`
- `seed20261024`（Swin3D-T Spatial AdamW vs Adam）：[diagnostic.json](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_spatial_adamw1e2_seed20261024_val_diagnostic_20261126_v1/diagnostic.json)｜`per_clip_predictions.jsonl` SHA-256：`a8c7d55682c2f7e96c076fd1343bc4cf1c6366cbfbd3d570b895fabc5a5c4d1b`
- `seed20261024`（Swin3D-T Spatial 224×224 vs 120×160）：[diagnostic.json](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_spatial_224_seed20261024_val_diagnostic_20261126_v1/diagnostic.json)｜`per_clip_predictions.jsonl` SHA-256：`6fa2a418522f34dd95f0eaf8654db7d9a50efd2181b64e384e25b063ac7db476`
- `seed20261024`（Swin3D-T Spatial TAttn vs MeanPool）：[diagnostic.json](file:///home/tsai/projects/basketball/outputs/01_current/swin3d_t_spatial_tattn_seed20261024_val_diagnostic_20261126_v1/diagnostic.json)｜`per_clip_predictions.jsonl` SHA-256：`b4e3e398a6b1c0023ea2798cfde8e772c594febf54a19b610ce581edae410dde`

---

## 7. 科學限制與誠實揭露

1. **樣本與種子數限制**：四組主線設定已完成對稱 2 組 random seed（`20261024`、`20261025`），`AdamW(wd=1e-2)`、`224×224` 與 `Gated Temporal Attention` 皆為 1-seed 探索實驗；**兩個 seed 共享同一份資料而非兩份獨立資料**，兩組平均僅屬描述性摘要，尚不宣稱統計顯著。
2. **評估集界線**：全部數字均為**反覆用於選模與研究決策之 8 場未見比賽驗證集（1,547 支）**結果；本輪未接觸 `test` split（`test_run: false`），並持續揭露專案歷史 `test` 接觸紀錄，尚不能宣稱獨立 `test` 泛化或達成 90% 準確率。
3. **跨種子校準波動與罰球結構瓶頸**：`AdamW`、`224×224`、`Gated Temporal Attention` 三組單種子探索實驗均未改善 `ft0`（分別為 `38.89%`、`33.33%`、`36.67%`），罰球家族 ROC-AUC（`0.782–0.795`）與運動戰投籃（`0.952–0.957`）之結構落差依然存在。

---

## 8. 下一階段最值得推進的優先順序建議

1. **優先 1（確認 `Gated Temporal Attention` 有限候選之跨種子穩定性 或 純單因子 `CosineAnnealingLR` 對照）**：
   - `Gated Temporal Attention` 為三組探索中唯一小幅改善主指標 Val CE（`0.3559 → 0.3515`）之有限候選，但有類別取捨，可補跑 `seed20261025` 確認其跨種子一致性；
   - 或依 Codex 先前建議，在 `120×160 Spatial+Temp50 + Adam (wd=0)` 平均池化主線進行純單因子 `CosineAnnealingLR` 對照（先鎖定 `T_max`、`eta_min`、步進單位與 scheduler checkpoint 續訓一致性）。
2. **優先 2（與教授討論後評估針對 `ft`／`ft0` 之任務／空間假說）**：
   - 既有證據顯示模型在運動戰已達 `~0.957` AUC，但罰球家族僅 `~0.788–0.795` AUC 且 `ft0` 召回偏低；目前證據尚不支持直接跳至 RL 選幀或 `SpinalNet`，建議今晚與教授討論：(a) 8 類／家族內均衡監督；(b) 籃框 ROI 偵測與局部聚焦（確認實驗室有無現成籃框偵測權重）；(c) 整體後續研究優先序。
