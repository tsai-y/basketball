# Basketball-51 階段性資料品質技術報告（第三次修訂版）

## 1. 報告元資料與研究任務定義

### 核心元資料

| 元資料項目 | 詳細資訊／設定值 |
| --- | --- |
| 快照日期 | 2026-10-16（Asia/Taipei） |
| Final commit hash | `af06065e38a9e6263a19105332a1dc2c81f7cc4e` |
| Git branch | `codex/data-quality-audit-v1` |
| Audit schema | `basketball.train-quality-associations.v1` |

### 研究任務

本專案的核心任務為 **Full-video Make/Miss classification**。模型輸入未強制限制於出手前，因此可能包含出手後內容，但不保證每支影片完整涵蓋飛行、碰框、入網或球員反應等階段。目前沒有出手點標註或 pre-release 時間限制，因此不得宣稱本研究執行 pre-release prediction。

### 資料切分邊界

- Train、validation 與 test 的 `source_group` 沒有交叉；目前只能稱為 **source-group holdout**，尚未證實為 game-level holdout。
- 跨 split 的 near-duplicate content scan 尚未完成。
- Test 維持 sealed，不得參與門檻、採樣器、增強政策或其他模型選擇。
- 本階段 validation/test quality mode 為 `disabled`。本次 Candidate–Label Association Audit 僅分析 train derived metadata；audit 本身的影片操作為 opens/probes/decodes = 0/0/0，且 validation/test quality data analyzed = 0。
- 未來若要調整 held-out data 的 quality 政策，必須另立並凍結 protocol；test 仍不得參與模型選擇。

## 2. 資料品質管線演進：v1 至 v3.1

```text
v1 探索 → v2 校準與選規則 → v3 全量 Train 掃描 → v3.1 驗證與發布
```

### v1：探索

v1 比較原始 Laplacian variance 與 Tenengrad 等影像訊號。探索結果顯示，全域絕對門檻容易受到來源、畫面紋理與拍攝條件影響，因此未將 v1 規則作為正式資料刪除或排除政策。

### v2：校準與選規則

v2 使用合成模糊、source-disjoint calibration/verification 與影片內 local robust z 比較四種策略。Laplacian 規則在 verification subset 取得最高 F1，因此凍結供 v3 掃描使用。

### v3：全量 Train 掃描

v3 使用凍結的 v2 Laplacian 規則掃描 7,193 支 train 影片，實際解碼 1,105,146 幀並產生 candidate metadata。這一階段只建立品質註記，沒有刪除影片、丟幀、重編碼、修改標籤或訓練模型。

### v3.1：驗證與發布

v3.1 驗證 manifest、影片 ID、config/hash、ledger、run identity、derived-output 完整性與 aggregate 一致性，並發布版本化、可驗證且具備來源追溯資訊的 sidecar。此階段不重新計算 candidate，也不 stat、open、probe 或 decode 原始影片。v3.1 不是新的模糊偵測器，也沒有重新擬合門檻。

必須區分：1,105,146 幀是 v3 full-train scanner 的解碼結果；opens/probes/decodes = 0 是後續 association audit 與 v3.1 metadata 驗證／發布流程的影片存取結果。

| 階段 | 主要目標 | 規則狀態 | 影片操作 | 主要產出 |
| --- | --- | --- | --- | --- |
| v1 | 訊號探索 | 全域門檻未採納 | Train pilot | 探索性診斷 |
| v2 | 校準與策略比較 | 選定 Laplacian local-z 規則 | Train calibration/verification subsets | 凍結規則 |
| v3 | 全量 train 掃描 | 使用凍結 v2 規則 | 解碼 1,105,146 幀 | Candidate metadata |
| v3.1 | 完整性、一致性與 provenance 驗證 | 不改規則 | 不操作原始影片 | Tracked sidecars |

## 3. 影像品質指標與數學公式

### Raw Laplacian score

Raw Laplacian score 為影像經 Laplacian 算子處理後 response 的 population variance。

### Tenengrad

本專案使用的 Tenengrad 分數為：

```text
Tenengrad = mean(Gx² + Gy²)
```

其中 `Gx` 與 `Gy` 分別為 Sobel x/y 梯度。此實作不包含平方根、像素門檻 `T`，也沒有只累加高於門檻的梯度。Tenengrad 在 v3 僅作診斷，不參與 primary candidate 判定。

### 影片內 3 秒 local robust z

```text
log_score_t = log(max(score_t, 1e-12))

window_width = ceil(3 × FPS)
若 window_width 為偶數，則調整為奇數。

median_t = median(log_score_j for j in W_t)
MAD_t = median(abs(log_score_j - median_t) for j in W_t)

local_z_t = (log_score_t - median_t)
            / max(1.4826 × MAD_t, 0.05)
```

`W_t` 是以 frame `t` 為中心的視窗，在影片首尾處截短。`1e-12` 避免對零取 logarithm；分母下限 `0.05` 避免 MAD 過小造成數值不穩定。

### v2 四策略比較

| 策略 | z 門檻 | 最短持續時間 | Calibration F1 | Verification F1 | 決策 |
| --- | ---: | ---: | ---: | ---: | --- |
| Laplacian | -2.0 | 0.16 秒 | 0.9012 | 0.8617 | v3 primary rule |
| Tenengrad | -2.5 | 0 秒 | 0.8752 | 0.8108 | 診斷比較 |
| Both | -1.0 | 0.16 秒 | 0.8904 | 0.8590 | 未採用 |
| Either | -3.0 | 0 秒 | 0.8923 | 0.8318 | 未採用 |

v3 的 primary candidate rule 是 Laplacian `local_z <= -2.0`，且候選持續時間至少 0.16 秒。

## 4. v3.1 Sidecar 與全量統計

### 發布檔案

```text
development/data_quality/sidecars/blur_quality_sidecar_v3_1.json
development/data_quality/sidecars/blur_video_summary_v3_1.csv
development/data_quality/sidecars/blur_candidate_segments_v3_1.csv
```

目前沒有發布 Parquet sidecar。

### `blur_video_summary_v3_1.csv` 欄位

```text
video_id,label,source_group,fps_numerator,fps_denominator,fps_rational,
frame_count,duration_seconds,candidate_frame_count,candidate_ratio,
candidate_segment_count,longest_candidate_segment_frames,
has_clean_16,has_clean_32,has_clean_64,config_hash,run_identity
```

### `blur_candidate_segments_v3_1.csv` 欄位

```text
video_id,start_frame,end_frame_exclusive,start_timestamp,
end_timestamp_exclusive,candidate_frame_count,duration_seconds,
policy_version,scanner_version,config_hash,run_identity
```

### Full-train 統計

| 項目 | 數值 |
| --- | ---: |
| Train videos | 7,193 |
| Decoded frames | 1,105,146 |
| Candidate frames | 18,390（約 1.664%） |
| Candidate segments | 1,798 |
| Affected videos | 1,698 |
| Unaffected videos | 5,495 |
| Corrupt / incomplete / unresolved | 0 / 0 / 0 |

## 5. Quality-Aware Clip Sampler 與 Fallback

### 採樣規則

對預設長度 64 幀的連續視窗：

1. 優先從 candidate overlap = 0 的可行視窗池中選取。
2. 如果不存在零重疊視窗，則從全體 minimum-overlap pool 選取並標記 `fallback=true`。
3. 視窗保留原始連續 frame index；不丟幀、不拼接、不插值、不修改標籤。
4. 在固定 global seed、epoch 與 hash 規則下，選取結果可以重現。

Sampler 程式本身不硬編碼 clip length；目前 I3D pilot 設定為每支影片選一個 64-frame clip。

### Fallback 統計

- 7,188 支影片選到 zero-overlap 64-frame window。
- 5 支影片使用 minimum-overlap fallback。
- 五個 fallback 視窗的 overlap 合計為 20 幀，個別為 1、8、5、5、1；最大個別 overlap 是 8。
- 系統沒有「單支影片最多允許 overlap 20 幀」的硬限制。

| video_id | label | frames | segments | selected window | overlap | clean 16/32/64 |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `ft0/ft0_v166_002702_x264.mp4` | Miss | 146 | 2 | `[0,64)` | 1 | true/true/false |
| `ft0/ft0_v181_002018_x264.mp4` | Miss | 145 | 3 | `[65,129)` | 8 | true/true/false |
| `ft1/ft1_v111_005842_x264.mp4` | Make | 180 | 4 | `[22,86)` | 5 | true/true/false |
| `ft1/ft1_v133_011206_x264.mp4` | Make | 146 | 3 | `[0,64)` | 5 | true/true/false |
| `ft1/ft1_v169_004222_x264.mp4` | Make | 146 | 2 | `[82,146)` | 1 | true/true/false |

目前政策保留全部影片；candidate 與 fallback 都不是刪除授權。

## 6. Candidate–Label 描述性關聯

本報告中的 candidate 僅表示 **low-local-Laplacian candidate**，不是 confirmed blur、模糊真值、壞影片、品質瑕疵或錯誤標籤。

### 未調整結果

| 指標 | Make | Miss | Make − Miss |
| --- | ---: | ---: | ---: |
| Mean candidate ratio | 2.25% | 1.01% | 1.24 個百分點 |
| Affected rate | 30.99% | 15.46% | 15.53 個百分點 |

Cliff's delta 為 0.1622，表示小幅正向 rank effect。正值表示隨機抽取一支 Make 的 candidate ratio 高於 Miss 的機率較高；這不代表 label 造成 candidate。

### 描述性分層調整

在同一 `source_group × shot_type` 且兩個 label 都存在的共同支撐範圍內，以 overlap weight 進行描述性分層調整後：

- Mean candidate-ratio difference：0.93 個百分點。
- Affected-risk difference：12.29 個百分點。

此處的 `shot_type` 是投籃類型，例如 `2p`、`3p`、`ft`、`mp`，不是 camera shot、鏡位或運鏡模式。分層後差距縮小但仍存在；這是描述性調整，不是因果估計，也不能證明差距由運鏡、球網震動或其他物理機制造成。

## 7. 20-Epoch Counterfactual Sensitivity

### 設定

- Sampler global seed：20261009
- Counterfactual epochs：20
- Bootstrap replicates：5,000
- Bootstrap seed：20261016
- Bootstrap cluster：`source_group`
- Epoch 0 selected-center Make − Miss：-0.0116
- Epoch 0 source-cluster 95% CI：[-0.0187, -0.0041]

這是固定 global seed、改變 sampler epoch 的 deterministic sensitivity sweep，不是 multi-seed 實驗，也不把 epochs 視為獨立統計樣本。

| 指標 | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| Quality-aware Make−Miss selected-center | -0.0062 | -0.0117 | 0.0046 |
| Unrestricted Make−Miss selected-center | -0.0019 | -0.0084 | 0.0107 |
| Difference-in-differences | -0.0043 | -0.0061 | -0.0019 |

`±0.05 video length` 是本次整合前使用的實質偏移**警示尺度**，不是安全性、統計無偏性或下游準確率的數學證明。本次 epoch-0 CI、全部 20 個 quality-aware label shifts，以及全部 quality-induced difference-in-differences 都位於該尺度內；未觀察到超過此警示尺度的 label-dependent position shift。

## 8. 視覺前處理與增強政策

### Reference preprocessing

1. Center maximum 4:3 crop。
2. 使用 `INTER_AREA` resize 至高 192、寬 256。
3. 轉為 `float32 CTHW`，數值縮放到 `[0,1]`。
4. Metadata 中 `normalization = none`；是否再執行 mean/std normalization 取決於模型與 pretrained checkpoint。

`[0,1]` 是數值縮放，不等同於已完成 mean/std normalization。

### Augmentation policy

- Reference 作為第一版研究對照。
- Mild spatial 保留為獨立 A/B 實驗候選，仍需邊緣人物、球、籃框與人工時序檢查。
- 目前 photometric 參數因 clipping 個案而暫緩作為預設；gamma enhancement 的可見性收益尚未定案。
- Horizontal flip 目前不加入第一版預設。左右手、球場方向、比分文字等是需驗證的風險；這不是永久禁令，未來可另行評估 label invariance 與模型效益。

第一版不另行加入 candidate-based sample weighting，避免在 sampler 之外增加尚未校準的超參數；若未來要引入，應另立 validation protocol。

## 9. DataLoader Gate 與 Runtime Logging

### Gate 結果

| 檢查項目 | 警示尺度 | 結果 |
| --- | --- | --- |
| Epoch-0 source-cluster CI | 完整位於 ±0.05 video length 內 | PASS |
| 20 quality-aware epoch label shifts | 全部位於 ±0.05 video length 內 | PASS |
| Quality-induced counterfactual DiD | 全部位於 ±0.05 video length 內 | PASS |

Gate PASS 表示目前檢查結果未阻止 train-only、可觀測、可回退的 DataLoader 整合；它不代表沒有偏差、沒有資料洩漏或模型準確率一定提升。

### Runtime logging 建議

下一階段每個 sample/log 應保留：

- `video_id`
- `clip_start_frame` / `clip_end_frame`
- `normalized_start`
- `selected_center_normalized_position`（可選衍生欄位）
- `candidate_overlap_count`
- `fallback_flag`
- `label`
- `source_group`
- `shot_type`
- `fps`
- `epoch`
- `sampler_global_seed`
- `policy_version` / `config_hash` / `run_identity`

每個 epoch 應監控 label/source/shot 的位置分布。以上是實作指引，不是已凍結的 JSON Schema 或 API 契約；正式欄位型別與 required constraints 應在 DataLoader 實作時另行凍結。

## 10. 技術限制

1. Local normalization 可能漏掉持續性、整段式低品質。
2. Source-group holdout 尚未證實為 game-level holdout。
3. 跨 split near-duplicate scan 尚未完成。
4. 20-epoch sweep 不是 multi-seed 或獨立樣本推論。
5. Validation/test quality data 未納入本次 Candidate–Label Association Audit。
6. Candidate 尚未經足量人工標註確認為肉眼可見模糊。
7. Gate PASS 不代表準確率會提升，也不代表已排除資料偏差或洩漏。
8. Centered 3-second window 使用未來影格，因此本方法只支持離線分析，不支持即時或 pre-release 偵測宣稱。

## 11. 結論與下一階段

v3/v3.1 已建立 train 資料的 low-local-Laplacian candidate metadata。Candidate–Label Association Audit 顯示 candidate 與 label 存在描述性關聯；但本次對 Quality-Aware Sampler 選中窗口的稽核中，量得的 label-dependent selected-position shifts 均未超過預設的 ±0.05 video-length 警示尺度。

因此，現有結果不阻止下一階段進行 **train-only、可觀測、可回退** 的 DataLoader 整合。Candidate 仍不是 confirmed blur，實際模型效益必須由 validation 上的 A/B 實驗與人工標註校準確認；test 不參與政策選擇。

下一階段建議依序為：

1. 整合 train-only DataLoader 與逐樣本 runtime logging，保留原始連續 64-frame window。
2. 執行 reference 與 mild-spatial 的 validation A/B 實驗，固定模型設定與 seeds。
3. 建立 `z=-2.5` derived sensitivity 版本，比較 Candidate–Label 關聯與選窗穩定度；不得覆寫正式 v3.1 sidecar，也不得使用 test 選擇門檻。這是敏感度分析，不代表重校正式門檻。
4. 建立依 source/shot type/label/時間位置分層的小型人工標註集，校準 candidate 與可見模糊的對應；人工標註不是刪除資料授權。
5. 補做跨 split near-duplicate scan，並釐清 source group 與實際比賽場次的對應，評估 game-level holdout 可行性。

## 12. 稽核驗證與重現資訊

### 驗收結果

| 項目 | 結果 |
| --- | --- |
| Targeted audit tests | 12/12 PASS |
| Full data-quality tests | 95/95 PASS |
| Audit check-only | `CHECK_PASS` |
| `git diff --check` | PASS |
| Audit video opens/probes/decodes | 0/0/0 |
| Validation/test quality data analyzed | 0 |

### 設定檔 SHA-256

| 設定檔 | SHA-256 |
| --- | --- |
| `blur_train_v3.yaml` | `ef73aea0b9cdd6656d8d017a90552b8737207fd7ee58a925b7d0dca45e5f9e20` |
| `quality_aware_sampler_v1.yaml` | `86cd23900b3673def08c0b5d6654fe11c4576d3b65b1d777071bb1517c1f5d74` |
| `visual_preprocessing_pilot_v1.yaml` | `c10fe6cd5709ffe6b6d54fb193a2a491141c2e570e6d98d3605959b6e8f956e0` |

### 來源文件

- `docs/data_quality/blur_screening_method_v1.md`
- `docs/data_quality/blur_screening_pilot_v2.md`
- `docs/data_quality/blur_screening_method_v3.md`
- `docs/data_quality/blur_screening_publisher_v3_1.md`
- `docs/data_quality/quality_aware_clip_sampler_v1.md`
- `docs/data_quality/visual_preprocessing_pilot_v1.md`
- `docs/data_quality/visual_preprocessing_method_v1.md`
- `docs/data_quality/train_quality_associations_v1.md`
- `docs/data_quality/train_quality_associations_v1.json`
