# Basketball-51 自動模糊篩檢 Pilot v2

**COMPLETE — train-only 合成模糊校準、來源混雜分析與時間偵測驗證；沒有正式資料清理。**

開始時 HEAD `7325dab feat(data-quality): add train-only blur screening pilot`；`git status --short` 為空。Pilot v1 已提交。完成後初始 113 個 tracked files（含 v1 scripts/config/manifest/報告及 frozen evidence）全部 hash 不變。本輪全部交付為新增 v2 檔案，未 commit/push。

方法細節見 [v2 方法](blur_screening_method_v2.md)；執行入口見 [README_v2](../../development/data_quality/README_v2.md)。

## 選樣與資料範圍

固定 train：7,193 支、35 個 source groups；每來源 41–339 支。v1 有候選比例資料的 30 個來源可分為 high/middle/low；另 5 個 unobserved 只列庫存、不納入本次目的性樣本。seed=20261009，取 12 來源×4 支=48 支，各來源 Make/Miss 各2。數量無不足，未因解碼或分數更換任何影片。

high/middle 依 v1 正候選比例的中位數分界；low=0。各層 4 來源；high 強制納入 v184。low 預留兩個 23.976 FPS 提示來源與兩個 29.97 FPS 提示來源。来源與 subset 的 SHA-256 排序規則見方法文件。v1 每來源只有一支影片，因此分層不是品質真值。

| 來源 | Subset | v1 層 | v1 雙低比例 | Train 可用 | 選取 | 實際 FPS |
| --- | --- | --- | --- | --- | --- | --- |
| v115 | verification | high | 20.55% | 214 | 4 | 23.976 |
| v122 | calibration | middle | 2.74% | 230 | 4 | 23.976 |
| v131 | verification | low | 0.00% | 214 | 4 | 29.97 |
| v133 | calibration | high | 20.55% | 179 | 4 | 23.976 |
| v156 | calibration | middle | 12.33% | 230 | 4 | 23.976 |
| v160_v166 | verification | high | 28.08% | 339 | 4 | 23.976 |
| v161 | verification | middle | 4.11% | 204 | 4 | 23.976 |
| v167 | calibration | low | 0.00% | 218 | 4 | 23.976 |
| v169 | verification | low | 0.00% | 204 | 4 | 23.976 |
| v175 | calibration | low | 0.00% | 207 | 4 | 29.97 |
| v183 | verification | middle | 6.16% | 244 | 4 | 23.976 |
| v184 | calibration | high | 58.90% | 223 | 4 | 23.976 |

Calibration/verification 各24支、6來源、Make/Miss各12；來源完全不交集，兩者均20支23.976 FPS與4支29.97 FPS。v184 位於 calibration，因此本輪沒有對 v184 做獨立 verification 的泛化宣稱。來源代碼不等於已核實 game ID。

完整35來源庫存：[source_inventory_v2.csv](../../development/data_quality/source_inventory_v2.csv)。選定身份、subset、label、來源、FPS、frame count、duration：[pilot_manifest_v2.csv](../../development/data_quality/pilot_manifest_v2.csv)。

## 固定畫面與合成尺度

成功解碼7,280影格；每影片按有效範圍10/30/50/70/90%取樣，共240 anchors。保存零起始 index、decoder timestamp及index/FPS估計時間；本次全部通過CFR一致性檢查。沒有根據分數重新挑選畫面。

Gaussian sigma 0/0.5/1/1.5/2/3，核寬 `2*ceil(3*sigma)+1`，0完全不修改。Motion length 3/5/9/13 × 0/45/90/135度，中心對稱bilinear-splat線核、總和1。灰階float64、BORDER_REFLECT_101；計分定義沿用v1。每anchor共22版本，合計5,280筆，包含absolute、relative、log drop。全部比值有效，沒有零分母。

**sigma與length只是工程尺度，不是真實模糊嚴重度標準。** 靜態單張配對relative score需要原圖參考，不能直接用於未知真實影片；時間偵測使用觀測串流內部local z。

## 指標敏感度與互補性

| Subset | Family | 指標/組合 | 曲線數 | ρ中位數 | 嚴格下降 | 非單調數 |
| --- | --- | --- | --- | --- | --- | --- |
| calibration | gaussian | geometric_relative | 120 | -1.000 | 100.0% | 0 |
| calibration | gaussian | laplacian_variance | 120 | -1.000 | 100.0% | 0 |
| calibration | gaussian | tenengrad | 120 | -1.000 | 100.0% | 0 |
| calibration | motion | geometric_relative | 480 | -1.000 | 100.0% | 0 |
| calibration | motion | laplacian_variance | 480 | -1.000 | 100.0% | 0 |
| calibration | motion | tenengrad | 480 | -1.000 | 100.0% | 0 |
| verification | gaussian | geometric_relative | 120 | -1.000 | 100.0% | 0 |
| verification | gaussian | laplacian_variance | 120 | -1.000 | 100.0% | 0 |
| verification | gaussian | tenengrad | 120 | -1.000 | 100.0% | 0 |
| verification | motion | geometric_relative | 480 | -1.000 | 100.0% | 0 |
| verification | motion | laplacian_variance | 480 | -1.000 | 100.0% | 0 |
| verification | motion | tenengrad | 480 | -1.000 | 100.0% | 0 |

每subset有120張畫面；Gaussian每張一條6點曲線，motion每角度一條4點曲線，因此motion的480條不是480張獨立畫面。每條曲線相鄰步驟下降率也全部100%。幾何平均組合沒有提升已達100%的單調性。

以下為relative score中位數；完整每個severity/angle/source/FPS/label的P10/median/P90見本機 `analysis/sensitivity_distributions.csv`，JSON保留各transform整體與FPS/label分布。

| Subset | Family | 尺度 | Angle | Laplacian relative | Tenengrad relative |
| --- | --- | --- | --- | --- | --- |
| calibration | gaussian | 0.5 | — | 0.4611 | 0.7558 |
| calibration | gaussian | 1.5 | — | 0.0169 | 0.2045 |
| calibration | gaussian | 3 | — | 0.0012 | 0.0600 |
| calibration | motion | 5 | 0 | 0.2636 | 0.4910 |
| calibration | motion | 5 | 45 | 0.1097 | 0.3678 |
| calibration | motion | 5 | 90 | 0.2331 | 0.5072 |
| calibration | motion | 5 | 135 | 0.1089 | 0.3640 |
| calibration | motion | 13 | 0 | 0.1341 | 0.2742 |
| verification | gaussian | 0.5 | — | 0.4609 | 0.7600 |
| verification | gaussian | 1.5 | — | 0.0176 | 0.2114 |
| verification | gaussian | 3 | — | 0.0014 | 0.0656 |
| verification | motion | 5 | 0 | 0.2921 | 0.5555 |
| verification | motion | 5 | 45 | 0.1056 | 0.3734 |
| verification | motion | 5 | 90 | 0.2365 | 0.4700 |
| verification | motion | 5 | 135 | 0.1121 | 0.3726 |
| verification | motion | 13 | 0 | 0.1624 | 0.3390 |

Laplacian的配對衰減更大，但較大數值變化本身不等於實際辨識更好；時間驗證另提供判斷。斜角核的離散化不同，角度差異不能全部解讀為影像方向效應。

絕對P10對照只用calibration原始anchors擬合：laplacian_variance ≤ 1803.977383, tenengrad ≤ 27394.887857。verification原封套用；完整單指標、AND/OR旗標率見 `analysis/absolute_threshold_response.csv`，不是刪除門檻。

## 來源、FPS與label混雜

先按影片聚合5個anchor，再計算log(影片中位數)的單因子eta²。下表為描述性組間變異比例；來源/FPS/label的數字不能相加，也沒有因果或統計顯著性宣稱。

| Subset | 指標 | Source eta² | FPS eta² | Label eta² |
| --- | --- | --- | --- | --- |
| calibration | laplacian_variance | 0.760 | 0.317 | 0.038 |
| calibration | tenengrad | 0.763 | 0.194 | 0.001 |
| verification | laplacian_variance | 0.258 | 0.086 | 0.014 |
| verification | tenengrad | 0.673 | 0.030 | 0.008 |

兩個subset的source+label design rank皆7；加入FPS後8欄仍rank7。樣本中每來源FPS固定，因此無法分離FPS與來源效果。Label在每來源內平衡，但仍可能與拍攝情境、投籃類型相關；JSON保留來源內Make−Miss差，不把label差異歸因於模糊。

原始anchor分數按來源如下；同一來源仍有影片間與畫面間差異。

| Subset | Source | Anchors | Laplacian median | Tenengrad median |
| --- | --- | --- | --- | --- |
| calibration | v122 | 20 | 2603.53 | 32275.27 |
| calibration | v133 | 20 | 2534.44 | 32050.13 |
| calibration | v156 | 20 | 2928.52 | 40633.14 |
| calibration | v167 | 20 | 3623.10 | 41178.58 |
| calibration | v175 | 20 | 4116.68 | 41059.12 |
| calibration | v184 | 20 | 1963.35 | 28995.05 |
| verification | v115 | 20 | 2180.52 | 27043.91 |
| verification | v131 | 20 | 3031.69 | 40362.33 |
| verification | v160_v166 | 20 | 2539.86 | 30468.64 |
| verification | v161 | 20 | 2250.91 | 23961.14 |
| verification | v169 | 20 | 2836.80 | 47602.03 |
| verification | v183 | 20 | 2697.04 | 36559.97 |

配對relative/log drop控制同張原圖的基線，但不保證不同紋理、鏡頭尺度或方向具有相同響應。每transform的來源eta²保留在 `analysis/source_confounding.csv`。本輪沒有人工紋理/鏡頭尺度真值，不能將來源差異全部命名為場景或FPS的影響。

## 時間區段校準與門檻凍結

每影片66個單段注入案例，共3,168案例，另48條原始control。中心30/70%；要求duration=0.25/0.5/1秒，實際採ceil(duration×FPS)格；Gaussian sigma 0.5/1.5/3；motion length5/13×四角度。只处理記憶體frame stream，保存分數陣列，無重新編碼影片。

每條原始/注入串流分別重算log-score、中心3秒rolling median/MAD；73/91格，邊界截短，MAD尺度下限0.05，score log下限1e−12。沒有用乾淨原始串流作oracle。中心窗使用未來格，只驗證離線偵測。

每策略在calibration搜尋7個z×3個minimum duration，共84候選。先限制control旗標率≤5%，再最大化source-macro frame F1，依預定同分規則選解，寫出frozen thresholds後才載入verification時間分數。

| 策略 | z ≤ | 最短秒數 | Calibration F1 | Calibration control旗標率 |
| --- | --- | --- | --- | --- |
| both | -1.0 | 0.16 | 0.8904 | 1.74% |
| either | -3.0 | 0.0 | 0.8923 | 0.83% |
| laplacian | -2.0 | 0.16 | 0.9012 | 0.77% |
| tenengrad | -2.5 | 0.0 | 0.8752 | 1.03% |

每案例frame F1先平均到影片，再平均到來源；motion/ Gaussian案例數不同，overall反映預定grid權重。事件檢出定義為最大連續區段IoU≥0.5，起訖為半開index區間。

| Subset | 策略 | Frame F1 | Event recall | 注入外旗標率 | Control旗標率 |
| --- | --- | --- | --- | --- | --- |
| calibration | both | 0.8904 | 94.00% | 1.34% | 1.74% |
| calibration | either | 0.8923 | 91.35% | 0.75% | 0.83% |
| calibration | laplacian | 0.9012 | 92.74% | 0.69% | 0.77% |
| calibration | tenengrad | 0.8752 | 89.33% | 0.81% | 1.03% |
| verification | both | 0.8590 | 95.71% | 2.43% | 2.90% |
| verification | either | 0.8318 | 91.73% | 2.28% | 2.75% |
| verification | laplacian | 0.8617 | 94.82% | 2.14% | 2.64% |
| verification | tenengrad | 0.8108 | 88.32% | 2.10% | 2.26% |

原始control可能已有模糊；control及注入外旗標率不是真實模糊FPR。事件recall與frame F1回答不同問題，少量多餘區段會降低frame F1而不影響正確事件是否被檢出。

按family分開的frame F1（同一組凍結門檻）：

| Subset | 策略 | Gaussian F1 | Motion F1 |
| --- | --- | --- | --- |
| calibration | laplacian | 0.8996 | 0.9017 |
| calibration | tenengrad | 0.8339 | 0.8907 |
| calibration | both | 0.8804 | 0.8941 |
| calibration | either | 0.8783 | 0.8976 |
| verification | laplacian | 0.8478 | 0.8669 |
| verification | tenengrad | 0.7756 | 0.8240 |
| verification | both | 0.8404 | 0.8660 |
| verification | either | 0.8152 | 0.8381 |

Laplacian依要求duration分組：

| Subset | 要求秒數 | Frame F1 | Event recall | 匹配事件數 | 起/訖絕對誤差P90秒 |
| --- | --- | --- | --- | --- | --- |
| calibration | 0.25 | 0.9078 | 95.83% | 506 | 0.0000/0.0000 |
| calibration | 0.5 | 0.9027 | 93.18% | 492 | 0.0000/0.0000 |
| calibration | 1.0 | 0.8929 | 89.20% | 471 | 0.0000/0.0000 |
| verification | 0.25 | 0.8403 | 99.05% | 523 | 0.0000/0.0000 |
| verification | 0.5 | 0.8788 | 95.64% | 505 | 0.0000/0.0000 |
| verification | 1.0 | 0.8660 | 89.77% | 474 | 0.0000/0.0000 |

匹配事件多能定位人為瞬間邊界，但未檢出者不在誤差統計中；長1秒的區段事件recall下降，仍需研究背景變動、rolling baseline吸收與漸變模糊。不能由P90=0宣稱真實時間定位完美。各來源/FPS/label/angle/severity/position的結果保留在JSON與本機group summary。

兩單指標各自凍結門檻的事件互補性：

| Subset | Both detected | Laplacian only | Tenengrad only | Neither |
| --- | --- | --- | --- | --- |
| calibration | 1405 | 64 | 10 | 105 |
| verification | 1398 | 104 | 1 | 81 |

## 研究判斷

1. **Gaussian**：兩項都能感知；Laplacian配對衰減較大，Gaussian時間F1 calibration 0.8996、verification 0.8478，高於Tenengrad的0.8339、0.7756，可作下一輪主要候選。
2. **Motion**：本次四方向線核兩項皆單調；Laplacian的motion F1為0.9017、0.8669，Tenengrad為0.8907、0.8240。結論限於這套合成核，不是所有真實運動模糊的排名。
3. **組合是否更好**：AND/OR未提升overall F1。AND verification F1 0.8590略低於Laplacian 0.8617，但事件recall 95.71%略高於94.82%；差距小、僅6個verification來源，沒有統計優越性證據。
4. **是否保留兩項**：保留兩項作診斷與敏感度對照，下一輪可優先驗證Laplacian；不將AND訂為正式規則，也不因本次結果刪除Tenengrad。
5. **第三指標**：目前兩項明確能感知合成motion blur，本輪沒有加入第三指標。後續優先處理來源泛化、局部主體品質與真實模糊真值，再依具體失敗模式決定是否增加指標。

此pilot沒有真實模糊標籤，原圖可能已模糊；合成邊界突然、全畫面一致，與真實漸變/局部模糊不同。配對下降不等於正式可用性；局部normalization可能忽略整片持續低品質。沒有證明清理可提升Make/Miss分類，不能建立刪除門檻、排除名單或直接擴展完整train。

## 存取證據與完成驗證

```json
{
  "all_video_opens_read_only": true,
  "anchor_frames": 240,
  "decoded_frames": 7280,
  "encoded_videos": 0,
  "independent_detection_rows": 12864,
  "independent_segments": 16228,
  "independently_recomputed_threshold_candidates": 84,
  "independently_verified_normalized_streams": 3216,
  "selected_videos": 48,
  "source_disjoint": true,
  "status": "PASS",
  "synthetic_score_rows": 5280,
  "test_opens_probes_decodes": 0,
  "tests": "10 v2 + 6 v1 PASS (logs); no videos opened in guard tests",
  "thresholds_calibration_only": true,
  "train_decodes": 48,
  "train_guard_opens": 96,
  "train_probes": 48,
  "unchanged_preexisting_tracked_files": 113,
  "unique_video_ids_opened": 48,
  "validation_opens_probes_decodes": 0,
  "verification_thresholds_unchanged": true,
  "video_open_syscalls": 192
}
```

Syscall trace顯示48支選定train，各一次probe、一次decode；含FD reopen共192次唯讀開檔。validation/test open、probe、decode全部0。strace初次sandbox能力檢查被ptrace限制，正式經核准執行追蹤成功；未在失敗檢查中執行資料處理。

獨立驗證重算3,216條local z、84門檻候選、12,864策略案例與16,228區段；檢查CSV相對分數、Spearman、anchors、選樣、來源分隔、固定門檻。10個v2與6個v1測試通過；guard tests不開影片。JSON附逐artifact hash，可核對ignored產物。

## 產物

`development/data_quality/artifacts/blur_pilot_v2/`（ignored）：

- `source_inventory.csv`、`pilot_manifest_v2.csv`、`calibration_frames.csv`：來源庫存、選样與固定位置。
- `synthetic_scores.csv`：5,280筆靜態absolute/relative/log-drop分數。
- `streams/*.npz`、`temporal_scenarios.csv`：48份原始/注入分數與local z，3,216條串流metadata。無影像或影片。
- `analysis/`：敏感度曲線、來源/FPS/label分布、混雜統計、凍結門檻、calibration搜尋grid、時間偵測、區段與分組摘要。
- `runtime_logs/`：metadata、decode status、ledger、versions/build、syscall trace、測試與獨立驗證。

原始資料與Pilot v1未修改。未訓練、未載入checkpoint、未執行正式清理；Gate 1C及原D維持暫停。
