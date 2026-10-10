# Basketball-51 資料品質技術使用入口

本頁是 `development/data_quality/` 的技術導覽；研究脈絡、一般組員閱讀順序與完整中文產物表見[資料品質研究產物中文索引](../../docs/data_quality/README.md)。

目前完成狀態：

- 模糊候選探索 v1：**完成，只供歷史理解**。
- 模糊指標校準 v2：**完成，提供 v3 凍結參數**。
- 完整 train 品質掃描 v3/v3.1：**完成**。
- Quality-aware Clip Sampler v1：**完成**。
- 影像增強、I3D 特徵擷取、YOLO 特徵／ROI、多模型融合：**尚未開始**。

本輪只整理文件。下列入口不代表需要重跑：沒有執行影片解碼、掃描、訓練或評估，也沒有存取 validation/test 影片。A1/A2/B/C、設定、sidecar、研究數值與歷史 artifacts 均維持原狀。

## 技術使用原則

- 所有相對路徑都以 repository root `/home/tsai/projects/basketball` 為基準。
- v1／v2 是歷史研究與稽核入口，不是新一輪正式掃描入口。
- v3/v3.1 的 scan／resume／verify／publish 細節只以 [tmux runbook](runbooks/blur_train_v3_tmux.md) 與 [publisher contract](../../docs/data_quality/blur_screening_publisher_v3_1.md) 為準。
- Sidecar 是 train-only、tracked、machine-readable 介面，不得手工改值；validation/test quality mode disabled。
- `candidate` 不是確認模糊、排除名單或刪除授權；原始影片保持不變。
- `artifacts/` 與 `__pycache__/` 是 ignored 本機產物，不得加入 Git。

## 指標探索與校準

### v1：模糊候選探索

狀態為完成，只供歷史理解。v1 回答全畫面清晰度指標與候選分布問題，不能直接制定刪除門檻。

| 中文入口 | 英文檔名 | 技術用途 |
| --- | --- | --- |
| v1 方法 | [`blur_screening_method_v1.md`](../../docs/data_quality/blur_screening_method_v1.md) | 指標、取樣、防護與限制 |
| v1 報告／JSON | [`blur_screening_pilot_v1.md`](../../docs/data_quality/blur_screening_pilot_v1.md)／[`blur_screening_pilot_v1.json`](../../docs/data_quality/blur_screening_pilot_v1.json) | 人讀報告與結構化證據 |
| v1 設定 | [`configs/blur_pilot_v1.yaml`](configs/blur_pilot_v1.yaml) | 凍結的取樣、指標與門檻契約 |
| v1 樣本清單 | [`pilot_manifest_v1.csv`](pilot_manifest_v1.csv) | 30 支 train pilot 選樣 |
| v1 執行器 | [`scripts/audit_blur_pilot.py`](scripts/audit_blur_pilot.py) | 歷史 train-only pilot 重現 |
| v1 驗證器 | [`scripts/verify_blur_pilot.py`](scripts/verify_blur_pilot.py) | 僅以 derived CSV／JSON 檢查一致性 |

### v2：模糊指標校準

狀態為完成，提供 v3 凍結參數。v2 的 calibration／verification 都是 train 內部研究子集，不是資料集 validation/test split。

| 中文入口 | 英文檔名 | 技術用途 |
| --- | --- | --- |
| v2 方法 | [`blur_screening_method_v2.md`](../../docs/data_quality/blur_screening_method_v2.md) | 合成模糊、來源分離與門檻凍結 |
| v2 報告／JSON | [`blur_screening_pilot_v2.md`](../../docs/data_quality/blur_screening_pilot_v2.md)／[`blur_screening_pilot_v2.json`](../../docs/data_quality/blur_screening_pilot_v2.json) | 人讀報告與結構化證據 |
| v2 歷史重現 | [`README_v2.md`](README_v2.md) | 保存當時命令與交付語境 |
| v2 設定 | [`configs/blur_pilot_v2.yaml`](configs/blur_pilot_v2.yaml) | 凍結的合成與 temporal calibration 契約 |
| v2 樣本／來源 | [`pilot_manifest_v2.csv`](pilot_manifest_v2.csv)／[`source_inventory_v2.csv`](source_inventory_v2.csv) | 48 支 train 選樣與來源覆蓋 |
| v2 執行器 | [`scripts/audit_blur_pilot_v2.py`](scripts/audit_blur_pilot_v2.py) | paired synthetic calibration |
| v2 分析器 | [`scripts/analyze_blur_pilot_v2.py`](scripts/analyze_blur_pilot_v2.py) | derived scores 與 calibration-only fitting |
| v2 驗證器 | [`scripts/verify_blur_pilot_v2.py`](scripts/verify_blur_pilot_v2.py) | derived artifacts 與 syscall 證據 |
| v2 發布器 | [`scripts/publish_blur_pilot_v2.py`](scripts/publish_blur_pilot_v2.py) | 從已驗證產物建立新報告／manifest |

## 完整品質掃描

完整 train 品質掃描 v3/v3.1 已完成；scanner、verifier、publisher 與完整 train publication 都有既有產物。本區只提供維護入口，不要求重新執行。

| 中文入口 | 英文檔名 | 技術用途 |
| --- | --- | --- |
| v3 方法 | [`blur_screening_method_v3.md`](../../docs/data_quality/blur_screening_method_v3.md) | train-only scanner、immutable attempt、resume／retry 契約 |
| v3 smoke 報告／JSON | [`blur_screening_smoke_v3.md`](../../docs/data_quality/blur_screening_smoke_v3.md)／[`blur_screening_smoke_v3.json`](../../docs/data_quality/blur_screening_smoke_v3.json) | 固定 20 支工程驗證 |
| v3.1 publisher contract | [`blur_screening_publisher_v3_1.md`](../../docs/data_quality/blur_screening_publisher_v3_1.md) | 完整度狀態、schema 與 fail-closed 規則 |
| v3 設定 | [`configs/blur_train_v3.yaml`](configs/blur_train_v3.yaml) | v2 凍結規則與 evidence hashes |
| v3 smoke 清單 | [`smoke_manifest_v3.csv`](smoke_manifest_v3.csv) | 固定 20 支 train smoke 選樣 |
| v3/v3.1 掃描器 | [`scripts/audit_blur_train_v3.py`](scripts/audit_blur_train_v3.py) | scan、resume、retry、derived aggregate rebuild |
| v3 驗證器 | [`scripts/verify_blur_train_v3.py`](scripts/verify_blur_train_v3.py) | 不開影片驗證 smoke／full-train derived outputs |
| v3.1 發布器 | [`scripts/publish_blur_train_v3.py`](scripts/publish_blur_train_v3.py) | fail-closed 驗證及 Markdown／JSON publication |
| 主機操作手冊 | [`runbooks/blur_train_v3_tmux.md`](runbooks/blur_train_v3_tmux.md) | Ubuntu／tmux checklist 與正式命令 |

全量既有結果為 7,193/7,193 成功、1,105,146 frames、18,390 candidate frames、1,798 segments、1,698 affected、5,495 unaffected、0 failure；這些數值只引用既有報告，本輪沒有重算或修改。

## Sidecar

Sidecar 是 v3.1 完整掃描到 sampler v1 的唯一精簡介面。三份檔案都已 tracked，內容由匯出器產生，不應手工維護。

| 中文入口 | 英文檔名 | 技術用途 |
| --- | --- | --- |
| Sidecar 中繼資料 | [`sidecars/blur_quality_sidecar_v3_1.json`](sidecars/blur_quality_sidecar_v3_1.json) | policy、來源、hash、schema 與統計 |
| 每影片品質摘要 | [`sidecars/blur_video_summary_v3_1.csv`](sidecars/blur_video_summary_v3_1.csv) | train 每影片 frame count、候選與 clean-window 摘要 |
| 候選區段表 | [`sidecars/blur_candidate_segments_v3_1.csv`](sidecars/blur_candidate_segments_v3_1.csv) | 半開 frame／timestamp candidate segments |
| Sidecar 匯出器 | [`scripts/export_blur_quality_sidecar_v3_1.py`](scripts/export_blur_quality_sidecar_v3_1.py) | 從 verified v3.1 metadata 匯出／check compact sidecars |
| 使用契約 | [`quality_aware_clip_sampler_v1.md`](../../docs/data_quality/quality_aware_clip_sampler_v1.md) | schema、防護與 sampler 整合規則 |

validation/test 沒有品質 sidecar，不得以缺少 sidecar 為由讀取或建立 held-out 品質資料。

## 品質感知取樣

Quality-aware Clip Sampler v1 已完成。它從原影片選擇連續半開 `[start, end)` clip，不刪格、不拼接；優先 zero candidate overlap，無可行窗口時才使用可追蹤 minimum-overlap fallback。

| 中文入口 | 英文檔名 | 技術用途 |
| --- | --- | --- |
| 方法與契約 | [`quality_aware_clip_sampler_v1.md`](../../docs/data_quality/quality_aware_clip_sampler_v1.md) | 演算法、sidecar 契約與 train-only 邊界 |
| Sampler package | [`samplers/__init__.py`](samplers/__init__.py) | metadata-only package 入口 |
| Clip 選擇器 | [`samplers/quality_aware_clip_sampler.py`](samplers/quality_aware_clip_sampler.py) | deterministic continuous clip selection |
| Sampler 設定 | [`configs/quality_aware_sampler_v1.yaml`](configs/quality_aware_sampler_v1.yaml) | clip length、seed、zero-overlap、fallback |
| Sampler audit | [`scripts/audit_quality_aware_sampler_v1.py`](scripts/audit_quality_aware_sampler_v1.py) | 讀取 sidecar 的 train-only metadata audit |
| Smoke 發布器 | [`scripts/publish_quality_aware_sampler_smoke_v1.py`](scripts/publish_quality_aware_sampler_smoke_v1.py) | 固定 20 支 smoke 報告發布 |
| Smoke 清單 | [`smoke_manifest_quality_sampler_v1.csv`](smoke_manifest_quality_sampler_v1.csv) | 固定 metadata-only smoke 選樣 |
| Smoke 報告／JSON | [`quality_aware_clip_sampler_smoke_v1.md`](../../docs/data_quality/quality_aware_clip_sampler_smoke_v1.md)／[`quality_aware_clip_sampler_smoke_v1.json`](../../docs/data_quality/quality_aware_clip_sampler_smoke_v1.json) | 人讀與結構化 smoke evidence |

## 設定

| 階段 | 設定檔 | 狀態 | 維護規則 |
| --- | --- | --- | --- |
| v1 指標探索 | [`configs/blur_pilot_v1.yaml`](configs/blur_pilot_v1.yaml) | 已凍結／歷史 | 不為索引整理改值 |
| v2 指標校準 | [`configs/blur_pilot_v2.yaml`](configs/blur_pilot_v2.yaml) | 已凍結／歷史 | 提供 v3 參數依據 |
| v3/v3.1 完整掃描 | [`configs/blur_train_v3.yaml`](configs/blur_train_v3.yaml) | 已凍結 | 受 frozen evidence hashes 約束 |
| Sampler v1 | [`configs/quality_aware_sampler_v1.yaml`](configs/quality_aware_sampler_v1.yaml) | 已凍結 | validation quality mode disabled |

## 測試

這些測試的設計契約都是不讀取資料集影片；本輪文件整理沒有執行它們。

| 中文測試 | 英文檔名 | 覆蓋範圍 |
| --- | --- | --- |
| v1 安全性與決定性 | [`tests/test_blur_pilot.py`](tests/test_blur_pilot.py) | guard 與 deterministic selection |
| v2 數值與來源分離 | [`tests/test_blur_pilot_v2.py`](tests/test_blur_pilot_v2.py) | numeric PSF、source separation、temporal regressions |
| v3 掃描器 | [`tests/test_blur_train_v3.py`](tests/test_blur_train_v3.py) | 數學、access guard、atomic commit、recovery |
| v3.1 發布器 | [`tests/test_blur_train_publisher_v3.py`](tests/test_blur_train_publisher_v3.py) | metadata fixture、完整度、fail-closed |
| Sampler v1 | [`tests/test_quality_aware_clip_sampler_v1.py`](tests/test_quality_aware_clip_sampler_v1.py) | zero-overlap、fallback、seed、邊界 |

## Runbook

[`runbooks/blur_train_v3_tmux.md`](runbooks/blur_train_v3_tmux.md) 是唯一完整 train 主機操作手冊，包含：

- 執行前 frozen hashes、manifest 與 train-only checklist。
- dry-run、scan、resume、retry、rebuild、verify、publish 命令。
- ignored output、runtime log 與 publication 位置。
- COMPLETE／PARTIAL／COMPLETE_WITH_FAILURES／INVALID 的處理方式。

本輪不要依 runbook 啟動工作；它保留給未來需要稽核或在新目錄重現時使用。

## 本機 ignored artifacts

`.gitignore` 排除整個 `development/data_quality/artifacts/`，也排除各 Python `__pycache__/`。以下為 2026-10-10 的目錄級盤點；完整說明與「是否可提交 Git」欄位見[中文主索引](../../docs/data_quality/README.md#本機-ignored-runtime-artifacts)。

| 目錄 | 階段 | 主要內容 | Git |
| --- | --- | --- | --- |
| `artifacts/blur_pilot_v1/` | v1 | frame scores、segments、contact sheets、runtime logs（53 檔） | ignored，不可提交 |
| `artifacts/blur_pilot_v2/` | v2 | streams、analysis、runtime logs（78 檔） | ignored，不可提交 |
| `artifacts/blur_train_v3_smoke/` | v3 smoke | 20 支 attempts、ledger、aggregates（70 檔） | ignored，不可提交 |
| `artifacts/blur_train_v3_smoke_preflight_failed/` | v3 fixture | fail-closed preflight（2 檔） | ignored，不可提交 |
| `artifacts/blur_train_v3_smoke_recovery/` | v3 fixture | recovery／retry（75 檔） | ignored，不可提交 |
| `artifacts/blur_train_v3_smoke_repeat/` | v3 fixture | repeat／determinism（65 檔） | ignored，不可提交 |
| `artifacts/blur_train_v3_validation/` | v3 工程驗證 | logs、JSON、fixture（26 檔）；不是 dataset validation split | ignored，不可提交 |
| `artifacts/blur_train_v3_full_train/` | v3/v3.1 full train | 7,193 組 attempts 與 aggregate（21,586 檔） | ignored，大型，不可提交 |
| `artifacts/blur_train_v3_full_train_runtime/` | v3/v3.1 full train | scan／verify／publish logs（12 檔） | ignored，不可提交 |
| `artifacts/blur_train_v3_full_train_publication/` | v3.1 | Markdown／JSON publication（2 檔） | ignored，不可提交 |
| `artifacts/blur_train_v3_publisher_validation/` | v3.1 工程驗證 | test／syscall logs（2 檔）；不是 dataset validation split | ignored，不可提交 |
| `artifacts/quality_aware_sampler_v1/` | Sampler v1 | selections、summary、timelines、log（5 檔） | ignored，不可提交 |
| `samplers/__pycache__/` | Sampler v1 | Python bytecode（1 檔） | ignored，不可提交 |
| `scripts/__pycache__/` | 多階段 | Python bytecode（7 檔） | ignored，不可提交 |
| `tests/__pycache__/` | 多階段 | Python bytecode（2 檔） | ignored，不可提交 |

Ignored artifacts 是本機執行證據，不是缺少待補的 tracked 檔案；不得用 `git add -f` 納入版本控制。
