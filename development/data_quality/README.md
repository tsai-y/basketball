# Basketball-51 資料品質技術入口

> - **給誰看：** 要重現、稽核、維護或整合資料品質元件的實作者。
> - **回答什麼：** configs、scripts、samplers、transforms、tests、sidecars、runbooks 與安全 CLI 在哪裡。
> - **不包含：** 完整研究結果、一般組員導覽或 A1/A2/B/C 技術細節。
> - **上一層入口：** [Development 規則](../README.md)。
> - **建議下一份：** 依工作選擇下表的 method 或 runbook；研究脈絡先讀[中文研究索引](../../docs/data_quality/README.md)。

這是資料品質實作的唯一技術入口。歷史檔名不代表目前執行版本；完整 train 操作以 [v3/v3.1 runbook](runbooks/blur_train_v3_tmux.md) 為準。

## 使用原則

- 所有命令從 repository root 執行，相對路徑也以 root 為基準。
- v1/v2 只供歷史重現與稽核；不要把它們當作新的完整掃描入口。
- v3/v3.1 scanner、verifier、publisher 與 full-train publication 已完成；歷史入口不表示需要重跑。
- Sidecar 是 train-only、tracked、machine-readable 介面，不得手工改值；validation/test quality mode disabled。
- `candidate` 不是確認模糊、排除名單或刪除授權；原始影片保持不變。
- `artifacts/` 與 `__pycache__/` 是 ignored 本機產物，不得加入 Git。

## 元件總覽

| 階段 | Method／report | Config | 主要程式 | 重現入口 |
| --- | --- | --- | --- | --- |
| v1 指標探索 | [method](../../docs/data_quality/blur_screening_method_v1.md)／[report](../../docs/data_quality/blur_screening_pilot_v1.md) | [blur_pilot_v1.yaml](configs/blur_pilot_v1.yaml) | [audit](scripts/audit_blur_pilot.py)、[verify](scripts/verify_blur_pilot.py) | method；歷史用途 |
| v2 指標校準 | [method](../../docs/data_quality/blur_screening_method_v2.md)／[report](../../docs/data_quality/blur_screening_pilot_v2.md) | [blur_pilot_v2.yaml](configs/blur_pilot_v2.yaml) | [audit](scripts/audit_blur_pilot_v2.py)、[analyze](scripts/analyze_blur_pilot_v2.py)、[verify](scripts/verify_blur_pilot_v2.py)、[publish](scripts/publish_blur_pilot_v2.py) | [歷史相容附錄](README_v2.md) |
| v3/v3.1 完整掃描 | [method](../../docs/data_quality/blur_screening_method_v3.md)／[publisher contract](../../docs/data_quality/blur_screening_publisher_v3_1.md) | [blur_train_v3.yaml](configs/blur_train_v3.yaml) | [scan](scripts/audit_blur_train_v3.py)、[verify](scripts/verify_blur_train_v3.py)、[publish](scripts/publish_blur_train_v3.py) | [唯一完整 runbook](runbooks/blur_train_v3_tmux.md) |
| Sampler v1 | [method](../../docs/data_quality/quality_aware_clip_sampler_v1.md)／[smoke](../../docs/data_quality/quality_aware_clip_sampler_smoke_v1.md) | [quality_aware_sampler_v1.yaml](configs/quality_aware_sampler_v1.yaml) | [sampler](samplers/quality_aware_clip_sampler.py)、[audit](scripts/audit_quality_aware_sampler_v1.py)、[publish smoke](scripts/publish_quality_aware_sampler_smoke_v1.py) | method 與 tests |
| Visual Preprocessing Pilot v1 | [method](../../docs/data_quality/visual_preprocessing_method_v1.md)／[report](../../docs/data_quality/visual_preprocessing_pilot_v1.md) | [config](configs/visual_preprocessing_pilot_v1.yaml) | [transforms](transforms/video_clip_transforms.py)、[runner](scripts/audit_visual_preprocessing_pilot_v1.py)、[verifier](scripts/verify_visual_preprocessing_pilot_v1.py) | method；Pilot 已完成 |

固定研究清單依階段保留在 [v1 Pilot manifest](pilot_manifest_v1.csv)、[v2 Pilot manifest](pilot_manifest_v2.csv)、[v2 source inventory](source_inventory_v2.csv)、[v3 smoke manifest](smoke_manifest_v3.csv)、[Sampler smoke manifest](smoke_manifest_quality_sampler_v1.csv) 與 [Visual Pilot manifest](manifests/visual_preprocessing_pilot_manifest_v1.csv)。它們是既有輸入或選樣證據，不由 README 維護內容。

`README_v2.md` 不是第二個現行入口。它保留 v2 當時的命令與交付語境，且既有 publisher 與 frozen 研究文件仍直接引用該路徑；在不修改 Python、frozen report 與 JSON 的前提下不能安全移除或改名。

## Sidecar

v3.1 到 Sampler v1 的精簡介面只有下列三份 tracked、train-only 檔案：

| 檔案 | 用途 |
| --- | --- |
| [blur_quality_sidecar_v3_1.json](sidecars/blur_quality_sidecar_v3_1.json) | policy、來源、hash 與 schema metadata |
| [blur_video_summary_v3_1.csv](sidecars/blur_video_summary_v3_1.csv) | 每影片 frame count、候選與 clean-window 摘要 |
| [blur_candidate_segments_v3_1.csv](sidecars/blur_candidate_segments_v3_1.csv) | 半開 frame／timestamp candidate segments |

[export_blur_quality_sidecar_v3_1.py](scripts/export_blur_quality_sidecar_v3_1.py) 從 verified v3.1 metadata 匯出或檢查 sidecar。validation/test 沒有 sidecar，不得以缺少 sidecar 為由建立 held-out 品質資料。

## 品質感知取樣

Sampler v1 從原影片選擇連續半開 `[start, end)` clip，不刪格、不拼接；優先 zero candidate overlap，無可行窗口時使用可追蹤的 minimum-overlap fallback。

- Package：[samplers/__init__.py](samplers/__init__.py)
- 實作：[quality_aware_clip_sampler.py](samplers/quality_aware_clip_sampler.py)
- 設定：[quality_aware_sampler_v1.yaml](configs/quality_aware_sampler_v1.yaml)
- Smoke manifest：[smoke_manifest_quality_sampler_v1.csv](smoke_manifest_quality_sampler_v1.csv)
- 契約與限制：[Quality-aware Clip Sampler v1](../../docs/data_quality/quality_aware_clip_sampler_v1.md)

## 視覺前處理／增強 Pilot v1

Pilot 使用固定 train-only manifest；正式 enhancement／augmentation 政策尚未定案。整合時保持 transform、normalization 與 metadata 責任分離。

- 固定名單：[visual_preprocessing_pilot_manifest_v1.csv](manifests/visual_preprocessing_pilot_manifest_v1.csv)
- Transform：[video_clip_transforms.py](transforms/video_clip_transforms.py)
- Runner／verifier：[audit](scripts/audit_visual_preprocessing_pilot_v1.py)、[verify](scripts/verify_visual_preprocessing_pilot_v1.py)
- 研究限制：[method](../../docs/data_quality/visual_preprocessing_method_v1.md)、[Pilot report](../../docs/data_quality/visual_preprocessing_pilot_v1.md)

## 測試

| 範圍 | 測試 |
| --- | --- |
| v1 | [test_blur_pilot.py](tests/test_blur_pilot.py) |
| v2 | [test_blur_pilot_v2.py](tests/test_blur_pilot_v2.py) |
| v3 scanner | [test_blur_train_v3.py](tests/test_blur_train_v3.py) |
| v3.1 publisher | [test_blur_train_publisher_v3.py](tests/test_blur_train_publisher_v3.py) |
| Sampler v1 | [test_quality_aware_clip_sampler_v1.py](tests/test_quality_aware_clip_sampler_v1.py) |
| Visual transforms | [test_video_clip_transforms_v1.py](tests/test_video_clip_transforms_v1.py) |

這些測試設計為不讀取資料集影片；本次文件整理不執行測試、掃描、verify、publish、訓練或評估。

## CLI 與重現方式

先用對應入口的 `--help` 確認參數；不要假設 `check`、`dry-run` 或 `verify` 一定不寫檔。完整 train 的 preflight、scan、resume、retry、rebuild、verify、publish 與輸出位置只依 [tmux runbook](runbooks/blur_train_v3_tmux.md)。v2 的原始命令只依[歷史相容附錄](README_v2.md)。

```bash
python3 -B development/data_quality/scripts/<entry>.py --help
python3 -B -m unittest discover -s development/data_quality/tests -p 'test_*.py'
```

執行前仍需確認既有環境、輸入、輸出與防護契約；上例不是啟動完整掃描或重跑歷史研究的指示。

## 本機 ignored runtime artifacts

`development/data_quality/artifacts/`、各 `__pycache__/`、逐幀資料、影像、NPZ、log、lock 與 publication 副本都是本機執行證據或 cache。它們不是缺少待補的 tracked 檔案，不得以 `git add -f` 納入版本控制，也不得在文件整理中移動或重建。
