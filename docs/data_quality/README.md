# Basketball-51 資料品質研究產物中文索引

本頁是一般組員理解 Basketball-51 資料品質研究的中文主索引；實際程式入口集中在[技術使用入口](../../development/data_quality/README.md)。本索引盤點 `docs/data_quality/` 與 `development/data_quality/` 內全部 48 份 tracked 文件、JSON、CSV、YAML、Python 程式與測試，並分組記錄本機 ignored runtime artifacts。核心英文檔名與程式路徑維持不變。

本輪只整理文件與導覽，沒有修改演算法、設定、sidecar 數值、研究結果或歷史紀錄；沒有執行影片解碼、掃描、訓練或評估，也沒有存取 validation/test 影片或修改 A1/A2/B/C frozen experiments。

## 組員最短閱讀路徑

1. [Repository README](../../README.md)：先理解研究範圍、frozen baseline 與 repository 責任。
2. [目前研究方向](../current_research_direction.md)：確認當前優先順序與暫停項目。
3. [本中文主索引](README.md)：找到資料品質各階段的文件、資料與技術入口。
4. [Quality-aware Clip Sampler v1](quality_aware_clip_sampler_v1.md)：理解已完成品質結果如何接到 train-only 連續 clip 選擇。

一般組員讀完以上四份即可掌握現況；需要重現、稽核或維護程式時，再進入[技術使用入口](../../development/data_quality/README.md)。

## 目前階段狀態

| 中文名稱 | 英文版本 | 所屬階段 | 目前狀態 | 說明與入口 |
| --- | --- | --- | --- | --- |
| 模糊候選探索 | Quality Screening v1 | 指標探索 | **完成，只供歷史理解** | 小樣本探索不能當作刪除規則；見 [v1 方法](blur_screening_method_v1.md) 與 [v1 報告](blur_screening_pilot_v1.md) |
| 模糊指標校準 | Quality Screening v2 | 指標校準 | **完成，提供 v3 凍結參數** | 合成模糊與時間規則校準；見 [v2 方法](blur_screening_method_v2.md) 與 [v2 報告](blur_screening_pilot_v2.md) |
| 完整 train 品質掃描 | Quality Screening v3/v3.1 | 完整品質掃描 | **完成** | 全量 scan、verify、publication 均完成；見 [v3 方法](blur_screening_method_v3.md) 與 [v3.1 publisher contract](blur_screening_publisher_v3_1.md) |
| 品質感知連續片段取樣 | Quality-aware Clip Sampler v1 | 品質感知取樣 | **完成** | 以 tracked sidecar 選擇 train clips；見 [sampler 方法](quality_aware_clip_sampler_v1.md) |
| 影像增強 | Image Enhancement | 後續研究 | **尚未開始** | 未建立或選定增強策略 |
| I3D 特徵擷取 | I3D Feature Extraction | 後續研究 | **尚未開始** | 既有 Gate 1A/1B 不代表已啟動特徵擷取 |
| YOLO 特徵／ROI | YOLO Features / ROI | 後續研究 | **尚未開始** | 未建立特徵或 ROI pipeline |
| 多模型融合 | Multi-model Fusion | 後續研究 | **尚未開始** | 未決定模型組合或融合方式 |

`candidate` 只代表凍結規則標出的候選，不等於確認模糊、排除名單或 cleaned dataset；未標記也不等於已確認清晰。

## 欄位判讀

- 「一般組員閱讀」：**必讀**是最短閱讀路徑，**選讀**依工作需要，**否**主要供程式或維護者使用。
- 「程式讀取資料」：該檔案本身是否為 YAML／JSON／CSV 等 machine-readable input 或 evidence；Python 程式與 Markdown 文件標為「否」。
- 「可提交 Git」：「是」表示目前為 tracked 小型研究產物；「否」表示 `.gitignore` 排除的本機 runtime artifact 或 cache，不得加入 Git。
- 歷史報告保留當時交付語境；當前狀態以本頁與[技術使用入口](../../development/data_quality/README.md)為準。

## 研究文件與報告

| 中文名稱 | 英文版本／檔名 | 所屬階段 | 用途 | 完成狀態 | 一般組員閱讀 | 程式讀取資料 | 可提交 Git | 相關入口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 資料品質中文主索引 | [`README.md`](README.md) | 全階段 | 中文導覽、狀態與完整產物盤點 | 現行 | **必讀** | 否 | 是 | [技術入口](../../development/data_quality/README.md) |
| 模糊篩檢方法 v1 | [`blur_screening_method_v1.md`](blur_screening_method_v1.md) | v1 指標探索 | 指標、取樣、防護與限制 | 完成／歷史 | 選讀 | 否 | 是 | [v1 報告](blur_screening_pilot_v1.md) |
| 模糊探索報告 v1 | [`blur_screening_pilot_v1.md`](blur_screening_pilot_v1.md) | v1 指標探索 | 30 支 train pilot 結果與限制 | 完成／歷史 | 選讀 | 否 | 是 | [v1 JSON](blur_screening_pilot_v1.json) |
| 模糊探索結構化報告 v1 | [`blur_screening_pilot_v1.json`](blur_screening_pilot_v1.json) | v1 指標探索 | 報告、驗證與存取證據 | 完成／歷史 | 否 | **是** | 是 | [v1 方法](blur_screening_method_v1.md) |
| 模糊校準方法 v2 | [`blur_screening_method_v2.md`](blur_screening_method_v2.md) | v2 指標校準 | 合成模糊、來源分離與門檻凍結方法 | 完成／歷史 | 選讀 | 否 | 是 | [v2 報告](blur_screening_pilot_v2.md) |
| 模糊校準報告 v2 | [`blur_screening_pilot_v2.md`](blur_screening_pilot_v2.md) | v2 指標校準 | 校準、verification 與 v3 凍結規則依據 | 完成／歷史 | 選讀 | 否 | 是 | [v2 JSON](blur_screening_pilot_v2.json) |
| 模糊校準結構化報告 v2 | [`blur_screening_pilot_v2.json`](blur_screening_pilot_v2.json) | v2 指標校準 | 分析、門檻與結論的結構化證據 | 完成／歷史 | 否 | **是** | 是 | [v2 方法](blur_screening_method_v2.md) |
| 完整掃描方法 v3 | [`blur_screening_method_v3.md`](blur_screening_method_v3.md) | v3 完整掃描 | train-only scanner、恢復與輸出契約 | 完成 | 選讀 | 否 | 是 | [v3 runbook](../../development/data_quality/runbooks/blur_train_v3_tmux.md) |
| 掃描 smoke 報告 v3 | [`blur_screening_smoke_v3.md`](blur_screening_smoke_v3.md) | v3 工程驗證 | 固定 20 支 smoke 與限制 | 完成 | 選讀 | 否 | 是 | [v3 smoke JSON](blur_screening_smoke_v3.json) |
| 掃描 smoke 結構化報告 v3 | [`blur_screening_smoke_v3.json`](blur_screening_smoke_v3.json) | v3 工程驗證 | smoke aggregate 與驗證證據 | 完成 | 否 | **是** | 是 | [v3 方法](blur_screening_method_v3.md) |
| 完整掃描發布契約 v3.1 | [`blur_screening_publisher_v3_1.md`](blur_screening_publisher_v3_1.md) | v3.1 發布 | COMPLETE／PARTIAL／INVALID 與 fail-closed 行為 | 完成 | 選讀 | 否 | 是 | [v3 runbook](../../development/data_quality/runbooks/blur_train_v3_tmux.md) |
| 品質感知取樣方法 v1 | [`quality_aware_clip_sampler_v1.md`](quality_aware_clip_sampler_v1.md) | Sampler v1 | sidecar 契約、零 overlap 與 fallback | 完成 | **必讀** | 否 | 是 | [技術入口](../../development/data_quality/README.md#品質感知取樣) |
| 品質感知取樣 smoke 報告 v1 | [`quality_aware_clip_sampler_smoke_v1.md`](quality_aware_clip_sampler_smoke_v1.md) | Sampler v1 工程驗證 | 固定 20 支 metadata-only smoke | 完成 | 選讀 | 否 | 是 | [smoke JSON](quality_aware_clip_sampler_smoke_v1.json) |
| 品質感知取樣 smoke JSON v1 | [`quality_aware_clip_sampler_smoke_v1.json`](quality_aware_clip_sampler_smoke_v1.json) | Sampler v1 工程驗證 | clip selections 與檢查結果 | 完成 | 否 | **是** | 是 | [sampler 方法](quality_aware_clip_sampler_v1.md) |

## 技術文件、設定與小型資料

| 中文名稱 | 英文版本／檔名 | 所屬階段 | 用途 | 完成狀態 | 一般組員閱讀 | 程式讀取資料 | 可提交 Git | 相關入口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 技術使用總入口 | [`development/data_quality/README.md`](../../development/data_quality/README.md) | 全階段 | 依工作類型找程式、資料與測試 | 現行 | 選讀 | 否 | 是 | [本主索引](README.md) |
| v2 歷史重現說明 | [`README_v2.md`](../../development/data_quality/README_v2.md) | v2 指標校準 | 保存 v2 當時命令與交付語境 | 完成／歷史 | 否 | 否 | 是 | [v2 報告](blur_screening_pilot_v2.md) |
| 完整掃描 tmux 操作手冊 | [`blur_train_v3_tmux.md`](../../development/data_quality/runbooks/blur_train_v3_tmux.md) | v3/v3.1 完整掃描 | 主機 checklist、scan／resume／verify／publish | 完成 | 選讀 | 否 | 是 | [v3 方法](blur_screening_method_v3.md) |
| v1 探索設定 | [`blur_pilot_v1.yaml`](../../development/data_quality/configs/blur_pilot_v1.yaml) | v1 指標探索 | 取樣、指標、門檻與輸出契約 | 已凍結 | 否 | **是** | 是 | [v1 方法](blur_screening_method_v1.md) |
| v2 校準設定 | [`blur_pilot_v2.yaml`](../../development/data_quality/configs/blur_pilot_v2.yaml) | v2 指標校準 | 合成尺度、時間搜尋與來源分離設定 | 已凍結 | 否 | **是** | 是 | [v2 方法](blur_screening_method_v2.md) |
| v3 掃描設定 | [`blur_train_v3.yaml`](../../development/data_quality/configs/blur_train_v3.yaml) | v3/v3.1 完整掃描 | 完整掃描規則與 frozen evidence hashes | 已凍結 | 否 | **是** | 是 | [v3 方法](blur_screening_method_v3.md) |
| 品質感知取樣設定 | [`quality_aware_sampler_v1.yaml`](../../development/data_quality/configs/quality_aware_sampler_v1.yaml) | Sampler v1 | clip 長度、zero-overlap、fallback 與 seed | 已凍結 | 否 | **是** | 是 | [sampler 方法](quality_aware_clip_sampler_v1.md) |
| v1 pilot 清單 | [`pilot_manifest_v1.csv`](../../development/data_quality/pilot_manifest_v1.csv) | v1 指標探索 | 30 支 train 樣本與選樣證據 | 完成／歷史 | 否 | **是** | 是 | [v1 報告](blur_screening_pilot_v1.md) |
| v2 pilot 清單 | [`pilot_manifest_v2.csv`](../../development/data_quality/pilot_manifest_v2.csv) | v2 指標校準 | 48 支 train calibration／verification 樣本 | 完成／歷史 | 否 | **是** | 是 | [v2 報告](blur_screening_pilot_v2.md) |
| v2 來源庫存 | [`source_inventory_v2.csv`](../../development/data_quality/source_inventory_v2.csv) | v2 指標校準 | 來源覆蓋與 v1 band 資訊 | 完成／歷史 | 否 | **是** | 是 | [v2 方法](blur_screening_method_v2.md) |
| v3 smoke 清單 | [`smoke_manifest_v3.csv`](../../development/data_quality/smoke_manifest_v3.csv) | v3 工程驗證 | 固定 20 支 train scanner smoke 選樣 | 完成 | 否 | **是** | 是 | [v3 smoke 報告](blur_screening_smoke_v3.md) |
| Sampler smoke 清單 | [`smoke_manifest_quality_sampler_v1.csv`](../../development/data_quality/smoke_manifest_quality_sampler_v1.csv) | Sampler v1 工程驗證 | 固定 20 支 metadata-only smoke 選樣 | 完成 | 否 | **是** | 是 | [sampler smoke](quality_aware_clip_sampler_smoke_v1.md) |

## Sidecar

這三份 sidecar 是完整 v3.1 結果匯出的精簡、tracked、train-only machine-readable 介面；不得手工改值。validation/test 沒有 sidecar，quality mode 維持停用。

| 中文名稱 | 英文版本／檔名 | 所屬階段 | 用途 | 完成狀態 | 一般組員閱讀 | 程式讀取資料 | 可提交 Git | 相關入口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 品質 sidecar 中繼資料 | [`blur_quality_sidecar_v3_1.json`](../../development/data_quality/sidecars/blur_quality_sidecar_v3_1.json) | v3.1 → Sampler v1 | policy、來源、hash、schema 與統計 | 完成／現行 | 否 | **是** | 是 | [sampler 方法](quality_aware_clip_sampler_v1.md) |
| 每影片品質摘要 | [`blur_video_summary_v3_1.csv`](../../development/data_quality/sidecars/blur_video_summary_v3_1.csv) | v3.1 → Sampler v1 | 7,193 支 train 的 frame count、候選與 clean-window 摘要 | 完成／現行 | 否 | **是** | 是 | [Sidecar 技術入口](../../development/data_quality/README.md#sidecar) |
| 候選區段表 | [`blur_candidate_segments_v3_1.csv`](../../development/data_quality/sidecars/blur_candidate_segments_v3_1.csv) | v3.1 → Sampler v1 | 半開 frame／timestamp candidate segments | 完成／現行 | 否 | **是** | 是 | [sampler 方法](quality_aware_clip_sampler_v1.md) |

## 程式入口

| 中文名稱 | 英文版本／檔名 | 所屬階段 | 用途 | 完成狀態 | 一般組員閱讀 | 程式讀取資料 | 可提交 Git | 相關入口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 品質取樣套件初始化 | [`samplers/__init__.py`](../../development/data_quality/samplers/__init__.py) | Sampler v1 | 宣告 metadata-only sampling package | 完成 | 否 | 否 | 是 | [取樣技術入口](../../development/data_quality/README.md#品質感知取樣) |
| 品質感知 clip 選擇器 | [`quality_aware_clip_sampler.py`](../../development/data_quality/samplers/quality_aware_clip_sampler.py) | Sampler v1 | 依 candidate segments 決定連續 clip start | 完成 | 否 | 否 | 是 | [sampler 方法](quality_aware_clip_sampler_v1.md) |
| v1 探索執行器 | [`audit_blur_pilot.py`](../../development/data_quality/scripts/audit_blur_pilot.py) | v1 指標探索 | train-only 影片品質 pilot；歷史重現用 | 完成／歷史 | 否 | 否 | 是 | [v1 方法](blur_screening_method_v1.md) |
| v1 derived artifact 驗證器 | [`verify_blur_pilot.py`](../../development/data_quality/scripts/verify_blur_pilot.py) | v1 指標探索 | 只用 CSV／JSON 重算一致性 | 完成／歷史 | 否 | 否 | 是 | [v1 報告](blur_screening_pilot_v1.md) |
| v2 校準執行器 | [`audit_blur_pilot_v2.py`](../../development/data_quality/scripts/audit_blur_pilot_v2.py) | v2 指標校準 | 產生 paired synthetic calibration scores | 完成／歷史 | 否 | 否 | 是 | [v2 方法](blur_screening_method_v2.md) |
| v2 分析器 | [`analyze_blur_pilot_v2.py`](../../development/data_quality/scripts/analyze_blur_pilot_v2.py) | v2 指標校準 | 分析 derived scores 並限制 calibration fitting | 完成／歷史 | 否 | 否 | 是 | [v2 報告](blur_screening_pilot_v2.md) |
| v2 驗證器 | [`verify_blur_pilot_v2.py`](../../development/data_quality/scripts/verify_blur_pilot_v2.py) | v2 指標校準 | 檢查 derived artifacts 與 syscall 證據 | 完成／歷史 | 否 | 否 | 是 | [v2 報告](blur_screening_pilot_v2.md) |
| v2 發布器 | [`publish_blur_pilot_v2.py`](../../development/data_quality/scripts/publish_blur_pilot_v2.py) | v2 指標校準 | 從已驗證產物建立新報告／manifest | 完成／歷史 | 否 | 否 | 是 | [v2 歷史重現](../../development/data_quality/README_v2.md) |
| v3/v3.1 掃描器 | [`audit_blur_train_v3.py`](../../development/data_quality/scripts/audit_blur_train_v3.py) | v3/v3.1 完整掃描 | train-only scan、resume、retry、aggregate rebuild | 完成 | 否 | 否 | 是 | [v3 runbook](../../development/data_quality/runbooks/blur_train_v3_tmux.md) |
| v3 derived artifact 驗證器 | [`verify_blur_train_v3.py`](../../development/data_quality/scripts/verify_blur_train_v3.py) | v3/v3.1 完整掃描 | 不開影片驗證 smoke／full-train 輸出 | 完成 | 否 | 否 | 是 | [publisher contract](blur_screening_publisher_v3_1.md) |
| v3.1 發布器 | [`publish_blur_train_v3.py`](../../development/data_quality/scripts/publish_blur_train_v3.py) | v3.1 發布 | fail-closed 驗證並產生 Markdown／JSON | 完成 | 否 | 否 | 是 | [publisher contract](blur_screening_publisher_v3_1.md) |
| v3.1 sidecar 匯出器 | [`export_blur_quality_sidecar_v3_1.py`](../../development/data_quality/scripts/export_blur_quality_sidecar_v3_1.py) | v3.1 → Sampler v1 | 從 verified metadata 匯出 compact sidecars | 完成 | 否 | 否 | 是 | [Sidecar 技術入口](../../development/data_quality/README.md#sidecar) |
| 品質感知取樣稽核器 | [`audit_quality_aware_sampler_v1.py`](../../development/data_quality/scripts/audit_quality_aware_sampler_v1.py) | Sampler v1 | 讀取 sidecar 執行 train-only metadata audit | 完成 | 否 | 否 | 是 | [sampler 方法](quality_aware_clip_sampler_v1.md) |
| 品質感知 smoke 發布器 | [`publish_quality_aware_sampler_smoke_v1.py`](../../development/data_quality/scripts/publish_quality_aware_sampler_smoke_v1.py) | Sampler v1 | 發布固定 20 支 metadata-only smoke 報告 | 完成 | 否 | 否 | 是 | [sampler smoke](quality_aware_clip_sampler_smoke_v1.md) |

## 測試

| 中文名稱 | 英文版本／檔名 | 所屬階段 | 用途 | 完成狀態 | 一般組員閱讀 | 程式讀取資料 | 可提交 Git | 相關入口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 安全性與決定性測試 | [`test_blur_pilot.py`](../../development/data_quality/tests/test_blur_pilot.py) | v1 指標探索 | 不開資料集影片的 guard／determinism 測試 | 完成 | 否 | 否 | 是 | [測試入口](../../development/data_quality/README.md#測試) |
| v2 數值與來源分離測試 | [`test_blur_pilot_v2.py`](../../development/data_quality/tests/test_blur_pilot_v2.py) | v2 指標校準 | 非真實影片的 PSF、guard、temporal regression | 完成 | 否 | 否 | 是 | [測試入口](../../development/data_quality/README.md#測試) |
| v3 掃描器測試 | [`test_blur_train_v3.py`](../../development/data_quality/tests/test_blur_train_v3.py) | v3/v3.1 完整掃描 | 數學、存取防護、atomic commit、recovery | 完成 | 否 | 否 | 是 | [測試入口](../../development/data_quality/README.md#測試) |
| v3.1 發布器測試 | [`test_blur_train_publisher_v3.py`](../../development/data_quality/tests/test_blur_train_publisher_v3.py) | v3.1 發布 | metadata fixture 的完整度與 fail-closed | 完成 | 否 | 否 | 是 | [publisher contract](blur_screening_publisher_v3_1.md) |
| 品質感知取樣測試 | [`test_quality_aware_clip_sampler_v1.py`](../../development/data_quality/tests/test_quality_aware_clip_sampler_v1.py) | Sampler v1 | zero-overlap、fallback、seed 與邊界 | 完成 | 否 | 否 | 是 | [sampler 方法](quality_aware_clip_sampler_v1.md) |

## 本機 ignored runtime artifacts

以下是 2026-10-10 的目錄級盤點。它們受 `.gitignore` 保護，clone 不會取得；檔案數只用於確認盤點範圍，不是研究結果。不要將大型逐幀資料、NPZ、PNG、log、lock、fixture、publication 副本或 Python cache 加入 Git。

| 中文名稱 | 英文版本／目錄 | 所屬階段 | 用途與主要內容 | 完成狀態 | 一般組員閱讀 | 程式讀取資料 | 可提交 Git | 相關入口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 本機研究產物 | `artifacts/blur_pilot_v1/`（53 檔） | v1 指標探索 | frame scores、segments、contact sheets、logs、報告副本 | 完成／歷史 | 否 | 是 | **否** | [v1 報告](blur_screening_pilot_v1.md) |
| v2 本機研究產物 | `artifacts/blur_pilot_v2/`（78 檔） | v2 指標校準 | streams、analysis CSV／JSON、logs、報告輸入 | 完成／歷史 | 否 | 是 | **否** | [v2 報告](blur_screening_pilot_v2.md) |
| v3 smoke 主產物 | `artifacts/blur_train_v3_smoke/`（70 檔） | v3 工程驗證 | 20 支 attempts、frames、summary、ledger、aggregates | 完成 | 否 | 是 | **否** | [v3 smoke](blur_screening_smoke_v3.md) |
| v3 smoke preflight 失敗 fixture | `artifacts/blur_train_v3_smoke_preflight_failed/`（2 檔） | v3 工程驗證 | fail-closed preflight 歷史證據 | 完成／歷史 | 否 | 是 | **否** | [v3 方法](blur_screening_method_v3.md) |
| v3 smoke recovery fixture | `artifacts/blur_train_v3_smoke_recovery/`（75 檔） | v3 工程驗證 | partial／failure／retry／recovery 產物 | 完成／歷史 | 否 | 是 | **否** | [publisher contract](blur_screening_publisher_v3_1.md) |
| v3 smoke repeat fixture | `artifacts/blur_train_v3_smoke_repeat/`（65 檔） | v3 工程驗證 | 重跑與決定性驗證產物 | 完成／歷史 | 否 | 是 | **否** | [v3 smoke](blur_screening_smoke_v3.md) |
| v3 驗證工作目錄 | `artifacts/blur_train_v3_validation/`（26 檔） | v3 工程驗證 | tests、syscall logs、驗證 JSON、recovery fixture；不是 dataset validation split | 完成／歷史 | 否 | 是 | **否** | [v3 方法](blur_screening_method_v3.md) |
| 完整 train 掃描產物 | `artifacts/blur_train_v3_full_train/`（21,586 檔） | v3/v3.1 完整掃描 | 7,193 組 immutable frames／summary／completion、ledgers、aggregates | 完成 | 否 | 是 | **否** | [v3 runbook](../../development/data_quality/runbooks/blur_train_v3_tmux.md) |
| 完整掃描 runtime logs | `artifacts/blur_train_v3_full_train_runtime/`（12 檔） | v3/v3.1 完整掃描 | dry-run、scan、verify、publish、hash、time、exit logs | 完成 | 否 | 是 | **否** | [v3 runbook](../../development/data_quality/runbooks/blur_train_v3_tmux.md) |
| 完整掃描 publication | `artifacts/blur_train_v3_full_train_publication/`（2 檔） | v3.1 發布 | 同源完整 train Markdown／JSON publication | 完成 | 否 | 是 | **否** | [publisher contract](blur_screening_publisher_v3_1.md) |
| v3.1 publisher 驗證產物 | `artifacts/blur_train_v3_publisher_validation/`（2 檔） | v3.1 發布 | tests 與 syscall log；不是 dataset validation split | 完成／歷史 | 否 | 是 | **否** | [publisher contract](blur_screening_publisher_v3_1.md) |
| 品質感知取樣 runtime 產物 | `artifacts/quality_aware_sampler_v1/`（5 檔） | Sampler v1 | selected manifest、selections、summary、timelines、syscall log | 完成 | 否 | 是 | **否** | [sampler smoke](quality_aware_clip_sampler_smoke_v1.md) |
| Sampler Python cache | `samplers/__pycache__/`（1 檔） | Sampler v1 | Python bytecode cache | 可再生 | 否 | 否 | **否** | [取樣技術入口](../../development/data_quality/README.md#品質感知取樣) |
| Script Python cache | `scripts/__pycache__/`（7 檔） | 多階段 | Python bytecode cache | 可再生 | 否 | 否 | **否** | [技術入口](../../development/data_quality/README.md) |
| Test Python cache | `tests/__pycache__/`（2 檔） | 多階段 | Python bytecode cache | 可再生 | 否 | 否 | **否** | [測試入口](../../development/data_quality/README.md#測試) |

## 使用邊界

- v1／v2 僅供歷史理解與稽核；不得以其探索分數直接刪除資料。
- v3/v3.1 與 sampler v1 均已完成；本輪不要求重跑任何 scan、verify、publish 或 test 命令。
- 原始影片、固定 manifest／split、A1/A2/B/C、歷史 runs、設定、sidecar 與報告數值均不可由索引整理回寫。
- validation/test 影片不得存取、probe 或 decode；表中名稱含 `validation` 的 ignored 目錄是程式驗證工作目錄，不是資料集 validation split。
- 影像增強、I3D 特徵擷取、YOLO 特徵／ROI 與多模型融合都尚未開始。
