# Repository 整理紀錄

日期：2026-10-07。範圍限結構整理、必要路徑與團隊文件；未開始資料清洗、模型實作或訓練。整理前 Git HEAD：`273b8808955c983a30499dc9064c933b7359a34e`。

## 開始狀態與盤點

- 已閱讀 root README、progress log、Research Baseline 與 Gate 1A／1B 文件，檢查程式用途、imports、設定、CLI 與文件中的路徑引用。repository 與適用上層未找到 AGENTS.md；`.agents/`、`.codex/` 為空。
- 開始時 `git status --short` 為空；98 個 tracked 檔，無非 ignored 的 untracked 檔。沒有需避讓的使用者未提交變更。搬移前再次比對來源內容。
- ignored 項包括原始 `archive/`、`papers/`、`TOPIC 1.pdf`、best／last checkpoints、run claim／training marker、C 的 manifest 快照、既有 Python cache。只盤點 Git 清單及目錄，不遞迴掃描原始影片內容。
- 原結構為 `baseline/`（A1/A2）、`reference_c3d/`（B 與 B/C 共用 data）、`experiment_c/`（C）、root `configs/`／`scripts/`，尚無 `development/`。
- 共六份 YAML，未發現專案 shell scripts、其他專案設定或測試框架。盤點 Python 的實際引用後才決定責任邊界。

## 舊路徑／新路徑

| 舊路徑 | 新路徑 | 處理 |
| --- | --- | --- |
| `readme.md` | [README.md](../README.md) | 統一 root 文件名稱，重寫為繁體中文團隊入口；詳細結果改由既有 Research Baseline、protocol、runs 索引提供 |
| Gate 1B 使用的 `../readme.md` | [readme.md](../readme.md) → `README.md` | 本機連結檢查發現兩處保護報告連結；保留小寫 symlink 別名，報告原文不動、README 內容僅一份 |
| `scripts/audit_c3d_shape.py` | [baseline/scripts/audit_c3d_shape.py](../baseline/scripts/audit_c3d_shape.py) | B 專用歷史 shape 輔助入口，非共用資料工具 |
| `scripts/benchmark_c3d_reference.py` | [baseline/scripts/benchmark_c3d_reference.py](../baseline/scripts/benchmark_c3d_reference.py) | B 專用歷史可行性 benchmark 入口 |
| `baseline/`、`reference_c3d/`、`experiment_c/` 核心 Python | 原路徑 | 有 frozen source hash、cross-package imports，保留單一實作 |
| `configs/*.yaml`、其餘 `scripts/*.py` | 原路徑 | hash／protocol、歷史產物或共用工具相依；由 baseline／repository map 建立索引 |
| 原始資料、固定 manifest／split、`runs/`、既有報告 | 原路徑 | 內容與位置不變；progress log 僅尾端追加，docs/README 僅補導航 |

Research Baseline、Gate 報告及歷史 logs／protocol 中的舊路徑維持當時寫法，不批次取代。舊 root README 全文可從既有 Git 歷史取回：`git show 273b8808955c983a30499dc9064c933b7359a34e:readme.md`。本次 root README 是新版入口，不宣稱它與舊版內容相同。

## 必要程式與設定修改

| 檔案／項目 | 修改前 | 修改後 | 原因 |
| --- | --- | --- | --- |
| 兩個搬移腳本的 `ROOT` | `Path(__file__).resolve().parents[1]` | `Path(__file__).resolve().parents[2]` | 多一層目錄後仍定位同一 repository root |
| `docs/README.md` root 連結 | `../readme.md` | `../README.md` | 配合 root README 命名 |
| 六份 YAML 全部路徑與值 | 既有內容 | **完全不變** | 不改寫 frozen config 或 run config snapshot |

兩個搬移腳本各只改一行，沿用原本的 ROOT／`sys.path.insert` 慣例，未新增 path hack、shim、相容性 hash 特例或重複程式。imports、預設 config、資源與輸出目錄均解析到原位置。模型、sampling／preprocessing、超參數、random seed、split、指標及 checkpoint 選擇邏輯未修改。

新增 [baseline 說明](../baseline/README.md)、[development 規則](../development/README.md)、[data_quality 說明](../development/data_quality/README.md)、[目前研究方向](current_research_direction.md)、[repository map](repository_map.md) 及本紀錄。`docs/README.md` 補目前方向入口；`docs/progress_log.md` 僅追加本次紀錄。

## 保留原位的理由

- A1／A2／B 及 B gates 的 fingerprint 包含入口本身；搬移後改 ROOT 也會改 protocol hash。A2／B 還有針對歷史 checkpoint 的 legacy 相容分支。本輪不修改其邏輯，也不以新增 hash 特例掩蓋差異。
- 正式 evaluator 比對 root config 與 run config 的 bytes hash；B 設定引用 `configs/c3d_reference.yaml`，B/C 共用其前處理契約。C environment 記錄 model／data／aggregate hash。整包搬移需更廣的相容性驗證，因此核心、設定與正式入口保留。
- 搬移的兩個輔助入口沒有其他 Python caller，也未列入上述固定 source hash 表；歷史文字引用依對照表追溯。它們正常模式仍會寫入舊 feasibility 目錄，未改變輸出以免偷改歷史行為，本輪不執行其正常模式。
- `inspect_dataset.py` 被 `build_manifest.py` 直接匯入；兩者屬共用資料準備工具，保持單一來源。Gate 1A 報告釘住腳本 hash 與命令，因此 Gate 1A 工具也保持原位。
- 原始影片、固定 manifests／split、runs／checkpoints／logs、formal test metrics／predictions、其他歷史 gates 與研究報告均保留；沒有刪除疑似過時檔案。

## 驗證結果與限制

驗證使用現有 `yolo11` Python 3.10.20／PyTorch 2.11.0+cu128／PyYAML 6.0.3，未安裝套件。標準函式庫的檔案與連結檢查使用現有 Python。檢查用暫存資料留在 `/tmp`，不寫入資料集或歷史 run。

| 檢查 | 結果與範圍 |
| --- | --- |
| Markdown 本機連結 | **PASS**：22 份獨立 Markdown、212 個本機連結，0 失效；既有 pooling／FC anchor 也通過。symlink 指向同一 README，不重複計數。未檢查外部網址可用性 |
| Python syntax | **PASS**：21 份 Python 以 `compile(..., 'exec')` 檢查，不執行程式、不寫 pyc |
| Import | **PASS**：12 個 package／模組項目；已先檢查模組頂層與 main guard，不建立模型／Dataset 或執行訓練。兩個搬移入口的 ROOT 均等於 repository root |
| CLI | **PASS**：全部 11 個入口 `--help` exit 0，包括兩個新位置、六個 frozen 訓練／gate／評估入口及三個資料／稽核工具；使用 `python -B` |
| YAML 與路徑 | **PASS**：六份設定解析成功，29 個 root-relative 資源／輸出路徑存在；只確認 path 存在，不開啟 checkpoint／影片，也未逐列檢查影片存在性 |
| Gate 1A metadata check | **PASS**：`python -B scripts/audit_i3d_temporal_feasibility.py --mode check`；selected 8,740，video opens／probe count／test video accesses 均 0；只讀 manifest／split，不重做 Gate 1A |
| 搬移程式差異 | **PASS**：與原始 Git bytes 比對，每支只替換 ROOT 的 `parents[1]` → `parents[2]`；其餘內容相同 |
| 保護文字檔 | **PASS**：起始記錄 99 份文字檔（98 tracked，加 C 的 ignored manifest 快照）；扣除 5 個預期變動舊路徑後，94 份 SHA-256 完全相同，涵蓋 configs、固定 manifest／split、Research Baseline、Gate MD／JSON、所有 run 文字產物及其餘程式 |
| Progress log | **PASS**：原檔 bytes 為新檔完整前綴，只追加，既有段落與路徑不變 |
| 舊路徑引用 | **PASS**：目前 Python／YAML 不引用已搬走的兩個腳本；新操作文件使用新入口。舊 protocol／logs 的文字引用刻意保留，對照列於上表；Gate 1B 的舊 README 連結由 alias 維持 |
| Git diff | **PASS**：`git diff --check` 無 whitespace 錯誤；另外檢查 untracked 文件與搬移後程式，沒有 stage、commit、push |

首次連結檢查找出 Gate 1B 兩處 `../readme.md` 連結；建立相容 symlink 後重新檢查通過，Gate 報告原文未修改。固定 manifest SHA-256 為 `e7bba1acd6028cf67cdf28251e69852e82cff201e13519ed05970ac483cce6b6`，split-summary 為 `953da31313976763c8f5d597da5f0fe5a6a45e10f5d5edcd42fb0c120af2f7c5`，與既有值一致。

不執行 evaluator 的 `--check`（它會載入 checkpoint），也不執行 shape audit／benchmark 正常模式、訓練 smoke／dry-run、`inspect_dataset` 掃描或 `build_manifest` 建置。沒有對影片或 checkpoint 做 hash；因此保護檔案的 byte-level 驗證只涵蓋上述文字檔，不能外推為整個資料集／checkpoint 的重新稽核。

不啟動訓練、evaluation、GPU benchmark、完整資料掃描、影片讀取／probe／解碼，不載入 checkpoint 或下載模型。**不宣稱模型數值等價、GPU 可行性或歷史結果已重新驗證。**

## 待審核項目

1. 部分 frozen 元件仍在 root：若未來需要完全搬入 baseline，須另定 protocol hash／resume／formal evaluator 相容性方案；不是本輪已完成的工作。
2. 歷史 protocol、config 註解、progress log 頂部仍有「尚未訓練」、D／pre-release 下一步等當時狀態。原文保留；目前排程以 [研究方向](current_research_direction.md) 為準。原 SPL skeleton 字樣不代表現階段已選定模型模組。
3. B 小樣本失敗紀錄、claim／training marker、cache 等疑似過時項目全部保留，未判定可刪。
4. 影片、PDF、checkpoint、C manifest 快照等 ignored 本機檔案不隨 clone 取得；團隊分發／備份方式待確認，本輪沒有下載或複製。
5. 新研究產物路徑目前只是規則，尚未建立 outputs 或 ignore 規則；品質指標、門檻、增強策略、模型組合與融合方式均待後續研究。
6. 本機為區分大小寫的 Linux filesystem；`readme.md` 是指向唯一內容 `README.md` 的相容 symlink，以保留 Gate 1B 原文連結。不區分大小寫或不支援 symlink 的 checkout 需另行驗證；本輪不宣稱跨平台 checkout 已驗證，不可同時維護兩份 README 內容。

本輪不執行 `git add`、commit 或 push；最終狀態需同時查看 tracked diff 與 untracked 新檔，因未 stage 的搬移在 Git status 會顯示刪除舊檔＋未追蹤新檔。
