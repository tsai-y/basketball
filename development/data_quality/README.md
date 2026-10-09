# Basketball-51 Data Quality：組員唯一技術入口

本頁統一 v1／v2／v3 的定位、研究連結與主機執行入口。目前研究仍為 full-video Make/Miss classification；A1/A2/B/C frozen，Gate 1C／原 Experiment D paused。

| 版本 | 用途 | 狀態 | 可否正式使用 | 主要入口 |
| --- | --- | --- | --- | --- |
| v1 | 絕對清晰度探索 | Complete | 不可作刪除規則 | [v1 報告](../../docs/data_quality/blur_screening_pilot_v1.md) |
| v2 | 合成模糊與時間校準 | Complete | 提供 v3 凍結參數 | [v2 報告](../../docs/data_quality/blur_screening_pilot_v2.md) |
| v3 | 完整 train 候選掃描器 | Scanner complete | 可於主機執行候選掃描 | [v3 runbook](runbooks/blur_train_v3_tmux.md) |

**v1、v2 是研究證據與歷史重現文件，不是正式執行入口；v3 是目前唯一的完整 train scanner。** Scanner complete 表示掃描器及既有 smoke 工程驗證完成，不表示全量已執行或資料已清理。

v3 只產生 candidate flags；**candidate 不等於確認模糊，也不是排除名單**，未標記也不等於清晰。原始影片保持不變，不刪格、拼接、重編碼或刪除 candidate 資料。validation/test 影片不得存取、probe 或 decode；固定 manifest 中的 split 字串只供身分與 denylist 核對。完整掃描由組員依 runbook 在 **Ubuntu／tmux** 執行。

## 閱讀與研究證據

建議依序讀 [研究索引](../../docs/data_quality/README.md) → [v3 方法](../../docs/data_quality/blur_screening_method_v3.md) → [v3 smoke 報告](../../docs/data_quality/blur_screening_smoke_v3.md) → [v3 tmux runbook](runbooks/blur_train_v3_tmux.md)，需要背景時再讀 v1／v2。

| 版本 | 方法 | Markdown 報告 | JSON | 設定與樣本 |
| --- | --- | --- | --- | --- |
| v1 | [方法](../../docs/data_quality/blur_screening_method_v1.md) | [Pilot](../../docs/data_quality/blur_screening_pilot_v1.md) | [JSON](../../docs/data_quality/blur_screening_pilot_v1.json) | [config](configs/blur_pilot_v1.yaml)、[manifest](pilot_manifest_v1.csv) |
| v2 | [方法](../../docs/data_quality/blur_screening_method_v2.md) | [Pilot](../../docs/data_quality/blur_screening_pilot_v2.md) | [JSON](../../docs/data_quality/blur_screening_pilot_v2.json) | [config](configs/blur_pilot_v2.yaml)、[manifest](pilot_manifest_v2.csv)、[來源庫存](source_inventory_v2.csv)、[歷史重現](README_v2.md) |
| v3 | [方法](../../docs/data_quality/blur_screening_method_v3.md) | [Smoke](../../docs/data_quality/blur_screening_smoke_v3.md) | [JSON](../../docs/data_quality/blur_screening_smoke_v3.json) | [config](configs/blur_train_v3.yaml)、[smoke manifest](smoke_manifest_v3.csv)、[runbook](runbooks/blur_train_v3_tmux.md) |

方法與報告中的「本輪／尚未提交／尚未執行」保留原交付時點的語境；目前操作流程以本頁及 runbook 為準，歷史數值不重寫。[進度紀錄](../../docs/progress_log.md) 追加各輪查核。

## v3 主機執行流程

v3 原封沿用 v2 凍結規則：primary 為 Laplacian local z ≤ −2.0，log-score 中心 3 秒 median/MAD，scale floor 0.05、log floor 1e-12，minimum duration 0.16 秒。Tenengrad 僅診斷，不加 AND/OR；16/32/64 windows 僅為資訊。中心窗是離線篩檢，正式清理政策尚未制定。

正式 scan、resume、retry、rebuild、verify 命令集中在 [Ubuntu／tmux runbook](runbooks/blur_train_v3_tmux.md)，執行前完成其中 checklist。由 repository root 可先執行下列零影片存取檢查：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/audit_blur_train_v3.py --help
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/audit_blur_train_v3.py \
  --config development/data_quality/configs/blur_train_v3.yaml \
  --manifest data/processed/dataset_manifest.csv \
  --output-dir development/data_quality/artifacts/blur_train_v3_full_train \
  --scope full-train --confirm-full-train --seed 20261009 --dry-run
```

`--dry-run` 只讀 metadata／凍結證據，不 stat/open/probe/decode 影片，也不建立 run；預期 `DRY_RUN_PASS`、`selected: 7193`、opens/probes/decodes 全為 0。它會在 output 檢查之前返回，故不能代替輸出目錄與有效 resume 的確認。不可省略 `--dry-run` 來做文件驗證。

正式輸出為 ignored 的 `development/data_quality/artifacts/blur_train_v3_full_train/`，runtime logs 另存相鄰 `blur_train_v3_full_train_runtime/`。不放進原始資料、baseline、歷史 runs 或既有 v1/v2/smoke 目錄。每影片 immutable attempts；`--resume` 跳過有效完成項，`--resume --retry-failures` 才重試失敗／partial，`--resume --rebuild-summary` 只重建 derived aggregate，舊產物保留。

**Publish 限制：**現有 [publisher](scripts/publish_blur_train_v3.py) 只支援固定 20 支 smoke，且拒絕覆寫既有報告；沒有 full-train publish CLI。全量完成後先 verify 並保存 run／runtime logs，全量 Markdown／JSON 發表仍待另行實作。不得將 smoke publisher 指向 full-train，也不能將 verify 稱為 publish。

## v1 歷史重現文件（非正式入口）

以下原文保留 v1 當時的重現命令與結果語境，不是本輪要執行的步驟。v2 歷史重現另見 [README_v2.md](README_v2.md)，不重新命名或搬移。

---

# 自動品質篩檢 — train-only Pilot v1

Pilot v1 已完成 30 支 train 影片、4,617 幀自動評分與候選影響分析。**這不是正式清理**：沒有排除影片、刪格、拼接或生成清理後影片；原始資料、固定 manifest/split、baseline 與歷史 runs 保持不變。沒有開始影像增強、特徵擷取、融合、訓練或評估。

- [方法及限制](../../docs/data_quality/blur_screening_method_v1.md)
- [Pilot 報告](../../docs/data_quality/blur_screening_pilot_v1.md)／[JSON](../../docs/data_quality/blur_screening_pilot_v1.json)
- [Pilot manifest](pilot_manifest_v1.csv)：以既有 video ID 關聯，不取代原始 manifest。
- [設定](configs/blur_pilot_v1.yaml)：取樣 seed、兩指標契約、P10、平滑與持續時間、contact sheet 規則。

## 從 repository root 執行

使用現有 Conda `yolo11`；沒有新增套件。先查看說明與零影片存取的 dry-run：

```bash
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/audit_blur_pilot.py --help
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/audit_blur_pilot.py --config development/data_quality/configs/blur_pilot_v1.yaml --dry-run --seed 20261008
conda run --no-capture-output -n yolo11 python -B development/data_quality/tests/test_blur_pilot.py
```

重算評分會只開啟選定 train 影片。**本次已完成，不需再次執行**；以下保留可重現入口，預設目錄存在時直接拒絕：

```bash
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/audit_blur_pilot.py --config development/data_quality/configs/blur_pilot_v1.yaml --output-dir development/data_quality/artifacts/blur_pilot_v1 --seed 20261008
```

`--output-dir` 必須是 ignored artifacts 下的新目錄，無 force/覆寫功能；需要另次重跑時用新的 run 名稱並保留舊輸出。`--check-video-id ID --check-video-path RELATIVE_PATH` 只驗證固定 train allowlist，未知、非 train ID/path 或兩者不相符皆 exit 2，無影片開啟。CLI 僅限 2–40 支的偶數 Pilot，不能以更改 sample_size 擴展完整 train。

本次獨立 syscall trace 已保留。下列重算只讀 derived CSV/JSON/trace，不開啟資料集影片：

```bash
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/verify_blur_pilot.py --trace development/data_quality/artifacts/blur_pilot_v1/runtime_logs/syscalls.log
```

驗證器的 `--publish` 僅供本次首次建立 report/manifest，遇到既有檔即拒絕，不應當成更新歷史報告的指令。一般驗證不加 `--publish`。若另次重新評分並需要同等零存取證據，依方法文件用 `strace -f -qq -yy -e trace=openat,execve,rename,renameat,renameat2,unlink,unlinkat` 包住正式評分命令，將 trace 存在本機 ignored 目錄；某些 sandbox 需要允許 ptrace，未成功追蹤不能宣稱具備獨立 syscall 證據。

## 產物

Tracked 候選檔包括 scripts、configs、tests、Pilot manifest 與 docs/data_quality 報告。沒有 git add/commit。

本機 ignored 根目錄：`development/data_quality/artifacts/blur_pilot_v1/`。

| 位置 | 內容 |
| --- | --- |
| `frame_scores/` | 30 份逐影片 CSV，共 4,617 rows；位置、兩分數、decode status、四策略 flags/reasons |
| `segments.csv` | 各策略所有候選及未標記區段起訖 frame/timestamp、幀數/秒數 |
| `video_summary.json` | 影片低分比例、最長候選區段、未標記區段與 16/32/64 連續窗口摘要 |
| `contact_sheets/` | 每指標低/中/高三組、每組三影片，共兩張 PNG/18 tiles；附 index.csv |
| `runtime_logs/` | 設定、metadata、decode status、ledger、版本、syscall trace、CLI/統計/保護檔驗證 |
| 根目錄 JSON/MD/CSV | 執行時原始報告與 Pilot manifest；tracked 報告另附驗證結果 |

Gate 1A per-video cache 不存在，因此此次只 probe 30 支已選定 train；未掃描其他影片。Source group 不是已核實 game ID。兩指標只能反映全画面局部強度變化，候選旗標與 contact sheets 都不是人工模糊真值。完整 train 評分、正式清理政策及後續影像增強仍待另外決定；本次完成後停止。
