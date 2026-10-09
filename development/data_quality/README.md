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
