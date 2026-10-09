# 自動模糊篩檢 Pilot v2

> **v2 歷史重現文件，非正式執行入口。** 組員請先讀 [統一技術入口](README.md)；完整 train 候選掃描只使用 [v3 runbook](runbooks/blur_train_v3_tmux.md)。以下「本輪／本次」均指 v2 研究交付時點。

Pilot v2完成48支train影片、12個來源的合成模糊與時間偵測校準。這不是正式清理，不產生排除名單或重編碼影片。所有Pilot v1檔案、報告及原始資料保持不變；本頁保留v2重現步驟，版本定位與目前執行流程見統一README。

- [方法](../../docs/data_quality/blur_screening_method_v2.md)
- [報告](../../docs/data_quality/blur_screening_pilot_v2.md)／[JSON](../../docs/data_quality/blur_screening_pilot_v2.json)
- [選樣manifest](pilot_manifest_v2.csv)／[全部train來源庫存](source_inventory_v2.csv)
- [設定](configs/blur_pilot_v2.yaml)

Laplacian與Tenengrad對本次Gaussian/motion均單調下降；verification時間F1為0.862/0.811，AND為0.859。來源與FPS仍混雜，不能直接把分數用作資料刪除門檻。保留兩項診斷，本輪未新增第三指標。

## 安全檢查與重現

由repository root使用既有Conda `yolo11`，沒有新增套件。以下不開影片：

```bash
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/audit_blur_pilot_v2.py --help
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/audit_blur_pilot_v2.py --dry-run
conda run --no-capture-output -n yolo11 python -B development/data_quality/tests/test_blur_pilot_v2.py
conda run --no-capture-output -n yolo11 python -B development/data_quality/tests/test_blur_pilot.py
```

`--check-video-id ID --check-video-path RELATIVE_PATH`只核對固定train身份，不讀取影片。未知或validation/test ID、偽裝路徑皆exit 2；拒絕測試使用manifest字串且mock opener。

本次已完成，不需再次評分。獨立產物根目錄為ignored的 `development/data_quality/artifacts/blur_pilot_v2/`。評分程式拒絕既有目錄，固定seed與12×4上限，沒有完整train執行或force選項。

```bash
# 重現須自行指定新的ignored run名稱；下列預設因本次產物已存在而拒絕。
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/audit_blur_pilot_v2.py
```

若要重現同等存取證據，需以strace包住上述評分入口：

```bash
strace -f -qq -yy -e trace=openat,execve,rename,renameat,renameat2,unlink,unlinkat \
  -o /tmp/basketball_v2_syscalls.log \
  conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/audit_blur_pilot_v2.py
```

正式本次執行使用同一環境的Python `/home/tsai/miniconda3/envs/yolo11/bin/python`直接執行。strace需允許ptrace；不得在未成功trace時宣稱獨立syscall驗證。全部影片入口重用v1 TrainGuard與Gate固定manifest/split hash。

## Derived-only分析、驗證與出版

```bash
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/analyze_blur_pilot_v2.py
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/verify_blur_pilot_v2.py
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/publish_blur_pilot_v2.py
```

本次完成後可用以下命令重新查核，完全不寫入或覆蓋產物：

```bash
conda run --no-capture-output -n yolo11 python -B development/data_quality/scripts/verify_blur_pilot_v2.py --check-only
```

三個入口都只讀derived artifacts、metadata或trace，不開影片。分析拒絕既有 `analysis/`；驗證結果以exclusive create保存 `runtime_logs/verification.json`；出版拒絕既有tracked v2報告/manifest，因此上述是首次生成入口，這次已完成後不應覆寫執行。另次run須使用新的根目錄（`--output-dir`）並保留舊結果，publication固定檔名仍會拒絕。

獨立驗證依賴執行前儲存的 `runtime_logs/initial_tracked_hashes.json`（只對tracked files計算hash，絕不掃描/雜湊原始影片）、`runtime_logs/syscalls.log`、測試log與完整scores。此次全部113個原tracked files保持不變。第一次發表前另檢查artifact hashes，避免使用已變動的驗證證據。

詳細CSV與分數陣列均ignored；Git新增的交付包括4個scripts、10項測試、config、選樣/來源CSV、本頁與方法/報告。未自動git add、commit或push。
