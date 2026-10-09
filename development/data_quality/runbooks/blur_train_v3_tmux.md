# Ubuntu／tmux：Blur Train Scanner v3 全量執行

返回 [唯一技術入口](../README.md)／[研究索引](../../../docs/data_quality/README.md)。本手冊供組員在 Ubuntu／tmux 主機執行完整 train 候選掃描。**Candidate 不等於確認模糊，也不是排除名單或 cleaned dataset。** 原影片、label、manifest、split、frozen artifacts 一律唯讀。validation/test 影片不得存取，不得將輸出放進原始資料、baseline 或歷史 runs。

2026-10-09 文件整理僅執行帶 `--dry-run` 的 full-train 檢查；`--confirm-full-train` 也由 dry-run 要求，不表示已執行掃描。沒有執行 smoke、正式掃描、清理、訓練或評估。歷史方法／報告的執行紀錄保留原文。

## 主機執行前與完成後 checklist

- [ ] 確認 v3 已 commit／push，主機 checkout 與 remote 保存的執行版本一致（第 1 節）。
- [ ] `git status` clean，review 後由操作者處理未提交文件，不帶未審閱變更執行。
- [ ] 記錄完整 commit hash；執行途中不換 branch 或修改程式。
- [ ] 確認並保存 config hash，與下方凍結值及 smoke 報告一致。
- [ ] Full-train dry-run 成功：7,193 支、video opens/probes/decodes 全為 0。
- [ ] 確認 output directory 不存在；若已存在，須是相同 context 的有效 resume，保留原目錄並依第 7 節處理。
- [ ] 建立 tmux session，從 repository root 執行第 4 節正式命令。
- [ ] 保存 runtime log、exit code、commit/config hash 與起訖時間。
- [ ] 完成後 verify PASS；有失敗先處理原因，依需要 retry／rebuild 再 verify。
- [ ] 完成後 publish：**full-train publisher 尚未實作，此項保持待辦**；先保存完整 run／runtime logs，依第 9 節區分 smoke publication，不宣稱全量已發表。
- [ ] 不直接刪除 candidate 資料；任何清理政策另行制定。

## CLI 與固定路徑對照

| 項目 | 實際入口／限制 |
| --- | --- |
| Scanner | [audit_blur_train_v3.py](../scripts/audit_blur_train_v3.py) |
| Config | [blur_train_v3.yaml](../configs/blur_train_v3.yaml)；不修改凍結演算法參數 |
| Manifest | [dataset_manifest.csv](../../../data/processed/dataset_manifest.csv)；只接受此固定路徑，搭配 [split_summary.json](../../../data/processed/split_summary.json) |
| Output | `development/data_quality/artifacts/blur_train_v3_full_train/`；直接位於 ignored artifacts 下，名稱以 `blur_train_v3_` 開頭，拒絕 symlink |
| Runtime logs | `development/data_quality/artifacts/blur_train_v3_full_train_runtime/`；與 run 分開 |
| 全量／seed | `--scope full-train --confirm-full-train --seed 20261009`；正式執行必須明示 `--output-dir` |
| Resume | `--resume`；相同 config/code/environment/seed/scope/selection/input hashes |
| Retry failures | `--resume --retry-failures`；新 attempt，保留舊 attempt |
| Rebuild summary | `--resume --rebuild-summary`；只讀 derived outputs，不 stat/open/probe/decode 影片 |
| Verify | [verify_blur_train_v3.py](../scripts/verify_blur_train_v3.py) `--output-dir`；可選 `--trace`、`--write-report` |
| Publish | [publish_blur_train_v3.py](../scripts/publish_blur_train_v3.py) 僅限固定 20 支 smoke；full-train 未支援（第 9 節） |

Output 與 runtime 目錄皆由 [.gitignore](../../../.gitignore) 的 `development/data_quality/artifacts/` 排除，不位於原資料、baseline 或歷史 runs。

## 1. 確認 v3 已提交／推送、branch／commit／工作樹

v3 已完成並提交；在執行主機確認該版本已 push 且 checkout 正確，不需重新建立 v3 提交。以下命令由 repository root 執行。工作樹須乾淨，remote 須已保存該 commit；本手冊不自動 commit/push。

```bash
cd /home/tsai/projects/basketball
git branch --show-current
git log -1 --oneline
git status --short
test -z "$(git status --porcelain)"
git rev-parse HEAD
git log -1 --oneline origin/main
```

`origin/main` 是本機 remote-tracking ref，不保證反映遠端即時狀態；主機操作者須確認 push 成功及 ref 已更新。若不是 main，請依實際 branch 核對 remote 上相同 commit；不可在 scanner 執行途中換 branch、修改 config 或程式。run.json 會記錄啟動 commit，resume 另比對 scanner/dependency/config hashes 與版本，不接受不同環境或選樣。

## 2. 確認環境及保存稽核紀錄

使用既有 Ubuntu Conda `yolo11`，無需 GPU、不載入模型。以下為這台主機已驗證的 Python 路徑。需可用 `ffprobe`、`tmux`，以及足夠磁碟空間；保守建議預留 2 GiB，實測估算見 smoke 報告。輸出 run 目錄不要預先 mkdir，scanner 會 exclusive-create；runtime logs 使用另一目錄。

```bash
cd /home/tsai/projects/basketball
command -v tmux
command -v ffprobe
/home/tsai/miniconda3/envs/yolo11/bin/python -B -c 'import cv2, numpy, yaml; print(cv2.__version__, numpy.__version__)'
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/audit_blur_train_v3.py --help
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/tests/test_blur_train_v3.py
mkdir -p development/data_quality/artifacts/blur_train_v3_full_train_runtime
git rev-parse HEAD | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/commits.log
sha256sum development/data_quality/configs/blur_train_v3.yaml | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/config_hashes.log
date -u --iso-8601=seconds | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/times.log
df -h development/data_quality/artifacts
```

凍結 v3 config SHA-256 必須為 `ef73aea0b9cdd6656d8d017a90552b8737207fd7ee58a925b7d0dca45e5f9e20`，對照 [smoke 報告](../../../docs/data_quality/blur_screening_smoke_v3.md)。Hash 不符時停止，不修改參數來通過檢查。

## 3. Full-train dry-run（不開影片）

這條命令只讀 manifest／設定／已提交凍結證據；不開影片、不 probe、不 decode、不建立 run。確認 `selected: 7193`，input/config hashes 與 smoke 報告一致。Full-train dry-run 也需要確認旗標；它不代表已掃描影片。

```bash
set -o pipefail
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/audit_blur_train_v3.py \
  --config development/data_quality/configs/blur_train_v3.yaml \
  --manifest data/processed/dataset_manifest.csv \
  --output-dir development/data_quality/artifacts/blur_train_v3_full_train \
  --scope full-train --confirm-full-train --seed 20261009 --dry-run \
  | tee development/data_quality/artifacts/blur_train_v3_full_train_runtime/dry_run.json
```

預期 `status: DRY_RUN_PASS`、`scope: full-train`、`selected: 7193`、`video_opens/video_probes/video_decodes: 0`。2026-10-09 文件整理已用上述參數完成檢查，未建立 run，未開啟任何影片；validation/test opens/probes/decodes 亦為 0。依據為 CLI 回報及 dry-run 分支／metadata guard 的靜態核對，本輪沒有新 syscall trace。主機執行前仍需再跑一次。

Dry-run 在 output path、既有 run 與 resume context 檢查前即返回，不能據此宣稱 output 可用或 resume 有效；目錄已存在時須按第 7 節處理。若任何 hash、split、selected count 或防護失敗，停止並查原因，不改門檻或替換 manifest。Full run 不帶 `--max-videos`／`--video-id`；這兩個選項會改變選樣並使後續全量 resume context 不符。

## 4. 建立 tmux、正式執行

```bash
tmux new-session -s basketball-blur-v3
```

在新 tmux 視窗貼上以下命令。正式目錄為 `development/data_quality/artifacts/blur_train_v3_full_train/`，只能初次執行；若目錄已存在，使用第 7 節 resume，不刪除它。

```bash
cd /home/tsai/projects/basketball
set -o pipefail
date -u --iso-8601=seconds | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/times.log
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/audit_blur_train_v3.py \
  --config development/data_quality/configs/blur_train_v3.yaml \
  --manifest data/processed/dataset_manifest.csv \
  --output-dir development/data_quality/artifacts/blur_train_v3_full_train \
  --scope full-train --confirm-full-train --seed 20261009 \
  2>&1 | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/scan.log
scan_exit=${PIPESTATUS[0]}
printf 'scanner exit=%s\n' "$scan_exit" | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/exit_codes.log
date -u --iso-8601=seconds | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/times.log
```

0 表示完成；2 表示拒絕或未完成／失敗，查看 ledger；130 表示安全中斷。Scanner 一支失败會繼續下一支，沒有樣本替換。`run.json` 保存完整命令、config snapshot/hash、commit、code hashes；`invocations/*.json` 保存各次起訖 UTC、耗時、計數與 RSS。stdout 中的 `[i/7193]` 是該次進度，不是成功數。

## 5. Detach、attach、查看進度

按 `Ctrl-b` 後按 `d` detach，工作持續。重新連線：

```bash
tmux ls
tmux attach-session -t basketball-blur-v3
```

另一個 shell 查看 log／完成狀態（只讀 derived files）：

```bash
cd /home/tsai/projects/basketball
tail -n 30 development/data_quality/artifacts/blur_train_v3_full_train_runtime/scan.log
/home/tsai/miniconda3/envs/yolo11/bin/python -B - <<'PY'
from pathlib import Path
from collections import Counter
import json
out = Path('development/data_quality/artifacts/blur_train_v3_full_train')
counts = Counter()
for directory in (out / 'videos').iterdir():
    attempts = sorted(directory.glob('attempt-*'))
    if not attempts:
        continue
    record = attempts[-1] / 'completion.json'
    counts[json.loads(record.read_text())['status'] if record.exists() else 'in_progress_or_partial'] += 1
print(dict(counts))
print('These counts are progress only; the verifier checks hashes.')
for name in ['errors.jsonl', 'decode_failures.jsonl']:
    path = out / name
    if path.exists():
        print(name, sum(1 for _ in path.open()), 'events (historical retries included)')
PY
```

Aggregate 在結束或安全中斷時生成；執行中不另啟動 rebuild/resume，`flock` 會拒絕第二個 writer。不要用 error ledger 歷史行數當目前仍失敗影片數，以最新 attempt／aggregate 為準。

## 6. 安全中斷

Attach 後按 `Ctrl-C`，等待 scanner 寫出 aggregate／invocation 並回到 shell。亦可向 scanner PID 發 SIGTERM；兩者都保留目前部分嘗試。不要優先用 `kill -9`。若 SSH 中斷，tmux 仍可繼續，先 attach 確認狀態再決定是否中止。

若機器斷電或 SIGKILL，已有 completion 的影片仍可驗證；缺 record／temp／hash 不符會列 suspect，永不當完成、不直接覆寫。若 `run.json` 本身不完整，保留原目錄並改用新的獨立 run 名稱；不可手工補 completion。

## 7. Resume、最後重試失敗項

確認原設定、程式、環境版本、manifest/split 與原 run 完全相同。普通 resume 驗證並跳過完成項，掃描 pending，保留 suspect/failed；因此可能 exit 2。正常 resume 會核對原影片 stat identity，無需 hash 全部原始影片。

```bash
cd /home/tsai/projects/basketball
set -o pipefail
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/audit_blur_train_v3.py \
  --config development/data_quality/configs/blur_train_v3.yaml \
  --manifest data/processed/dataset_manifest.csv \
  --output-dir development/data_quality/artifacts/blur_train_v3_full_train \
  --scope full-train --confirm-full-train --seed 20261009 --resume \
  2>&1 | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/resume.log
resume_exit=${PIPESTATUS[0]}
printf 'resume exit=%s\n' "$resume_exit" | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/exit_codes.log
```

先檢查並解決缺檔／權限／磁碟／decoder 等原因；中斷的部分項也需以下明確 retry。此命令最多重試每個尚未成功項一次，成功項不重算；建立新 attempt，舊分數／可疑檔及 ledger 原封保留。

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/audit_blur_train_v3.py \
  --config development/data_quality/configs/blur_train_v3.yaml \
  --manifest data/processed/dataset_manifest.csv \
  --output-dir development/data_quality/artifacts/blur_train_v3_full_train \
  --scope full-train --confirm-full-train --seed 20261009 --resume --retry-failures \
  2>&1 | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/retry.log
retry_exit=${PIPESTATUS[0]}
printf 'retry exit=%s\n' "$retry_exit" | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/exit_codes.log
```

不同 config/code/selection 不允許在同一 run 續跑。不要改 config 來消除錯誤，也不要拿已完成 smoke 目錄當 full-train resume 目錄。

## 8. 重建 aggregate、驗證、保存結果

停止 scanner 後執行；重建只讀 derived outputs，不存取影片，保留所有旧 aggregate：

```bash
cd /home/tsai/projects/basketball
set -o pipefail
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/audit_blur_train_v3.py \
  --config development/data_quality/configs/blur_train_v3.yaml \
  --manifest data/processed/dataset_manifest.csv \
  --output-dir development/data_quality/artifacts/blur_train_v3_full_train \
  --scope full-train --confirm-full-train --seed 20261009 --resume --rebuild-summary
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/verify_blur_train_v3.py \
  --output-dir development/data_quality/artifacts/blur_train_v3_full_train \
  | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/verification.log
verify_exit=${PIPESTATUS[0]}
printf 'verify exit=%s\n' "$verify_exit" | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/exit_codes.log
git rev-parse HEAD | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/commits.log
sha256sum development/data_quality/configs/blur_train_v3.yaml | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/config_hashes.log
date -u --iso-8601=seconds | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/times.log
git status --short
```

確認 `verify_exit=0` 且 JSON `status: PASS`，否則不得宣稱驗證完成。Verifier PASS 要求所有所選影片 completion/output 有效、獨立重算一致；只依 access ledger 時不宣稱獨立 syscall 證據。Full-run 預設不用 strace，以免額外 I/O；如需同等 syscall 證據，可由主機操作者用本次 smoke 的 strace flags 包住 scanner，trace 必須另存在 runtime 目錄，再傳 `--trace` 給 verifier。續跑多次的 trace 必須涵蓋全部存取，不能以單次 trace 比對累積 ledger。

## 9. Publish 的支援範圍與全量交付

**目前沒有 full-train publish 命令。** `publish_blur_train_v3.py` 的 `publish()` 強制 `scope == smoke`、固定 20 支選樣，輸出固定的既有 smoke Markdown／JSON，且拒絕覆寫。`--output-dir` 不是通用 publication 介面，不能指定 full-train；`--check-only` 只核對既有 smoke 報告與證據，不能發表全量結果。

以下只顯示 publisher 支援的 CLI，適合核對入口：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/publish_blur_train_v3.py --help
```

既有 smoke 已完成發表；若日後需要唯讀核對其 publication，可用下列命令（依賴原 smoke/repeat/recovery 及 validation evidence；不是本輪文件驗證步驟）：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/publish_blur_train_v3.py --check-only
```

全量完成後先保存整個 run（含 `run.json`、所有 attempts、ledgers、invocations、aggregates）及 runtime logs／verify 結果。Full-train Markdown／JSON publisher 與發表流程留待後續任務實作、審閱；checklist 的 publish 保持待辦，不以 verify 或保存 logs 代替。這不阻擋已有 scanner 的主機候選掃描，但不能宣稱全量「scan → verify → publish」流程已全部具備。

正式品質／清理政策另行制定；不得直接刪除 candidate 資料，本手冊沒有資料刪除或排除步驟。
