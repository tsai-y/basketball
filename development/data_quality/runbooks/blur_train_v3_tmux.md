# Ubuntu／tmux：Blur Train Scanner v3 全量執行

本手冊供使用者在主機執行。**Codex 本輪沒有執行 full train，也沒有使用 `--confirm-full-train`。** 這是離線 candidate 掃描，輸出不是排除名單或 cleaned dataset。原影片、label、manifest、split、frozen artifacts 一律唯讀。不得改指 validation/test，也不得將輸出放進原始資料目錄。

## 1. 提交 v3、確認 branch／commit／工作樹

先由使用者 review、commit、push 本次 v3 交付；不要混入 v1/v2 修改。以下命令由 repository root 執行。工作樹須乾淨，remote 須已保存該 commit；本手冊不自動 commit/push。

```bash
cd /home/tsai/projects/basketball
git branch --show-current
git log -1 --oneline
git status --short
test -z "$(git status --porcelain)"
git rev-parse HEAD
git log -1 --oneline origin/main
```

若不是 main，請依實際 branch 核對 remote 上相同 commit；不可在 scanner 執行途中換 branch、修改 config 或程式。run.json 會記錄啟動 commit，resume 另比對 scanner/dependency/config hashes 與版本，不接受不同環境或選樣。

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

## 3. Full-train dry-run（使用者執行）

這條命令只讀 manifest／設定／已提交凍結證據；不開影片、不 probe、不 decode、不建立 run。確認 `selected: 7193`，input/config hashes 與 smoke 報告一致。Full-train dry-run 也需要確認旗標；它不代表已掃描影片。

```bash
set -o pipefail
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/audit_blur_train_v3.py \
  --config development/data_quality/configs/blur_train_v3.yaml \
  --manifest data/processed/dataset_manifest.csv \
  --scope full-train --confirm-full-train --seed 20261009 --dry-run \
  | tee development/data_quality/artifacts/blur_train_v3_full_train_runtime/dry_run.json
```

若任何 hash、split、selected count 或防護失敗，停止並查原因，不改門檻或替換 manifest。Full run 不帶 `--max-videos`／`--video-id`；這兩個選項會改變選樣並使後續全量 resume context 不符。

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
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/audit_blur_train_v3.py \
  --config development/data_quality/configs/blur_train_v3.yaml \
  --manifest data/processed/dataset_manifest.csv \
  --output-dir development/data_quality/artifacts/blur_train_v3_full_train \
  --scope full-train --confirm-full-train --seed 20261009 --resume --rebuild-summary
/home/tsai/miniconda3/envs/yolo11/bin/python -B development/data_quality/scripts/verify_blur_train_v3.py \
  --output-dir development/data_quality/artifacts/blur_train_v3_full_train \
  | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/verification.log
git rev-parse HEAD | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/commits.log
sha256sum development/data_quality/configs/blur_train_v3.yaml | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/config_hashes.log
date -u --iso-8601=seconds | tee -a development/data_quality/artifacts/blur_train_v3_full_train_runtime/times.log
git status --short
```

Verifier PASS 要求所有所選影片 completion/output 有效、獨立重算一致；只依 access ledger 時不宣稱獨立 syscall 證據。Full-run 預設不用 strace，以免額外 I/O；如需同等 syscall 證據，可由主機操作者用本次 smoke 的 strace flags 包住 scanner，trace 必須另存在 runtime 目錄，再傳 `--trace` 給 verifier。續跑多次的 trace 必須涵蓋全部存取，不能以單次 trace 比對累積 ledger。

勿對全量輸出執行 smoke publisher，它只發表固定 20 支報告，且拒絕覆寫已存在 tracked 報告。保存整個 full-train run 與 runtime logs；正式品質／清理政策另行制定，本手冊沒有資料刪除或排除步驟。
