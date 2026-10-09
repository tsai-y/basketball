# Basketball-51 Blur Train Scanner v3

v3 是 full-video Make/Miss classification 的 train 候選品質掃描器。它產生品質註記，沒有正式清理、排除、刪格、拼接、重編碼、增強、特徵擷取、checkpoint 載入或模型訓練／評估。A1/A2/B/C frozen；Gate 1C 與原 Experiment D paused。完整 train 的執行由使用者在主機上另行啟動，本次只處理固定 20 個 train IDs。

## 凍結來源

開始時 HEAD `dab6c46`，Pilot v2 的 12 個交付檔案已提交，工作樹乾淨。v3 的 `load_config` 每次檢查 v2 config、JSON 報告、方法文件、評分程式的 SHA-256，並與 `git show HEAD:<path>` 比對；任何未提交、hash 或參數差異均停止，不自行修正或重新擬合。

| 證據 | SHA-256 |
| --- | --- |
| v2 config | `80fe9ad128d284777eb51a428ec9eabc9a6c3d4c0256637a522b1aa44d607473` |
| v2 JSON 報告 | `95f95463ccf1cbb0e5b24e73dca60e375c71be7fcc0261bdd15cd40cf96e341f` |
| v2 方法 | `ac1c75ebaa967282ab24e994fc7ea6bbf8f266dbd41b137e9cea9ee2385e0582` |
| v2 評分程式 | `d86e91472c010d99a02b4c97c4dd96da95262c6468de81bcd5e8d76ce671407b` |

精確採用 v2 config 的 3 秒中心窗、MAD scale floor 0.05、log floor 1e-12，以及 JSON `temporal_thresholds.laplacian` 的 z ≤ −2.0、minimum duration 0.16 秒。Prompt 與已提交證據一致。Tenengrad 照常計算分數及 local z，僅供診斷，沒有 AND/OR 參與 primary candidate。

## 數值與時間契約

重用 v2 `score` 與 `local_normalize`，不修改 v1/v2。OpenCV CAP_FFMPEG，threads=1、OpenCL=false、optimized=false；逐格保留原始 320×240，BGR→uint8 gray→float64 [0,255]。Laplacian ksize=1 population variance；Tenengrad 為 Sobel ksize=3 的 mean(Gx²+Gy²)；BORDER_REFLECT_101。沒有 resize、crop、resample 或影像編碼。

```text
log_score = log(max(score, 1e-12))
width = ceil(3 * FPS)，偶數再加 1
median_t = median(log_score 在以 t 為中心、兩端截短的 window)
MAD_t = median(abs(window_log_scores - median_t))
local_z_t = (log_score_t - median_t) / max(1.4826 * MAD_t, 0.05)
```

只保留 Laplacian local z ≤ −2.0 且連續長度/FPS ≥ 0.16 秒的 runs。不填 gap、不合併隔開區段、不做形態學操作。23.976 FPS 最少 4 幀；29.97 FPS 最少 5 幀。中心窗使用前後畫面，**屬離線篩檢，不是即時偵測器**。

frame index 零起始，segments 使用半開 `[start_frame, end_frame_exclusive)`；時間端點以 index/FPS 明示為 CFR 估計，duration 為幀數/FPS。逐幀另保留 decoder timestamp：有限、嚴格遞增、與 index/FPS 相差不超過半幀且 avg/r FPS 相同才視為可信，否則留空並標示 `unverified_use_frame_index`。未標記不等於清晰或內容可用；16/32/64 非 candidate windows 僅為資訊，未選定模型輸入長度。

整片持續低品質可能被局部正規化消去。另輸出各指標 P0/P10/P50/P90/P100、低於 **v2 calibration 固定 P10** 的比例、最長連續時間及全片低於參考值旗標。參考 Laplacian 1803.9773826943294、Tenengrad 27394.88785677083；它們只作診斷，並非新增 candidate 或排除規則，沒有用本輪或 full train 重新擬合。

## 輸入防護與抽樣

唯一 allowlist 為 `data/processed/dataset_manifest.csv` 的 train rows，搭配 `split_summary.json` 的 source assignment；重用 Gate 1A 固定 manifest/split hash。`--manifest` 只接受此固定路徑。未知 ID、非 train split、錯配路徑、任意路徑、symlink 及 hardlink aliases 均拒絕。每次 probe/decode 前先授權 exact ID/path，再檢查 resolved path、stat identity、唯讀 O_NOFOLLOW FD；ffprobe/OpenCV 只接收已授權 `/proc/self/fd`。validation/test 的拒絕測試只讀 manifest 字串，opener 與 stat 入口被 mock 封鎖。

Smoke seed=20261009，硬上限 20 支；交替 Miss/Make、優先尚未使用來源、SHA-256(seed:video_id) 破同分。先用已提交 v1/v2 train manifests 的 source/FPS 欄位為各已知 FPS 群組預留兩支，再按相同來源分散規則補足；沒有讀取任何品質分數。FPS hint 是來源推測，實際 FPS 仍逐支 probe。最後依 video_id 排序，固定 CSV 必須與重算選樣完全一致。`--video-id` 在 smoke 模式只能縮小這 20 支，不能換入其他影片；失敗不替換。Full-train 模式同樣依 video_id 排序。

dry-run 只讀設定、tracked 證據及 manifest metadata，不 stat、open、probe、decode 影片，也不建立輸出。Full-train（含 dry-run）必須同時帶 `--confirm-full-train`；本輪 Codex 沒有使用這個 flag，僅驗證缺少確認時拒絕。

## 記憶體、落盤及失敗

一次一支影片、一張解碼畫面；不保存灰階影片或全資料集 frame scores。RAM 僅保存單影片分數、timestamp/local z/flags，影片完成即寫 CSV、summary、completion。工程上限 10,000 frames、FPS ≤120、320×240；超出是明確工程 failure，不是品質或排除規則，不偷偷截斷為成功。Aggregate 逐支讀取輸出，只累積影片層級資訊，空間 O(影片數)，沒有全資料集 frame arrays。

每次執行持有 `flock`；第二個 writer 拒絕。所有正式輸出先寫同目錄唯一 temporary file、flush/fsync，再 atomic rename 及 fsync directory。正式檔名必須不存在。最後寫 completion 作 commit marker，含 input stat identity（path、size、mtime/ctime、device/inode、manifest hash）、config hash、scanner/dependency hashes、expected/processed count、兩個輸出 hash、UTC 及 status。原始影片不做全量 hash；stat identity 不是內容雜湊保證，複製到另一主機後需開新 run。

每個 video ID 以 SHA-256 作目錄名稱，每次嘗試為遞增 `attempt-000001` 等。解碼不足、probe/decoder failure 與候選分開：failure summary/frames 的 candidate 欄為 unavailable，不將已讀部分當品質判定；`decode_failures.jsonl` 獨立保存。授權、輸入缺失或輸出 I/O 問題進 `errors.jsonl`，後續影片繼續；沒有因失敗替換影片。

`access.jsonl` 在每次 open 前、成功開 FD 後立即 append/flush/fsync，包含 ID、path、split、operation。SIGINT/SIGTERM 停止目前嘗試、保留部分輸出，寫入 error ledger／當下 aggregate；SIGKILL 可能留下 temp 或缺 completion，但不會被視為成功。若連初始化 `run.json` 都未完成，resume 拒絕；保留該目錄，另選新的輸出名稱。

## Resume 與驗證

既有目錄必須 `--resume`，且 config（含檔案 bytes hash）、code/dependency hash、環境版本、seed、scope、選樣及輸入 manifest/split hashes 完全相同。只有完成 record、input identity、CSV row count/order、輸出 inventory/hash、status 均一致且 decode 成功才 SKIP。不同 config hash 在影片存取前拒絕並寫 error ledger。

失敗／不完整／hash mismatch 皆不當成完成。普通 resume 跳過有效完成項，處理尚未開始項，記錄可疑項；要重試 failed/suspect，明確加 `--retry-failures`，保留舊 attempt，在新 attempt 中重算。同一命令只有一次嘗試，不做無限 retry。若新 attempt 本身失敗，舊成功結果也不會遮蓋最新失敗。磁碟滿／權限等問題應先解決，不能靠反覆重試。

`--resume --rebuild-summary` 僅讀 per-video outputs；不 stat/open/probe/decode 原影片，會檢查 completion/output integrity 並建立新的 immutable aggregate。這是既有輸出的重建，不重新證明目前原影片 identity；正常 resume 才再次核對原影片 stat。每次 aggregate 使用唯一時間名稱，舊版本保留。`status=COMPLETE` 表示該選樣的 derived outputs 完整；執行 exit code／errors 還需檢查，因 live input 改變仍可能使該次 resume 失敗。

退出碼：0 完整；2 未完成、拒絕或失敗；130 安全中斷。Verifier 逐支讀 derived CSV，以 `statistics.median`、`math.log` 獨立重算 local z，以 `itertools.groupby` 獨立重算連續區段、duration、windows、分位數及 aggregate，完全不讀影片。選擇 `--trace` 可另核對 syscall/ledger。Publisher 只在驗證通過後 exclusive-create 本輪報告，`--check-only` 只檢查既有 JSON/Markdown/CSV 一致性。

## 輸出及下一步

忽略目錄：`development/data_quality/artifacts/blur_train_v3_smoke/`；同樣以 `blur_train_v3_` 前綴、直接位於 ignored artifacts 下的獨立目錄供重跑或 full train。不能使用既有 v1/v2 artifacts、歷史 runs、原資料目錄或 symlink 路徑。

- `run.json`：selection、config snapshot/hash、code hashes、environment、commit 與命令。
- `videos/<ID hash>/attempt-*/frames.csv`：每幀兩分數、local z、timestamp、candidate/reason、metadata、版本。
- 同目錄 `summary.json`、`completion.json`：品質診斷、完整性與輸入身分。
- `access.jsonl`、`errors.jsonl`、`decode_failures.jsonl`：存在相關事件時建立；append-only。
- `invocations/*.json`：各次時間、跳過／嘗試數、failure、RSS。
- `aggregates/summary-*.json`：可重建且不可覆写的 aggregate snapshots；執行中逐片看 completion，aggregate 在中斷或結束時生成。

[Smoke 報告](blur_screening_smoke_v3.md) 含實測、保護檔驗證與外推；[Ubuntu/tmux 手冊](../../development/data_quality/runbooks/blur_train_v3_tmux.md) 提供實際 CLI。Full train 執行前由使用者 commit/push v3，再記錄正式 commit；不得中途改 config／程式或指向 validation/test。品質真值、主體 ROI、來源差異、長期低品質、人工複核、錯誤成本與任何清理／保留／排除政策仍未決。
