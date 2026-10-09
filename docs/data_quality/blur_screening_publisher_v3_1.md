# Basketball-51 Blur Screening Publisher v3.1

v3.1 補齊 v3 scanner 的 derived-output 完整度驗證與 full-train Markdown／JSON publication。它只讀固定 manifest/split metadata、run metadata、per-video CSV/JSON、completion、immutable attempts、failure/access ledgers、aggregate、config/code/artifact hashes；不 stat、open、probe 或 decode 影片，也不重新計算 candidate 分數。Candidate 指「低局部 Laplacian 候選」，不是真實模糊標籤或資料排除名單。

## 完整度狀態

| 狀態 | 定義 | 正式完整發表 |
| --- | --- | --- |
| `COMPLETE` | 固定 scope 的精確 ID 全部有一份有效 latest successful completion；無未知/held-out/duplicate/corrupt；config/run identity 一致；artifact hashes 通過；per-video 重建 aggregate 與最新快照一致；unresolved failure=0 | 可使用 `--require-complete` |
| `COMPLETE_WITH_FAILURES` | 全部 ID 已進 terminal state，但仍有列名的 decode/processing failure；沒有完整 scored coverage | `--require-complete` 拒絕；只能明確 `--allow-partial` |
| `PARTIAL` | 有未處理 ID、completion 缺失、選樣未完成或 aggregate 尚未完成 | `--require-complete` 拒絕；只能明確 `--allow-partial` |
| `INVALID` | held-out/unknown ID、config/run identity 混合、duplicate successful completion、artifact hash 損壞、非法紀錄或 aggregate 不一致 | 永遠拒絕發表 |

驗證比較實際 ID 集合而非只比數量：expected/selected/completion/attempt/ledger/aggregate 都會輸出 missing、unexpected、duplicate、successful、terminal failure、incomplete、corrupt、held-out 差異。歷史 failed attempt 可保留；同一 ID 出現多份有效 successful completion 則視為 duplicate。所有主要 candidate 統計逐支讀取 per-video outputs 重建，不信任既有 aggregate，也不把全資料集逐幀分數留在記憶體。

## CLI 與輸出契約

Publisher 支援 `--run-dir`、`--manifest`、`--config`、`--scope smoke|full-train`、`--output-md`、`--output-json`、`--require-complete`、`--allow-partial`、`--verify-artifacts`、`--dry-run` 與 `--help`。`--require-complete`／`--allow-partial` 互斥；full train 未明示 allow-partial 時預設要求 complete。Dry-run 只檢查與規劃，不寫報告。兩個正式輸出都使用 exclusive create，任一既有即停止，不提供 force/overwrite。

JSON schema version 為 `basketball.blur-screening-report.v3.1`，包含 `run_status`、`provenance`、`coverage`、`integrity`、`candidate_summary`、`group_summaries`、`distributions`、`failures`、`limitations`、`verification`。Run provenance 保存 scanner commit、publisher 執行 commit/worktree 狀態、config/manifest hash、scanner/publisher version、seed、scope、時間、run directory、resume/retry 次數與 run identity。Markdown 核心表格由同一份 JSON summary render，另由 parser 回讀 status/coverage 核心值驗證一致性。

報告按 label（Make/Miss）、FPS、source group 提供影片數、frames、candidate frames/ratio、segments、affected videos，另含每影片 candidate ratio 與 segment duration 分位數、最長 segment、最高 candidate-ratio 影片/來源、16/32/64 non-candidate window 可用性。中心三秒窗使用未來影格、local-z 對全片持續低品質的限制、synthetic calibration 非真值、來源/FPS 混雜、尚無 Make/Miss 效益證據及「不是排除名單」均固定列入 limitations。

正式主機命令與 ignored runtime 報告路徑以 [tmux runbook](../../development/data_quality/runbooks/blur_train_v3_tmux.md) 為準。既有 tracked smoke report 可用 `publish_blur_train_v3.py --check-only` 核對 20 支、3,055 幀、54 candidate frames、1.7676%、5 segments、0 failure；不會重新 decode smoke。
