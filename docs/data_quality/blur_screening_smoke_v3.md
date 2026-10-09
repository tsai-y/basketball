# Basketball-51 Blur Train Scanner v3 Smoke Report

**SCANNER IMPLEMENTATION COMPLETE — 僅完成固定 20 支 train smoke；完整 7,193 支尚未執行。**

開始 HEAD `dab6c46e7ae4ea5395d2bbccfee9dfac3feb8e71`，Pilot v2 已提交且工作樹乾淨。沒有 git add、commit 或 push；既有 tracked 檔僅預期更新 README 與追加 progress log。

v3 是離線資料品質候選掃描器；沒有正式清理、排除、刪格、拼接、重編碼、增強、特徵擷取、checkpoint 或模型訓練／評估。Gate 1C／原 Experiment D 維持 paused。

[方法與防護](blur_screening_method_v3.md)；[完整 Ubuntu/tmux 手冊](../../development/data_quality/runbooks/blur_train_v3_tmux.md)；[固定 smoke manifest](../../development/data_quality/smoke_manifest_v3.csv)。

## 凍結規則與版本

Primary 僅 Laplacian variance；log-score 中心 3 秒 rolling median/MAD，scale=max(1.4826×MAD,0.05)，log floor=1e-12，z≤−2.0，連續 duration≥0.16 秒。Tenengrad 分數／local z 繼續保存，但不使用 AND/OR 改變候選。Prompt 與已提交 v2 證據沒有差異，也未重調參數。

v3 config SHA-256：`ef73aea0b9cdd6656d8d017a90552b8737207fd7ee58a925b7d0dca45e5f9e20`；scanner/dependency fingerprint：`19372422e554779a09d54ca93e4927bf91d5665af5e6c127904727d4b5edc02a`。

| v2 frozen evidence | SHA-256 |
| --- | --- |
| development/data_quality/configs/blur_pilot_v2.yaml | `80fe9ad128d284777eb51a428ec9eabc9a6c3d4c0256637a522b1aa44d607473` |
| development/data_quality/scripts/audit_blur_pilot_v2.py | `d86e91472c010d99a02b4c97c4dd96da95262c6468de81bcd5e8d76ce671407b` |
| docs/data_quality/blur_screening_method_v2.md | `ac1c75ebaa967282ab24e994fc7ea6bbf8f266dbd41b137e9cea9ee2385e0582` |
| docs/data_quality/blur_screening_pilot_v2.json | `95f95463ccf1cbb0e5b24e73dca60e375c71be7fcc0261bdd15cd40cf96e341f` |

## 樣本、候選與每支執行時間

seed=20261009，固定 20 個 train IDs、20 來源、Make/Miss 各 10。先保留既有 train FPS metadata 所示的兩群，再交替 label、優先未使用來源、SHA-256 破同分；不讀品質分數選樣，失敗不替換。

成功解碼 3,055 幀；候選 54 幀（1.7676%）、5 區段、涉及 5 支；實際 decode failures=0。所有影片均有 16/32/64 非 candidate window（僅資訊，非模型長度選擇）。

| video_id | label | source | FPS | frames | 候選幀／段 | 秒 | 輸出 bytes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2p0/2p0_v150_003503_x264.mp4 | Miss | v150 | 23.976 | 146 | 0/0 | 0.262685 | 48166 |
| 2p0/2p0_v181_000500_x264.mp4 | Miss | v181 | 23.976 | 145 | 8/1 | 0.272808 | 48466 |
| 2p1/2p1_v118_004008_x264.mp4 | Make | v118 | 23.976 | 146 | 0/0 | 0.268755 | 48256 |
| 2p1/2p1_v124_000756_x264.mp4 | Make | v124 | 23.976 | 146 | 0/0 | 0.271029 | 48247 |
| 2p1/2p1_v169_011006_x264.mp4 | Make | v169 | 23.976 | 146 | 7/1 | 0.272199 | 48744 |
| 2p1/2p1_v183_003903_x264.mp4 | Make | v183 | 23.976 | 146 | 0/0 | 0.272574 | 48213 |
| 3p0/3p0_v119_012427_x264.mp4 | Miss | v119 | 23.976 | 146 | 0/0 | 0.277288 | 48204 |
| 3p0/3p0_v122_004044_x264.mp4 | Miss | v122 | 23.976 | 146 | 0/0 | 0.265343 | 48249 |
| 3p0/3p0_v157_001615_x264.mp4 | Miss | v157 | 23.976 | 146 | 15/1 | 0.263432 | 49156 |
| 3p0/3p0_v158_011740_x264.mp4 | Miss | v158 | 29.97 | 180 | 0/0 | 0.320709 | 58388 |
| 3p1/3p1_v159_014222_x264.mp4 | Make | v159 | 29.97 | 180 | 0/0 | 0.311015 | 58474 |
| ft0/ft0_v117_000311_x264.mp4 | Miss | v117 | 23.976 | 146 | 0/0 | 0.264116 | 48326 |
| ft1/ft1_v120_003638_x264.mp4 | Make | v120 | 23.976 | 146 | 0/0 | 0.259565 | 48075 |
| ft1/ft1_v156_001230_x264.mp4 | Make | v156 | 23.976 | 146 | 13/1 | 0.264096 | 48523 |
| ft1/ft1_v161_004901_x264.mp4 | Make | v161 | 23.976 | 146 | 11/1 | 0.273884 | 48716 |
| mp0/mp0_v111_003913_x264.mp4 | Miss | v111 | 29.97 | 180 | 0/0 | 0.321890 | 58267 |
| mp0/mp0_v145_003246_x264.mp4 | Miss | v145 | 23.976 | 146 | 0/0 | 0.271536 | 48174 |
| mp0/mp0_v160_000637_x264.mp4 | Miss | v160_v166 | 23.976 | 146 | 0/0 | 0.270928 | 48995 |
| mp1/mp1_v108_003320_x264.mp4 | Make | v108 | 23.976 | 146 | 0/0 | 0.272835 | 48145 |
| mp1/mp1_v116_010540_x264.mp4 | Make | v116 | 29.97 | 180 | 0/0 | 0.339202 | 58420 |

每支秒數包含 probe、decode、評分、local z 與 frame CSV 寫入；不包含最後 summary/completion 寫入。總 scanner wall time 另含初始化、所有寫入與 aggregate，初次是在 strace 下量測。

## 描述性分布

### label

| 群組 | 影片 | 影格 | 候選幀 | 候選比例 |
| --- | --- | --- | --- | --- |
| Make | 10 | 1528 | 31 | 2.0288% |
| Miss | 10 | 1527 | 23 | 1.5062% |

### fps_rational

| 群組 | 影片 | 影格 | 候選幀 | 候選比例 |
| --- | --- | --- | --- | --- |
| 2997/100 | 4 | 720 | 0 | 0.0000% |
| 2997/125 | 16 | 2335 | 54 | 2.3126% |

### source_group

| 群組 | 影片 | 影格 | 候選幀 | 候選比例 |
| --- | --- | --- | --- | --- |
| v108 | 1 | 146 | 0 | 0.0000% |
| v111 | 1 | 180 | 0 | 0.0000% |
| v116 | 1 | 180 | 0 | 0.0000% |
| v117 | 1 | 146 | 0 | 0.0000% |
| v118 | 1 | 146 | 0 | 0.0000% |
| v119 | 1 | 146 | 0 | 0.0000% |
| v120 | 1 | 146 | 0 | 0.0000% |
| v122 | 1 | 146 | 0 | 0.0000% |
| v124 | 1 | 146 | 0 | 0.0000% |
| v145 | 1 | 146 | 0 | 0.0000% |
| v150 | 1 | 146 | 0 | 0.0000% |
| v156 | 1 | 146 | 13 | 8.9041% |
| v157 | 1 | 146 | 15 | 10.2740% |
| v158 | 1 | 180 | 0 | 0.0000% |
| v159 | 1 | 180 | 0 | 0.0000% |
| v160_v166 | 1 | 146 | 0 | 0.0000% |
| v161 | 1 | 146 | 11 | 7.5342% |
| v169 | 1 | 146 | 7 | 4.7945% |
| v181 | 1 | 145 | 8 | 5.5172% |
| v183 | 1 | 146 | 0 | 0.0000% |

FPS 實測為 23.976×16、29.97×4，包含 145/146/180 幀影片。來源代碼未核實為 game ID；本樣本刻意分散來源且每來源一支，不能估計 full train 的模糊盛行率，不能把 label/FPS/source 差異當因果。

## 時間、容量與記憶體

| 量測 | 值 |
| --- | --- |
| 初次 scanner wall seconds | 5.784127 |
| 每支平均／P50／P90 seconds | 0.279794 / 0.271867 / 0.320827 |
| 每支輸出平均／P50／P90 bytes | 50410.2 / 48396.0 / 58391.2 |
| 20 支 per-video files bytes | 1008204 |
| 目前主 smoke run bytes（含 resume/rebuild snapshots） | 1079174 |
| scanner peak RSS MiB | 128.680 |
| 子程序 peak RSS KiB | 131768 |
| 父＋子 peak RSS 保守上界 MiB | 257.359 |
| 20/20 已完成 resume seconds | 0.075305 |
| aggregate rebuild seconds | 0.060465 |

RSS 由 Linux getrusage 取得；父＋子各自峰值相加是總記憶體的保守上界，**不是同步採樣的 process-tree 峰值**，可能包含 fork 時繼承的 RSS。單 worker、單畫面、單影片最多 10,000 筆分數，沒有保存整個資料集的分數或影像；詳細邊界見方法文件。

## Resume／recovery 與一致性

- 初次 20/20 完成；普通 resume 跳過 20 支、attempted=0，access ledger 未增加。
- 同設定另開獨立目錄重跑同一 20 支，逐影片 frames.csv 位元組完全一致；summary 排除 processing_seconds 後完全一致。
- Derived-only fixture：複製 18 支完成結果、建立 1 支無 completion 的部分輸出、以 mock 空 decoder 產生 1 支 failure。建置 fixture 沒有開資料集影片。
- 普通 resume 保留 incomplete/failed、exit 2；--retry-failures 跳過 18 支，只重試原 smoke 中的 2 支，最後 20/20，與初次科學結果一致。舊 partial、failed completion、decode failure ledger 都保留。
- Unit tests 另覆蓋真正 KeyboardInterrupt、輸出 hash 毀損、atomic rename 失敗、輸入 identity 改變及 concurrent writer lock。
- 相同參數但不同 config bytes hash 的實際 CLI resume 被拒絕（exit 2）；error ledger 留存，video access ledger 不變。
- Aggregate 從 per-video outputs 重建一致；獨立 median/MAD、segment/window、quantile、JSON/Markdown/CSV 檢查通過。

單元測試：v3 15、v2 10、v1 6，全部通過。Syntax/import、CLI help、smoke dry-run、全 train 正向抽樣 guard、全部 held-out ID/path 字串拒絕、seed 重現與 v2 hashes 通過。Full-train 缺确认拒絕；本輪未執行 full-train dry-run，因其也要求禁止由 Codex 使用的確認 flag。

## 存取及原始檔保護證據

| 執行 | 唯一 IDs | guarded opens | probes | decodes | 唯讀 open syscalls |
| --- | --- | --- | --- | --- | --- |
| initial | 20 | 40 | 20 | 20 | 80 |
| repeat | 20 | 40 | 20 | 20 | 80 |
| recovery | 2 | 4 | 2 | 2 | 8 |

三次實際存取涵蓋同一固定 20 個 IDs：初次 20、重跑 20、恢復重試 2；合計 42 probe/decode sessions，各次都有 durable access ledger 與獨立 strace。含 descriptor reopen 共 168 次唯讀 open syscall。**validation/test opens、probes、decodes 均為 0**，沒有 encoder 或 dataset write syscall。
最初 strace capability test 被 sandbox ptrace 限制，未執行 scanner；取得允許後成功追蹤。第一次實際 smoke 嘗試遇 ledger 參數命名錯誤，在影片 open 前停止；修正並新增回歸測試，原目錄保留為 blur_train_v3_smoke_preflight_failed，該 trace 影片開檔為 0。

初始 125 個 tracked files 的 SHA-256 比對：123 個不變；預期修改 development/data_quality/README.md, docs/progress_log.md。Progress log 舊內容以 HEAD 前綴比對保留。20 支原影片 stat identity 前後一致，未計算全資料集影片 hash。原始 manifest/split、v1/v2、baseline、runs 與 frozen evidence 未改；沒有資料刪除。

## 完整 7,193 支的 smoke extrapolation

以初次 wall time /20×7,193，預估 34.67 分鐘；以每支 P50/P90×7,193 分別為 32.59/38.46 分鐘。這些是 smoke extrapolation，**不是信賴區間或實際時間保證**；冷 cache、磁碟、主機負載、失敗及 retry 都會改變時間。

每支檔案容量線性外推約 0.338 GiB；將本次 run 的 logs/多次 snapshots 一起按影片數保守外推約 0.361 GiB。建議預留至少 2 GiB，若保存額外 strace、重跑／重試目錄需再增加。Full-run 預設只寫 access ledger，不產生 contact sheets 或逐影片影像。

正式執行請先 review、commit/push v3，記錄 commit/config hash，依 [tmux 手冊](../../development/data_quality/runbooks/blur_train_v3_tmux.md) 執行 dry-run、scan、detach/attach、監看、resume、retry、rebuild、verify。推薦 output：`development/data_quality/artifacts/blur_train_v3_full_train/`。本次不自動執行。

## 未決政策

Local z 可能漏掉整片持續低品質；低絕對分數只列診斷。沒有人工模糊真值、球／籃框 ROI、真實模糊嚴重度或下游分類效益證據。來源/FPS/紋理/字幕與拍攝情境混雜，品質定義、人工複核、錯誤成本及任何保留／排除／清理策略仍待決定。Scanner implementation complete 不代表資料已清理，也不代表全量結果已完成。
