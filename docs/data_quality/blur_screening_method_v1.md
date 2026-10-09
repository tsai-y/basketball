# Basketball-51 自動模糊品質篩檢方法 v1

本方法只產生 train-only 探索性品質分數與 candidate flags，不是正式資料清理。研究仍為 full-video Make/Miss classification。A1/A2/B/C frozen、Gate 1A/1B completed、Gate 1C 與原 Experiment D paused；Pre-release 與 SPL 延伸不在範圍。沒有影像增強、特徵擷取、模型融合、訓練、評估或 checkpoint loading。

## 輸入、防護與取樣

唯一身份來源為 `data/processed/dataset_manifest.csv`；群組 split 與 label counts 交叉核對 `data/processed/split_summary.json`。重用 Gate 1A 工具的固定 SHA-256、safe-relative path、allowlist 比對與唯讀 guarded descriptor，不修改 Gate 1A。新 TrainGuard 只保存 train rows，補上 validation denylist；每次 probe/decode 都先驗 exact train ID/path，不接受絕對影片路徑、未知 ID、路徑替換或被改成非 train 的 row。拒絕 symlink、hardlink、非一般檔案；唯讀 FD 交給 ffprobe/OpenCV，避免解碼器重新按未授權名稱選檔。這是單一可信 repository 執行環境的防護，不宣稱可抵禦同時控制檔案系統的攻擊者。

優先讀取 Gate 1A 原 cache，若存在需比對 Gate 報告的 cache hash，僅取 train records。此次 cache 已不存在，Gate 1A 公開報告只有彙總；因此先決定樣本，再只 probe 樣本。重用 Gate 1A **train** FPS 群組資訊作覆蓋檢查，不使用 validation/test metadata 分布制定選樣、門檻或結論。

固定 seed `20261008`，規則 `balanced_label_unique_source_sha256_v1`：由 video_id 排序的 train pool 開始，交替 Miss/Make；每輪優先目前入選數最少的 source_group，同分依 SHA-256(`seed:video_id`) 再依 video_id 排序；取 30 支後依 video_id 排序。這使類別各 15 支、來源盡量多樣，沒有模型預測、test 結果或人工選片。FPS 未知時不在選樣前掃描；probe 後檢查是否涵蓋已知 train FPS，若未涵蓋則標記 incomplete，不因分數換片。Pilot 不是完整 train 的比例代表樣本，source_group 不是已驗證的 game ID。

所有數值、時間門檻、量化方式及主要旗標由 [設定檔](../../development/data_quality/configs/blur_pilot_v1.yaml) 控制。`--seed` 只改取樣；CLI 不支持重新分割資料或完整 train 執行，上限 40 支。原 manifest、label、split 不改寫。

## 固定解碼及數值契約

- Metadata：ffprobe 僅讀 selected train 的 stream/format metadata，不 `-count_frames`；保留 avg/r FPS、nb_frames、stream/format duration、解析度等原值。實際版本寫入報告與 runtime_logs。
- 解碼：OpenCV `VideoCapture(..., CAP_FFMPEG)` 循序解碼整支原始影片，BGR uint8，無 seek、抽格、時間重採樣、crop 或 resize；單執行緒、OpenCL 關閉、OpenCV optimized dispatch 關閉。
- 灰階：`cv2.cvtColor(BGR, COLOR_BGR2GRAY)` 先得到 uint8 [0,255]。概念權重為 B 0.114、G 0.587、R 0.299；實際 uint8 取整依記錄的 OpenCV 實作。轉 float64 [0,255]，不除 255、不作額外正規化。
- 全畫面原始解析度計分；此次均 320×240。計分後才為 contact sheets 產生記憶體內 JPEG previews；preview 的 JPEG 壓縮及版面縮放不參與分數計算，不生成或重新編碼任何影片。
- 所有導數以 float64 計算，scale=1、delta=0、`BORDER_REFLECT_101`（不重複邊緣像素的鏡射）。包含邊界像素在內，以全部 H×W 像素平均，沒有 ROI 或邊緣裁切。

設灰階影像為 I，像素數 N：

1. **Variance of Laplacian**：`cv2.Laplacian(I, CV_64F, ksize=1)`，核為 `[[0,1,0],[1,-4,1],[0,1,0]]`。令 L 為輸出，分數為 `sum((L - mean(L))²) / N`，population variance，ddof=0。
2. **Tenengrad / Sobel gradient energy**：`cv2.Sobel(I, CV_64F, dx=1,dy=0,ksize=3)` 及 dx=0,dy=1。Gx 核 `[[−1,0,1],[−2,0,2],[−1,0,1]]`，Gy 為其轉置；分數 `mean(Gx² + Gy²)`。不開根號、不額外設 gradient cutoff。

分數依解碼、灰階數值範圍、空間尺寸、導數核及 border 而變；不能跨其他實作直接套用。

## 時間、解碼失敗與可追溯性

每筆記錄 video_id、零起始 frame_index、decoder timestamp、timestamp_status、獨立的 frame_index/FPS 估計時間、兩分數、decode_status、四策略 flags、主要 candidate_flag、觸發策略 flag_reason 及 threshold_version。

Timestamp 取 OpenCV `CAP_PROP_POS_MSEC / 1000`；只有全部有限、嚴格遞增、与 index/avg_FPS 誤差不超過半個 frame 且 avg_FPS==r_FPS 時，保留為 `decoder_monotonic_cfr_consistent`。否則 timestamp 留空、以 frame index 為權威位置並單獨保存 estimated_timestamp，不能把估計當真實 PTS。這是 decoder/CFR 一致性檢查，沒有獨立逐格 ffprobe PTS 認證，不保證任意 VFR 來源可用。

逐格讀取到 false 後，解碼數必須等於 metadata nb_frames；不相等記為 `frame_count_mismatch`，保存獨立 decode status、停止門檻擬合並標記未完成。沒有把 decode failure 當低分、沒有猜測失敗影格位置、没有跳過失敗片再補樣本。即使 frame count 相等，也不宣稱檢出全部 codec error concealment。

## 探索性候選規則

僅將 Pilot 成功解碼的 train 影格合併，每影格等權，兩指標各取 P10（NumPy linear quantile：排序位置 `(n−1)q` 线性插值）。較長影片因此權重稍高，報告按 FPS 分組檢查影響。沒有外部固定模糊分數，也沒有人工真值校準或 test 調參。

| 策略 | 候選定義 |
| --- | --- |
| laplacian_low | Laplacian ≤ Pilot Laplacian P10 |
| tenengrad_low | Tenengrad ≤ Pilot Tenengrad P10 |
| both_low | 同一影格兩指標均 ≤ 各自 P10；主要 candidate_flag |
| smoothed_both_low | 各影片兩分數分別中心 median 平滑，皆低於上述 raw P10，且連續時間至少 0.20 秒 |

平滑窗為 `ceil(0.25×FPS)` 再上調為奇數，此次兩 FPS 群組對應 7 或 9 frames；邊界截短 window，沒有補值、不跨影片。平滑後仍使用 raw 分布 P10，便於比較同一門檻的持續性；不重新 fit 另一套門檻。連續區段長度以 `n/FPS` 判定；不跨 gap 合併，不按分數事後調整參數。

每策略輸出所有 candidate 與 unflagged 區段的包含端點 frame/timestamp、frame count、名義持續秒數；影片摘要有候選比例、最長候選區段、未標記區段數/總幀數/最長幀數，及 16/32/64 連續滑動窗數 `sum(max(0, run_length−W+1))`。未標記並不等於真實可用；窗口只提供可行性資訊，不決定後續模型輸入長度。

## Contact sheets 與限制

每指標按分數、video_id、frame index 穩定排序，在 P05/P50/P95 附近各取 3 個不同影片的畫面，合計每張 sheet 9 tiles、兩張 18 tiles。每 tile 顯示 video ID、frame/timestamp、兩分數、主要 candidate_flag、門檻版本。圖片與逐影格分數均為 ignored 本機產物；用於檢視演算法反應，不是人工逐筆清理任務、不是 ground-truth labels。

兩項指標只能描述全畫面局部亮度變化與邊緣強弱。低紋理畫面可能低分；運動模糊與失焦不一定能可靠區分；高紋理背景可能掩蓋主體模糊；全畫面清晰不等於球或投籃者清晰；分數不是人工模糊真值。壓縮、噪聲、字幕、場景與鏡頭尺度也可能影響分數。不能用本 Pilot 宣稱清理會提高 Make/Miss 分類表現。

## 執行、保護與驗證

執行方式見 [README](../../development/data_quality/README.md)。預設所有產物先寫到新的 ignored `development/data_quality/artifacts/blur_pilot_v1/`；存在則拒絕，不提供 force。報告與 Pilot manifest 經驗證後才另存為 tracked 候選檔；不寫 frozen 路徑。程式重用 Gate 1A metadata/guard；既有模型 Dataset decoder 會建 tensor、套模型前處理且接受其他 splits，因此這裡另立原始解析度品質 decoder，不匯入模型 Dataset。

安全測試涵蓋所有 manifest 中 validation/test/excluded ID 与偽裝成 train ID 的非 train path、未知/absolute/traversal path、選樣重現與解碼失敗分離，拒絕測試 mock 掉實際 opener。正式執行加 syscall trace，獨立從 trace 對照 manifest 與選定 ID，核對唯一影片、probe/decode 次數、唯讀開檔與無原始資料 mutation。CSV/JSON/Markdown、逐影片總數、區段與旗標由 derived artifacts 重算，不再開啟影片。只雜湊文字輸入、設定、程式及產物，不掃描或雜湊整個資料集。
