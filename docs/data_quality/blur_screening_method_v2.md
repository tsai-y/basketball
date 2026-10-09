# Basketball-51 自動模糊校準方法 v2

本輪是 train-only 工程校準，不是正式清理。原始資料、label、固定 manifest/split、A1/A2/B/C、Pilot v1 全部保持不變；不建立清理影片、不刪格、不做訓練、特徵擷取或模型下載。方法只使用 Laplacian variance 與 Tenengrad，沿用 v1 的原始解析度評分定義。

## 選樣與封存

輸入為固定 `data/processed/dataset_manifest.csv` 與 `split_summary.json`，使用 Gate 1A 固定 hash 與 v1 `TrainGuard`，每次 probe/decode 先驗 train ID/path，再以唯讀 FD 開啟。validation/test 的 guard 測試只使用 manifest 字串，opener 被 mock 封鎖。沒有為了 FPS 覆蓋而掃描未選影片。

先統計全部 35 個 train source groups 的可用數與 Make/Miss 數，再以 Pilot v1 的來源 `both_low` 比例分層：0 為 low；正值中位數以下（含）為 middle，其餘 high。這是 v1 單影片候選比例的抽樣提示，不是來源品質真值。v1 未觀測的 5 個來源列為 unobserved，這次不從它們選取；此為目的性校準樣本，不估計 train 的模糊盛行率。

固定 seed 20261009，以 SHA-256(`seed:source:source_group`) 排序，high 強制包含 v184，各層取 4 來源。low 層依 v1 已知 FPS，各選 2 個來源；v1 FPS 只是提示，新樣本仍須逐片 probe。每層枚舉 2/2 的來源分配，優先使 v1 FPS 提示在 calibration/verification 之間平衡，SHA-256(`seed:subset:逗號連接的排序後calibration來源`) 破同分。來源不得交集。

每來源依 SHA-256(`seed:video_id`) 排序，各取 2 Make、2 Miss；不足才由同來源剩餘候選補至最多 4 支，若來源不足則記錄限制，不擴展其他來源補足數量。本次無不足。全部樣本先固定，再 probe/decode；解碼失敗不得替換，記錄 selection/failure/decode status 並停止門檻擬合。

## 固定校準畫面與數值定義

循序解碼各選定影片，不 seek、resize、crop、resample。OpenCV CAP_FFMPEG，threads=1、OpenCL=false、optimized=false。BGR uint8 轉灰階 uint8，再轉 float64 [0,255]；所有合成操作在此灰階 float64 上進行，不重新 uint8 量化、不重新編碼。

有效 frame range 為成功且與 metadata frame count 一致的 `[0,N−1]`。位置 10/30/50/70/90%，index=`floor(position*(N−1)+0.5)`，不依分數挑畫面。保留 index、比例、decoder timestamp、index/FPS 估計 timestamp 與可信狀態。Decoder 時間須有限、嚴格遞增、與 CFR 估計差不超過半格、avg/r FPS 相同；否則只採 index 作權威位置，不將估計時間當 PTS。

- Gaussian：sigma 0、0.5、1、1.5、2、3；radius=`ceil(3*sigma)`，kernel=`2*radius+1`；sigmaX=sigmaY。sigma 0 直接回傳原圖，沒有濾波操作。
- Motion：length 3、5、9、13；angle 0/45/90/135 度。在 length×length 奇數網格，以中心為原點，沿方向放置 length 個等距（1 pixel）樣本，用 bilinear splatting 分配到鄰近網格，再除以核總和。0 度向右、90 度向下；核中心對稱且總和為 1。斜角核的網格離散化與水平/垂直核不同，因此跨角度差異同時反映影像方向與核離散化，不可全歸因於畫面。
- 全部合成濾波與導數：`BORDER_REFLECT_101`、float64。Laplacian ksize=1 population variance；Tenengrad Sobel ksize=3 的 mean(Gx²+Gy²)，scale=1、delta=0，不裁切邊界。
- relative=`blurred/original`；log drop=`log(blurred)−log(original)`。分母/對數不合法時保留 undefined，不捏造靈敏度；本次全部有效。正規化用的 score epsilon=1e−12 與這些精確 paired ratios 分開。

sigma、length 僅為可重現的工程校準尺度，不是真實世界的模糊嚴重度標準。合成操作在記憶體中進行，落地只有分數、metadata 與報告。

## 靜態敏感度、組合與來源混雜

每張原畫面個別計算 severity-score Spearman：Gaussian 含 sigma 0 共 6 點；motion 每 angle 各自以 length 3/5/9/13 共 4 點。對每條曲線記錄非增加、嚴格下降、下降步驟比例、增加步驟數；容忍度=`1e−10*max(1,abs(first_score))`。motion 統計單位是「畫面×angle」，不可冒充 4 倍獨立畫面。relative/log drop 為單畫面的固定比例／單調轉換，與 absolute score 的 Spearman 相同。

雙指標無權重校準的工程組合為 relative scores 的幾何平均，只檢查敏感度。另一對照是 calibration 原始 anchor scores 的 P10（linear quantile），分別比較 Laplacian、Tenengrad、AND、OR 的絕對低分反應；verification 沿用原 P10。P10 不是刪除門檻，原始畫面也不是「清晰」真值。

每個 subset、family、severity、angle 分別輸出 source/FPS/label 的 absolute、relative、log drop 的 P10/median/P90。互補性包含 paired log-drop Spearman、絕對低分旗標分歧，以及時間測試的兩單指標與 AND/OR 比較。

來源分析先把每影片 5 個 anchors 聚合為影片中位數，再對 log(影片原始中位數) 計算單因子 eta²（組間平方和／總平方和）。另對每 transform 的影片 paired log-drop 中位數計算 eta²；它們是不同 response，不應直接當成「消除了多少混雜」的因果百分比。保留各來源內 Make−Miss log-score 差，以及 source+label、source+label+FPS 的 design rank。無 p-value、無獨立影格假設，也未量測鏡頭尺度、字幕、主體清晰度或場景紋理真值。

## 合成時間區段與 local robust normalization

每影片在 30% 與 70% 中心位置分開注入（每案例只有一段，案例間不疊加）。duration 0.25/0.5/1 秒；長度=`ceil(duration*FPS)`，中心 index 與 anchor 相同取整，start=`clip(center_index−length//2,0,N−length)`，區段為 `[start,start+length)`。保存實際 frame count/FPS 秒數，不能把要求秒數當成精確實際秒數。

Gaussian sigma 0.5/1.5/3；motion length 5/13 × angle 0/45/90/135，共 11 transforms × 3 durations × 2 positions=66 案例/影片，另有 1 原始 control。逐格灰階影像只保留單影片於 RAM；同 transform、同 frame 的分數可以重用。每案例以 transformed frame scores 取代注入範圍，範圍外保留原始分數，等價於在 frame stream 注入後逐格評分，不輸出重編碼影片。

對**每條觀測串流本身**（原始或注入後）重算：

```text
log_score(t) = log(max(score(t), 1e-12))
rolling_median(t) = median(log_score(window(t)))
rolling_MAD(t) = median(abs(log_score(window(t)) - rolling_median(t)))
local_z(t) = (log_score(t) - rolling_median(t)) / max(1.4826*rolling_MAD(t), 0.05)
```

window 為 `ceil(3*FPS)` 後上調為奇數（本次 73/91 frames）；中心對齊，影片兩端截短，不跨影片，不把每一格各自的 rolling median 放入 MAD 計算。epsilon 0.05 是 log-score 單位的尺度下限，避免近乎零 MAD 爆大。注入後重算局部統計，不使用乾淨原始序列作 oracle baseline。此為離線中心窗偵測，使用未來 frame，不能聲稱即時可用。

四策略為 `z_L <= threshold`、`z_T <= threshold`、AND、OR；比較共用一個 z threshold，AND/OR 不另外學權重。連續低分 run 小於 minimum duration 則移除旗標，不合併 gap、不做形態學操作。

預先固定搜尋網格：z −1/−1.5/−2/−2.5/−3/−4/−6；minimum duration 0/0.08/0.16 秒，共 21 組/策略。只用 calibration，優先最大化 source-macro frame F1，受 calibration control flag rate ≤5% 約束。若無可行解就停止，不能用 verification 放寬。精確同分時依較低 control flag rate、較嚴 z、較長 minimum duration 決定。每策略獨立選門檻，保存 `frozen_thresholds.json` 後才載入 verification 的時間分數；verification 不重新選門檻。

F1 先每案例算，依序案例→影片→來源平均，避免不同長度影片與來源數量改變權重。66 案例等權，motion 有 48、Gaussian 有 18，因此 overall 是此工程 grid 的加權結果，另報 family 分開結果。Control 不計入 positive-case F1，只計旗標比例。

事件評估以單一真實注入區段與預測連續 run 的最大 IoU，≥0.5 才算檢出；匹配之外的 runs 記為 injection-relative false positive segments。輸出 frame precision/recall/F1、事件 recall、IoU、未注入範圍旗標率、匹配起訖時間誤差。邊界統一採半開 frame endpoints，秒數採 index/FPS。起訖誤差只在成功匹配事件計算，不能忽略未檢出比例來宣稱全部定位準確。

Control 可能已有真實模糊；「未注入旗標率」與注入外旗標不是真實世界 FPR。注入的瞬間邊界、人為均勻全畫面模糊有利於偵測，不代表漸變、局部球體模糊、壓縮、鏡頭變換或噪聲環境的效果。持續整片的低品質也可能被影片內 normalization 消去。

## 產物與驗證

本機根目錄 `development/data_quality/artifacts/blur_pilot_v2/` 已由既有 ignore 規則保護；所有入口拒絕覆寫。逐影片 `.npz` 只有 scores/local_z/rolling statistics/timestamps，沒有影像。v2 程式引用既有 guard，不修改 v1 scripts 或報告。

獨立 verifier 從 derived arrays/CSV 重算全部 local z、calibration 84 個門檻候選、全部偵測、區段、relative/log drops、Spearman，核對來源分隔、閾值凍結與固定 anchors。syscall trace 核對每筆資料集開檔皆為指定 train，且唯讀、無 encoder。初始 113 個 tracked files SHA-256 全部須保持一致。只 hash 已有文字/程式與衍生產物，不 hash、probe 或掃描其他資料集影片。

可重現命令、產物與結論見 [v2 入口](../../development/data_quality/README_v2.md) 與 [v2 報告](blur_screening_pilot_v2.md)。
