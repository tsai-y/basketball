# Data Pipeline Integration Gate v1 Protocol

版本：`data_pipeline_integration_gate_v1`  
凍結日期：2026-10-16

## 範圍與停止條件

本 Gate 只驗收 `sampler -> guarded decoder -> reference transform -> explicit normalization -> PyTorch tensor -> metadata`。它不建立或載入 I3D／YOLO／其他模型，不執行 optimizer step、訓練或正式評估，也不讀取 test 影片。A1／A2／B／C、`baseline/`、`reference_c3d/`、`experiment_c/`、既有 `configs/`、固定 manifest／split 與歷史 `runs/` 維持唯讀。

通過本 Gate 只表示資料介面可重現且防護條件成立；不表示 I3D、YOLO、reference 或任何 preprocessing candidate 已證明能改善 Make/Miss 分類。

## 輸入身份

- 固定 manifest：`data/processed/dataset_manifest.csv`。
- 固定 split：`data/processed/split_summary.json`。
- train quality sidecar：`development/data_quality/sidecars/blur_quality_sidecar_v3_1.json` 及其宣告的兩份 CSV。
- Sampler：既有 `quality_aware_sampler_v1`；candidate margin、v3.1 threshold 與 fallback 定義不得改動。
- Transform：既有 `video_clip_transforms_v1` 的 `reference` strategy。
- Gate config 必須保存上述檔案的 SHA-256／run identity；任何不一致在影片 open 前 fail closed。

## Train sampling

每支 train 影片以既有 Quality-aware Clip Sampler v1、global seed `20261009`、指定 epoch 與 clip index 選一個連續 64-frame 半開區間。優先選擇 candidate overlap 為 0 的窗口；沒有零 overlap 窗口時保留 minimum-overlap fallback。已知 5 支 fallback 影片不得刪除，並須回傳 overlap、`fallback=true` 與 `no_zero_candidate_overlap_window`。

不實作最後一秒尾部錨定，也不改 candidate segment、margin 或 v3.1 threshold。

## Validation reference sampling

Validation quality-aware mode 固定為 disabled，且 validation dataset 不載入、不查詢 train quality sidecar。

對 frame count 為 `N` 的 validation 影片：

1. 僅允許 `N >= 64`；短片 fail closed，不重複、補幀或拼接。
2. 固定 `start = floor((N - 64) / 2)`。
3. 回傳連續半開區間 `[start, start + 64)`。
4. `candidate_overlap` 固定為 `null`，`fallback=false`，`fallback_reason=null`。
5. sampler policy/version 固定為 `validation_temporal_center_contiguous_v1`。

此規則只依 immutable manifest row 與 decoder 回報且經驗證的 frame count；沒有 seed、epoch、quality threshold、sidecar或尾部錨定依賴。

## Test 邊界

只接受 `train` 或 `validation` dataset。`test`、`excluded`、未知 split、test video ID、test path 或與 manifest 不一致的 row/path，必須在 path resolve、`stat`、open、probe、decode 之前拒絕。Dry-run、unit tests 與 smoke test 的 test open／probe／decode count 必須為 0。

## Decoder 與 preprocessing

- 影片只以固定 manifest row 授權的 read-only descriptor 開啟；拒絕 symlink、hardlink、path/ID mismatch 與來源 identity 改變。
- 依序解碼至 `end_frame_exclusive`，只保留 sampler 的 `[start,end)`；不可使用不精確 seek 取代 frame identity。
- retained frame indices 必須逐項等於 `range(start,end)`，且正好 64 格。
- Decoder 產生 contiguous RGB `np.uint8 [64,H,W,3]`。
- Reference transform 使用最大 4:3 中心裁切與 OpenCV `INTER_AREA` resize 至 H=192、W=256。
- 禁止 spatial augmentation、gamma、photometric 與 horizontal flip。
- Transform 輸出 contiguous `np.float32 [3,64,192,256]`、有限值、範圍 `[0,1]`。
- normalization 是獨立且顯式的 adapter；v1 僅允許 identity。不得隱含 ImageNet mean/std 或 `[-1,1]`。
- 最終以 `torch.from_numpy` 建立 contiguous `torch.float32` tensor，數值不得改變。

## Sample contract

每筆至少包含 tensor、binary label、video ID、split、source group、shot type、FPS、start/end 半開區間、normalized start、candidate overlap、fallback/reason、sampler policy/version、transform policy/version、normalization metadata，以及 config、manifest、split 與 sidecar identity/hash。Validation 的 sidecar metadata 必須明示 `used=false` 且 identity/hash 為 `null`。

`normalized_start = start / (N - 64)`；若 `N == 64` 則為 `0.0`。

## Gate 驗收

- 同 seed、video、epoch、clip index 得到相同 train frame indices。
- 相同 RGB 輸入得到 bitwise-identical reference tensor。
- tensor shape、dtype、contiguous 與 range 正確。
- decoder retained indices 與 sampler 半開區間完全一致。
- 5 支 fallback 均可產生 sample 且 metadata 正確。
- validation 不載入 train sidecar。
- test row/path 在任何影片 open/decode 前拒絕。
- 短片、錯誤 dtype、錯誤 frame count 與 sidecar identity mismatch 均 fail closed。
- import、`--help`、unit test 與 dry-run 不寫入歷史 `runs/`。
- smoke 只使用少量 train／validation 樣本、使用 PyTorch `DataLoader`、不建立模型，並輸出小型 JSON；其中 test open/probe/decode 必須為 0。

