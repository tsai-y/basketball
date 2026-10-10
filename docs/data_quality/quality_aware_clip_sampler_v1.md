# Basketball-51 Quality-aware Clip Sampler v1

## 範圍與來源

本版把完整 train 的 blur-screening v3.1 `COMPLETE` derived metadata 接到後續 clip 取樣；不開啟、probe 或 decode 影片。來源涵蓋固定 train 7,193 支、1,105,146 frames、18,390 candidate frames、1,798 segments；candidate 只表示低局部 Laplacian，不是真實模糊標註或排除名單。

Tracked sidecar 位於 `development/data_quality/sidecars/`：

- `blur_candidate_segments_v3_1.csv`：1,798 個已排序、不重疊的半開區間 `[start_frame,end_frame_exclusive)`。
- `blur_video_summary_v3_1.csv`：固定 train 每支恰一列，共 7,193 列，含 FPS、frame count、candidate 統計及 16/32/64 clean availability。
- `blur_quality_sidecar_v3_1.json`：來源報告／full JSON／verification、config、manifest、commit、run identity 與 CSV SHA-256。

匯出器 `development/data_quality/scripts/export_blur_quality_sidecar_v3_1.py` 只讀 ignored scanner metadata 與固定 manifest，驗證 `COMPLETE`、來源雜湊、completion/summary hash、train ID 精確集合與 totals 後才輸出；`--check-only` 可逐 byte 重生核對。Tracked sidecar 不含 `/home/tsai/` 絕對路徑，5.3 MB full/verification JSON 保持 ignored。

## 選擇契約

給定 `video_id`、`frame_count`、candidate segments、`clip_length`、`num_clips`、seed、epoch 與 selection mode，合法 start 為 `0 <= start <= frame_count - clip_length`，clip 永遠是原始 frame index 中連續的 `[start,start+clip_length)`。每個 start 的 cost 是該窗口與所有 candidate 半開區間的聯集 overlap frames；v1 candidate margin 固定為 0。

1. 若存在 cost=0 的窗口，只從零 overlap pool 選擇。
2. 否則只從全體 minimum-cost pool 選擇，設 `fallback=true`、記錄 overlap 與 `no_zero_candidate_overlap_window`。
3. Train 預設以 SHA-256 排序；seed material 包含 global seed、video ID、epoch、clip index、policy version 與 start，不使用 Python `hash()`。
4. 多 clip 在 pool 足夠時不重複 start；後續 clip 先最大化到已選 starts 的最小距離，再以 SHA-256 破同分以增加時間覆蓋。pool 不足時才依每個 clip index 的穩定 hash 重複。
5. `clip_length > frame_count`、非正長度、越界、未排序或重疊 candidate segments 均明確失敗；不正規化、不拼接兩段乾淨影格。

設定為 `development/data_quality/configs/quality_aware_sampler_v1.yaml`，目前 I3D Pilot 預設 64 frames、每影片 1 clip、seed 20261009；程式本身不硬編碼 clip length。Validation/test 尚無品質 sidecar，`validation_quality_mode: disabled`；若未來要套用，必須另行設計並凍結規則。

## Metadata audit 與結果

`audit_quality_aware_sampler_v1.py` 支援 config/manifest/segments/video-summary/output、seed、epoch、clip length、clip count、max videos 與 dry-run。它依 label、source group、FPS 分組，列出 16/32/64 clean availability，輸出逐 clip selection、summary、固定 cohort manifest 與 frame-index timeline；既有 output directory 拒絕覆寫。

完整 7,193 支、clip length 64、epoch 0 結果：7,188 支選到零 overlap；5 支進 minimum-overlap fallback，共 overlap 20 frames（各為 1、8、5、5、1）。Clean availability 為 16: 7,193、32: 7,193、64: 7,188，與 scanner publication 一致。結果位於 ignored `development/data_quality/artifacts/quality_aware_sampler_v1/`。

固定 20 支 smoke 為 Make/Miss 各 10、20 個來源，包含 8 支 affected-but-clean、7 支 unaffected 與全部 5 支 64-frame fallback；不依模型結果選樣。[Markdown](quality_aware_clip_sampler_smoke_v1.md)／[JSON](quality_aware_clip_sampler_smoke_v1.json) 保存逐支窗口與純 frame-index timeline；[manifest](../../development/data_quality/smoke_manifest_quality_sampler_v1.csv) 固定 cohort。

## 使用方式

從 repository root：

```bash
python3 -B development/data_quality/scripts/export_blur_quality_sidecar_v3_1.py --check-only
python3 -B development/data_quality/scripts/audit_quality_aware_sampler_v1.py --help
python3 -B development/data_quality/scripts/audit_quality_aware_sampler_v1.py --dry-run
python3 -B development/data_quality/scripts/audit_quality_aware_sampler_v1.py \
  --output-dir development/data_quality/artifacts/quality_aware_sampler_v1_next
python3 -B development/data_quality/scripts/publish_quality_aware_sampler_smoke_v1.py --check-only
python3 -B -m unittest development/data_quality/tests/test_quality_aware_clip_sampler_v1.py
```

後續 I3D DataLoader 應先以 `video_id` 查 sidecar，呼叫 sampler 取得 `[start,end)`，再由既有 decoder 依該連續原始窗口載入 frames；fallback/overlap 應一路記入 sample metadata 與訓練 log。這是下一輪整合界面，本輪沒有實作 I3D、YOLO、augmentation、訓練或評估，也沒有修改、刪除、重編碼任何影片、label、manifest 或 split。
