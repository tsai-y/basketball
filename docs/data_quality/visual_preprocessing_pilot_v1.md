# Basketball-51 Visual Preprocessing / Augmentation Pilot v1

本輪完成可重現的 train-only Pilot；正式 enhancement／augmentation 政策尚未定案。

## 組成與解碼

| 項目 | 數值 |
| --- | --- |
| videos | 48 |
| clips | 48 |
| retained_frames | 3072 |
| decoder_read_frames | 5930 |
| prefix_frames | 2858 |
| sources | 32 |
| fallback_clips | 5 |
| candidate_overlap | 20 |
| same_seed_checks | 240 |

- Label：`{"Make": 24, "Miss": 24}`。
- FPS：`{"2997/100": 24, "2997/125": 24}`（23.976／29.97）。
- Category：`{"affected_clean": 22, "fallback": 5, "unaffected": 21}`。
- Label × FPS：`{"Make / 2997/100": 12, "Make / 2997/125": 12, "Miss / 2997/100": 12, "Miss / 2997/125": 12}`。
- Pilot seed=20261010；sampler seed=20261009；epoch=0。
- 所有 5 支 fallback 都保留原始連續 64 格，總 overlap 20；其他 43 支窗口 overlap=0。
- 先納入 5 支 fallback，再以 label × FPS 最少格、交替 category、優先未用來源、SHA-256 破同分選樣。
- 扣除 5 支 fallback 後剩 43 個位置，因此 affected-clean／unaffected 為 22／21；不是偷偷替換。
- 涵蓋 35 個可用 train 來源中的 32 個；逐步優先未用來源的貪婪策略不保證全域最大來源覆蓋。來源代碼不等於已確認比賽 ID；配額取樣不能估計完整 train 品質比例。
- 依序解碼至 end，必要前綴只供精確 frame 定位、即讀即丟；僅保留選中窗口，不 decode 窗口後影格。
- 原始連續 index 與 decoder timestamps 誤差均小於半個 frame；不跳格、拼接或插格。

## 四類處理

1. Deterministic preprocessing：guarded OpenCV/FFmpeg decode、BGR→RGB、最大 4:3 中心裁切、INTER_AREA resize 到 192×256、uint8 THWC→float32 CTHW [0,1]。
2. Image enhancement：固定 gamma=0.95，公式 x^gamma；無自動逐幀估參，不是模糊修復或 data augmentation。
3. Training augmentation：brightness／contrast／saturation 各 [0.95,1.05]；另試線性 crop scale [0.92,1]。每個 clip 共用參數。水平翻轉為獨立方向敏感選項。
4. Quality-aware sampling：沿用 frozen sampler v1 選低 candidate overlap 的連續窗口；candidate 不是人工確認的模糊真值。

## Pixel 與時間診斷

下表為每 clip 指標的等權平均；完整 mean/min/max 在同名 JSON。Brightness 為 RGB Rec.709 權重；contrast 為各幀 luma std 的平均；endpoint/new clip 以 RGB channel elements 計。

| 策略 | Mean | Std | Brightness | Contrast | Saturation | Endpoint % | New clip % | Laplacian | Tenengrad | Δbrightness | Δ殘差 MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gamma_enhancement | 0.422169 | 0.279161 | 0.411668 | 0.264807 | 0.361892 | 2.787881 | 0.000000 | 2326.483883 | 33217.466142 | 0.002865 | 0.000046 |
| horizontal_flip | 0.408991 | 0.280769 | 0.398550 | 0.265771 | 0.374080 | 2.787881 | 0.000000 | 2340.174127 | 33307.339101 | 0.002884 | 0.000000 |
| photometric | 0.408440 | 0.278987 | 0.398227 | 0.264306 | 0.364406 | 2.610354 | 2.610353 | 2325.885130 | 32974.788525 | 0.002829 | 0.000121 |
| reference | 0.408991 | 0.280769 | 0.398550 | 0.265771 | 0.374080 | 2.787881 | 0.000000 | 2340.174127 | 33307.339101 | 0.002884 | 0.000000 |
| spatial | 0.415241 | 0.279592 | 0.404702 | 0.263669 | 0.375136 | 2.156125 | 0.000000 | 2147.756281 | 32795.312595 | 0.002946 | 0.000169 |

- 所有輸出 float32、[3,64,192,256]、range [0,1]，無 padding；保留比例與 aspect rounding 見 JSON。
- crop、flip、色彩參數共用一份 metadata，240 次同 seed 完整 tensor bitwise 比較通過。
- 合成重複 frame 測試檢查不新增參數性 flicker；不同 seed 只驗證預期的 photometric／spatial 變化。
- 真實影片逐幀 brightness delta 與對 reference 的 delta residual 包含動作、鏡頭切換及 crop 內容改變，不能單憑它宣稱無感知閃爍。
- Laplacian／Tenengrad 在相同輸出尺寸、luma×255 上計算，只是診斷；分數增加不等於品質或準確率改善。

## 視覺檢查與決策

- 由 Codex 逐張檢查 9 張 contact sheets（36 個時間點 × source/reference/4 candidates），不是人工標註或連續播放審查。
- Reference 保留 320×240 的全幅 4:3 語義；抽查未見新增 padding、嚴重拉伸、色彩通道顛倒或明顯 halo。原片 v111 自帶上下黑邊，不能將其誤記為 transform 新增空白邊。
- Spatial 平均保留原面積 92.74%，最少 84.89%，最大比例 rounding 誤差 0.333%；抽查主要投籃人物仍可見，但部分邊緣人物、看板與比分區變緊，不能保證未檢查幀的球／籃框不被裁切。
- Photometric 平均 clamp 前超界 RGB 元素比例 2.6104%，最高 36.7377% 出現在 ft1_v111_005842；該片 reference endpoint 原已有 26.6561%，且帶原生黑邊。超界比例不能直接稱為新損失像素比例，但當前參數仍暫緩作預設。
- v111 的 photometric 亮部與黃色場地更亮；gamma=0.95 整體略提亮，未見明顯新 halo，gamma 新增 clamp 為 0，但看不出可據以定案的資訊恢復。
- 水平翻轉可見比分文字反轉，亦改變球場方向與慣用手；保持 Make/Miss 語義只是待驗證假設。
- v166、v111、v169 的抽樣畫面包含原生鏡頭切換；64 格連續 index 與共用 crop/色彩參數已驗證，靜態 contact sheets 不能排除所有感知 flicker。

- 保留：reference 作為研究對照；clip 共用參數、seed 與 metadata 契約。
- 暫留候選：mild spatial，仍需邊緣人物／籃框與人工時序檢查。Photometric 僅保留研究實作，當前參數因 clipping 個案而暫緩採用。
- 尚待決定：gamma enhancement 是否有可見性收益；horizontal flip 改變左右與慣用手，Make/Miss label 不變性尚未實證。
- 拒絕作為本輪正式預設：當前 photometric 參數、自動 enhancement／水平翻轉、sharpness score 單獨決策，以及需求禁止的時序／重度幾何策略。

## 介面、安全與驗證

- `ClipTransform(config, strategy)(rgb_uint8_THWC, seed=..., split="train", sample_metadata=selection)` → `(float32_CTHW, metadata)`。
- `normalize_tensor` 獨立接受 explicit identity／mean_std／affine；未指定 checkpoint，因此不預設 I3D mean/std 或 [-1,1]。可由後續 DataLoader 使用 `torch.from_numpy`，本輪沒有 DataLoader/model 實作。
- reference 介面可供未來固定評估使用；所有候選遇 validation/test split 都拒絕。本輪沒有讀取其影片。
- Syscall trace：48 支 Pilot，validation=0、test=0、名單外=0、write=0。
- Trace SHA-256：`c041d53f4fadfd5bc8ddf3dba10b59409ab90862c0bb3925b0e70704fdcec31f`；檔案位於 `development/data_quality/artifacts/visual_preprocessing_pilot_v1_runtime/run.strace`。
- 驗證：Python syntax（4 檔）、import、兩個 CLI help、dry-run、39 個 unit/regression tests 均通過；測試／dry-run trace 零影片存取；既有 tracked 檔案僅兩份導覽更新。
- JSON／Markdown 由同一 verified summary 產生並逐 byte check；完整驗證記錄見 JSON validation。

## 產物與重現

- [方法、定義與重現命令](visual_preprocessing_method_v1.md)
- [結構化摘要](visual_preprocessing_pilot_v1.json)
- [固定 Pilot manifest](../../development/data_quality/manifests/visual_preprocessing_pilot_manifest_v1.csv)
- [設定](../../development/data_quality/configs/visual_preprocessing_pilot_v1.yaml)
- Ignored contact sheets（9 張）：`development/data_quality/artifacts/visual_preprocessing_pilot_v1/contact_sheets/`；包含全部 fallback、2 個 unaffected、2 個 affected-clean。

## 研究限制與停止點

48 支、單一 epoch、每支一個窗口，不代表完整 train 或未來 checkpoint 的最佳輸入。Contact sheets 為 4 個時間點的稀疏靜態檢查，未完成人工連續播放審查；沒有 NR-IQA 指標或分類 accuracy。

本輪只使用 train 影片；未修改原始影片、label、固定 manifest/split、門檻、sidecar、既有 sampler 結果或 frozen A1/A2/B/C。未完成正式影像增強政策，未執行 I3D／YOLO／其他模型訓練或評估，未下載 checkpoint，未證明任何策略改善 Make/Miss 準確率。停止於 full-video Pilot；不是 pre-release prediction。沒有 git add、commit 或 push。
