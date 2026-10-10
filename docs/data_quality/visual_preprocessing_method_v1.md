# Basketball-51 視覺前處理與增強 Pilot v1 方法

本方法只為 full-video Make/Miss pipeline 校準視覺處理與可重用介面，不訓練模型。正式影像增強政策尚未定案。結果見 [Pilot 報告](visual_preprocessing_pilot_v1.md)／[JSON](visual_preprocessing_pilot_v1.json)。

## 四個獨立責任

| 責任 | 本 Pilot 實作 | 不能解讀為 |
| --- | --- | --- |
| Deterministic preprocessing | read-only decode、BGR→RGB、中心最大比例 crop、INTER_AREA resize、float tensor、獨立 normalization adapter | 模型專屬 I3D preprocessing 已定案 |
| Image enhancement | 固定 gamma 0.95：`x ** 0.95` | data augmentation、修復模糊真值 |
| Training augmentation | mild photometric、mild spatial，獨立水平翻轉比較 | validation/test 政策、已證明 label invariance |
| Quality-aware sampling | frozen sampler v1；zero overlap，必要時 minimum overlap fallback | enhancement、刪除影格、人工模糊真值 |

## 名單與選樣

固定主 manifest／split 只讀，sidecar 的原 SHA-256 契約由既有 loader 驗證。選取 48 支 train，每支一個連續 64-frame clip。先加入全部 5 支 `has_clean_64=false`，然後按 Make/Miss × FPS 四格中目前最少的格選取；22 個 affected-clean 與 21 個 unaffected 位置交替，優先目前較少使用的來源，再以 seed=20261010 的 SHA-256 破同分。若格內無可選資料立即失敗，不替換條件。

四格各 12 支，共 Make/Miss 各 24，23.976/29.97 FPS 各 24。實際涵蓋 35 個可用 train 來源群組中的 32 個；來源多樣性為每步貪婪偏好，優先滿足 label/FPS/category 配額，不保證全域最大來源覆蓋，亦不代表真正 game-level split。5 支 fallback 必須全納入，剩餘 43 個名額不能把其餘兩類完全平分。

[新 Pilot manifest](../../development/data_quality/manifests/visual_preprocessing_pilot_manifest_v1.csv) 是衍生選樣紀錄，不修改主資料表。它保存 train ID、label、source、FPS、frame count、category、start/end、overlap、fallback/reason、epoch、clip index、sampler policy、seed 及 seed provenance。重新產生的 CSV 必須逐 byte 相同才允許解碼。Sampler seed=20261009、epoch=0，引用 [原設定](../../development/data_quality/configs/quality_aware_sampler_v1.yaml)，不更動原結果。

## 解碼防護

`PilotGuard` 繼承既有 v3 `Guard`，重用 immutable manifest/split hashes、精確 train allowlist、path/ID 一致性、拒絕 symlink/hardlink 與 read-only descriptor open。新增的 48-ID 限制在任何 video path resolve/stat/open 之前執行。讀取完整主 manifest 的 held-out ID 字串只用於拒絕，不接觸那些影片。

既有 decoder 不是 full-video grayscale 就是與 frozen C3D preprocessing/dataset 綁定；因此新增有界 RGB decoder adapter，而不複製或修改 frozen dataset。OpenCV/FFmpeg 從已授權 descriptor 依序解碼至 end，僅保留 `[start,end)`，停止於第 64 個所選影格。H.264 精確 frame-index 定位所需前綴與真正保留的 clip frames 分別計數；不以不精確 seek 取代 frame 身分。以原 FPS 比較 64 個 decoded timestamps，允許半格誤差，失敗即停止。讀取前後檔案 size、inode、mtime、ctime 等 identity 必須相同。

## Transform 契約與策略

[獨立模組](../../development/data_quality/transforms/video_clip_transforms.py) 接受 `np.uint8 [64,H,W,3]` RGB，拒絕空、短、錯誤 shape/dtype；輸出 contiguous `np.float32 [3,64,192,256]`、[0,1]。沒有 normalization 隱含值、temporal sampling 或模型相依。

- Reference：最大 in-bounds 4:3 crop，中心定位，INTER_AREA resize。原始 320×240 的比例完整保留；其他比例會明確裁切而不強制拉伸或 padding，metadata 保存 crop 與像素 rounding 的 aspect error。
- Photometric：brightness、contrast、saturation 各均勻抽取 [0.95,1.05]；依序 brightness 乘法、以固定 0.5 為 pivot 的 contrast、Rec.709 權重 grayscale 周圍的 saturation。所有參數一個 clip 只抽一次，最後 clamp [0,1]，記錄 clamp 前超界 RGB 元素比例。沒有逐幀 histogram 或 mean 自適應。
- Spatial：從 reference 最大 crop 的線性寬高 scale [0.92,1.0] 抽一次（不是 area scale）；同一個 in-bounds crop 用於 64 格，再 resize。僅保守縮放與平移，不做旋轉、padding 或 vertical flip。
- Horizontal flip：獨立候選每次翻轉，以便清楚比較；並未作為 train 預設機率。metadata 明示左右／慣用手改變，Make/Miss invariance 未實證。
- Gamma enhancement：固定 gamma=0.95；亮部不被拉到 >1，但不等於能復原模糊或增加任務資訊。不引入 CLAHE、sharpening、denoise 或更多候選。

PCG64 seed 由 SHA-256 的前 64 bits 衍生，輸入包含 global seed、video ID、epoch、clip index、transform version 與 strategy。相同 config/library/input/seed 重現相同結果；不同 seed 只要求隨機候選有變化，reference/gamma/flip 應保持不變。每份 sample metadata 保存原 sampler 記錄及完整 transform 參數。版本、config 與 seed 共同提供重播依據；不保證不同 OpenCV/NumPy 版本 bitwise 相同。

## 未來 DataLoader 的介面

```python
# rgb: decoded contiguous RGB uint8 [64,H,W,3]
seed = derived_seed(global_seed, selection['video_id'], epoch,
                    selection['clip_index'], strategy)
x, metadata = ClipTransform(transform_config, strategy)(
    rgb, seed=seed, split='train', sample_metadata=selection)
x, normalization = normalize_tensor(x, mode='identity')
metadata['normalization_adapter'] = normalization
# A future DataLoader may use torch.from_numpy(x); no model is created here.
```

`normalize_tensor` 另外支援 explicit RGB mean/std（輸入單位為 [0,1]）或 affine scale/offset；呼叫者必須依已選 checkpoint 的實際契約設定，不能假定所有 I3D 都用同一組 mean/std 或 [-1,1]。Adapter 不原地修改 augmentation 結果。Output H/W 可配置，192×256 是本 Pilot 的保留比例選擇，並非宣稱相容任何特定 I3D checkpoint。

候選 transform 拒絕 validation/test split。Reference 函式是可重用的 deterministic 純陣列介面；本輪安全 runner 完全不允許 held-out 影片，即使呼叫 reference 也不能存取。

## 診斷與限制

所有 64 格均納入統計。每 clip 指標包括全 RGB mean/std/min/max、luma mean brightness、逐幀 luma std 平均 contrast、OpenCV float HSV saturation 平均、輸出 0/1 endpoint fraction、clamp 前超界 fraction。Endpoint 包含原有飽和／黑色像素，不能當作新增 clipping；有些超界值經最後的 saturation 回到範圍，因此 new clip 僅指最後 clamp 前的超界。

Laplacian variance（ksize=1）及 Tenengrad（Sobel ksize=3，平方梯度平均）以輸出 luma×255 四捨五入 float64 計算，OpenCV 預設 reflect101 邊界。它們不是重算 blur sidecar 或更改 frozen 門檻；其絕對值不能直接與原尺度 gray 指標相比，也不能單獨決定策略。

Temporal 檢查包含重複 frame 的 synthetic 無參數閃爍測試、所有 240 組真實 clip/strategy 的同 seed bitwise 重現、逐幀 brightness 差與相對 reference 的差分殘差。動作、鏡頭切換與裁切內容差異都會影響差分，未設定無參考品質門檻，也不把低差分當作感知品質證明。

9 張 ignored contact sheets 每張呈現原圖 context 與 5 個候選的 4 個時間點，全部 5 支 fallback 加兩類各 2 支。檢查 clipping、空白邊、拉伸、主要人物／籃框裁掉及色彩 artifact；這只是稀疏靜態視覺檢查，不是完整人工連續播放審查。

## 命令與輸出

使用既有 Conda `yolo11` 的 Python。所有命令由 repository root 執行；沒有新增依賴或下載 checkpoint。

```bash
python -B development/data_quality/scripts/audit_visual_preprocessing_pilot_v1.py --help
python -B development/data_quality/scripts/audit_visual_preprocessing_pilot_v1.py --dry-run
python -B -m unittest development/data_quality/tests/test_video_clip_transforms_v1.py
# Only when reproducing into a NEW ignored directory:
strace -f -yy -e trace=%file -o development/data_quality/artifacts/visual_preprocessing_pilot_v1_runtime/repeat.strace \
  python -B development/data_quality/scripts/audit_visual_preprocessing_pilot_v1.py \
  --output-dir development/data_quality/artifacts/visual_preprocessing_pilot_v1_repeat
# Default verifier reads derived files and verifies the existing publication:
python -B development/data_quality/scripts/verify_visual_preprocessing_pilot_v1.py
```

初次固定名單的 `--create-manifest` 僅允許專用新路徑且拒絕已存在檔案。正常 runner 拒絕已存在的 output directory，不會重寫本次或歷史成果；失敗輸出保留調查，不覆蓋重跑。Verifier 的 `--publish` 亦拒絕已有報告；預設 check 模式無寫入。

本次專用 ignored 目錄為 `development/data_quality/artifacts/visual_preprocessing_pilot_v1/`，保存 samples、summary、contact sheets、review、validation；syscall/runtime 證據另存 `visual_preprocessing_pilot_v1_runtime/`。新目錄沿用既有 `artifacts/` ignore 規則，不新增「清理後影片資料集」。Tracked JSON 僅保存 aggregate、provenance 及驗證摘要，不放逐幀像素／指標。新小型 manifest 依本次指定命名放 `manifests/`，沒有複製既有入口。

安全驗證使用 `strace -f -yy -e trace=%file`，所有 path-based syscalls 包括失敗的存取嘗試均檢查，descriptor path 有標註；與固定主 manifest 的 validation/test ID 字串及 Pilot 白名單比較。這證明受追蹤程序的檔案存取範圍；不是系統所有程序的監控。合成測試與 dry-run 另有零影片 trace。

停止於 Pilot。未修改原始影片、主 manifest/split、labels、門檻、sidecar、sampler 歷史結果、frozen baselines；未完成正式影像增強政策、未實作 I3D、未執行 I3D/YOLO/其他模型訓練或評估，未證明 Make/Miss accuracy 改善，沒有 git add/commit/push。
