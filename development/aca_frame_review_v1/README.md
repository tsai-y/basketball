# 原片與模型實際輸入：24 部 validation 抽幀對照

固定使用 `aca_review_v1/artifacts/review24_ckaqp3hd/` 已收齊的24部人工初檢案例，先R02/R24，再原順序其餘案例。不選新影片，不讀test，不跑模型推論或GPU，不更新模型、標籤、人工欄位或validation成績。原人工作為觀察來源，不把人類約略秒數當成精確事件標註。

每片用guarded descriptor順序完整解碼原片，並重用原ACA dataset得到50張輸入；另外以獨立CPU worker在原主線checkout執行原 `BasketballThreeWindowDataset`，得到原三個64張視窗。ACA由原loader張量反正規化得到的uint8 RGB必須與獨立逐片resize完全一致；主線實際loader的unit-range float32張量必須可逐位由uint8 RGB/255重建，且與獨立參考transform重建的三視窗RGB逐像素一致。任何不一致停止，不悄悄改前處理。

主線worker明確將Python package root設為 `/home/tsai/projects/basketball`，因為主線guarded dataset的archive根目錄固定在原checkout。工作樹與主線相關transform原始碼SHA256必須相同，gate config必須對上正式主線run。所有來源、checkpoint、人工紀錄與程式指紋前後相同，並核對選定原片bytes SHA256前後不變。

輸出每片原始全部影格、ACA50張、主線3×64張的無損PNG分頁；每張圖保留原尺寸RGB，標題/時間/橘框放在像素外，不放大、不銳化、不做ROI、不用生成影像。原片6×4影格一頁，ACA10×5，主線各視窗8×4分兩頁。`key_comparison.png`在人工約略秒數±0.25秒列出原片、最近ACA影格及包含該影格的一個主線視窗。橘框僅提示人工秒數前後0.5秒，不是已確認的事件範圍。

每片 `comparison.json` 保存完整影格索引、frame/FPS時間、最近ACA時間差、主線視窗半開區間及時間涵蓋；`semantic_key_event_presence_confirmed` 保持null，時間有涵蓋不等於球框關係可辨識。主線後續temporal downsampling不在這次RGB對照的驗證範圍。`mainline_rgb.npz` 是可重播的原loader RGB，不是訓練checkpoint。

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.aca_frame_review_v1.test_build
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -u -m development.aca_frame_review_v1.build --build-comparisons
```

每次建立新 `artifacts/frame_compare_<random>/`，總覽 `index.html`、每片獨立 `Rxx/index.html` 與 `report.json`；失敗保留failure與已完成案例，不自動重試或覆寫。影像可供獨立複核R02/R24，但不自動更改原標籤或宣稱因果。
