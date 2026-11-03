# ACA 256×192：正式訓練入口已準備，尚未啟動

使用者要求停在正式開始前。**本次沒有呼叫 `--start-training`，沒有建立正式 run，沒有安排自動啟動。** 舊 baseline、dense、DW3D、15 輪 extension、三模型 test、解析度成本預檢的程式與產物保持唯讀。這個模組不含 test 入口。

## 定位與固定協定

這是 **dense ACA 概念重建的高解析度二分類候選**，不是作者原版，也不是三模型融合的重新訓練。相對既有低解析度 dense 實驗，模型計算圖與參數量不變，輸入從寬×高 160×120 改成 256×192。從原始 320×240 影片直接縮圖，不放大舊快取，不裁 ROI、不加 YOLO、不增強、不刪除或改標籤。

- 輸入 `[B,3,50,192,256]`；全片取樣 `floor(i*(N-1)/49)`，RGB、INTER_AREA、ImageNet V1 mean/std。
- 新鮮 ImageNet V1 骨幹、新 ACA、同 seed 配對二類頭；**不讀任何訓練過的低解析度權重或工程權重**。
- 模型 93,163,870 參數，16 個 ACA；dense 3D gating、LSTA、TSCI 保持原實作。
- seed 20261025；先沿用總上限 **15** 輪、patience **5**，不是額外 15 輪；此預算尚待正式開始前由使用者確認，並非論文的 30 輪。
- Adam lr=1e-4、weight decay=0、無 scheduler、FP32、無 AMP/TF32；micro 1 累積至 effective 16；最後一批 9 部也正確平均。
- 所有 BN affine 與 running buffers 凍結，包含新 TSCI BN；其他參數訓練。workers=2、spawn、pin memory。
- 原固定 train 7193、validation 1547；每輪 450 更新，上限 6750 更新。選最小 validation CE，平手保留較早輪，不按最高 accuracy 選模。
- `last.pt` 包含模型、Adam moments/step、Python/NumPy/Torch/CUDA RNG、epoch/cursor、當輪累積指標、history、最佳 loss、早停計數。每 25 更新、每輪末或安全中斷點存檔。恢復時不得換設定、程式、切分或尺寸。

資料路徑沿用唯讀的 `ResolutionDataset`：每次原片解碼會同時產生兩種縮圖，但只送 256×192 張量進模型；這個額外 CPU 縮圖不改變模型輸入語意。它與先前工程量測輸入逐位相同，沒有改動舊 loader 全域變數。

## 本次驗收與停止點

CPU：本模組 13 項測試、解析度模組 9 項，共 22 項通過。CPU preparation：

`/home/tsai/.codex/worktrees/aca-net-reproduction/basketball/development/aca_highres_v1/artifacts/cpu_prepare_l92d6as2/`

專用 GPU readiness：

`/home/tsai/.codex/worktrees/aca-net-reproduction/basketball/development/aca_highres_v1/artifacts/gpu_readiness_8v1g98iq/report.json`

status=`passed`。固定 4 部 train 各重複 4 次，只做 **1 次隔離工程更新**；4 部 validation 只檢查前向有限值，不計新 validation accuracy；test 讀取 0。工程 checkpoint 明確標示為 `real_highres_engineering`，正式入口拒絕載入；它與純合成續跑測試 checkpoint 分開標記。

已驗證：來源／初始化／輸入指紋、所有分支梯度有限且非零、凍結 BN 不變、真實高解析度模型／Adam／RNG／progress 存讀逐位相同、logits 往返逐位相同、GPU 合成中斷續跑與不中斷軌跡逐位相同、末批正確、正式 workers=2 loader、舊低解析度與成本報告不能解鎖正式訓練。

此 readiness 的峰值 allocated 約 8.79 GiB（reserved 約 9.19 GiB），不是全程訓練顯存上限；上一份成本預檢約 9.50 GiB。正式啟動前還會檢查 GPU 其他運算、至少 11 GiB 可用 GPU 顯存及 8 GiB 磁碟。失敗不自動改 AMP／batch／尺寸，不停止其他程序。工程產物保留作稽核，但不作初始化。

## 指令

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
# 預設：只列 plan，絕不啟動
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_highres_v1.train
# CPU 驗收
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.aca_highres_v1.test_highres development.aca_resolution_v1.test_resolution
```

以下是**未執行**的正式啟動命令，只供之後另行授權。不能把「準備入口」解讀成授權啟動：

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
/home/tsai/miniconda3/envs/yolo11/bin/python -B -u -m development.aca_highres_v1.train \
  --start-training --accept-candidate-architecture --confirm-formal-training \
  --readiness-report /home/tsai/.codex/worktrees/aca-net-reproduction/basketball/development/aca_highres_v1/artifacts/gpu_readiness_8v1g98iq/report.json
```

不設定空的 CUDA_VISIBLE_DEVICES。輸出只會建立本模組 `artifacts/aca_highres_train_<unique>/`。若中斷，之後以同一命令加上 `--resume /ABSOLUTE/HIGHRES/RUN`；只能恢復本模組相同計畫的 `last.pt`，不能拿 best.pt 重設 optimizer／早停。單程序 flock 防止同時啟動兩個正式 worker。

程式、設定、資料或環境指紋若變動，現有 readiness 會失效，必須重新 prepare 與 preflight，不能手動改報告 hash。新 prepare 命令為 `--prepare`，GPU 驗證為 `--preflight --preparation /ABSOLUTE/NEW/PREPARATION --accept-candidate-architecture`，兩者均不接續呼叫正式入口。

## 論文差異與解讀

完整對照見同目錄 [PAPER_COMPARISON.html](PAPER_COMPARISON.html)，已重新使用 PDF 閱讀流程檢查論文 Figures 1–3、Eqs.1–8、§3.3 與 Table 3。作者：Zhang et al., 2024，DOI 10.3389/fnbot.2024.1471327。

關鍵結論：拓樸概念接近，但參數量、標籤任務、切分、輸入、訓練預算和未驗證實作細節不同，不能稱為精確復刻，也不能把論文八類 92.05% 當成本版二分類應達分數。升解析度不會自動修正 93.16M 與 26.8M 的差異。

高解析度構想在查看低解析度 test 成績前已有工程準備，但現在開發過程已接觸同一 test 結果，因此整個 test 已不是全專案未見 holdout。此入口只用 validation 選模；是否另做 test、採何種獨立驗證，須另外決定。本輪不重測或依 test 調參。

## 時間風險

沿用先前 4 片快取成本外推：約 89 分鐘／輪的 train＋validation 純計算，15 輪約 22.1 小時；還不含正式原片解碼、checkpoint I/O 和工作負載差異，早停也會改變總工時。這不是完整 epoch 實測或完成時限保證。高解析度是否改善結果，正式訓練之前仍未知。
