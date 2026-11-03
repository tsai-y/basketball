# ACA 輸入解析度：單變因可行性檢查

使用同一個未經作者驗證的dense ACA候選（93,163,870參數），比較全畫面160×120與256×192。固定50張uniform endpoint取樣、原片直接INTER_AREA縮圖、RGB/ImageNet正規化、同seed全新ImageNet＋ACA＋配對head、凍結BN、FP32、Adam與有效batch16。無ROI、增強、樣本剔除或標籤修改。不接續已訓練的低解析度權重，避免把初始化與解析度混為一談。

本模組只實作資料與成本／記憶體預檢，**沒有正式訓練或test入口**。通過不等於高解析度成績更好，也不等於正式訓練resume/checkpoint已驗證。舊baseline、ACA、DW3D、extension程式與產物保持唯讀。

CPU準備選擇既有metadata規則固定的4部train，從原片同次解碼建立兩種尺寸，低解析度張量／取樣逐位對上原loader；核對ImageNet初始化、meta形狀與所有來源指紋。只解碼4部train兩次（新paired與舊loader各一次），不解碼validation/test。

GPU預檢依序測兩種尺寸，使用完全相同的初始模型，各做2次有界工程Adam更新；每次4部train各重複4遍，micro1累積至16。重複只用於工程成本量測，不是正式訓練資料擴增。驗證各ACA分支梯度、BN不變、完整Adam狀態、實際顯存及單批時間；工程模型全部丟棄，**不儲存checkpoint或產生validation準確率**。GPU忙時拒絕，絕不停止其他程序；失敗不自動改AMP／batch／尺寸或重試。

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.aca_resolution_v1.test_resolution
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_resolution_v1.probe --cpu-prepare
# 將PREPARATION換成上一步印出的絕對路徑；不得設定CUDA_VISIBLE_DEVICES空值
/home/tsai/miniconda3/envs/yolo11/bin/python -B -u -m development.aca_resolution_v1.probe --gpu-preflight --preparation /ABSOLUTE/PREPARATION --accept-candidate-architecture
```

像素數增加2.56倍不是顯存／時間的保證比例。報告的每輪估計只是四片快取工程計算時間外推，不包含原片解碼、dataloader、checkpoint I/O等，亦非完整epoch實測；不可拿兩次小樣本工程更新的結果宣稱泛化、論文復刻或test90%。是否建立正式訓練版本及其預算需另行決定。

## 2026-10-27 實測結果

30項CPU測試／相關回歸測試通過。CPU準備為 `artifacts/cpu_prepare_vygk24ls/report.json`；GPU成本量測為 `artifacts/gpu_cost_ba0juxok/report.json`，status=`feasibility_passed_not_training_readiness`。在宿主RTX5070Ti16GB上，兩路均通過各ACA分支梯度、凍結BN、Adam完整狀態與有限值檢查。

| 輸入寬×高（均50張） | 峰值allocated，含Adam | 第二次16片工程更新 | 外推一輪train＋validation純計算 |
| --- | ---: | ---: | ---: |
| 160×120 | 約4.71GiB | 4.98秒 | 約40分鐘 |
| 256×192 | 約9.50GiB | 11.05秒 | 約89分鐘 |

峰值為PyTorch allocated，不是整張GPU的總用量；桌面及其他服務另占空間。第二次工程批次計算不含逐參數梯度稽核時間，含forward/backward、gradient clipping、Adam及CPU到GPU輸入傳輸。每輪外推是450次更新加1547片eval估值；eval耗時亦由4片train工程推論估計，並未讀取全validation影片。

若將89分鐘單純乘以5／15輪，純計算約7.4／22.1小時，還需額外的原片解碼、checkpoint寫入等；早停與正式工作負載會改變實際耗時，這不是完整訓練工時保證。尚未建立正式訓練入口或驗證其checkpoint／續跑機制，不得拿此成本報告替舊入口解鎖或直接修改舊設定啟動。

工程GPU輸出只有plan/report，沒有模型checkpoint；CPU的 `engineering_inputs.pt` 只是4片輸入張量快取，不是可訓練權重。未計算新validation/test準確率，未啟動正式訓練。
