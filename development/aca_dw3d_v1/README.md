# ACA-DW3D 候選實驗

這是針對 dense ACA 重建參數量過大而建立的獨立候選，不是已驗證的作者原版 ACA-Net。原本 baseline 與 dense 候選程式、權重和訓練產物均不改寫。

## 架構與修正範圍

輸入為每片 50 張 RGB，張量 `[B,3,50,120,160]`。ResNet50 bottleneck 共 3/4/6/3 個，在窄通道 64/128/256/512 的 conv1 後插入 ACA；ACA 包含 LSTA 與 TSCI，最後使用既有融合與時間/空間 pooling，輸出 Make/Miss 二分類。

只把每個 LSTA 的 gate_control、gate_output 之 3×3×3 Conv3D 設為 groups=C，保留 gate_value 的 dense 1×1×1 通道混合、IN/ReLU、short_term、動態 kernel generator 及 TSCI。共 16 個 ACA、32 個 depthwise 3D 卷積。

| 版本 | 二分類總參數 |
| --- | ---: |
| ResNet50 baseline | 23,512,130 |
| 原 dense ACA 候選 | 93,163,870 |
| 本 ACA-DW3D 候選 | 25,464,286 |

8 類版本為 25,476,580，仍不等於論文約 26.8M；不可為湊數而增加無依據層。Depthwise 是工程假設，尚無作者設定佐證。通道已處於 bottleneck 內，這次不再額外縮減通道。

## 固定訓練協定

- 固定 train 7,193 / validation 1,547；沒有 test 介面，不讀取 test 影片。
- ImageNet V1 backbone + 全新 ACA；分類頭與 baseline 同 seed 初始值配對。不載入已訓練 baseline 或 engineering checkpoint。
- FP32，所有 BN 的 affine 與 running buffers 凍結。Adam lr=1e-4、weight decay=0、clip norm=1；micro-batch=1，累積至有效 batch=16（尾批9片按實際數平均）。
- 最多30輪、patience=5；依最低 validation CE 選 best.pt，不依 test 或最高 accuracy 挑選。
- 獨立 checkpoint schema，拒絕 baseline、dense ACA、synthetic 與 GPU 工程檢查 checkpoint。可手動嚴格續跑，但排程不會自動重試或續跑。

## 驗證與自動接續

CPU prepare 驗證 ImageNet 載入、初始權重有限、paired head、完整尺寸 meta forward、16個 ACA 實際 groups、來源及程式指紋，不解碼影片、不建立 CUDA context。

GPU 預檢需 baseline summary.status=completed，且沒有其他非桌面 GPU compute 程序：測試 CUDA 中斷續跑一致、4片 train 的一次工程更新、全部 ACA 分支梯度、凍結 BN 不變、4片 validation、checkpoint 載入 logits 完全一致。通過後重新從 ImageNet 初始化正式模型，工程權重不進正式訓練。

自動流程由產品排程觸發 `after_baseline --launch-once`，以持久 exclusive claim 限制僅啟動一次；預檢失敗、來源變動、GPU忙碌或訓練異常均不得默默改參數、重啟或換模型。GPU忙碌時可等待下次排程。`artifacts/automatic_launch/state.json` 記錄 preflight/training/completed/failed 及正式 run；`console.log` 保存輸出。程式/設定在 CPU prepare 後必須保持不變，否則拒絕啟動。

唯讀查看狀態：

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_dw3d_v1.after_baseline
```

具體 CPU prepare、GPU 預檢與正式 run 路徑由當次執行產物記錄；不要把 synthetic/engineering 指標當成泛化成績。此候選不保證達到固定 test accuracy 90%。
