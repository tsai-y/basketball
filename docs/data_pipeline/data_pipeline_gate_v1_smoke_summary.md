# Data Pipeline Integration Gate v1 Smoke Summary

日期：2026-10-16  
結果：**SMOKE PASS**

## 結果

以 PyTorch `DataLoader(num_workers=0)` 讀取 6 支 train 與 2 支 validation，共 8 個 sample。所有 tensor 均為 contiguous `torch.float32 [3,64,192,256]`，值域 `[0,1]`；decoder retained indices 逐項等於 sampler 的半開區間。

Train cohort 包含全部 5 支 minimum-overlap fallback，candidate overlap 依序為 1、8、5、5、1，另含 1 支 overlap=0 sample。Validation 兩支均使用 `validation_temporal_center_contiguous_v1`，其 sample metadata 明示 train sidecar `used=false`。

| Split | Samples | Video opens | Decodes | Train quality sidecar |
| --- | ---: | ---: | ---: | --- |
| train | 6 | 6 | 6 | used |
| validation | 2 | 2 | 2 | not used |
| test | 0 | **0** | **0** | not used |

沒有建立模型或 optimizer，optimizer step 為 0；沒有下載權重、訓練、正式評估或 test access。Runtime JSON 暫存於 `/tmp/data_pipeline_gate_v1_smoke_20261016.json`，SHA-256 為 `cb0d08bb205571cfd5e6b188e00ce61e4b78bc36c0283e7b327a08febfb555be`；tracked machine summary 見同目錄 JSON。

## 驗證環境

Python 3.10.20、PyTorch 2.11.0+cu128、OpenCV 4.13.0、NumPy 2.1.3、PyYAML 6.0.3。Gate config SHA-256 為 `4b5c4f60d952c3fe2aefb3d435de9e3a6b05e892deef241b9bab3657dc215541`。

## 結論邊界

本結果只證明資料管線的 deterministic sampling、bounded decode、reference transform、identity normalization、PyTorch tensor 與 metadata 契約可整合並通過 smoke。它不代表 I3D、YOLO 或任何 preprocessing candidate 已證明有效，也不提供分類 accuracy 證據。

