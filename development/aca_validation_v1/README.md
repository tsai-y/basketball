# ACA validation 預測與固定融合比較

此模組不訓練模型，不讀 test 影片，不更新模型或 BN，不覆寫任何既有實驗產物。不是作者原版 ACA-Net 的驗證。

固定使用 dense 延長訓練第8輪與 DW3D 第4輪的 `best.pt`（各自最低 validation CE 選模），以原有 full-video uniform50、120×160、ImageNet normalization，在同一組1547部 validation 影片推論。模型分別載入，micro-batch=1、workers=2、FP32，無 AMP/TF32，模型輸出 logits 計算類別1 Make 的 softmax 機率。

先檢查 checkpoint schema/plan/progress、資料與程式指紋、原始最小 CE 選模；逐片預測的 confusion、分類指標及 loss 必須重現訓練摘要（loss 容差1e-6），否則失敗，不產出完成報告。推論前後全模型參數與 buffer 內容摘要相同，所有原始checkpoint/run metadata/source SHA256 相同，才視為成功。

固定融合為 `p_make = (p_dense + p_dw3d)/2`，不搜尋權重或門檻；`p_make > 0.5` 判 Make，恰好相等判 Miss，與兩類 argmax 平手取 class0 一致。比較以 video ID 對齊，不按檔案列序猜測。拒絕重複、遺漏、額外、跨split或標籤不一致預測。

```bash
cd /home/tsai/.codex/worktrees/aca-net-reproduction/basketball
# 唯讀 metadata 計畫，無GPU/影片推論
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.aca_validation_v1.analyze --source-root /home/tsai/projects/basketball
# 僅CPU合成資料測試
CUDA_VISIBLE_DEVICES='' /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.aca_validation_v1.test_analyze
# 明確執行 validation GPU 推論；每次建立全新資料夾，不自動重試
/home/tsai/miniconda3/envs/yolo11/bin/python -B -u -m development.aca_validation_v1.analyze --source-root /home/tsai/projects/basketball --execute-validation
```

輸出 `artifacts/validation_compare_<random>/`：兩模型的 `*_predictions.jsonl`、`*_verification.json`、`fusion_predictions.jsonl`、`provenance.json` 和 `report.json`。若失敗則 `failure.json`，沒有成功完成摘要。不產生新訓練 checkpoint。

報告列出 accuracy、macro-F1、CE、confusion、各投籃類型、錯誤互補四象限，以及融合相較各單模型的救回/新增錯誤數。label oracle ceiling 使用真實標籤選對的模型，只是診斷用不可部署上限，不是實際達成成績。所有結果均為開發用 validation，不能當成 test、宣稱統計優越或保證90%；最終模型、權重與門檻固定後另行授權 test 評估。

## 2026-10-26 執行結果

13項 CPU 測試通過；正式推論完成於 `artifacts/validation_compare_vykz5xz0/`，結果以 `report.json` 為準。兩模型均重現原訓練最佳輪的 validation 指標，全模型與 BN buffer 不變，受保護的 checkpoint、run metadata、原程式指紋不變。未讀 test、未進行任何 optimizer update。

| Validation 方法 | Accuracy | Macro-F1 | CE loss |
| --- | ---: | ---: | ---: |
| Dense 第8輪 | 80.61% | 80.39% | 0.44056 |
| DW3D 第4輪 | 80.16% | 80.03% | 0.43035 |
| 固定50/50融合 | 82.22% | 82.07% | 0.39032 |

融合判對1272/1547，比 Dense 多25部（救回60、新增錯誤35），約提高1.62個百分點。兩模型都對1139部、僅 Dense 對108部、僅 DW3D 對101部、都錯199部。若用真實標籤在兩者現有類別判斷間選擇，診斷上限為87.14%，不是可部署成績，也不是其他門檻、第三模型或test表現的上限。因此不宜把這兩模型的簡單融合視為已接近保證test90%的方案；後續較值得比較主線 MixNet-I3D 的 validation 預測與共同錯誤。
