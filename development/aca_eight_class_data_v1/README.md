# 固定基準資料流程（2026-11-15）

目前使用 `foundation.py`：保留整段原片、原生幀序均勻50幀、全圖160×120及既有ImageNet RGB正規化。RGB三通道是顏色，沒有另行抽取球員、球或籃框特徵。

原資料來源分組不變，套用既定八類歧義隔離後為7192 train、1547 validation、1538 test、34 excluded。八類順序為2p0、2p1、3p0、3p1、ft0、ft1、mp0、mp1；二類對映為0、1、0、1、0、1、0、1。test僅保留metadata，loader拒絕解碼。

`preprocessing.py` 是先前暫存的虛擬25Hz候選，不是此基準的執行入口。沒有修改歷史訓練程式，也沒有將新loader接上任何訓練排程。

## 使用

在儲存庫根目錄執行Python：

```python
from development.aca_eight_class_data_v1.foundation import FoundationDataset

dataset = FoundationDataset(source_root="/home/tsai/projects/basketball", split="train")
item = dataset[0]
# item["tensor"]: CPU float32，shape [3, 50, 120, 160]
# item["class_id"]: 八類0..7；item["binary_label"]: Miss=0 / Make=1
# 另含video_id、source_group、split、original_class、解碼metadata
```

八類與二類標籤同時輸出，但未宣稱已由人工逐片裁定真值。來源統計卡、特寫、遮擋保留；固定16支的觀察另存品質註記，沒有建立全資料內容品質標籤。

## 驗收入口

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.aca_eight_class_data_v1.test_foundation
```

全量驗收模組為 `development.aca_eight_class_data_v1.audit_foundation`。只允許最多4個CPU worker，完整解碼train／validation、檢查EOF及輸入張量，保存來源／張量SHA；遇錯停止並保存failure.json，不自動重試或修正。輸出目錄必須是新的isolated artifacts子目錄，不能覆蓋已有驗收。

固定契約見 `docs/research/specs/input_foundation_v1.json`；進度及凍結結果見 `docs/research/INPUT_PROCESSING_FOUNDATION_STATUS.md`。此為專案基準資料處理，不代表作者精確前處理、人工品質驗收或模型成果。
