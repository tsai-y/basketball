# 更正標籤的獨立載入介面

本模組固定使用 `diagnostic32_label_revision_v1`，只將已由使用者裁定的第15支train目標由3p1改為2p1。原片ID、原始類別與舊程式保留；有效class_id來自固定雜湊清單。

CPU驗收已通過：[報告](/home/tsai/projects/basketball/outputs/01_current/input_diagnostic32_20261119_v1/label_revision_v1/cpu_acceptance_v1/README.md)。未啟動訓練，沒有模型或optimizer入口。

## 使用方式

從ACA複刻工作樹執行。以下是接入範例，不會自動啟動訓練：

```python
from torch.utils.data import DataLoader
from development.adjudicated_input_v1.adapter import (
    AdjudicatedDataset, AdjudicatedCollator, LabelRevision,
    VERSION, MANIFEST_SHA, require_revision_identity,
)

# 只查標籤：不開啟影片。
revision = LabelRevision()
record = revision.record('3p1/3p1_v150_003223_x264.mp4', 'train')
assert record['original_class'] == '3p1'
assert record['effective_class'] == '2p1'
assert record['class_id'] == 1

# 建立載入器不會解碼；迭代時才會逐片核對來源並載入RGB。
dataset = AdjudicatedDataset(
    source_root='/home/tsai/projects/basketball', split='train',
    video_ids=['3p1/3p1_v150_003223_x264.mp4'],
)
loader = DataLoader(dataset, batch_size=1, shuffle=False,
                    num_workers=0, collate_fn=AdjudicatedCollator())
```

`targets`只取有效class_id；`records`同時保存original_class、original_class_id、effective_class及資料版本。不要把更正item交給舊`collate_eight_class`，它會因原始類別與目標不一致而拒絕。這是刻意保留的舊版約束。

## 版本與接入邊界

- 清單SHA256固定為 `1d2df00e61e4aa0f8646f03315e24574e77c28c4246c976b58bf54c0b74aa85f`；無任意覆寫標籤或跳過雜湊的參數。
- 只接受train／validation；test在建立介面時即拒絕。資料集的完整預設順序沿用原正式plan。
- 呼叫`__getitem__`會核對來源SHA與原張量SHA；張量仍由凍結FoundationDataset產生。
- 未來新run身分應包含`data_version`、`label_revision_sha256`及`pixel_contract_parent_sha256`。`require_revision_identity`須在恢復權重前呼叫；目前沒有自動修改任何trainer或resume程式。
- 本介面通過CPU驗收，尚未驗證多worker或GPU訓練接入。既有run不會自行採用本模組。

## 驗收與重現

不解碼影片的檢查：

```bash
/home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest development.adjudicated_input_v1.test_adapter -v
```

`cpu_acceptance.py`另包含只解碼已裁定第15支一次的端到端CPU檢查；結果已保存。預設拒絕覆寫既有驗收目錄，避免因重跑掩蓋歷史紀錄。正常使用時直接閱讀既有驗收結果即可。
