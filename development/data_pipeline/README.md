# Data Pipeline Integration Gate v1

本目錄是獨立於 frozen baseline、既有 `configs/` 與歷史 `runs/` 的 PyTorch 資料管線入口。它重用：

- `development/data_quality/samplers/quality_aware_clip_sampler.py`
- `development/data_quality/transforms/video_clip_transforms.py`
- v3.1 train-only quality sidecar
- Gate 1A 的固定 manifest/split allowlist 與 guarded read-only open

本 Gate 不包含模型、optimizer、訓練或 test 影片存取。

## 結構

- `configs/data_pipeline_gate_v1.yaml`：完整凍結的 policy、input path 與 identity。
- `pipeline.py`：PyTorch `Dataset`、validation center sampler、guarded sequential RGB decoder 與 sample contract。
- `scripts/run_data_pipeline_gate_v1.py`：無影片 dry-run 與 6 train + 2 validation smoke。
- `tests/test_data_pipeline_gate_v1.py`：reproducibility、fallback、validation/test boundary 與 fail-closed tests。
- Protocol：[Data Pipeline Integration Gate v1](../../docs/data_pipeline/data_pipeline_integration_gate_v1_protocol.md)。

## 環境

沿用既有 Conda `yolo11` 環境；本次驗收使用 Python 3.10.20、PyTorch 2.11.0+cu128、OpenCV 4.13.0、NumPy 2.1.3、PyYAML 6.0.3。沒有下載模型權重或新增 checkpoint。

## 執行與重現

所有命令從 repository root 執行，並加 `-B` 避免產生 bytecode cache。

```bash
# 只顯示參數；不開影片、不寫 output
conda run -n yolo11 python -B development/data_pipeline/scripts/run_data_pipeline_gate_v1.py --help

# 驗證 config/manifest/split/sidecar identity 並建構 smoke datasets；影片存取為 0
conda run -n yolo11 python -B development/data_pipeline/scripts/run_data_pipeline_gate_v1.py --dry-run

# Unit tests；使用 synthetic decoder，不開任何 dataset 影片
conda run -n yolo11 python -B -m unittest discover \
  -s development/data_pipeline/tests -p 'test_*.py' -v

# 真實 bounded smoke；只開 6 train + 2 validation，stdout 是 JSON
conda run -n yolo11 python -B development/data_pipeline/scripts/run_data_pipeline_gate_v1.py

# 若需要保存 runtime JSON，必須指定一個尚不存在的新檔；程式拒絕覆寫
conda run -n yolo11 python -B development/data_pipeline/scripts/run_data_pipeline_gate_v1.py \
  --output /tmp/data_pipeline_gate_v1_smoke.json
```

Runner 不使用 `runs/`。`--help`、unit test 與 `--dry-run` 不建立資料管線 artifact。`--output` 只會寫使用者明確指定的新 JSON；若目的檔已存在則 fail closed。

## 使用介面

```python
from development.data_pipeline import BasketballClipDataset, load_gate_config

config = load_gate_config()
train = BasketballClipDataset(config, split="train", epoch=0)
validation = BasketballClipDataset(config, split="validation")
sample = train[0]
```

Dataset 僅接受 `train`／`validation`。Validation 使用 protocol 先行凍結的時間中心連續 64-frame 規則，且不建立 train quality index。傳入 `split="test"` 或 test ID 會在任何 path resolve/stat/open/decode 前拒絕。

每個 sample 回傳 tensor 與完整 selection/decoder/transform/provenance metadata。Normalization v1 固定為顯式 `identity`；不得把此介面解讀為任何 pretrained checkpoint 的 normalization 契約。

## 已知限制

- Sequential decoder 為了精確原始 frame index 會解碼 start 之前的 prefix；沒有用 seek 加速。
- Bitwise reference 重現保證限於相同 input 與相同 library/config 環境；不宣稱跨 OpenCV/NumPy 版本 bitwise 相同。
- Validation frame count/FPS 來自已授權 descriptor 的 decoder metadata，短片或與 train sidecar 不一致會停止。
- Source group 尚未證明是真實 game ID，near duplicate 也尚未完成內容稽核。
- Gate pass 不代表 I3D、YOLO、reference 或其他 preprocessing candidate 已證明有效。

