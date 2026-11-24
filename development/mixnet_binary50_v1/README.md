# `development/mixnet_binary50_v1`

Isolated single-clip 50-frame direct binary classifier wrapper around the existing
`development.mixnet_i3d.backbone.MixNetI3DBackbone` (`timm` `mixnet_s.ft_in1k` 3D inflation).

## Interface Contract

- **Model**: `MixNetBinary50Model` (built via `build_mixnet_binary50(pretrained=False, ...)`)
- **Input**: single-clip FP32 tensor `[B, 3, 50, 120, 160]` (`B, C, T, H, W`), already RGB ImageNet-normalized once upstream (`scale=255.0`, `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`).
- **No double normalization**: model contains no `rgb_mean` / `rgb_std` buffers and passes `clip` directly to `backbone(clip)`.
- **Pooling & Head**: explicit global average pooling over `(T, H, W)` (`features_3d.mean(dim=(2, 3, 4))`) followed by `nn.Linear(backbone.num_features, 2)` (`1536 -> 2` raw logits).
- **Target & Loss**: `torch.long` targets (`Miss=0`, `Make=1`) with unweighted `CrossEntropyLoss` on raw logits (`development.aca_binary_v1.metrics.binary_ce`).
- **Label Provenance**: uses `diagnostic32_label_revision_v1` via `development.adjudicated_input_v1.adapter.LabelRevision`; sample #15 (`3p1/3p1_v150_003223_x264.mp4`) keeps `original_class="3p1"`, `effective_class="2p1"`, `class_id=1`, `binary_label=1`.

## Explicit Boundaries (Not Included in v1)

- Legacy `development/mixnet_i3d/` (including `ThreeWindowMakeMissModel`) is untouched.
- No LSTA or TSCI modules are attached.
- No training, optimizer, GPU, validation sweep, or test entry point is included.
- `pretrained=True` is explicitly blocked in `build_mixnet_binary50` until formal MixNet pretrained provenance, hash, and comparability are approved in Codex review.
- All 49 `nn.BatchNorm3d` layers inherit `timm` defaults (`momentum=0.1`, `eps=1e-5`, `affine=True`, `track_running_stats=True`); formal training BN policy remains undecided.

## CPU Verification Commands

```bash
PYTHONPATH=/home/tsai/.codex/worktrees/antigravity-mixnet-binary50/basketball \
  /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest \
  development.mixnet_binary50_v1.test_mixnet_binary50 -v

PYTHONPATH=/home/tsai/.codex/worktrees/antigravity-mixnet-binary50/basketball \
  /home/tsai/miniconda3/envs/yolo11/bin/python -B -m development.mixnet_binary50_v1.cpu_acceptance \
  --output-json /home/tsai/projects/basketball/outputs/01_current/mixnet_binary50_cpu_acceptance_20261123_v1/cpu_acceptance.json
```
