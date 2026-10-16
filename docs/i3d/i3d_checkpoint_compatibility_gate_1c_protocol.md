# I3D Checkpoint Compatibility Gate 1C Protocol

Version: `i3d_checkpoint_compatibility_gate_1c`  
Frozen before artifact acquisition: 2026-10-17 (Asia/Taipei)

## Scope and stop conditions

This Gate performs provenance, inventory, strict loading, normalization, endpoint and bounded forward-only checks. It does not implement YOLO or masked pooling, train a Make/Miss model, construct an optimizer, call backward, access test videos, or modify frozen baselines, historical runs, existing configs, Data Pipeline Gate v1 files, or processed data.

The Gate reports `BLOCKED` instead of substituting random or unrelated weights if the source chain, exact strict load, finite deterministic forward, official 224×224 path, project 192×256 path, or train/validation boundary cannot be established.

## Candidate frozen before download

- Gate 1B candidate: `C03`, hassony2/kinetics_i3d_pytorch commit `786fbae06a6fd5df419e1961cd32e32fbe8b396d`.
- Architecture: RGB Inception-v1 Inflated 3D ConvNet; not I3D-ResNet or R(2+1)D.
- Converted artifact: `model/model_rgb.pth`, expected 50,887,222 bytes, hosted at the pinned repository commit.
- Upstream chain: Gate 1B `A01`, google-deepmind/kinetics-i3d commit `0667e889a5904b4dc122e0ded4c332f49f8df42c`, RGB ImageNet-initialized Kinetics-400 TensorFlow checkpoint.
- Conversion implementation: pinned `i3d_tf_to_pt.py` plus `src/i3dpt.py`; the conversion uses TF `[D,H,W,in,out]` to PyTorch `[out,in,D,H,W]` transpose.
- License/notice: port code is MIT. Upstream repository root is Apache-2.0; the release contains no separate weights-specific grant. Both notice chains must be retained, and external redistribution of derived weights remains a documented limitation.
- Exact URLs, filenames and expected component sizes are frozen in `development/i3d_gate/configs/i3d_gate_1c.yaml`.

Artifacts are acquired only from the canonical pinned URLs into ignored `development/i3d_gate/artifacts/`. Every downloaded file must match its predeclared byte count. The acquisition manifest records canonical URL, redirect chain, bytes, local SHA-256, acquisition time, source commit and license; Git SHA-1, ETag and filenames are never treated as SHA-256.

## Loading and inventory

The downloaded PyTorch object is loaded with the current safe `weights_only=True` mode. The exact state dictionary is inventoried before model construction. Every tensor record includes key, shape, dtype and whether the matching model slot is a parameter or buffer.

The pinned topology is reproduced without heuristic shape matching or key renaming. `load_state_dict(..., strict=True)` is mandatory. Missing keys, unexpected keys, unused source tensors and shape mismatches must all be zero. The 400-class classifier remains present for compatibility inspection, but the Gate forward endpoint is `Mixed_5c`, before fixed pooling and logits. No classification head is added or adapted.

## Inputs and normalization

The official-reference path is deterministic float32 RGB `[N,C,T,H,W] = [1,3,64,224,224]` in `[-1,1]`; it does not use ImageNet mean/std. The project path reuses Data Pipeline Gate v1 float32 contiguous `[3,64,192,256]` RGB in `[0,1]`, adds the batch axis, and applies exactly one explicit out-of-place adapter `x*2-1` (`i3d_rgb_minus1_plus1_v1`). The adapter must preserve shape and float32 dtype, map endpoints exactly, leave the source tensor unchanged, and add version/scale/bias/in-place metadata.

## Endpoint and forward checks

The last pre-pooling spatiotemporal endpoint is `Mixed_5c`. Shape is measured, never hard-coded. Temporal/spatial stride is reported from input and output dimensions. Ordinary Gate-only GAP is `mean(dim=(2,3,4))`; it must produce finite `[N,C]` output. The fixed historical `[2,7,7]` classifier pooling is not invoked for endpoint extraction and therefore cannot block the 192×256 project path.

Both synthetic paths use seed `20261017`, batch size 1, `model.eval()` and `torch.inference_mode()`. A repeated forward of identical input must be bitwise identical in the same environment. Feature and pooled tensors record shape, finite status, min/max/mean/population-std, L1/L2 norm and a SHA-256 digest over contiguous little-endian float32 bytes.

The real-data smoke uses exactly one train and one validation sample selected by the existing Data Pipeline Gate v1 smoke-ID function. Dataset/test guards remain authoritative. Split open/decode counts are reported and test must be 0/0. Optimizer steps, backward calls, gradient accumulation and training remain zero.

## Cross-framework parity rule

Before acquisition, absolute tolerance is frozen at `1e-4` and relative tolerance at `1e-3` for `Conv3d_1a_7x7` and `Mixed_5c`. A PASS parity comparison requires a reliably pinned official TensorFlow/Sonnet runtime and the same fixed input. If that historical runtime cannot be established, parity is explicitly `NOT_RUN`, never inferred from strict loading or successful execution, and full official numerical equivalence is not claimed.

## PASS criteria

PASS requires complete acquisition hashes, pinned implementation/checkpoint, safe inventory, strict loading with zero discrepancies, verified RGB/normalization contracts, successful deterministic finite official/project synthetic forwards, finite GAP, successful bounded real train/validation forwards, zero test access, zero optimizer/backward/training activity, unchanged protected areas, and passing unit/regression tests. Cross-framework parity may remain `NOT_RUN` only if the limitation is prominently reported; in that case the Gate establishes checkpoint/implementation compatibility but not full conversion equivalence.
