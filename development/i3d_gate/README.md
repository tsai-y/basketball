# I3D Checkpoint Compatibility Gate 1C

This isolated area validates Gate 1B candidate `C03` against upstream `A01`. It performs safe checkpoint inventory, exact TensorFlow-to-PyTorch tensor mapping, `strict=True` loading, the explicit `[0,1] -> [-1,1]` adapter, `Mixed_5c` extraction, ordinary GAP and bounded forward-only smoke. It contains no optimizer, backward, YOLO, masked pooling or Make/Miss head.

Downloaded checkpoint/source/runtime artifacts live under ignored `development/i3d_gate/artifacts/`. Small config, implementation, tests and reports are tracked.

## Reproduction

From the repository root in the existing `yolo11` environment:

```bash
# Help only: no video access and no output
conda run -n yolo11 python -B development/i3d_gate/scripts/run_gate_1c.py --help

# One-time controlled acquisition (about 153 MB; refuses an existing artifact directory)
conda run -n yolo11 python -B development/i3d_gate/scripts/acquire_artifacts.py

# Safe inventory plus exact source-to-converted tensor mapping
TF_CPP_MIN_LOG_LEVEL=3 MPLCONFIGDIR=/tmp/mpl \
  conda run -n yolo11 python -B development/i3d_gate/scripts/inspect_checkpoint.py

# No-video dry-run
conda run -n yolo11 python -B development/i3d_gate/scripts/run_gate_1c.py \
  --dry-run --output /tmp/i3d_gate_1c_dry_run.json

# Unit/regression tests; no dataset video is opened
conda run -n yolo11 python -B -m unittest discover \
  -s development/i3d_gate/tests -p 'test_*.py' -v

# Full forward-only smoke; creates the tracked machine summary once
conda run -n yolo11 python -B development/i3d_gate/scripts/run_gate_1c.py
```

The acquisition, inventory and smoke commands refuse to overwrite their outputs. Delete or move only the specific Gate 1C output you intentionally want to regenerate.

## Exact implementation notes

The checkpoint has 287 tensors and the Gate model has exactly 287 parameter/buffer slots. A custom eval-only legacy BatchNorm omits modern PyTorch's automatically synthesized `num_batches_tracked`, so strict loading is genuinely one-to-one. The port's pinned source accidentally calls `mixed_4b` twice in `forward()`; this Gate corrects that unusable path while retaining all layer/module names and weights, and records the discrepancy in the compatibility report.

The fixed historical `[2,7,7]` average pool remains present only for architecture/head inventory. Gate feature extraction stops at `Mixed_5c`; project inputs therefore do not depend on a 7×7 feature grid. Ordinary GAP is a parameter-free mean over `[T,H,W]` used only for compatibility verification.
