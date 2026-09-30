# Experiment A2: Modified C3D-style Extended Training

Date: 2026-09-30. **Protocol prepared; A2 full training has not started.**
This is a follow-up full-video Make/Miss classification experiment, not
pre-release prediction and not an untouched continuation of Experiment A1.

## Motivation

A1 validation loss fell at every observed epoch: `0.677480`, `0.584887`,
`0.574223`, `0.556761`, `0.525438` over epochs 1–5. Validation accuracy rose
from `0.563672` to `0.760181`. Because the best checkpoint is the final epoch,
the five-epoch limit ended while the validation curve was still improving.
This supports a bounded extended-training follow-up; it does not prove that
additional epochs will improve validation or test performance.

## Relation to A1

- A1 remains frozen at `runs/full_video_c3d_style_v1/`. Its test accuracy is
  already known (`0.747888`), so A2 is a follow-up experiment whose choices
  must be based on the A1 **training/validation trend**, not the A1 test result.
- A1 `best.pt` contains model weights, epoch 5, validation loss, seed, config,
  manifest hash and protocol hash. It has **no optimizer, scheduler, RNG state,
  training history, or `last.pt`**. Therefore A1 cannot be truly resumed.
- Classification: **WARM_START_ONLY**. A2 is initialized from the frozen A1
  best model weights; its optimizer is new. Epoch labels 6–15 describe the
  total model-training budget, but the optimizer trajectory is discontinuous
  at the A1→A2 boundary. A2 logs also include `a2_epoch` 1–10.

## Initialization

`configs/c3d_style_extended.yaml` pins the A1 `best.pt` SHA-256, original
manifest and split SHA-256, and the current A1 model/data source hashes. The
A2 script requires strict state-dict compatibility with `baseline.C3DStyle`
and compares the A2 data, preprocessing and optimizer settings to the frozen
A1 config. No A1 file is written. A2's initial best candidate is a new
checkpoint in its own run directory containing the A1 epoch-5 model and
validation loss. A later A2 checkpoint replaces it only if validation loss
falls below the current best.

## Training Protocol

- Run path: `runs/full_video_c3d_style_extended_v1/` (must not exist before
  starting formal training). The script creates it exclusively; it never
  writes to A1.
- Same eight-convolution modified C3D-style architecture, 16 uniform-bin RGB
  frames, 96×96 letterbox, `[-1,1]` normalization, training temporal jitter
  and horizontal flip `p=0.5`, deterministic validation bin centers, fixed
  cleaned manifest/split, binary labels Miss `0` and Make `1`.
- Same AdamW, LR `0.001`, weight decay `0.0001`, batch size 16 and seed
  `20260927`. **AdamW state starts fresh** because A1 did not save it.
- At most **15 total epoch labels**, meaning at most ten A2 epochs (6–15)
  after A1's five epochs. Early stopping allows **5** consecutive epochs
  without a lower validation loss; `min_delta=0`. The best checkpoint is
  selected by lowest validation loss, including the A1 initialization as
  the initial candidate. The separate A2 log records train/validation loss
  and accuracy per epoch.
- **No LR scheduler** is added. A1 validation loss was still decreasing at
  its stop point, so retaining its constant LR avoids introducing a second
  optimization change during this bounded follow-up. This choice was made
  before A2 training and is a project adaptation, not a C3D-paper setting.
- New run files on formal training: config, manifest/split snapshots,
  environment, `best.pt`, per-epoch `last.pt`, `training_log.json`, and
  `training_status.json`. A2 `last.pt` saves model, new optimizer, RNG, epoch,
  best validation state and A2 history for audit. The initial preparation
  did not add an interruption-resume command; the later audit below adds one.

## Validation Policy

The fixed **train** split updates weights; the fixed **validation** split
chooses the A2 best checkpoint and early-stop epoch. Training uses the same
Make/Miss mapping and reports the same loss/accuracy definitions as A1.
No split is redrawn. A1's previously observed test result is not a criterion
for LR, duration, patience, preprocessing, or model selection.

## Test Policy

No A2 test dataset or test-evaluation stage exists in this preparation
script. After A2 training and its protocol are fixed, a separate evaluation
step may evaluate its selected best checkpoint **once** with A1's metrics:
accuracy, Make precision/recall/F1 and confusion matrix. This round did not
read test videos or evaluate A1 test again.

## Resume / Warm-Start Status

**A1 → A2 is warm-start, not true resume.** A1 model weights are preserved;
its AdamW moments, worker RNG trajectory and training history are unavailable
for restoration. The original five A1 log entries stay in A1; A2 logs only
new epochs 6–15 and records its A1 checkpoint hash for provenance.

## Reproducibility Notes

The script validates source hashes and refuses an existing A2 directory.
It seeds Python, NumPy, PyTorch and CUDA and uses deterministic per-epoch
DataLoader generators. A2 retains A1's sampling/preprocessing rules, but
its random draw sequence need not match a hypothetical uninterrupted A1 run.
CUDA max-pool3d backward was previously observed to lack a deterministic
implementation, so identical seeds do not guarantee bitwise-identical GPU
weights. The one-batch A2 dry-run used CPU and produced finite train and
validation losses; those two losses are pipeline checks, not research metrics.

Read-only config/checkpoint check:

```bash
conda run -n yolo11 python -B scripts/run_c3d_style_extended.py --config configs/c3d_style_extended.yaml --stage check
```

Ephemeral train/validation dry-run (no run files):

```bash
conda run -n yolo11 python -B scripts/run_c3d_style_extended.py --config configs/c3d_style_extended.yaml --stage dry-run
```

Formal A2 training command, **not executed in this audit**:

```bash
conda run -n yolo11 python -B scripts/run_c3d_style_extended.py --config configs/c3d_style_extended.yaml --stage train
```

## A2 Epoch-15 Resume Audit and Budget Extension

Date: 2026-09-30. **Decision made before any A2 test evaluation.** The
completed A2 `last.pt` contains epoch-15 model weights, AdamW state (36
parameter states), Python/NumPy/PyTorch/CUDA RNG state, ten A2 history records
for epochs 6–15, best epoch 15, best validation loss
`0.48392225430138125`, and stale counter 0. `best.pt` matches that best
epoch/loss. The optimizer LR is `0.001`; there is no scheduler. A2 is thus
**TRUE_RESUME** from its own epoch 15 checkpoint, although A1→A2 remains a
warm-start boundary.

The training budget is extended with **resume-only** `--max-epochs 50` and
`--early-stopping-patience 7`. The original A2 config remains at total epoch
15/patience 5 as a historical snapshot. Model, optimizer, LR `0.001`, RNG,
best loss/epoch, all ten history records and the stale counter 0 are restored
from `last.pt`. No counter reset or optimizer restart is permitted. The
existing `best.pt` is retained unless a later validation loss is **strictly
lower**; the JSON training log is atomically rewritten after each completed
new epoch with all existing records plus the appended record. Test videos and
test metrics are not read for this adjustment.

Read-only preflight (verified current epoch 15, next epoch 16):

```bash
conda run -n yolo11 python -B scripts/run_c3d_style_extended.py --config configs/c3d_style_extended.yaml --resume --check --max-epochs 50 --early-stopping-patience 7
```

Ephemeral one-batch train/validation dry-run from the epoch-15 state (no run
files written):

```bash
conda run -n yolo11 python -B scripts/run_c3d_style_extended.py --config configs/c3d_style_extended.yaml --resume --stage dry-run --max-epochs 50 --early-stopping-patience 7
```

Formal resume command, **not executed in this audit**:

```bash
conda run -n yolo11 python -B scripts/run_c3d_style_extended.py --config configs/c3d_style_extended.yaml --resume --stage train --max-epochs 50 --early-stopping-patience 7
```
