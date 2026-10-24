# Quality-weighted frozen-feature Transformer v1

Controlled experiment: alpha **0.25** and an additional exploratory strength
**0.5**, paired seeds **20261024, 20261025, 20261026**.
The three alpha=0.25 runs are already complete and must be preserved.
all **7,193 train** and **1,547 validation** videos. No test samples are read.
This is not RGB partial fine-tuning, YOLO integration, RL, or a blur detector.

## Only intentional training change

For video i, r_i = sum(three candidate-overlap frame counts) / 192.
Raw coefficient w_i = 1 - alpha*r_i. Normalize every coefficient by the SAME
full-train mean: w_norm_i = w_i / mean(w_train).
The objective per batch is mean(w_norm_i * BCE(logit_i, label_i)).
There is **no batchwise, classwise, or shotwise normalization**.
There is no feature/window scaling or weighted resampling.
The unit of loss is a video, not three independent windows.
Candidate flags are a proxy, not confirmed blur or unusability truth.

The existing frozen Transformer baseline is preserved. Its recorded config,
PyTorch version and source hashes must match. The head is freshly initialized
with the same seed; its trained best.pt is NOT used as initialization.
Each seed has its own explicitly mapped original baseline; missing or mismatched
references fail closed, rather than falling back to seed 20261024 or selecting
the highest-scoring reference. Within each three-seed set, alpha, data,
architecture and optimizer stay fixed. CLI accepts only alpha=0.25 or 0.5;
the default remains 0.25 for backward compatibility. The 0.5 comparison is
exploratory, added after inspecting the 0.25 results. Finish all three seeds
regardless of the first seed's performance, then stop this strength comparison.
The model, shuffle generator, batch size 64 (last batch 25 retained), AdamW
lr=1e-4, weight decay=1e-4, clip norm=1, CPU threads=4, float32 precision,
maximum 50 epochs and validation protocol match that reference.

Validation uses ordinary UNWEIGHTED global BCE and a probability threshold of
0.5. Every completed epoch evaluates all validation videos. Best checkpoint:
lowest validation loss. Early stopping: patience 10, min_delta 1e-4, separate
from saving the absolute lowest-loss checkpoint. No epoch-0 candidate is added,
matching the original frozen-head experiment. Test remains untouched.

## Pre-training check (safe; no optimizer or output files)

From `/home/tsai/projects/basketball`, in the yolo11 environment:

```bash
python -B -m unittest discover -s development/temporal_quality_v1/tests -p 'test_*.py' -v
python -B -m development.temporal_quality_v1.train_quality --dry-run --alpha 0.5 --seed 20261024
python -B -m development.temporal_quality_v1.train_quality --dry-run --alpha 0.5 --seed 20261025
python -B -m development.temporal_quality_v1.train_quality --dry-run --alpha 0.5 --seed 20261026
```

The dry run fully validates feature hashes and shapes, ID/source-group split
separation, train sidecars and all cached overlap counts. It performs a small
real validation forward and a two-class real train weighted backward, verifies
finite gradients and unchanged model state, clears gradients, and exits.
It does NOT create an optimizer, step parameters, make a run directory, or save
checkpoints. Unit tests may optimize small artificial tensors only, never the
actual feature cache. Existing feature/model files are not modified.

## Training (user starts this separately)

```bash
python -B -m development.temporal_quality_v1.train_quality --train --alpha 0.5 --epochs 50 --seed 20261024
python -B -m development.temporal_quality_v1.train_quality --train --alpha 0.5 --epochs 50 --seed 20261025
python -B -m development.temporal_quality_v1.train_quality --train --alpha 0.5 --epochs 50 --seed 20261026
```

Every invocation creates a distinct directory under
`development/temporal_quality_v1/artifacts/quality_alpha050_seed<seed>_*` for
alpha=0.5. Alpha=0.25 retains the `quality_alpha025_seed<seed>_*` naming.
Experiment names, summary alpha, config alpha, weight audit and weight hashes
are strength-specific. No trained checkpoint is loaded when changing alpha.
Keep all completed results; do not rerun or replace them to select a more
favorable outcome. Compare all three paired seeds, not only the best seed.
Outputs: config.json (formula, normalization, initialization/weight/source
hashes), weight_audit.json, train_weights.pt, history.csv (both weighted and
unweighted training loss), best.pt, last.pt, summary.json (unweighted validation,
per-shot diagnostics and same-seed baseline differences), and ordered validation
predictions. No former run is overwritten. The artifact folder is ignored by
Git. There is **no resume support**: last.pt is a model snapshot, not a complete
optimizer/RNG checkpoint. Restarting the command starts a fresh run.

Do not judge quality-proxy correctness from classification accuracy alone.
One seed is an initial controlled result, not evidence of stable improvement.
