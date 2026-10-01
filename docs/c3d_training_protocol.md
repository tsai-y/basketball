# Reference C3D Training Protocol

Date: 2026-09-30. **Status: proposed after short feasibility tests; no full epoch, full-dataset training, or test evaluation was run.** This protocol uses the completed [C3D audit](c3d_reproduction_spec.md) and [3B implementation](c3d_reference_implementation.md). Existing Experiment A and 3B artifacts were not overwritten. Benchmark artifacts are isolated in `runs/full_video_c3d_reference_v1/feasibility/`.

## Goal

Train a Basketball-51 **full-video** Make (`1`) / Miss (`0`) C3D adaptation with the existing cleaned manifest and fixed train/validation/test split. This is not pre-release prediction and is not a reproduction of the original paper's benchmark score. The proposed primary objective is **one loss per video after mean-fc6 aggregation and L2 normalization**.

## Initialization Decision

| Strategy | Evidence and judgment | Current decision |
|---|---|---|
| From scratch on Basketball-51 | The [C3D paper](https://openaccess.thecvf.com/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html), §3.3, reports a Sports-1M from-scratch network and an I380K-pretrained/fine-tuned network. It does **not** show that a ~78M-parameter C3D will generalize from 10,279 cleaned Basketball-51 videos. The current local model declares the [author Caffe fillers](https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/examples/c3d_feature_extraction/prototxt/c3d_sport1m_feature_extractor_video.prototxt), so starting weights and preprocessing choices can be recorded exactly. GPU feasibility passed, but overfitting risk is high; this risk assessment is an inference, not a paper result. | **Primary runnable strategy now:** from-scratch, official filler declarations, video-level loss on all active conv→fc6 layers plus a binary linear head. This is PROJECT_ADAPTATION. |
| Author Sports-1M pretrained initialization | Likely reduces optimization burden and makes frozen fc6+SVM meaningful; that expectation is an inference from transfer learning and paper usage, not a measured Basketball-51 result. No compatible C3D checkpoint or Sports-1M mean file was found locally. Shape, color order, normalization, fc6 export point and weight provenance must be checked before loading. | **Secondary comparison, conditional:** verified author weights → frozen fc6 descriptors → binary linear SVM, followed by partial and potentially full fine-tuning. Do not silently use a third-party conversion. |

The 3B random-weight frozen descriptors of two train videos differed by only `0.0002373` after L2 normalization. The two-video diagnostic is **not** a performance estimate, but it gives no basis to promote **frozen random weights** as the primary baseline. A pretrained frozen feature study would be a different, separately recorded experiment.

### Required checkpoint compatibility gate before any pretrained experiment

1. Identify the author release, exact checkpoint URL/version and SHA-256; retain the matching network definition and original mean file. Do not download large weights or claim compatibility in this feasibility round.
2. Parse source tensors and verify every conv/fc weight shape; specifically check pool5 `512×1×4×4` and fc6 `4096×8192`. Decide how the Sports-1M 487-class head is discarded/replaced.
3. Verify source decoder channel order, numerical mean subtraction and pixel scale against the current RGB `[-1,1]` PROJECT_ADAPTATION. If they cannot be matched, do not load the weights into the current preprocessing path.
4. On identical 16-frame input, compare a Caffe reference forward and mapped PyTorch intermediate features, including the exact fc6 activation point. If comparison cannot be run reliably, record pretrained compatibility as unresolved and retain the from-scratch protocol.

## Freeze / Fine-tune Strategy

The current model has **78,003,970** parameters including its unused two-class `fc8`. A separate video-level `Linear(4096,2)` head adds **8,194** parameters. Because video loss is computed on **fc6**, `fc7`/`fc8` are **not in its computation graph**; calling all 78M parameters “trained” would be false. The proposed full fc6 path freezes fc7/fc8 and trains **61,222,658** active parameters including the video head.

| Option | Scope benchmarked or proposed | Trainable / active parameters | VRAM, speed, risk, research use |
|---|---|---:|---|
| 1. Frozen C3D + classifier | Freeze backbone; 4096-D descriptor → binary head (benchmark). A linear SVM would likewise train only its classifier weights. | **8,194** neural head parameters (SVM parameterization separate). | Lowest benchmark VRAM/fastest. Closest to paper §4's frozen C3D descriptor + linear SVM **only if C3D weights are relevant and verified**. Frozen random descriptors are not a credible formal transfer baseline. |
| 2. Freeze lower layers; tune upper | Benchmark unfreezes conv5a, conv5b, fc6 and video head; conv1a–conv4b frozen. | **47,723,522** | Intermediate VRAM and speed; fc6 alone is large. Recommended after a verified pretrained frozen baseline, not on a random frozen lower stack. |
| 3. Full end-to-end fine-tune | With compatible pretrained weights, train all **active conv→fc6** layers and video head. fc7/fc8 remain outside this objective. | **61,222,658 active**; naïvely enabling all model parameters reports 78,012,164 declared trainable but 16,789,506 get no gradient. | Higher compute/overfit risk; assess after frozen/partial transfer runs. |
| 4. From scratch | Same active path as option 3, initialized with official Caffe filler declarations; no pretrained weights. | **61,222,658 active** | **Current primary runnable protocol**. Scientifically useful as a clearly labeled dataset-specific C3D adaptation, but likely data hungry (inference). |

Recommended order **with the current verified assets**: option 4 first. If an author checkpoint passes the compatibility gate, add a separate comparison in the order **option 1 → option 2 → option 3**. These choices are PROJECT_ADAPTATION; the paper does not specify a Basketball-51 fine-tuning order.

## Video-level Objective

**Primary: video-level CrossEntropyLoss.** For a train video, decode all complete 16-frame clips, process them in GPU chunks of two, concatenate per-clip fc6 tensors **while retaining their computation graphs**, average fc6, L2-normalize, apply a two-output `Linear(4096,2)`, and compute one cross-entropy loss against the **video** label. Backpropagate once per video. With gradient accumulation four, scale each video loss by its group size before stepping. This aligns the supervision with the dataset's video-level outcome label and preserves the paper's mean-fc6 representation; the binary neural loss/head are PROJECT_ADAPTATION.

**Alternative: clip-level CrossEntropyLoss** would assign the final Make/Miss label to every early, middle and late clip. Basketball-51 provides video-level outcome labels, not verified clip-level outcome labels. Early clips may not show a shot or an observable outcome, while late clips may directly show the ball/rim/net result. Therefore label inheritance is weak supervision and can favor late result cues; clip-level objective is **not selected**. It is cheaper to train independent clip batches, but its loss no longer directly optimizes the paper-style mean-fc6 video descriptor. Full-video outcome visibility remains a study limitation under either loss; it must not be mistaken for pre-release prediction.

## Binary Classifier

| Choice | Relationship to source | Decision |
|---|---|---|
| Frozen 4096-D descriptor → linear SVM | Paper §4 uses averaged/L2-normalized fc6 with a **multi-class linear SVM** on UCF101; binary SVM on Basketball-51 is PROJECT_ADAPTATION. | Closest to the paper's feature-extraction usage. Conditional secondary experiment after verified pretrained weights; fit only train features, use validation for `C` selection, reserve test for the final fixed protocol. No SVM was fit this round. |
| Frozen descriptor → PyTorch linear(4096,2) + CE | Same features but replaces SVM objective. | Diagnostic/possible control, not the primary paper-pattern transfer comparison. The 3B two-video diagnostic does not count as formal training. |
| Trainable C3D fc6 path → video linear head + CE | Allows gradient through C3D and video aggregation. The paper does not report this Basketball-51 objective. | Primary from-scratch adaptation and the natural head for later partial/full fine-tuning. |

## Preprocessing

| Item | Source class | Current/proposed handling |
|---|---|---|
| 16 consecutive frames, resize 128×171, training random 112×112 crop and 0.5 horizontal flip | **PAPER_CONFIRMED**, paper §3.3 | Retained in `reference_c3d/data.py`. |
| Conv padding 1, official Caffe fillers, ReLU, dropout 0.5 after fc6/fc7 | **OFFICIAL_IMPLEMENTATION_CONFIRMED** in [author prototxt](https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/examples/c3d_feature_extraction/prototxt/c3d_sport1m_feature_extractor_video.prototxt) | Implemented; dropout does not lie on the chosen pre-dropout fc6 descriptor path. |
| Validation center crop, OpenCV decoding, RGB float32 `[-1,1]`, same crop per video, discard incomplete tail | **PROJECT_ADAPTATION** | Fixed in existing 3B config; no dataset-derived or test-derived statistics. |
| Exact author mean tensor, channel order/pixel scale, pretrained weight equivalence | **UNRESOLVED** | No pretrained loading until compatibility gate above passes. |

## Pooling / FC Shape Audit

The [author Caffe `pool3d_layer.cpp`](https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/src/caffe/layers/pool3d_layer.cpp), `Reshape` lines 52–57, computes pooled sizes with `ceil((input + 2·pad − kernel)/stride) + 1`. The [official Sports-1M prototxt](https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/examples/c3d_feature_extraction/prototxt/c3d_sport1m_feature_extractor_video.prototxt) supplies the five pool kernels/strides. `scripts/audit_c3d_shape.py` applied that formula and actually ran PyTorch max-pool at each stage; both gave temporal/spatial sizes `16×56×56 → 8×28×28 → 4×14×14 → 2×7×7 → 1×4×4`. Hence pool5 is **`[B,512,1,4,4]`**, flatten is **8,192**, and the current fc6 expects **8,192→4,096**. The minimal audit **PASS** is saved as `shape_audit.json`. This verifies shape compatibility with that official definition, **not** numerical equivalence of the full network or checkpoint.

## VRAM Benchmark

Actual RTX 5070 Ti (16,303 MiB reported by `nvidia-smi`), PyTorch 2.11.0+cu128. All batches are **clips from one train video**, not video batch sizes; one video label is used after feature aggregation. FP32 AdamW, three warmups and five synchronized timed training iterations per point; `torch.cuda.reset_peak_memory_stats()` runs before measurement. Values are peak PyTorch allocated/reserved, which are not total device usage. Full mode declared fc7/fc8 trainable but its fc6 loss gives them no gradients; active count is above.

| Mode | Clips/batch | Peak allocated MiB | Peak reserved MiB | Mean ms/iter | Clips/s | Status |
|---|---:|---:|---:|---:|---:|---|
| Full fc6 path | 1 | 1258 | 1366 | 17.54 | 57.0 | PASS |
| Full fc6 path | 2 | 1360 | 1566 | 26.01 | 76.9 | PASS |
| Full fc6 path | 4 | 1683 | 2450 | 42.26 | 94.6 | PASS |
| Full fc6 path | 8 | 2334 | 3916 | 73.56 | 108.8 | PASS |
| Frozen backbone | 1 | 420 | 532 | 3.91 | 255.9 | PASS |
| Frozen backbone | 2 | 519 | 752 | 6.86 | 291.5 | PASS |
| Frozen backbone | 4 | 720 | 724 | 11.98 | 333.9 | PASS |
| Frozen backbone | 8 | 1139 | 1154 | 22.57 | 354.4 | PASS |
| Partial conv5+fc6 | 1 | 1051 | 1100 | 9.74 | 102.7 | PASS |
| Partial conv5+fc6 | 2 | 1051 | 1266 | 12.70 | 157.5 | PASS |
| Partial conv5+fc6 | 4 | 1084 | 1306 | 17.94 | 222.9 | PASS |
| Partial conv5+fc6 | 8 | 1503 | 1958 | 29.14 | 274.5 | PASS |

Every tested batch passed forward/backward with finite loss; no OOM. The maximum tested safe clip batch was **8 for all three modes**. This is a short throughput comparison only; it cannot establish stable epoch throughput or accuracy. Source JSON: `vram_benchmark.json`, `throughput_benchmark.json`, and `training_mode_comparison.json`.

### Video-level memory strategies

One **actual 17-clip train video**, single warmup followed by one measured video-level optimizer step:

| Strategy | Peak allocated MiB | Peak reserved MiB | Seconds/step | Gradient through C3D? |
|---|---:|---:|---:|---|
| All 17 clips in one GPU batch | 3796 | 8974 | 0.1465 | Yes, conv1a and fc6 |
| Two-clip chunks, retain all graphs until video loss | 3537 | 3840 | 0.1804 | Yes, conv1a and fc6 |
| Two-clip chunks, frozen C3D under `no_grad` | 519 | 974 | 0.0560 | No; classifier only |

**Primary:** chunk by two, then average **all** 17 fc6 outputs and backpropagate once. Chunking lowers instantaneous convolution workspace and observed reserved VRAM, but it **does not eliminate** the memory cost of retaining all clip graphs. The measured 17-clip allocated value is already ~3.5 GiB; longer videos and future changes can increase it. Never infer that a video batch of 8 is safe from the clip-batch-8 table. A one-video GPU batch plus gradient accumulation is the conservative choice.

**Frozen precompute option:** under `no_grad`, write one normalized float32 4096-D descriptor per video, keyed by the fixed manifest video ID/split and source/checkpoint/preprocessing hashes. Arithmetic storage estimate for **all 10,279 usable videos** is **168,411,136 bytes (160.61 MiB)** before metadata/filesystem overhead; no all-dataset features were extracted this round. Train/validation extraction would come first, and test extraction only after protocol lock. This reduces repeated decoding and makes SVM training cheap, but prevents backbone gradients and requires artifact versioning.

## Throughput / DataLoader Benchmark

Six usable **train** videos, all with 17 clips, deterministic center crop for the benchmark. GPU was warmed before timing; every video was decoded into clips and **all 17 clips** were forwarded in chunks of two. Data wait means time spent waiting for `next(DataLoader)`; GPU step includes chunk copy, forward and aggregation. Each worker count was measured once, so differences are indicative only. GPU utilization was **not directly sampled**; wait fraction is evidence about possible starvation, not a measured utilization percentage.

| Workers | Average data wait/video | Average GPU step/video | Data wait fraction | Status |
|---:|---:|---:|---:|---|
| 0 (current 3B setting) | 0.0908 s | 0.0633 s | 58.9% | PASS |
| 2 | 0.0156 s | 0.0538 s | 22.5% | PASS |
| 4 | 0.0189 s | 0.0537 s | 26.0% | PASS |

Choose **2 workers** initially: lowest wait in this six-video check and no observed OpenCV/multiprocessing failure. Eight workers were not tested because the six-item check would mostly measure process startup and contention, not useful steady throughput. Reassess from actual epoch logs only if a concrete bottleneck appears.

## Proposed Hyperparameters

All entries below are **PROJECT_ADAPTATION** for Basketball-51, except the network topology/filler elements attributed above. They are fixed in [`configs/c3d_reference_train.yaml`](../configs/c3d_reference_train.yaml); no tuning or sweep was performed.

| Parameter | Proposed value | Reason / limit |
|---|---|---|
| Initialization | From scratch, author prototxt filler declarations | Executable without incompatible/unknown checkpoint; transfer remains a conditional comparison. |
| Trainable scope | Entire active conv1a→fc6 path + separate 4096→2 head; freeze unused fc7/fc8 | Video-level descriptor loss cannot train fc7/fc8. |
| Loss | One video-level CrossEntropyLoss after mean fc6 and L2 | Matches available outcome label granularity. |
| Optimizer / LR / weight decay | AdamW / `1e-4` / `1e-4` | Same optimizer class/rate used in VRAM feasibility; not a claim about paper optimization. |
| Video batch / clip chunk / accumulation | 1 video / 2 clips / 4 videos | All clips contribute to one video loss; clip graphs retained only until each video's backward; accumulation averages four video gradients before optimizer step. |
| Epochs | Maximum 10 | Initial bounded plan, **not** a validated optimum. |
| Scheduler | ReduceLROnPlateau on validation loss, factor 0.5, patience 1 | PROJECT_ADAPTATION. |
| Early stopping / checkpoint | Stop after 3 non-improving validation epochs (min delta `0.001`); best checkpoint by minimum validation loss under that improvement rule | Test is not consulted. |
| Gradient clipping | Global norm 1.0 | Stability measure, not in paper protocol. |
| Mixed precision | Off (FP32) | Feasibility timings and memory are FP32; AMP numerical behavior untested. |
| Seed / workers | `20260930` / 2 | Fixed split unchanged; two-worker short check was stable. |
| Run path | `runs/full_video_c3d_reference_train_v1/` | Separate from Experiment A and all 3B/3C feasibility artifacts. |

### Training script status

[`scripts/run_c3d_reference_train.py`](../scripts/run_c3d_reference_train.py) is implemented but **full training was not started**. `--check` verified fixed manifest/split hashes and separate run path. A first one-train/one-validation-video `--dry-run` failed only when serializing a NumPy boolean after computation; this output bug was corrected. The repeated `--dry-run` **PASS** processed 17 train clips, took one video-level optimizer step, and forwarded 17 validation clips; both losses were finite, no test data or run files were used. PyTorch warned that CUDA max-pool3d backward lacks a deterministic implementation, so fixed seeds do **not** guarantee bitwise-identical weights. Neither dry-run loss nor tiny benchmark loss is a research metric.

## Validation Policy

Use **only the existing fixed train and validation splits**. Log per epoch: train/validation loss, accuracy, Make precision/recall/F1, confusion matrix, epoch time, peak allocated/reserved VRAM, and learning rate. Validation loss controls scheduler, early stopping and best checkpoint. The script snapshots config, manifest and split; checkpoints include model/head, optimizer, scheduler, RNG states, protocol hash and epoch history for **epoch-boundary resume**. Code/config/manifest mismatches prevent resume, apart from the documented one-time compatibility for the resume script fix and a resume-only increase to the epoch limit. Other setting changes require a new run name rather than rewriting this protocol after observing outcomes.

## Test Policy

The proposed training script does **not** create a test Dataset or evaluate test rows. Test must remain sealed until initialization, preprocessing, model/head, checkpoint rule and all validation decisions are fixed. Perform one final test evaluation in a later stage only. No test labels or test metric influenced this feasibility decision.

## Known Risks

- From-scratch active path has **61.2M** trainable parameters but only 10,279 usable videos; overfitting or slow convergence is plausible, not yet measured.
- Current RGB `[-1,1]` preprocessing and fc6 export point are explicit adaptations. Official weights may require different mean/color handling.
- Six-video loader timing and five-iteration clip benchmarks are short; they cannot predict full-epoch time, long-run GPU utilization or stability.
- CUDA `max_pool3d` backward is not bitwise deterministic in the current PyTorch build, despite fixed seeds and `warn_only=True`.
- True game identity and near duplicates remain unresolved; the full-video task can use outcome frames.

## Project Adaptations

Basketball-51 binary labels; existing manifest/split; discard incomplete tail; OpenCV RGB `[-1,1]`; deterministic center validation crop; fc6 post-ReLU/pre-dropout extraction; two-output video head; video-level CE; AdamW/scheduler/early stopping; PyTorch `ceil_mode=True` (shape now confirmed against official Caffe formula); gradient accumulation and checkpoint policy. These choices are not asserted as the original paper's training protocol.

## Full Training Command

**For the next approved stage only; do not run in this feasibility round.** From project root in Ubuntu terminal/tmux:

```bash
conda run -n yolo11 python -B scripts/run_c3d_reference_train.py --config configs/c3d_reference_train.yaml
```

Epoch-boundary resume after interruption (same epoch limit):

```bash
conda run -n yolo11 python -B scripts/run_c3d_reference_train.py --config configs/c3d_reference_train.yaml --resume
```

To extend the completed 10-epoch run to a maximum of 20 epochs, keep the
original config and run snapshot unchanged. The resume-only CLI override
changes only the stopping limit; all other settings and the fixed split must
match. First perform the read-only preflight:

```bash
conda run -n yolo11 python -B scripts/run_c3d_reference_train.py --config configs/c3d_reference_train.yaml --resume --max-epochs 20 --check
```

Then, when the extension is approved, run:

```bash
conda run -n yolo11 python -B scripts/run_c3d_reference_train.py --config configs/c3d_reference_train.yaml --resume --max-epochs 20
```

The preflight checks `last.pt`, the original run config and fixed-split
snapshots, existing history, and `best.pt` without writing run files. The
first resumed epoch is 11. The ten existing log records are preserved when
new epoch records are added. `best.pt` changes only if a later validation
loss improves the recorded best under the same criterion. Checkpoints from
the extension record the effective maximum epoch. The earlier checkpoint's
protocol hash included the old training script; the resume fix accepts only
that known 10-epoch hash with unchanged reference model, data, aggregation,
and preprocessing source hashes. New checkpoints use the updated script hash;
an older `best.pt` remains valid if later epochs have not improved validation
loss.

Read-only preflight:

```bash
conda run -n yolo11 python -B scripts/run_c3d_reference_train.py --config configs/c3d_reference_train.yaml --check
```

## Early-Stopping Adjustment Audit

**2026-09-30; PROJECT_ADAPTATION; before any test evaluation.** The run was
extended from the original 10-epoch limit to 20 but stopped at epoch 13:
`last.pt` records best epoch 10, best validation loss
`0.6015637946479461`, and three consecutive non-improving epochs (11–13).
The original early-stopping patience was **3**. `ReduceLROnPlateau` uses
patience **1**, factor **0.5**. It stepped from `1e-4` to `5e-5` **after**
epoch 12 validation; the log's LR is recorded after `scheduler.step`, so only
epoch 13 was actually trained at `5e-5` before early stopping. This supports
the research judgment that patience 3 may be too short to assess the lower
LR; it does not establish that more epochs will improve validation loss.

Proposed resume-only adjustment: **patience 6, maximum 20 epochs**. Keep the
original config at patience 3/max 10, and preserve epoch 13 model/head,
optimizer, scheduler, current `5e-5` LR, RNG, best loss/epoch, and the existing
counter **3**. The counter is **not reset**. If validation never improves by
the existing `0.001` minimum delta, early stopping will occur after epoch 16.
The unchanged scheduler may reduce LR again after epoch 14 validation if its
plateau condition persists; this adjustment does not guarantee several epochs
at exactly `5e-5`.
The existing 13 log records remain in the history when new records are added;
the JSON log is atomically rewritten with that extended history after each
completed epoch. `best.pt` changes only on a qualifying validation-loss
improvement. The test set remains untouched; `training_status.json` records
`test_evaluation_performed=false`.

Read-only preflight:

```bash
conda run -n yolo11 python -B scripts/run_c3d_reference_train.py --config configs/c3d_reference_train.yaml --resume --check --max-epochs 20 --early-stopping-patience 6
```

Proposed resume command, **not executed during this audit**:

```bash
conda run -n yolo11 python -B scripts/run_c3d_reference_train.py --config configs/c3d_reference_train.yaml --resume --max-epochs 20 --early-stopping-patience 6
```

## B Early-Stopping Extension Audit

**2026-09-30; PROJECT_ADAPTATION; before Experiment B test evaluation.**
This audit revisits the 20/6 proposal above, which was **not executed**. The
current `last.pt` remains at epoch 13: best epoch 10 and validation loss
`0.6015637946479461`, with stale counter 3. `best.pt` is epoch 10. The
checkpoint saves model/head, AdamW, ReduceLROnPlateau, Python/NumPy/PyTorch/CUDA
RNG states and all 13 history records. No new model or optimizer is created
for the proposed continuation.

The original early-stopping patience is **3**. Epochs 11–13 had validation
losses `0.6138866176008793`, `0.6028682707622888`, and
`0.6149046720663578`; none improved on epoch 10. The counter therefore
progressed **0 → 1 → 2 → 3**, triggering early stop after epoch 13.
ReduceLROnPlateau has factor `0.5` and patience `1`: it lowered AdamW's LR
from `1e-4` to `5e-5` **after epoch 12 validation**. The per-epoch log records
post-scheduler LR, so epoch 12's `5e-5` entry does not mean its training used
that rate. Only epoch 13 trained at `5e-5` before the stop. The saved
scheduler has `last_epoch=13`, `best=0.6015637946479461`,
`num_bad_epochs=1`, and `_last_lr=[5e-5]`.

**Research assessment:** patience 3 may give too little time after the first
LR reduction. This does not establish that more training will improve
validation loss. The conservative extension selected here is **maximum 30
epochs, early-stopping patience 7**, with the same scheduler and optimizer
states. The original config remains 10/3; these are resume-only CLI
overrides. The historical counter stays **3**, not zero. With no significant
validation improvement, four more non-improving epochs would stop at epoch
17. The scheduler may lower LR again during that interval.

The original script used `early_stopping_min_delta=0.001` for both the stop
counter and `best.pt`. That could skip a validation loss that was strictly
lower but improved by less than `0.001`. Before this continuation, checkpoint
selection was separated from the counter: **`best.pt` updates for any
strictly lower validation loss**, while a decrease greater than `0.001`
resets the early-stopping counter. An independent
`early_stopping_reference_loss` is saved in future checkpoints; for the
unchanged epoch-13 checkpoint it is reconstructed as the saved best loss.
The existing best epoch/loss and counter are unchanged. This is a documented
Basketball-51 training-policy clarification, not a C3D paper setting.

The read-only preflight confirms next epoch 14, LR `5e-5`, best epoch 10,
best loss `0.6015637946479461`, patience 7, maximum 30, and
`test_evaluation_performed=false`. Existing 13 log records are retained in
the checkpoint history; after each future completed epoch the JSON log is
atomically rewritten with the old records plus the new one. No test videos
or test metrics were accessed in this audit.

Read-only check:

```bash
conda run -n yolo11 python -B scripts/run_c3d_reference_train.py --config configs/c3d_reference_train.yaml --resume --check --max-epochs 30 --early-stopping-patience 7
```

Formal resume command, **not executed in this audit**:

```bash
conda run -n yolo11 python -B scripts/run_c3d_reference_train.py --config configs/c3d_reference_train.yaml --resume --max-epochs 30 --early-stopping-patience 7
```
