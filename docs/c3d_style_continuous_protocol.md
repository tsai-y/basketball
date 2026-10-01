# Experiment C — Modified C3D-style with Continuous Clips

Date: 2026-10-01. **Implementation and short validation complete; full training not started.** This is full-video Basketball-51 Make/Miss classification, not pre-release prediction. All Experiment C choices below are **PROJECT_ADAPTATION**; C is not a reproduction of the original C3D benchmark.

## Research question and comparisons

- **C vs A2:** both use the Modified C3D-style eight-convolution feature stack. A2 sees one sparse 16-frame sample across a whole video; C sees all complete overlapping 16-frame clips and aggregates their features. C also changes 96×96 letterbox input to B's 112×112 crop and uses a different video-level head, so this is **not a pure single-variable temporal ablation**. Initialization and optimizer trajectories also differ: A2 warm-started A1; C is proposed from scratch.
- **C vs B:** both use the same continuous sampler, resize/crop/normalization, 16-frame length, stride 8, and mean-then-L2 video aggregation. They differ in backbone, feature dimension (C 256; B 4096), classifier input, and training history. Validation comparisons should describe the entire system, not infer an isolated architecture effect.

## Fixed dataset and test policy

Use the existing cleaned manifest `data/processed/dataset_manifest.csv`, fixed `data/processed/split_summary.json`, and original Make=1/Miss=0 labels. `configs/c3d_style_continuous.yaml` pins their SHA-256 hashes. The Experiment C script permits only `train` and `validation` datasets; it has no test evaluation stage. No test video was decoded during this implementation. Formal test evaluation remains a separate future decision after training and protocol selection.

## Clip pipeline and preprocessing

The script directly imports `ContinuousClipSampler` and `C3DVideoDataset` from `reference_c3d/data.py`; it does not copy the decoder or use `UniformBinSampler`. Clips are contiguous frames `[0..15]`, `[8..23]`, `[16..31]`, and so on. Only complete clips are retained; an incomplete tail is discarded, matching B. A 40-frame synthetic index check yields four clips ending `[24..39]` with eight-frame overlap. An actual 146-frame train video yielded 17 clips.

The same B preprocessing code resizes each RGB frame to height 128, width 171; uses one spatial crop/flip decision per video; train random crop 112×112 and horizontal flip probability 0.5; validation center crop; then converts pixels to float32 `[-1,1]`. This color normalization is a documented **PROJECT_ADAPTATION**, not a claim about paper-exact C3D mean/color preprocessing. Input clips are `[K,3,16,112,112]` per video.

## Model and video-level objective

`experiment_c/model.py` instantiates the unchanged `baseline.model.C3DStyle` and takes its `features` and `pool` modules. The original A1/A2 `forward` and head are not modified or included in C's trainable model. Eight Conv3D layers with channels `3→32→64→128→128→256→256→256→256`, BatchNorm3d, ReLU and five MaxPool3d layers feed `AdaptiveAvgPool3d(1)`. A 256-D vector is extracted after GAP for each clip. C concatenates all clip features, averages them, applies L2 normalization and then a **new** `Linear(256,2)` video classifier. One CrossEntropyLoss is computed per video; there is no clip-level Make/Miss supervision or probability averaging.

The observed model contains **6,919,234 total parameters, all trainable**. The old A1/A2 `256→128→2` head is intentionally absent. For `[2,3,16,112,112]`, the actual feature map is `[2,256,1,3,3]`, GAP is `[2,256,1,1,1]`, clip features are `[2,256]`, the video descriptor is `[256]`, and logits are `[2]`. The spatial shape therefore supports B-style 112×112 input without changing A's backbone.

## Proposed training protocol

The initial C protocol is **PROJECT_ADAPTATION** and has not been evaluated on a full epoch: train from scratch; AdamW, LR `1e-4`, weight decay `1e-4`; one video per DataLoader item; two clips per GPU chunk; accumulate gradients across four videos; FP32; ReduceLROnPlateau on validation loss (factor 0.5, patience 1); maximum 50 epochs; early stopping after seven epochs without a strictly lower validation loss; save best checkpoint by strictly lowest validation loss. This chooses B's video-level optimizer/LR/scheduler as a conservative starting point for the shared clip pipeline; A2 used AdamW `1e-3` without a scheduler for its sparse, warm-started path. **Neither rate is a C3D paper setting or proven optimal for C.** Fixed seed `20260930`, two DataLoader workers, and source/config/data hashes are recorded in the proposed new run. Training writes only to `runs/full_video_c3d_style_continuous_v1/` under the lifecycle rule below.

Training must record train/validation loss, accuracy, Make precision/recall/F1, confusion matrix, LR, epoch time and GPU peak allocation when available. Validation controls scheduler, early stopping and best checkpoint. The current script does not implement resume; interruption would require a separately audited resume path before continuation. CUDA memory/throughput for C have **not** been measured in this round, because the available PyTorch process reported CPU. A full video retains gradients through all clip chunks until its single video-level backward, so GPU memory needs monitoring when formal training begins. Fixed seeds do not guarantee bitwise-identical CUDA max-pool backward.

## Checks actually performed

With the existing `yolo11` environment, `--stage check` **PASS** verified fixed manifest/split hashes, preprocessing equality with B, and an unused C run path. `--stage shape` **PASS** produced the shapes and count above. `--stage smoke` **PASS** on CPU: the 40-frame clip indices were contiguous with stride 8/overlap 8; forward, aggregation, one video-level loss and backward had finite loss and finite nonzero gradients. `--stage dry-run` **PASS** on CPU: one actual train video (146 frames, 17 clips) completed one optimizer step through all clips; one actual validation video (146 frames, 17 clips) completed inference with finite loss. These tiny losses are **not** research results. Neither stage created a run/checkpoint or accessed test videos.

## Commands

Read-only configuration check and ephemeral validation:

```bash
conda run -n yolo11 python -B scripts/run_c3d_style_continuous.py --config configs/c3d_style_continuous.yaml --stage check
conda run -n yolo11 python -B scripts/run_c3d_style_continuous.py --config configs/c3d_style_continuous.yaml --stage shape
conda run -n yolo11 python -B scripts/run_c3d_style_continuous.py --config configs/c3d_style_continuous.yaml --stage smoke
conda run -n yolo11 python -B scripts/run_c3d_style_continuous.py --config configs/c3d_style_continuous.yaml --stage dry-run
```

**Next stage only, not run in this round:**

```bash
conda run -n yolo11 python -B scripts/run_c3d_style_continuous.py --config configs/c3d_style_continuous.yaml --stage train
```

## Run-directory lifecycle audit (2026-10-01)

An attempted formal C start left exactly four metadata files (`config.yaml`, `dataset_manifest.csv`, `split_summary.json`, `environment.json`) and no checkpoint, training log or status. The previous script rejected **any** existing directory in `main()` and called `mkdir(exist_ok=False)` again in `train()`, so the second invocation raised `FileExistsError` before epoch 1.

The revised read-only preflight accepts either an **absent** run directory or an **exact metadata-only** directory. For the latter, it checks that the config/manifest/split bytes match current pinned inputs and that recorded package/source versions match. Formal training additionally requires the recorded GPU/device to match the active environment. It reuses the four metadata files without rewriting them and exclusively creates a `.training_started` marker before optimization, preventing two processes from claiming the same metadata-only run. A directory containing any `best.pt`, `last.pt`, `training_log.json` or `training_status.json` is treated as started and blocks new training. Missing/extra files (including a leftover marker), symlinks and mismatched metadata also block reuse; no automatic deletion or migration is performed. `check`, `shape`, `smoke` and `dry-run` do not write to the formal run.

The read-only `--stage check` now reports `run_directory_state=metadata_only` and `new_training_allowed=true` for the observed directory. This permission is conditional on the formal training process using the same recorded CUDA GPU; the current audit process sees CPU, so GPU matching is deferred to the terminal start. No epoch or test evaluation was executed in this audit.
