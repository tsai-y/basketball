# Basketball-51 Reference C3D Adaptation: Implementation and Gates

Date: 2026-09-30. Scope: clip extraction, C3D topology, fc6 video descriptor, clip verification, CUDA smoke, and a tiny train-only pipeline diagnostic. **No full-dataset training, validation model selection, or test evaluation was run.** Experiment A remains separate in `runs/full_video_c3d_style_v1/`.

## Sources and status labels

Primary specification: [C3D reproduction specification](c3d_reproduction_spec.md), derived from Tran et al., *Learning Spatiotemporal Features with 3D Convolutional Networks*, ICCV 2015 (local PDF `papers/c3d_tran_2015.pdf`) and the linked author Caffe prototxt. **PAPER_CONFIRMED** and **OFFICIAL_IMPLEMENTATION_CONFIRMED** below refer to that audit; **PROJECT_ADAPTATION** is specific to this Basketball-51 implementation. This is a reference-faithful **adaptation**, not a replication of the paper's benchmark or pretrained model.

## Files and fixed data

- `reference_c3d/data.py`: new `ContinuousClipSampler`, RGB preprocessing, video-level PyTorch Dataset. Experiment A's `UniformBinSampler` is untouched.
- `reference_c3d/model.py`: separate reference-topology network with fc6 extraction.
- `reference_c3d/aggregate.py`: independently callable mean-fc6-then-L2 aggregation.
- `configs/c3d_reference.yaml`: all clip, image, model, run, and diagnostic settings.
- `scripts/run_c3d_reference_checks.py`: only `verify`, `smoke`, and `subset` stages; it refuses Experiment A's run directory. Gate files include a hash of config and new source files.
- Uses existing `data/processed/dataset_manifest.csv` and `split_summary.json` without regeneration. Their SHA-256 values in this run are `e7bba1acd6028cf67cdf28251e69852e82cff201e13519ed05970ac483cce6b6` and `953da31313976763c8f5d597da5f0fe5a6a45e10f5d5edcd42fb0c120af2f7c5`. All gate samples are usable rows from the **existing train split**.

## Continuous clip extraction

**PAPER_CONFIRMED:** 16 consecutive frames per clip, 8-frame overlap; start stride 8 is derived from these values. `ContinuousClipSampler.indices(F)` emits starts `0, 8, 16, …` while `start + 16 ≤ F`.

**PROJECT_ADAPTATION:** Discard an incomplete final tail for all splits. This is deterministic and requires no frame replication or random padding; the source does not specify tail behavior. Videos shorter than 16 frames raise an explicit error. Index numbering in code/artifacts starts at zero.

Actual verification used the first usable Miss and Make train videos, both with 146 decoded frames. Each produced **17** clips and tensor shape `[17,3,16,112,112]`: first `[0..15]` (one-based `[1..16]`), second `[8..23]` (one-based `[9..24]`), last `[128..143]`; frames 144–145 are discarded. Every clip's full indices were checked for unit increments, and all starts for stride 8. Complete bounds and first two full index arrays are saved in `runs/full_video_c3d_reference_v1/clip_verification.json`.

Additional executed boundary check: frame counts 16, 23, 24, 146 produced 1, 1, 2, 17 complete clips respectively. An independent two-vector check confirmed `aggregate_fc6()` computes **mean first, then L2 normalization**; both checks passed.

## Preprocessing

| Setting | Implementation | Source status |
|---|---|---|
| RGB, 16 frames, resize to height 128 × width 171, 112×112 crop | OpenCV decode; resize each BGR frame to `(width=171,height=128)`, crop consistently across a video, convert BGR→RGB | RGB/resize/crop dimensions **PAPER_CONFIRMED**; OpenCV decoder and consistent crop across this video's clips **PROJECT_ADAPTATION** |
| Training crop/flip | Random spatial crop; horizontal flip probability 0.5, with seeded Python RNG | Random crop and 0.5 flip **PAPER_CONFIRMED**; per-video shared random draw **PROJECT_ADAPTATION** |
| Validation/test crop | Deterministic center crop, no flip | Center crop appears in the paper's Sports-1M clip classification; applying it to Basketball-51 validation/test is **PROJECT_ADAPTATION** |
| Pixel values | RGB float32 `((x/255)-0.5)/0.5`, range `[-1,1]` | **PROJECT_ADAPTATION**: original model mean-file contents, exact decoder color order, and pixel scale are **UNRESOLVED**. This transform must not be presented as source-compatible mean subtraction. |
| Gate preprocessing | Train manifest samples with deterministic validation-style center crop/no flip | **PROJECT_ADAPTATION** to make the smoke and subset checks repeatable; no validation/test samples were accessed. |

## Implemented architecture

**PAPER_CONFIRMED:** `conv1a 3→64; pool1; conv2a 64→128; pool2; conv3a 128→256; conv3b 256→256; pool3; conv4a 256→512; conv4b 512→512; pool4; conv5a 512→512; conv5b 512→512; pool5; fc6=4096; fc7=4096`. Convolutions use 3×3×3 kernel and stride 1. Pool1 uses 1×2×2 kernel/stride; pool2–5 use 2×2×2 kernel/stride.

**OFFICIAL_IMPLEMENTATION_CONFIRMED:** ReLU after each convolution and fc6/fc7, dropout 0.5 after fc6/fc7, spatial/temporal convolution padding 1, trainable conv/FC bias, Gaussian weight filler declarations (`0.01` conv, `0.005` fc6/fc7, `0.01` fc8), and official bias constants. The new model initializes these fillers; **no pretrained checkpoint was loaded**. No BatchNorm, LayerNorm, or global average pooling was added.

**PROJECT_ADAPTATION:** PyTorch max-pools use `ceil_mode: true`, giving pool5 `[batch,512,1,4,4]` from 112×112 input. The exact Caffe/PyTorch pool rounding equivalence remains to be verified. `fc8` has **two** outputs for Miss/Make; it is a task head and is not used to make the fc6 descriptor. The implemented model contains **78,003,970 parameters**, counted from its actual PyTorch tensors, including the two-class fc8 head. This is **not** claimed as the original C3D checkpoint's measured parameter count.

## fc6 feature and video aggregation

`ReferenceC3D.extract_fc6()` returns `[batch,4096]` at **fc6 linear → ReLU → before drop6**. This precise export point is **PROJECT_ADAPTATION** because the audited paper says “fc6 activations” but does not establish the exact exported blob state. `aggregate_fc6()` takes all per-video clip fc6 features, computes their arithmetic mean, then L2-normalizes that mean into one `[4096]` video descriptor. This **PAPER_CONFIRMED aggregation order** is separate from the paper's Sports-1M ten-clip prediction averaging; no probabilities or votes are aggregated here.

## Executed gates and artifacts

All gate outputs are in `runs/full_video_c3d_reference_v1/`; the directory is independent of Experiment A.

| Gate | Actual result |
|---|---|
| Clip verification | **PASS**, two train videos, 146 frames/17 clips each, complete index and shape checks. |
| CUDA smoke | **PASS** on RTX 5070 Ti. Input `[2,3,16,112,112]`; pool1 `[2,64,16,56,56]`; pool2 `[2,128,8,28,28]`; pool3 `[2,256,4,14,14]`; pool4 `[2,512,2,7,7]`; pool5 `[2,512,1,4,4]`; fc6 `[2,4096]`; descriptor `[4096]`, L2 norm `1.0`. `fc6.mean()` loss `1.1276508569717407`; conv1a/fc6 gradients finite and nonzero. Peak allocated GPU memory `935,785,984` bytes. |
| Small-subset first attempt | **FAIL**, preserved as `small_subset_attempt1.json`. Two train videos each yielded 17 clip features. Frozen randomly initialized C3D descriptors differed by only `0.00023732025874778628` after L2 normalization; raw linear diagnostic head reached accuracy 1.0 by step 190 but loss remained `0.6929361820220947` at step 200, above the required `<0.05`. |
| Small-subset corrected diagnostic | **PASS**, in `small_subset.json`. Only the two **train** descriptors were centered and divided by their global train-subset scale `0.00011866012937389314` before the diagnostic linear head. Accuracy `1.0`, loss `0.00010281255526933819` at recorded step 1. Both video descriptors have shape `[4096]`, are finite and have unit norm. |

The subset gate trains **only a tiny linear head on frozen, randomly initialized C3D descriptors**; no C3D backbone optimization or formal classifier fitting was performed. The perfect two-video result is a wiring/overfit diagnostic, **not an accuracy estimate or evidence that C3D learned useful basketball features**. The failed attempt remains visible. A later full training protocol still needs its own end-to-end optimization and validation gates.

The run directory contains `config.yaml`, `environment.json`, `clip_verification.json`, `smoke.json`, `small_subset_attempt1.json`, and `small_subset.json`; it contains **no trained C3D checkpoint or test metrics**. `environment.json` records manifest/split hashes, seed, PyTorch/CUDA versions, GPU and selected device.

## Remaining unresolved / decisions before full experiment

1. Verify whether an original C3D checkpoint and mean-file preprocessing can be used together; current weights are freshly initialized and RGB `[-1,1]` is a project choice.
2. Confirm exact Caffe pool rounding and fc6 input shape against official code/weights. `ceil_mode` is explicit rather than silently assumed paper fact.
3. Confirm author-exported fc6 activation point and test-time crop/clip position details.
4. Decide the full experiment's classifier protocol: paper-pattern binary linear SVM trained on train descriptors with validation-only selection, or a separately named end-to-end two-class adaptation. Neither formal protocol was trained here.
5. The existing split is source-group based; true game identity and near-duplicate leakage remain unverified. Full-video clips can show the outcome, so this is not pre-release prediction.

## Re-run only the permitted gates

From the project root with the existing `yolo11` environment:

```bash
conda run -n yolo11 python -B scripts/run_c3d_reference_checks.py --stage verify
conda run -n yolo11 python -B scripts/run_c3d_reference_checks.py --stage smoke
conda run -n yolo11 python -B scripts/run_c3d_reference_checks.py --stage subset
```

These commands update only `runs/full_video_c3d_reference_v1/`. The script has no train or test stage.
