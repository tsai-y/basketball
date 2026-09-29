# Experiment A: Basketball-51 Make/Miss Video Classification Baseline

Run completed 2026-09-27. These numbers come from the fixed manifest, five-epoch training log and one test evaluation in `runs/full_video_c3d_style_v1/`. Positive class: **Make = 1**. This is a full-video classifier, **not** pre-release shot outcome prediction.

## Data and split

| Stage | Miss | Make | Total |
| --- | ---: | ---: | ---: |
| Raw | 4,901 | 5,410 | 10,311 |
| Excluded | 9 | 23 | 32 |
| Train | 3,421 | 3,772 | 7,193 |
| Validation | 740 | 807 | 1,547 |
| Test | 731 | 808 | 1,539 |

The 32 exclusions are six files in three binary-label-conflict duplicate groups plus 26 consistent-label exact-duplicate copies. The split is by filename source group, with `v160`/`v166` merged. **True game identity is not independently verified.** Seed: `20260927`. The reproducible split assignment is saved in `data/processed/split_summary.json`; all decisions are in `data/processed/dataset_manifest.csv`.

## Preprocessing and model

16 RGB frames sampled across the whole video using uniform temporal bins. Train samples randomly within bins and may horizontally flip; validation/test use fixed bin centers. Aspect-preserving resize/pad to 96×96, fixed RGB normalization to [-1, 1]. Modified C3D-style model with eight 3×3×3 convolutions, BatchNorm3d, five max-pools, global average pooling and binary head; **6,951,874 trainable parameters**. See [protocol](baseline_protocol.md) and [references](references.md) for paper differences.

## Gates and debugging record

- First smoke attempt **failed** because this PyTorch/CUDA build lacks deterministic `max_pool3d` backward. Changed deterministic enforcement to warn for unsupported operations while keeping fixed seeds and max pooling. The subsequent two-class GPU smoke test **passed**: input `[2,3,16,96,96]`, output `[2,2]`, labels `[0,1]`, loss `0.732536`, finite gradients and successful optimizer step. Peak allocated GPU memory: 356,121,088 bytes.
- First eight-video overfit attempt **failed** with subset accuracy 0.75 at 120 steps and eval loss 0.692883. Verified actual input variation and correct 4/4 labels. Added BatchNorm3d after convolutions, recorded as an architecture modification, and reran both gates.
- Final eight-video overfit test **passed** at step 40: subset accuracy 1.0 and subset eval loss 0.000387. This is a pipeline check, not a research accuracy result. Peak allocated GPU memory: 924,558,848 bytes.

## Training

AdamW, learning rate 0.001, weight decay 0.0001, batch size 16, five epochs, CrossEntropyLoss. Best checkpoint selected by minimum validation loss. Train/validation logs:

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.655424 | 0.616016 | 0.677480 | 0.563672 |
| 2 | 0.614763 | 0.670096 | 0.584887 | 0.688429 |
| 3 | 0.593522 | 0.694842 | 0.574223 | 0.699418 |
| 4 | 0.568486 | 0.716530 | 0.556761 | 0.717518 |
| **5** | **0.546568** | **0.729737** | **0.525438** | **0.760181** |

Best epoch: **5**. Training elapsed 319.81 seconds. Peak allocated GPU memory: **1,648,338,432 bytes** (1.535 GiB). Full checkpoint, copied manifest/config, environment, and logs are in `runs/full_video_c3d_style_v1/`.

## One-time test evaluation

| Metric | Value |
| --- | ---: |
| Accuracy | 0.747888 |
| Precision, Make | 0.711694 |
| Recall, Make | 0.873762 |
| F1, Make | 0.784444 |

Confusion matrix, rows = true `Miss, Make`, columns = predicted `Miss, Make`:

| | Predicted Miss | Predicted Make |
| --- | ---: | ---: |
| True Miss | 445 | 286 |
| True Make | 102 | 706 |

`runs/full_video_c3d_style_v1/test_predictions.csv` has 1,539 unique relative video IDs, ground truth, predicted class and probability of Make. A post-run audit recomputed the confusion matrix, checked every prediction against the manifest label, and confirmed the copied manifest SHA-256 matches the training record. No test value was used to choose an epoch, split, architecture or preprocessing.

## Environment and limits

- Ran in existing `yolo11` Conda environment; **no packages installed or upgraded**. Python 3.10.20, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.1.3, OpenCV 4.13.0, PyYAML 6.0.3, CUDA available, NVIDIA GeForce RTX 5070 Ti. The `basketball` environment exists but lacks the required training packages. scikit-learn was not required; metrics were computed directly from counts.
- GPU `max_pool3d` backward is not deterministic in this build; identical seeds need not yield bitwise-identical weights. The fixed manifest, split, code, config and per-video test protocol remain auditable.
- The complete clip reveals ball flight, rim/net response and possibly broadcast graphics or reactions. These results measure full-video Make/Miss classification only. Unverified true game identity, near duplicates and other label errors may affect the test estimate. This run is not an exact reproduction of either the C3D or Basketball-51 paper.
