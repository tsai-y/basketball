# Basketball-51 Make/Miss Video Classification Baseline protocol

Protocol frozen: 2026-09-27, before model results. Experiment A only. Sources and paper/implementation distinctions: [references.md](references.md). The local professor note `TOPIC 1.pdf` describes later possible work and contains placeholder results; none of those architectures or numbers belong to this baseline.

Actual gate, training and test results are reported separately in [baseline_results.md](baseline_results.md). The architecture amendment after the first failed overfit gate is documented below; the manifest and split were not changed.

## Research question

Can a modified C3D-style 3D CNN classify Make vs Miss from the **full six-second Basketball-51 RGB video**? This is a video classification baseline. It is **not** pre-release prediction or early shot outcome prediction.

## Dataset and cleaning

- Original: 10,311 H.264/MP4 videos in eight class directories; 320×240, 23.976/25/29.97 FPS, approximately six seconds. The original [Basketball-51 paper](https://csitcp.org/paper/11/117csit12.pdf) defines `2p0`, `3p0`, `ft0`, `mp0` as Miss (`0`) and `2p1`, `3p1`, `ft1`, `mp1` as Make (`1`). The local directory names agree. Raw totals: Miss 4,901, Make 5,410.
- Exact duplicates: 29 SHA-256 groups containing 58 files. Three groups (six files) disagree on binary outcome and are all excluded with `conflicting_outcome`. Of the remaining 26 two-file groups, the lexicographically first relative path is retained and the other copy excluded with `duplicate_copy`. This yields 32 excluded and **10,279 usable** videos: Miss 4,892, Make 5,387. Choosing one copy prevents duplicate weighting and guarantees exact copies cannot cross split. A same-outcome duplicate with differing original shot-type remains eligible because only the binary target is used; shot type metadata is not fed to the model.
- Original videos are untouched. [dataset_manifest.csv](../data/processed/dataset_manifest.csv) has one row for every source file. `video_path`/`video_id` are relative to the configured dataset root for portability; `game_code` and `source_timestamp` parse the filename; `source_group` encodes known `v160`/`v166` merging; `duplicate_group`, `excluded`, `exclusion_reason`, and `split` make all decisions auditable.

## Split

- [build_manifest.py](../scripts/build_manifest.py) uses fixed seed **20260927**. Fifty `source_group`s (51 `v` codes with `v160` and `v166` merged) are assigned in 35/8/7 group quotas to train/validation/test. Of 20,000 seeded candidate assignments, the one minimizing absolute deviations from target 70/15/15 total and Make counts is chosen **before model training**. This uses labels for stratification, never images, losses or test metrics. No resplitting after seeing model results.
- Counts from [split_summary.json](../data/processed/split_summary.json): train 7,193 (Miss 3,421; Make 3,772); validation 1,547 (Miss 740; Make 807); test 1,539 (Miss 731; Make 808).
- The split is **source-code grouped, not independently verified game-level**. The dataset provides 51 `v` codes and the paper says 51 games, but no original game identity table. Cross-code exact copies are known for `v160`/`v166` and merged. Other near duplicates, alternate broadcast edits, and repeats may remain; a verified game holdout claim requires external game mapping and visual near-duplicate audit. Exact duplicate groups, conflicting labels, and known duplicate source group cannot cross usable splits.

## Preprocessing and temporal sampler

- RGB only. Fixed **16 frame** input follows the temporal dimension of the [C3D paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf). All local clips have at least 145 frames, so 16 temporal bins cover the whole video without padding. Validation/test select bin centers deterministically; training draws one frame per bin using seeded random jitter. This is a **modified full-video sampling strategy**, not the paper's contiguous 16-frame crop. One 16-frame tensor and one prediction per source video; no extra overlapping clips enter separate splits.
- Decode all frames in file order with OpenCV, convert BGR to RGB, resize with preserved aspect ratio and black padding to **96×96**, divide by 255 and apply fixed `(x - 0.5)/0.5` normalization. No dataset-wide statistics or test-derived preprocessing. Training only: horizontal flip probability 0.5, using seeded worker RNG. Validation/test: no augmentation. Frame paths, directory strings, class names, timestamps and labels are returned as metadata only, never concatenated to tensors.
- Temporal sampling is a separate component from the Dataset so a future experiment can replace its strategy. Only Experiment A's full-video strategy is implemented this round.

## Model

- **Modified/simplified C3D-style network**, initialized from scratch; not strict C3D reproduction. The [Tran et al. ICCV 2015 paper](https://openaccess.thecvf.com/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html) has eight 3×3×3 convolution layers, five max-pooling layers with first temporal pooling of 1, two 4096-unit FC layers, 112×112 crops, and Sports-1M pretraining/training. This project retains 3D convolutions and five-pool temporal pattern but reduces channel widths and the head, adds BatchNorm3d after each convolution, uses adaptive global average pooling, 96×96 padded video frames, full-video temporal bins, and two output logits. The two-class output is Miss `0`, Make `1`. BatchNorm was added after the first eight-video overfit gate stalled near 0.693 loss; the input variance and labels had been checked and were valid. Both gates must be rerun after this change.
- Parameter count: measured from actual model at run time, not asserted in advance.

## Training

- Configuration: [baseline.yaml](../configs/baseline.yaml). Seed Python, NumPy, PyTorch, CUDA; set cuDNN deterministic mode and request deterministic algorithms with warnings for unsupported operations; record environment. Device preference is CUDA, with CPU fallback. On the available PyTorch/CUDA build, `max_pool3d` backward lacks a deterministic CUDA implementation, so bitwise reproducibility of GPU training cannot be guaranteed even with fixed seeds. This was found by the first smoke test; strict deterministic mode failed at backward, then the warning mode was adopted without changing the model.
- Optimizer AdamW, initial learning rate `0.001`, batch size `16`, CrossEntropyLoss, maximum `5` epochs, four DataLoader workers. These are project choices, not C3D-paper reproduction. No learning-rate or architecture search in this run.
- Per epoch log train/validation loss and accuracy. Choose best checkpoint by **lowest validation loss**, with first epoch winning ties. Save model, config, manifest copy, split summary, random seed, environment, and log. Run smoke test and tiny-subset overfit gate first; if either fails, stop without full training. Test is not read for model/epoch/preprocessing selection.

## Evaluation

- Evaluate the fixed best-validation checkpoint once on test: accuracy, precision, recall, F1 and confusion matrix. **Make (`1`) is positive**. Save one row per video with relative `video_id`, ground truth, prediction and Make probability. Report actual values only after successful execution.

## Limitations

- The full clip includes ball trajectory, rim contact, net motion, reactions and possibly score graphics or replay. A high score can reflect visible outcome rather than anticipation from shooting motion. This experiment must be called **Basketball-51 Make/Miss Video Classification Baseline**, never pre-release or early outcome prediction.
- The source-group split is not verified game identity. Exact-hash cleaning cannot detect visually equivalent re-encodes or broadcast replay. Original labels may contain further mistakes beyond observed exact-hash conflicts. Fixed temporal bin sampling and 96×96 input differ from published C3D and Basketball-51 model protocols; paper numbers are not directly comparable.
