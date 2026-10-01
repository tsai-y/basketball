# A2 / B / C Formal Test Evaluation Protocol

Date: 2026-10-01. **Formal evaluation COMPLETE.** A2, B and C were each evaluated once using their validation-selected frozen `best.pt`. The existing `test_metrics.json` files are the authoritative numerical records. Do not retrain, retune, re-evaluate or overwrite any checkpoint, preprocessing, split, metrics or predictions.

## Frozen inputs and integrity gate

Use [`scripts/evaluate_full_video_experiments.py`](../scripts/evaluate_full_video_experiments.py). Each invocation checks all three completed runs, their unchanged config/manifest/split snapshots, source compatibility, training status and validation log. Formal inference loads only the selected run's `best.pt`; there is no `last.pt` fallback. Required best epochs are A2 **21**, B **25**, C **4**. B's checkpoint lacks a direct manifest hash, so its run manifest/split snapshots, status manifest hash, and frozen config are checked together.

The metadata-only `--check` creates the three test Dataset objects but never calls `__getitem__`, creates a DataLoader iterator, or decodes a test video. It compares the sorted identities resolved by A2/B/C, requires exactly **1,539** unique test videos and the fixed label counts **731 Miss / 808 Make**, excludes every non-test or excluded row, verifies the eight-class `0=Miss,1=Make` mapping, and checks disjointness from train/validation IDs. The checked sorted-ID SHA-256 is `2c6c37cc846179aea27e1b2a0726a4210c50a184357792202791ddd9752d3eba`.

## Frozen deterministic preprocessing

| Experiment | Test input | Preprocessing |
| --- | --- | --- |
| A2 | One `[3,16,96,96]` tensor per video | A2/A1 `UniformBinSampler(jitter=False)` takes each bin center; RGB letterbox and `[-1,1]`; no flip. |
| B | All complete continuous `[3,16,112,112]` clips | Stride 8, discard incomplete tail; resize 128×171, center crop 112, RGB `[-1,1]`; no random crop or flip. Extract fc6, mean, L2, then frozen `Linear(4096,2)`. |
| C | Same clip indices and spatial preprocessing as B | Extract 256-D GAP feature, mean, L2, then frozen `Linear(256,2)`. |

No probability threshold is tuned. The predicted class is `argmax(logits)`. Make (`1`) is the positive class. The evaluator uses `model.eval()` and `torch.inference_mode()` and contains no optimizer or parameter update path.

## Metrics and artifacts

Each one-time run computes video-mean cross-entropy test loss, accuracy, Make precision/recall/F1, and a confusion matrix with rows `true [Miss=0, Make=1]` and columns `predicted [Miss=0, Make=1]`. It records total, true-class, and predicted-class counts. The prediction CSV contains only relative `video_id`, `true_label`, `predicted_label`, and `probability_make` (softmax class 1). Each experiment writes `test_metrics.json` and `test_predictions.csv` in its own existing run directory only after all 1,539 predictions are available.

The formal path refuses any existing test output or `.test_evaluation_claim`. It exclusively creates the claim before decoding and opens final outputs in exclusive-create mode; there is no `--force`. All three runs now contain their one-time claim, `test_metrics.json` and `test_predictions.csv`; these are **immutable formal result artifacts**. Do not delete the marker, rerun evaluation, or use test outcomes to change checkpoint, architecture, preprocessing, threshold or training budget.

## Final formal results

Values below are copied from the existing result JSON files. Confusion matrices have rows true `[Miss, Make]` and columns predicted `[Miss, Make]`.

| Experiment | Best epoch | Test loss | Accuracy | Precision Make | Recall Make | F1 Make | Confusion matrix |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A2 | 21 | 0.5014848665729446 | 0.7738791423001949 | 0.7572706935123042 | 0.8378712871287128 | 0.7955346650998824 | `[[514,217],[131,677]]` |
| B | 25 | 0.5671578797525675 | 0.7342430149447693 | 0.7484433374844334 | 0.7438118811881188 | 0.7461204220980757 | `[[529,202],[207,601]]` |
| C | 4 | 0.6629814902747738 | 0.6049382716049383 | 0.6265822784810127 | 0.6126237623762376 | 0.6195244055068836 | `[[436,295],[313,495]]` |

Each result records **1,539** test videos: **731 Miss** and **808 Make**. A2/B/C result JSONs record identical manifest SHA-256 `e7bba1acd6028cf67cdf28251e69852e82cff201e13519ed05970ac483cce6b6`, split-summary SHA-256 `953da31313976763c8f5d597da5f0fe5a6a45e10f5d5edcd42fb0c120af2f7c5`, and sorted test-ID SHA-256 `2c6c37cc846179aea27e1b2a0726a4210c50a184357792202791ddd9752d3eba`. The prediction files remain in their respective run directories and must not be edited.

A2/B/C compare **complete experimental pipelines**, not controlled single-variable ablations. A2 vs C changes spatial preprocessing, aggregation/head and initialization along with temporal sampling; C vs B changes backbone, feature dimension and training history. The observed ranking does not establish that sparse sampling is inherently better than continuous sampling or that Modified C3D is inherently better than Reference C3D. All four experiments classify **full videos** that may contain post-release ball flight, rim/net interaction and reactions. None demonstrates pre-release prediction capability. Test scores must not be used to retune A2/B/C.

## Read-only checks performed before formal evaluation

All three commands below **PASS**; they reported identical test IDs, strict state-dict compatibility, the expected best epoch, no output collision, `test_video_decoded=false`, `test_metrics_computed=false`, and `output_files_written=false`:

```bash
conda run -n yolo11 python -B scripts/evaluate_full_video_experiments.py --experiment a2 --check
conda run -n yolo11 python -B scripts/evaluate_full_video_experiments.py --experiment b --check
conda run -n yolo11 python -B scripts/evaluate_full_video_experiments.py --experiment c --check
```

## Historical formal commands — completed once; do not rerun

```bash
conda run -n yolo11 python -B scripts/evaluate_full_video_experiments.py --experiment a2
conda run -n yolo11 python -B scripts/evaluate_full_video_experiments.py --experiment b
conda run -n yolo11 python -B scripts/evaluate_full_video_experiments.py --experiment c
```
