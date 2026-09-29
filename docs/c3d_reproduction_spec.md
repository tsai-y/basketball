# C3D Reproduction Specification

Audit date: 2026-09-29. This document specifies a future implementation; no model or training run was changed in this audit.

## Source Paper

Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. “Learning Spatiotemporal Features with 3D Convolutional Networks.” *IEEE International Conference on Computer Vision (ICCV)*, 2015, pp. 4489–4497. [CVF record](https://openaccess.thecvf.com/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html); local complete paper and supplement: [`papers/c3d_tran_2015.pdf`](../papers/c3d_tran_2015.pdf). Page numbers below are printed article pages; PDF physical pages are printed page minus 4488. The DOI remains unverified.

Source labels: **PAPER_CONFIRMED** means the PDF states the setting; **OFFICIAL_IMPLEMENTATION_CONFIRMED** means an author-released Caffe definition states it; **PROJECT_ADAPTATION** means a proposed Basketball-51 choice; **UNRESOLVED** means the source does not determine it. A derived size/stride is identified as arithmetic, not a separate paper claim.

The checked official files are from the [author-released C3D repository](https://github.com/facebookarchive/C3D): [Sports-1M video feature extractor prototxt](https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/examples/c3d_feature_extraction/prototxt/c3d_sport1m_feature_extractor_video.prototxt) and [UCF101 fine-tuning train prototxt](https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/examples/c3d_finetuning/c3d_ucf101_finetuning_train.prototxt). The former describes a **487-class Sports-1M network**; the latter describes a separate **101-class fine-tuning example**. They verify architectural implementation details, but do not prove every published experiment used identical preprocessing or weights.

## Paper-Confirmed Architecture

| Setting | PAPER_CONFIRMED evidence | Meaning / limit |
|---|---|---|
| Tensor notation and RGB | p. 4491, §3.1, “Notations” and “Common network settings” | `c × l × h × w`: channels, frames, height, width. `3 × 16 × 128 × 171` is the resized RGB clip before crop; `3 × 16 × 112 × 112` is the training crop. The paragraph initially describes the **five-convolution UCF101 architecture search**; its non-overlapping clip rule is not the later C3D video-descriptor rule. |
| Final C3D input and training crop | p. 4492, §3.3, “Training” | Sports-1M frames resized to `128 × 171`; random `16 × 112 × 112` crop with spatial and temporal jitter. Together with the RGB convention, network crop is `3 × 16 × 112 × 112`. Horizontal flip probability 0.5. |
| Final network depth | p. 4492, §3.3; p. 4493, Fig. 3 and caption | Eight 3D convolution layers, five max-pool layers, two fully connected layers, then softmax output. |
| Convolution sequence | p. 4493, Fig. 3 | `conv1a:64 → conv2a:128 → conv3a:256 → conv3b:256 → conv4a:512 → conv4b:512 → conv5a:512 → conv5b:512` output channels. All use `3×3×3` kernels, temporal and spatial stride 1. RGB gives conv1a three input channels. |
| Pooling sequence | p. 4493, Fig. 3 and caption; p. 4492, §3.3 | Max-pool after conv1a, conv2a, conv3b, conv4b, conv5b. Pool1 kernel and stride `1×2×2`; pool2–pool5 kernel and stride `2×2×2`. The p. 4492 text states these strides, and explains Pool1 preserves early temporal information. |
| Temporal sizes | Arithmetic from the above paper-confirmed pool strides | For 16 frames, `16 → 16 → 8 → 4 → 2 → 1`. This is a derivation, not a separately measured output. |
| Fully connected layers | p. 4493, Fig. 3 and caption | fc6 and fc7 each have 4096 units; a softmax classification output follows. The Sports-1M experiment has 487 categories (p. 4492, §3.3, “Dataset”), so its head is for that task, not Make/Miss. |
| Training initialization regimes | p. 4492, §3.3, “Training” and Table 2 | Reports both C3D trained from scratch on Sports-1M and C3D fine-tuned from an I380K pretrained model. The paper does **not** specify the numerical random weight filler. |
| Sports-1M clip classification | p. 4492, §3.3, “Sports-1M classification results” | One center crop per clip; video prediction averages predictions of ten randomly extracted clips. This is **prediction averaging**, separate from the fc6 descriptor method below. Exact test random draws/seed are unstated. |
| Generic video descriptor | p. 4493, §3.3, “C3D video descriptor” | Split a video into continuous 16-frame clips, with 8-frame overlap; run each clip through C3D; average fc6 activations into a 4096-D video vector; then L2-normalize. Start-to-start stride 8 is arithmetic from length 16 and overlap 8. |
| Action recognition classifier | p. 4493, §4, “Classification model” | C3D video descriptors are supplied to a multi-class **linear SVM** for UCF101 action recognition. This is not the Sports-1M softmax evaluation. |

The paper says “appropriate padding” for the architecture-search networks (p. 4491, §3.1), but gives no final C3D per-layer numeric padding there. The figure does not state ReLU, dropout, bias, numeric initialization, or normalization layers. Those details require the official definition or remain unresolved; they are not PAPER_CONFIRMED.

## Official-Implementation Details

These observations refer to the two linked official prototxt files, not to prose in the paper.

| Setting | OFFICIAL_IMPLEMENTATION_CONFIRMED evidence | Relation to paper |
|---|---|---|
| Activation | The Sports-1M definition contains `relu1a`, `relu2a`, `relu3a`, `relu3b`, `relu4a`, `relu4b`, `relu5a`, `relu5b`, `relu6`, `relu7`, each immediately after its named convolution or FC layer. The UCF101 training definition has the same pattern. | Adds detail absent from Fig. 3. |
| Dropout | `drop6` follows fc6/ReLU and `drop7` follows fc7/ReLU; both have `dropout_ratio: 0.5` in the official definitions. | Adds detail absent from paper; the inference prototxt also lists these layers, but dropout is a training-time operation. |
| Conv padding | Every listed convolution has spatial `pad: 1` and `temporal_pad: 1`, with kernel size/depth 3. | Numerically resolves paper’s “appropriate padding” for these checked definitions. |
| Bias | Convolution and inner-product layers declare `bias_filler` and a separate bias learning-rate blob (`blobs_lr: 2`), indicating trainable bias parameters. | Bias is not stated in paper. |
| Initialization declaration | Conv layers declare Gaussian weight filler, `std: 0.01`; fc6/fc7 Gaussian `std: 0.005`; fc8 Gaussian `std: 0.01`. Bias constants are 0 for conv1a/fc8 and 1 for conv2a–conv5b/fc6/fc7 in these definitions. | These are prototxt **declarations**. A run that loads pretrained weights uses those weights instead; the declarations alone cannot establish the exact checkpoint initialization history. |
| Normalization layers | Neither checked network definition includes BatchNorm or LayerNorm layers. The Sports-1M video data layer specifies `mean_file: sport1m_train16_128_mean.binaryproto`, which is input mean subtraction, not feature BatchNorm. | Supports a reference network **without** BatchNorm/LayerNorm for these definitions; does not prove no other unretrieved variant exists. |
| Example input data layer | Sports-1M feature extractor: `new_length: 16`, `new_height: 128`, `new_width: 171`, `crop_size: 112`, `mirror: false`, `shuffle: false`; UCF101 fine-tuning train example has `mirror: true`, `shuffle: true`. | Consistent with clip length and resized/cropped dimensions. The exact Caffe test crop and clip-position algorithm are not fully specified by these fields. |
| Example classification head | Sports-1M feature extractor `fc8-1` has 487 outputs and softmax; UCF101 fine-tuning train example `fc8` has 101 outputs. | Shows task-specific heads; Basketball-51 requires two outputs if using an end-to-end binary classifier. |

## Unresolved Details

| Question | Status and next verification |
|---|---|
| ReLU | **Resolved by official implementation**, not by paper text. Exact ten ReLU positions are listed above. |
| Dropout | **Resolved by official implementation** for the checked definitions: after fc6 and fc7, probability 0.5. |
| Padding | **Resolved by official implementation** for conv layers: spatial and temporal 1. Pooling uses implementation-specific output rounding; verify Caffe vs PyTorch shapes before porting fc6. |
| Bias | **Resolved by official implementation** for the checked conv/FC definitions: bias filler and bias blob present. Exact PyTorch port must preserve biases. |
| Weight initialization | **Partly resolved**: official fillers are above, paper states from-scratch versus pretrained/fine-tuned regimes. **UNRESOLVED:** which precise weights/seed produced each published model, and whether a downloadable checkpoint exactly matches the paper’s descriptor network. |
| Test-time clip sampling | **Partly resolved by paper**: Sports-1M classification uses 10 randomly extracted clips and center crop; generic descriptor uses all specified overlapping clips. **UNRESOLVED:** random seed/positions for 10 clips, end-of-video tail handling, short-video padding, and exact Caffe data-layer crop behavior. |
| Input color order and pixel scaling | **UNRESOLVED:** the paper says RGB; the checked Caffe data layer points to a mean file, but exact decoder channel order, numerical mean contents, and pixel scale for the published weights were not validated. Do not assume the current project’s `[-1,1]` transform is equivalent. |
| Exact original parameter total | **UNRESOLVED:** not reported in the cited paper passages and not measured from an official loaded model in this audit. Caffe/PyTorch pooling rounding affects the fc6 input width. Do not present an inferred total as measured. |
| Exact video descriptor activation point | The paper says “fc6 activations.” The prototxt names fc6, in-place ReLU, and dropout. **UNRESOLVED:** whether published feature export selected the post-ReLU blob before dropout or another internal state; verify the author extraction script/checkpoint path before implementation. |

## Video Clip Processing

The **PAPER_CONFIRMED generic descriptor path** (p. 4493, §3.3) is:

```text
full video
  → consecutive 16-frame RGB clips, next start 8 frames later (8-frame overlap)
  → same C3D applied to each clip
  → per-clip fc6 feature
  → arithmetic average of clip features
  → 4096-D descriptor
  → L2 normalization
  → task classifier (linear SVM in paper §4 action-recognition experiments)
```

For `F ≥ 16`, the complete clips defined by starts `0, 8, 16, …` satisfy `start + 16 ≤ F`. The paper does not specify how to handle a remaining tail or clips shorter than 16; these are **UNRESOLVED** and need a documented project policy. The UCF101 five-layer architecture search uses **non-overlapping** clips (p. 4491, §3.1); do not apply that rule to the final descriptor. The Sports-1M **video classification** path instead averages predictions from ten random clips (p. 4492, §3.3); it is a different evaluation method.

## Current Baseline Differences

Current values below were read from `baseline/data.py`, `baseline/model.py`, `configs/baseline.yaml`, and the completed run report. “Reference” combines paper statements with explicitly identified official details.

| Item | Current Modified C3D-style | Reference C3D |
|---|---|---|
| Input frames | One 16-frame tensor per complete video | Continuous 16-frame clips; multiple clips per video for descriptor (paper p. 4493) |
| Frame sampling | One frame from each of 16 bins over the full ~6 s video; seeded train jitter, fixed bin centers for validation/test | Consecutive frames within each clip; 8-frame overlap for descriptor; 10 random clips for separate Sports-1M prediction experiment |
| Resolution | Aspect-preserving letterbox to `96×96` | Resize frame to `128×171`, crop `112×112` (paper §3.3) |
| Pixel preprocessing | RGB, normalize to `[-1,1]` | Official input layer specifies mean file; exact equivalence to project transform unverified |
| Conv layers | 8 | 8 (paper Fig. 3) |
| Channels | 32, 64, 128, 128, 256, 256, 256, 256 | 64, 128, 256, 256, 512, 512, 512, 512 (paper Fig. 3) |
| Kernel / conv stride / padding | `3×3×3` / 1 / 1 | `3×3×3` / 1 (paper); spatial and temporal pad 1 (official definition) |
| Pooling | 5 max-pools; `(1,2,2)` then four `(2,2,2)` | Same kernel/stride pattern (paper Fig. 3) |
| BatchNorm | After all 8 convolutions | No BatchNorm/LayerNorm in checked official definitions |
| ReLU | After all 8 convolutions and the 128-unit head layer | After each conv and fc6/fc7 (official definitions) |
| Global Average Pool | `AdaptiveAvgPool3d(1)` | Absent; pool5 output is flattened into fc6 (official definitions) |
| FC6 / FC7 | Absent | 4096 / 4096 (paper Fig. 3) |
| Dropout | 0.3 after 128-unit head layer | 0.5 after fc6 and fc7 (official definitions) |
| Output head | 256→128→2 logits, Make/Miss | Sports-1M official example fc8=487 + softmax; head is task-specific |
| Video aggregation | None; one output per uniformly sampled video | Default descriptor: mean fc6 then L2 norm; §4 uses linear SVM. Separate Sports-1M evaluation averages 10 clip predictions. |
| Parameters | **6,951,874**, measured in completed project run | **Not measured in this audit**; Caffe pooling output shape and task-specific head must be fixed before comparing a total |

## Basketball-51 Adaptation Plan

This is a **PROJECT_ADAPTATION** specification for a later turn, not an implemented method.

1. Retain the paper’s 8-conv, 5-pool, 4096/4096-FC topology and continuous 16-frame input. Port official ReLU, dropout, conv padding, and bias settings. Verify intermediate tensor shapes against the official Caffe definition before comparing parameter counts.
2. Retain the fixed cleaned Basketball-51 manifest and train/validation/test assignment. Make=`1`, Miss=`0`; keep exact duplicates and known source groups inside one split. Never generate overlapping clips before assigning videos to splits.
3. Use multiple clips per video, with an explicit 8-frame-overlap descriptor mode. Decide and document tail handling, short-video handling, FPS policy, color/mean preprocessing, and deterministic validation/test behavior. The source paper’s settings were developed for other datasets and do not automatically determine these Basketball-51 choices.
4. Define **one** task output protocol before training: either (a) C3D fc6 mean → L2 → **binary linear SVM**, following the paper’s descriptor/classifier pattern with a new binary target, or (b) C3D clip classifier with a **two-output** head and a declared video aggregation rule. Option (b) is a stronger Basketball-51 adaptation and is not the paper’s fc6+SVM descriptor. Do not silently replace mean fc6 with mean softmax probabilities.
5. Choose training initialization (from scratch or verified C3D pretrained weights) and schedule only after weight provenance and preprocessing compatibility are checked. Report a “reference-faithful C3D adaptation on Basketball-51,” never an exact reproduction of the paper’s benchmark scores. Full-video clips can include ball flight and result frames; this is not pre-release prediction.

## Reproduction Checklist

- [x] Input verified — paper’s 16 RGB frames, resize and training crop; unresolved dataset-specific tail/test policy documented.
- [x] Conv verified — paper Fig. 3 topology/channels/kernel/stride.
- [x] Pool verified — paper Fig. 3 kernel/stride and derived temporal sizes; framework shape rounding still to test.
- [x] Activation verified — official prototxt, not paper prose.
- [x] Dropout verified — official prototxt, not paper prose.
- [x] Padding verified — official conv prototxt; pool rounding still to test.
- [x] FC verified — paper’s 4096/4096 and official task heads.
- [x] Clip extraction verified — paper’s continuous 16-frame, 8-frame-overlap descriptor rule; tail behavior unresolved.
- [x] Video aggregation verified — paper’s mean fc6 + L2 descriptor, separately from ten-clip prediction averaging.
- [ ] Basketball-51 adaptation defined — alternatives and constraints are recorded, but head, weights, preprocessing, and tail policy require a final decision before implementation.
