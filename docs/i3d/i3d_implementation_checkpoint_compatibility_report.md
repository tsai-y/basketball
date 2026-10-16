# I3D Implementation / Checkpoint Compatibility Report

Gate: `i3d_checkpoint_compatibility_gate_1c`  
Completed: 2026-10-17 (Asia/Taipei)  
Status: **PASS** for checkpoint/implementation compatibility and forward-only use; cross-framework endpoint parity remains **NOT_RUN**.

## Frozen selection and provenance

- Implementation/conversion: Gate 1B `C03`, `hassony2/kinetics_i3d_pytorch` commit `786fbae06a6fd5df419e1961cd32e32fbe8b396d`.
- Converted checkpoint: `model/model_rgb.pth`, RGB Inception-v1 Inflated 3D ConvNet, ImageNet-initialized then Kinetics-400, 400 classes.
- Official upstream: Gate 1B `A01`, `google-deepmind/kinetics-i3d` commit `0667e889a5904b4dc122e0ded4c332f49f8df42c`, TensorFlow/Sonnet RGB ImageNet checkpoint.
- Tested runtime: Python 3.10.20, PyTorch 2.11.0+cu128 on CPU, NumPy 2.1.3, OpenCV 4.13.0. CUDA was unavailable and was not required.
- License chain: MIT port plus upstream Apache-2.0 repository notice. The upstream release has no separate weights-specific grant; external derived-weight redistribution remains a limitation.

The acquisition manifest records canonical URLs, an empty redirect chain for each direct raw GitHub URL, exact bytes, SHA-256, acquisition time, commit and notice. Publisher checkpoint SHA-256 values were not provided; all checkpoint hashes below are locally computed raw-file identities, not Git blob SHA-1 or ETag.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| converted `model_rgb.pth` | 50,887,222 | `9729d6f029170c682ffe8d57ddda2ead49cf232297db7b7bdec0fa316fe6764a` |
| official `checkpoint` | 173 | `642bdb899ccd0655219ec352324b8f4c65e153b2ee847104691c1aee153c7398` |
| official `.data-00000-of-00001` | 50,818,176 | `f703e980b577f39ad69125cbb2ad5cc59f72b3f7606634ddb02db5bd45f786c0` |
| official `.index` | 10,953 | `f4bfd6f4af485f276c68333d3eca3697b691bb647f0167a03fe021147ca7f8eb` |
| official `.meta` | 51,144,099 | `a2049468749ceb21b84db071d4918e396cd802f622ebb65b6e5eb20fa25598d2` |

## Inventory, mapping and strict load

The converted artifact was deserialized with `torch.load(..., weights_only=True)` and contains only 287 finite float32 tensors. The matching Gate model has exactly 287 state slots:

- input convolution `[64,3,7,7,7]`, proving three RGB channels;
- 57 BatchNorm groups, each with gamma, beta, running mean and running variance;
- `Mixed_5c` output width 1024;
- classifier weight `[400,1024,1,1,1]` and bias `[400]`, proving a present 400-class Kinetics head.

The official TensorFlow checkpoint contains 230 tensors. The saved mapping uses every official tensor exactly once and no shape guessing: conv kernels use explicit transpose `[D,H,W,in,out] -> [out,in,D,H,W]`; beta and running statistics are identity copies; logits kernel/bias are mapped explicitly; and 57 BN gamma vectors are documented all-ones constants because the official Sonnet BatchNorm did not create scale tensors. Every converted value exactly matches its mapped source or declared constant. Unused official tensors, duplicate source uses and value mismatches are all zero.

`load_state_dict(..., strict=True)` succeeded: source tensors 287, model slots 287, missing 0, unexpected 0, unused converted source tensors 0, shape mismatches 0. A Gate-local eval-only legacy BatchNorm intentionally omits modern PyTorch's automatically synthesized `num_batches_tracked`, preventing a misleading “strict” success with 57 implicit buffers.

The pinned port source contains an unusable duplicate `mixed_4b` invocation in `forward()` that would send 512 channels into a 480-channel block. The Gate model corrects this to the declared official sequence (one call per endpoint), keeps checkpoint key names/state slots unchanged, and exposes endpoint extraction. This correction is explicit, not an unrecorded key rename or shape match.

## Input, endpoint and numerical smoke

Official-reference input is deterministic RGB float32 NCTHW `[1,3,64,224,224]` in `[-1,1]`, without ImageNet mean/std. Its `Mixed_5c` is `[1,1024,8,7,7]`; ordinary GAP is finite `[1,1024]`.

The Basketball-51 path starts from Data Pipeline Gate v1 contiguous float32 CTHW `[3,64,192,256]` in `[0,1]`, adds the batch axis, then applies exactly one explicit out-of-place `i3d_rgb_minus1_plus1_v1` adapter: `x*2-1`. Shape/dtype are preserved, the source digest is unchanged, and the output contract is `[-1,1]`. Its `Mixed_5c` is `[1,1024,8,6,8]`; ordinary GAP is finite `[1,1024]`. Both paths have measured temporal stride 8, spatial stride 32×32 and channel count 1024. The project path stops before the historical fixed `[2,7,7]` pool, so it is compatible with a 6×8 feature grid.

Repeated identical synthetic forwards were bitwise identical for `Conv3d_1a_7x7`, `Mixed_5c` and GAP. The machine summary records shape, finite state, min/max/mean/population-std, L1/L2 norm and little-endian float32 SHA-256 for every output.

Real smoke reused Data Pipeline Gate v1 for one train and one validation video. Both produced finite `[1,1024,8,6,8]` features and `[1,1024]` GAP vectors. Access counts were train 1/1 open/decode, validation 1/1, test **0/0**. The model remained in eval/inference mode, dropout was disabled, all BN running statistics were unchanged, and optimizer creation/steps, backward calls, gradient accumulation and training samples were all zero.

## Parity and final interpretation

Cross-framework numerical endpoint parity is `NOT_RUN`. Tolerances were frozen before acquisition (`atol=1e-4`, `rtol=1e-3`) for `Conv3d_1a_7x7` and `Mixed_5c`, but the exact historical Sonnet 1.x runtime is not available; TensorFlow 2.19 alone is not a reliable official reference runtime. The Gate therefore does not claim full official runtime equivalence. Exact source-to-converted tensor mapping is `PASS_EXACT` and is distinct from endpoint parity.

All Gate and Data Pipeline regression tests pass (7 + 8). No test sample was accessed, no frozen baseline/config/data-quality/data-pipeline file or historical run was modified, and no Git commit or push was made.

Conclusion: the repository now has the technical prerequisites for a separately specified **I3D ordinary GAP baseline** using the verified `Mixed_5c` endpoint and normalization adapter. That future experiment still needs its own training protocol, classifier, optimization policy, validation stopping rule and isolated outputs; Gate 1C itself performed no training.
