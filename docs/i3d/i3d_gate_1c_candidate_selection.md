# Gate 1C Candidate Selection Decision

Date: 2026-10-17 (Asia/Taipei). Decision: select Gate 1B `C03` with upstream `A01`.

`C03` is the only registered RGB, Kinetics-400, ImageNet-initialized PyTorch Inception-v1 I3D candidate in Gate 1B that also preserves a pinned conversion implementation. `C01` has a widely used checkpoint but no conversion script in its pinned tree, so it cannot satisfy this Gate's source-tensor mapping requirement. `B01` changes padding/GAP/head behavior and was already ineligible; `B02` is ResNet-50 and is not the required architecture. RGB scratch, flow and Kinetics-600 candidates do not match the stated priority.

The selected chain is:

1. Official `google-deepmind/kinetics-i3d` commit `0667e889a5904b4dc122e0ded4c332f49f8df42c`, checkpoint `data/checkpoints/rgb_imagenet/model.ckpt*` (`A01`).
2. Third-party `hassony2/kinetics_i3d_pytorch` commit `786fbae06a6fd5df419e1961cd32e32fbe8b396d`, conversion code `i3d_tf_to_pt.py` + `src/i3dpt.py`, checkpoint `model/model_rgb.pth` (`C03`).
3. Gate-local modern-runtime topology with identical checkpoint keys and state slots, an explicit correction for the pinned source's duplicate `Mixed_4b` forward call, and an eval-only legacy BatchNorm state contract.

The downloaded official checkpoint contains 230 tensors. Exact mapping proves each official tensor is consumed once: 57 convolution weights are transposed `[D,H,W,in,out] -> [out,in,D,H,W]`, beta/moving-mean/moving-variance are copied, and logits weight/bias are copied. Sonnet's absent BN scale is represented by 57 predeclared all-ones gamma tensors. All 287 converted tensors match these source or derived values exactly. This establishes parameter-conversion identity, not cross-framework endpoint numerical parity.

License notice chain: the port is MIT; upstream repository root is Apache-2.0. No separate weight-specific grant was found, so external redistribution of derived weights remains a limitation even though local research validation proceeds with both notices preserved.
