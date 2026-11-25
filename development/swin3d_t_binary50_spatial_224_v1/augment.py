"""Clip-consistent spatial/color augmentation at 224x224 preserving exact SHA256 draws."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from development.aca_net_v1.data import MEAN, STD, require
from development.aca_spatial_v1 import augment as base_aug

FRAMES = 50
HEIGHT = 224
WIDTH = 224

VERSION = "clip_flip_color_light_v2_224x224"
DRAW_VERSION = base_aug.DRAW_VERSION  # 'clip_flip_color_light_v1' -> exact same draws per (seed, epoch, video_id)
POLICY = dict(
    version=VERSION,
    train_only=True,
    draw_version=DRAW_VERSION,
    frames=FRAMES,
    height=HEIGHT,
    width=WIDTH,
    resize_interpolation="cv2.INTER_AREA",
    resize_source_geometry="direct_full_frame_320x240_to_224x224_no_crop_no_120x160_intermediate",
    aspect_ratio_disclosure="4:3_to_1:1_full_frame_resize_preserves_entire_frame_including_rim",
    horizontal_flip_probability=0.5,
    brightness=[0.95, 1.05],
    contrast=[0.95, 1.05],
    saturation=[0.95, 1.05],
    hue=False,
    order=["flip", "brightness", "contrast", "saturation", "clamp_0_1", "ImageNet_normalize_once"],
    contrast_center="single_clip_gray_mean_after_brightness_numpy_float64_reduce_cast_float32",
    gray_weights=[0.2989, 0.5870, 0.1140],
    draw="SHA256_JSON_[draw_version,seed,zero_based_epoch,video_id,field]; first_53_bits / 2**53",
    parameter_scope="same_parameters_for_all_50_frames",
    validation="direct_full_frame_320x240_to_224x224_cv2_INTER_AREA_zero_aug",
    color_implementation="explicit_fixed_order_float32_blend_not_torchvision_random_ColorJitter",
)
POLICY_SHA = hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

parameters = base_aug.parameters
validate_params = base_aug.validate_params


def normalize_float_224(x: torch.Tensor) -> torch.Tensor:
    require(
        isinstance(x, torch.Tensor)
        and x.dtype == torch.float32
        and tuple(x.shape) == (3, FRAMES, HEIGHT, WIDTH)
        and x.device.type == "cpu",
        f"Invalid 224x224 RGB float clip: {type(x).__name__} {getattr(x, 'shape', None)}",
    )
    out = ((x - x.new_tensor(MEAN)[:, None, None, None]) / x.new_tensor(STD)[:, None, None, None]).contiguous()
    require(torch.isfinite(out).all().item(), "Non-finite normalized 224x224 tensor")
    return out


def normalize_rgb_224(rgb: np.ndarray) -> torch.Tensor:
    require(
        isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (FRAMES, HEIGHT, WIDTH, 3),
        f"Expected (50, 224, 224, 3) uint8 RGB clip, got {getattr(rgb, 'shape', None)}",
    )
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(3, 0, 1, 2).float() / 255.0
    return normalize_float_224(x)


def apply_rgb_224(rgb: np.ndarray, p: dict) -> torch.Tensor:
    require(
        isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.shape == (FRAMES, HEIGHT, WIDTH, 3),
        f"Expected resized (50, 224, 224, 3) uint8 RGB clip, got {getattr(rgb, 'shape', None)}",
    )
    validate_params(p)
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(3, 0, 1, 2).float() / 255.0
    if p["flip"]:
        x = x.flip(-1)
    if p["brightness"] != 1:
        x = x * float(p["brightness"])
    gray_weights = x.new_tensor(POLICY["gray_weights"])[:, None, None, None]
    if p["contrast"] != 1:
        center = x.new_tensor(float((x * gray_weights).sum(0).numpy().mean(dtype=np.float64)))
        x = x * float(p["contrast"]) + center * (1 - float(p["contrast"]))
    if p["saturation"] != 1:
        gray = (x * gray_weights).sum(0, keepdim=True)
        x = x * float(p["saturation"]) + gray * (1 - float(p["saturation"]))
    return x.clamp(0, 1).contiguous()


def transform_224(rgb: np.ndarray, split: str, seed: int, epoch: int, video_id: str, *, enabled: bool = True):
    require(split in ("train", "validation"), "Only train/validation")
    if split == "validation" or not enabled:
        return normalize_rgb_224(rgb), None
    p = parameters(seed, epoch, video_id)
    return normalize_float_224(apply_rgb_224(rgb, p)), dict(
        version=VERSION,
        policy_sha256=POLICY_SHA,
        seed=seed,
        epoch_index=epoch,
        video_id=video_id,
        parameters=p,
    )
