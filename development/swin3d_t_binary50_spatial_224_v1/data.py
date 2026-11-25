"""Isolated 224x224 direct full-frame cv2.INTER_AREA dataset, collator, and epoch stream.

Preserves exact split isolation, label revision, stratified50 train indices, uniform50
validation indices, and SHA256 spatial augmentation draws; updates all resize and shape
guards consistently to (3, 50, 224, 224).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os

import cv2
import numpy as np
import torch

from development.aca_binary_v1.data import RevisedDataset
from development.aca_binary_v1.runtime import EpochStream as BaseStream
from development.aca_eight_class_data_v1.gpu_readiness import tensor_sha
from development.aca_net_v1.data import _identity, gate1a, require, uniform_indices
from development.aca_temporal_v1 import sampling
from development.adjudicated_input_v1 import adapter as labels

from . import augment

SEED = 20261024
FRAMES = augment.FRAMES
HEIGHT = augment.HEIGHT
WIDTH = augment.WIDTH
need = require

INPUT_CONTRACT_224 = {
    "schema": "basketball51_direct_full_frame_224x224_inter_area_v1",
    "frames": FRAMES,
    "height": HEIGHT,
    "width": WIDTH,
    "channels": 3,
    "color_order": "RGB",
    "source_frame_expected": [240, 320, 3],
    "resize_interpolation": "cv2.INTER_AREA",
    "crop": "none_full_frame_retained",
    "intermediate_downscale": "none_direct_from_320x240",
    "aspect_ratio_change": "4:3_to_1:1_disclosed",
    "normalization": "ImageNet_RGB_once_float32",
    "train_temporal_sampling": sampling.VERSION,
    "validation_temporal_sampling": "native_uniform50_whole_clip",
}
INPUT_CONTRACT_224_SHA = hashlib.sha256(
    json.dumps(INPUT_CONTRACT_224, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def decode_224(scope, video_id: str, index_fn):
    """Decode authorized 320x240 video and resize selected 50 frames directly to 224x224 (INTER_AREA)."""
    row, path = scope.authorize(video_id)
    fd = gate1a.open_guarded(path)
    cap = None
    try:
        before = os.fstat(fd)
        cap = cv2.VideoCapture(f"/proc/self/fd/{fd}", cv2.CAP_FFMPEG)
        require(cap.isOpened(), "Cannot decode authorized source")
        raw_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        require(
            math.isfinite(raw_count) and raw_count >= FRAMES and abs(raw_count - round(raw_count)) < 1e-6,
            "Invalid/short source frame count",
        )
        require(math.isfinite(fps) and fps > 0, "Invalid source FPS")
        n = int(round(raw_count))
        indices = index_fn(n)
        require(len(indices) == FRAMES and len(set(indices)) == FRAMES, "Invalid 50-frame indices")
        wanted = set(indices)
        retained = []
        for index in range(n):
            ok, bgr = cap.read()
            require(ok, f"Truncated source at frame {index}")
            require(
                isinstance(bgr, np.ndarray) and bgr.dtype == np.uint8 and bgr.shape == (240, 320, 3),
                "Expected original 320x240 BGR frame",
            )
            if index in wanted:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                retained.append(cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
        require(not cap.read()[0], "Metadata frame count is smaller than decoded count")
        require(len(retained) == FRAMES, "Incomplete 50-frame selection")
        require(_identity(os.fstat(fd)) == _identity(before), "Source changed while decoding")
        metadata = {
            "frame_count": n,
            "fps": fps,
            "selected_indices": indices,
            "selected_times_seconds": [i / fps for i in indices],
            "decoder": "OpenCV_CAP_FFMPEG_sequential_authorized_fd",
            "resize_interpolation": "cv2.INTER_AREA",
            "resized_hw": [HEIGHT, WIDTH],
            "video_open_count": 1,
            "decoded_frames": n,
            "eof_check_reads": 1,
            "source_unchanged_during_decode": True,
            "source_size_bytes": before.st_size,
        }
        return np.stack(retained), metadata, row
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)


class Spatial224Dataset(RevisedDataset):
    def __init__(self, source_root, split, ids, reference):
        super().__init__(source_root, split, ids, reference)
        self.epoch = None

    def set_epoch(self, epoch: int) -> None:
        need(type(epoch) is int and 0 <= epoch < 30, "Invalid epoch")
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        vid = self._ids[index]
        ref = self.revision.record(vid, self.split)
        source_sha = labels.f.baseline.guarded_source_sha256(self._base.scope, vid)
        need(source_sha == ref["source_sha256"], "Source SHA changed")

        if self.split == "validation":
            rgb, m, row = decode_224(self._base.scope, vid, uniform_indices)
            need(
                row["original_class"] == ref["original_class"] and row["source_group"] == ref["source_group"],
                "Validation source provenance mismatch",
            )
            tensor, spatial = augment.transform_224(rgb, "validation", SEED, 0, vid, enabled=False)
            need(spatial is None, "Validation spatial augmentation forbidden")
            m.update(
                data_version=labels.VERSION,
                label_revision_sha256=labels.MANIFEST_SHA,
                label_patch_sha256=labels.PATCH_SHA,
                source_sha256=source_sha,
                tensor_sha256=tensor_sha(tensor),
                pixel_contract_parent_sha256=labels.f.CONTRACT_SHA256,
                input_contract_224_sha256=INPUT_CONTRACT_224_SHA,
                fixed_uniform_120x160_reference_tensor_sha256=ref["tensor_sha256"],
                fixed_uniform_120x160_tensor_match_claimed=False,
                times_are_nominal_index_over_fps_not_verified_pts=True,
            )
            return dict(
                tensor=tensor,
                video_id=vid,
                split="validation",
                source_group=ref["source_group"],
                class_id=ref["class_id"],
                binary_label=ref["binary_label"],
                original_class=ref["original_class"],
                original_class_id=ref["original_class_id"],
                effective_class=ref["effective_class"],
                metadata=m,
            )

        need(self.split == "train" and self.epoch is not None, "Explicit train epoch required")
        rgb, m, row = decode_224(self._base.scope, vid, lambda n: sampling.indices(n, SEED, self.epoch, vid))
        need(
            row["original_class"] == ref["original_class"] and row["source_group"] == ref["source_group"],
            "Train source provenance mismatch",
        )
        tensor, spatial = augment.transform_224(rgb, "train", SEED, self.epoch, vid, enabled=True)
        m.update(
            data_version=labels.VERSION,
            label_revision_sha256=labels.MANIFEST_SHA,
            label_patch_sha256=labels.PATCH_SHA,
            source_sha256=source_sha,
            tensor_sha256=tensor_sha(tensor),
            temporal_sampling_version=sampling.VERSION,
            temporal_seed=SEED,
            epoch_index=self.epoch,
            pixel_contract_parent_sha256=labels.f.CONTRACT_SHA256,
            input_contract_224_sha256=INPUT_CONTRACT_224_SHA,
            fixed_uniform_120x160_reference_tensor_sha256=ref["tensor_sha256"],
            fixed_uniform_120x160_tensor_match_claimed=False,
            times_are_nominal_index_over_fps_not_verified_pts=True,
            spatial_augmentation=spatial,
        )
        return dict(
            tensor=tensor,
            video_id=vid,
            split="train",
            source_group=ref["source_group"],
            class_id=ref["class_id"],
            binary_label=ref["binary_label"],
            original_class=ref["original_class"],
            original_class_id=ref["original_class_id"],
            effective_class=ref["effective_class"],
            metadata=m,
        )


class Spatial224Collator:
    def __init__(self):
        self.revision = labels.LabelRevision()

    def __call__(self, items: list[dict]) -> dict:
        need(bool(items) and len({x["split"] for x in items}) == 1, "Empty/mixed splits")
        split = items[0]["split"]
        need(split in ("train", "validation"), "Only train/validation")
        need(len({x["video_id"] for x in items}) == len(items), "Duplicate IDs")

        for item in items:
            ref = self.revision.record(item["video_id"], split)
            m = item["metadata"]
            x = item["tensor"]
            need(
                all(
                    item[k] == ref[k]
                    for k in ("class_id", "binary_label", "original_class", "original_class_id", "effective_class", "source_group", "split")
                ),
                "Label/split provenance mismatch",
            )
            need(
                isinstance(x, torch.Tensor)
                and x.device.type == "cpu"
                and x.dtype == torch.float32
                and tuple(x.shape) == (3, FRAMES, HEIGHT, WIDTH)
                and x.is_contiguous()
                and torch.isfinite(x).all().item(),
                f"Invalid 224x224 tensor: {getattr(x, 'shape', None)}",
            )
            need(
                m["data_version"] == labels.VERSION
                and m["label_revision_sha256"] == labels.MANIFEST_SHA
                and m["label_patch_sha256"] == labels.PATCH_SHA
                and m["source_sha256"] == ref["source_sha256"]
                and m["source_size_bytes"] == ref["source_size_bytes"]
                and m["resize_interpolation"] == "cv2.INTER_AREA"
                and m["resized_hw"] == [HEIGHT, WIDTH]
                and m["input_contract_224_sha256"] == INPUT_CONTRACT_224_SHA
                and m["tensor_sha256"] == tensor_sha(x)
                and m["fixed_uniform_120x160_reference_tensor_sha256"] == ref["tensor_sha256"]
                and m["fixed_uniform_120x160_tensor_match_claimed"] is False,
                "Input 224x224 contract/provenance mismatch",
            )
            if split == "train":
                need(
                    m["temporal_sampling_version"] == sampling.VERSION
                    and m["temporal_seed"] == SEED
                    and tuple(m["selected_indices"]) == sampling.indices(m["frame_count"], SEED, m["epoch_index"], item["video_id"]),
                    "Train sampling identity mismatch",
                )
                expected_sp = dict(
                    version=augment.VERSION,
                    policy_sha256=augment.POLICY_SHA,
                    seed=SEED,
                    epoch_index=m["epoch_index"],
                    video_id=item["video_id"],
                    parameters=augment.parameters(SEED, m["epoch_index"], item["video_id"]),
                )
                need(m.get("spatial_augmentation") == expected_sp, "Stale/incorrect 224x224 spatial identity")
            else:
                need(
                    "spatial_augmentation" not in m
                    and "temporal_sampling_version" not in m
                    and "epoch_index" not in m
                    and list(m["selected_indices"]) == list(uniform_indices(m["frame_count"])),
                    "Validation augmentation forbidden or wrong uniform50 indices",
                )

        if split == "train":
            need(len({x["metadata"]["epoch_index"] for x in items}) == 1, "Mixed epochs")

        targets = torch.tensor([x["binary_label"] for x in items], dtype=torch.long)
        batch = dict(
            tensor=torch.stack([x["tensor"] for x in items]),
            targets=targets,
            binary_targets=targets.clone(),
            eight_class_targets=torch.tensor([x["class_id"] for x in items], dtype=torch.long),
            records=[
                {
                    k: deepcopy(x[k])
                    for k in ("video_id", "original_class", "original_class_id", "effective_class", "source_group", "split", "metadata")
                }
                for x in items
            ],
            task="basketball51_direct_binary",
            data_version=labels.VERSION,
            label_revision_sha256=labels.MANIFEST_SHA,
            pixel_contract_parent_sha256=labels.f.CONTRACT_SHA256,
            input_contract_224_sha256=INPUT_CONTRACT_224_SHA,
        )
        if split == "train":
            batch["temporal_sampling_version"] = sampling.VERSION
            batch["spatial_policy_sha256"] = augment.POLICY_SHA
        return batch


class Spatial224EpochStream(BaseStream):
    def __init__(self, datasets, plan, state_getter, *, workers: int = 2):
        need(
            plan["seed"] == SEED
            and plan["temporal_sampling"] == sampling.POLICY
            and plan["spatial_augmentation"] == augment.POLICY
            and plan["input_contract_224_sha256"] == INPUT_CONTRACT_224_SHA,
            "224x224 sampling/spatial policy mismatch",
        )
        labels.require_revision_identity(plan)
        super().__init__(datasets, plan, state_getter, collate=Spatial224Collator(), workers=workers)

    def __call__(self, split, index):
        state = self.state_getter()
        if (state["epoch"], split) != self.key:
            self.datasets[split].set_epoch(state["epoch"])
        batch = super().__call__(split, index)
        if split == "train":
            need(all(r["metadata"]["epoch_index"] == state["epoch"] for r in batch["records"]), "Stale worker epoch")
        return batch
