"""Isolated train temporal augmentation; validation delegates the frozen loader."""
import math
import os
from copy import deepcopy
import cv2
import numpy as np
import torch
from development.aca_net_v1.data import FRAMES, HEIGHT, WIDTH, require, gate1a, _identity, normalize_rgb
from development.adjudicated_input_v1 import adapter as labels
from development.aca_binary_v1.data import RevisedDataset, BinaryCollator
from development.aca_binary_v1.runtime import EpochStream as BaseStream
from development.aca_eight_class_data_v1.gpu_readiness import tensor_sha
from development.aca_temporal_v1 import sampling

SEED = 20261025
need = require

def decode_stratified50(scope, video_id, seed, epoch):
    row, path = scope.authorize(video_id)
    fd = gate1a.open_guarded(path)
    cap = None
    try:
        before = os.fstat(fd)
        cap = cv2.VideoCapture(f"/proc/self/fd/{fd}", cv2.CAP_FFMPEG)
        require(cap.isOpened(), "Cannot decode authorized source")
        raw_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        require(math.isfinite(raw_count) and raw_count >= FRAMES and
                abs(raw_count - round(raw_count)) < 1e-6, "Invalid/short source frame count")
        require(math.isfinite(fps) and fps > 0, "Invalid source FPS")
        n = int(round(raw_count))
        indices = sampling.indices(n, seed, epoch, video_id)
        wanted = set(indices)
        retained = []
        for index in range(n):
            ok, bgr = cap.read()
            require(ok, f"Truncated source at frame {index}")
            require(isinstance(bgr, np.ndarray) and bgr.dtype == np.uint8 and
                    bgr.shape == (240, 320, 3), "Expected original 320x240 BGR frame")
            if index in wanted:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                retained.append(cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA))
        require(not cap.read()[0], "Metadata frame count is smaller than decoded count")
        require(len(retained) == FRAMES, "Incomplete 50-frame selection")
        require(_identity(os.fstat(fd)) == _identity(before), "Source changed while decoding")
        metadata = {
            "frame_count": n, "fps": fps, "selected_indices": indices,
            "selected_times_seconds": [i / fps for i in indices],
            "decoder": "OpenCV_CAP_FFMPEG_sequential_authorized_fd",
            "video_open_count": 1, "decoded_frames": n,
            "eof_check_reads": 1, "source_unchanged_during_decode": True,
            "source_size_bytes": before.st_size,
        }
        return np.stack(retained), metadata, row
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)

class TemporalDataset(RevisedDataset):
    def __init__(self, source_root, split, ids, reference):
        super().__init__(source_root, split, ids, reference)
        self.epoch = None

    def set_epoch(self, epoch):
        need(type(epoch) is int and 0 <= epoch < 30, 'Invalid epoch')
        self.epoch = epoch

    def __getitem__(self, index):
        if self.split == 'validation':
            return super().__getitem__(index)  # Byte-identical old loader, including its reference SHA guard.
        need(self.split == 'train' and self.epoch is not None, 'Explicit train epoch required')
        vid = self._ids[index]; ref = self.revision.record(vid, 'train')
        source_sha = labels.f.baseline.guarded_source_sha256(self._base.scope, vid)
        need(source_sha == ref['source_sha256'], 'Source SHA changed')
        rgb, metadata, row = decode_stratified50(self._base.scope, vid, SEED, self.epoch)
        need(row['original_class'] == ref['original_class'] and row['source_group'] == ref['source_group'],
             'Source provenance mismatch')
        tensor = normalize_rgb(rgb)
        metadata.update(data_version=labels.VERSION, label_revision_sha256=labels.MANIFEST_SHA,
            label_patch_sha256=labels.PATCH_SHA, source_sha256=source_sha, tensor_sha256=tensor_sha(tensor),
            temporal_sampling_version=sampling.VERSION, temporal_seed=SEED, epoch_index=self.epoch,
            pixel_contract_parent_sha256=labels.f.CONTRACT_SHA256,
            fixed_uniform_reference_tensor_sha256=ref['tensor_sha256'],
            input_contract_sha256=None, fixed_uniform_tensor_match_claimed=False,
            times_are_nominal_index_over_fps_not_verified_pts=True)
        return dict(tensor=tensor, video_id=vid, split='train', source_group=ref['source_group'],
            class_id=ref['class_id'], binary_label=ref['binary_label'], original_class=ref['original_class'],
            original_class_id=ref['original_class_id'], effective_class=ref['effective_class'], metadata=metadata)


class TemporalCollator:
    def __init__(self):
        self.delegate = BinaryCollator()
        self.revision = self.delegate.delegate.revision

    def __call__(self, items):
        need(bool(items) and len({x['split'] for x in items}) == 1, 'Empty/mixed splits')
        if items[0]['split'] == 'validation':
            return self.delegate(items)
        need(all(x['split'] == 'train' for x in items), 'Train only')
        need(len({x['video_id'] for x in items}) == len(items), 'Duplicate IDs')
        for item in items:
            ref=self.revision.record(item['video_id'], 'train'); m=item['metadata']; x=item['tensor']
            need(all(item[k] == ref[k] for k in ('class_id','binary_label','original_class','original_class_id',
                  'effective_class','source_group','split')), 'Label/split provenance mismatch')
            need(isinstance(x,torch.Tensor) and x.device.type=='cpu' and x.dtype==torch.float32 and
                 tuple(x.shape)==(3,50,120,160) and x.is_contiguous() and torch.isfinite(x).all().item(), 'Tensor invalid')
            need(m['data_version']==labels.VERSION and m['label_revision_sha256']==labels.MANIFEST_SHA and
                 m['label_patch_sha256']==labels.PATCH_SHA and m['source_sha256']==ref['source_sha256'] and
                 m['source_size_bytes']==ref['source_size_bytes'], 'Input source/revision mismatch')
            need(m['temporal_sampling_version']==sampling.VERSION and m['temporal_seed']==SEED and
                 tuple(m['selected_indices'])==sampling.indices(m['frame_count'],SEED,m['epoch_index'],item['video_id']),
                 'Sampling identity mismatch')
            need(m['tensor_sha256']==tensor_sha(x) and m['fixed_uniform_reference_tensor_sha256']==ref['tensor_sha256']
                 and m['fixed_uniform_tensor_match_claimed'] is False and m['input_contract_sha256'] is None,
                 'Dynamic tensor identity mismatch')
        need(len({x['metadata']['epoch_index'] for x in items})==1, 'Mixed epochs')
        targets=torch.tensor([x['binary_label'] for x in items],dtype=torch.long)
        return dict(tensor=torch.stack([x['tensor'] for x in items]),targets=targets,
            binary_targets=targets.clone(),eight_class_targets=torch.tensor([x['class_id'] for x in items],dtype=torch.long),
            records=[{k:deepcopy(x[k]) for k in ('video_id','original_class','original_class_id','effective_class',
                'source_group','split','metadata')} for x in items],task='basketball51_direct_binary',
            data_version=labels.VERSION,label_revision_sha256=labels.MANIFEST_SHA,
            pixel_contract_parent_sha256=labels.f.CONTRACT_SHA256,temporal_sampling_version=sampling.VERSION)


class EpochStream(BaseStream):
    def __init__(self, datasets, plan, state_getter, *, workers=2):
        need(plan['seed']==SEED and plan['temporal_sampling']==sampling.POLICY, 'Sampling plan mismatch')
        labels.require_revision_identity(plan)
        super().__init__(datasets,plan,state_getter,collate=TemporalCollator(),workers=workers)

    def __call__(self, split, index):
        state=self.state_getter()
        if (state['epoch'],split) != self.key:
            self.datasets[split].set_epoch(state['epoch'])  # Before spawn; no persistent workers.
        batch=super().__call__(split,index)
        if split=='train':
            need(all(r['metadata']['epoch_index']==state['epoch'] for r in batch['records']), 'Stale worker epoch')
        return batch
