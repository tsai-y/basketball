"""Model-independent RGB clip transforms; no decoder, dataset or model dependency.

Input: uint8 [T,H,W,3], exactly configured T (64 in this pilot).
Output: contiguous float32 [3,T,Hout,Wout], [0,1], plus replayable metadata.
All random parameters are drawn once per clip. No temporal sampling occurs here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math

import cv2
import numpy as np

VERSION = 'video_clip_transforms_v1'
STRATEGIES = ('reference', 'photometric', 'spatial', 'horizontal_flip', 'gamma_enhancement')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def validate_config(cfg):
    required = {'version', 'clip_length', 'output_height', 'output_width', 'brightness',
                'contrast', 'saturation', 'spatial_scale', 'enhancement_gamma'}
    need(isinstance(cfg, dict) and set(cfg) == required, 'Unknown/missing transform config keys')
    need(cfg['version'] == VERSION, 'Unknown transform version')
    for key in ('clip_length', 'output_height', 'output_width'):
        need(type(cfg[key]) is int and 1 <= cfg[key] <= 1024, f'Invalid {key}')
    for key in ('brightness', 'contrast', 'saturation', 'spatial_scale'):
        pair = cfg[key]
        need(isinstance(pair, (list, tuple)) and len(pair) == 2, f'Invalid {key}')
        need(all(type(v) in (int, float) and math.isfinite(v) for v in pair), f'Invalid {key}')
        lo, hi = pair
        bounds = (0.9, 1.0) if key == 'spatial_scale' else (0.9, 1.1)
        need(bounds[0] <= lo <= hi <= bounds[1], f'Nonconservative {key}')
    gamma = cfg['enhancement_gamma']
    need(type(gamma) in (int, float) and math.isfinite(gamma) and 0.9 <= gamma <= 1.1,
         'Nonconservative gamma')
    return deepcopy(cfg)


def derived_seed(global_seed, video_id, epoch, clip_index, strategy):
    material = f'{global_seed}\0{video_id}\0{epoch}\0{clip_index}\0{VERSION}\0{strategy}'
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], 'big')


class ClipTransform:
    def __init__(self, config, strategy='reference'):
        self.config = validate_config(config)
        need(strategy in STRATEGIES, 'Unknown strategy')
        self.strategy = strategy

    def __call__(self, clip, *, seed, split='train', sample_metadata=None):
        need(split in ('train', 'validation', 'test'), 'Unknown split')
        need(split == 'train' or self.strategy == 'reference', 'Pilot candidates are train-only')
        need(type(seed) is int and 0 <= seed < 2**64, 'Seed must be uint64 integer')
        need(isinstance(clip, np.ndarray) and clip.dtype == np.uint8, 'Expected uint8 RGB ndarray')
        need(clip.ndim == 4 and clip.shape[0] == self.config['clip_length'] and
             clip.shape[-1] == 3 and min(clip.shape[1:3]) >= 2, 'Invalid/empty/short RGB clip')
        cfg = self.config
        _, height, width, _ = clip.shape
        oh, ow = cfg['output_height'], cfg['output_width']
        # Largest in-bounds rectangle matching output aspect ratio, up to pixel rounding.
        base_w = min(width, int(height * ow / oh))
        base_h = min(height, int(width * oh / ow))
        rng = np.random.Generator(np.random.PCG64(seed))
        scale = float(rng.uniform(*cfg['spatial_scale'])) if self.strategy == 'spatial' else 1.0
        cw, ch = max(1, int(base_w * scale)), max(1, int(base_h * scale))
        if self.strategy == 'spatial':
            x, y = int(rng.integers(width-cw+1)), int(rng.integers(height-ch+1))
        else:
            x, y = (width-cw)//2, (height-ch)//2
        params = dict(crop_xywh=[x, y, cw, ch], scale=scale,
                      horizontal_flip=self.strategy == 'horizontal_flip',
                      brightness=1.0, contrast=1.0, saturation=1.0, gamma=1.0)
        if self.strategy == 'photometric':
            for key in ('brightness', 'contrast', 'saturation'):
                params[key] = float(rng.uniform(*cfg[key]))
        if self.strategy == 'gamma_enhancement':
            params['gamma'] = cfg['enhancement_gamma']
        frames, clipped = [], 0
        for frame in clip:
            crop = frame[y:y+ch, x:x+cw]
            v = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
            if params['horizontal_flip']:
                v = v[:, ::-1, :]
            if self.strategy == 'photometric':
                # Fixed pivot avoids per-frame contrast adaptation and parameter flicker.
                v = (v * params['brightness'] - 0.5) * params['contrast'] + 0.5
                gray = np.sum(v * np.array([0.2126, 0.7152, 0.0722], np.float32), axis=-1, keepdims=True)
                v = gray + params['saturation'] * (v-gray)
            clipped += int(np.count_nonzero((v < 0) | (v > 1)))
            v = np.clip(v, 0, 1)
            if params['gamma'] != 1:
                v = np.power(v, np.float32(params['gamma']))
            frames.append(v)
        tensor = np.ascontiguousarray(np.stack(frames).transpose(3, 0, 1, 2), dtype=np.float32)
        metadata = dict(sample=deepcopy(sample_metadata or {}), transform=dict(
            version=VERSION, strategy=self.strategy, seed=seed, rng='numpy.PCG64',
            split=split, parameters=params, shared_across_all_frames=True,
            input_shape=list(clip.shape), output_shape=list(tensor.shape),
            input_dtype='uint8', output_dtype='float32', color='RGB', layout='CTHW',
            range=[0.0, 1.0], interpolation='opencv.INTER_AREA',
            normalization='none', newly_clipped_channel_fraction=clipped/tensor.size,
            retained_area_fraction=cw*ch/(width*height),
            aspect_ratio_relative_error=abs((cw/ch)/(ow/oh)-1),
            flip_semantics='changes_left_right_and_handedness; Make/Miss_invariance_unverified'
            if params['horizontal_flip'] else 'no_flip'))
        return tensor, metadata


def normalize_tensor(tensor, *, mode='identity', mean=None, std=None, scale=None, offset=None):
    """Explicit model adapter, separate from augmentation; never assumes an I3D checkpoint.

    mean_std uses RGB means/stds in [0,1] units; affine uses x*scale+offset.
    Always returns a separate float32 CTHW array and normalization metadata.
    """
    need(isinstance(tensor, np.ndarray) and tensor.dtype == np.float32 and tensor.ndim == 4
         and tensor.shape[0] == 3 and tensor.size > 0 and np.isfinite(tensor).all()
         and tensor.min() >= 0 and tensor.max() <= 1, 'Expected finite float32 unit-range CTHW')
    if mode == 'identity':
        need(all(v is None for v in (mean, std, scale, offset)), 'Unexpected identity parameters')
        output = tensor.copy()
    elif mode == 'mean_std':
        need(scale is None and offset is None, 'Unexpected affine parameters')
        m, s = np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)
        need(m.shape == s.shape == (3,) and np.isfinite(m).all() and np.isfinite(s).all()
             and (s > 0).all(), 'Explicit finite RGB mean/std required')
        output = (tensor-m[:, None, None, None])/s[:, None, None, None]
    elif mode == 'affine':
        need(mean is None and std is None and type(scale) in (int, float) and
             type(offset) in (int, float) and math.isfinite(scale) and math.isfinite(offset),
             'Explicit finite scale/offset required')
        output = tensor*np.float32(scale)+np.float32(offset)
    else:
        raise ValueError('Unknown normalization mode')
    need(np.isfinite(output).all(), 'Normalization overflow')
    return np.ascontiguousarray(output, dtype=np.float32), dict(
        mode=mode, mean=None if mean is None else list(mean), std=None if std is None else list(std),
        scale=scale, offset=offset, output_range=[float(output.min()), float(output.max())])
