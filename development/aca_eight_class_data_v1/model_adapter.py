"""Pinned CPU eight-class baseline initialization, with no training entry point."""
import hashlib
import io
import json
from pathlib import Path

import torch
from torch import nn

from development.aca_net_v1.model import VideoResNet50
from .foundation import ROOT, need

SPEC_PATH = ROOT / 'docs/research/specs/eight_class_baseline_train_v1.json'
SPEC_SHA256 = '1398ee82bccbc9171357f3090b1ab716c3d228d41db95869ab1f8eabc5571086'


def load_model_contract():
    raw = SPEC_PATH.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == SPEC_SHA256, 'Baseline spec changed')
    spec = json.loads(raw)
    for name, expected in spec['source_sha256'].items():
        need(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
             f'Pinned dependency changed: {name}')
    return spec


def verified_checkpoint_bytes(path, identity):
    """Verify the same bytes that will be deserialized, without downloads."""
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'Expected regular local weight file')
    need(path.stat().st_size == identity['pretrained_size_bytes'], 'Weight byte count mismatch')
    raw = path.read_bytes()
    need(len(raw) == identity['pretrained_size_bytes'] and
         hashlib.sha256(raw).hexdigest() == identity['pretrained_sha256'], 'Weight SHA mismatch')
    return raw


def baseline_training_mode(model):
    """Keep backbone/head trainable but freeze all BN affine and statistics."""
    need(isinstance(model, VideoResNet50) and model.variant == 'baseline' and
         model.fc.out_features == 8, 'Expected eight-class baseline')
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)


def initialized_baseline8(seed=20261024):
    """Fresh eight-class head plus complete pinned ImageNet backbone, on CPU.

    Seeding intentionally follows the finalized contract (global CPU torch RNG).
    No old task checkpoint, old two-class initializer, or model download is used.
    """
    spec = load_model_contract()
    need(type(seed) is int and seed == spec['initialization']['seed'], 'Seed differs from fixed contract')
    identity = spec['initialization']
    raw = verified_checkpoint_bytes(ROOT / identity['pretrained_path'], identity)
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    need(isinstance(state, dict) and len(state) == 267, 'Unexpected ImageNet tensor inventory')
    torch.manual_seed(seed)
    model = VideoResNet50(variant='baseline', num_classes=8)
    head_before = {name: value.detach().clone() for name, value in model.fc.state_dict().items()}
    transfer = model.load_imagenet_state(state)
    missing_counters = transfer['legacy_bn_counters_initialized_to_zero']
    need(len(missing_counters) == identity['legacy_missing_bn_counters']['expected'],
         'Unexpected legacy BN counter count')
    current = model.state_dict()
    copied = [name for name in state if not name.startswith('fc.')]
    need(all(torch.equal(current[name], state[name]) for name in copied), 'Backbone transfer mismatch')
    need(all(torch.count_nonzero(current[name]).item() == 0 for name in missing_counters),
         'Legacy BN counters not zero')
    need(all(torch.equal(model.fc.state_dict()[name], value) for name, value in head_before.items()),
         'Fresh eight-class head changed during transfer')
    baseline_training_mode(model)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = spec['model']['parameters_from_saved_inventory']
    need(total == expected['total'] and trainable == expected['trainable'] and
         total - trainable == expected['frozen_bn_affine'], 'Parameter inventory mismatch')
    need(all(p.device.type == 'cpu' for p in model.parameters()), 'Initializer must return CPU model')
    return model, {
        'schema': 'baseline8_initialization_v1', 'seed': seed, 'spec_sha256': SPEC_SHA256,
        'pretrained_sha256': identity['pretrained_sha256'],
        'source_tensor_count': len(state), 'backbone_tensors_equal_to_source': len(copied),
        'transfer': transfer, 'fresh_head_preserved': True, 'head_shape': list(model.fc.weight.shape),
        'parameters': {'total': total, 'trainable': trainable, 'frozen_bn_affine': total - trainable},
        'device': 'cpu', 'old_task_checkpoint_loaded': False, 'optimizer_created': False,
    }
