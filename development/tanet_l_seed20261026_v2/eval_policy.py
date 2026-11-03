"""Fixed inference/validation rule: local TAM normalization within one video.

Each L channel's BN statistics are computed across that video's temporal
positions, without updating any buffers. G/BN2d use saved statistics; dropout is off.
Training still uses the original training_mode and updates both TAM BN branches.
"""
from contextlib import contextmanager

from torch import nn

from .config import need

POLICY = 'TAM_L_per_video_stats_batch1_no_buffer_update'


def tam_bn_modules(model):
    return {name: module for name, module in model.named_modules()
            if '.tam.' in name and isinstance(module, nn.BatchNorm1d)}


def assert_evaluation_mode(model):
    modules = tam_bn_modules(model)
    need(modules and any('.L.' in name for name in modules), 'Missing TAM L normalization')
    for name, module in model.named_modules():
        local = name in modules and '.L.' in name
        need(module.training == local, 'Unexpected evaluation training flag: '+name)
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            need(module.track_running_stats == (not local), 'Unexpected evaluation statistics: '+name)


@contextmanager
def per_video_L_stats(model):
    """Restore flags even if forward raises; buffers are never mutated by BN."""
    need(all(not module.training for module in model.modules()), 'Begin with model.eval()')
    modules = tam_bn_modules(model)
    need(all('.G.' in name or '.L.' in name for name in modules), 'Unknown TAM branch')
    old = {name: (module.training, module.track_running_stats) for name, module in modules.items()}
    try:
        for name, module in modules.items():
            module.training = '.L.' in name
            module.track_running_stats = not module.training
        assert_evaluation_mode(model)
        yield
    finally:
        for name, module in modules.items():
            module.training, module.track_running_stats = old[name]
