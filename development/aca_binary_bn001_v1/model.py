"""Single-variable TSCI running-stat momentum experiment; old modules untouched."""
from torch import nn
from development.aca_binary_v1.model import initialized as baseline_initialized
from development.aca_net_v1.modules import TSCI

MOMENTUM = 0.01


def tsci_bn_names(model):
    owned = {id(g.bn) for m in model.modules() if isinstance(m, TSCI)
             for g in (m.channel, m.height, m.width)}
    names = [name for name, m in model.named_modules() if id(m) in owned]
    if len(names) != 48:
        raise ValueError('Expected exactly 48 TSCI BN modules')
    return names


def apply_momentum(model):
    names = tsci_bn_names(model)
    for name in names:
        m = model.get_submodule(name)
        if not isinstance(m, nn.BatchNorm2d) or m.momentum != 0.1:
            raise ValueError('Unexpected original TSCI normalization')
        m.momentum = MOMENTUM
    validate_momentum(model)
    return names


def validate_momentum(model):
    names = tsci_bn_names(model)
    if any(model.get_submodule(n).momentum != MOMENTUM for n in names):
        raise ValueError('TSCI momentum differs from this experiment')
    return names


def initialized(variant='aca', seed=20261024):
    model, evidence = baseline_initialized(variant, seed)
    names = apply_momentum(model)
    evidence = {**evidence, 'head_shape': list(model.fc.weight.shape),
                'single_change': {'field': 'TSCI_BN_momentum', 'from': 0.1, 'to': MOMENTUM},
                'tsci_bn_names': names, 'tsci_bn_count': len(names),
                'trained_parent_weights_loaded': False}
    return model, evidence
