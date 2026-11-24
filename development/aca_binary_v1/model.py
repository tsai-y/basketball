"""Exact ACA non-head initialization; isolated deterministic two-logit head."""
import torch
from torch import nn
from development.aca_reconstruction_v2.model import initialized as original_initialized, training_mode

def initialized(variant='aca', seed=20261024):
    if variant != 'aca' or seed != 20261024: raise ValueError('Only ACA seed24')
    model, evidence = original_initialized(variant, seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 3000)
        model.fc = nn.Linear(2048, 2)
    training_mode(model)
    evidence = {**evidence, 'task':'basketball51_direct_binary', 'class_order':['Miss','Make'],
                'head_initialization':'torch_Linear_default_seed_plus_3000_rng_isolated',
                'trained_parent_weights_loaded':False,
                'parameters': {'total':sum(p.numel() for p in model.parameters()),
                 'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}}
    return model, evidence
