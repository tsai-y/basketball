"""One explicitly assumed ACA reconstruction; no formal training entry point."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.modules import LSTA, TSCI, ReconstructionConfig
from development.aca_eight_class_data_v1.model_adapter import initialized_baseline8
from development.aca_eight_class_data_v1.foundation import ROOT

PROPOSAL = ROOT / 'docs/research/specs/aca_reconstruction_proposal_v2.json'
PROPOSAL_SHA = 'a4066366c987b4dda66a90debb6496983fa839ed8aad1e3f2bc60d19c0f302e0'
TOTAL = {'baseline':23524424, 'lsta':25471780, 'tsci':23529224, 'aca':25476580}


def contract():
    raw = PROPOSAL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROPOSAL_SHA:
        raise ValueError('Proposal changed; review a new version')
    return json.loads(raw)


class SeparableLSTA(LSTA):
    """Inherited forward; directly initialize dense1 and depthwise3 weights."""
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.config = ReconstructionConfig()
        def block(k, groups):
            return nn.Sequential(nn.Conv3d(channels, channels, k, padding=k//2,
                                           groups=groups, bias=False),
                                 nn.InstanceNorm3d(channels, eps=1e-5, affine=False,
                                                   track_running_stats=False), nn.ReLU())
        self.gate_value = block(1, 1)
        self.gate_control = block(3, channels)
        self.gate_output = block(3, channels)
        h = max(1, channels//16)
        self.short_term = nn.Sequential(nn.Conv1d(channels,h,3,padding=1),nn.ReLU(),
                                        nn.Conv1d(h,channels,3,padding=1),nn.Sigmoid())
        self.kernel_generator = nn.Sequential(nn.Linear(50,12),nn.ReLU(),nn.Linear(12,3))


class ParallelACA(nn.Module):
    def __init__(self, lsta, tsci):
        super().__init__()
        self.lsta, self.tsci = lsta, tsci

    def forward(self, x):
        return self.lsta(x) + self.tsci(x)


def training_mode(model):
    """Only TSCI-owned BN learns statistics/affine; all backbone BN is frozen."""
    model.train()
    new_bn = {id(g.bn) for m in model.modules() if isinstance(m,TSCI)
              for g in (m.channel,m.height,m.width)}
    for m in model.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):
            learn = id(m) in new_bn
            m.train(learn)
            for p in m.parameters(): p.requires_grad_(learn)


def _attach(model, variant, seed):
    if variant not in TOTAL: raise ValueError('Unknown arm')
    index = 0
    for name in ('layer1','layer2','layer3','layer4'):
        for block in getattr(model,name):
            c = block.conv1.out_channels
            lsta = tsci = None
            if variant in ('lsta','aca'):
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed+1000+index)
                    lsta = SeparableLSTA(c)
            if variant in ('tsci','aca'):
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(seed+2000+index)
                    tsci = TSCI()
            block.aca = (ParallelACA(lsta,tsci) if variant=='aca' else
                         lsta if variant=='lsta' else tsci if variant=='tsci' else nn.Identity())
            index += 1
    model.variant = variant
    training_mode(model)
    assert sum(p.numel() for p in model.parameters()) == TOTAL[variant]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == TOTAL[variant]-53120
    return model


def initialized(variant='aca', seed=20261024):
    contract()
    if type(seed) is not int or seed != 20261024 or variant not in TOTAL:
        raise ValueError('Only reviewed first-seed arms are available')
    # Keep the precise baseline construction/transfer/head RNG sequence.
    model, evidence = initialized_baseline8(seed)
    model = _attach(model,variant,seed)
    evidence = {**evidence,'arm':variant,'proposal_sha256':PROPOSAL_SHA,
                'parameters':{'total':TOTAL[variant],'trainable':TOTAL[variant]-53120}}
    return model,evidence


def meta_model(variant):
    contract()
    with torch.device('meta'):
        return _attach(VideoResNet50(variant='baseline',num_classes=8),variant,20261024)
