"""Pinned offline initialization and explicitly frozen pretrained BN3d."""
import hashlib
from pathlib import Path
import torch
from torch import nn
import timm
from safetensors.torch import load_file
from development.mixnet_i3d.backbone import MixNetI3DBackbone
from development.mixnet_binary50_v1.model import MixNetBinary50Model

WEIGHTS = Path('/home/tsai/.cache/huggingface/hub/models--timm--mixnet_s.ft_in1k/snapshots/94149f6f14da590e98eeaa4dd9867b97a23bd3f4/model.safetensors')
WEIGHT_SHA = '5e1b5f59be7d42376359bf6cd3a4782eb53a14da018a52cc5fa53f568ed38314'
BN_POLICY = 'all_49_pretrained_BN3d_eval_running_stats_and_affine_frozen; other_convs_and_head_trainable'

def need(ok, message):
    if not ok:
        raise ValueError(message)

class FrozenBNMixNet(MixNetBinary50Model):
    def train(self, mode=True):
        super().train(mode)
        layers = [m for m in self.modules() if isinstance(m, nn.BatchNorm3d)]
        need(len(layers) == 49, 'BN count changed')
        for m in layers:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        return self

def validate_bn(model):
    layers = [m for m in model.modules() if isinstance(m, nn.BatchNorm3d)]
    need(len(layers) == 49 and all(not m.training and all(not p.requires_grad for p in m.parameters())
                                for m in layers), 'Frozen BN policy violated')
    bn_ids = {id(p) for m in layers for p in m.parameters()}
    need(all(p.requires_grad == (id(p) not in bn_ids) for p in model.parameters()), 'Unexpected frozen parameter')

def training_mode(model):
    model.train()
    validate_bn(model)

def initialized(variant='mixnet_i3d', seed=20261024):
    need(variant == 'mixnet_i3d' and type(seed) is int and seed == 20261024, 'Only authorized single seed')
    need(hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == WEIGHT_SHA, 'Pretrained file changed')
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        old = timm.create_model('mixnet_s.ft_in1k', pretrained=False).cpu().eval()
        state = load_file(str(WEIGHTS), device='cpu')
        old.load_state_dict(state, strict=True)
        need(all(torch.equal(old.state_dict()[k], v) for k,v in state.items()), '2D weight load mismatch')
        backbone = MixNetI3DBackbone(old)
        torch.random.default_generator.manual_seed(seed + 3000)
        model = FrozenBNMixNet(backbone)
        training_mode(model)
    need(torch.equal(before, torch.get_rng_state()), 'Initialization changed caller CPU RNG')
    need(sum(p.numel() for p in model.parameters()) == 3047704, 'Model parameter count changed')
    return model, dict(task='basketball51_direct_binary', variant=variant, seed=seed,
        pretrained_path=str(WEIGHTS), pretrained_sha256=WEIGHT_SHA,
        initialization='offline_2D_ImageNet_MixNetS_inflated_to_3D_fresh_binary_head',
        head_seed=seed+3000, head_shape=[2,1536], normalization_policy=BN_POLICY,
        bn_count=49, backbone_num_features=1536,
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)),
        source_classifier_discarded=True, caller_cpu_rng_preserved=True,
        author_exact_ACA=False, LSTA=False, TSCI=False)
