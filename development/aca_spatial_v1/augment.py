"""Clip-consistent spatial/color augmentation with stateless named SHA256 draws."""
import hashlib
import json
import numpy as np
import torch
from development.aca_net_v1.data import normalize_rgb, MEAN, STD, require

VERSION='clip_flip_color_light_v2'
# Preserve the exact original draws: only rescale color factors toward 1.0.
DRAW_VERSION='clip_flip_color_light_v1'
POLICY=dict(version=VERSION,train_only=True,draw_version=DRAW_VERSION,frames=50,height=120,width=160,
    horizontal_flip_probability=0.5,brightness=[0.95,1.05],contrast=[0.95,1.05],saturation=[0.95,1.05],hue=False,
    order=['flip','brightness','contrast','saturation','clamp_0_1','ImageNet_normalize_once'],
    contrast_center='single_clip_gray_mean_after_brightness_numpy_float64_reduce_cast_float32',gray_weights=[0.2989,0.5870,0.1140],
    draw='SHA256_JSON_[draw_version,seed,zero_based_epoch,video_id,field]; first_53_bits / 2**53',
    parameter_scope='same_parameters_for_all_50_frames',validation='exact_original_loader',
    color_implementation='explicit_fixed_order_float32_blend_not_torchvision_random_ColorJitter')
POLICY_SHA=hashlib.sha256(json.dumps(POLICY,sort_keys=True,separators=(',',':')).encode()).hexdigest()
IDENTITY=dict(flip=False,brightness=1.0,contrast=1.0,saturation=1.0)

def parameters(seed,epoch,video_id):
    require(type(seed) is int and type(epoch) is int and 0<=epoch<30 and isinstance(video_id,str) and bool(video_id),'Invalid augmentation identity')
    def draw(field):
        raw=json.dumps([DRAW_VERSION,seed,epoch,video_id,field],ensure_ascii=True,separators=(',',':')).encode()
        return (int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')>>11)/(2**53)
    return dict(flip=draw('flip')<0.5,**{k:0.95+0.1*draw(k) for k in ('brightness','contrast','saturation')})

def validate_params(p):
    require(set(p)==set(IDENTITY) and type(p['flip']) is bool,'Bad augmentation parameters')
    require(all(type(p[k]) in (float,int) and 0.95<=p[k]<=1.05 for k in ('brightness','contrast','saturation')),'Factors outside reviewed range')

def apply_rgb(rgb,p):
    require(isinstance(rgb,np.ndarray) and rgb.dtype==np.uint8 and rgb.shape==(50,120,160,3),'Expected resized uint8 RGB clip')
    validate_params(p)
    x=torch.from_numpy(np.ascontiguousarray(rgb)).permute(3,0,1,2).float()/255.0
    if p['flip']:x=x.flip(-1)
    if p['brightness']!=1:x=x*float(p['brightness'])
    gray_weights=x.new_tensor(POLICY['gray_weights'])[:,None,None,None]
    if p['contrast']!=1:
        # Fixed CPU reduction avoids Torch thread-count-dependent rounding in workers.
        center=x.new_tensor(float((x*gray_weights).sum(0).numpy().mean(dtype=np.float64)))
        x=x*float(p['contrast'])+center*(1-float(p['contrast']))
    if p['saturation']!=1:
        gray=(x*gray_weights).sum(0,keepdim=True)
        x=x*float(p['saturation'])+gray*(1-float(p['saturation']))
    return x.clamp(0,1).contiguous()

def normalize_float(x):
    require(x.dtype==torch.float32 and tuple(x.shape)==(3,50,120,160) and x.device.type=='cpu','Invalid RGB float clip')
    return ((x-x.new_tensor(MEAN)[:,None,None,None])/x.new_tensor(STD)[:,None,None,None]).contiguous()

def transform(rgb,split,seed,epoch,video_id,*,enabled=True):
    require(split in ('train','validation'),'Only train/validation')
    if split=='validation' or not enabled:
        return normalize_rgb(rgb),None
    p=parameters(seed,epoch,video_id)
    return normalize_float(apply_rgb(rgb,p)),dict(version=VERSION,policy_sha256=POLICY_SHA,
        seed=seed,epoch_index=epoch,video_id=video_id,parameters=p)
