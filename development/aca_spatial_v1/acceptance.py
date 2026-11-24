"""Bounded CPU acceptance on four saved RGB arrays; no video/GPU/model access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image,ImageDraw
from development.aca_temporal_v1 import data as td
from development.aca_net_v1.data import normalize_rgb
from development.aca_eight_class_data_v1.gpu_readiness import equal,worker_init
from development.mixnet_binary50_v1.data_adapter import verify_sample15_metadata
from . import augment as aug
from .data import SpatialDataset,SpatialCollator

BASE=Path('/home/tsai/projects/basketball/outputs/01_current')
SAVED=BASE/'input_diagnostic32_20261119_v1'
IDS=('04','16','22','27')
ROOT=Path(__file__).resolve().parents[2]
need=td.need

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(t):return hashlib.sha256(t.numpy().tobytes()).hexdigest()
def rng():return (random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
def rng_equal(a,b):
    return a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2])

def saved(key,epoch):
    p=SAVED/key;ev=json.loads((p/'evidence.json').read_text())
    need(sha(p/'source_frames_rgb.npy')==ev['frames_array_sha256'],'Saved source mismatch')
    frames=np.load(p/'source_frames_rgb.npy',mmap_mode='r')
    indices=td.sampling.indices(len(frames),td.SEED,epoch,ev['video_id'])
    rgb=np.stack([cv2.resize(frames[i],(160,120),interpolation=cv2.INTER_AREA) for i in indices])
    return rgb,ev,indices

class SavedInputDataset(SpatialDataset):
    """Exercise production adapter using saved pixels instead of opening video."""
    def __init__(self,plan,epoch):
        vids=[json.loads((SAVED/k/'evidence.json').read_text())['video_id'] for k in IDS]
        super().__init__(plan['source_root'],'train',vids,plan['reference'])
        self.set_epoch(epoch)
    def __getitem__(self,index):
        rgb,ev,indices=saved(IDS[index],self.epoch)
        ref=self.revision.record(ev['video_id'],'train')
        meta=dict(frame_count=ev['frame_count'],fps=ev['fps'],selected_indices=indices,
            source_size_bytes=ref['source_size_bytes'],video_open_count=0,decoded_frames=0,
            decoder='saved_RGB_fixture_only',source_unchanged_during_decode=True)
        with patch.object(td,'decode_stratified50',return_value=(rgb,meta,ref)), \
             patch.object(td.labels.f.baseline,'guarded_source_sha256',return_value=ref['source_sha256']), \
             patch.object(cv2,'VideoCapture',side_effect=AssertionError('Video decode forbidden')):
            return super().__getitem__(index)

def reference_float(rgb,p):
    # Independent NumPy arithmetic, same explicitly stated formula; float tolerance only here.
    x=rgb.astype(np.float64)/255
    if p['flip']:x=x[:,:,::-1,:]
    x=x*p['brightness']
    gray=x@np.array([.2989,.587,.114])
    x=x*p['contrast']+gray.mean()*(1-p['contrast'])
    gray=x@np.array([.2989,.587,.114])
    x=x*p['saturation']+gray[...,None]*(1-p['saturation'])
    return np.clip(x,0,1).transpose(3,0,1,2)

def image_row(x):
    return x.mul(255).round().clamp(0,255).byte().permute(1,2,3,0).numpy()

def run(out):
    torch.set_num_threads(4);cv2.setNumThreads(1)
    plan=json.loads((BASE/'aca_binary_temporal50_seed20261024_labelrev1_v1/plan.json').read_text())
    need(hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':')).encode()).hexdigest()==
         'ee331c184ca8b0cea54550098bd926929e9f1b74398528948c62410a70a2495a','Baseline plan changed')
    protected={str(ROOT/p):sha(ROOT/p) for p in plan['code_sha256']}
    need(all(protected[str(ROOT/p)]==h for p,h in plan['code_sha256'].items()),'Baseline code mismatch')
    report=dict(status='running',policy=aug.POLICY,policy_sha256=aug.POLICY_SHA,
        baseline_run='aca_binary_temporal50_seed20261024_labelrev1_v1',checks={},
        resource_scope=dict(saved_train_arrays=list(IDS),new_video_decodes=0,validation_video_access=0,
                            test_video_access=0,model_instances=0,optimizer_updates=0,GPU_executions=0))
    start=time.monotonic()
    g=np.random.default_rng(2718)
    frame=g.integers(0,256,(1,120,160,3),dtype=np.uint8)
    repeated=np.repeat(frame,50,axis=0)
    synthetic=g.integers(0,256,(50,120,160,3),dtype=np.uint8)
    p=dict(flip=True,brightness=1.05,contrast=.95,saturation=1.05)
    result=aug.apply_rgb(repeated,p)
    need(torch.equal(result,result[:,:1].expand_as(result)),'Repeated frames drifted')
    need(torch.equal(aug.apply_rgb(synthetic,{**aug.IDENTITY,'flip':True}),
                     torch.from_numpy(synthetic).permute(3,0,1,2).float().div(255).flip(-1)),'Flip axes wrong')
    ref=reference_float(synthetic,p);actual=aug.apply_rgb(synthetic,p).numpy()
    error=float(np.max(np.abs(ref-actual)));need(error<5e-7,'Formula mismatch')
    perm=g.permutation(50)
    permuted=aug.apply_rgb(synthetic[perm],p)
    # One clip-level reduction may differ by FP32 rounding when frame order changes.
    need(torch.allclose(permuted,aug.apply_rgb(synthetic,p)[:,perm],rtol=0,atol=2e-7),'Temporal order dependent transform')
    need(torch.equal(aug.normalize_float(aug.apply_rgb(synthetic,aug.IDENTITY)),normalize_rgb(synthetic)),'Identity path differs')
    with patch.object(aug,'parameters',side_effect=AssertionError('validation sampled RNG')):
        val,meta=aug.transform(synthetic,'validation',None,None,None)
    need(meta is None and torch.equal(val,normalize_rgb(synthetic)),'Validation transform differs')
    with patch.object(td.TemporalDataset,'__getitem__',return_value=val) as delegate:
        obj=SpatialDataset.__new__(SpatialDataset);obj.split='validation'
        need(obj[0] is val and delegate.call_count==1,'Validation failed direct delegation')
    report['checks']['clip_consistency']=dict(status='PASS',identical_frames_bitwise_equal=True,
        pure_flip_exact=True,independent_numpy_max_error=error,temporal_permutation_atol=2e-7,identity_exact=True)
    report['checks']['validation']=dict(status='PASS',original_normalize_tensor_SHA_equal=True,
        RNG_parameter_call_forbidden=True,production_dataset_direct_delegate=True,
        scope='synthetic pixel identity plus dispatch test; no actual validation sample opened')
    before=rng()
    identities=[(td.SEED,e,'probe') for e in (0,1,2)]
    params=[aug.parameters(*x) for x in identities]
    need(params==list(reversed([aug.parameters(*x) for x in reversed(identities)])),'Order dependence')
    need(len({json.dumps(x,sort_keys=True) for x in params})==3,'Epoch parameters unchanged')
    need(rng_equal(before,rng()),'Parameter generation changed global RNG')
    prior=json.loads((BASE/'aca_spatial_cpu_acceptance_20261123_v1/host_cpu_retry/cpu_report.json').read_text())
    for clip in prior['visuals']:
        old=clip['scenarios'][1]['parameters'];new=aug.parameters(td.SEED,0,clip['video_id'])
        need(new['flip']==old['flip'],'Flip decision changed while narrowing color')
        for field in ('brightness','contrast','saturation'):
            need(.95<=new[field]<=1.05 and abs(new[field]-(1+(old[field]-1)/2))<1e-15,'Color draw changed beyond rescaling')
    for factor in (.9,1.1):
        try:aug.validate_params({**aug.IDENTITY,'brightness':factor})
        except ValueError:pass
        else:raise AssertionError('Old out-of-range factors accepted')
    report['checks']['narrowing']=dict(status='PASS',same_four_flip_decisions=True,
        same_color_quantiles_rescaled=True,old_out_of_range_rejected=True)
    dataset=SavedInputDataset(plan,0)
    before=rng();serial=[SpatialCollator()([dataset[i]]) for i in range(4)]
    need(rng_equal(before,rng()),'Serial transform consumed global RNG')
    repeat=[SpatialCollator()([dataset[i]]) for i in reversed(range(4))]
    need(all(equal(a,b) for a,b in zip(serial,reversed(repeat))),'Repeated/reordered retrieval differs')
    before=rng()
    loader=DataLoader(dataset,batch_size=1,num_workers=2,multiprocessing_context='spawn',
        collate_fn=SpatialCollator(),worker_init_fn=worker_init,persistent_workers=False,
        generator=torch.Generator().manual_seed(2468),prefetch_factor=2,timeout=45)
    workers=list(loader)
    need(rng_equal(before,rng()),'Worker loader changed caller global RNG')
    need(all(equal(a,b) for a,b in zip(serial,workers)),'Worker/serial tensors or metadata differ')
    report['checks']['determinism']=dict(status='PASS',same_identity_bitwise=True,order_independent=True,
        epoch_parameters_differ=True,workers_0_vs_2_equal=True,Python_NumPy_Torch_global_RNG_unchanged=True)
    item=dataset[0];bad=copy.deepcopy(item);bad['metadata']['spatial_augmentation']['parameters']['flip']=not bad['metadata']['spatial_augmentation']['parameters']['flip']
    try:SpatialCollator()([bad])
    except ValueError:pass
    else:raise AssertionError('Tampered spatial metadata accepted')
    invalids=[('test',td.SEED,0,'v'),('train',td.SEED,-1,'v')]
    for split,seed,epoch,vid in invalids:
        try:aug.transform(synthetic,split,seed,epoch,vid)
        except ValueError:pass
        else:raise AssertionError('Invalid identity accepted')
    report['checks']['contracts']=dict(status='PASS',shape=[3,50,120,160],dtype='float32',
        stale_metadata_rejected=True,test_split_rejected=True,sample15=verify_sample15_metadata())
    visuals=[]
    positions=[0,9,19,29,39,49]
    for key,batch in zip(IDS,serial):
        rgb,ev,idx=saved(key,0);typical=aug.parameters(td.SEED,0,ev['video_id'])
        scenarios=[('original',aug.IDENTITY),('epoch0 sampled',typical),
            ('low boundary + flip',dict(flip=True,brightness=.95,contrast=.95,saturation=.95)),
            ('high boundary + flip',dict(flip=True,brightness=1.05,contrast=1.05,saturation=1.05))]
        canvas=Image.new('RGB',(960,4*148+30),'white');draw=ImageDraw.Draw(canvas)
        draw.text((5,5),key+' '+ev['video_id'],fill='black')
        stats=[]
        for row,(title,factors) in enumerate(scenarios):
            x=aug.apply_rgb(rgb,factors);pixels=image_row(x)
            y=30+row*148;draw.text((5,y),title+' '+str(factors),fill='black')
            for col,j in enumerate(positions):canvas.paste(Image.fromarray(pixels[j]),(col*160,y+22))
            stats.append(dict(name=title,parameters=factors,zero_fraction=float((x==0).float().mean()),one_fraction=float((x==1).float().mean())))
        dest=out/(key+'_comparison.png');canvas.save(dest)
        need(torch.equal(aug.normalize_float(aug.apply_rgb(rgb,typical)),batch['tensor'][0]),'Visual differs from loader')
        need(torch.isfinite(batch['tensor']).all().item(),'Nonfinite tensor')
        visuals.append(dict(fixture=key,video_id=ev['video_id'],image=str(dest),image_sha256=sha(dest),
            selected_input_positions=positions,source_frame_indices=[idx[j] for j in positions],
            augmented_tensor_sha256=tensor_sha(batch['tensor'][0]),scenarios=stats))
    report['visuals']=visuals
    need(all(sha(p)==h for p,h in protected.items()),'Protected source changed')
    report['protected_source_sha256']=protected
    report['new_source_sha256']={str(p):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    report.update(status='PASS_CPU_ONLY',elapsed_seconds=round(time.monotonic()-start,4),
        visual_review_status='PENDING_MANUAL_REVIEW',formal_training_started=False)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    try:report=run(args.output)
    except BaseException as exc:
        import traceback
        report=dict(status='FAIL',error=repr(exc),traceback=traceback.format_exc())
    (args.output/'cpu_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','elapsed_seconds','error','traceback')},indent=2))
    return 0 if report['status']=='PASS_CPU_ONLY' else 1

if __name__=='__main__':raise SystemExit(main())
