"""Bounded acceptance: fixed32 train only; no new samples, no test, no scoring."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from unittest.mock import patch
import cv2
import numpy as np
import torch
from torch import nn
from development.aca_eight_class_data_v1.gpu_readiness import configure, equal
from development.aca_binary_v1 import core
from development.adjudicated_input_v1 import adapter as labels
from . import formal_train as formal, sampling
from .data import TemporalDataset, TemporalCollator, EpochStream, RevisedDataset, BinaryCollator
from development.aca_binary_bn001_v1.model import initialized


def report(plan, fields, name):
    path=formal.QA/name
    formal.need(not path.exists(),'Acceptance exists; inspect before changes')
    value=dict(status='passed',at_taiwan=formal.runtime.now(),draft_plan_sha256=core.plan_digest(plan),
        code_sha256={k:v for k,v in plan['code_sha256'].items() if k.startswith(formal.PREFIX)},
        test_video_decodes=0,validation_video_decodes=0,formal_training_started=False,**fields)
    formal.runtime.atomic_json(path,value);print(json.dumps(value,ensure_ascii=False),flush=True)


def rejects(fn):
    try: fn()
    except (ValueError,KeyError): return
    raise AssertionError('Expected rejection')


def cpu():
    configure();plan=formal.draft_plan();e=formal.ORIGINAL.read(formal.QA/'evidence.json')
    assert formal.ORIGINAL.sha(formal.ROOT/'development/aca_temporal_v1/sampling.py')==e['sampling_source_sha256']
    vids=[r['video_id'] for r in e['records']];assert len(vids)==32 and len(set(vids))==32
    ds=TemporalDataset(plan['source_root'],'train',vids,plan['reference']);ds.set_epoch(0)
    collate=TemporalCollator();checks=[];samples=[]
    old=RevisedDataset(plan['source_root'],'train',vids,plan['reference'])
    for i,row in enumerate(e['records']):
        x=ds[i];batch=collate([x]);olditem=old[i]
        rgb=labels.f.baseline.preview_rgb(x['tensor'])
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['epoch1_rgb_sha256']
        assert tuple(x['metadata']['selected_indices'])==tuple(row['epochs_indices'][0])
        assert labels.tensor_sha(olditem['tensor'])==plan['reference'][vids[i]]['tensor_sha256']
        assert batch['targets'].item()==row['binary_label'] and batch['eight_class_targets'].item()==row['effective_class_id']
        assert np.array_equal(rgb[0],labels.f.baseline.preview_rgb(olditem['tensor'])[0])
        assert np.array_equal(rgb[-1],labels.f.baseline.preview_rgb(olditem['tensor'])[-1])
        samples.append(dict(video_id=vids[i],tensor_sha256=labels.tensor_sha(x['tensor']),source_sha256=x['metadata']['source_sha256']))
        print('checked fixed32',i+1,flush=True)
    checks.extend(['32_new_decodes_match_visually_reviewed_RGB_bytes','32_old_uniform_tensors_match_frozen_reference',
                   '50_RGB_shape_dtype_order_endpoints','32_effective_labels_unchanged_including_2p1_revision'])
    # Structural properties across all candidate epochs and many lengths, independently checking each bin.
    rng=core.capture_rng()
    for n in (50,51,97,145,146,179,180,1000):
        for ep in range(30):
            ix=sampling.indices(n,plan['seed'],ep,vids[0]);assert len(set(ix))==50 and tuple(sorted(ix))==ix
            assert ix[0]==0 and ix[-1]==n-1
            assert all(1+j*(n-2)//48<=ix[j+1]<1+(j+1)*(n-2)//48 for j in range(48))
    assert equal(rng,core.capture_rng())
    rejects(lambda:sampling.indices(49,plan['seed'],0,vids[0]))
    rejects(lambda:TemporalDataset(plan['source_root'],'test',[],plan['reference']))
    rejects(lambda:TemporalDataset(plan['source_root'],'train',[plan['validation_ids'][0]],plan['reference']))
    bad=deepcopy(ds[0]);bad['class_id']=7;rejects(lambda:collate([bad]))
    bad=deepcopy(ds[0]);bad['metadata']['selected_indices']=(0,)*50;rejects(lambda:collate([bad]))
    bad=deepcopy(ds[0]);bad['tensor'][0,0,0,0]+=1;rejects(lambda:collate([bad]))
    checks.append('invalid_split_label_index_tensor_rejected_and_global_RNG_unchanged')
    # Validation is an exact delegate, tested with sentinel before any decode; no new validation clip opened.
    val=TemporalDataset(plan['source_root'],'validation',plan['validation_ids'][:1],plan['reference'])
    sentinel=object()
    for ep in (0,29):
        val.set_epoch(ep)
        with patch.object(RevisedDataset,'__getitem__',return_value=sentinel) as method:
            assert val[0] is sentinel;method.assert_called_once_with(0)
        with patch.object(BinaryCollator,'__call__',return_value=sentinel) as method:
            assert collate([dict(split='validation')]) is sentinel;method.assert_called_once()
    checks.append('validation_exact_frozen_loader_and_collator_delegation_no_augmentation')
    # Spawn-worker and resume-cursor tests use only first two fixed train clips.
    tiny={**plan,'train_ids':vids[:2]};dataset=TemporalDataset(plan['source_root'],'train',vids[:2],plan['reference'])
    state=dict(epoch=0,cursor=0)
    stream=EpochStream({'train':dataset},tiny,lambda:state)
    for ep in (0,1,29):
        state.update(epoch=ep,cursor=0);dataset.set_epoch(ep)
        expected={i:labels.tensor_sha(dataset[i]['tensor']) for i in range(2)}
        order=torch.randperm(2,generator=torch.Generator().manual_seed(plan['seed']+ep)).tolist()
        before=core.capture_rng()
        for i in order:
            batch=stream('train',i);assert labels.tensor_sha(batch['tensor'][0])==expected[i]
            state['cursor']+=1
        assert equal(before,core.capture_rng())
        # A recreated loader after a persisted cursor must produce the exact same suffix.
        state['cursor']=1
        resumed=EpochStream({'train':dataset},tiny,lambda:state)
        assert labels.tensor_sha(resumed('train',order[1])['tensor'][0])==expected[order[1]]
        resumed.close()
    stream.close();checks.append('spawn_workers_epoch_changes_and_resume_suffix_bit_identical')
    report(plan,dict(checks=checks,fixed_train_video_ids=vids,unique_train_videos=32,samples=samples,
        engineering_optimizer_updates=0,model_initialization='unchanged_function_reused_no_model_code_change',
        validation_identity_scope='exact_delegate_proof_no_new_validation_decode',bounded_visual_epoch=1),'cpu_report.json')


def gpu():
    configure();plan=formal.draft_plan();formal.ORIGINAL.assert_idle_gpu(plan['gpu_policy'])
    e=formal.ORIGINAL.read(formal.QA/'evidence.json');ids=[e['records'][3]['video_id'],e['records'][5]['video_id']]
    ds=TemporalDataset(plan['source_root'],'train',ids,plan['reference']);ds.set_epoch(0)
    batches=[TemporalCollator()([ds[i]]) for i in range(2)]
    model,info=initialized();model.cuda();core.training_mode(model)
    frozen={n:b.detach().cpu().clone() for n,b in model.named_buffers() if '.aca.' not in n}
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,
                         weight_decay=0.,foreach=False,fused=False)
    before=model.fc.weight.detach().clone();opt.zero_grad()
    for b in batches:
        z=model(b['tensor'].cuda());assert z.shape==(1,2)
        (core.summed_ce(z,b['targets'].cuda())/2).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if '.aca.' in n)
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    opt.step();opt.zero_grad();assert not torch.equal(before,model.fc.weight)
    assert all(torch.equal(b,dict(model.named_buffers())[n].cpu()) for n,b in frozen.items())
    model.eval();buffers={n:b.clone() for n,b in model.named_buffers()}
    with torch.no_grad():
        for b in batches:core.summed_ce(model(b['tensor'].cuda()),b['targets'].cuda())
    assert all(torch.equal(b,dict(model.named_buffers())[n]) for n,b in buffers.items())
    peak=torch.cuda.max_memory_allocated();del model,opt;gc.collect();torch.cuda.empty_cache()
    report(plan,dict(train_video_ids=ids,unique_train_videos=2,engineering_optimizer_updates=1,
        engineering_weights_discarded=True,output_shape=[1,2],targets=[b['targets'].item() for b in batches],
        peak_allocated_bytes=peak,scope='engineering_only_no_classification_performance',
        backbone_buffers_unchanged=True,eval_buffers_unchanged=True),'gpu_report.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);a=p.parse_args()
    cpu() if a.mode=='cpu' else gpu()
