"""Bounded integration checks: two train clips; at most four CUDA updates."""
import argparse
import copy
import gc
import json
from pathlib import Path
import tempfile
import time
import traceback
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from development.aca_binary_v1 import core, runtime
from development.aca_binary_v1.data import RevisedDataset, BinaryCollator
from development.aca_temporal_v1.data import TemporalDataset, TemporalCollator, EpochStream
from development.aca_temporal_v1 import sampling
from development.aca_net_v1.data import normalize_rgb
from development.aca_eight_class_data_v1.gpu_readiness import configure, reset_seed, worker_init, equal, cpu_tree
from development.mixnet_binary50_v1.data_adapter import load_saved_rgb_fixture_item, verify_sample15_metadata
from development.mixnet_binary50_v1.model import snapshot_bn3d_buffers, compare_bn3d_snapshots
from . import formal_train as f
from .model import initialized, training_mode, validate_bn, need

FIXTURES=('04','22')
DIAGNOSTIC=f.CURRENT/'input_diagnostic32_20261119_v1'

def fixture_items(plan):
    ids=[json.loads((DIAGNOSTIC/x/'evidence.json').read_text())['video_id'] for x in FIXTURES]
    ds=TemporalDataset(plan['source_root'],'train',plan['train_ids'],plan['reference'])
    ds.set_epoch(0)
    indices=[plan['train_ids'].index(vid) for vid in ids]
    return ds, indices, ids

def check_cpu(plan):
    torch.set_num_threads(4);cv2.setNumThreads(1)
    f.install_runtime()
    model,init=initialized(); model.eval();model.train();validate_bn(model)
    bn_before=snapshot_bn3d_buffers(model)
    ds, indices, ids=fixture_items(plan)
    items=[ds[i] for i in indices]
    for key,item in zip(FIXTURES,items):
        ev=json.loads((DIAGNOSTIC/key/'evidence.json').read_text())
        rgb_path=DIAGNOSTIC/key/'source_frames_rgb.npy'
        need(f.sha(rgb_path)==ev['frames_array_sha256'],'Saved RGB fixture changed')
        rgb=np.load(rgb_path,mmap_mode='r')
        chosen=sampling.indices(len(rgb),plan['seed'],0,item['video_id'])
        expected=normalize_rgb(np.stack([cv2.resize(rgb[i],(160,120),interpolation=cv2.INTER_AREA) for i in chosen]))
        need(torch.equal(expected,item['tensor']),'Temporal decode differs from saved RGB reference')
    # Exercise the same two-worker spawn/collator path as formal training.
    before=core.capture_rng()
    loader=DataLoader(Subset(ds,indices),batch_size=1,num_workers=2,collate_fn=TemporalCollator(),
        worker_init_fn=worker_init,multiprocessing_context='spawn',prefetch_factor=2,
        generator=torch.Generator().manual_seed(20261024),persistent_workers=False)
    workers=list(loader)
    need(equal(before,core.capture_rng()),'Workers consumed model RNG')
    for a,b in zip(items,workers):
        need(torch.equal(a['tensor'],b['tensor'][0]),'Worker tensor differs')
        need(b['targets'].tolist()==[a['binary_label']],'Worker target differs')
    # Both loaders use the unchanged fixed validation delegation.
    need(issubclass(TemporalDataset,RevisedDataset),'Validation delegate changed')
    for key in FIXTURES:
        item,_=load_saved_rgb_fixture_item(key)
        # Fixed-loader fixture contract; no validation video is opened here.
        need(BinaryCollator()([item])['targets'].tolist()==[item['binary_label']],'Fixed collator changed')
    with torch.no_grad():
        for batch in workers:
            need(torch.isfinite(model(batch['tensor'])).all().item(),'Nonfinite frozen-BN forward')
    model.strict_clip_shape=False
    x=torch.randn(2,3,8,64,64,generator=torch.Generator().manual_seed(24))
    loss=core.summed_ce(model(x),torch.tensor([0,1]))/2
    loss.backward()
    bn_params={id(p) for m in model.modules() if isinstance(m,nn.BatchNorm3d) for p in m.parameters()}
    for p in model.parameters():
        need((p.grad is None) if id(p) in bn_params else (p.grad is not None and torch.isfinite(p.grad).all().item()),'Gradient coverage mismatch')
    need(compare_bn3d_snapshots(bn_before,snapshot_bn3d_buffers(model))['all_buffers_bitwise_unchanged'],'BN buffers changed in train')
    model.zero_grad(set_to_none=True)
    tiny_plan={**plan,'train_ids':ids,'validation_ids':ids}
    loop=f.TrainingLoop(model,tiny_plan,lambda s,i: workers[i],f.QA/'never_written.pt','cpu_contract_only')
    need(sum(p.numel() for g in loop.optimizer.param_groups for p in g['params'])==init['parameters']['trainable'],'Adam includes frozen BN')
    need(core.summed_ce(torch.tensor([[1.,2.]]),torch.tensor([1])).item()==torch.nn.functional.cross_entropy(torch.tensor([[1.,2.]]),torch.tensor([1])).item(),'CE definition mismatch')
    return dict(initialization=init,source_train_decodes=4,train_ids=ids,worker_tensor_equal=True,
        saved_RGB_temporal_tensor_equal=True,frozen_bn_in_train=True,frozen_affine_gradients_none=True,
        trainable_gradients_finite=True,optimizer_parameter_coverage=True,
        sample15=verify_sample15_metadata(),validation_video_decodes=0,test_video_decodes=0,
        optimizer_updates=0,fullsize_CPU_forward=True,small_backward_shape=[2,3,8,64,64])

def gpu_batches(plan):
    # CPU verified production decoding, limited to the same two train clips.
    ds,indices,ids=fixture_items(plan)
    return [TemporalCollator()([ds[i]]) for i in indices],ids

def check_gpu(plan):
    f.gpu_guard.assert_idle_gpu(plan['gpu_policy'])
    with f.shell.run_lock(f.CURRENT.parent/'00_admin','baseline8_gpu.lock'):
        f.gpu_guard.assert_idle_gpu(plan['gpu_policy'])
        configure();f.install_runtime()
        batches,ids=gpu_batches(plan)
        # Repeated engineering clips exercise effective16 and tail8; not a dataset accuracy measurement.
        probe=copy.deepcopy(plan)
        probe.update(train_ids=[ids[i%2] for i in range(24)],validation_ids=ids,
            max_epochs=1,checkpoint_every=1)
        def fetch(split,index):
            need(split=='train','Engineering validation data access forbidden')
            return batches[index%2]
        torch.cuda.reset_peak_memory_stats()
        with tempfile.TemporaryDirectory(prefix='mixnet_gpu_probe_',dir=f.QA) as temp:
            folder=Path(temp)
            def new_loop(path):
                reset_seed();m,_=initialized();m.cuda()
                return f.TrainingLoop(m,probe,fetch,path,'engineering_only')
            continuous=new_loop(folder/'continuous.pt')
            bn=snapshot_bn3d_buffers(continuous.model)
            need(continuous.run(lambda phase,state,seen:phase=='validation')=='paused','Probe failed to stop before validation')
            expected=torch.load(folder/'continuous.pt',map_location='cpu',weights_only=True)
            need(expected['state']['updates']==2 and expected['state']['cursor']==24,'Effective16/tail8 update count wrong')
            need(compare_bn3d_snapshots(bn,snapshot_bn3d_buffers(continuous.model))['all_buffers_bitwise_unchanged'],'Train BN changed')
            need(all(p.grad is None for p in continuous.model.parameters()),'Gradients retained after update')
            del continuous;gc.collect();torch.cuda.empty_cache()
            partial=new_loop(folder/'resume.pt')
            need(partial.run(lambda phase,state,seen:state['updates']==1)=='paused','Pause boundary failed')
            need(partial.committed['cursor']==16,'Pause not at effective16 boundary')
            del partial;gc.collect();torch.cuda.empty_cache()
            resumed=new_loop(folder/'resume.pt');resumed.resume()
            need(resumed.run(lambda phase,state,seen:phase=='validation')=='paused','Resume stop failed')
            actual=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True)
            matches={k:equal(expected[k],actual[k]) for k in ('model','optimizer','rng','cuda_rng','state','best_model')}
            need(all(matches.values()),'Continuous/resume differs '+str(matches))
            # Exercise eval and CPU validation metric accumulation on train fixtures only.
            resumed.model.eval();metric=core.BinaryMetrics()
            with torch.no_grad():
                for batch in batches:
                    logits=resumed.model(batch['tensor'].cuda());target=batch['targets'].cuda()
                    core.summed_ce(logits,target);metric.add(logits.cpu(),target.cpu())
            need(metric.result()['count']==2,'Metric accumulation failed')
            need(compare_bn3d_snapshots(bn,snapshot_bn3d_buffers(resumed.model))['all_buffers_bitwise_unchanged'],'Eval BN changed')
            peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            del resumed;gc.collect();torch.cuda.empty_cache()
        need(not folder.exists(),'Engineering checkpoint remains')
        return dict(real_optimizer_updates_engineering_only=4,effective=16,tail=8,
            microbatch=1,input_shape=[1,3,50,120,160],continuous_vs_resumed_equal=matches,
            bn_buffers_unchanged=True,temporary_checkpoints_removed=True,source_train_decodes=2,
            validation_video_decodes=0,test_video_decodes=0,engineering_eval_train_clips=2,
            accuracy_reported=False,peak_memory=peak,formal_weights_from_engineering=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cpu','gpu']);a=p.parse_args()
    f.QA.mkdir(exist_ok=True)
    output=f.QA/(a.phase+'_report.json');need(not output.exists(),'Do not overwrite readiness result')
    start=time.monotonic();report=dict(status='running',phase=a.phase)
    try:
        plan=f.draft_plan();report['draft_plan_sha256']=core.plan_digest(plan)
        if a.phase=='gpu':
            cpu=f.read(f.QA/'cpu_report.json')
            need(cpu['status']=='passed' and cpu['draft_plan_sha256']==report['draft_plan_sha256'],'CPU gate mismatch')
        report.update(check_cpu(plan) if a.phase=='cpu' else check_gpu(plan))
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    report['elapsed_seconds']=round(time.monotonic()-start,4)
    runtime.atomic_json(output,report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0 if report['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())
