"""CPU preparation and bounded GPU engineering benchmark; never starts training.

The 4 metadata-selected train videos are repeated four times per effective batch.
No validation scoring, test access, trained checkpoint loading or checkpoint save.
"""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import cv2
import torch
from torch import nn
from development.aca_candidate_v1 import experiment as exp
from development.aca_net_v1 import train as base
from development.aca_net_v1.baseline_preflight import check_gradients,check_state,frozen_snapshot,training_mode
from development.aca_net_v1.data import Basketball50Dataset,VideoScope,file_sha256,guarded_source_sha256,require,select_smoke_ids
from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.modules import ReconstructionConfig
from .data import RESOLUTIONS,decode_pair,normalize

FOLDER=Path(__file__).resolve().parent
ROOT=FOLDER.parents[1]
SCHEMA='aca_resolution_cost_probe_v1'


def read(path):return json.loads(Path(path).read_text())


def config():
    c=read(FOLDER/'config.json')
    require(c['schema']=='aca_dense_resolution_feasibility_v1' and c['model']=='aca_dense_candidate' and
            c['seed']==20261025 and c['frames']==50 and c['resolutions_height_width']==[list(r) for r in RESOLUTIONS] and
            c['micro_batch']==1 and c['effective_batch']==16 and c['engineering_updates_per_resolution']==2 and
            c['lr']==1e-4 and c['weight_decay']==0 and c['clip_norm']==1 and c['formal_training_enabled'] is False,
            'Pinned feasibility protocol changed')
    return c


def hashes():
    old=read(ROOT/'development/aca_review_v1/artifacts/review24_ckaqp3hd/sampling_summary.json')['protected_files_sha256']
    verify(old)
    current=dict(old)
    for folder in ('aca_net_v1','aca_candidate_v1','aca_extension_v1','aca_resolution_v1'):
        for p in (ROOT/'development'/folder).glob('*'):
            if p.is_file() and p.suffix in ('.py','.json'):current[str(p)]=file_sha256(p)
    return current


def verify(expected):
    for path,digest in expected.items():require(file_sha256(Path(path))==digest,f'Protected source changed: {path}')


def state_digest(model):
    import hashlib
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare(source_root):
    require(not torch.cuda.is_initialized(),'CPU preparation must not use CUDA')
    torch.set_num_threads(2);cv2.setNumThreads(1);c=config();protected=hashes();verify(protected)
    train=VideoScope(source_root=source_root,split='train');validation=VideoScope(source_root=source_root,split='validation')
    ids=select_smoke_ids(train)
    paired=[];metadata=[];source_hashes={}
    for video_id in ids:
        source_hashes[video_id]=guarded_source_sha256(train,video_id)
        pair,meta,row=decode_pair(train,video_id)
        # Validate the new 120x160 branch against the unmodified actual loader.
        legacy=Basketball50Dataset(source_root=source_root,split='train',video_ids=[video_id])[0]
        low=normalize(pair[RESOLUTIONS[0]]);high=normalize(pair[RESOLUTIONS[1]])
        require(torch.equal(low,legacy['tensor']) and list(legacy['metadata']['selected_indices'])==meta['selected_indices'],
                'Control input differs from frozen original dataset')
        require(torch.isfinite(high).all().item(),'Nonfinite high-resolution input')
        require(int(row['binary_label'])==legacy['binary_label'],'Label mismatch')
        paired.append({size:normalize(pair[size]) for size in RESOLUTIONS});metadata.append(meta)
        require(guarded_source_sha256(train,video_id)==source_hashes[video_id],'Original video changed')
    base.configure_runtime(c['seed'],cuda=False)
    model,initialization=exp.initialized_candidate(c['seed'])
    require(all(torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite initialization')
    initial_sha=state_digest(model);del model
    shapes={}
    for h,w in RESOLUTIONS:
        with torch.device('meta'):
            m=VideoResNet50(variant='aca',num_classes=2);out=m(torch.empty(1,3,50,h,w))
            require(sum(p.numel() for p in m.parameters())==exp.PARAMETERS and tuple(out.shape)==(1,2),'Shape/architecture mismatch')
        shapes[f'{w}x{h}']=list(out.shape);del m,out
    out=Path(tempfile.mkdtemp(prefix='cpu_prepare_',dir=artifact_parent()))
    # Cache only these 4 train inputs; no optimizer/model checkpoint in this file.
    torch.save({'schema':'aca_resolution_engineering_inputs_v1','train_ids':ids,
                'labels':[int(train._allowed[i]['binary_label']) for i in ids],
                'inputs':[{f'{w}x{h}':x for (h,w),x in pair.items()} for pair in paired]},out/'engineering_inputs.pt')
    plan={'schema':SCHEMA,'source_root':str(Path(source_root).resolve()),'config':c,'protected_sha256':protected,
          'input_hashes':train.hashes,'train_count':len(train.rows),'validation_count':len(validation.rows),
          'ordered_train_ids_sha256':base.canonical_hash([r['video_id'] for r in train.rows]),
          'ordered_validation_ids_sha256':base.canonical_hash([r['video_id'] for r in validation.rows]),
          'train_ids':ids,'source_video_sha256':source_hashes,'sample_metadata':metadata,
          'initialization':initialization,'initial_state_sha256':initial_sha,'parameters':exp.PARAMETERS,
          'architecture':asdict(ReconstructionConfig()),
          'engineering_inputs_sha256':file_sha256(out/'engineering_inputs.pt'),
          'control_parity':'all_four_normalized_tensors_and_indices_bitwise_equal_to_frozen_original_loader',
          'normalization':'same_ImageNet_V1','single_variable':'full_frame_spatial_resolution',
          'gpu_preflight_ready':True,'formal_training_enabled':False,'existing_runs_read_only':True}
    verify(protected);require(not torch.cuda.is_initialized(),'CPU unexpectedly initialized CUDA')
    base.atomic_json(out/'plan.json',plan)
    base.atomic_json(out/'report.json',{'status':'cpu_passed_gpu_pending','parameters':exp.PARAMETERS,'meta_shapes':shapes,
        'low_resolution_pixel_parity':True,'direct_original_resize':True,'unique_train_videos':4,
        'video_decodes':8,'validation_videos_decoded':0,'test_videos_opened':0,'optimizer_steps':0,
        'cuda_initialized':False,'formal_training_started':False,'formal_training_entry_implemented':False})
    print(f'CPU_PREPARATION={out}',flush=True);return out


def artifact_parent():
    p=FOLDER/'artifacts';p.mkdir(exist_ok=True);return p


def gpu_idle():
    raw=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    desktop={'gnome-remote-desktop-daemon','Discord','nautilus','ptyxis','gnome-shell','Xwayland','chrome','codex','Codex',
             'gnome-text-editor','snapd-desktop-integration'}
    for value in raw.stdout.splitlines():
        pid=int(value.strip())
        if pid==os.getpid():continue
        try:name=(Path('/proc')/str(pid)/'exe').resolve(strict=True).name
        except FileNotFoundError:continue
        require(name in desktop,f'GPU busy with pid={pid} exe={name}; no process stopped')


def run_gpu(preparation,accepted):
    require(accepted,'Candidate is not verified author ACA-Net; explicit acceptance required')
    preparation=Path(preparation).resolve()
    require(preparation.parent==artifact_parent().resolve() and preparation.name.startswith('cpu_prepare_'),'Wrong preparation path')
    require(read(preparation/'report.json')['status']=='cpu_passed_gpu_pending','Missing CPU preparation')
    plan=read(preparation/'plan.json');c=config();require(c==plan['config'],'Preparation config changed')
    verify(plan['protected_sha256']);require(file_sha256(preparation/'engineering_inputs.pt')==plan['engineering_inputs_sha256'],'Input cache changed')
    gpu_idle();require(torch.cuda.is_available(),'Host CUDA unavailable')
    free,total=torch.cuda.mem_get_info();require(free>=10*1024**3,'Less than 10GiB GPU free; do not stop other processes')
    require(shutil.disk_usage(FOLDER).free>=2*1024**3,'Insufficient free disk')
    out=Path(tempfile.mkdtemp(prefix='gpu_cost_',dir=artifact_parent()));print(f'GPU_OUTPUT={out}',flush=True)
    base.atomic_json(out/'plan.json',plan)
    cache=torch.load(preparation/'engineering_inputs.pt',map_location='cpu',weights_only=True)
    require(cache['schema']=='aca_resolution_engineering_inputs_v1' and cache['train_ids']==plan['train_ids'],'Wrong engineering cache')
    require(cache['labels']==[0,1,0,1] and len(cache['inputs'])==4,'Wrong metadata subset')
    train_scope=VideoScope(source_root=plan['source_root'],split='train')
    for video_id,expected in plan['source_video_sha256'].items():require(guarded_source_sha256(train_scope,video_id)==expected,'Source changed')
    report={'schema':SCHEMA,'status':'running','engineering_only':True,'formal_training_started':False,
            'checkpoint_saved':False,'may_start_training_from_this_report':False,'test_videos_opened':0,'results':[]}
    try:
        for h,w in RESOLUTIONS:
            gpu_idle()
            key=f'{w}x{h}';base.configure_runtime(c['seed']);model,_=exp.initialized_candidate(c['seed'])
            require(state_digest(model)==plan['initial_state_sha256'],'Low/high initial states differ')
            frozen=frozen_snapshot(model);head=model.fc.weight.detach().clone();model=model.cuda()
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=c['weight_decay'])
            training_mode(model);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            times=[];wall_times=[];branch_sums={}
            for step in range(2):
                torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
                for index in range(16):
                    i=index%4;x=cache['inputs'][i][key].unsqueeze(0).cuda();y=torch.tensor([cache['labels'][i]],device='cuda')
                    logits=model(x);loss=nn.functional.cross_entropy(logits,y,reduction='sum')
                    require(torch.isfinite(loss).item(),'Nonfinite engineering loss');(loss/16).backward()
                    del x,y,logits,loss
                torch.cuda.synchronize();forward_backward_seconds=time.perf_counter()-start
                check_gradients(model)
                for branch in ('gate_value','gate_control','gate_output','short_term','kernel_generator','tsci'):
                    branch_sums[branch]=sum(p.grad.abs().sum().item() for n,p in model.named_parameters() if f'.{branch}.' in n and p.grad is not None)
                require(all(v>0 for v in branch_sums.values()),'Missing ACA branch gradient')
                torch.cuda.synchronize();optimizer_started=time.perf_counter()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],c['clip_norm'],error_if_nonfinite=True)
                opt.step();torch.cuda.synchronize()
                times.append(forward_backward_seconds+time.perf_counter()-optimizer_started)
                wall_times.append(time.perf_counter()-start)
                check_state(model,frozen);require(not torch.equal(model.fc.weight.detach().cpu(),head),'Head did not update')
                require(all(torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite updated parameters')
                print(f'{key} engineering_step={step+1}/2 effective16_seconds={times[-1]:.3f}',flush=True)
            require(len(opt.state)==sum(p.requires_grad for p in model.parameters()),'Adam moments incomplete')
            require(all(s['step'].item()==2 for s in opt.state.values()),'Adam step mismatch')
            require(all(torch.isfinite(s['exp_avg']).all().item() and torch.isfinite(s['exp_avg_sq']).all().item()
                        for s in opt.state.values()),'Nonfinite Adam moments')
            peak=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
            opt.zero_grad(set_to_none=True);model.eval()
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.no_grad():
                for inputs in cache['inputs']:
                    logits=model(inputs[key].unsqueeze(0).cuda());require(tuple(logits.shape)==(1,2) and torch.isfinite(logits).all().item(),'Eval shape/nonfinite')
            torch.cuda.synchronize();eval_seconds=(time.perf_counter()-start)/4;check_state(model,frozen)
            row={'resolution_height_width':[h,w],'parameters':exp.PARAMETERS,'same_initial_state':True,
                 'effective_batch':16,'micro_batch':1,'engineering_updates':2,'effective_batch_seconds':times,
                 'batch_wall_seconds_including_gradient_audits':wall_times,
                 'effective_batch_timing_excludes_gradient_audit_overhead':True,
                 'peak_allocated_bytes_with_Adam':peak,'peak_reserved_bytes':reserved,'all_BN_unchanged':True,
                 'all_trainable_gradients_finite':True,'branch_gradient_l1':branch_sums,
                 'eval_seconds_per_video_on_four_train_clips':eval_seconds,
                 'train_compute_only_seconds_per_epoch_estimate':times[-1]*450,
                 'validation_compute_only_seconds_estimate':eval_seconds*1547,
                 'timing_caveat':'Four cached train clips repeated; estimates exclude decoding, workers, checkpoint I/O and variability; not a full epoch measurement.'}
            report['results'].append(row);base.atomic_json(out/'report.json',report)
            del model,opt,head,frozen,logits;gc.collect();torch.cuda.empty_cache()
        verify(plan['protected_sha256'])
        for video_id,expected in plan['source_video_sha256'].items():require(guarded_source_sha256(train_scope,video_id)==expected,'Source changed')
        report.update(status='feasibility_passed_not_training_readiness',all_existing_sources_unchanged=True,
                      formal_training_entry_implemented=False,architecture='unverified_dense_ACA_candidate_not_author_original')
        base.atomic_json(out/'report.json',report);print(json.dumps({'output':str(out),'status':report['status'],'results':report['results']},ensure_ascii=False),flush=True)
    except Exception as error:
        report.update(status='failed_no_automatic_retry',error=f'{type(error).__name__}: {error}')
        base.atomic_json(out/'report.json',report);raise
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu-prepare',action='store_true');mode.add_argument('--gpu-preflight',action='store_true')
    p.add_argument('--source-root',type=Path,default=Path('/home/tsai/projects/basketball'))
    p.add_argument('--preparation',type=Path);p.add_argument('--accept-candidate-architecture',action='store_true');a=p.parse_args()
    if a.cpu_prepare:prepare(a.source_root)
    else:require(a.preparation is not None,'Need --preparation');run_gpu(a.preparation,a.accept_candidate_architecture)


if __name__=='__main__':main()
