"""Source-separated 14/4 ball-head trainer, with a separate one-epoch smoke gate."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import shutil
import tempfile
import traceback

from .probe import read, save, sha, require, within
from .ball_dataset_v2 import check as check_dataset, SCHEMA
from .ball_only_probe_v1 import runtime_at, freeze_backbone_eval
from .train_expansion_v1 import training_args, new_training_epoch


def checked_loader_paths(actual, expected):
    actual=[str(Path(p).resolve()) for p in actual]
    expected={str(Path(p).resolve()) for p in expected}
    require(len(actual)==len(set(actual)) and set(actual)==expected, 'Loader escaped fixed split or duplicated an image')


def parameter_scope(model, head_index):
    prefix=f'model.{head_index}.'
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    require(params and all(n.startswith(prefix) and '.dfl.' not in n for n,p in params),
            'Only Detect parameters (excluding DFL) may train')
    return params


def gradient_summary(model, head_index):
    """Called before optimizer.zero_grad, with AMP disabled by the fixed policy."""
    import torch
    allowed={n for n,p in parameter_scope(model,head_index)}
    present=0;nonzero=0
    for n,p in model.named_parameters():
        if p.grad is None:continue
        require(n in allowed, 'Frozen parameter received a gradient')
        require(torch.isfinite(p.grad).all().item(), 'Non-finite Detect gradient')
        present+=1;nonzero+=int(torch.count_nonzero(p.grad).item()>0)
    require(present>0 and nonzero>0, 'No finite nonzero Detect gradient')
    return {'tensors_with_gradient':present,'tensors_with_nonzero_gradient':nonzero}


def smoke_gate(state, dataset, dataset_sha, initializer_sha, code_sha):
    require(state.get('schema')==SCHEMA and state.get('status')=='completed' and state.get('stage')=='smoke',
            'A completed independent v2 smoke is required')
    require(state['dataset']==str(Path(dataset).resolve()) and state['dataset_manifest_sha256']==dataset_sha
            and state['initializer_sha256']==initializer_sha and state['source_code_sha256']==code_sha,
            'Smoke used different data, initializer or trainer source')
    require(state['epochs_completed']==1 and state['optimizer_steps']==7 and state['training_image_exposures']==14,
            'Smoke did not complete the fixed14-image epoch')
    for flag in ('actual_loader_allowlists_checked','all_optimizer_gradients_finite_nonzero',
                 'backbone_parameters_and_BN_buffers_unchanged','frozen_head_parameters_unchanged',
                 'checkpoint_round_trip','head_updated','source_data_unchanged'):
        require(state.get(flag) is True, 'Smoke check missing: '+flag)


def train(dataset, *, epochs=1, smoke=True, approved_smoke=None):
    dataset=Path(dataset).resolve();m=check_dataset(dataset)
    training_args(dataset,dataset/'unused',epochs,smoke)
    manifest_sha=sha(dataset/'dataset_manifest.json')
    preflight=read(within(dataset,'preflight.json'))
    require(preflight['status']=='completed' and not preflight['training_started']
            and preflight['initializer_sha256']==m['initializer_sha256'], 'Dataset CPU preflight required')
    require({s:r['images_loaded'] for s,r in preflight['splits'].items()}=={'train':14,'val':4}, 'Wrong preflight counts')
    if not smoke:
        require(approved_smoke is not None, 'Full training requires --approved-smoke')
        approved_smoke=Path(approved_smoke).resolve()
        require(approved_smoke.is_relative_to(dataset), 'Foreign smoke run')
        smoke_gate(read(within(approved_smoke,'run_state.json')),dataset,manifest_sha,m['initializer_sha256'],sha(__file__))
    import torch
    require(torch.cuda.is_available(), 'CUDA required; no silent CPU training')
    require(torch.cuda.device_count()==1, 'This isolated trainer supports one visible GPU only')
    run=Path(tempfile.mkdtemp(prefix='smoke_v2_' if smoke else 'headtrain_v2_',dir=dataset))
    runtime_at(run)
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    import ultralytics
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    wrapper=YOLO(m['initializer']);head_index=len(wrapper.model.model)-1
    require(type(wrapper.model.model[9].cv1.act).__name__=='SiLU', 'Initializer structure changed')
    args=training_args(dataset,run,epochs,smoke);args['freeze']=head_index
    dependencies={}
    for filename in ('ball_train_v2.py','ball_dataset_v2.py','ball_only_probe_v1.py','train_expansion_v1.py'):
        path=Path(__file__).with_name(filename);shutil.copy2(path,run/filename)
        dependencies[str(path)]=sha(path)
    save(run/'dataset_manifest_snapshot.json',m)
    allowed={s:{str((dataset/r['image']).resolve()) for r in m['records'] if r['partition']==s} for s in ('train','val')}
    log={'schema':SCHEMA,'status':'starting','stage':'smoke' if smoke else 'ball_only_head_training_v2',
         'dataset':str(dataset),'dataset_manifest_sha256':manifest_sha,'initializer_sha256':m['initializer_sha256'],
         'source_code_sha256':sha(__file__),'dependency_source_sha256':dependencies,'training_args':args,
         'head_index':head_index,'image_counts':m['image_counts'],'source_group_counts':m['source_group_counts'],
         'epochs_completed':0,'optimizer_steps':0,'training_image_exposures':0,'gradient_checks':[],
         'torch':torch.__version__,'ultralytics':ultralytics.__version__,'GPU':torch.cuda.get_device_name(0),
         'approved_smoke':str(approved_smoke) if approved_smoke else None,
         'full_run_restarts_original_initializer_not_smoke_weights':True,
         'generalization_metrics_valid':False,'original_classification_validation_test_used':False,
         'negative_crops_used':False,'MixNet_features_changed':False,
         'started_at_utc':datetime.now(timezone.utc).isoformat()}
    frozen={};frozen_head={};initial_head={};observed=set();seen=Counter();prefix=f'model.{head_index}.'

    class PreservedV2Trainer(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            require(weights is not None, 'Must receive actual initializer modules, not YAML reconstruction')
            result=deepcopy(weights).float()
            require(result.names=={0:'ball'} and result.model[-1].nc==1
                    and type(result.model[9].cv1.act).__name__=='SiLU', 'Wrong checkpoint structure/semantics')
            return result

        def _model_train(self):
            super()._model_train();freeze_backbone_eval(self.model,head_index)

        def preprocess_batch(self,batch):
            paths=[str(Path(p).resolve()) for p in batch['im_file']]
            require(set(paths)<=allowed['train'], 'Development/nontrain image entered a training batch')
            seen.update(paths);log['training_image_exposures']+=len(paths)
            return super().preprocess_batch(batch)

        def optimizer_step(self):
            require(not self.amp, 'Fixed FP32 gradient audit requires AMP disabled')
            summary=gradient_summary(self.model,head_index)
            super().optimizer_step()
            log['optimizer_steps']+=1
            log['gradient_checks'].append({'step':log['optimizer_steps'],**summary})

    def start(trainer):
        require(trainer.data['names']=={0:'ball'}, 'Wrong class mapping')
        require(trainer.batch_size==2 and trainer.accumulate==1, 'Fixed update budget changed')
        for split,loader in [('train',trainer.train_loader),('val',trainer.test_loader)]:
            checked_loader_paths(loader.dataset.im_files,allowed[split])
        params=parameter_scope(trainer.model,head_index)
        freeze_backbone_eval(trainer.model,head_index)
        frozen.update({n:v.detach().cpu().clone() for n,v in trainer.model.state_dict().items() if not n.startswith(prefix)})
        frozen_head.update({n:p.detach().cpu().clone() for n,p in trainer.model.named_parameters() if n.startswith(prefix) and not p.requires_grad})
        initial_head.update({n:p.detach().cpu().clone() for n,p in params})
        log.update(status='training',actual_loader_allowlists_checked=True,
                   trainable_parameters=sum(p.numel() for n,p in params),backbone_bn_eval_enforced=True,
                   Detect_BN_statistics_trainable=True,SPPF_cv1_activation='SiLU')
        save(run/'run_state.json',log)

    def epoch_start(trainer):
        seen.clear();freeze_backbone_eval(trainer.model,head_index)

    def batch_end(trainer):
        require(torch.isfinite(trainer.loss_items).all().item(), 'Non-finite loss')
        require(all(not layer.training for layer in trainer.model.model[:head_index]), 'Backbone entered train mode')

    def epoch_end(trainer):
        require(seen==Counter({p:1 for p in allowed['train']}), 'Training epoch must expose all14 images once')
        require(trainer.batch_size==2 and trainer.accumulate==1, 'Batch/update policy changed')
        observed.add(trainer.epoch)

    def fit_end(trainer):
        if not new_training_epoch(trainer.epoch,log['epochs_completed'],observed):return
        require(log['optimizer_steps']==7*(trainer.epoch+1), 'Expected7 optimizer steps per epoch')
        log.update(epochs_completed=trainer.epoch+1,
                   training_losses=[float(v) for v in trainer.tloss.detach().cpu()],
                   development_metrics={k:float(v) for k,v in trainer.metrics.items()},
                   peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
        save(run/'run_state.json',log);print('BALL_V2_EPOCH:',log['epochs_completed'],log['training_losses'],flush=True)

    for event,callback in [('on_train_start',start),('on_train_epoch_start',epoch_start),
                           ('on_train_batch_end',batch_end),('on_train_epoch_end',epoch_end),('on_fit_epoch_end',fit_end)]:
        wrapper.add_callback(event,callback)
    save(run/'run_state.json',log);print('RUN_DIR:',run,flush=True)
    try:
        wrapper.train(trainer=PreservedV2Trainer,**args)
        actual=wrapper.trainer.model.state_dict()
        require(frozen and all(torch.equal(v,actual[n].detach().cpu()) for n,v in frozen.items()), 'Frozen feature state changed')
        require(frozen_head and all(torch.equal(v,actual[n].detach().cpu()) for n,v in frozen_head.items()), 'Frozen DFL changed')
        updates={n:float((actual[n].detach().cpu()-v).abs().max()) for n,v in initial_head.items()}
        require(any(v>0 for v in updates.values()), 'Detect parameters did not update')
        checkpoints={}
        for name in ('best','last'):
            path=within(run,f'fit/weights/{name}.pt');loaded=YOLO(str(path))
            require(loaded.names=={0:'ball'} and type(loaded.model.model[9].cv1.act).__name__=='SiLU', 'Reload changed checkpoint semantics/structure')
            results=loaded.predict(source=sorted(allowed['val']),device=0,imgsz=640,conf=.25,iou=.7,
                                   max_det=100,half=False,save=False,verbose=False)
            require(len(results)==m['image_counts']['val'] and
                    all(torch.isfinite(r.boxes.data).all().item() and tuple(r.orig_shape)==(240,320) for r in results),
                    'Checkpoint reload on4 development images failed')
            checkpoints[name]={'path':str(path),'sha256':sha(path),'development_images_inferred':len(results)}
        check_dataset(dataset)
        require(sha(dataset/'dataset_manifest.json')==manifest_sha, 'Dataset manifest changed during training')
        for p,h in dependencies.items():require(sha(p)==h, 'Trainer source changed during run')
        log.update(status='completed',head_parameter_max_abs_updates=updates,head_updated=True,
                   checkpoint_round_trip=True,checkpoints=checkpoints,all_optimizer_gradients_finite_nonzero=True,
                   backbone_parameters_and_BN_buffers_unchanged=True,frozen_head_parameters_unchanged=True,
                   source_data_unchanged=True,stop_reason='early_stopping' if log['epochs_completed']<epochs else 'epoch_limit',
                   selected_development_metrics={k:float(v) for k,v in wrapper.trainer.metrics.items()},
                   completed_at_utc=datetime.now(timezone.utc).isoformat())
        if smoke:smoke_gate(log,dataset,manifest_sha,m['initializer_sha256'],sha(__file__))
        save(run/'run_state.json',log)
    except BaseException as exc:
        log.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        save(run/'run_state.json',log);raise
    finally:check_dataset(dataset)
    print('COMPLETED:',run,flush=True);return run


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke',action='store_true');mode.add_argument('--train',action='store_true')
    p.add_argument('--dataset-dir',required=True);p.add_argument('--epochs',type=int)
    p.add_argument('--approved-smoke');a=p.parse_args()
    train(a.dataset_dir,epochs=a.epochs if a.epochs is not None else (1 if a.smoke else 100),
          smoke=a.smoke,approved_smoke=a.approved_smoke)


if __name__=='__main__':main()
