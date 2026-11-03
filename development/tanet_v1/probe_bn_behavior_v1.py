"""Bounded train-only BN forward experiment; no gradients, state updates or test.

Six frozen checkpoints, sixteen metadata-selected train clips, disabled Dropout.
Compare saved running statistics with TAM G/L batch statistics, and batch2
numerical/context controls. BN track_running_stats is temporarily disabled only
on an in-memory diagnostic model, so even its running buffers never update.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
import gc
import hashlib
import html
import json
from pathlib import Path
import shutil
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn

from .config import FOLDER, InputConfig, need, sha256
from .data import VideoScope, decode
from .model import TANetBinary
from .training import atomic_save, configure_runtime, state_digest, autocast
from .diagnose_stability_v1 import prepare_inputs, finite_summary, write_csv

SEED = 20261030
MODES = (('saved_b1', 'saved', 1), ('G_batch_b1', 'G', 1),
         ('L_batch_b1', 'L', 1), ('GL_batch_b1', 'GL', 1),
         ('saved_b2', 'saved', 2), ('GL_batch_b2', 'GL', 2))


def read(path):
    return json.loads(Path(path).read_text())


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def select_train_cases(rows):
    need(rows and all(r['split'] == 'train' and str(r['excluded']) == '0' for r in rows), 'Train metadata only')
    need(len({r['video_id'] for r in rows}) == len(rows), 'Duplicate train metadata')
    classes = sorted({r['original_class'] for r in rows})
    need(classes == ['2p0','2p1','3p0','3p1','ft0','ft1','mp0','mp1'], 'Expected8 original classes')
    rank = lambda value: hashlib.sha256(f'bn-behavior-v1:{SEED}:{value}'.encode()).hexdigest()
    selected, groups = [], Counter()
    for cls in classes:
        pool = [r for r in rows if r['original_class'] == cls]
        need(len(pool) >= 2, 'Insufficient class; no replacement')
        local = set()
        for _ in range(2):
            row = min(pool, key=lambda r: (r['source_group'] in local, groups[r['source_group']],
                                           rank(r['source_group']), rank(r['video_id'])))
            pool.remove(row); local.add(row['source_group']); groups[row['source_group']] += 1
            selected.append(dict(row))
    return selected


def prepare():
    need(not torch.cuda.is_initialized(), 'Preparation is metadata-only')
    summary, protected, _ = prepare_inputs()
    scope = VideoScope('train')
    cases = select_train_cases(scope.rows)
    protected[str(Path(__file__).resolve())] = sha256(__file__)
    diagnostic = FOLDER/'artifacts/stability_v1_km0d6q26'
    for name in ('report.json','interpretation.json'):
        protected[str(diagnostic/name)] = sha256(diagnostic/name)
    checkpoints = []
    for d in summary['details']:
        folder = Path(d['TANet_run'])
        for tag in ('best','last'):
            checkpoints.append({'seed':d['seed'], 'snapshot':tag,
                'checkpoint':str(folder/f'{tag}.pt'), 'contract':str(folder/'contract.json'),
                'epoch': d['TANet_best_epoch'] if tag == 'best' else read(folder/'run_state.json')['epoch']})
    out = Path(tempfile.mkdtemp(prefix='bn_behavior_v1_',dir=FOLDER/'artifacts'))
    (out/'inputs').mkdir();(out/'plots').mkdir()
    shutil.copy2(__file__,out/'inputs/probe_bn_behavior_v1.py')
    plan = {'schema':'TAM_BN_train_forward_fixed16_v1','sampling_seed':SEED,
        'created_at_taiwan':datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        'sampling':'2 per original train class; source spread then SHA ranking, before pixels/predictions',
        'cases':cases,'pairs':[[cases[i]['video_id'],cases[i+1]['video_id']] for i in range(0,16,2)],
        'pair_policy':'within same original class, source-spread; context dependence is conditional on these fixed pairs',
        'checkpoints':checkpoints,'modes':[list(m) for m in MODES],
        'expected_model_forward_calls':6*(16*4+8*2+1),
        'expected_video_forward_instances':6*(16*6+1),
        'absolute_PMake_shift_thresholds':[.01,.05,.1],
        'dropout':'disabled for every condition', 'pretrained_BN2d':'saved running statistics for every condition',
        'TAM_batch_mode':'training=True, track_running_stats=False, restores flags; no running-buffer updates',
        'precision':'same deterministic BF16 autocast/FP32 parameters as formal training',
        'repeat_control':'replay first saved_b1 video after all conditions; require bitwise logits equality',
        'BN_layer_capture':'32 TAM BN1d outputs per forward; relative L2 against same-video saved_b1; input batch moment vs saved running stats',
        'source_data_identity':scope.identity,'protected_sha256':protected,
        'optimizer_steps':0,'parameter_updates':0,'validation_video_access':False,'test_video_access':False,
        'accuracy_or_CE_computed':False,'formal_training_started':False}
    atomic_save(out/'plan.json',plan,json_format=True)
    atomic_save(out/'run_state.json',{'status':'prepared','source_video_opens':0},json_format=True)
    print(out,flush=True)
    return out


def tam_bn_modules(model):
    return {name:m for name,m in model.named_modules() if '.tam.' in name and isinstance(m,nn.BatchNorm1d)}


@contextmanager
def bn_condition(model, branch):
    need(branch in ('saved','G','L','GL'), 'Unknown BN condition')
    need(not model.training, 'Start with eval model, not global train mode')
    modules = tam_bn_modules(model)
    old = {name:(m.training,m.track_running_stats) for name,m in modules.items()}
    try:
        for name,m in modules.items():
            use_batch = branch == 'GL' or branch != 'saved' and f'.{branch}.' in name
            m.train(use_batch)
            # This preserves training-time normalization without updating the
            # diagnostic model's running_mean/var/num_batches_tracked.
            m.track_running_stats = not use_batch
        need(all(not m.training for m in model.modules() if isinstance(m,(nn.Dropout,nn.BatchNorm2d))),
             'Dropout or pretrained BN2d accidentally enabled')
        yield
    finally:
        for name,m in modules.items():
            m.training,m.track_running_stats = old[name]


class BNCapture:
    def __init__(self, model, seed, snapshot):
        self.seed,self.snapshot=seed,snapshot
        self.baseline={};self.rows=[];self.ids=[];self.mode=None
        self.handles=[m.register_forward_hook(self.hook(name)) for name,m in tam_bn_modules(model).items()]

    def hook(self,name):
        def capture(module,inputs,output):
            bs=len(self.ids)
            need(bs>0, 'Capture requires fixed batch IDs')
            # Clone: the author's following ReLU(inplace=True) must not mutate
            # captured CPU baselines, including when this helper is CPU-tested.
            y=output.detach().float().reshape(bs,-1).cpu().clone()
            x=inputs[0].detach().float().cpu()
            dims=(0,) if x.ndim==2 else (0,2)
            need(x.ndim in (2,3), 'Unexpected BN1d input')
            mean=x.mean(dim=dims).double();var=x.var(dim=dims,unbiased=False).double()
            running_mean=module.running_mean.detach().cpu().double()
            running_var=module.running_var.detach().cpu().double()
            standardized=(mean-running_mean).abs()/torch.sqrt(running_var+module.eps)
            logratio=torch.log10((var+module.eps)/(running_var+module.eps))
            need(torch.isfinite(y).all().item() and torch.isfinite(standardized).all().item(), 'Nonfinite BN activation')
            for i,vid in enumerate(self.ids):
                if self.mode=='saved_b1':self.baseline[(name,vid)]=y[i].clone()
                ref=self.baseline[(name,vid)]
                denom=float(ref.double().norm())
                delta=y[i].double()-ref.double()
                self.rows.append({'seed':self.seed,'snapshot':self.snapshot,'mode':self.mode,
                    'video_id':vid,'paired_batch_ids':'|'.join(self.ids),'layer':name,
                    'relative_L2_output':float(delta.norm())/max(denom,1e-12),
                    'baseline_output_norm':denom,'baseline_norm_below_1e-12':denom<1e-12,
                    'output_RMSE':float(delta.square().mean().sqrt()),
                    'input_mean_gap_saved_sd_median':float(standardized.median()),
                    'input_mean_gap_saved_sd_p95':float(torch.quantile(standardized,.95)),
                    'input_log10_variance_ratio_median':float(logratio.median())})
        return capture

    def close(self):
        for h in self.handles:h.remove()


def compare_logits(logits, reference):
    a,b=logits.detach().cpu().double(),reference.detach().cpu().double()
    need(a.shape==b.shape==(2,) and torch.isfinite(a).all().item() and torch.isfinite(b).all().item(), 'Invalid two-logit pair')
    p,q=float(a.softmax(0)[1]),float(b.softmax(0)[1])
    return {'p_make':p,'saved_b1_p_make':q,'signed_PMake_shift':p-q,'absolute_PMake_shift':abs(p-q),
            'prediction':int(a.argmax()),'saved_b1_prediction':int(b.argmax()),
            'prediction_flipped':int(a.argmax())!=int(b.argmax()),
            'margin':float(a[1]-a[0]),'saved_b1_margin':float(b[1]-b[0]),
            'absolute_margin_shift':float(((a[1]-a[0])-(b[1]-b[0])).abs())}


def summarize_outputs(rows):
    need(rows, 'No paired outputs')
    return {'videos':len(rows),'prediction_flips':sum(r['prediction_flipped'] for r in rows),
            'absolute_PMake_shift':finite_summary([r['absolute_PMake_shift'] for r in rows]),
            'absolute_margin_shift':finite_summary([r['absolute_margin_shift'] for r in rows]),
            'shift_at_least':{str(t):sum(r['absolute_PMake_shift']>=t for r in rows) for t in (.01,.05,.1)}}


def execute(plan_path):
    plan_path=Path(plan_path).resolve();out=plan_path.parent
    need(plan_path.name=='plan.json' and out.parent==(FOLDER/'artifacts').resolve() and
         out.name.startswith('bn_behavior_v1_'), 'Foreign probe plan')
    plan=read(plan_path)
    need(plan['schema']=='TAM_BN_train_forward_fixed16_v1' and plan['sampling_seed']==SEED and
         plan['modes']==[list(m) for m in MODES] and read(out/'run_state.json')['status']=='prepared',
         'Plan changed or probe already executed')
    need(all(sha256(p)==h for p,h in plan['protected_sha256'].items()), 'Protected original/source changed')
    scope=VideoScope('train',[r['video_id'] for r in plan['cases']])
    canonical=VideoScope('train')
    need(select_train_cases(canonical.rows)==plan['cases'] and scope.identity==plan['source_data_identity'], 'Fixed sample/data changed')
    # Validate GPU before opening any original video. No CPU substitute/OOM retry.
    first_contract=read(plan['checkpoints'][0]['contract'])
    device=configure_runtime(first_contract['recipe'])
    free,total=torch.cuda.mem_get_info()
    need(free>=8*1024**3, 'Need8GiB GPU headroom; do not interrupt other jobs')
    state={'status':'running','source_video_opens':0,'completed_checkpoints':0,'model_forward_calls':0}
    atomic_save(out/'run_state.json',state,json_format=True)
    print('OUTPUT:',out,flush=True)
    started=time.monotonic();samples=[];outputs=[];activations=[];checks=[];metadata=[]
    try:
        for row in plan['cases']:
            sample=decode(scope,row['video_id'],InputConfig())
            need(sample['split']=='train' and sample['metadata']['video_open_count']==1, 'Train-only decode required')
            samples.append(sample)
            metadata.append({k:v for k,v in sample.items() if k!='tensor'}|
                            {'tensor_sha256':tensor_sha(sample['tensor'])})
            state['source_video_opens']+=1
        atomic_save(out/'decoded_inputs.json',metadata,json_format=True)
        for item in plan['checkpoints']:
            cfg=read(item['contract'])['recipe'];configure_runtime(cfg)
            saved=torch.load(item['checkpoint'],map_location='cpu',weights_only=True)
            need(saved['kind']=='formal' and saved['epoch']==item['epoch'] and
                 saved['contract_sha256']==read(item['contract'])['contract_sha256'], 'Wrong frozen checkpoint')
            weights=saved['model'];del saved
            expected_digest=state_digest(weights)
            model=TANetBinary(frames=50,dropout=.5)
            model.load_state_dict(weights,strict=True);del weights
            model.requires_grad_(False).eval().to(device)
            need(len(tam_bn_modules(model))==32 and sum(isinstance(m,nn.BatchNorm2d) for m in model.modules())==53,
                 'Unexpected BN inventory')
            need(state_digest(model.state_dict())==expected_digest, 'Checkpoint load changed state')
            capture=BNCapture(model,item['seed'],item['snapshot'])
            baseline={};mode_outputs={}
            torch.cuda.reset_peak_memory_stats()
            local_calls=0
            with torch.inference_mode():
                for mode,branch,batchsize in MODES:
                    mode_outputs[mode]={}
                    capture.mode=mode
                    with bn_condition(model,branch):
                        for start in range(0,16,batchsize):
                            batch=samples[start:start+batchsize]
                            ids=[s['video_id'] for s in batch];capture.ids=ids
                            x=torch.stack([s['tensor'] for s in batch]).to(device)
                            with autocast(device):y=model(x)
                            y=y.detach().float().cpu()
                            need(y.shape==(len(batch),2) and torch.isfinite(y).all().item(), 'Invalid probe output')
                            local_calls+=1;state['model_forward_calls']+=1
                            for vid,z in zip(ids,y):
                                if mode=='saved_b1':baseline[vid]=z.clone()
                                mode_outputs[mode][vid]=z.clone()
                                outputs.append({'seed':item['seed'],'snapshot':item['snapshot'],'epoch':item['epoch'],
                                    'mode':mode,'video_id':vid,'paired_batch_ids':'|'.join(ids),
                                    'logit_Miss':float(z[0]),'logit_Make':float(z[1]),**compare_logits(z,baseline[vid])})
                            del x,y
                    atomic_save(out/'run_state.json',state|{'active_seed':item['seed'],'active_snapshot':item['snapshot'],'last_completed_mode':mode},json_format=True)
                    print(item['seed'],item['snapshot'],mode,'done',flush=True)
                # Reproducibility check after every mode, with all flags restored.
                capture.mode='saved_repeat_b1';capture.ids=[samples[0]['video_id']]
                with autocast(device):again=model(samples[0]['tensor'].unsqueeze(0).to(device))
                repeat_equal=torch.equal(again[0].float().cpu(),baseline[capture.ids[0]])
                need(repeat_equal, 'Saved-mode replay changed logits')
                local_calls+=1;state['model_forward_calls']+=1
            capture.close()
            need(all(not m.training for m in model.modules()) and
                 all(m.track_running_stats for m in tam_bn_modules(model).values()), 'Diagnostic flags not restored')
            need(all(p.grad is None and not p.requires_grad for p in model.parameters()), 'Unexpected gradient')
            need(state_digest(model.state_dict())==expected_digest, 'Parameters or buffers changed during forwards')
            context=[{'video_id':s['video_id'],**compare_logits(mode_outputs['GL_batch_b2'][s['video_id']],
                        mode_outputs['GL_batch_b1'][s['video_id']])} for s in samples]
            subset=[r for r in outputs if r['seed']==item['seed'] and r['snapshot']==item['snapshot']]
            checks.append({**item,'model_state_digest':expected_digest,'model_state_unchanged':True,
                'gradient_tensors_created':False,'dropout_disabled':True,'BN2d_always_eval':True,
                'saved_replay_bitwise_equal':repeat_equal,'forward_calls':local_calls,
                'modes':{mode:summarize_outputs([r for r in subset if r['mode']==mode]) for mode,_,_ in MODES},
                'GL_batch2_vs_GL_batch1':summarize_outputs(context),
                'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,
                'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3})
            write_csv(out/f'GL_batch_context_{item["seed"]}_{item["snapshot"]}.csv',context)
            activations.extend(capture.rows)
            state['completed_checkpoints']+=1
            del capture,model,baseline,mode_outputs,again
            gc.collect();torch.cuda.empty_cache()
        need(state['model_forward_calls']==plan['expected_model_forward_calls'], 'Unexpected forward budget')
        need(all(tensor_sha(s['tensor'])==m['tensor_sha256'] for s,m in zip(samples,metadata)), 'Cached input changed')
        need(all(sha256(p)==h for p,h in plan['protected_sha256'].items()), 'Original checkpoint/source changed')
        write_csv(out/'per_video_outputs.csv',outputs)
        write_csv(out/'BN_activation_comparisons.csv',activations)
        report={'status':'numeric_complete_render_pending','plan_sha256':sha256(plan_path),'checkpoints':checks,
            'source_video_opens':{'train':16,'validation':0,'test':0},
            'model_forward_calls':state['model_forward_calls'],'video_forward_instances':plan['expected_video_forward_instances'],
            'unique_videos':16,'optimizer_steps':0,'parameter_updates':0,'in_memory_model_state_unchanged':True,
            'original_artifacts_sha256_unchanged':True,'dropout_disabled':True,'formal_training_started':False,
            'accuracy_or_CE_computed':False,'validation_used':False,'test_used':False,
            'environment':{'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
                           'free_GiB_before':free/1024**3,'total_GiB':total/1024**3},
            'limitations':[
                '16 class-stratified train examples are diagnostic probes, not population performance or validation/test.',
                'Batch statistics in this experiment are not a deployable inference policy; no BN recalibration is saved.',
                'The batch2 comparison depends on the declared same-class/source-spread pairs.',
                'Dropout is off: this isolates BN behavior, not the complete stochastic training forward.',
                'G/L changes propagate through downstream layers; large later activation shifts are not independent causal localization.',
                'Relative L2 can be large when reference activation norm is small; absolute RMSE/norm are also saved.',
                'A demonstrated BN-induced output gap does not prove BN alone caused cross-seed validation instability.',
                'No retraining or larger-batch performance has been measured.'],
            'elapsed_seconds':time.monotonic()-started}
        render(out,report,activations)
        report['status']='completed';report['completed_at_taiwan']=datetime.now(ZoneInfo('Asia/Taipei')).isoformat()
        atomic_save(out/'report.json',report,json_format=True)
        atomic_save(out/'run_state.json',state|{'status':'completed','report_sha256':sha256(out/'report.json')},json_format=True)
        return out
    except BaseException as exc:
        atomic_save(out/'run_state.json',state|{'status':'failed','error':f'{type(exc).__name__}: {exc}'},json_format=True)
        raise


def render(out,report,activations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes=[m[0] for m in MODES if m[0]!='saved_b1']
    cp=report['checkpoints'];x=np.arange(len(cp));width=.15
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for j,mode in enumerate(modes):
        axes[0].bar(x+(j-2)*width,[100*c['modes'][mode]['absolute_PMake_shift']['mean'] for c in cp],width,label=mode)
        axes[1].bar(x+(j-2)*width,[c['modes'][mode]['prediction_flips'] for c in cp],width,label=mode)
    labels=[f'{c["seed"]}\n{c["snapshot"]}' for c in cp]
    for ax in axes:ax.set_xticks(x,labels);ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
    axes[0].set(ylabel='Mean absolute P(Make) shift (percentage points)',title='Same16 train clips vs saved BN, batch1')
    axes[1].set(ylabel='Prediction flips /16',ylim=(0,16),title='Flips relative to saved BN, batch1')
    fig.savefig(out/'plots/output_shifts.png',dpi=140);plt.close(fig)
    layers=sorted({r['layer'] for r in activations});labels=[];data=[]
    for c in cp:
        for mode in ('G_batch_b1','L_batch_b1','GL_batch_b1'):
            selected=[r for r in activations if r['seed']==c['seed'] and r['snapshot']==c['snapshot'] and r['mode']==mode]
            labels.append(f'{c["seed"]} {c["snapshot"]} {mode}')
            data.append([np.median([r['relative_L2_output'] for r in selected if r['layer']==name]) for name in layers])
    fig,ax=plt.subplots(figsize=(16,8),constrained_layout=True)
    im=ax.imshow(np.log10(np.asarray(data)+1e-6),aspect='auto',cmap='viridis',vmin=-3,vmax=2)
    ax.set(xticks=range(32),xticklabels=[n.removeprefix('backbone.').removesuffix('.1') for n in layers],
           yticks=range(len(labels)),yticklabels=labels,title='Median per-video relative L2 of TAM BN outputs vs saved_b1 (log10)')
    ax.tick_params(axis='x',rotation=90,labelsize=7);ax.tick_params(axis='y',labelsize=8)
    fig.colorbar(im,ax=ax,label='log10(relative L2 +1e-6); fixed display range')
    fig.savefig(out/'plots/BN_activation_shifts.png',dpi=140);plt.close(fig)
    table=[]
    for c in cp:
        for mode in modes:
            m=c['modes'][mode]
            table.append(f'<tr><td>{c["seed"]}</td><td>{c["snapshot"]}</td><td>{mode}</td><td>{100*m["absolute_PMake_shift"]["mean"]:.2f}</td><td>{100*m["absolute_PMake_shift"]["max"]:.2f}</td><td>{m["prediction_flips"]}/16</td></tr>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TAM BN行為對照</title><style>body{font:18px system-ui;line-height:1.8;max-width:1250px;margin:32px auto;padding:0 20px;color:#192333}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:9px}img{max-width:100%}.notice{background:#fff1cc;padding:18px}</style><h1>TAM BN行為對照：固定16支train影片</h1><p class="notice">關閉Dropout，凍結所有參數，保持預訓練BN2d推論模式。只改記憶體副本中的TAM BN1d統計來源；甚至副本的running buffers也沒有更新。未做optimizer更新、重訓、BN校正存檔或validation/test。</p><p><a href="plan.json">看像素前固定的樣本／配對／條件</a> · <a href="report.json">完整數據與限制</a> · <a href="per_video_outputs.csv">逐片輸出與差異</a> · <a href="BN_activation_comparisons.csv">逐層BN輸入統計與輸出差異</a></p><p>saved_b1為保存統計、batch1基準；G／L／GL_batch_b1分別只改G、只改L、同時改兩者為當批統計；saved_b2為保持保存統計的batch2數值控制；GL_batch_b2為固定配對batch2當批統計。所有差異均對同片saved_b1，並非準確率或效能提升。</p><img src="plots/output_shifts.png"><table><tr><th>Seed</th><th>權重</th><th>條件</th><th>平均絕對P(Make)差（百分點）</th><th>最大差（百分點）</th><th>判斷翻轉</th></tr>'''+''.join(table)+'''</table><h2>逐層變化</h2><img src="plots/BN_activation_shifts.png"><p>G／L的改動會沿網路傳遞，不把下游變化直接當作該層的獨立原因。相對L2分母小時會偏大，CSV同時保留絕對RMSE及參考norm。</p><h2>驗證與限制</h2><p>六份checkpoint在全部條件後，模型參數與buffers的雜湊皆不變；第一支片的saved_b1重播logits皆逐值相同。原權重／程式／資料清單SHA256亦不變。</p><ul>'''+''.join('<li>'+html.escape(s)+'</li>' for s in report['limitations'])+'</ul></html>'
    (out/'index.html').write_text(page)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare',action='store_true')
    modes.add_argument('--run',type=Path,help='Existing prepared plan.json; bounded GPU forward probe only')
    args=parser.parse_args()
    print(prepare() if args.prepare else execute(args.run))
