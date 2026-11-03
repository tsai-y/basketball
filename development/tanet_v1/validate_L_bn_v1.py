"""Predeclared full-validation comparison: saved BN vs per-video TAM L stats.

Fixed batch1, G/pretrained BN saved statistics, disabled Dropout, frozen model.
L uses batch statistics without updating any running buffers. No training,
checkpoint reselection, calibration fitting, threshold changes or test access.
"""
import argparse
from collections import Counter
from datetime import datetime
import gc
import html
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch import nn

from .config import FOLDER, InputConfig, need, sha256
from .data import VideoScope, decode
from .model import TANetBinary
from .training import atomic_save, configure_runtime, state_digest, autocast, metrics
from .diagnose_stability_v1 import (prepare_inputs, validate_ids, rows_from_logits,
                                    grouped_metrics, write_csv)
from .probe_bn_behavior_v1 import bn_condition, tam_bn_modules, tensor_sha
from .repeat_seeds_v1 import process_identity

POLICY = {
    'schema':'full_validation_TAM_L_per_video_stats_v1',
    'modes':['saved_b1','L_batch_b1'], 'batch_size':1,
    'candidate':'TAM_L_BN1d_per_video_batch_statistics_no_buffer_update',
    'G_BN':'saved_running_stats', 'pretrained_BN2d':'saved_running_stats',
    'dropout':'disabled', 'precision':'formal_deterministic_BF16_autocast_FP32_parameters',
    'checkpoint_selection':'preserve_original_best_epoch_selected_by_original_validation_CE',
    'last_snapshots':'auxiliary_diagnostic_only_not_reselection',
    'threshold':'unchanged_argmax_two_logits', 'temperature_fitting':False,
    'optimizer_steps':0, 'parameter_updates':0, 'BN_buffer_updates':0,
    'validation_count':1547, 'expected_model_forward_calls':6*1547*2,
    'baseline_replay':'require_bitwise_saved_float32_logit_equality_for_all9282_pairs',
    'test_access':False, 'automatic_formal_training':False,
}


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat()


def validate_policy(plan):
    need(plan['policy']==POLICY, 'Predeclared L-only/batch1 policy changed')
    need([(c['seed'],c['snapshot']) for c in plan['checkpoints']]==
         [(seed,tag) for seed in (20261024,20261025,20261026) for tag in ('best','last')],
         'Wrong checkpoint cohort; no seed or epoch selection')
    need(plan['validation_ids']==sorted(set(plan['validation_ids'])) and len(plan['validation_ids'])==1547,
         'Incomplete/reordered validation cohort')


def prepare():
    need(not torch.cuda.is_initialized(), 'Prepare is metadata-only')
    summary,protected,_=prepare_inputs()
    previous=FOLDER/'artifacts/bn_behavior_v1_2q0xo24h'
    oldplan=read(previous/'plan.json')
    need(read(previous/'run_state.json')['status']=='completed', 'Train-only BN probe incomplete')
    protected.update(oldplan['protected_sha256'])
    for name in ('plan.json','report.json','interpretation.json'):
        protected[str(previous/name)]=sha256(previous/name)
    protected[str(Path(__file__).resolve())]=sha256(__file__)
    scope=VideoScope('validation')
    need(len(scope.rows)==1547, 'Canonical validation count changed')
    checkpoints=[]
    for d in summary['details']:
        run=Path(d['TANet_run'])
        for tag in ('best','last'):
            checkpoints.append({'seed':d['seed'],'snapshot':tag,'checkpoint':str(run/f'{tag}.pt'),
                'contract':str(run/'contract.json'),
                'epoch':d['TANet_best_epoch'] if tag=='best' else read(run/'run_state.json')['epoch']})
    plan={'policy':POLICY,'created_at_taiwan':now(),'checkpoints':checkpoints,
          'validation_ids':[r['video_id'] for r in scope.rows],
          'data_identity':scope.identity,'protected_sha256':protected,
          'basis':'L branch chosen from the preceding fixed train16 diagnostic before new validation outputs',
          'mainline_reference_summary':str(FOLDER/'artifacts/multiseed_c0qf_w7l/summary.json')}
    validate_policy(plan)
    need(all(sha256(p)==h for p,h in protected.items()), 'Original source/artifact changed')
    out=Path(tempfile.mkdtemp(prefix='validation_L_v1_',dir=FOLDER/'artifacts'))
    (out/'inputs').mkdir();(out/'plots').mkdir()
    shutil.copy2(__file__,out/'inputs/validate_L_bn_v1.py')
    atomic_save(out/'plan.json',plan,json_format=True)
    atomic_save(out/'run_state.json',{'status':'prepared','validation_videos_processed':0,'model_forward_calls':0},json_format=True)
    print(out,flush=True)
    return out


def require_replay(actual,expected):
    need(actual.shape==expected.shape==(2,) and actual.dtype==expected.dtype==torch.float32 and
         torch.equal(actual,expected), 'Baseline logits did not reproduce saved validation bitwise; stop before interpreting candidate')


def forward_one(model,x,branch):
    need(x.shape==(1,3,50,192,256), 'Fixed batch1/full input required')
    need(branch in ('saved','L'), 'Only saved or predeclared L strategy permitted')
    with bn_condition(model,branch), torch.inference_mode(), autocast(x.device):
        result=model(x)
    need(result.shape==(1,2) and torch.isfinite(result).all().item(), 'Invalid validation output')
    return result[0].detach().float().cpu()


def make_scores(logits,labels,ids,allowed):
    score=metrics(logits,labels)
    saved={'validation':{'logits':logits,'labels':labels,'video_ids':ids,'metrics':score}}
    rows=rows_from_logits(saved,allowed)
    return {'CE':score['CE'],**grouped_metrics(rows)},rows


def paired_change(baseline,candidate):
    need(len(baseline)==len(candidate)>0, 'Paired validation rows required')
    a={r['video_id']:r for r in baseline};b={r['video_id']:r for r in candidate}
    need(len(a)==len(b)==len(baseline) and set(a)==set(b), 'Duplicate or mismatched pairing')
    counts=Counter();rows=[]
    for vid in sorted(a):
        x,y=a[vid],b[vid]
        need(x['true_label']==y['true_label'], 'Paired labels changed')
        left=x['prediction']==x['true_label'];right=y['prediction']==y['true_label']
        pattern='both_correct' if left and right else 'both_wrong' if not left and not right else 'L_only_correct' if right else 'saved_only_correct'
        counts[pattern]+=1
        rows.append({'video_id':vid,'true_label':x['true_label'],'source_group':x['source_group'],
            'shot_type':x['shot_type'],'saved_prediction':x['prediction'],'L_prediction':y['prediction'],
            'saved_p_make':x['p_make'],'L_p_make':y['p_make'],
            'saved_nll':x['nll'],'L_nll':y['nll'],'pattern':pattern})
    return {'counts':{k:counts[k] for k in ('both_correct','both_wrong','L_only_correct','saved_only_correct')},
            'net_additional_correct':counts['L_only_correct']-counts['saved_only_correct'],
            'prediction_flips':counts['L_only_correct']+counts['saved_only_correct']},rows


def cohort_summary(details):
    result={}
    for tag in ('best','last'):
        cohort=[d for d in details if d['snapshot']==tag]
        need([d['seed'] for d in cohort]==[20261024,20261025,20261026], 'Report all three original seeds')
        result[tag]={}
        for mode in POLICY['modes']:
            values={k:[d[mode]['overall']['metrics'][k] for d in cohort] for k in ('accuracy','macro_f1')}
            values['CE']=[d[mode]['CE'] for d in cohort]
            for label in ('Miss','Make'):
                values[label+'_recall']=[d[mode]['overall']['metrics']['per_class'][label]['recall'] for d in cohort]
            result[tag][mode]={k:{'values':v,'mean':statistics.mean(v),'sample_sd':statistics.stdev(v)} for k,v in values.items()}
    return result


def execute(plan_path):
    plan_path=Path(plan_path).resolve();out=plan_path.parent
    need(plan_path.name=='plan.json' and out.parent==(FOLDER/'artifacts').resolve() and out.name.startswith('validation_L_v1_'), 'Foreign validation plan')
    plan=read(plan_path);validate_policy(plan)
    need(read(out/'run_state.json')['status']=='prepared', 'This plan already started; no silent restart or extra validation')
    need(all(sha256(p)==h for p,h in plan['protected_sha256'].items()), 'Source/checkpoint changed')
    scope=VideoScope('validation')
    ids=[r['video_id'] for r in scope.rows];labels=torch.tensor([int(r['binary_label']) for r in scope.rows])
    need(ids==plan['validation_ids'] and scope.identity==plan['data_identity'], 'Canonical validation identity changed')
    device=configure_runtime(read(plan['checkpoints'][0]['contract'])['recipe'])
    free,total=torch.cuda.mem_get_info();need(free>=8*1024**3, 'Need8GiB free GPU memory; do not interrupt other jobs')
    state={'status':'running','validation_videos_processed':0,'source_video_opens':{'train':0,'validation':0,'test':0},
           'model_forward_calls':0,'baseline_logits_bitwise_matched':0,'started_at_taiwan':now(),
           'process':process_identity(os.getpid())}
    atomic_save(out/'run_state.json',state,json_format=True)
    print('OUTPUT:',out,flush=True)
    started=time.monotonic();loaded=[];input_meta=[];details=[];all_rows=[]
    with (out/'progress.log').open('x',buffering=1) as logfile:
        def log(text):
            print(text,flush=True);logfile.write(now()+' '+text+'\n')
        try:
            for item in plan['checkpoints']:
                cfg=read(item['contract'])['recipe'];configure_runtime(cfg)
                saved=torch.load(item['checkpoint'],map_location='cpu',weights_only=True)
                need(saved['kind']=='formal' and saved['epoch']==item['epoch'] and
                     saved['contract_sha256']==read(item['contract'])['contract_sha256'], 'Wrong fixed checkpoint')
                need(saved['runtime']['torch']==str(torch.__version__) and saved['runtime']['cuda']==torch.version.cuda and
                     saved['runtime']['gpu']==torch.cuda.get_device_name(), 'Runtime changed since saved baseline')
                original=saved['validation']
                validate_ids(original['video_ids'],original['labels'].tolist(),scope.allowed)
                reference={vid:logit.clone() for vid,logit in zip(original['video_ids'],original['logits'])}
                before=state_digest(saved['model'])
                model=TANetBinary(frames=50,dropout=.5)
                model.load_state_dict(saved['model'],strict=True);model.requires_grad_(False).eval().to(device)
                need(len(tam_bn_modules(model))==32 and sum(isinstance(m,nn.BatchNorm2d) for m in model.modules())==53,
                     'Unexpected TAM/BN inventory')
                need(state_digest(model.state_dict())==before, 'Checkpoint load changed model state')
                loaded.append({'item':item,'model':model,'digest':before,'reference':reference,
                               'original_metrics':original['metrics'],
                               'saved_b1':[],'L_batch_b1':[]})
                del saved,original
            torch.cuda.reset_peak_memory_stats()
            for index,vid in enumerate(ids):
                sample=decode(scope,vid,InputConfig())
                need(sample['split']=='validation' and sample['binary_label']==int(labels[index]) and
                     sample['metadata']['video_open_count']==1, 'Wrong decoded sample')
                state['source_video_opens']['validation']+=1
                x=sample['tensor'].unsqueeze(0).to(device)
                for entry in loaded:
                    baseline=forward_one(entry['model'],x,'saved');state['model_forward_calls']+=1
                    require_replay(baseline,entry['reference'][vid]);state['baseline_logits_bitwise_matched']+=1
                    candidate=forward_one(entry['model'],x,'L');state['model_forward_calls']+=1
                    entry['saved_b1'].append(baseline);entry['L_batch_b1'].append(candidate)
                input_meta.append({'video_id':vid,'tensor_sha256':tensor_sha(sample['tensor']),
                                   'metadata':sample['metadata']})
                del sample,x
                state['validation_videos_processed']=index+1
                if (index+1)%100==0 or index+1==len(ids):
                    atomic_save(out/'run_state.json',state,json_format=True)
                    log(f'validation={index+1}/{len(ids)} checkpoints=6 modes=2 forwards={state["model_forward_calls"]} baseline_matches={state["baseline_logits_bitwise_matched"]} elapsed_seconds={time.monotonic()-started:.1f}')
            peak_allocated=torch.cuda.max_memory_allocated()/1024**3
            peak_reserved=torch.cuda.max_memory_reserved()/1024**3
            for entry in loaded:
                item,model=entry['item'],entry['model']
                need(state_digest(model.state_dict())==entry['digest'], 'Parameters or buffers changed during validation')
                need(all(p.grad is None and not p.requires_grad for p in model.parameters()) and
                     all(not m.training for m in model.modules()) and all(m.track_running_stats for m in tam_bn_modules(model).values()),
                     'Gradients or diagnostic flags remained')
                detail={**item,'model_state_unchanged':True,'baseline_all1547_logits_bitwise_equal':True}
                paired_rows={}
                for mode in POLICY['modes']:
                    logits=torch.stack(entry[mode])
                    scores,rows=make_scores(logits,labels,ids,scope.allowed)
                    detail[mode]=scores;paired_rows[mode]=rows
                    if mode=='saved_b1':
                        need(scores['overall']['metrics']['confusion_matrix']==entry['original_metrics']['confusion_matrix_Miss_Make'] and
                             abs(scores['CE']-entry['original_metrics']['CE'])<1e-7, 'Saved baseline metrics changed')
                    all_rows.extend({'seed':item['seed'],'snapshot':item['snapshot'],'epoch':item['epoch'],
                        'mode':mode,'logit_Miss':float(z[0]),'logit_Make':float(z[1]),**r} for z,r in zip(logits,rows))
                detail['paired'],paired=paired_change(paired_rows['saved_b1'],paired_rows['L_batch_b1'])
                write_csv(out/f'paired_seed{item["seed"]}_{item["snapshot"]}.csv',paired)
                details.append(detail)
                log(f'{item["seed"]} {item["snapshot"]} complete; original epoch={item["epoch"]}; net_correct_change={detail["paired"]["net_additional_correct"]}')
            need(state['model_forward_calls']==POLICY['expected_model_forward_calls'] and
                 state['baseline_logits_bitwise_matched']==6*1547, 'Forward/replay budget mismatch')
            need(all(sha256(p)==h for p,h in plan['protected_sha256'].items()), 'Protected original changed')
            write_csv(out/'per_video_outputs.csv',all_rows)
            atomic_save(out/'decoded_inputs.json',input_meta,json_format=True)
            report={'status':'numeric_complete_render_pending','policy':POLICY,'plan_sha256':sha256(plan_path),
                'details':details,'aggregate':cohort_summary(details),
                'mainline_reference':read(plan['mainline_reference_summary'])['aggregate']['mainline'],
                'unique_validation_videos':1547,'prediction_rows':len(all_rows),
                'source_video_opens':state['source_video_opens'],'model_forward_calls':state['model_forward_calls'],
                'baseline_logits_bitwise_matches':state['baseline_logits_bitwise_matched'],
                'parameters_and_buffers_unchanged':True,'original_artifacts_sha256_unchanged':True,
                'optimizer_steps':0,'parameter_updates':0,'checkpoint_reselection':False,'test_used':False,
                'environment':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
                    'free_GiB_before':free/1024**3,'peak_allocated_GiB':peak_allocated,'peak_reserved_GiB':peak_reserved},
                'elapsed_seconds':time.monotonic()-started,
                'limitations':[
                    'New inference strategy on existing validation, not test or an independent holdout.',
                    'L strategy was proposed using train16 behavior, but these validation data already informed earlier model selection/research.',
                    'Original best checkpoints are retained; last checkpoints are auxiliary diagnostics, not a new selection pool.',
                    'Fixed batch1 means L statistics use only the current video across time; no statistics are pooled across validation videos.',
                    'No running buffers are fitted or updated; no weights or decision thresholds change.',
                    'Three seeds are descriptive repeats on the same1547 videos; sample SD is not a confidence interval.',
                    'A validation gain would support the BN normalization hypothesis, not prove it is the only cause or guarantee test performance.',
                    'No additional G/GL strategies, temperatures, batches, epochs or source exclusions were searched in this validation run.']}
            render(out,report)
            report['status']='completed';report['completed_at_taiwan']=now()
            atomic_save(out/'report.json',report,json_format=True)
            atomic_save(out/'run_state.json',state|{'status':'completed','completed_at_taiwan':now(),
                        'report_sha256':sha256(out/'report.json')},json_format=True)
            log('Completed full validation comparison; no training/test/checkpoint changes.')
            return out
        except BaseException as exc:
            atomic_save(out/'run_state.json',state|{'status':'failed','error':f'{type(exc).__name__}: {exc}'},json_format=True)
            log(f'FAILED: {type(exc).__name__}: {exc}')
            raise
        finally:
            loaded.clear();gc.collect();torch.cuda.empty_cache()


def render(out,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    details=report['details'];x=np.arange(len(details));fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for shift,mode,label in [(-.18,'saved_b1','Original saved BN'),(.18,'L_batch_b1','TAM L per-video stats')]:
        for ax,metric in zip(axes,('accuracy','macro_f1','CE')):
            values=[d[mode]['CE'] if metric=='CE' else d[mode]['overall']['metrics'][metric] for d in details]
            ax.bar(x+shift,values,.36,label=label)
            ax.set(title=metric,xticks=x,xticklabels=[f'{d["seed"]}\n{d["snapshot"]}' for d in details])
            ax.tick_params(axis='x',rotation=35,labelsize=8);ax.grid(axis='y',alpha=.2)
            ax.legend(fontsize=8)
    fig.savefig(out/'plots/metric_comparison.png',dpi=140);plt.close(fig)
    sources=sorted(details[0]['saved_b1']['by_source'])
    differences=[[100*(d['L_batch_b1']['by_source'][g]['metrics']['accuracy']-d['saved_b1']['by_source'][g]['metrics']['accuracy']) for g in sources] for d in details]
    limit=max(1.,float(np.abs(differences).max()))
    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    image=ax.imshow(differences,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    for i,row in enumerate(differences):
        for j,v in enumerate(row):ax.text(j,i,f'{v:+.1f}',ha='center',va='center',color='white' if abs(v)>.65*limit else 'black')
    ax.set(xticks=range(len(sources)),xticklabels=[f'{g}\nn={details[0]["saved_b1"]["by_source"][g]["metrics"]["videos"]}' for g in sources],
           yticks=range(len(details)),yticklabels=[f'{d["seed"]} {d["snapshot"]}' for d in details],title='Validation accuracy change: L strategy minus original (percentage points)')
    fig.colorbar(image,ax=ax);fig.savefig(out/'plots/source_accuracy_change.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(len(details),2,figsize=(8,3*len(details)),squeeze=False,constrained_layout=True)
    for i,d in enumerate(details):
        for j,mode in enumerate(POLICY['modes']):
            cm=d[mode]['overall']['metrics']['confusion_matrix'];ax=axes[i,j]
            ax.imshow(cm,cmap='Blues',vmin=0,vmax=807)
            for y in (0,1):
                for z in (0,1):ax.text(z,y,str(cm[y][z]),ha='center',va='center',color='white' if cm[y][z]>500 else 'black')
            ax.set(xticks=[0,1],xticklabels=['Miss','Make'],yticks=[0,1],yticklabels=['Miss','Make'],
                   xlabel='Prediction',ylabel='Dataset label',title=f'{d["seed"]} {d["snapshot"]}: {mode}')
    fig.savefig(out/'plots/confusion_matrices.png',dpi=140);plt.close(fig)
    rows=[]
    for d in details:
        for mode in POLICY['modes']:
            m=d[mode]['overall']['metrics']
            rows.append(f'<tr><td>{d["seed"]}</td><td>{d["snapshot"]} / {d["epoch"]}</td><td>{mode}</td><td>{d[mode]["CE"]:.4f}</td><td>{100*m["accuracy"]:.2f}%</td><td>{m["macro_f1"]:.4f}</td><td>{100*m["per_class"]["Miss"]["recall"]:.2f}%</td><td>{100*m["per_class"]["Make"]["recall"]:.2f}%</td></tr>')
    aggregate=[]
    for mode in POLICY['modes']:
        a=report['aggregate']['best'][mode]
        aggregate.append(f'<tr><td>{mode}</td><td>{100*a["accuracy"]["mean"]:.2f}% ± {100*a["accuracy"]["sample_sd"]:.2f} pp</td><td>{a["macro_f1"]["mean"]:.4f} ± {a["macro_f1"]["sample_sd"]:.4f}</td><td>{a["CE"]["mean"]:.4f} ± {a["CE"]["sample_sd"]:.4f}</td></tr>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TAM L分支完整validation對照</title><style>body{font:18px system-ui;line-height:1.8;max-width:1250px;margin:32px auto;padding:0 20px;color:#192333}td,th{border:1px solid #ccd;padding:9px}table{border-collapse:collapse}img{max-width:100%}.notice{padding:18px;background:#fff1cc}</style><h1>TAM L分支完整validation對照</h1><p class="notice">這是新推論策略的validation診斷，非test。固定batch1，只把TAM L BN改用當片統計；G、預訓練BN2d、Dropout與其他設定維持原推論。參數與buffers不變，未重訓或校正存檔。原best epoch選模不變；last只作輔助診斷。</p><p><a href="plan.json">新推論前固定的策略與範圍</a> · <a href="report.json">完整指標、來源、類別與限制</a> · <a href="per_video_outputs.csv">18564筆輸出與逐片NLL</a> · <a href="progress.log">執行日誌</a></p><h2>原best固定快照：三seed平均±樣本標準差</h2><p>標準差來自三個seed，不是對來源或影片的信賴區間。</p><table><tr><th>策略</th><th>Accuracy</th><th>Macro-F1</th><th>CE</th></tr>'''+''.join(aggregate)+'''</table><img src="plots/metric_comparison.png"><h2>全部快照</h2><table><tr><th>Seed</th><th>快照／epoch</th><th>策略</th><th>CE</th><th>Accuracy</th><th>Macro-F1</th><th>Miss recall</th><th>Make recall</th></tr>'''+''.join(rows)+'''</table><img src="plots/source_accuracy_change.png"><img src="plots/confusion_matrices.png"><h2>驗證與限制</h2><p>所有9282筆原策略logits均逐值重現保存結果；6份模型在完整推論後參數／buffers雜湊不變，原權重檔與來源SHA256也維持不變。原片各解碼一次，train／test開啟皆0，validation開啟1547。</p><ul>'''+''.join('<li>'+html.escape(s)+'</li>' for s in report['limitations'])+'</ul></html>'
    (out/'index.html').write_text(page)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--run',type=Path)
    args=parser.parse_args()
    print(prepare() if args.prepare else execute(args.run))
