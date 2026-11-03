"""Fixed-update positive-exposure-matched negative-crop pilot; no formal test."""
from copy import deepcopy
from pathlib import Path
import argparse
import math
import os
import random
import shutil
import tempfile
import traceback

import cv2
import numpy as np
import torch
from torch import nn

from .probe import ARTIFACTS,read,save,sha,require,within
from .ball_negative_crops_v1 import check as check_candidates
from .ball_only_probe_v1 import check as check_ball,runtime_at
from .ball_pretraining_diagnostic_v1 import map_ball_predictions
from .visual_check_expansion_v1 import count_matches,labeled

SCHEMA='paired_fixed_update_ball_negative_crop_v1'
SEED=20261025
CYCLES=30
NEGATIVE_WEIGHT=.25


def schedule(ids,cycles):
    require(len(ids)==11 and len(set(ids))==11,'Fixed11 unique positive images required')
    require(type(cycles) is int and 1<=cycles<=CYCLES,'Invalid fixed cycle budget')
    rng=random.Random(SEED);result=[]
    for cycle in range(cycles):
        order=sorted(ids);rng.shuffle(order)
        result.extend({'cycle':cycle+1,'source_image_id':iid} for iid in order)
    return result


def encode(raw,boxes):
    """Deterministic square640 letterbox: BGR->RGB, normalized xywh targets."""
    require(raw is not None and raw.dtype==np.uint8 and raw.ndim==3 and raw.shape[2]==3,'Invalid native pixels')
    h,w=raw.shape[:2];require(w*3==h*4 and (w,h) in ((320,240),(160,120),(128,96)),'Unexpected native aspect/size')
    out=cv2.copyMakeBorder(cv2.resize(raw,(640,480),interpolation=cv2.INTER_LINEAR),80,80,0,0,cv2.BORDER_CONSTANT,value=(114,114,114))
    image=torch.from_numpy(np.ascontiguousarray(out[:,:,::-1].transpose(2,0,1))).float().unsqueeze(0)/255
    converted=[]
    for box in boxes:
        require(len(box)==4 and all(math.isfinite(float(v)) for v in box),'Invalid box')
        x1,y1,x2,y2=box;require(0<=x1<x2<=w and 0<=y1<y2<=h,'Out-of-frame ball')
        converted.append([(x1+x2)/(2*w),((y1+y2)*480/(2*h)+80)/640,(x2-x1)/w,(y2-y1)*480/(h*640)])
    return {'img':image,'batch_idx':torch.zeros(len(boxes)),
            'cls':torch.zeros((len(boxes),1)),'bboxes':torch.tensor(converted,dtype=torch.float32).reshape(-1,4)}


def train_mode(model):
    """Only Detect parameters train; all BN statistics fixed in BOTH arms."""
    model.eval();model.model[-1].train()
    for layer in model.modules():
        if isinstance(layer,nn.modules.batchnorm._BatchNorm):layer.eval()


def combined_loss(positive,negative,weight):
    require(weight in (0.,NEGATIVE_WEIGHT),'Only preregistered negative weights allowed')
    return positive if weight==0 else positive+weight*negative


def prepare(candidate_dir):
    candidate_dir=Path(candidate_dir).resolve();c=check_candidates(candidate_dir)
    source=Path(c['source_ball_dataset']);m=check_ball(source)
    folder=Path(tempfile.mkdtemp(prefix='ball_negative_pair_',dir=ARTIFACTS))
    for name in ('positive/train','positive/val','negative','inputs'):(folder/name).mkdir(parents=True)
    protected={str(candidate_dir/'manifest.json'):sha(candidate_dir/'manifest.json'),str(source/'dataset_manifest.json'):sha(source/'dataset_manifest.json'),m['initializer']:m['initializer_sha256']}
    rows=[];neg={x['source_image_id']:x for x in c['candidates']}
    for r in m['records']:
        original=within(source,r['image']);dest=folder/f'positive/{r["partition"]}/{r["image_id"]}.png'
        shutil.copy2(original,dest);protected[str(original)]=sha(original)
        row={'image_id':r['image_id'],'partition':r['partition'],'source_group':r['source_group'],
             'video_id':r['video_id'],'source_split':r['source_split'],'image':str(dest.relative_to(folder)),
             'image_sha256':sha(dest),'ball_boxes':[a['xyxy'] for a in r['annotations']]}
        if r['partition']=='train':
            n=neg[r['image_id']];p=within(candidate_dir,n['image']);dst=folder/'negative'/f'{r["image_id"]}.png'
            shutil.copy2(p,dst);protected[str(p)]=sha(p)
            row.update(negative_image=str(dst.relative_to(folder)),negative_image_sha256=sha(dst),
                       negative_provenance='deterministic_crop_excludes_complete_human_ball_boxes_plus12px',
                       negative_crop_proof=deepcopy(n['geometry_proof']))
        rows.append(row)
    save(folder/'inputs/candidates.json',c)
    spec={'schema':SCHEMA,'candidate_dir':str(candidate_dir),'source_ball_dataset':str(source),'records':rows,
          'initializer':m['initializer'],'initializer_sha256':m['initializer_sha256'],'protected_input_sha256':protected,
          'seed':SEED,'cycles':CYCLES,'updates_per_arm':11*CYCLES,'negative_weight':NEGATIVE_WEIGHT,
          'schedule':schedule([r['image_id'] for r in rows if r['partition']=='train'],CYCLES),
          'common_policy':{'imgsz':640,'batch_per_forward':1,'positive_forwards_per_update':1,
                           'learning_rate':1e-4,'weight_decay_weights_only':.0005,'optimizer':'AdamW',
                           'betas':[.9,.999],'grad_clip':10.,'all_BN_running_statistics_frozen':True,
                           'trainable_module':'Detect_only_except_fixed_DFL','AMP':False,'EMA':False,
                           'augmentation':False,'scheduler':False,'early_stopping':False},
          'primary_endpoint':'final_update330_no_model_selection; all30 cycles completed for BOTH arms',
          'validation_policy':'same3 full-frame development images after each11 updates; no threshold/weight search',
          'comparison':'control=L_positive; negative_crop=L_positive+0.25*L_negative, separate forwards/backwards then one optimizer step',
          'negative_compute_is_extra':True,'positive_exposure_and_optimizer_update_budgets_equal':True,
          'not_exact_replay_of_previous_trainer':True,'new_independent_data_added':False,
          'original_classification_validation_test_used':False,'feature_cache_modified':False,'generalization_metrics_valid':False}
    save(folder/'experiment.json',spec);check(folder);return folder


def check(folder):
    folder=Path(folder).resolve();require(folder.is_relative_to(ARTIFACTS.resolve()),'Foreign pair experiment')
    p=read(folder/'experiment.json');require(p['schema']==SCHEMA,'Wrong schema')
    candidates=check_candidates(p['candidate_dir']);m=check_ball(p['source_ball_dataset'])
    for path,h in p['protected_input_sha256'].items():require(sha(path)==h,'Protected input changed')
    expected={r['image_id']:r for r in m['records']};require(len(p['records'])==14,'Wrong image count')
    require({r['image_id'] for r in p['records']}==set(expected),'Wrong source allowlist')
    for r in p['records']:
        old=expected[r['image_id']]
        require(all(r[k]==old[k] for k in ('partition','source_group','video_id','source_split')),'Split identity changed')
        require(r['ball_boxes']==[a['xyxy'] for a in old['annotations']],'Ball geometry changed')
        require(sha(within(folder,r['image']))==r['image_sha256']==old['image_sha256'],'Positive pixels changed')
        if r['partition']=='train':
            original=next(x for x in candidates['candidates'] if x['source_image_id']==r['image_id'])
            require(r['negative_crop_proof']==original['geometry_proof'],'Negative proof changed')
            require(sha(within(folder,r['negative_image']))==r['negative_image_sha256']==original['image_sha256'],'Negative pixels changed')
        else:require('negative_image' not in r,'Development negative leaked into training')
    require(p['schedule']==schedule([r['image_id'] for r in p['records'] if r['partition']=='train'],CYCLES),'Schedule changed')
    require(p['cycles']==CYCLES and p['negative_weight']==NEGATIVE_WEIGHT and p['updates_per_arm']==330,'Budget changed')
    require(sha(p['initializer'])==p['initializer_sha256'],'Initializer changed')
    return p


def run_pair(folder,smoke=False):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    folder=Path(folder).resolve();p=check(folder);require(torch.cuda.is_available(),'CUDA required')
    run=Path(tempfile.mkdtemp(prefix='smoke_' if smoke else 'paired_train_',dir=folder));runtime_at(run)
    shutil.copy2(__file__,run/'source_snapshot.py')
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    import ultralytics
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=False)
    batches={}
    for r in p['records']:
        b=encode(cv2.imread(str(folder/r['image'])),r['ball_boxes'])
        batches[r['image_id']]={k:v.cuda() for k,v in b.items()}
        if r['partition']=='train':
            b=encode(cv2.imread(str(folder/r['negative_image'])),[])
            batches['N_'+r['image_id']]={k:v.cuda() for k,v in b.items()}
    plan=p['schedule'][:11] if smoke else p['schedule']
    state={'schema':SCHEMA,'status':'running','smoke':smoke,'experiment':str(folder),
           'source_code_sha256':sha(__file__),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
           'GPU':torch.cuda.get_device_name(0),'planned_updates_per_arm':len(plan),'arms':{}}
    save(run/'run_state.json',state);print('RUN:',run,flush=True)
    try:
        for arm,weight in [('control',0.),('negative_crop',NEGATIVE_WEIGHT)]:
            torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.cuda.reset_peak_memory_stats()
            wrapper=YOLO(p['initializer']);model=wrapper.model.float().cuda();head_index=len(model.model)-1
            require(wrapper.names=={0:'ball'},'Wrong class mapping');prefix=f'model.{head_index}.'
            for n,v in model.named_parameters():v.requires_grad_(n.startswith(prefix) and '.dfl.' not in n)
            model.args=get_cfg(overrides={'box':7.5,'cls':.5,'dfl':1.5})
            frozen={n:v.detach().cpu().clone() for n,v in model.state_dict().items()
                    if not n.startswith(prefix) or 'running_' in n or 'num_batches_tracked' in n or '.dfl.' in n}
            parameters=[(n,v) for n,v in model.named_parameters() if v.requires_grad]
            first_name,first=parameters[0];first_initial=first.detach().cpu().clone()
            groups=[{'params':[v for n,v in parameters if v.ndim>1 and not n.endswith('.bias')],'weight_decay':.0005},
                    {'params':[v for n,v in parameters if v.ndim<=1 or n.endswith('.bias')],'weight_decay':0.}]
            optimizer=torch.optim.AdamW(groups,lr=1e-4,betas=(.9,.999))
            info={'status':'running','negative_weight':weight,'updates':0,'positive_exposures':0,'negative_exposures':0,
                  'history':[],'trainable_parameters':sum(v.numel() for n,v in parameters)}
            state['arms'][arm]=info
            for step,item in enumerate(plan,1):
                train_mode(model);optimizer.zero_grad(set_to_none=True)
                require(all(not layer.training for layer in model.modules() if isinstance(layer,nn.modules.batchnorm._BatchNorm)),'BN mode changed')
                iid=item['source_image_id'];positive,parts=model(batches[iid]);positive=positive.sum()
                require(torch.isfinite(positive).item(),'Non-finite positive loss');positive.backward()
                neg_value=0.
                if weight:
                    negative,_=model(batches['N_'+iid]);negative=negative.sum()
                    require(torch.isfinite(negative).item(),'Non-finite negative loss')
                    (weight*negative).backward();neg_value=float(negative.detach());info['negative_exposures']+=1
                require(all(v.grad is None or torch.isfinite(v.grad).all().item() for n,v in parameters),'Non-finite gradient')
                torch.nn.utils.clip_grad_norm_([v for n,v in parameters],10.);optimizer.step()
                info['updates']=step;info['positive_exposures']+=1
                if step%11==0:
                    model.eval()
                    with torch.inference_mode():
                        validation=[float(model(batches[r['image_id']])[0].sum()) for r in p['records'] if r['partition']=='val']
                    require(all(math.isfinite(x) for x in validation),'Non-finite development loss')
                    info['history'].append({'cycle':item['cycle'],'updates':step,'development_loss':sum(validation)/3,
                                            'last_positive_loss':float(positive.detach()),'last_negative_loss':neg_value})
                    save(run/'run_state.json',state);print(arm,'updates',step,'dev_loss',sum(validation)/3,flush=True)
            actual=model.state_dict();require(all(torch.equal(v,actual[n].detach().cpu()) for n,v in frozen.items()),'Frozen parameters or BN statistics changed')
            require(not torch.equal(first_initial,dict(model.named_parameters())[first_name].detach().cpu()),'No Detect update')
            path=run/f'{arm}_final.pt';wrapper.model=model.eval().cpu()
            wrapper.ckpt={'train_args':{'task':'detect','model':str(path)}};wrapper.save(path)
            restored=YOLO(str(path));require(restored.names=={0:'ball'},'Checkpoint class round trip failed')
            sample=next(r for r in p['records'] if r['partition']=='val')
            result=restored.predict(source=str(folder/sample['image']),device=0,imgsz=640,conf=.25,save=False,verbose=False)[0]
            require(torch.isfinite(result.boxes.xyxy).all().item(),'Non-finite restored prediction')
            info.update(status='completed',checkpoint={'path':str(path),'sha256':sha(path)},checkpoint_round_trip=True,
                        frozen_parameters_and_all_BN_buffers_unchanged=True,peak_GPU_GiB=torch.cuda.max_memory_allocated()/1024**3)
            save(run/'run_state.json',state)
            del optimizer,model,wrapper,restored;torch.cuda.empty_cache()
        a,b=(state['arms'][x] for x in ('control','negative_crop'))
        require(a['updates']==b['updates']==a['positive_exposures']==b['positive_exposures']==len(plan),'Unequal update/positive budget')
        require(a['negative_exposures']==0 and b['negative_exposures']==len(plan),'Wrong negative exposure')
        state.update(status='completed',budgets_matched=True,primary_endpoint='final_fixed_update',early_stopping_used=False,
                     generalization_metrics_valid=False,original_classification_validation_test_used=False)
        save(run/'run_state.json',state);check(folder)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());save(run/'run_state.json',state);raise
    print('COMPLETED:',run,flush=True);return run


def evaluate(folder,run,reference):
    folder,run,reference=map(lambda x:Path(x).resolve(),(folder,run,reference));p=check(folder)
    require(run.is_relative_to(folder) and reference.is_relative_to(ARTIFACTS.resolve()),'Foreign run/reference')
    state=read(run/'run_state.json');require(state['status']=='completed' and not state['smoke'] and state['budgets_matched'],'Full matched run required')
    cfg=read(reference/'config.json');inputs=cfg['inputs']
    require(len(inputs)==53 and cfg['parameters']=={'imgsz':640,'conf':.25,'iou':.7,'max_det':100,'half':False},'Wrong fixed diagnostic inputs')
    require({i['item_id'] for i in inputs if i['ground_truth'] is not None}=={r['image_id'] for r in p['records'] if r['partition']=='val'},'Development allowlist changed')
    used={r['source_group'] for r in p['records']}
    require(all(i['source_group'] not in used for i in inputs if i['ground_truth'] is None),'Diagnostic source overlaps actual training/development')
    require(torch.cuda.is_available(),'CUDA required')
    output=Path(tempfile.mkdtemp(prefix='fixed53_comparison_',dir=run));(output/'frames').mkdir();(output/'overview').mkdir();runtime_at(output)
    shutil.copy2(__file__,output/'source_snapshot.py')
    from ultralytics import YOLO
    from collections import Counter
    torch.set_num_threads(4);cv2.setNumThreads(1)
    models={};protected={str(run/'run_state.json'):sha(run/'run_state.json'),str(reference/'config.json'):sha(reference/'config.json')}
    for arm,data in state['arms'].items():
        path=data['checkpoint']['path'];require(sha(path)==data['checkpoint']['sha256'],'Checkpoint changed')
        models[arm]=YOLO(path);require(models[arm].names=={0:'ball'},'Wrong semantic class');protected[path]=sha(path)
    totals={arm:Counter() for arm in models};records=[];panels=[];cards=[]
    for item in inputs:
        require(sha(item['path'])==item['image_sha256'],'Input PNG changed');protected[item['path']]=item['image_sha256']
        raw=cv2.imread(item['path']);require(raw is not None and raw.shape==(240,320,3),'Wrong original pixels')
        gt=None if item['ground_truth'] is None else [a for a in item['ground_truth'] if a['class_id']==1]
        predictions={}
        for arm,model in models.items():
            result=model.predict(source=raw,classes=[0],device=0,save=False,verbose=False,**cfg['parameters'])[0]
            predictions[arm]=map_ball_predictions(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist(),0,arm)
        row={**deepcopy(item),'ground_truth':gt,'predictions':predictions}
        if gt is not None:
            row['matching']={arm:count_matches(gt,boxes)['ball'] for arm,boxes in predictions.items()}
            for arm,v in row['matching'].items():totals[arm].update(v)
        panel=np.zeros((514,1920,3),np.uint8);panel[34:,:640]=labeled(raw,gt,True) if gt is not None else cv2.resize(raw,(640,480),interpolation=cv2.INTER_NEAREST)
        for x,arm in ((640,'control'),(1280,'negative_crop')):panel[34:,x:x+640]=labeled(raw,predictions[arm])
        for x,txt in ((4,item['item_id']+(' HUMAN BALL DEV' if gt is not None else ' NO GT')),(644,'CONTROL: POSITIVE ONLY / FINAL330'),(1284,'POSITIVE + 0.25 NEGATIVE / FINAL330')):
            cv2.putText(panel,txt,(x,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        filename=f'frames/{item["item_id"]}.png';require(cv2.imwrite(str(output/filename),panel),'Panel write failed');row['panel']=filename;records.append(row)
        if gt is not None or any(predictions.values()):
            panels.append(panel);cards.append(f'<section><h2>{item["item_id"]}</h2><img src="{filename}"></section>')
    pages=[]
    for start in range(0,len(panels),4):
        name=f'overview/page_{start//4+1}.png';require(cv2.imwrite(str(output/name),np.vstack([cv2.resize(x,(1280,343)) for x in panels[start:start+4]])),'Overview failed');pages.append(name)
    presence={arm:{'boxes':sum(len(r['predictions'][arm]) for r in records if r['ground_truth'] is None),
                   'frames_with_detection':sum(bool(r['predictions'][arm]) for r in records if r['ground_truth'] is None)} for arm in models}
    report={'schema':SCHEMA,'status':'completed','images':records,'overview':pages,'parameters':cfg['parameters'],
            'development_fixed_matching':{arm:dict(v) for arm,v in totals.items()},'unlabeled_presence_NOT_precision_or_recall':presence,
            'primary_endpoint':'both_final_update330','positive_exposures_each':330,'updates_each':330,
            'protected_input_sha256':protected,'AI_review_status':'pending','generalization_metrics_valid':False,
            'new_human_labels_created':False,'feature_cache_changed':False,'training_started_by_evaluation':False}
    save(output/'report.json',report)
    text='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>固定更新數：負例對照</title><style>body{font:18px system-ui;line-height:1.8;max-width:1500px;margin:30px auto;padding:0 20px}img{max-width:100%}section{margin:20px 0}</style><h1>固定330次更新、相同330次全圖正例：負例增強對照</h1><p>左：原圖／3張人工開發球框；中：只有正例；右：同一正例加0.25倍衍生負例loss。相同初始化、固定所有BN統計、同一正例順序、AdamW與固定learning rate；沒有early stopping、EMA或best epoch選擇，主要比較兩組最終權重。額外負例有額外計算量，但不增加optimizer更新數或減少正例曝光。</p><p>固定相同53張全圖與640/conf0.25。3張可作IoU0.5匹配；其他50張沒有完整真值，預測數量不是precision/recall。這是單一seed、極小且已反覆檢視的開發診斷，不是正式泛化。人工標註、分類validation/test與MixNet特徵不改。</p><p><a href="report.json">全部預測與統計</a> · <a href="../run_state.json">固定預算訓練紀錄</a> · <a href="AI_review.html">AI初審（若有）</a></p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(text,encoding='utf-8')
    require(all(sha(path)==h for path,h in protected.items()),'Evaluation changed protected inputs');check(folder)
    print({'output':str(output),'matching':report['development_fixed_matching'],'presence':presence},flush=True);return output


def training_check(folder,run):
    """Resubstitution diagnoses memorization only, never estimates generalization."""
    folder,run=Path(folder).resolve(),Path(run).resolve();p=check(folder)
    require(run.is_relative_to(folder),'Foreign training run');state=read(run/'run_state.json')
    require(state['status']=='completed' and not state['smoke'],'Full pair required')
    require(torch.cuda.is_available(),'CUDA required')
    output=Path(tempfile.mkdtemp(prefix='training_resubstitution_',dir=run));runtime_at(output)
    from ultralytics import YOLO
    from collections import Counter
    torch.set_num_threads(4);totals={};records={};protected={}
    for arm,data in state['arms'].items():
        path=data['checkpoint']['path'];require(sha(path)==data['checkpoint']['sha256'],'Weight changed')
        protected[path]=sha(path);model=YOLO(path);require(model.names=={0:'ball'},'Wrong class')
        total=Counter();rows=[]
        for r in p['records']:
            if r['partition']!='train':continue
            image=folder/r['image'];require(sha(image)==r['image_sha256'],'Training PNG changed');protected[str(image)]=sha(image)
            raw=cv2.imread(str(image));require(raw is not None and raw.shape==(240,320,3),'Wrong full training image')
            result=model.predict(source=raw,classes=[0],device=0,imgsz=640,conf=.25,iou=.7,max_det=100,half=False,save=False,verbose=False)[0]
            pred=map_ball_predictions(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist(),0,arm)
            gt=[{'class_id':1,'xyxy':box} for box in r['ball_boxes']];match=count_matches(gt,pred)['ball'];total.update(match)
            rows.append({'image_id':r['image_id'],'ground_truth':gt,'predictions':pred,'matching':match})
        totals[arm]=dict(total);records[arm]=rows
    report={'scope':'11 original full training positives; resubstitution not independent evaluation',
            'training_started':False,'matching':totals,'images':records,'generalization_metrics_valid':False,
            'protected_input_sha256':protected,'parameters':{'imgsz':640,'conf':.25,'iou':.7,'max_det':100,'half':False}}
    save(output/'report.json',report);require(all(sha(path)==h for path,h in protected.items()),'Protected inputs changed');check(folder)
    print({'output':str(output),'training_matching_not_generalization':totals},flush=True);return output


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    for x in ('prepare','check','smoke','train','evaluate','training-check'):mode.add_argument('--'+x,action='store_true')
    p.add_argument('--candidate-dir');p.add_argument('--experiment-dir');p.add_argument('--run-dir');p.add_argument('--reference');a=p.parse_args()
    if a.prepare:print(prepare(a.candidate_dir))
    elif a.check:print({'updates_per_arm':check(a.experiment_dir)['updates_per_arm']})
    elif a.train or a.smoke:print(run_pair(a.experiment_dir,smoke=a.smoke))
    elif a.training_check:print(training_check(a.experiment_dir,a.run_dir))
    else:print(evaluate(a.experiment_dir,a.run_dir,a.reference))


if __name__=='__main__':main()
