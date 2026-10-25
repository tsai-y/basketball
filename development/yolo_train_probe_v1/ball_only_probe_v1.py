"""Ball-only semantic COCO warm start with frozen backbone: tiny engineering branch."""
from copy import deepcopy
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import os
import re
import shutil
import tempfile
import traceback

import cv2
import numpy as np
import yaml

from .probe import ARTIFACTS,read,save,sha,require,within,xyxy_to_yolo
from .train_expansion_v1 import check_dataset as check_source,training_args,new_training_epoch
from .ball_pretraining_diagnostic_v1 import ball_class_id,map_ball_predictions
from .visual_check_expansion_v1 import count_matches,labeled

SCHEMA='ball_only_semantic_warmstart_probe_v1'


def ball_annotations(annotations):
    """Only inherit accepted original ball boxes; person/rim are not new negatives."""
    balls=[{**deepcopy(a),'class_id':0,'source_class_id':1} for a in annotations if a['class_id']==1]
    require(balls,'This pilot exports only existing positive, human-complete ball images')
    return balls


def transfer_ball_state(source,target,source_class_id):
    import torch
    require(type(source_class_id) is int and source_class_id>=0,'Invalid semantic class')
    mapped={};rows=[];sliced=[]
    for key,value in target.items():
        require(key in source,'Missing pretrained tensor '+key)
        old=source[key]
        if tuple(old.shape)==tuple(value.shape):
            mapped[key]=old.detach().clone();method='exact_shared_tensor'
        else:
            require(re.fullmatch(r'model\.\d+\.cv3\.[0-2]\.2\.(weight|bias)',key) is not None,
                    'Unexpected non-classifier mismatch: '+key)
            require(value.shape[0]==1 and old.shape[0]>source_class_id and
                    tuple(old.shape[1:])==tuple(value.shape[1:]),'Invalid semantic output mapping')
            mapped[key]=old[source_class_id:source_class_id+1].detach().clone()
            method='COCO_sports_ball_row_to_single_ball';sliced.append(key)
        require(tuple(mapped[key].shape)==tuple(value.shape),'Mapped shape mismatch')
        rows.append({'key':key,'source_shape':list(old.shape),'target_shape':list(value.shape),'method':method})
    require(len(sliced)==6,'Expected exactly three classifier weight/bias pairs')
    require(all(torch.isfinite(v).all().item() for v in mapped.values()),'Non-finite mapped weight')
    return mapped,{'tensor_count':len(rows),'exact_shared_count':len(rows)-len(sliced),
        'semantic_row_count':len(sliced),'source_class_id':source_class_id,'target_class_id':0,'tensors':rows}


def clone_ball_model(original,source_class_id):
    """Preserve checkpoint operations/attributes, not merely YAML tensor shapes."""
    import torch
    from torch import nn
    target=deepcopy(original).float().eval();head=target.model[-1]
    require(not getattr(head,'end2end',False),'Only the existing non-end2end head is supported')
    for branch in head.cv3:
        old=branch[-1]
        require(isinstance(old,nn.Conv2d) and old.groups==1 and old.out_channels>source_class_id,
                'Unexpected class-output layer')
        branch[-1]=nn.Conv2d(old.in_channels,1,old.kernel_size,old.stride,old.padding,
            old.dilation,old.groups,old.bias is not None,old.padding_mode).to(device=old.weight.device,dtype=old.weight.dtype)
    head.nc=1;head.no=head.reg_max*4+1;head.shape=None
    target.yaml=deepcopy(original.yaml);target.yaml['nc']=1;target.names={0:'ball'}
    mapped,audit=transfer_ball_state(original.state_dict(),target.state_dict(),source_class_id)
    target.load_state_dict(mapped,strict=True)
    audit['architecture_policy']='Deep-copy checkpoint modules and attributes; replace only final class-output Conv2d layers'
    return target,audit


def runtime_at(folder):
    runtime=Path(folder)/'runtime';runtime.mkdir(exist_ok=True)
    os.environ.update(YOLO_CONFIG_DIR=str(runtime),MPLCONFIGDIR=str(runtime/'matplotlib'),
        YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',HF_HUB_OFFLINE='1',
        WANDB_MODE='disabled',COMET_MODE='DISABLED',CLEARML_OFFLINE_MODE='1')
    font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    require(font.is_file(),'Local font missing');shutil.copy2(font,runtime/'Arial.ttf')
    from ultralytics.utils import USER_CONFIG_DIR,SETTINGS
    Path(USER_CONFIG_DIR).mkdir(parents=True,exist_ok=True);shutil.copy2(font,Path(USER_CONFIG_DIR)/'Arial.ttf')
    SETTINGS.update({k:False for k in ('sync','hub','wandb','mlflow','clearml','comet','neptune','tensorboard','dvc','raytune') if k in SETTINGS})


def prepare(source_dataset):
    source_dataset=Path(source_dataset).resolve();m=check_source(source_dataset)
    require(m['image_counts']=={'train':11,'val':3},'Frozen source counts changed')
    output=Path(tempfile.mkdtemp(prefix='ball_only_probe_',dir=ARTIFACTS))
    for sub in ('images/train','images/val','labels/train','labels/val','inputs'):(output/sub).mkdir(parents=True)
    runtime_at(output)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    weights=Path(m['weights']);require(sha(weights)==m['weights_sha256'],'Original COCO weights changed')
    wrapper=YOLO(str(weights));original=wrapper.model.float().eval()
    cid=ball_class_id(wrapper.names,'sports ball')
    target,audit=clone_ball_model(original,cid)
    # Compare raw decoded box and ball-score tensors, before class argmax/NMS.
    x=torch.linspace(0,1,3*64*64).reshape(1,3,64,64)
    with torch.inference_mode():
        old=original(x)[0];new=target(x)[0]
    require(torch.allclose(old[:,:4],new[:,:4],atol=1e-4,rtol=1e-5),'Transferred box outputs differ')
    require(torch.allclose(old[:,4+cid:5+cid],new[:,4:5],atol=1e-6,rtol=1e-5),'COCO ball score not preserved')
    initializer=output/'coco_ball_preserved.pt'
    wrapper.model=target;wrapper.ckpt={'train_args':{'task':'detect','model':str(initializer)}}
    wrapper.save(initializer)
    restored=YOLO(str(initializer)).model.float().eval()
    with torch.inference_mode():reloaded=restored(x)[0]
    require(torch.allclose(new,reloaded,atol=1e-4,rtol=1e-5),'Initializer checkpoint round trip failed')
    audit.update(raw_box_outputs_preserved=True,raw_sports_ball_scores_preserved=True,
        maximum_raw_box_absolute_difference=float((old[:,:4]-new[:,:4]).abs().max()),
        maximum_raw_ball_score_absolute_difference=float((old[:,4+cid:5+cid]-new[:,4:5]).abs().max()),
        test_input_shape=list(x.shape),initializer_round_trip=True,classification_competition_changed_by_nc1=True,
        raw_score_equivalence_does_not_guarantee_identical_NMS_detections=True,
        original_weights_sha256=sha(weights),initializer_sha256=sha(initializer))
    save(output/'initialization_audit.json',audit);save(output/'inputs/source_dataset.json',m)
    files={};records=[]
    protected=dict(m['protected_source_sha256'])
    protected[str(source_dataset/'dataset_manifest.json')]=sha(source_dataset/'dataset_manifest.json')
    for r in m['records']:
        source=within(source_dataset,r['image']);protected[str(source)]=sha(source)
        balls=ball_annotations(r['annotations']);require(len(balls)==1,'Expected one accepted ball per fixed frame')
        image=output/f'images/{r["partition"]}/{r["image_id"]}.png'
        label=output/f'labels/{r["partition"]}/{r["image_id"]}.txt'
        shutil.copy2(source,image)
        label.write_text('\n'.join(xyxy_to_yolo(0,a['xyxy']) for a in balls)+'\n',encoding='utf-8')
        require(sha(image)==r['image_sha256'],'Native copy changed')
        records.append({**deepcopy(r),'image':str(image.relative_to(output)),'label':str(label.relative_to(output)),
            'annotations':balls,'original_human_accepted_annotations':deepcopy(r['annotations'])})
        for p in (image,label):files[str(p.relative_to(output))]=sha(p)
    data={'path':str(output),'train':'images/train','val':'images/val','names':{0:'ball'}}
    (output/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    for p in [output/'data.yaml',output/'initialization_audit.json',initializer,output/'inputs/source_dataset.json']:
        files[str(p.relative_to(output))]=sha(p)
    report={'schema':SCHEMA,'source_dataset':str(source_dataset),'records':records,
        'image_counts':{'train':11,'val':3},'ball_counts':{'train':11,'val':3},
        'source_assignments':m['source_assignments'],'source_split_unchanged':True,
        'initializer':str(initializer),'initializer_sha256':sha(initializer),
        'dataset_file_sha256':files,'protected_source_sha256':protected,
        'new_independent_data_added':False,'source_annotations_modified':False,'AI_only_labels_used':False,
        'original_classification_validation_test_used':False,'generalization_metrics_valid':False,
        'backbone_policy':'Freeze all layers before Detect, including BatchNorm running buffers; train detection head only',
        'scope':'Isolated ball-only model adaptation diagnosis, does not replace existing3-class model',
        'limitation':'Only11 train balls and3 development balls from2 dev sources; development re-use and model-selection bias',
        'created_at_utc':datetime.now(timezone.utc).isoformat()}
    save(output/'dataset_manifest.json',report);check(output);return output


def check(output):
    output=Path(output).resolve();require(output.is_relative_to(ARTIFACTS.resolve()),'Foreign artifact')
    m=read(output/'dataset_manifest.json');require(m['schema']==SCHEMA,'Wrong schema')
    for p,h in m['dataset_file_sha256'].items():require(sha(within(output,p))==h,'Ball dataset changed: '+p)
    for p,h in m['protected_source_sha256'].items():require(sha(p)==h,'Original source changed: '+p)
    require(yaml.safe_load((output/'data.yaml').read_text())=={'path':str(output),
        'train':'images/train','val':'images/val','names':{0:'ball'}},'Wrong ball dataset config')
    for split in ('train','val'):
        rows=[r for r in m['records'] if r['partition']==split];ids={r['image_id'] for r in rows}
        require(len(ids)==m['image_counts'][split],'Wrong split count')
        require({p.name for p in (output/f'images/{split}').iterdir()}=={i+'.png' for i in ids},'Unexpected images')
        require({p.name for p in (output/f'labels/{split}').iterdir()}=={i+'.txt' for i in ids},'Unexpected labels')
        for r in rows:
            require(r['source_split']=='train','Non-original-train source')
            expected='detector_train' if split=='train' else 'detector_dev_validation'
            require(m['source_assignments'][r['source_group']]==expected,'Source reservation violated')
            require(r['annotations']==ball_annotations(r['original_human_accepted_annotations']),'Unapproved ball geometry')
            require((output/r['label']).read_text().splitlines()==[xyxy_to_yolo(0,a['xyxy']) for a in r['annotations']], 'Ball label mismatch')
    for field in ('source_group','video_id','image_sha256'):
        groups={s:{r[field] for r in m['records'] if r['partition']==s} for s in ('train','val')}
        require(groups['train'].isdisjoint(groups['val']),'Train/dev overlap')
    return m


def freeze_backbone_eval(model,head_index):
    for layer in model.model[:head_index]:layer.eval()


def train(output,*,epochs=100,smoke=False):
    output=Path(output).resolve();m=check(output);training_args(output,output/'unused',epochs,smoke)
    import torch
    require(torch.cuda.is_available(),'CUDA required')
    folder=Path(tempfile.mkdtemp(prefix='smoke_' if smoke else 'headtrain_',dir=output));runtime_at(folder)
    shutil.copy2(__file__,folder/'training_source_snapshot.py')
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    import ultralytics
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    model=YOLO(m['initializer']);head_index=len(model.model.model)-1
    class PreservedCheckpointTrainer(DetectionTrainer):
        def get_model(self,cfg=None,weights=None,verbose=True):
            require(weights is not None,'Checkpoint-preserving trainer requires initializer module')
            result=deepcopy(weights).float()
            require(result.model[-1].nc==1 and result.names=={0:'ball'},'Wrong preserved initializer')
            return result
        def _model_train(self):
            # The base trainer calls this AFTER on_train_epoch_start callbacks.
            super()._model_train()
            freeze_backbone_eval(self.model,head_index)
    args=training_args(output,folder,epochs,smoke);args['freeze']=head_index
    log={'schema':SCHEMA,'status':'starting','stage':'smoke' if smoke else 'ball_only_head_training',
        'dataset':str(output),'training_args':args,'head_index':head_index,'epochs_completed':0,
        'initializer_sha256':m['initializer_sha256'],'source_code_sha256':sha(__file__),
        'torch':torch.__version__,'ultralytics':ultralytics.__version__,'GPU':torch.cuda.get_device_name(0),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'generalization_metrics_valid':False}
    frozen={};initial={};observed=set();prefix=f'model.{head_index}.'
    def start(trainer):
        require(trainer.data['names']=={0:'ball'},'Wrong one-class mapping')
        for split,loader in [('train',trainer.train_loader),('val',trainer.test_loader)]:
            expected={str((output/r['image']).resolve()) for r in m['records'] if r['partition']==split}
            require({str(Path(p).resolve()) for p in loader.dataset.im_files}==expected,'Loader escaped source split')
        params=[(n,p) for n,p in trainer.model.named_parameters() if p.requires_grad]
        require(params and all(n.startswith(prefix) for n,p in params),'Backbone parameter unexpectedly trainable')
        freeze_backbone_eval(trainer.model,head_index)
        frozen.update({n:v.detach().cpu().clone() for n,v in trainer.model.state_dict().items() if not n.startswith(prefix)})
        for n,p in (params[0],params[-1]):initial[n]=p.detach().cpu().clone()
        log.update(status='training',actual_loader_allowlists_checked=True,
            trainable_parameters=sum(p.numel() for n,p in params),backbone_bn_eval_enforced=True)
        save(folder/'run_state.json',log)
    def epoch_start(trainer):freeze_backbone_eval(trainer.model,head_index)
    def batch_end(trainer):
        require(torch.isfinite(trainer.loss_items).all().item(),'Non-finite ball loss')
        require(all(not layer.training for layer in trainer.model.model[:head_index]),'Frozen backbone entered train mode')
    def epoch_trained(trainer):observed.add(trainer.epoch)
    def fit_end(trainer):
        if not new_training_epoch(trainer.epoch,log['epochs_completed'],observed):return
        p=dict(trainer.model.named_parameters())
        log.update(epochs_completed=trainer.epoch+1,training_losses=[float(x) for x in trainer.tloss.detach().cpu()],
            head_parameter_max_abs_updates={n:float((p[n].detach().cpu()-v).abs().max()) for n,v in initial.items()},
            development_metrics={k:float(v) for k,v in trainer.metrics.items()},
            peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
        save(folder/'run_state.json',log);print('BALL_EPOCH:',trainer.epoch+1,log['training_losses'],flush=True)
    for key,callback in [('on_train_start',start),('on_train_epoch_start',epoch_start),
        ('on_train_batch_end',batch_end),('on_train_epoch_end',epoch_trained),('on_fit_epoch_end',fit_end)]:model.add_callback(key,callback)
    save(folder/'run_state.json',log);print('RUN_DIR:',folder,flush=True)
    try:
        model.train(trainer=PreservedCheckpointTrainer,**args)
        actual=model.trainer.model.state_dict()
        require(frozen and all(torch.equal(v,actual[n].detach().cpu()) for n,v in frozen.items()),'Frozen backbone parameters/buffers changed')
        require(any(x>0 for x in log['head_parameter_max_abs_updates'].values()),'No head update')
        checkpoints={}
        for name in ('best','last'):
            path=folder/f'fit/weights/{name}.pt';require(path.is_file(),'Missing checkpoint')
            loaded=YOLO(str(path));require(loaded.names=={0:'ball'},'Checkpoint class map wrong')
            results=loaded.predict(source=[str(output/r['image']) for r in m['records'] if r['partition']=='val'],
                device=0,imgsz=640,conf=.25,save=False,verbose=False)
            require(len(results)==3 and all(torch.isfinite(x.boxes.xyxy).all().item() for x in results),'Checkpoint reload failed')
            checkpoints[name]={'path':str(path),'sha256':sha(path)}
        log.update(status='completed',checkpoint_round_trip=True,checkpoints=checkpoints,
            backbone_parameters_and_BN_buffers_unchanged=True,
            stop_reason='early_stopping' if log['epochs_completed']<epochs else 'epoch_limit',
            selected_development_metrics={k:float(v) for k,v in model.trainer.metrics.items()},
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        save(folder/'run_state.json',log)
    except BaseException as exc:
        log.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());save(folder/'run_state.json',log);raise
    finally:check(output)
    print('COMPLETED:',folder,flush=True);return folder


def evaluate(output,training_run,reference):
    output,training_run,reference=map(lambda p:Path(p).resolve(),(output,training_run,reference))
    m=check(output);require(training_run.is_relative_to(output),'Foreign ball training run')
    state=read(training_run/'run_state.json');require(state['status']=='completed' and state['checkpoint_round_trip'],'Training incomplete')
    require(reference.is_relative_to(ARTIFACTS.resolve()),'Foreign reference')
    cfg=read(reference/'config.json');inputs=cfg['inputs'];require(len(inputs)==53,'Fixed53 inputs required')
    require(cfg['parameters']=={'imgsz':640,'conf':.25,'iou':.7,'max_det':100,'half':False},'Reference parameters changed')
    dev={r['image_id'] for r in m['records'] if r['partition']=='val'}
    require({r['item_id'] for r in inputs if r['ground_truth'] is not None}==dev,'Development allowlist changed')
    used_sources={r['source_group'] for r in m['records']}
    require(all(r['source_group'] not in used_sources for r in inputs if r['ground_truth'] is None),
            'Extra source overlaps actually used train/development source')
    import torch
    require(torch.cuda.is_available(),'CUDA required')
    folder=Path(tempfile.mkdtemp(prefix='before_after_',dir=training_run));(folder/'frames').mkdir();runtime_at(folder)
    shutil.copy2(__file__,folder/'evaluation_source_snapshot.py')
    from ultralytics import YOLO
    torch.set_num_threads(4);cv2.setNumThreads(1)
    paths={'before':Path(m['initializer']),'after':Path(state['checkpoints']['best']['path'])}
    require(sha(paths['after'])==state['checkpoints']['best']['sha256'],'Trained checkpoint changed')
    models={n:YOLO(str(p)) for n,p in paths.items()};require(all(model.names=={0:'ball'} for model in models.values()),'Wrong ball map')
    protected={str(p):sha(p) for p in paths.values()};protected.update({x['path']:x['image_sha256'] for x in inputs})
    totals={n:Counter() for n in models};records=[]
    save(folder/'run_state.json',{'status':'running'});print('OUTPUT:',folder,flush=True)
    for item in inputs:
        require(sha(item['path'])==item['image_sha256'],'Input changed')
        raw=cv2.imread(item['path']);require(raw is not None and raw.shape==(240,320,3),'Wrong native input')
        gt=None if item['ground_truth'] is None else [a for a in item['ground_truth'] if a['class_id']==1]
        pred={}
        for name,model in models.items():
            result=model.predict(source=raw,classes=[0],device=0,imgsz=640,conf=.25,iou=.7,max_det=100,half=False,save=False,verbose=False)[0]
            pred[name]=map_ball_predictions(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist(),0,name)
        panel=np.zeros((514,1920,3),dtype=np.uint8)
        panel[34:,:640]=labeled(raw,gt,True) if gt is not None else cv2.resize(raw,(640,480),interpolation=cv2.INTER_NEAREST)
        panel[34:,640:1280]=labeled(raw,pred['before']);panel[34:,1280:]=labeled(raw,pred['after'])
        for x,text in [(4,item['item_id']+(' BALL HUMAN / DEV' if gt is not None else ' ORIGINAL / NO GT')),
            (644,'BEFORE: COCO BALL ROW PRESERVED'),(1284,'AFTER: FROZEN BACKBONE HEAD FINETUNE')]:
            cv2.putText(panel,text,(x,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        rel=f'frames/{item["item_id"]}.png';require(cv2.imwrite(str(folder/rel),panel),'Panel write failed')
        row={**item,'ground_truth':gt,'predictions':pred,'panel':rel}
        if gt is not None:
            row['matching']={n:count_matches(gt,p)['ball'] for n,p in pred.items()}
            for n,count in row['matching'].items():totals[n].update(count)
        records.append(row)
    presence={n:{'boxes':sum(len(x['predictions'][n]) for x in records if x['ground_truth'] is None),
        'frames_with_detection':sum(bool(x['predictions'][n]) for x in records if x['ground_truth'] is None)} for n in models}
    report={'schema':SCHEMA,'status':'completed','images':records,'images_checked':53,
        'development_matching_conf025_iou05':{n:dict(v) for n,v in totals.items()},
        'unlabeled_presence_NOT_recall':presence,'same_model_architecture_before_after':True,
        'generalization_metrics_valid':False,'new_training_started_by_evaluation':False,'source_annotations_modified':False,
        'parameters':{'imgsz':640,'conf':.25,'iou':.7,'max_det':100,'half':False},
        'protected_input_sha256':protected,'AI_review_status':'pending'}
    save(folder/'report.json',report)
    cards=[]
    for row in records:
        if row['ground_truth'] is None and not any(row['predictions'][n] for n in models):continue
        cards.append(f'<section><h2>{row["item_id"]}</h2><img loading="lazy" src="{row["panel"]}"></section>')
    (folder/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>球專用分支：訓練前後</title><style>body{font:18px system-ui;line-height:1.8;max-width:1600px;margin:25px auto;padding:0 18px}img{max-width:100%}section{margin:20px 0}</style><h1>球專用分支：保留COCO球分類輸出、凍結骨幹</h1><p>左人工球框／原圖，中訓練前單類初始化，右只訓練偵測頭後。兩者同單類架構，固定640/conf0.25/IoU0.7；3張開發球可固定IoU0.5匹配，其他50張沒有完整真值。有框不是recall，分數不是正式泛化。</p><p>沿用11個既有訓練球框，並未增加獨立資料；骨幹參數及BN緩衝需保持不變。原三類YOLO、標註與MixNet特徵不改。<a href="report.json">全部預測</a> · <a href="AI_review.html">AI初審（若有）</a></p>'''+''.join(cards),encoding='utf-8')
    require(all(sha(p)==h for p,h in protected.items()),'Protected inputs changed');check(output)
    save(folder/'run_state.json',{'status':'completed','images_completed':53})
    print({'output':str(folder),'development_matching':report['development_matching_conf025_iou05'],'presence':presence},flush=True)
    return folder


def review_crops(folder):
    """Derived display-only native context crops; never model inputs or labels."""
    import math
    folder=Path(folder).resolve();require(folder.is_relative_to(ARTIFACTS.resolve()),'Foreign comparison')
    report=read(folder/'report.json');require(report['status']=='completed','Incomplete comparison')
    panels=[];metadata=[]
    for row in report['images']:
        for i,pred in enumerate(row['predictions']['after'],1):
            raw=cv2.imread(row['path']);require(raw is not None and sha(row['path'])==row['image_sha256'],'Source changed')
            x1,y1,x2,y2=pred['xyxy'];cx,cy=(x1+x2)/2,(y1+y2)/2
            left,top=max(0,math.floor(cx)-24),max(0,math.floor(cy)-24)
            right,bottom=min(320,math.ceil(cx)+24),min(240,math.ceil(cy)+24)
            image=cv2.resize(raw[top:bottom,left:right],(294,294),interpolation=cv2.INTER_NEAREST)
            tile=np.zeros((340,320,3),dtype=np.uint8);tile[42:336,13:307]=image
            cv2.putText(tile,f'{row["item_id"]} B{i} {pred["confidence"]:.3f}',(4,17),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
            cv2.putText(tile,f'box {x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}',(4,35),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1)
            panels.append(tile);metadata.append({'item_id':row['item_id'],'box_index':i,'xyxy':pred['xyxy'],
                'crop_xyxy':[left,top,right,bottom],'image_sha256':row['image_sha256']})
    require(panels,'No candidate crops')
    pages=[]
    for start in range(0,len(panels),6):
        tiles=panels[start:start+6]
        while len(tiles)<6:tiles.append(np.zeros((340,320,3),dtype=np.uint8))
        filename=f'candidate_native_context_{start//6+1}.png'
        require(cv2.imwrite(str(folder/filename),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])])),'Crop write failed')
        pages.append(filename)
    save(folder/'candidate_native_context.json',{'display_only':True,'inference_input_changed':False,'pages':pages,'crops':metadata})
    return pages


def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','check','train','evaluate'):m.add_argument('--'+name,action='store_true')
    p.add_argument('--source-dataset');p.add_argument('--dataset-dir');p.add_argument('--training-run');p.add_argument('--reference')
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    if a.prepare:print(prepare(a.source_dataset))
    elif a.check:print(check(a.dataset_dir)['ball_counts'])
    elif a.train:print(train(a.dataset_dir,epochs=a.epochs,smoke=a.smoke))
    else:print(evaluate(a.dataset_dir,a.training_run,a.reference))


if __name__=='__main__':main()
