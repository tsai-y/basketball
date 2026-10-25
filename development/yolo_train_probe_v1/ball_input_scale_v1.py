"""Paired640/1280 ball inference; same frozen53 inputs/weights/confidence, no training."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import html
import os
import tempfile

import cv2
import numpy as np

from .probe import CLASS_NAMES, read,save,require,sha
from .train_expansion_v1 import check_dataset
from .visual_check_expansion_v1 import labeled,count_matches
from .ball_pretraining_diagnostic_v1 import ball_class_id,map_ball_predictions


def parameters_for_scale(reference, size):
    require(type(size) is int and size in (640,1280), 'Only preregistered640/1280 scales allowed')
    require(reference=={'imgsz':640,'conf':.25,'iou':.7,'max_det':100,'half':False}, 'Frozen reference parameters changed')
    return {**reference,'imgsz':size}


def same_predictions(a,b):
    """Order-independent repeatability check for frozen640 control predictions."""
    if len(a)!=len(b):return False
    key=lambda x:tuple(x['xyxy'])+(x['confidence'],)
    return all(all(abs(x-y)<1e-4 for x,y in zip(p['xyxy'],q['xyxy'])) and
               abs(p['confidence']-q['confidence'])<1e-5
               for p,q in zip(sorted(a,key=key),sorted(b,key=key)))


def execute(dataset,run,reference,*,dry_run=False):
    dataset,run,reference=map(lambda x:Path(x).resolve(),(dataset,run,reference))
    require(run.is_relative_to(dataset) and reference.is_relative_to(run),'Foreign run/reference')
    m=check_dataset(dataset);config=read(reference/'config.json');previous=read(reference/'report.json')
    require(previous['status']=='completed' and previous['images_checked']==53,'Completed53-image reference required')
    inputs=config['inputs'];require(len(inputs)==53 and sum(x['ground_truth'] is not None for x in inputs)==3,'Wrong inputs')
    for size in (640,1280):parameters_for_scale(config['parameters'],size)
    for p,h in config['protected_input_sha256'].items():require(sha(p)==h,'Reference source changed')
    weights={n:Path(config[n+'_weights']).resolve() for n in ('coco','custom')}
    require(sha(weights['coco'])==m['weights_sha256'],'COCO initializer changed')
    state=read(run/'run_state.json');require(state['status']=='completed' and
        sha(weights['custom'])==state['checkpoints']['best']['sha256'],'Custom checkpoint changed')
    if dry_run:return {'dry_run':True,'images':53,'models':2,'scales':[640,1280],
        'forward_predictions':212,'files_written':False,'training_started':False,'confidence_tuning':False}
    import torch
    require(torch.cuda.is_available(),'CUDA required; no silent CPU fallback')
    output=Path(tempfile.mkdtemp(prefix='ball_scale_check_',dir=run))
    for sub in ('frames','overview','crops','runtime'):(output/sub).mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(output/'runtime'),YOLO_AUTOINSTALL='false',
                      YOLO_OFFLINE='true',HF_HUB_OFFLINE='1')
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(4);cv2.setNumThreads(1)
    models={n:YOLO(str(p)) for n,p in weights.items()}
    ids={'coco':ball_class_id(models['coco'].names,'sports ball'),'custom':ball_class_id(models['custom'].names,'ball')}
    require(ids==config['semantic_class_ids'] and models['custom'].names==dict(enumerate(CLASS_NAMES)), 'Class mapping changed')
    # Model pre-hook records the actual network tensor after letterboxing, including padding.
    shapes={}
    def capture(name):
        def hook(module,args):
            if args and isinstance(args[0],torch.Tensor):shapes[name]=list(args[0].shape)
        return hook
    hooks=[model.model.register_forward_pre_hook(capture(n)) for n,model in models.items()]
    protected={str(reference/'config.json'):sha(reference/'config.json'),str(reference/'report.json'):sha(reference/'report.json'),
        str(dataset/'dataset_manifest.json'):sha(dataset/'dataset_manifest.json'),**{str(p):sha(p) for p in weights.values()}}
    protected.update({x['path']:x['image_sha256'] for x in inputs})
    cfg={'schema':'fixed_ball_input_scale_diagnostic_v1','inputs':inputs,
        'weights':{n:{'path':str(p),'sha256':sha(p)} for n,p in weights.items()},
        'conditions':{str(s):parameters_for_scale(config['parameters'],s) for s in (640,1280)},
        'only_manipulated_factor':'imgsz_max_input_dimension','confidence_tuning':False,
        'native_shape_hw':[240,320],'upsampling_is_not_detail_restoration':True,
        'protected_input_sha256':protected,'source_code_sha256':sha(__file__),
        'ultralytics':ultralytics.__version__,'torch':torch.__version__,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'training_started':False,'original_video_opens':0}
    save(output/'config.json',cfg);print('OUTPUT:',output,flush=True)
    totals={n:{str(s):Counter() for s in (640,1280)} for n in models};records=[];crops=[]
    save(output/'run_state.json',{'status':'running','images_completed':0,'images_total':53})
    for index,item in enumerate(inputs,1):
        require(sha(item['path'])==item['image_sha256'],'Image changed')
        raw=cv2.imread(item['path']);require(raw is not None and raw.shape==(240,320,3),'Unexpected native pixels')
        gt=None if item['ground_truth'] is None else [a for a in item['ground_truth'] if a['class_id']==1]
        predictions={n:{} for n in models};network_shapes={n:{} for n in models}
        for name,model in models.items():
            for size in (640,1280):
                result=model.predict(source=raw,classes=[ids[name]],device=0,save=False,verbose=False,
                    **parameters_for_scale(config['parameters'],size))[0]
                pred=map_ball_predictions(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),ids[name],name+'_'+str(size))
                predictions[name][str(size)]=pred;network_shapes[name][str(size)]=shapes.get(name)
                require(shapes.get(name) is not None and max(shapes[name][-2:])==size,'Actual network input size differs')
                if size==640:
                    prev=next(x for x in previous['images'] if x['item_id']==item['item_id'])
                    require(same_predictions(pred,prev['ball_predictions'][name]),'640 control not repeatable; do not attribute changes to scale')
                for j,obj in enumerate(pred,1):
                    x1,y1,x2,y2=obj['xyxy'];cx=(x1+x2)/2;cy=(y1+y2)/2
                    radius=max(20.,max(x2-x1,y2-y1)*1.5)
                    left=max(0,int(cx-radius));right=min(320,int(cx+radius+1))
                    top=max(0,int(cy-radius));bottom=min(240,int(cy+radius+1))
                    crop=raw[top:bottom,left:right]
                    factor=240/max(crop.shape[:2]);resized=cv2.resize(crop,None,fx=factor,fy=factor,interpolation=cv2.INTER_NEAREST)
                    cell=np.zeros((285,260,3),dtype=np.uint8);h,w=resized.shape[:2];cell[45:45+h,:w]=resized
                    p1=(round((x1-left)*factor),45+round((y1-top)*factor))
                    p2=(round((x2-left)*factor),45+round((y2-top)*factor))
                    cv2.rectangle(cell,p1,p2,(70,230,70),2)
                    cv2.putText(cell,item['item_id'],(3,17),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                    cv2.putText(cell,f'{name} {size} B{j} {obj["confidence"]:.2f}',(3,36),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                    rel=f'crops/{item["item_id"]}_{name}_{size}_B{j}.png';require(cv2.imwrite(str(output/rel),cell),'Crop diagnostic write failed')
                    crops.append({'image_id':item['item_id'],'model':name,'imgsz':size,'prediction_index':j-1,
                                  'prediction':obj,'path':rel,'native_context_xyxy':[left,top,right,bottom]})
        panels=[]
        for name in ('coco','custom'):
            panel=np.zeros((514,1920,3),dtype=np.uint8)
            panel[34:,:640]=labeled(raw,gt,True) if gt is not None else cv2.resize(raw,(640,480),interpolation=cv2.INTER_NEAREST)
            for x,text in [(4,item['item_id']+(' HUMAN BALL / DEV' if gt is not None else ' ORIGINAL / NO GT')),
                           (644,name.upper()+' INPUT640'),(1284,name.upper()+' INPUT1280')]:
                cv2.putText(panel,text,(x,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            panel[34:,640:1280]=labeled(raw,predictions[name]['640'])
            panel[34:,1280:]=labeled(raw,predictions[name]['1280']);panels.append(panel)
        rel=f'frames/{item["item_id"]}.png';require(cv2.imwrite(str(output/rel),np.vstack(panels)),'Panel write failed')
        record={**item,'ground_truth':gt,'ball_predictions':predictions,'actual_network_shapes':network_shapes,'panel':rel}
        if gt is not None:
            record['ball_matching']={n:{str(s):count_matches(gt,predictions[n][str(s)])['ball'] for s in (640,1280)} for n in models}
            for n in models:
                for s in ('640','1280'):totals[n][s].update(record['ball_matching'][n][s])
        records.append(record)
        save(output/'run_state.json',{'status':'running','images_completed':index,'images_total':53})
    for hook in hooks:hook.remove()
    presence={n:{s:{'boxes':sum(len(x['ball_predictions'][n][s]) for x in records if x['ground_truth'] is None),
        'frames_with_detection':sum(bool(x['ball_predictions'][n][s]) for x in records if x['ground_truth'] is None)}
        for s in ('640','1280')} for n in models}
    candidate_ids=[x['item_id'] for x in records if any(x['ball_predictions'][n][s] for n in models for s in ('640','1280'))]
    overview=[]
    for start in range(0,len(crops),12):
        cells=[cv2.imread(str(output/c['path'])) for c in crops[start:start+12]]
        while len(cells)%4:cells.append(np.zeros((285,260,3),dtype=np.uint8))
        image=np.vstack([np.hstack(cells[k:k+4]) for k in range(0,len(cells),4)])
        rel=f'overview/ball_crops_{start//12+1}.png';require(cv2.imwrite(str(output/rel),image),'Crop overview write failed');overview.append(rel)
    report={'schema':cfg['schema'],'status':'completed','images':records,'images_checked':53,
        'forward_predictions':212,'development_ball_matching_conf025_iou05':{n:{s:dict(v) for s,v in d.items()} for n,d in totals.items()},
        'unlabeled_presence_NOT_recall':presence,'ball_crop_records':crops,'crop_overview':overview,
        'candidate_image_ids':candidate_ids,'640_control_matches_previous_results':True,
        'input_ids_and_SHA_identical_to_reference':True,'weights_unchanged':True,'confidence_tuning':False,
        'matching_is_not_AP':True,'generalization_metrics_valid':False,'AI_review_status':'pending',
        'training_started':False,'annotations_modified':False,'original_video_opens':0,
        'original_classification_validation_test_used':False,'feature_cache_modified':False,
        'source_videos_cropped_or_filtered':False,'diagnostic_display_crops_only':True}
    save(output/'report.json',report)
    cards=[]
    for x in records:
        if x['ground_truth'] is None and x['item_id'] not in candidate_ids:continue
        cards.append(f'<section><h2>{html.escape(x["item_id"])}</h2><img loading="lazy" src="{x["panel"]}"></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>小球640／1280輸入尺度診斷</title><style>body{font:18px system-ui;line-height:1.8;max-width:1600px;margin:25px auto;padding:0 18px;background:#f3f5f8}img{max-width:100%}section{background:white;padding:15px;margin:20px 0}</style><h1>只改輸入尺度：640／1280</h1><p>相同53張原生320×240無損PNG、兩組固定權重，conf0.25／NMS IoU0.7／max_det100完全不變。每圖上排COCO、下排自訓版；左人工球框或原圖、中640、右1280。另記錄模型實際tensor尺寸。</p><p>1280是放大既有像素，不是取得高解析原片或超解析還原。3張開發圖僅有3個球框，固定同類IoU0.5匹配不是AP；另50張沒有完整真值，檢出數不是recall。這是開發診斷，不能宣稱泛化。</p><p>沒有重訓、改標註、調門檻或改MixNet特徵；局部裁圖只供可視化，不是模型輸入裁切或影片前處理。<a href="report.json">全部預測</a> · <a href="AI_review.html">AI初審（若已產生）</a></p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    require(all(sha(p)==h for p,h in protected.items()),'Protected source changed');check_dataset(dataset)
    save(output/'run_state.json',{'status':'completed','images_completed':53,'images_total':53})
    print({'output':str(output),'development_matching':report['development_ball_matching_conf025_iou05'],
           'unlabeled_presence':presence,'ball_crop_count':len(crops),'candidate_images':len(candidate_ids)},flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',required=True)
    p.add_argument('--training-run',required=True);p.add_argument('--reference',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();print(execute(a.dataset_dir,a.training_run,a.reference,dry_run=a.dry_run))


if __name__=='__main__':main()
