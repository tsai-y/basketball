"""Fixed-input COCO sports-ball vs custom ball diagnostic; no tuning or training."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import html
import math
import os
import re
import tempfile

import cv2
import numpy as np

from .probe import CLASS_NAMES, read, save, require, sha, within
from .train_expansion_v1 import check_dataset
from .visual_check_expansion_v1 import labeled, count_matches


def ball_class_id(names, expected):
    ids = [i for i,n in names.items() if n == expected]
    require(len(ids) == 1, 'Missing/ambiguous semantic class '+expected)
    return ids[0]


def map_ball_predictions(boxes, classes, scores, source_class_id, model_name):
    require(len(boxes) == len(classes) == len(scores), 'Prediction length mismatch')
    result=[]
    for box,cls,score in zip(boxes,classes,scores):
        require(float(cls).is_integer() and int(cls) == source_class_id, 'Unexpected semantic output class')
        require(len(box)==4 and all(math.isfinite(float(v)) for v in box), 'Non-finite geometry')
        x1,y1,x2,y2=box
        require(0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 240, 'Invalid native geometry')
        require(math.isfinite(float(score)) and .25 <= score <= 1, 'Unexpected confidence')
        result.append({'class_id':1,'class_name':'ball','source_class_id':int(cls),
            'xyxy':list(map(float,box)),'confidence':float(score),
            'origin':model_name+'_prediction_not_ground_truth'})
    return result


def state_shape_compatibility(source_shapes, target_shapes):
    rows=[]
    for key,shape in target_shapes.items():
        old=source_shapes.get(key)
        rows.append({'key':key,'source_shape':list(old) if old is not None else None,
            'target_shape':list(shape),'compatible_by_key_and_shape':old is not None and tuple(old)==tuple(shape)})
    return {'target_tensor_count':len(rows),'compatible_tensor_count':sum(r['compatible_by_key_and_shape'] for r in rows),
        'incompatible_tensors':[r for r in rows if not r['compatible_by_key_and_shape']],
        'all_tensor_shapes':rows}


def run_check(dataset, run, prior_visual, *, dry_run=False):
    dataset,run,prior_visual=map(lambda p:Path(p).resolve(),(dataset,run,prior_visual))
    require(run.is_relative_to(dataset) and prior_visual.is_relative_to(run), 'Foreign diagnostic/run')
    m=check_dataset(dataset);state=read(run/'run_state.json')
    prior_config=read(prior_visual/'config.json');prior_report=read(prior_visual/'report.json')
    require(state['status']=='completed' and state['checkpoint_round_trip'] and prior_report['status']=='completed', 'Incomplete training/diagnostic')
    inputs=prior_config['inputs']
    require(len(inputs)==53 and sum(i['ground_truth'] is not None for i in inputs)==3, 'Fixed53 inputs required')
    for item in inputs:require(sha(item['path'])==item['image_sha256'], 'Fixed input changed')
    for p,h in prior_config['protected_input_sha256'].items():require(sha(p)==h,'Prior protected input changed')
    baseline=Path(m['weights']);custom=Path(state['checkpoints']['best']['path'])
    require(sha(baseline)==m['weights_sha256'] and sha(custom)==state['checkpoints']['best']['sha256'], 'Weights changed')
    parameters=prior_config['parameters']
    require(parameters=={'imgsz':640,'conf':.25,'iou':.7,'max_det':100,'half':False}, 'Do not tune thresholds or resolution')
    if dry_run:return {'dry_run':True,'images':53,'files_written':False,'training_started':False,'threshold_tuning':False}
    import torch
    require(torch.cuda.is_available(),'CUDA required; no silent CPU fallback')
    output=Path(tempfile.mkdtemp(prefix='ball_pretraining_check_',dir=run))
    for sub in ('frames','overview','runtime'):(output/sub).mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(output/'runtime'),YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',HF_HUB_OFFLINE='1')
    from ultralytics import YOLO
    import ultralytics
    from ultralytics.nn import tasks
    from ultralytics.utils import torch_utils
    torch.set_num_threads(4);cv2.setNumThreads(1)
    models={'coco':YOLO(str(baseline)),'custom':YOLO(str(custom))}
    class_ids={'coco':ball_class_id(models['coco'].names,'sports ball'),'custom':ball_class_id(models['custom'].names,'ball')}
    require(models['custom'].names==dict(enumerate(CLASS_NAMES)), 'Wrong custom class map')
    shapes={n:{k:tuple(v.shape) for k,v in model.model.state_dict().items()} for n,model in models.items()}
    compatibility=state_shape_compatibility(shapes['coco'],shapes['custom'])
    missing=compatibility['incompatible_tensors']
    require(len(missing)==6 and all(re.fullmatch(r'model\.\d+\.cv3\.[0-2]\.2\.(weight|bias)',r['key']) for r in missing),
            'Unexpected transfer mismatch; inspect rather than claim classifier-only reset')
    compatibility.update({'scope':'Static state-key/shape comparison consistent with installed BaseModel.load intersection; not direct replay of original initialization',
        'last_class_output_only_shape_mismatch':True,'entire_detection_head_reset':False,
        'COCO_sports_ball_output_row_automatically_mapped_to_custom_ball':False,
        'not_a_failed_checkpoint_load':True,
        'causal_finetuning_regression_proven':False,
        'framework_source':{'tasks_file':tasks.__file__,'tasks_sha256':sha(tasks.__file__),
            'torch_utils_file':torch_utils.__file__,'torch_utils_sha256':sha(torch_utils.__file__)}})
    save(output/'initialization_audit.json',compatibility)
    protected={str(baseline):sha(baseline),str(custom):sha(custom),str(dataset/'dataset_manifest.json'):sha(dataset/'dataset_manifest.json'),
        str(prior_visual/'config.json'):sha(prior_visual/'config.json'),str(prior_visual/'report.json'):sha(prior_visual/'report.json')}
    protected.update({i['path']:i['image_sha256'] for i in inputs})
    config={'schema':'ball_pretraining_fixed_comparison_v1','inputs':inputs,'parameters':parameters,
        'semantic_class_ids':class_ids,'coco_class_name':'sports ball','custom_class_name':'ball',
        'coco_weights':str(baseline),'custom_weights':str(custom),'protected_input_sha256':protected,
        'training_started':False,'threshold_tuning':False,'resolution_tuning':False,'original_video_opens':0,
        'ultralytics':ultralytics.__version__,'torch':torch.__version__,'source_code_sha256':sha(__file__),
        'created_at_utc':datetime.now(timezone.utc).isoformat()}
    save(output/'config.json',config);print('OUTPUT:',output,flush=True)
    totals={n:Counter() for n in models};records=[]
    save(output/'run_state.json',{'status':'running','images_total':53,'images_completed':0})
    for index,item in enumerate(inputs,1):
        raw=cv2.imread(item['path']);require(raw is not None and raw.shape==(240,320,3),'Wrong native image')
        predictions={}
        for name,model in models.items():
            pred=model.predict(source=raw,classes=[class_ids[name]],device=0,save=False,verbose=False,**parameters)[0]
            predictions[name]=map_ball_predictions(pred.boxes.xyxy.cpu().tolist(),pred.boxes.cls.cpu().tolist(),
                pred.boxes.conf.cpu().tolist(),class_ids[name],name)
        gt=None if item['ground_truth'] is None else [a for a in item['ground_truth'] if a['class_id']==1]
        first=labeled(raw,gt,True) if gt is not None else cv2.resize(raw,(640,480),interpolation=cv2.INTER_NEAREST)
        panel=np.zeros((514,1920,3),dtype=np.uint8);panel[34:,:640]=first
        panel[34:,640:1280]=labeled(raw,predictions['coco']);panel[34:,1280:]=labeled(raw,predictions['custom'])
        for x,txt in [(4,item['item_id']+(' BALL HUMAN LABEL / DEV' if gt is not None else ' ORIGINAL / NO GT')),
                      (644,'COCO sports ball'),(1284,'CUSTOM ball - SAME PARAMETERS')]:
            cv2.putText(panel,txt,(x,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        filename='frames/'+item['item_id']+'.png';require(cv2.imwrite(str(output/filename),panel),'Cannot write comparison')
        record={**item,'ground_truth':gt,'ball_predictions':predictions,'panel':filename}
        if gt is not None:
            record['ball_matching']={n:count_matches(gt,p)['ball'] for n,p in predictions.items()}
            for n,count in record['ball_matching'].items():totals[n].update(count)
            # A class-only filter must not be mistaken for a changed threshold experiment.
            previous=next(r for r in prior_report['images'] if r['item_id']==item['item_id'])
            previous_ball=[p for p in previous['predictions']['new'] if p['class_id']==1]
            require(len(previous_ball)==len(predictions['custom']), 'Custom ball count differs from prior all-class inference')
        records.append(record)
        save(output/'run_state.json',{'status':'running','images_total':53,'images_completed':index})
    presence={n:{'boxes':sum(len(r['ball_predictions'][n]) for r in records if r['ground_truth'] is None),
                'frames_with_detection':sum(bool(r['ball_predictions'][n]) for r in records if r['ground_truth'] is None)} for n in models}
    candidate_ids=[r['item_id'] for r in records if r['ground_truth'] is None and
                   any(r['ball_predictions'][n] for n in models)]
    overview=[]
    # All predicted ball candidate panels are retained, independent of confidence ranking/outcome.
    candidates=[r for r in records if r['item_id'] in candidate_ids]
    for start in range(0,len(candidates),6):
        panels=[]
        for r in candidates[start:start+6]:
            panel=cv2.imread(str(output/r['panel']));panels.append(cv2.resize(panel,(1280,343),interpolation=cv2.INTER_AREA))
        name=f'overview/candidates_{start//6+1}.png';require(cv2.imwrite(str(output/name),np.vstack(panels)),'Overview write failed');overview.append(name)
    report={'schema':config['schema'],'status':'completed','images':records,'images_checked':53,
        'development_ball_matching_conf025_iou05':{n:dict(c) for n,c in totals.items()},
        'matching_is_not_AP':True,'unlabeled_presence_NOT_recall':presence,'unlabeled_candidate_ids':candidate_ids,
        'overview':overview,'input_ids_and_SHA_identical_to_prior_diagnostic':True,
        'parameters_identical_except_semantic_class_filter':True,'training_started':False,
        'source_annotations_modified':False,'threshold_tuning':False,'original_classification_validation_test_used':False,
        'feature_cache_modified':False,'cropping_used':False,'generalization_metrics_valid':False,
        'AI_review_status':'pending','no_finetuning_causal_claim_from_three_development_images':True}
    save(output/'report.json',report)
    cards=[]
    for r in records:
        if r['ground_truth'] is None and r['item_id'] not in candidate_ids:continue
        cards.append(f'<section><h2>{html.escape(r["item_id"])}</h2><p>{html.escape(r["scope"])}</p><img loading="lazy" src="{r["panel"]}"></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>原始COCO與自訓YOLO：球偵測診斷</title><style>body{font:18px system-ui;line-height:1.8;max-width:1600px;margin:25px auto;padding:0 18px;background:#f4f6f8}img{max-width:100%}section{background:white;padding:15px;margin:20px 0}</style><h1>只比較球：原始COCO權重 vs 目前自訓權重</h1><p>左：人工球框（3張開發驗證）或原圖；中：原始YOLO11m的sports ball；右：三類自訓模型的ball。COCO類別為所有運動用球，並非專用籃球；在這批籃球畫面上只作語義對齊診斷。</p><p>完全相同53張無損PNG、imgsz640、conf0.25、NMS IoU0.7、max_det100；沒有調閾值、解析度或再訓練。3張開發驗證可比對球框；額外50張沒有完整真值，預測數量不是recall。來源先前已用於開發檢查，不是正式盲測。</p><p>80改3類時，最後3個尺度分類輸出的weight/bias共6個tensor形狀不符，不自動載入；其餘相容tensor沿用，並非整個偵測頭重置。這是一般類別數轉換行為，不等於程式載錯，也不能單憑此證明退化原因。</p><p><a href="report.json">全部預測與匹配</a> · <a href="initialization_audit.json">逐層形狀相容性</a> · <a href="AI_review.html">AI初審（若已產生）</a></p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    require(all(sha(p)==h for p,h in protected.items()),'Protected inputs changed');check_dataset(dataset)
    save(output/'run_state.json',{'status':'completed','images_total':53,'images_completed':53})
    print({'output':str(output),'development_ball_matching':report['development_ball_matching_conf025_iou05'],
           'unlabeled_presence':presence,'candidate_images':len(candidate_ids),
           'compatible_tensors':compatibility['compatible_tensor_count'],'total_tensors':compatibility['target_tensor_count']},flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',required=True);p.add_argument('--training-run',required=True)
    p.add_argument('--prior-visual',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();print(run_check(a.dataset_dir,a.training_run,a.prior_visual,dry_run=a.dry_run))


if __name__=='__main__':main()
