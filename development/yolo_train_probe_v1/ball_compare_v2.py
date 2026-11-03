"""Fixed34 full-frame ball comparison: initializer, old11 best, new14 best; no training."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import html
import math
import shutil
import tempfile

import cv2
import numpy as np

from .probe import read,save,sha,within,require
from .ball_dataset_v2 import check as check_dataset,SCHEMA
from .ball_only_probe_v1 import check as check_old,runtime_at
from .ball_pretraining_diagnostic_v1 import map_ball_predictions
from .visual_check_expansion_v1 import count_matches,labeled

PARAMETERS={'imgsz':640,'conf':.25,'iou':.7,'max_det':100,'half':False}
ARMS=('initializer','old11','new14')


def validate_inputs(plan,records):
    require(plan['parameters']==PARAMETERS, 'Frozen inference parameters changed')
    inputs=plan['inputs']
    require(len(inputs)==34 and len({r['item_id'] for r in inputs})==34, 'Expected34 unique fixed inputs')
    dev={r['image_id']:r for r in records if r['partition']=='val'}
    labeled_rows=[r for r in inputs if r['ground_truth'] is not None]
    require(len(labeled_rows)==4 and {r['item_id'] for r in labeled_rows}==set(dev), 'Wrong development allowlist')
    for r in labeled_rows:
        old=dev[r['item_id']]
        require(all(r[k]==old[k] for k in ('source_group','video_id','frame_index','image_sha256')), 'Development identity changed')
        require(r['ground_truth']==[{'class_id':1,'xyxy':a['xyxy']} for a in old['annotations']], 'Development GT differs from human acceptance')
    used={r['source_group'] for r in records};used_videos={r['video_id'] for r in records}
    extra=[r for r in inputs if r['ground_truth'] is None]
    require(len(extra)==30 and len({r['source_group'] for r in extra})==6, 'Wrong unlabeled diagnostic sources')
    require(all(r['source_group'] not in used and r['video_id'] not in used_videos for r in extra), 'Diagnostic source leakage')
    require(len({r['image_sha256'] for r in inputs})==34, 'Duplicate input pixels')
    return deepcopy(inputs)


def completed_best(run,dataset,manifest_sha=None):
    run=Path(run).resolve();require(run.is_relative_to(Path(dataset).resolve()), 'Foreign trained run')
    state_path=within(run,'run_state.json');s=read(state_path)
    require(s['status']=='completed' and s['checkpoint_round_trip'] and s['epochs_completed']>1
            and s['backbone_parameters_and_BN_buffers_unchanged'], 'Full completed head training required')
    require(s['stage']!='smoke' and s['dataset']==str(Path(dataset).resolve()), 'Wrong training dataset/stage')
    if manifest_sha is not None:
        require(s['schema']==SCHEMA and s['dataset_manifest_sha256']==manifest_sha, 'New training manifest changed')
    p=Path(s['checkpoints']['best']['path']).resolve()
    require(p.is_relative_to(run) and sha(p)==s['checkpoints']['best']['sha256'], 'Selected checkpoint changed')
    return p,state_path,s


def make_plan(dataset,new_run,old_run):
    dataset=Path(dataset).resolve();m=check_dataset(dataset)
    old_dataset=Path(m['base_dataset']);old=check_old(old_dataset)
    require(m['initializer_sha256']==old['initializer_sha256'], 'Initialization differs between versions')
    new,new_state,ns=completed_best(new_run,dataset,sha(dataset/'dataset_manifest.json'))
    previous,old_state,os=completed_best(old_run,old_dataset)
    require(ns['initializer_sha256']==os['initializer_sha256']==m['initializer_sha256'], 'Trained runs used different initialization')
    diagnostic=within(dataset,'diagnostic_plan.json');inputs=validate_inputs(read(diagnostic),m['records'])
    paths={'initializer':m['initializer'],'old11':str(previous),'new14':str(new)}
    protected={str(p):sha(p) for p in (dataset/'dataset_manifest.json',old_dataset/'dataset_manifest.json',new_state,old_state,diagnostic)}
    protected.update({p:sha(p) for p in paths.values()})
    for r in inputs:
        require(sha(r['path'])==r['image_sha256'], 'Fixed input changed')
        protected[r['path']]=r['image_sha256']
    return {'schema':'fixed34_ball_only_three_version_comparison_v2','dataset':str(dataset),
            'new_run':str(Path(new_run).resolve()),'old_run':str(Path(old_run).resolve()),
            'inputs':inputs,'models':paths,'parameters':PARAMETERS,'protected_input_sha256':protected,
            'training_started':False,'original_classification_validation_test_used':False,
            'annotations_modified':False,'feature_cache_changed':False,'generalization_metrics_valid':False,
            'limitations':'4 repeatedly inspected easy-selected development images,30 unlabeled images; not blind or representative; training budgets and image counts differ'}


def pages(tiles,columns,rows,folder,prefix):
    result=[]
    for start in range(0,len(tiles),columns*rows):
        group=list(tiles[start:start+columns*rows])
        while len(group)<columns*rows:group.append(np.zeros_like(tiles[0]))
        canvas=np.vstack([np.hstack(group[i:i+columns]) for i in range(0,len(group),columns)])
        filename=f'{prefix}_{len(result)+1}.png'
        require(cv2.imwrite(str(folder/filename),canvas), 'Display write failed');result.append(filename)
    return result


def execute(dataset,new_run,old_run):
    plan=make_plan(dataset,new_run,old_run)
    import torch
    require(torch.cuda.is_available(), 'CUDA required; no silent fallback')
    folder=Path(tempfile.mkdtemp(prefix='fixed34_comparison_',dir=Path(plan['new_run'])))
    (folder/'frames').mkdir();runtime_at(folder);shutil.copy2(__file__,folder/'source_snapshot.py')
    from ultralytics import YOLO
    import ultralytics
    torch.set_num_threads(4);cv2.setNumThreads(1)
    models={n:YOLO(p) for n,p in plan['models'].items()}
    require(all(model.names=={0:'ball'} for model in models.values()), 'Wrong semantic class')
    save(folder/'config.json',{**plan,'source_code_sha256':sha(__file__),
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'ultralytics':ultralytics.__version__,'torch':torch.__version__})
    totals={n:Counter() for n in ARMS};records=[];native=[];contexts=[];crop_meta=[];cards=[]
    print('OUTPUT:',folder,flush=True);save(folder/'run_state.json',{'status':'running'})
    try:
        for item in plan['inputs']:
            raw=cv2.imread(item['path']);require(raw is not None and raw.shape==(240,320,3), 'Wrong native pixels')
            predictions={}
            for name,model in models.items():
                r=model.predict(source=raw,classes=[0],device=0,save=False,verbose=False,**PARAMETERS)[0]
                predictions[name]=map_ball_predictions(r.boxes.xyxy.cpu().tolist(),r.boxes.cls.cpu().tolist(),r.boxes.conf.cpu().tolist(),0,name)
                for index,p in enumerate(predictions[name],1):
                    x1,y1,x2,y2=p['xyxy'];cx,cy=(x1+x2)/2,(y1+y2)/2
                    left,top=max(0,math.floor(cx)-24),max(0,math.floor(cy)-24)
                    right,bottom=min(320,math.ceil(cx)+24),min(240,math.ceil(cy)+24)
                    crop=raw[top:bottom,left:right].copy()
                    cv2.rectangle(crop,(round(x1)-left,round(y1)-top),(round(x2)-left,round(y2)-top),(0,180,255),1)
                    tile=np.zeros((344,320,3),np.uint8);tile[46:340,13:307]=cv2.resize(crop,(294,294),interpolation=cv2.INTER_NEAREST)
                    cv2.putText(tile,f'{item["item_id"]} {name} B{index}',(4,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                    cv2.putText(tile,f'conf {p["confidence"]:.3f} xy {cx:.0f},{cy:.0f}',(4,37),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
                    contexts.append(tile);crop_meta.append({'item_id':item['item_id'],'model':name,'box_index':index,
                        'prediction':p,'context_native_xyxy':[left,top,right,bottom]})
            row={**item,'predictions':predictions,'panel':f'frames/{item["item_id"]}.png'}
            gt=item['ground_truth']
            if gt is not None:
                row['matching']={n:count_matches(gt,p)['ball'] for n,p in predictions.items()}
                for n,v in row['matching'].items():totals[n].update(v)
            panel=np.zeros((514,2560,3),np.uint8)
            panel[34:,:640]=labeled(raw,gt,True) if gt is not None else cv2.resize(raw,(640,480),interpolation=cv2.INTER_NEAREST)
            titles=[item['item_id']+(' HUMAN BALL / DEVELOPMENT' if gt is not None else ' ORIGINAL / NO GT'),
                    'COCO BALL INITIALIZER','OLD11 / BEST','NEW14 / BEST']
            for i,n in enumerate(ARMS,1):panel[34:,i*640:(i+1)*640]=labeled(raw,predictions[n])
            for i,title in enumerate(titles):cv2.putText(panel,title,(i*640+4,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            require(cv2.imwrite(str(folder/row['panel']),panel), 'Comparison panel write failed')
            tile=np.zeros((264,320,3),np.uint8);tile[24:]=raw
            cv2.putText(tile,item['item_id'],(3,17),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1);native.append(tile)
            if gt is not None or any(predictions.values()):cards.append(f'<section id="{html.escape(item["item_id"])}"><h2>{html.escape(item["item_id"])}</h2><img src="{row["panel"]}"></section>')
            records.append(row)
        presence={n:{'boxes':sum(len(r['predictions'][n]) for r in records if r['ground_truth'] is None),
                     'frames_with_detection':sum(bool(r['predictions'][n]) for r in records if r['ground_truth'] is None)} for n in ARMS}
        report={'status':'completed','schema':plan['schema'],'images':records,'images_checked':34,'parameters':PARAMETERS,
                'development_matching_conf025_iou05':{n:dict(v) for n,v in totals.items()},
                'unlabeled_presence_NOT_precision_or_recall':presence,
                'native_overview_pages':pages(native,4,3,folder,'native_overview'),
                'candidate_context_pages':pages(contexts,3,2,folder,'candidate_context') if contexts else [],
                'candidate_contexts':crop_meta,'crops_are_display_only_not_model_inputs':True,
                'protected_input_sha256':plan['protected_input_sha256'],'AI_review_status':'pending',
                'generalization_metrics_valid':False,'training_started':False,'new_human_labels_created':False,
                'limitations':plan['limitations']}
        save(folder/'report.json',report)
        summary='<table><tr><th>版本</th><th>開發匹配/4</th><th>漏框</th><th>多餘框</th><th>未標註30圖框數（不是recall）</th></tr>'
        for n in ARMS:
            v=totals[n];summary+=f'<tr><td>{n}</td><td>{v["matched"]}/4</td><td>{v["missed"]}</td><td>{v["extra"]}</td><td>{presence[n]["boxes"]}</td></tr>'
        summary+='</table>'
        (folder/'index.html').write_text('''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>相同34全圖：球模型三版本對照</title><style>body{font:18px system-ui;line-height:1.8;max-width:1800px;margin:25px auto;padding:0 20px}img{max-width:100%}td,th{padding:8px 14px;border:1px solid #ccd}table{border-collapse:collapse}section{margin:25px 0}</style><h1>同一批34張全圖：初始化／舊11張／新14張</h1><p>依序為人工球框／原圖、COCO球語意初始化、舊11張best、新14張best。所有模型相同640、conf0.25、NMS IoU0.7、max_det100；人工開發球框以IoU0.5一對一匹配。未掃門檻、未再訓練、未改標註或MixNet特徵。</p><p>4張開發圖曾被反覆檢視、模型挑選，且偏向可辨識易例；30張其他來源沒有完整真值。不是盲測，框數不是precision/recall；新舊訓練預算不同，差異不作單一因素因果歸因。不直接部署到全部影片。</p>'''+summary+'<p><a href="report.json">完整預測</a> · <a href="AI_review.html">AI視覺初審（若有）</a></p>'+''.join(cards)+'</html>',encoding='utf-8')
        require(all(sha(p)==h for p,h in plan['protected_input_sha256'].items()), 'Protected inputs changed')
        check_dataset(dataset);save(folder/'run_state.json',{'status':'completed','images_completed':34})
        print({'output':str(folder),'development_matching':report['development_matching_conf025_iou05'],'presence':presence},flush=True)
        return folder
    except BaseException as exc:
        save(folder/'run_state.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}'});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',required=True)
    p.add_argument('--new-run',required=True);p.add_argument('--old-run',required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:print({'inputs':len(make_plan(a.dataset_dir,a.new_run,a.old_run)['inputs']),'training_started':False,'files_written':False})
    else:print(execute(a.dataset_dir,a.new_run,a.old_run))


if __name__=='__main__':main()
