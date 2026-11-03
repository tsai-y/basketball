"""Saved best/last inference on14 training full frames; memorization diagnosis only."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import html
import shutil
import tempfile

import cv2
import numpy as np

from .probe import read,save,sha,require
from .ball_dataset_v2 import check as check_dataset
from .ball_compare_v2 import completed_best,PARAMETERS
from .ball_only_probe_v1 import runtime_at
from .visual_check_expansion_v1 import count_matches,labeled
from .ball_pretraining_diagnostic_v1 import map_ball_predictions
from .ball_source_review_v2 import check_proposal


def training_rows(records):
    require(len({r['image_id'] for r in records})==len(records), 'Duplicate dataset IDs')
    require(Counter(r['partition'] for r in records)=={'train':14,'val':4}, 'Fixed14/4 dataset required')
    require(all(r['source_split']=='train' for r in records), 'Original classification train only')
    train=[r for r in records if r['partition']=='train'];dev=[r for r in records if r['partition']=='val']
    for field in ('source_group','video_id','image_sha256'):
        require({r[field] for r in train}.isdisjoint({r[field] for r in dev}), 'Train/development overlap')
    for r in train:
        require(r['annotations'] and all(a['class_id']==0 for a in r['annotations']), 'Expected approved ball-only training labels')
        check_proposal([a['xyxy'] for a in r['annotations']])
    return deepcopy(train)


def summarize(rows):
    require(len(rows)==14, 'Exactly14 training predictions required')
    totals=Counter();groups={};missed=[];extra=[]
    for row in rows:
        totals.update(row['matching'])
        group=row['annotation_origin'];groups.setdefault(group,Counter()).update(row['matching'])
        if row['matching']['missed']:missed.append(row['image_id'])
        if row['matching']['extra']:extra.append(row['image_id'])
    return {'fixed_conf025_iou05':dict(totals),'by_annotation_origin':{k:dict(v) for k,v in groups.items()},
            'images_with_missed_balls':missed,'images_with_extra_boxes':extra,
            'scope':'training resubstitution, NOT independent evaluation or Make/Miss accuracy'}


def render_page(output,report,checkpoint):
    require(checkpoint in ('best','last'), 'Unknown checkpoint role')
    t=report['summary']['fixed_conf025_iou05']
    cards=[f'<section id="{html.escape(r["image_id"])}"><h2>{html.escape(r["image_id"])}</h2><img src="{r["panel"]}"></section>' for r in report['images']]
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>14張訓練圖重代入：非泛化成績</title><style>body{font:18px system-ui;line-height:1.8;max-width:1400px;margin:30px auto;padding:0 20px}img{max-width:100%}section{margin:25px 0;border-top:1px solid #ccd}</style><h1>14張訓練全圖重代入</h1>'''
    page+=f'<p>左為已接受人工球框，右為新14張版的 {checkpoint} 權重。這些是訓練看過的影格，只診斷記憶／擬合，不是獨立泛化、不是Make/Miss分類準確率。不重訓、不降門檻、不改標註／MixNet，沒有對開發圖再推論。</p>'
    page+=f'<p>本次權重：{checkpoint}（不是根據訓練圖重挑best）。固定640/conf0.25/NMS IoU0.7，以IoU0.5一對一匹配：匹配{t["matched"]}/{t["ground_truth"]}、漏{t["missed"]}、多餘{t["extra"]}。</p>'
    page+='<p><a href="report.json">完整預測與SHA</a> · <a href="AI_review.html">AI複查與解讀（若有）</a></p>'+''.join(cards)+'</html>'
    (Path(output)/'index.html').write_text(page,encoding='utf-8')


def execute(dataset,run,dry_run=False,checkpoint='best'):
    require(checkpoint in ('best','last'), 'Only saved best/last checkpoints allowed')
    dataset,run=Path(dataset).resolve(),Path(run).resolve();m=check_dataset(dataset)
    rows=training_rows(m['records']);weights,state_path,state=completed_best(run,dataset,sha(dataset/'dataset_manifest.json'))
    if checkpoint=='last':
        weights=Path(state['checkpoints']['last']['path']).resolve()
        require(weights.is_relative_to(run) and sha(weights)==state['checkpoints']['last']['sha256'], 'Last checkpoint changed')
    protected={str(p):sha(p) for p in (weights,state_path,dataset/'dataset_manifest.json')}
    for r in rows:
        path=dataset/r['image'];require(sha(path)==r['image_sha256'], 'Training source pixels changed')
        protected[str(path)]=r['image_sha256']
    if dry_run:return {'images':len(rows),'development_images_inferred':0,'training_started':False,'files_written':False}
    import torch
    require(torch.cuda.is_available(), 'CUDA required; no silent CPU fallback')
    output=Path(tempfile.mkdtemp(prefix='training_resubstitution_v2_',dir=run));(output/'frames').mkdir()
    runtime_at(output);shutil.copy2(__file__,output/'source_snapshot.py')
    from ultralytics import YOLO
    import ultralytics
    torch.set_num_threads(4);cv2.setNumThreads(1);model=YOLO(str(weights))
    require(model.names=={0:'ball'}, 'Wrong semantic class')
    report={'status':'running','schema':'ball_v2_training_full_frame_resubstitution',
            'dataset':str(dataset),'training_run':str(run),'selected_checkpoint':str(weights),
            'checkpoint_role':checkpoint,'best_checkpoint_not_reselected_from_training_diagnosis':True,
            'parameters':PARAMETERS,'matching_IoU':.5,'ultralytics':ultralytics.__version__,'torch':torch.__version__,
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_code_sha256':sha(__file__),
            'training_started':False,'threshold_tuning':False,'new_human_labels_created':False,
            'annotations_modified':False,'original_classification_validation_test_used':False,
            'MixNet_features_changed':False,'generalization_metrics_valid':False,'development_images_inferred':0,
            'protected_input_sha256':protected,'AI_review_status':'pending','images':[]}
    save(output/'report.json',report);print('OUTPUT:',output,flush=True)
    panels=[]
    try:
        for row in rows:
            path=dataset/row['image'];raw=cv2.imread(str(path))
            require(raw is not None and raw.shape==(240,320,3), 'Expected native320x240 full frame')
            gt=[{'class_id':1,'xyxy':a['xyxy']} for a in row['annotations']]
            result=model.predict(source=raw,classes=[0],device=0,save=False,verbose=False,**PARAMETERS)[0]
            predictions=map_ball_predictions(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),
                                              result.boxes.conf.cpu().tolist(),0,'new14_'+checkpoint+'_training_resubstitution')
            matching=count_matches(gt,predictions)['ball']
            panel=np.zeros((514,1280,3),np.uint8)
            panel[34:,:640]=labeled(raw,gt,True);panel[34:,640:]=labeled(raw,predictions)
            title=f'{row["image_id"]} TRAIN GT / {row["source_group"]}'
            cv2.putText(panel,title,(4,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(panel,f'{checkpoint.upper()} / matched {matching["matched"]} miss {matching["missed"]} extra {matching["extra"]}',
                        (644,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            rel=f'frames/{row["image_id"]}.png';require(cv2.imwrite(str(output/rel),panel), 'Panel write failed')
            report['images'].append({'image_id':row['image_id'],'source_group':row['source_group'],
                'video_id':row['video_id'],'frame_index':row['frame_index'],'annotation_origin':row['annotation_origin'],
                'path':str(path),'image_sha256':row['image_sha256'],'ground_truth':gt,'predictions':predictions,
                'matching':matching,'panel':rel,'panel_sha256':sha(output/rel)})
            panels.append(cv2.resize(panel,(960,386),interpolation=cv2.INTER_AREA))
        overview=[]
        for start in range(0,len(panels),4):
            rel=f'overview_{len(overview)+1}.png';require(cv2.imwrite(str(output/rel),np.vstack(panels[start:start+4])), 'Overview failed');overview.append(rel)
        report.update(status='completed',images_checked=14,summary=summarize(report['images']),overview=overview)
        require(all(sha(p)==h for p,h in protected.items()), 'Protected model/data changed')
        check_dataset(dataset);save(output/'report.json',report)
        render_page(output,report,checkpoint)
        print({'output':str(output),'training_matching_NOT_generalization':report['summary']},flush=True);return output
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');save(output/'report.json',report);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset-dir',required=True)
    p.add_argument('--run-dir',required=True);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--checkpoint',choices=('best','last'),default='best');a=p.parse_args()
    print(execute(a.dataset_dir,a.run_dir,a.dry_run,a.checkpoint))


if __name__=='__main__':main()
