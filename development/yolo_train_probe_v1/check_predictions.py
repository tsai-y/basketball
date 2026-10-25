"""Visual and IoU checks on the SAME five training images, never held-out performance."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np

from .probe import CLASS_NAMES, check_dataset, read, save, sha, require


def box_iou(a,b):
    area=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union>0 else 0.


def matched_counts(gt,pred,threshold=.5):
    """Greedy confidence-ordered matching; this is not an AP implementation."""
    remaining=set(range(len(gt)));matched=[]
    for i,p in sorted(enumerate(pred),key=lambda x:-x[1]['confidence']):
        eligible=[(box_iou(g['xyxy'],p['xyxy']),j) for j,g in enumerate(gt)
                  if j in remaining and g['class_id']==p['class_id']]
        if eligible:
            overlap,j=max(eligible)
            if overlap>=threshold:
                remaining.remove(j);matched.append({'ground_truth_index':j,'prediction_index':i,'iou':overlap})
    return matched


def draw(image,box,label,color):
    x1,y1,x2,y2=[round(v*3) for v in box]
    cv2.rectangle(image,(x1,y1),(x2-1,y2-1),color,2)
    cv2.putText(image,label,(x1,max(15,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir',required=True);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();dataset=Path(args.dataset_dir).resolve();run=Path(args.run_dir).resolve()
    require(run.is_relative_to(dataset), 'Run outside dataset')
    m=check_dataset(dataset);state=read(run/'run_state.json')
    require(state['status']=='completed' and state['checkpoint_round_trip'], 'Training not completed')
    checkpoint=Path(state['checkpoint']);require(sha(checkpoint)==state['checkpoint_sha256'],'Checkpoint changed')
    os.environ['YOLO_CONFIG_DIR']=str(run/'runtime');os.environ['MPLCONFIGDIR']=str(run/'runtime/matplotlib')
    os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_OFFLINE']='true'
    import torch
    from ultralytics import YOLO
    require(torch.cuda.is_available(),'GPU unavailable')
    model=YOLO(str(checkpoint));require(model.names=={i:n for i,n in enumerate(CLASS_NAMES)},'Wrong classes')
    output=Path(tempfile.mkdtemp(prefix='training_image_visual_check_',dir=run))
    records=[];totals={i:Counter() for i in range(3)}
    for item in m['records']:
        iid=item['image_id'];raw=cv2.imread(str(dataset/item['image']))
        result=model.predict(raw,imgsz=640,conf=.25,device=0,save=False,verbose=False)[0]
        pred=[{'class_id':int(c),'xyxy':box,'confidence':float(score)}
              for box,c,score in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist())]
        gt=item['annotations'];matches=matched_counts(gt,pred)
        for cid in range(3):
            g=sum(a['class_id']==cid for a in gt);p=sum(a['class_id']==cid for a in pred)
            tp=sum(gt[e['ground_truth_index']]['class_id']==cid for e in matches)
            totals[cid].update({'ground_truth':g,'predicted':p,'matched':tp,'false_positive':p-tp,'missed':g-tp})
        left=cv2.resize(raw,(960,720),interpolation=cv2.INTER_NEAREST);right=left.copy()
        colors=[(255,205,40),(70,230,70),(0,230,255)]
        for a in gt:draw(left,a['xyxy'],f'GT {CLASS_NAMES[a["class_id"]]}',colors[a['class_id']])
        for a in pred:draw(right,a['xyxy'],f'{CLASS_NAMES[a["class_id"]]} {a["confidence"]:.2f}',colors[a['class_id']])
        panel=np.zeros((752,1920,3),dtype=np.uint8);panel[32:,:960]=left;panel[32:,960:]=right
        cv2.putText(panel,f'{iid} HUMAN LABELS / TRAINING IMAGE',(5,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        cv2.putText(panel,'PREDICTIONS ON SAME TRAINING IMAGE - NOT HELD OUT',(967,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        assert cv2.imwrite(str(output/f'{iid}_train_gt_prediction.png'),panel)
        records.append({'image_id':iid,'predictions':pred,'matches_at_iou_0_5':matches,
                        'ground_truth_count':len(gt),'prediction_count':len(pred)})
    diagnostics={CLASS_NAMES[i]:dict(totals[i]) for i in range(3)}
    all_memorized=all(v['missed']==0 and v['false_positive']==0 for v in diagnostics.values())
    report={'schema':'YOLO_training_image_matching_diagnostic_v1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'checkpoint':str(checkpoint),
        'checkpoint_sha256':sha(checkpoint),'confidence_threshold':.25,'IoU_threshold':.5,
        'matching_method':'confidence-ordered greedy class-wise IoU matching, not AP',
        'evaluation_scope':'exact_same_five_training_images_only',
        'independent_validation':False,'generalization_metrics_valid':False,'test_used':False,
        'by_class':diagnostics,'all_objects_memorized_at_fixed_threshold':all_memorized,'images':records}
    save(output/'training_image_diagnostic.json',report)
    cards=''.join(f'<h2>{r["image_id"]}</h2><img src="{r["image_id"]}_train_gt_prediction.png">' for r in records)
    (output/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>五張YOLO訓練影像工程驗證</title><style>body{{font:16px system-ui;max-width:1500px;margin:24px auto;line-height:1.7}}img{{max-width:100%}}</style><h1>YOLO五張small-overfit：只驗證訓練樣本記憶能力</h1><p>左側為人工接受標註，右側為同一訓練影像上的預測；不是獨立validation或test。框比對固定conf=0.25、同類IoU≥0.5，不是AP，不能宣稱實際偵測效能。</p><pre>{__import__('json').dumps(diagnostics,ensure_ascii=False,indent=2)}</pre>{cards}<a href="training_image_diagnostic.json">完整工程診斷JSON</a>''',encoding='utf-8')
    check_dataset(dataset)
    print(__import__('json').dumps({'output':str(output),'by_class':diagnostics,
        'all_objects_memorized_at_fixed_threshold':all_memorized,'generalization_metrics_valid':False},indent=2))


if __name__=='__main__':main()
