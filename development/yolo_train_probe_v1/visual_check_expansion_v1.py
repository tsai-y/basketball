"""Fixed-frame old/new YOLO diagnostics; no training, cropping or annotation promotion."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import html
import os
import tempfile

import cv2
import numpy as np

from .probe import ARTIFACTS, CLASS_NAMES, read, save, require, sha, within
from .train_expansion_v1 import check_dataset
from .check_predictions import matched_counts
from .unseen_video_audit import detections_from_arrays, COLORS


def eligible_samples(audit, records):
    require(audit.get('split') == 'train', 'Original train diagnostic only')
    samples = audit.get('samples', [])
    require(samples and all(s['split'] == 'train' and str(s['excluded']) == '0' for s in samples),
            'Non-train or excluded source rejected before PNG access')
    require(len({s['sample_id'] for s in samples}) == len(samples) and
            len({s['video_id'] for s in samples}) == len(samples), 'Duplicate diagnostic samples')
    groups = {r['source_group'] for r in records}
    videos = {r['video_id'] for r in records}
    selected = [dict(s) for s in samples if s['source_group'] not in groups and s['video_id'] not in videos]
    require(selected, 'No sources disjoint from both training and development validation')
    return selected


def count_matches(gt, pred):
    matches = matched_counts(gt, pred, .5)
    result = {}
    for c, name in enumerate(CLASS_NAMES):
        n = sum(a['class_id'] == c for a in gt)
        p = sum(a['class_id'] == c for a in pred)
        tp = sum(gt[m['ground_truth_index']]['class_id'] == c for m in matches)
        result[name] = {'ground_truth': n, 'predicted': p, 'matched': tp,
                        'missed': n-tp, 'extra': p-tp}
    return result


def plan(dataset, run, audit):
    dataset, run, audit = map(lambda p: Path(p).resolve(), (dataset, run, audit))
    require(run.is_relative_to(dataset) and audit.is_relative_to(ARTIFACTS.resolve()), 'Foreign run/audit')
    m = check_dataset(dataset);state = read(run/'run_state.json')
    require(state['status'] == 'completed' and state['checkpoint_round_trip'], 'Training not complete')
    new = Path(state['checkpoints']['best']['path']).resolve()
    require(new.is_relative_to(run) and sha(new) == state['checkpoints']['best']['sha256'], 'New checkpoint changed')
    audit_manifest = read(audit/'manifest.json');config = read(audit/'config.json')
    old = Path(config['weights']).resolve()
    require(old.is_relative_to(ARTIFACTS.resolve()) and sha(old) == config['weights_sha256'], 'Old checkpoint changed')
    selected = eligible_samples(audit_manifest, m['records'])
    inputs = []
    for r in m['records']:
        if r['partition'] != 'val': continue
        inputs.append({'item_id': r['image_id'], 'scope': 'development_with_human_labels',
            'path': str(within(dataset, r['image'])), 'image_sha256': r['image_sha256'],
            'source_group': r['source_group'], 'video_id': r['video_id'], 'frame_index': r['frame_index'],
            'ground_truth': r['annotations']})
    for s in selected:
        for q in (1, 3, 5, 7, 9):
            frame = s['frame_count']*q//10
            p = within(audit, f'original_frames/{s["sample_id"]}_f{frame:06d}.png')
            inputs.append({'item_id': f'{s["sample_id"]}_f{frame:06d}',
                'sample_id': s['sample_id'], 'scope': 'source_disjoint_no_ground_truth',
                'path': str(p), 'image_sha256': sha(p), 'source_group': s['source_group'],
                'video_id': s['video_id'], 'frame_index': frame, 'fixed_position_tenths': q,
                'original_class': s['original_class'], 'ground_truth': None})
    require(len(inputs) == 3+len(selected)*5, 'Incomplete fixed-frame selection')
    protected = {str(p): sha(p) for p in (new, old, run/'run_state.json', dataset/'dataset_manifest.json',
                 audit/'manifest.json', audit/'config.json')}
    protected.update({i['path']: i['image_sha256'] for i in inputs})
    return {'schema': 'sourcepilot_fixed_frame_visual_check_v1', 'inputs': inputs,
        'source_disjoint_samples': selected, 'new_weights': str(new), 'old_weights': str(old),
        'protected_input_sha256': protected, 'parameters': {'imgsz':640, 'conf':.25, 'iou':.7,
            'max_det':100, 'half':False}, 'dataset':str(dataset), 'training_run':str(run),
        'original_video_opens':0, 'original_classification_validation_test_used':False,
        'training_started':False, 'tracking_used':False, 'cropping_used':False,
        'annotations_modified':False, 'feature_cache_modified':False,
        'unseen_source_precision_recall_mAP_available':False,
        'limitation':'Previously inspected diagnostic train sources, not untouched test; five fixed stills per video, not temporal coverage evaluation'}


def labeled(raw, objects, gt=False):
    out = cv2.resize(raw, (640,480), interpolation=cv2.INTER_NEAREST)
    counts = Counter()
    for obj in objects:
        cid = obj['class_id'];counts[cid] += 1
        x1,y1,x2,y2 = [round(v*2) for v in obj['xyxy']]
        color = COLORS[cid]
        cv2.rectangle(out,(x1,y1),(x2-1,y2-1),color,2)
        label = ('GT ' if gt else '')+('P','B','R')[cid]+str(counts[cid])
        if not gt: label += f' {obj["confidence"]:.2f}'
        cv2.putText(out,label,(max(0,min(x1,540)),max(13,y1-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    return out


def execute(dataset, run, audit, *, dry_run=False):
    info = plan(dataset,run,audit)
    if dry_run: return {'dry_run':True, 'files_written':False, 'images':len(info['inputs']),
        'source_disjoint_videos':len(info['source_disjoint_samples']), 'original_video_opens':0}
    import torch
    require(torch.cuda.is_available(), 'CUDA required; no silent CPU fallback')
    output = Path(tempfile.mkdtemp(prefix='visual_check_',dir=Path(run).resolve()))
    for sub in ('frames','overview','runtime'): (output/sub).mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(output/'runtime'), YOLO_AUTOINSTALL='false',
                      YOLO_OFFLINE='true', HF_HUB_OFFLINE='1')
    import ultralytics
    from ultralytics import YOLO
    torch.set_num_threads(4);cv2.setNumThreads(1)
    models = {name:YOLO(info[name+'_weights']) for name in ('old','new')}
    require(all(m.names == dict(enumerate(CLASS_NAMES)) for m in models.values()), 'Wrong classes')
    save(output/'config.json',{**info,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'ultralytics':ultralytics.__version__,'torch':torch.__version__,'source_code_sha256':sha(__file__)})
    save(output/'run_state.json',{'status':'running','images_completed':0,'images_total':len(info['inputs'])})
    print('OUTPUT:',output,flush=True)
    records = []
    totals = {n:{c:Counter() for c in CLASS_NAMES} for n in ('old','new')}
    for index,item in enumerate(info['inputs'],1):
        require(sha(item['path']) == item['image_sha256'], 'Image changed')
        raw = cv2.imread(item['path']);require(raw is not None and raw.shape == (240,320,3), 'Wrong native still')
        predictions = {}
        for name,model in models.items():
            r = model.predict(source=raw,device=0,save=False,verbose=False,**info['parameters'])[0]
            pred = detections_from_arrays(r.boxes.xyxy.cpu().tolist(),r.boxes.cls.cpu().tolist(),r.boxes.conf.cpu().tolist())
            for p in pred: p['origin'] = name+'_checkpoint_prediction_not_ground_truth'
            predictions[name] = pred
        gt = item['ground_truth']
        left = labeled(raw,gt,True) if gt is not None else cv2.resize(raw,(640,480),interpolation=cv2.INTER_NEAREST)
        panel = np.zeros((514,1920,3),dtype=np.uint8)
        panel[34:,:640] = left;panel[34:,640:1280] = labeled(raw,predictions['old'])
        panel[34:,1280:] = labeled(raw,predictions['new'])
        for x,text in [(5,item['item_id']+(' HUMAN LABELS / DEV' if gt is not None else ' ORIGINAL / NO GT')),
                       (645,'OLD: FIVE-IMAGE OVERFIT'),(1285,'NEW: SOURCE-SEPARATED TINY PILOT')]:
            cv2.putText(panel,text,(x,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        filename='frames/'+item['item_id']+'.png';require(cv2.imwrite(str(output/filename),panel),'Cannot write panel')
        record={**item,'predictions':predictions,'panel':filename}
        if gt is not None:
            record['fixed_threshold_diagnostic']={name:count_matches(gt,pred) for name,pred in predictions.items()}
            for name,result in record['fixed_threshold_diagnostic'].items():
                for cls,count in result.items():totals[name][cls].update(count)
        records.append(record)
        save(output/'run_state.json',{'status':'running','images_completed':index,'images_total':len(info['inputs'])})
    overview = []
    samples = info['source_disjoint_samples']
    for start in range(0,len(samples),5):
        rows=[]
        for sample in samples[start:start+5]:
            cells=[]
            for pos in (5,9):
                r=next(r for r in records if r.get('sample_id')==sample['sample_id'] and r['fixed_position_tenths']==pos)
                panel=cv2.imread(str(output/r['panel']))
                crop=panel[34:,640:]
                cells.append(cv2.resize(crop,(640,240),interpolation=cv2.INTER_AREA))
            row=np.zeros((270,1280,3),dtype=np.uint8);row[30:]=np.hstack(cells)
            cv2.putText(row,sample['sample_id']+' '+sample['source_group']+'  50% OLD | NEW     90% OLD | NEW',
                (5,21),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);rows.append(row)
        filename=f'overview/page_{start//5+1}.png'
        require(cv2.imwrite(str(output/filename),np.vstack(rows)),'Cannot write overview');overview.append(filename)
    presence={name:{cls:{'boxes':sum(sum(p['class_name']==cls for p in r['predictions'][name]) for r in records if r['ground_truth'] is None),
        'frames_with_detection':sum(any(p['class_name']==cls for p in r['predictions'][name]) for r in records if r['ground_truth'] is None)}
        for cls in CLASS_NAMES} for name in ('old','new')}
    report={'schema':info['schema'],'status':'completed','images':records,'images_checked':len(records),
        'development_images':3,'unlabeled_source_disjoint_images':len(records)-3,
        'unlabeled_source_disjoint_videos':len(samples),
        'development_fixed_conf025_iou05_matching':{n:{c:dict(v) for c,v in d.items()} for n,d in totals.items()},
        'matching_method':'Confidence-ordered greedy class-wise IoU, not AP',
        'unlabeled_detection_presence_NOT_recall':presence,'overview':overview,
        'no_detection_does_not_mean_absence':True,'prediction_labels_are_ground_truth':False,
        'original_video_opens':0,'training_started':False,'tracking_used':False,'cropping_used':False,
        'source_annotations_modified':False,'feature_cache_modified':False,
        'generalization_metrics_valid':False,'unlabeled_precision_recall_mAP_available':False,
        'AI_review_status':'pending','human_review_status':'not_requested'}
    require(all(sha(p)==h for p,h in info['protected_input_sha256'].items()),'Protected input changed')
    check_dataset(dataset);save(output/'report.json',report)
    cards=[]
    for r in records:
        if r['ground_truth'] is None and r['fixed_position_tenths'] not in (5,9):continue
        cards.append(f'<section><h2>{html.escape(r["item_id"])}</h2><p>{html.escape(r["source_group"])} · {html.escape(r["scope"])}</p><a href="{r["panel"]}"><img loading="lazy" src="{r["panel"]}"></a></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>YOLO 新舊模型偵測診斷</title><style>body{font:17px system-ui;max-width:1700px;margin:25px auto;padding:0 18px;line-height:1.8;background:#f3f5f8}section{background:white;padding:14px;margin:18px 0}img{max-width:100%}</style><h1>YOLO新舊權重：相同畫面比較</h1><p>左：人工標註（3張開發驗證）或未標註原圖；中：舊5張overfit；右：新11張訓練模型。P=球員、B=球、R=籃圈；框號非追蹤ID。</p><p>固定conf=0.25、NMS IoU=0.7。只有3張開發驗證圖可作同類IoU≥0.5的一對一框比對（不是AP）；另外10個來源各5張、共50張原圖沒有完整真值，有框／沒框都不能換算recall或判定物件存在。來源曾被開發檢查，不是正式盲測。</p><p>額外來源同時排除訓練與開發驗證來源。只讀既有無損PNG，不重開原始影片、不訓練、不修改標註、不裁切、不改MixNet特徵。下方顯示50/90%兩位置，全部53張結果在JSON。</p><p><a href="report.json">完整預測與框比對JSON</a> · <a href="AI_review.json">AI初審（若已產生）</a></p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')
    save(output/'run_state.json',{'status':'completed','images_completed':len(records),'images_total':len(records)})
    print('COMPLETED:',output,flush=True)
    return {'output':str(output),'development_matching':report['development_fixed_conf025_iou05_matching'],
            'unlabeled_presence':presence}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',required=True);p.add_argument('--training-run',required=True)
    p.add_argument('--audit-run',required=True);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();print(execute(a.dataset_dir,a.training_run,a.audit_run,dry_run=a.dry_run))


if __name__=='__main__':main()
