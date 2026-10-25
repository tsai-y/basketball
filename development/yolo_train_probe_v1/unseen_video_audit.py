"""Source-disjoint original-train video visualization of the five-image YOLO probe.

Inference only: no training, tracking, cropping, deletion, feature-cache changes,
or original validation/test access. Detection presence is not object recall.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import html
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger, SplitGuard, load_gate_config
from .probe import ARTIFACTS, AUDITS, CLASS_NAMES, check_dataset, read, require, save, sha

COLORS = ((255, 205, 40), (70, 230, 70), (0, 230, 255))
PREFIXES = ('P', 'B', 'R')
VIDEO_FIELDS = ('player_coverage', 'nonplayer_false_boxes', 'ball_localization',
                'rim_localization', 'shot_critical_dropout', 'scene_cut_behavior', 'notes')


def select_samples(audit, training):
    require(audit.get('split') == 'train', 'Only original train audit allowed')
    rows = audit.get('samples', [])
    require(rows and all(r['split'] == 'train' and str(r['excluded']) == '0' for r in rows),
            'Non-train/excluded sample rejected before video access')
    require(len({r['video_id'] for r in rows}) == len(rows), 'Duplicate video IDs')
    groups = set(training['source_groups'])
    trained_ids = {r['video_id'] for r in training['records']}
    selected = [dict(r) for r in rows if r['source_group'] not in groups and r['video_id'] not in trained_ids]
    require(10 <= len(selected) <= 20, 'Expected 10..20 eligible source-disjoint diagnostic videos')
    require(not ({r['source_group'] for r in selected} & groups), 'Training-source overlap')
    return [{**r, 'prior_sample_id': r['sample_id'], 'sample_id': f'U{i:03d}'}
            for i, r in enumerate(selected, 1)]


def detections_from_arrays(boxes, classes, scores):
    require(len(boxes) == len(classes) == len(scores), 'Box/class/score length mismatch')
    detections = []
    counts = Counter()
    for box, cls, score in zip(boxes, classes, scores):
        require(float(cls).is_integer() and int(cls) in range(3), 'Unexpected detector class')
        cid = int(cls)
        require(len(box) == 4 and all(math.isfinite(float(v)) for v in box), 'Non-finite box')
        x1, y1, x2, y2 = box
        require(0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 240, 'Box outside native image')
        require(math.isfinite(float(score)) and .25 <= score <= 1, 'Unexpected confidence')
        counts[cid] += 1
        detections.append({'detection_id': PREFIXES[cid]+str(counts[cid]), 'class_id': cid,
            'class_name': CLASS_NAMES[cid], 'xyxy': list(map(float, box)), 'confidence': float(score),
            'track_id': None, 'origin': 'five_training_image_probe_prediction_not_ground_truth'})
    return detections


def draw_pair(raw, detections, sid, frame_index, fps):
    require(raw.dtype == np.uint8 and raw.shape == (240, 320, 3), 'Bad native frame')
    right = raw.copy()
    for det in detections:
        x1,y1,x2,y2 = [round(v) for v in det['xyxy']]
        color = COLORS[det['class_id']]
        cv2.rectangle(right, (x1,y1), (x2-1,y2-1), color, 1)
        label = f'{det["detection_id"]} {det["confidence"]:.2f}'
        cv2.putText(right, label, (max(0,min(x1,259)),max(9,y1-2)),
                    cv2.FONT_HERSHEY_SIMPLEX, .28, color, 1, cv2.LINE_AA)
    pair = np.zeros((264,640,3), dtype=np.uint8)
    pair[24:,:320] = raw;pair[24:,320:] = right
    cv2.putText(pair, f'{sid} f{frame_index} {frame_index/fps:.2f}s ORIGINAL',
                (3,15),cv2.FONT_HERSHEY_SIMPLEX,.33,(255,255,255),1)
    cv2.putText(pair, 'P=player B=ball R=rim; NO TRACKING',
                (323,15),cv2.FONT_HERSHEY_SIMPLEX,.30,(255,255,255),1)
    return pair


def summarize(frames):
    summary = {}
    for cid, name in enumerate(CLASS_NAMES):
        counts = [sum(d['class_id']==cid for d in f['detections']) for f in frames]
        summary[name] = {'boxes':sum(counts), 'frames_with_detection':sum(n>0 for n in counts),
            'frames_without_detection':sum(n==0 for n in counts),
            'detection_presence_fraction':sum(n>0 for n in counts)/len(counts)}
    return summary


def process_video(sample, guard, model, output, trained_groups):
    # These guards must precede identity/path lookup or descriptor open.
    require(sample['split']=='train', 'Non-train video rejected before path access')
    require(sample['source_group'] not in trained_groups, 'Training-source video rejected before path access')
    before = guard.identity(sample['video_id'])
    fd = guard.open(sample['video_id'], sample['video_path'], 'custom_yolo_source_disjoint_full_video_audit')
    cap = encoder = None
    n = sample['frame_count'];fps = float(Fraction(sample['fps_rational']))
    indices = tuple(n*q//10 for q in (1,3,5,7,9))
    frames=[];snapshots=[]
    source_identity = {'before':before}
    try:
        cap = cv2.VideoCapture(f'/proc/self/fd/{fd}', cv2.CAP_FFMPEG)
        require(cap.isOpened(), 'Cannot open guarded original video')
        require(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))==n, 'Frame count changed')
        require(abs(cap.get(cv2.CAP_PROP_FPS)-fps)<1e-4, 'FPS changed')
        destination = output/'videos'/f'{sample["sample_id"]}.mp4'
        require(not destination.exists(), 'Refusing video overwrite')
        encoder = subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','bgr24',
            '-s','640x264','-r',sample['fps_rational'],'-i','pipe:0','-an','-c:v','libx264',
            '-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart','-n',str(destination)],
            stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        for index in range(n):
            ok,raw=cap.read();require(ok and raw.shape==(240,320,3) and raw.dtype==np.uint8,'Bad/truncated original frame')
            result=model.predict(source=raw,imgsz=640,conf=.25,iou=.7,max_det=100,
                device=0,half=False,verbose=False,save=False,stream=False)[0]
            det=detections_from_arrays(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),
                                       result.boxes.conf.cpu().tolist())
            pair=draw_pair(raw,det,sample['sample_id'],index,fps)
            encoder.stdin.write(pair.tobytes())
            frames.append({'frame_index':index,'time_seconds':index/fps,'detections':det})
            if index in indices:
                filename=f'{sample["sample_id"]}_f{index:06d}.png'
                require(cv2.imwrite(str(output/'frames'/filename),pair),'Snapshot write failed')
                require(cv2.imwrite(str(output/'original_frames'/filename),raw),'Original snapshot write failed')
                snapshots.append(pair)
        require(not cap.read()[0], 'Extra decoded frames')
        encoder.stdin.close();error=encoder.stderr.read().decode(errors='replace')
        require(encoder.wait(timeout=30)==0,'Encoder failed: '+error[-500:])
    finally:
        if cap is not None:cap.release()
        os.close(fd)
        if encoder is not None and encoder.poll() is None:
            encoder.kill();encoder.wait()
    after=guard.identity(sample['video_id']);require(before==after,'Original video changed')
    source_identity['after']=after
    require(len(frames)==n and len(snapshots)==5,'Incomplete audit')
    require(cv2.imwrite(str(output/'contacts'/f'{sample["sample_id"]}.png'),np.vstack(snapshots)), 'Contact write failed')
    check=cv2.VideoCapture(str(destination));require(check.isOpened(),'Cannot reopen encoded diagnostic')
    try:
        require(round(check.get(cv2.CAP_PROP_FRAME_COUNT))==n,'Encoded frame count mismatch')
        require(abs(check.get(cv2.CAP_PROP_FPS)-fps)<1e-4,'Encoded FPS mismatch')
        require(round(check.get(cv2.CAP_PROP_FRAME_WIDTH))==640 and round(check.get(cv2.CAP_PROP_FRAME_HEIGHT))==264,
                'Bad encoded visualization size')
    finally:check.release()
    save(output/'detections'/f'{sample["sample_id"]}.json',{'sample_id':sample['sample_id'],
        'video_id':sample['video_id'],'original_split':'train','source_group':sample['source_group'],
        'source_identity':source_identity,'prediction_origin':'five_image_small_overfit_model_not_ground_truth',
        'frames':frames,'training_started':False,'tracking_used':False})
    return {'sample_id':sample['sample_id'],'prior_sample_id':sample['prior_sample_id'],
        'video_id':sample['video_id'],'source_group':sample['source_group'],'frames':n,
        'per_class':summarize(frames),'source_unchanged':True,'snapshot_indices':indices}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--audit-run',required=True);parser.add_argument('--dataset-dir',required=True)
    parser.add_argument('--training-run',required=True)
    args=parser.parse_args(argv)
    audit=Path(args.audit_run).resolve();require(audit.is_relative_to(AUDITS.resolve()),'Outside local audit artifacts')
    training=check_dataset(args.dataset_dir);run=Path(args.training_run).resolve()
    require(run.is_relative_to(Path(args.dataset_dir).resolve()),'Foreign training run')
    completed=read(run/'run_state.json');require(completed['status']=='completed' and completed['checkpoint_round_trip'],
        'Probe not completed')
    weights=Path(completed['checkpoint']);require(weights.is_relative_to(run) and sha(weights)==completed['checkpoint_sha256'],
        'Checkpoint missing/changed')
    audit_manifest=read(audit/'manifest.json');samples=select_samples(audit_manifest,training)
    config=load_gate_config();ledger=AccessLedger();guard=SplitGuard('train',ledger)
    for sample in samples:
        require(sample['video_id'] in guard.allowed,'Selected ID not in original train allowlist')
        official=guard.allowed[sample['video_id']]
        require(all(sample[k]==official[k] for k in ('video_path','source_group','original_class','binary_label','split')),
                'Audit row differs from fixed train manifest')
    plan={'schema':'source_disjoint_yolo_probe_video_audit_v1','split':'train','videos':len(samples),
        'samples':samples,'trained_source_groups':training['source_groups'],
        'source_groups_disjoint_from_probe_training':True,'training_started':False,'test_used':False,
        'tracking_used':False,'cropping_used':False,'feature_cache_modified':False,
        'not_formal_held_out_evaluation':True,
        'sampling_limitation':'Previously selected diagnostic train videos, not a representative/untouched detector validation set'}
    if args.dry_run:
        print(json.dumps({'dry_run':True,'video_opens':0,'model_loaded':False,'files_written':False,'plan':plan},indent=2))
        return
    output=Path(tempfile.mkdtemp(prefix='unseen_video_audit_',dir=ARTIFACTS))
    for name in ('videos','frames','original_frames','contacts','detections','runtime'):(output/name).mkdir()
    os.environ['YOLO_CONFIG_DIR']=str(output/'runtime');os.environ['MPLCONFIGDIR']=str(output/'runtime/matplotlib')
    os.environ['YOLO_AUTOINSTALL']='false';os.environ['YOLO_OFFLINE']='true'
    import torch
    import ultralytics
    from ultralytics import YOLO
    require(torch.cuda.is_available(),'GPU unavailable; no silent CPU fallback')
    torch.set_num_threads(4);cv2.setNumThreads(1)
    protected={str(audit/'manifest.json'):sha(audit/'manifest.json'),str(weights):sha(weights),
        str(Path(args.dataset_dir).resolve()/'dataset_manifest.json'):sha(Path(args.dataset_dir).resolve()/'dataset_manifest.json')}
    save(output/'manifest.json',plan)
    protocol={'weights':str(weights),'weights_sha256':sha(weights),'classes':CLASS_NAMES,
        'ultralytics':ultralytics.__version__,'torch':torch.__version__,'opencv':cv2.__version__,
        'parameters':{'imgsz':640,'conf':.25,'iou':.7,'max_det':100,'device':0,'half':False,'every_frame':True},
        'tracker':None,'source':'Original dataset train videos opened read-only with exact allowlist',
        'gate_config_sha256':config.sha256,'input_identities':config.raw['identities'],
        'source_code_sha256':sha(Path(__file__)),'protected_source_sha256':protected,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'training_started':False}
    save(output/'config.json',protocol)
    save(output/'run_state.json',{'status':'running','videos_completed':0,'videos_total':len(samples)})
    print('OUTPUT:',output,flush=True)
    model=YOLO(str(weights));require(model.names=={i:n for i,n in enumerate(CLASS_NAMES)},'Wrong class mapping')
    results=[]
    try:
        for index,sample in enumerate(samples,1):
            result=process_video(sample,guard,model,output,set(training['source_groups']));results.append(result)
            save(output/'run_state.json',{'status':'running','videos_completed':index,'videos_total':len(samples)})
            print(f'{index}/{len(samples)} {sample["sample_id"]} ({sample["prior_sample_id"]}) frames={result["frames"]} '+
                  str({k:v['frames_with_detection'] for k,v in result['per_class'].items()}),flush=True)
        access=ledger.statistics()
        require(access['train']['opens']==len(samples) and access['test']['opens']==access['validation']['opens']==0,
                'Unexpected original-video access')
        require(all(sha(Path(p))==h for p,h in protected.items()),'Protected input changed')
        check_dataset(args.dataset_dir)
        frames=sum(r['frames'] for r in results)
        totals={n:{'boxes':sum(r['per_class'][n]['boxes'] for r in results),
                   'frames_with_detection':sum(r['per_class'][n]['frames_with_detection'] for r in results)} for n in CLASS_NAMES}
        summary={**{k:v for k,v in plan.items() if k!='samples'},'frames':frames,'per_class':totals,
            'per_video':results,'source_videos_unchanged':True,'original_video_access':access,
            'precision_recall_mAP_available':False,'no_detection_does_not_mean_object_absent':True,
            'human_review_status':'pending','prediction_labels_are_ground_truth':False}
        save(output/'summary.json',summary)
        save(output/'human_review.json',{'schema':'source_disjoint_video_visual_review_v1',
            'review_origin':'unreviewed','videos':[{'sample_id':r['sample_id'],
                **{k:None for k in VIDEO_FIELDS}} for r in results],
            'no_auto_annotation_or_training':True})
        cards=[]
        for r in results:
            counts={n:r['per_class'][n]['frames_with_detection'] for n in CLASS_NAMES}
            cards.append(f'<section id="{r["sample_id"]}"><h2>{r["sample_id"]}（原診斷{r["prior_sample_id"]}）</h2><p>{html.escape(r["video_id"])} · {html.escape(r["source_group"])}</p><video controls preload="metadata" src="videos/{r["sample_id"]}.mp4"></video><p>共{r["frames"]}幀；有偵測框幀數（不是recall）：{html.escape(str(counts))}</p><details><summary>固定10/30/50/70/90%五幀</summary><img loading="lazy" src="contacts/{r["sample_id"]}.png"></details></section>')
        page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>YOLO小樣本模型：來源分離影片診斷</title><style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;line-height:1.7;background:#f5f7fa}section{background:white;padding:16px;margin:20px 0}video,img{width:960px;max-width:100%}button{padding:8px;margin:4px}</style><h1>五張small-overfit權重：16支來源不同train影片</h1><p>左原片、右預測；青色P=場上球員、綠色B=球、黃色R=籃圈。逐幀框號不是追蹤ID；目前未追蹤、不裁切、不遮罩。辨識球員不等於辨識投籃者或持球者。</p><p>來源與五張訓練圖分離，但這些是既有開發診斷樣本，不是正式獨立test。沒有逐物件真值，不能計算precision/recall/mAP；有框不代表框對，未檢出不代表沒有物件。</p><p>可用下方按鈕調整所有影片播放速度，先看完整影片再查固定幀。原始validation/test未讀、未訓練、不改MixNet-I3D或特徵快取。</p><button onclick="document.querySelectorAll('video').forEach(v=>v.playbackRate=.25)">0.25倍速</button><button onclick="document.querySelectorAll('video').forEach(v=>v.playbackRate=1)">正常速度</button>'''+''.join(cards)+'''<p><a href="summary.json">自動統計（不是效能指標）</a> · <a href="human_review.json">尚未填寫的可視化複查紀錄</a></p></html>'''
        (output/'index.html').write_text(page,encoding='utf-8')
        # Four overview pages: three fixed stills per video for limited AI screenshot triage.
        for start in range(0,len(samples),4):
            tiles=[]
            for sample in samples[start:start+4]:
                sid=sample['sample_id'];n=sample['frame_count']
                for q in (1,5,9):tiles.append(cv2.imread(str(output/'frames'/f'{sid}_f{n*q//10:06d}.png')))
            rows=[np.hstack(tiles[j:j+2]) for j in range(0,len(tiles),2)]
            require(cv2.imwrite(str(output/f'overview_{start//4+1}.png'),np.vstack(rows)),'Overview write failed')
        save(output/'run_state.json',{'status':'completed','videos_completed':len(results),'frames':frames})
        print(json.dumps({'output':str(output),'videos':len(results),'frames':frames,'per_class':totals,
                          'original_video_access':access,'precision_recall_mAP_available':False},indent=2),flush=True)
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','videos_completed':len(results),'error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__':main()
