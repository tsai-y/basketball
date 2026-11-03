"""Fixed80 source-expanded native candidates and weak ball proposals; never export labels/train."""
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import html
import math
import os
import shutil
import tempfile

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger,SplitGuard
from .probe import ARTIFACTS,read,save,sha,within,require
from .ball_dataset_v2 import check as check_dataset
from .ball_compare_v2 import completed_best,PARAMETERS
from .ball_only_probe_v1 import runtime_at
from .ball_pretraining_diagnostic_v1 import map_ball_predictions
from .visual_check_expansion_v1 import labeled

SCHEMA='fixed80_source_expanded_ball_review_pool_v3'
SEED=20261027
CLASSES=('2p0','2p1','3p0','3p1','ft0','ft1','mp0','mp1')
TRAIN='detector_train'
DEV='detector_dev_validation'


def rank(text):
    return hashlib.sha256(f'ball-review-v3:{SEED}:{text}'.encode()).hexdigest()


def select_sources(rows,existing):
    require(rows and all(r['split']=='train' and str(r['excluded'])=='0' for r in rows), 'Non-excluded original train only')
    require(len({r['video_id'] for r in rows})==len(rows), 'Duplicate original video ID')
    require(existing and all(v in (TRAIN,DEV) for v in existing.values()), 'Invalid prior source reservations')
    fresh=sorted({r['source_group'] for r in rows}-set(existing),key=rank)
    require(len(fresh)>=13, 'Need at least13 previously unreserved original-train groups')
    # Reserve one third before decoding or new ball inspection; never revise existing assignments.
    held_count=max(5,len(fresh)//3);require(len(fresh)-held_count>=8, 'Insufficient training groups')
    assignments={**existing,**{g:DEV if i<held_count else TRAIN for i,g in enumerate(fresh)}}
    selected=[];train_groups=set();dev_counts=Counter()
    for cls in CLASSES:
        pool=[r for r in rows if r['source_group'] in fresh and r['original_class']==cls]
        train=[r for r in pool if assignments[r['source_group']]==TRAIN and r['source_group'] not in train_groups]
        require(train,'No distinct new train source for '+cls)
        tr=min(train,key=lambda r:(rank(r['source_group']),rank(r['video_id'])))
        train_groups.add(tr['source_group'])
        dev=[r for r in pool if assignments[r['source_group']]==DEV]
        require(dev,'No reserved development source for '+cls)
        dr=min(dev,key=lambda r:(dev_counts[r['source_group']],rank(r['source_group']),rank(r['video_id'])))
        dev_counts[dr['source_group']]+=1
        for part,r in ((TRAIN,tr),(DEV,dr)):
            selected.append({**deepcopy(r),'sample_id':f'S{len(selected)+1:03d}','detector_partition':part})
    require(len({r['video_id'] for r in selected})==16,'Duplicate selected video')
    require({r['source_group'] for r in selected if r['detector_partition']==TRAIN}.isdisjoint(
            {r['source_group'] for r in selected if r['detector_partition']==DEV}), 'New source overlap')
    return selected,assignments


def indices(n):
    require(type(n) is int and n>=10,'Invalid frame count')
    result=tuple(n*q//10 for q in (1,3,5,7,9))
    require(len(set(result))==5 and all(0<=i<n for i in result),'Invalid fixed frame positions')
    return result


def check_pending_state(row):
    require(row['human_confirmed'] is False and row['annotation_complete'] is False and row['training_ready'] is False,
            'Unreviewed candidates cannot be promoted to human labels')
    require(row['visible_ball_status']=='unreviewed' and row['no_ball_confirmed'] is False,
            'Empty model prediction is not a no-ball label')


def prepare(dataset,dry_run=False):
    dataset=Path(dataset).resolve();base=check_dataset(dataset);ledger=AccessLedger();guard=SplitGuard('train',ledger)
    selected,assignments=select_sources(list(guard.allowed.values()),base['source_assignments'])
    require(all(assignments[g]==p for g,p in base['source_assignments'].items()), 'Prior reservations changed')
    plan={'schema':SCHEMA,'source_dataset':str(dataset),'source_dataset_manifest_sha256':sha(dataset/'dataset_manifest.json'),
          'seed':SEED,'samples':selected,'source_assignments':assignments,
          'existing_reserved_sources_excluded':sorted(base['source_assignments']),
          'sampling':'Before pixel/ball inspection: hash-reserve fresh source groups; one train+one development video per8 original classes; five fixed10/30/50/70/90% positions per video',
          'model_predictions_do_not_select_frames':True,'ball_visibility_does_not_select_frames':True,
          'target_candidate_counts':{TRAIN:40,DEV:40},'class_metadata_used_for_sampling_not_YOLO_labels':True,
          'prior_RGB_training_may_have_used_these_original_train_sources':True,
          'development_scope':'Detector development drawn within original classification train, not original classification validation/test or formal untouched test',
          'unavailable_or_uncertain_frames_not_replaced_with_easier_examples':True,
          'training_started':False,'yolo_labels_exported':False,'human_labels_created':False,
          'training_ready_images':0,'classification_validation_test_used':False,'feature_cache_changed':False}
    if dry_run:return {**plan,'files_written':False,'original_video_opens':0}
    output=Path(tempfile.mkdtemp(prefix='ball_review_pool_v3_',dir=ARTIFACTS))
    for sub in ('images','large','contacts','inputs'):(output/sub).mkdir()
    shutil.copy2(__file__,output/'source_snapshot.py');save(output/'sampling_plan.json',plan)
    save(output/'inputs/base_manifest.json',base);save(output/'run_state.json',{'status':'running','videos_completed':0})
    protected={str(dataset/'dataset_manifest.json'):sha(dataset/'dataset_manifest.json')}
    records=[];identities=[];decoded=0;cv2.setNumThreads(1)
    print('OUTPUT:',output,flush=True)
    try:
        for s in selected:
            official=guard.allowed[s['video_id']]
            require(all(s[k]==official[k] for k in official),'Canonical train row mismatch')
            before=guard.identity(s['video_id']);fd=guard.open(s['video_id'],s['video_path'],'ball_v3_fixed_frame_native_candidate_pool')
            cap=None;tiles=[]
            try:
                cap=cv2.VideoCapture(f'/proc/self/fd/{fd}',cv2.CAP_FFMPEG)
                require(cap.isOpened(),'Cannot decode guarded source')
                count=cap.get(cv2.CAP_PROP_FRAME_COUNT);require(math.isfinite(count) and abs(count-round(count))<1e-6,'Invalid frame count')
                n=int(round(count));fps=cap.get(cv2.CAP_PROP_FPS);require(math.isfinite(fps) and fps>0,'Invalid FPS')
                chosen=indices(n)
                for i in range(n):
                    ok,raw=cap.read();require(ok and raw.shape==(240,320,3) and raw.dtype==np.uint8,'Invalid/truncated native frame')
                    decoded+=1
                    if i not in chosen:continue
                    iid=f'N{len(records)+1:03d}';rel=f'images/{iid}.png';path=output/rel
                    require(cv2.imwrite(str(path),raw),'Native PNG write failed')
                    require(np.array_equal(cv2.imread(str(path)),raw),'Lossless native PNG mismatch')
                    preview=f'large/{iid}.png';require(cv2.imwrite(str(output/preview),cv2.resize(raw,(960,720),interpolation=cv2.INTER_NEAREST)),'Preview failed')
                    part=s['detector_partition'];tile=np.zeros((264,320,3),np.uint8);tile[24:]=raw
                    cv2.putText(tile,f'{iid} {"train" if part==TRAIN else "dev"} f{i}',(3,16),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);tiles.append(tile)
                    records.append({'image_id':iid,'image':rel,'image_sha256':sha(path),'large_preview':preview,
                        'sample_id':s['sample_id'],'video_id':s['video_id'],'source_group':s['source_group'],
                        'source_split':'train','original_class':s['original_class'],'detector_partition':part,
                        'frame_index':i,'frame_count':n,'fps':fps,'fixed_position_tenths':(1,3,5,7,9)[chosen.index(i)],
                        'status':'pending_AI_then_human_full_image_ball_review','human_confirmed':False,
                        'annotation_complete':False,'training_ready':False,'visible_ball_status':'unreviewed',
                        'no_ball_confirmed':False})
                require(not cap.read()[0], 'Extra frames after expected end')
            finally:
                if cap is not None:cap.release()
                os.close(fd)
            after=guard.identity(s['video_id']);require(before==after, 'Original source changed')
            identities.append({'video_id':s['video_id'],'before':before,'after':after})
            while len(tiles)<6:tiles.append(np.zeros((264,320,3),np.uint8))
            require(cv2.imwrite(str(output/f'contacts/{s["sample_id"]}.png'),np.vstack([np.hstack(tiles[j:j+3]) for j in (0,3)])), 'Contact write failed')
            save(output/'run_state.json',{'status':'running','videos_completed':len(identities),'images_completed':len(records)})
            print(f'{len(identities)}/16 sources, {len(records)}/80 native candidates',flush=True)
        stats=ledger.statistics();require(stats['train']['opens']==16 and stats['validation']['opens']==stats['test']['opens']==0,'Unexpected source access')
        m={**plan,'records':records,'source_identities':identities,'access_ledger':stats,'frames_sequentially_decoded':decoded,
           'original_video_opens':16,'protected_input_sha256':protected,'status':'completed',
           'source_group_counts':dict(Counter({p:len({s['source_group'] for s in selected if s['detector_partition']==p}) for p in (TRAIN,DEV)})),
           'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_code_sha256':sha(__file__)}
        save(output/'manifest.json',m);check(output)
        cards=''.join(f'<section><h2>{s["sample_id"]} / {"train" if s["detector_partition"]==TRAIN else "development"}</h2><img loading="lazy" src="contacts/{s["sample_id"]}.png"></section>' for s in selected)
        (output/'index.html').write_text('''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>80張固定位置候選：尚未標註／訓練</title><style>body{font:18px system-ui;line-height:1.8;max-width:1200px;margin:30px auto;padding:0 20px}img{max-width:100%}section{border-top:1px solid #ccd;padding-top:15px}</style><h1>新來源球專用候選池：80張，40train／40development</h1><p>抽樣前先預留來源，8種原始類別各1支train與1支development、每支固定5位置。這不是按球清晰度或模型偵測成功挑選，無球／轉場／遮擋／不確定都留在原位置；不必現在逐框回答。</p><p>每張皆為原生320×240無損PNG。尚無球框真值、無球負例確認、訓練labels或訓練。原始分類validation/test未開啟，與既有YOLO預留來源不交疊；這仍是分類train內的偵測器開發資料，不是最終盲測。</p><p>後續AI先分批複查，弱模型空輸出不能當成無球。要匯出完整正／負樣本仍須明確確認球框或完整無球狀態；看不清就保留uncertain，不改成空標籤。不改舊模型與MixNet。</p><p><a href="manifest.json">來源與像素稽核</a> · <a href="sampling_plan.json">看圖前固定抽樣／來源預留</a> · <a href="AI_proposals/index.html">弱模型提案（若已完成）</a></p>'''+cards+'</html>',encoding='utf-8')
        save(output/'run_state.json',{'status':'completed','videos_completed':16,'images_completed':80});return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','error':f'{type(exc).__name__}: {exc}'});raise


def check(pool):
    pool=Path(pool).resolve();require(pool.is_relative_to(ARTIFACTS.resolve()),'Foreign pool')
    m=read(pool/'manifest.json');require(m['schema']==SCHEMA and m['status']=='completed','Incomplete pool')
    base=check_dataset(m['source_dataset']);guard=SplitGuard('train',AccessLedger())
    samples,assignments=select_sources(list(guard.allowed.values()),base['source_assignments'])
    require(m['samples']==samples and m['source_assignments']==assignments,'Selection/reservations changed')
    require(sha(Path(m['source_dataset'])/'dataset_manifest.json')==m['source_dataset_manifest_sha256'],'Base changed')
    require(len(m['records'])==80 and len({r['image_id'] for r in m['records']})==80,'Wrong candidate IDs')
    require(Counter(r['detector_partition'] for r in m['records'])=={TRAIN:40,DEV:40},'Wrong partitions')
    for sample in samples:
        rows=[r for r in m['records'] if r['sample_id']==sample['sample_id']]
        require(len(rows)==5 and tuple(r['frame_index'] for r in rows)==indices(rows[0]['frame_count']),'Fixed positions changed')
        require(guard.identity(sample['video_id'])==next(x['before'] for x in m['source_identities'] if x['video_id']==sample['video_id']),'Original source identity changed')
        for r in rows:
            require(r['source_split']=='train' and all(r[k]==sample[k] for k in ('video_id','source_group','original_class','detector_partition')),'Candidate source mismatch')
            check_pending_state(r);require(sha(within(pool,r['image']))==r['image_sha256'],'Native pixels changed')
    require(not m['training_started'] and not m['yolo_labels_exported'] and m['training_ready_images']==0,'Wrong phase')
    for p,h in m['protected_input_sha256'].items():require(sha(p)==h,'Protected input changed')
    return m


def propose(pool,run):
    pool=Path(pool).resolve();m=check(pool);weights,state_path,state=completed_best(run,m['source_dataset'],m['source_dataset_manifest_sha256'])
    output=pool/'AI_proposals';require(not output.exists(),'Do not overwrite existing proposals')
    import torch
    require(torch.cuda.is_available(),'CUDA required; no silent CPU inference')
    output.mkdir();(output/'panels').mkdir();runtime_at(output)
    from ultralytics import YOLO
    torch.set_num_threads(4);cv2.setNumThreads(1);model=YOLO(str(weights));require(model.names=={0:'ball'},'Wrong class')
    protected={str(pool/'manifest.json'):sha(pool/'manifest.json'),str(state_path):sha(state_path),str(weights):sha(weights)}
    draft=[];cards=[];tiles=[]
    for r in m['records']:
        raw=cv2.imread(str(pool/r['image']))
        result=model.predict(source=raw,classes=[0],device=0,save=False,verbose=False,**PARAMETERS)[0]
        pred=map_ball_predictions(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist(),0,'weak_new14_best_AI_proposal')
        panel=np.zeros((514,1280,3),np.uint8);panel[34:,:640]=cv2.resize(raw,(640,480),interpolation=cv2.INTER_NEAREST);panel[34:,640:]=labeled(raw,pred)
        cv2.putText(panel,r['image_id']+' RAW / UNREVIEWED',(4,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        cv2.putText(panel,'WEAK MODEL / EMPTY DOES NOT MEAN NO BALL',(644,23),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
        rel=f'panels/{r["image_id"]}.png';require(cv2.imwrite(str(output/rel),panel),'Proposal panel failed')
        draft.append({**deepcopy(r),'model_predictions_not_ground_truth':pred,'shown_panel':rel,'shown_panel_sha256':sha(output/rel)})
        tile=np.zeros((264,320,3),np.uint8);tile[24:]=raw
        cv2.putText(tile,f'{r["image_id"]} {"train" if r["detector_partition"]==TRAIN else "dev"} weak boxes={len(pred)}',(3,16),cv2.FONT_HERSHEY_SIMPLEX,.39,(255,255,255),1);tiles.append(tile)
        cards.append(f'<section id="{r["image_id"]}"><h2>{r["image_id"]}</h2><img loading="lazy" src="{rel}"></section>')
    pages=[]
    for start in range(0,80,8):
        rel=f'overview_{len(pages)+1}.png';require(cv2.imwrite(str(output/rel),np.vstack([np.hstack(tiles[j:j+4]) for j in (start,start+4)])),'Overview failed');pages.append(rel)
    d={'schema':SCHEMA,'origin':'weak_model_AI_drafts_not_ground_truth','pool':str(pool),'images':draft,
       'parameters':PARAMETERS,'source_code_sha256':sha(__file__),'protected_input_sha256':protected,
       'overview_pages':pages,'AI_visual_review_status':'not_yet_reviewed','human_confirmation_events':[],
       'training_ready_images':0,'human_labels_created':False,'labels_exported':False,'training_started':False,
       'empty_predictions_are_not_confirmed_negatives':True,'images_with_proposals':sum(bool(r['model_predictions_not_ground_truth']) for r in draft)}
    save(output/'review_draft.json',d)
    (output/'index.html').write_text('''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>80張弱提案：不是人工真值</title><style>body{font:18px system-ui;line-height:1.8;max-width:1400px;margin:30px auto;padding:0 20px}img{max-width:100%}section{margin:20px 0}</style><h1>80張固定候選的弱模型提案</h1><p>左原圖、右球模型提案。所有影格不論有框／無框都保留，模型空輸出不是無球真值。全數仍待AI全圖檢查與人工確認，不匯出labels、不訓練。請勿直接整批接受未檢視的弱模型輸出。</p><p><a href="review_draft.json">提案與待複查狀態</a> · <a href="AI_review.html">AI初審（若有）</a></p>'''+''.join(cards)+'</html>',encoding='utf-8')
    require(all(sha(p)==h for p,h in protected.items()),'Protected model/source changed');check(pool)
    print({'output':str(output),'images':80,'images_with_proposals':d['images_with_proposals'],'training_ready':0},flush=True);return output


def main():
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    for flag in ('prepare','check','propose'):mode.add_argument('--'+flag,action='store_true')
    p.add_argument('--dataset-dir');p.add_argument('--pool');p.add_argument('--run-dir');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.prepare:
        value=prepare(a.dataset_dir,a.dry_run)
        if a.dry_run:print({k:value[k] for k in ('target_candidate_counts','files_written','original_video_opens')});print('selected_source_groups:',sorted({r['source_group'] for r in value['samples']}))
        else:print(value)
    elif a.check:print({'images':len(check(a.pool)['records']),'training_ready':0})
    else:print(propose(a.pool,a.run_dir))


if __name__=='__main__':main()
