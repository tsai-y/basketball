"""Bounded8-case temporal-retention review. No selector tuning, training, or crop adoption."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import html
import json
import shutil
import tempfile

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger,SplitGuard
from development.yolo_train_probe_v1.probe import read,save,sha,require
from .probe import FOLDER,ARTIFACTS,encode
from .frozen_probe_v2 import decode_new
from .batch_probe_v1 import cache_inventory
from .selector_v5 import ExposurePolicyV5

PARENT=ARTIFACTS/'local_exposure_v5_of7zq9p2'
SMOKE=ARTIFACTS/'adapter_smoke_v5_zisd_ekx'
# Fixed before this review. S026 is a no-eligible-candidate control, not a crop.
REQUESTS={'Y012':(37,142),'Y019':(0,89),'S002':(0,113),'S018':(0,126),
          'S025':(0,125),'S026':None,'S030':(34,146),'S052':(0,123)}
PURPOSE={'Y012':'Human identified later free throw; check event/result retention, not only coordinates',
         'Y019':'Preserve human source-start uncertainty; no forced release answer',
         'S002':'Three-point shot plus later closeup',
         'S018':'Two-point attempt plus later closeup',
         'S025':'Another two-point attempt and later camera view',
         'S026':'No single clean span reaches66 frames; unresolved recovery control',
         'S030':'Short early free-throw camera view may contain release; negative control',
         'S052':'Mid-range shot plus appearance-change/closeup at end'}
QUESTIONS=('same_labeled_event_preserved','preparation_and_release_preserved',
           'rim_and_result_context_preserved','removed_content_nonessential_to_shot')
PHASES=('preparation','release','rim','result')


def fixed_cases(guard):
    prior=read(PARENT/'report.json');require(prior['status']=='completed','Incomplete fixed v5 run')
    require(asdict(ExposurePolicyV5())==read(PARENT/'policy_v5.json'),'Frozen v5 settings changed')
    for name in ('selector.py','selector_v2.py','selector_v3.py','selector_v4.py','selector_v5.py'):
        require(sha(FOLDER/name)==sha(PARENT/'inputs'/name),'Frozen selector changed')
    by={r['sample_id']:r for r in prior['videos']};require(set(REQUESTS)<=by.keys(),'Missing fixed diagnostic cases')
    cases=[]
    for sid,interval in REQUESTS.items():
        old=by[sid];require(old['original_split']=='train' and old['video_id'] in guard.allowed,'Nontrain before open')
        require(guard.identity(old['video_id'])==old['source_identity']['before'],'Frozen source changed')
        clean={(s['start_frame'],s['end_frame_exclusive']) for s in old['v5']['all_segments_including_short']
               if not s['inside_proposed_change_span']}
        require(interval is None or interval in clean,'Request differs from existing frozen scene')
        if interval is None:
            require(not any(66<=b-a<=192 for a,b in clean),'No-candidate control unexpectedly became eligible')
        sample=guard.allowed[old['video_id']]
        cases.append({'sample_id':sid,'video_id':old['video_id'],'source_group':old['source_group'],
                      'source_identity':old['source_identity'],'frame_count':old['frame_count'],
                      'fps_rational':old['fps_rational'],'shot_type_context':sample['original_class'][:-1],
                      'candidate_interval':interval,'review_purpose':PURPOSE[sid],
                      'all_v5_clean_scene_spans':sorted(clean)})
    require(len(cases)==8 and len({c['video_id'] for c in cases})==8,'Expected fixed8, no replacement')
    return cases


def excluded_spans(n,interval):
    if interval is None:return []
    a,b=interval;require(type(a) is int and type(b) is int and 0<=a<b<=n,'Invalid half-open candidate')
    return ([ [0,a] ] if a else [])+([ [b,n] ] if b<n else [])


def sequence_pages(frames,sid,interval,folder):
    """Every frame in order; resized160x120 display tiles only, never feature input."""
    pages=[]
    for start in range(0,len(frames),64):
        tiles=[]
        for i in range(start,min(start+64,len(frames))):
            keep=None if interval is None else interval[0]<=i<interval[1]
            color=(20,150,20) if keep else ((20,20,180) if keep is False else (20,110,180))
            tile=np.zeros((140,160,3),np.uint8);tile[:20]=color
            tile[20:]=cv2.resize(frames[i],(160,120),interpolation=cv2.INTER_AREA)
            cv2.putText(tile,f'{sid} f{i} '+('?' if keep is None else ('KEEP' if keep else 'OUT')),
                        (2,14),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1);tiles.append(tile)
        while len(tiles)%8:tiles.append(np.zeros_like(tiles[0]))
        rel=f'{sid}_{start:03d}.png'
        require(cv2.imwrite(str(folder/rel),np.vstack([np.hstack(tiles[j:j+8]) for j in range(0,len(tiles),8)])), 'Sequence write failed')
        pages.append({'path':'sequence/'+rel,'first_frame':start,'end_frame_exclusive':min(start+64,len(frames)),
                      'all_frames_present':True,'display_size':[160,120]})
    return pages


def render_initial(output,rows):
    cards=[]
    for r in rows:
        sid=r['sample_id'];p=r['candidate_interval']
        crop=(f'<label>既有候選 [{p[0]},{p[1]})（未採用）<br><video controls preload="none" src="{r["candidate_preview"]}"></video></label>' if p else '<p>沒有長度合格的單一候選；不可任選最長段。</p>')
        seq=''.join(f'<a href="{x["path"]}"><img loading="lazy" src="{x["path"]}"></a>' for x in r['sequence_pages'])
        cards.append(f'<section id="{sid}"><h2>{sid}／{r["shot_type_context"]}</h2><p>原片 {r["frame_count"]}幀；候選 {p}；排除範圍 {r["excluded_spans"]}<br>{html.escape(r["review_purpose"])}</p><div><label>完整原片（有損預覽）<br><video controls preload="none" src="{r["original_preview"]}"></video></label>{crop}</div><p>逐幀查看原生PNG：<input type="range" min="0" max="{r["frame_count"]-1}" value="0" data-sid="{sid}"><span id="label{sid}">frame0</span></p><img class="native" id="img{sid}" src="native/{sid}/f000000.png"><details><summary>原片全部連續影格（縮圖，沒有跳幀）</summary>{seq}</details></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v5固定8例：事件完整性檢查</title><style>body{font:18px system-ui;line-height:1.8;max-width:1350px;margin:30px auto;padding:0 20px}img{max-width:100%}.native{width:640px;image-rendering:pixelated}video{width:480px;max-width:100%}section{border-top:1px solid #ccd;padding:20px 0}section div{display:flex;gap:18px;flex-wrap:wrap}input{width:55%}.warn{background:#fff1cc;padding:16px}</style><h1>v5固定8例：是否保留同一次投籃？</h1><p class="warn">既有目的性train案例，不是新盲測。候選在本輪看像素前固定，不調閾值、不換片、不調端點、不選最長、不標框／逐球位置；AI初審不是人工真值或訓練批准。S026無合格長段，明確作負例。Y019保留先前人類不確定，不強迫回答。未訓練／建特徵快取，不碰分類validation/test及TANet。</p><p>依序檢查同一事件、準備與出手、到框與結果所需上下文，以及被排除的內容。shot type只提供原資料的事件類型脈絡，不以Make/Miss或計分板反推球的位置／結果。沒有宣稱結果標籤盲化，因來源ID與原畫面可能透露資訊。</p><p><a href="sampling_plan.json">看像素前固定8例</a> · <a href="report.json">原生PNG／來源／SHA</a> · <a href="AI_review.html">AI初審（完成後）</a></p><p>綠色KEEP僅表示位於候選區間，紅色OUT僅表示區間外；不是好壞真值或已刪除影格。每個原片影格都在時間圖中，縮圖只供通覽；可用滑桿與原生細節頁查核。</p>'''+''.join(cards)+'''<script>document.querySelectorAll('input').forEach(e=>e.addEventListener('input',()=>{let s=e.dataset.sid,i=Number(e.value);document.getElementById('img'+s).src='native/'+s+'/f'+String(i).padStart(6,'0')+'.png';document.getElementById('label'+s).textContent='frame '+i;}));</script></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')


def execute(dry_run=False):
    ledger=AccessLedger();guard=SplitGuard('train',ledger);cases=fixed_cases(guard)
    if dry_run:return {'cases':[{k:r[k] for k in ('sample_id','candidate_interval','shot_type_context')} for r in cases],
                       'source_opens':0,'writes':False,'training_started':False}
    output=Path(tempfile.mkdtemp(prefix='event_review_v5_',dir=ARTIFACTS))
    for name in ('inputs','native','sequence','detail','previews'):(output/name).mkdir()
    protected=dict(read(SMOKE/'report.json')['protected_sha256'])
    for p in (PARENT/'report.json',PARENT/'policy_v5.json',PARENT/'AI_review.json',SMOKE/'report.json',Path(__file__)):
        protected[str(p)]=sha(p)
    shutil.copy2(Path(__file__),output/'inputs/event_review_v5.py')
    for name in ('selector_v5.py','window_adapter_v5.py','event_retention_v1.py'):
        shutil.copy2(FOLDER/name,output/'inputs'/name)
    for sid,path in [('Y012',ARTIFACTS/'temporal_probe_v2_xafv9o3e/inputs/Y012_human_target_clarification.json'),
                     ('Y019',ARTIFACTS/'event_retention_v1_vqoqjui0/human_Y019_start_uncertainty.json')]:
        protected[str(path)]=sha(path);shutil.copy2(path,output/'inputs'/f'{sid}_prior_human_scope.json')
    plan={'schema':'fixed8_v5_event_retention_diagnostic','cases':cases,'source_v5_report_sha256':sha(PARENT/'report.json'),
          'sampling':'Purposeful fixed existing examples covering four shot types plus known negative/uncertain controls',
          'maximum_cases':10,'actual_fixed_cases':8,'replace_after_review':False,'classification_accuracy_computed':False,
          'training_started':False,'production_crop_adoptions':0,'new_human_ground_truth':False}
    save(output/'sampling_plan.json',plan);before=cache_inventory();save(output/'inputs/cache_inventory_before.json',before)
    save(output/'run_state.json',{'status':'preparing','cases_completed':0});rows=[];cv2.setNumThreads(1)
    print('OUTPUT:',output,flush=True)
    try:
        for case in cases:
            sid=case['sample_id'];frames,fps,identity=decode_new(guard.allowed[case['video_id']],guard)
            require(identity==case['source_identity'] and len(frames)==case['frame_count'] and fps==case['fps_rational'],'Frozen source changed')
            (output/'native'/sid).mkdir();native=[]
            for i,f in enumerate(frames):
                rel=f'native/{sid}/f{i:06d}.png';require(cv2.imwrite(str(output/rel),f),'Native write failed')
                require(np.array_equal(cv2.imread(str(output/rel)),f),'Native PNG round trip mismatch')
                native.append({'frame_index':i,'path':rel,'sha256':sha(output/rel)})
            original=f'previews/{sid}_original.mp4';encode(frames,output/original,fps)
            interval=case['candidate_interval'];candidate=None
            if interval:
                a,b=interval;candidate=f'previews/{sid}_candidate_{a}_{b}.mp4';encode(frames[a:b],output/candidate,fps)
            seq=sequence_pages(frames,sid,interval,output/'sequence')
            rows.append({**case,'native_frames':native,'sequence_pages':seq,'original_preview':original,
                         'candidate_preview':candidate,'excluded_spans':excluded_spans(len(frames),interval),
                         'AI_review_status':'pending','training_use_approved':False})
            save(output/'run_state.json',{'status':'preparing','cases_completed':len(rows)})
            print(sid,'all native frames=',len(frames),'candidate=',interval,flush=True)
        require(all(sha(p)==h for p,h in protected.items()),'Protected source/history changed')
        after=cache_inventory();require(before==after,'Original feature cache inventory changed')
        access=ledger.statistics();require(access['train']['opens']==8 and access['validation']['opens']==access['test']['opens']==0,'Unexpected source access')
        report={'status':'prepared_for_AI_review','videos':rows,'protected_sha256':protected,
                'cache_inventory_before':before,'cache_inventory_after':after,'original_access':access,
                'native_frames_decoded':sum(r['frame_count'] for r in rows),'selector_or_thresholds_changed':False,
                'feature_tensors_read_or_written':False,'training_started':False,'TANet_modified':False,
                'all_original_frames_available_in_order':True,'automatic_crop_adoptions':0}
        save(output/'report.json',report);render_initial(output,rows)
        save(output/'run_state.json',{'status':'prepared_for_AI_review','cases_completed':8})
        return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','cases_completed':len(rows),'error':f'{type(exc).__name__}: {exc}'})
        raise


def detail_pages(run,sid,a,b):
    run=Path(run).resolve();require(run.is_relative_to(ARTIFACTS),'Foreign run')
    report=read(run/'report.json');row=next((r for r in report['videos'] if r['sample_id']==sid),None)
    require(row is not None and type(a) is int and type(b) is int and 0<=a<b<=row['frame_count'],'Invalid detail range')
    folder=Path(tempfile.mkdtemp(prefix=f'{sid}_{a}_{b}_',dir=run/'detail'));pages=[]
    for start in range(a,b,12):
        tiles=[]
        for i in range(start,min(start+12,b)):
            entry=row['native_frames'][i];require(entry['frame_index']==i and sha(run/entry['path'])==entry['sha256'],'Native changed')
            frame=cv2.imread(str(run/entry['path']));require(frame.shape==(240,320,3),'Invalid native detail')
            tile=np.zeros((264,320,3),np.uint8);tile[24:]=frame
            cv2.putText(tile,f'{sid} original f{i}',(3,17),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1);tiles.append(tile)
        while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
        p=folder/f'f{start:06d}.png';require(cv2.imwrite(str(p),np.vstack([np.hstack(tiles[j:j+4]) for j in range(0,len(tiles),4)])),'Detail write failed')
        pages.append(str(p.relative_to(run)))
    save(folder/'inspection_plan.json',{'sample_id':sid,'frame_range':[a,b],'all_native_frames_in_range':True,
         'detail_pages':pages,'new_original_video_opens':0,'display_only_not_training':True})
    return pages


def validate_assessment(assessment,row):
    require(assessment['sample_id']==row['sample_id'],'Assessment sample mismatch')
    status=assessment['status'];require(status in ('acceptable','uncertain','unacceptable'),'Invalid review status')
    questions=assessment['questions'];require(set(questions)==set(QUESTIONS),'Need four retention questions')
    require(all(v is None or type(v) is bool for v in questions.values()),'Only explicit bool/unknown question answers')
    phases=assessment['source_phase_evidence_brackets'];require(set(phases)==set(PHASES),'Need four phase brackets or unknown')
    interval=row['candidate_interval'];outside=False
    for bounds in phases.values():
        if bounds is None:continue
        require(isinstance(bounds,(list,tuple)) and len(bounds)==2 and all(type(v) is int for v in bounds) and
                0<=bounds[0]<bounds[1]<=row['frame_count'],'Invalid source evidence bracket')
        if interval and not(interval[0]<=bounds[0] and bounds[1]<=interval[1]):outside=True
    if status=='acceptable':
        require(interval is not None and all(v is True for v in questions.values()) and
                all(b is not None for b in phases.values()) and not outside,
                'Unknown/missing/outside evidence cannot pass as acceptable')
    if status=='unacceptable':require(False in questions.values() or outside,'Need explicit negative evidence for rejection')
    require(assessment.get('human_ground_truth') is False and assessment.get('training_use_approved') is False,
            'AI preliminary review cannot promote to human truth or training adoption')


def render_review(run,findings):
    run=Path(run).resolve();require(run.is_relative_to(ARTIFACTS),'Foreign run')
    report=read(run/'report.json');evidence=read(findings);rows={r['sample_id']:r for r in report['videos']}
    require(evidence['report_sha256']==sha(run/'report.json'),'Review uses another report')
    require(len(evidence['assessments'])==8 and {a['sample_id'] for a in evidence['assessments']}==set(rows),'Exactly the fixed8 cases required')
    for rel in evidence['viewed_pages']:
        require((run/rel).resolve().is_relative_to(run) and (run/rel).is_file(),'Missing/foreign viewed image')
    for a in evidence['assessments']:
        validate_assessment(a,rows[a['sample_id']]);require(set(a['evidence_pages'])<=set(evidence['viewed_pages']),'Unreviewed evidence page')
    evidence['viewed_page_sha256']={p:sha(run/p) for p in evidence['viewed_pages']}
    evidence['status_counts']=dict(Counter(a['status'] for a in evidence['assessments']))
    require(not (run/'AI_review.json').exists() and not (run/'AI_review.html').exists(),'Do not overwrite earlier reviews')
    save(run/'AI_review.json',evidence)
    label={'acceptable':'可接受（AI初審限定）','uncertain':'不確定，回退','unacceptable':'不可接受，回退'}
    table=''.join(f'<tr><td><a href="index.html#{a["sample_id"]}">{a["sample_id"]}</a></td><td>{label[a["status"]]}</td><td>{html.escape(a["reason"])}</td></tr>' for a in evidence['assessments'])
    cards=''.join(f'<section id="{a["sample_id"]}"><h2>{a["sample_id"]}：{label[a["status"]]}</h2><p>{html.escape(a["reason"])}</p><p>{html.escape(a["removed_content_observation"])}</p><pre>{html.escape(json.dumps(a["source_phase_evidence_brackets"],ensure_ascii=False))}</pre><p><a href="index.html#{a["sample_id"]}">原片／固定候選／原生滑桿</a></p>'+''.join(f'<a href="{p}"><img loading="lazy" src="{p}"></a>' for p in a['evidence_pages'])+'</section>' for a in evidence['assessments'])
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v5固定8例：事件完整性AI初審</title><style>body{font:18px system-ui;line-height:1.8;max-width:1350px;margin:30px auto;padding:0 20px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:10px}section{border-top:1px solid #ccc;padding:18px 0}.warn{background:#fff0cc;padding:16px}pre{white-space:pre-wrap}</style><h1>固定8例：事件完整性AI初審</h1><p class="warn">可接受只限這次AI視覺保留檢查，不是人工真值、分類準確率、全資料自動裁片或訓練批准。不確定／不可接受均保留原三窗口。此次不因結果調閾值、移端點或換片；未新增人工逐球標註。</p><p>'''+html.escape(evidence['review_scope'])+'</p><p>'+html.escape(evidence['conclusion'])+'</p><p><a href="index.html">全部原片與候選</a> · <a href="AI_review.json">逐題與來源座標證據</a> · <a href="sampling_plan.json">固定樣本</a></p><table><tr><th>案例</th><th>初審</th><th>理由</th></tr>'+table+'</table>'+cards+'</html>'
    (run/'AI_review.html').write_text(page,encoding='utf-8');return run/'AI_review.html'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    p.add_argument('--detail-run',type=Path);p.add_argument('--sample');p.add_argument('--start',type=int);p.add_argument('--end',type=int)
    p.add_argument('--review-run',type=Path);p.add_argument('--findings',type=Path);args=p.parse_args()
    require(not(args.detail_run and args.review_run),'Choose one action')
    if args.detail_run:print(detail_pages(args.detail_run,args.sample,args.start,args.end))
    elif args.review_run:print(render_review(args.review_run,args.findings))
    else:print(execute(args.dry_run))
