"""Frozen v4 replay vs v5 on the fixed100 +19 earlier train development cases."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import html
import json
import shutil
import tempfile
import time

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger,SplitGuard
from development.yolo_train_probe_v1.probe import read,save,sha,require
from .probe import FOLDER,ARTIFACTS,encode
from .batch_probe_v1 import cache_inventory,plan_input,contact
from .frozen_probe_v2 import decode_new,boundary_indices
from .compare_v3 import signatures
from .selector_v4 import BoundaryPolicy,analyze_v4
from .selector_v5 import ExposurePolicyV5,analyze_v5

OLD=ARTIFACTS/'boundary_refine_v4_se8f2bo6'
BATCH=ARTIFACTS/'batch100_v4_hdz2h_f4'
PREMEASURED_DIAGNOSTIC_SOURCE_OPENS=7


def load_cases():
    old=read(OLD/'report.json');batch=read(BATCH/'manifest.json')
    require(old['status']==batch['status']=='completed','Incomplete fixed regression parents')
    require(asdict(BoundaryPolicy())==read(OLD/'policy.json')==read(BATCH/'policy.json'),'v4 policy changed')
    for name in ('selector.py','selector_v2.py','selector_v3.py','selector_v4.py'):
        require(sha(FOLDER/name)==sha(OLD/'inputs'/name)==sha(BATCH/'inputs'/name),'Frozen v4 implementation changed: '+name)
    cases=[(OLD,r) for r in old['videos']]+[(BATCH,r) for r in batch['videos']]
    require(len(cases)==119 and len({r['video_id'] for _,r in cases})==119,'Expected fixed119 unique cases')
    return cases,batch


def new_flashes(old,new):
    coords={(e['start_frame'],e['end_frame_exclusive']) for e in old['exposure_flash_candidates']}
    return [e for e in new['exposure_flash_candidates'] if (e['start_frame'],e['end_frame_exclusive']) not in coords]


def render(output,rows,summary,checks,pulse_pages):
    cards=[]
    for r in rows:
        sid=r['sample_id'];s4=signatures(r['v4']);s5=signatures(r['v5'])
        detail=''.join(f'<a href="{p}"><img loading="lazy" src="{p}"></a>' for p in r['contact_pages'])
        vids=''.join(f'<label>v5場景 [{p["start_frame"]},{p["end_frame_exclusive"]})，不是目標認證<br><video controls preload="none" src="{p["path"]}"></video></label>' for p in r['scene_previews'])
        cards.append(f'<section id="{sid}"><h2>{sid}：'+('變更' if r['signature_changed'] else '場景／曝光座標不變')+f'</h2><p>{html.escape(r["video_id"])}<br>v4：{html.escape(str(s4))}<br>v5：{html.escape(str(s5))}<br>保守輸入計畫：{r["v5_input_plan"]["mode"]}／{r["v5_input_plan"]["reason"]}；訓練採用0。</p><div><label>原片（有損檢視，非特徵輸入）<br><video controls preload="none" src="{r["original_preview"]}"></video></label>{vids}</div><details><summary>原生曝光／邊界抽幀（僅變更案例）</summary>{detail}</details></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v5：局部曝光修正／119案例回歸</title><style>body{font:17px system-ui;line-height:1.8;max-width:1400px;margin:30px auto;padding:0 18px}img{max-width:100%}video{width:400px;max-width:100%}section{border-top:1px solid #ccd;padding:18px 0}section div{display:flex;gap:16px;flex-wrap:wrap}.warn{background:#fff1cb;padding:16px}pre{white-space:pre-wrap}</style><h1>v5：局部曝光修正，不是自動投籃裁段</h1><p class="warn">100支與舊19支都是已看過的train開發案例。不是新盲測、不是分類準確率。先逐項重現凍結v4，再跑固定v5；不做參數搜尋，不因結果換片。保留所有原片像素與原三窗口，未訓練或重建特徵，不碰分類validation/test及TANet。</p><p>原v3全畫面曝光規則與v4數值不動。補充3×4區塊亮度脈衝：至少兩個四鄰接區塊各自超過原0.10亮度峰值門檻，再以原4幀上限／2幀恢復持續／外觀相關、hist與亮度回復檢查。未恢復或不同場景不抑制切鏡；仍有相似短鏡頭／移動光源等混淆風險。只補曝光種子，不修特寫／圖卡漏切，不選第一段或最長段。</p><p><a href="report.json">119筆新舊逐幀／回歸／SHA</a> · <a href="sampling_plan.json">固定案例</a> · <a href="policy_v5.json">v5工程設定</a> · <a href="AI_review.html">AI視覺初審（產生後）</a></p><pre>'''+html.escape(json.dumps({'summary':summary,'checks':checks},ensure_ascii=False,indent=2))+'</pre><h2>新增曝光：每列前幀／起點／返回幀／返回後一幀</h2>'+''.join(f'<a href="{p}"><img loading="lazy" src="{p}"></a>' for p in pulse_pages)+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')


def execute(dry_run=False):
    cases,batch=load_cases();ledger=AccessLedger();guard=SplitGuard('train',ledger)
    for _,r in cases:
        require(r['video_id'] in guard.allowed and r['original_split']=='train','Nontrain before open')
        require(guard.identity(r['video_id'])==r['source_identity']['before'],'Parent source changed')
    policy=ExposurePolicyV5();old_policy=asdict(BoundaryPolicy())
    require(all(asdict(policy)[k]==v for k,v in old_policy.items() if k!='version'),'Existing thresholds changed')
    if dry_run:return {'fixed_cases':119,'fixed_batch_cases':100,'earlier_cases':19,'source_opens':0,'writes':False,'training_started':False}
    output=Path(tempfile.mkdtemp(prefix='local_exposure_v5_',dir=ARTIFACTS))
    for name in ('inputs','previews','contacts','pulses'):(output/name).mkdir()
    protected=dict(batch['protected_sha256'])
    for parent,name in ((OLD,'report.json'),(BATCH,'manifest.json'),(BATCH,'AI_review.json'),(BATCH,'sampling_plan.json')):
        protected[str(parent/name)]=sha(parent/name)
    for name in ('selector.py','selector_v2.py','selector_v3.py','selector_v4.py','selector_v5.py','compare_v5.py','batch_probe_v1.py'):
        p=FOLDER/name;protected[str(p)]=sha(p);shutil.copy2(p,output/'inputs'/name)
    test=FOLDER/'tests/test_selector_v5.py';protected[str(test)]=sha(test);shutil.copy2(test,output/'inputs'/test.name)
    save(output/'policy_v4.json',old_policy);save(output/'policy_v5.json',asdict(policy))
    save(output/'sampling_plan.json',{'cases':[{k:r[k] for k in ('sample_id','video_id','source_group')}|{'parent_run':str(p)} for p,r in cases],
         'all119_known_development_cases_no_replacement':True,'new_source_blind_evaluation':False,
         'labels_or_classifier_outputs_used':False,'parameter_sweep_used':False,
         'v5_settings_fixed_before_this_replay':True,'training_started':False,
         'separate_prior_cause_measurement_train_opens':PREMEASURED_DIAGNOSTIC_SOURCE_OPENS})
    before=cache_inventory();save(output/'inputs/cache_inventory_before.json',before)
    save(output/'run_state.json',{'status':'running','records_completed':0})
    print('OUTPUT:',output,flush=True);cv2.setNumThreads(1);started=time.monotonic()
    rows=[];pulse_strips=[];pulse_pages=[];pulse_log=[]
    try:
        for parent,old in cases:
            sid=old['sample_id'];sample=guard.allowed[old['video_id']]
            frames,fps,identity=decode_new(sample,guard)
            require(identity==old['source_identity'] and len(frames)==old['frame_count'],'Native parent geometry/identity changed')
            v4=analyze_v4(frames);require(v4==old['v4'],'Frozen v4 native replay differs: '+sid)
            v5=analyze_v5(frames,policy);added=new_flashes(v4,v5);changed=signatures(v4)!=signatures(v5)
            original=f'previews/{sid}_original.mp4';src=parent/old['original_preview']
            protected[str(src)]=sha(src);shutil.copy2(src,output/original)
            pages=[];previews=[]
            if changed:
                points=set(boundary_indices(len(frames),v4)+boundary_indices(len(frames),v5))
                for e in added:
                    a,b=e['start_frame'],e['end_frame_exclusive'];points.update(range(max(0,a-2),min(len(frames),b+3)))
                    indices=[a-1,a,b,b+1]
                    require(all(0<=i<len(frames) for i in indices),'Invalid recovered pulse preview')
                    tiles=[]
                    for j,i in enumerate(indices):
                        tile=np.zeros((264,320,3),np.uint8);tile[24:]=frames[i]
                        cv2.putText(tile,f'{sid} pulse[{a},{b}) f{i}',(4,17),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1);tiles.append(tile)
                    pulse_strips.append(np.hstack(tiles))
                    page_number=len(pulse_pages)+1
                    pulse_log.append({'sample_id':sid,'start_frame':a,'end_frame_exclusive':b,
                                      'frame_indices':indices,'page':f'pulses/pulse_{page_number:02d}.png',
                                      'row_on_page':len(pulse_strips)-1})
                    if len(pulse_strips)==4:
                        rel=f'pulses/pulse_{page_number:02d}.png';require(cv2.imwrite(str(output/rel),np.vstack(pulse_strips)),'Pulse panel failed')
                        pulse_pages.append(rel);pulse_strips=[]
                # Bounded pages of12 native frames, avoiding one unreadably tall sheet.
                ordered=sorted(points)
                for j in range(0,len(ordered),12):
                    rel=f'contacts/{sid}_{j//12+1:02d}.png';contact(frames,ordered[j:j+12],output/rel,sid);pages.append(rel)
                for scene in v5['contiguous_scene_candidates']:
                    if not scene['length_eligible_for_preview']:continue
                    a,b=scene['start_frame'],scene['end_frame_exclusive']
                    if (a,b)==(0,len(frames)):continue
                    rel=f'previews/{sid}_v5_{a}_{b}.mp4';encode(frames[a:b],output/rel,fps)
                    previews.append({**scene,'path':rel})
            old_global=v4['exposure_flash_candidates'];new_by={(e['start_frame'],e['end_frame_exclusive']):e for e in v5['exposure_flash_candidates']}
            global_preserved=all((e['start_frame'],e['end_frame_exclusive']) in new_by and
                all(new_by[(e['start_frame'],e['end_frame_exclusive'])][k]==v for k,v in e.items()) for e in old_global)
            row={'sample_id':sid,'video_id':old['video_id'],'source_group':old['source_group'],'original_split':'train',
                 'parent_run':str(parent),'frame_count':len(frames),'fps_rational':fps,'source_identity':identity,
                 'v4_native_replay_identical':True,'v4':v4,'v5':v5,'signature_changed':changed,
                 'v3_global_exposure_records_preserved':global_preserved,'new_local_exposures':added,
                 'removed_v4_hard_cuts':sorted(set(v4['hard_cut_candidates'])-set(v5['hard_cut_candidates'])),
                 'v4_input_plan':plan_input(len(frames),v4),'v5_input_plan':plan_input(len(frames),v5),
                 'original_preview':original,'scene_previews':previews,'contact_pages':pages,'training_use_approved':False}
            rows.append(row);del frames
            save(output/'run_state.json',{'status':'running','records_completed':len(rows)})
            print(f'{len(rows)}/119 {sid}: '+('CHANGED' if changed else 'same')+f' new_local={len(added)} removed_cuts={row["removed_v4_hard_cuts"]}',flush=True)
        if pulse_strips:
            rel=f'pulses/pulse_{len(pulse_pages)+1:02d}.png';require(cv2.imwrite(str(output/rel),np.vstack(pulse_strips)),'Pulse panel failed');pulse_pages.append(rel)
        after=cache_inventory();require(before==after,'Original cache inventory changed')
        require(all(sha(p)==h for p,h in protected.items()),'Protected source/history changed')
        access=ledger.statistics();require(access['train']['opens']==119 and access['validation']['opens']==access['test']['opens']==0,'Unexpected source access')
        by={r['sample_id']:r for r in rows};batch_rows=[r for r in rows if r['sample_id'].startswith('S')]
        known=('S012','S021','S026','S036','S051','S069','S094')
        # The patch does not relax recovery to force S026's rapidly moving scene to pass.
        fixed_false_wings=('S012','S021','S036','S051','S069','S094')
        checks={'v4_native_replay_all119_identical':all(r['v4_native_replay_identical'] for r in rows),
                'all_old_global_exposure_records_unchanged':all(r['v3_global_exposure_records_preserved'] for r in rows),
                'all_existing_v4_hard_cuts_retained':all(not r['removed_v4_hard_cuts'] for r in rows),
                'four_prior_verified_true_cuts_retained':all(c in by[s]['v5']['hard_cut_candidates'] for s,c in [('Y010',162),('Y011',44),('Y012',142),('T012',10)]),
                'prior_T004_and_Y013_full_scenes_retained':all([(s['start_frame'],s['end_frame_exclusive']) for s in by[i]['v5']['contiguous_scene_candidates']]==[(0,146)] for i in ('T004','Y013')),
                'all_scene_partition_frames_accounted':all(sum(s['length'] for s in r['v5']['all_segments_including_short'])==r['frame_count'] for r in rows),
                'known_six_exposure_wings_no_longer_regions':all(not by[i]['v5']['gradual_change_regions'] for i in fixed_false_wings),
                'no_automatic_event_or_training_approval':all(not r['v5']['target_event_selected'] and not r['v5']['training_use_approved'] for r in rows)}
        summary={'development_cases':119,'fixed_batch_cases':100,'earlier_regression_cases':19,
                 'signature_changed_ids':[r['sample_id'] for r in rows if r['signature_changed']],
                 'new_local_exposure_count':len(pulse_log),'new_local_exposure_videos':sum(bool(r['new_local_exposures']) for r in rows),
                 'batch_v4_input_reason_counts':dict(Counter(r['v4_input_plan']['reason'] for r in batch_rows)),
                 'batch_v5_input_reason_counts':dict(Counter(r['v5_input_plan']['reason'] for r in batch_rows)),
                 'target_case_change_regions':{sid:{'v4':[(x['start_frame'],x['end_frame_exclusive']) for x in by[sid]['v4']['gradual_change_regions']],
                                                   'v5':[(x['start_frame'],x['end_frame_exclusive']) for x in by[sid]['v5']['gradual_change_regions']]} for sid in known},
                 'removed_v4_hard_cuts':{r['sample_id']:r['removed_v4_hard_cuts'] for r in rows if r['removed_v4_hard_cuts']},
                 'primary_regression_original_access':access,'primary_regression_frames_decoded':sum(r['frame_count'] for r in rows),
                 'separate_cause_measurement_train_opens':PREMEASURED_DIAGNOSTIC_SOURCE_OPENS,
                 'elapsed_seconds':time.monotonic()-started,'classification_accuracy_computed':False,
                 'training_started':False,'new_crop_adoptions':0,'parameter_sweep_used':False}
        report={'status':'completed' if all(checks.values()) else 'completed_with_regression_failures',
                'summary':summary,'regression_checks':checks,'videos':rows,'pulse_pages':pulse_pages,'new_pulse_inspection_rows':pulse_log,
                'protected_sha256':protected,'cache_inventory_before':before,'cache_inventory_after':after,
                'not_blind_evaluation':True,'labels_or_classifier_outputs_used':False,'training_started':False,
                'feature_tensors_read_or_written':False,'TANet_modified':False,'AI_review_status':'pending',
                'automatic_training_use_videos':0,'target_event_selection_implemented':False,
                'unchanged_limitations':'Similar short cuts can imitate pulses; unconfirmed recovery remains unresolved; camera/graphic misses and event identity not solved.'}
        save(output/'report.json',report);save(output/'summary.json',summary);save(output/'pulse_index.json',pulse_log)
        render(output,rows,summary,checks,pulse_pages)
        save(output/'run_state.json',{'status':report['status'],'records_completed':119,'regression_checks':checks})
        print(json.dumps({'checks':checks,'summary':summary},ensure_ascii=False),flush=True);return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','records_completed':len(rows),'error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();print(execute(args.dry_run))
