"""Seven train-only development examples: v1/v2 scene previews and pending event references."""
from dataclasses import asdict
from pathlib import Path
import argparse
import html
import shutil
import tempfile

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger,SplitGuard
from development.yolo_train_probe_v1.probe import read,save,sha,require,within
from .probe import ROOT,FOLDER,ARTIFACTS,select_samples,decode,encode,frame_pages
from .selector_v2 import ScenePolicy,analyze_v2

CASES=('Y005','Y010','Y011','Y012','Y013','Y018','Y019')
DEFAULT_PARENT=ARTIFACTS/'temporal_probe_v1__lwxmu8j'
# Display-only inspection windows from prior train diagnostics, NOT algorithm parameters.
DISPLAY_RANGES={'Y005':(52,66),'Y010':(48,100),'Y011':(40,50),'Y012':(28,45),
                'Y013':(55,68),'Y018':(25,90),'Y019':(82,126)}


def inspection_indices(n,result,sid):
    points=set(range(0,n,6))|{n-1}
    a,b=DISPLAY_RANGES[sid];points.update(range(a,min(n,b),2))
    for c in result['hard_cut_candidates']:points.update(range(max(0,c-2),min(n,c+3)))
    for region in result['gradual_change_regions']+result['exposure_flash_candidates']:
        for c in (region['start_frame'],region['end_frame_exclusive']):points.update(range(max(0,c-2),min(n,c+3)))
    if sid=='Y012':points.update(range(134,n))
    return sorted(points)


def draw_timeline(v1,v2,n,path):
    canvas=np.full((230,1200,3),245,np.uint8)
    def x(i):return 20+round(1160*i/n)
    cv2.putText(canvas,'V1: RED hard cuts; BLUE old proposed interval',(20,22),cv2.FONT_HERSHEY_SIMPLEX,.49,(20,20,20),1)
    cv2.line(canvas,(20,65),(1180,65),(60,60,60),1)
    for c in v1['cut_frame_candidates']:cv2.line(canvas,(x(c),35),(x(c),75),(20,20,220),2)
    p=v1['proposed_contiguous_interval']
    if p:cv2.rectangle(canvas,(x(p['start_frame']),78),(x(p['end_frame_exclusive']),91),(220,120,30),-1)
    cv2.putText(canvas,'V2: RED cuts; BLUE possible gradual change; YELLOW exposure (kept)',(20,121),cv2.FONT_HERSHEY_SIMPLEX,.47,(20,20,20),1)
    cv2.line(canvas,(20,165),(1180,165),(60,60,60),1)
    for r in v2['gradual_change_regions']:cv2.rectangle(canvas,(x(r['start_frame']),140),(x(r['end_frame_exclusive']),173),(220,120,30),-1)
    for r in v2['exposure_flash_candidates']:cv2.rectangle(canvas,(x(r['start_frame']),140),(max(x(r['start_frame'])+3,x(r['end_frame_exclusive'])),173),(0,200,230),-1)
    for c in v2['hard_cut_candidates']:cv2.line(canvas,(x(c),135),(x(c),180),(20,20,220),2)
    cv2.putText(canvas,'All flags are unvalidated; no target-event choice; zero-based [start,end).',(20,216),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1)
    require(cv2.imwrite(str(path),canvas),'Timeline failed')


def render(output,rows):
    cards=[]
    for row in rows:
        sid=row['sample_id'];v=row['v2'];pieces=[]
        for preview in row['scene_previews']:
            pieces.append(f'<label>v2連續鏡頭 [{preview["start_frame"]},{preview["end_frame_exclusive"]})，非目標確認<br><video controls preload="metadata" src="{preview["path"]}"></video></label>')
        old=f'<label>v1候選<br><video controls preload="metadata" src="{row["v1_preview"]}"></video></label>' if row['v1_preview'] else '<p>v1無長候選。</p>'
        cards.append(f'<section id="{sid}"><h2>{sid}</h2><p>v2硬切鏡：{html.escape(str(v["hard_cut_candidates"]))}；閃光：{html.escape(str([(r["start_frame"],r["end_frame_exclusive"]) for r in v["exposure_flash_candidates"]]))}；漸變候選：{html.escape(str([(r["start_frame"],r["end_frame_exclusive"]) for r in v["gradual_change_regions"]]))}。</p><img src="{row["timeline"]}"><div><label>原片<br><video controls preload="metadata" src="{row["original_preview"]}"></video></label>{old}{"".join(pieces)}</div><details><summary>原生逐幀／邊界檢查</summary>'+''.join(f'<img loading="lazy" src="{p}">' for p in row['contact_pages'])+'</details></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v2：分鏡／漸變／閃光與待確認目標</title><style>body{font:18px system-ui;line-height:1.8;max-width:1450px;margin:30px auto;padding:0 20px}img{max-width:100%}video{width:400px;max-width:100%;background:#111}section{border-top:1px solid #ccd;padding:20px 0}section div{display:flex;gap:18px;flex-wrap:wrap}</style><h1>7支train診斷：v1／v2分鏡對照</h1><p>v2取消木地板色排除規則，分開提出硬切鏡、短曝光與多幀漸變。閃光像素保留、不跨鏡頭拼接；所有場景候選都保留，不用最早或最長鏡頭自選目標。漸變也可能是快速外觀變化，不保證是動畫。</p><p>Y012後段罰球目標身分已由使用者確認；精確邊界與結果完整性未確認。其他事件仍待核對。演算法只接影格，沒有讀ft1、Make/Miss、事件參考或人工時間。這7支是已看過的train開發案例，不是盲測。</p><p>候選最短64幀僅控制預覽；短鏡頭仍在JSON保留，不因長度認定它不是目標。這不是完整投籃選段器，沒有重訓、改原特徵或開分類validation/test；自動訓練使用=0。</p><p><a href="report.json">逐幀分數與全部場景</a> · <a href="event_reference_draft.json">目標事件／時間參考草稿</a> · <a href="AI_review.html">AI初審與需要確認的地方（若有）</a></p>'''+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')


def execute(parent=DEFAULT_PARENT,dry_run=False):
    parent=Path(parent).resolve();require(parent.is_relative_to(ARTIFACTS.resolve()),'Foreign v1 run')
    old=read(within(parent,'report.json'));require(old['status']=='completed','Incomplete v1 run')
    manifest=read(within(parent,'inputs/audit_manifest.json'));ledger=AccessLedger();guard=SplitGuard('train',ledger)
    all_samples=select_samples(manifest,guard.allowed);samples=[s for s in all_samples if s['sample_id'] in CASES]
    require(tuple(s['sample_id'] for s in samples)==CASES,'Fixed development sample selection changed')
    clarification=read(within(parent,'human_target_event_clarification.json'))
    require(clarification['sample_id']=='Y012' and clarification['target_event_identity_human_confirmed']
            and not clarification['exact_start_end_frames_human_confirmed'],'Y012 scoped clarification changed')
    if dry_run:return {'videos':7,'original_video_opens':0,'files_written':False,'training_started':False}
    output=Path(tempfile.mkdtemp(prefix='temporal_probe_v2_',dir=ARTIFACTS))
    for name in ('inputs','previews','contacts','native_frames','timelines'):(output/name).mkdir()
    protected={str(parent/'report.json'):sha(parent/'report.json'),
               str(parent/'human_target_event_clarification.json'):sha(parent/'human_target_event_clarification.json')}
    protected.update(old['protected_input_sha256'])
    for filename in ('probe_v2.py','selector_v2.py','selector.py','probe.py'):
        source=FOLDER/filename;shutil.copy2(source,output/'inputs'/filename)
    save(output/'inputs/parent_report.json',old)
    save(output/'inputs/Y012_human_target_clarification.json',clarification)
    policy=ScenePolicy();save(output/'policy.json',asdict(policy));save(output/'run_state.json',{'status':'running','videos_completed':0})
    # Known prior user range is approximate and is a review reference, never an algorithm input.
    old_provenance=ROOT/'development/yolo_audit_v1/artifacts/audit_seed20261024_7_aaaj0r/outputs/ai_preliminary_review_v1_27g4Gz/review_provenance.json'
    provenance=read(old_provenance);protected[str(old_provenance)]=sha(old_provenance)
    historical=[e for e in provenance.get('human_review_updates',[]) if e.get('sample_id')=='Y019']
    require(historical,'Prior Y019 scoped human observation missing')
    save(output/'inputs/Y019_prior_human_observations.json',historical)
    # Preserve entire provenance so the exact event key is auditable even on older schemas.
    save(output/'inputs/prior_human_review_provenance.json',provenance)
    cv2.setNumThreads(1);rows=[];events=[];before={r['sample_id']:r for r in old['videos']}
    print('OUTPUT:',output,flush=True)
    try:
        for sample in samples:
            sid=sample['sample_id'];frames,identity=decode(sample,guard);n=len(frames)
            result=analyze_v2(frames,policy)  # No sample ID, label, type, human range, or reference is passed.
            original=f'previews/{sid}_original.mp4';src=within(parent,before[sid]['original_preview']);shutil.copy2(src,output/original);protected[str(src)]=sha(src)
            v1preview=None
            if before[sid]['proposal_preview']:
                src=within(parent,before[sid]['proposal_preview']);v1preview=f'previews/{sid}_v1.mp4';shutil.copy2(src,output/v1preview);protected[str(src)]=sha(src)
            previews=[]
            for s in result['contiguous_scene_candidates']:
                if not s['length_eligible_for_preview']:continue
                rel=f'previews/{sid}_v2_{s["start_frame"]}_{s["end_frame_exclusive"]}.mp4'
                encode(frames[s['start_frame']:s['end_frame_exclusive']],output/rel,sample['fps_rational'])
                previews.append({**s,'path':rel})
            points=inspection_indices(n,result,sid);(output/'native_frames'/sid).mkdir()
            native={}
            for i in points:
                rel=f'native_frames/{sid}/f{i:06d}.png';require(cv2.imwrite(str(output/rel),frames[i]),'Native review frame failed')
                require(np.array_equal(cv2.imread(str(output/rel)),frames[i]),'Native PNG round trip changed pixels');native[str(i)]={'path':rel,'sha256':sha(output/rel)}
            sheets=frame_pages(frames,points,output/'contacts',sid,set(result['hard_cut_candidates']))
            t=f'timelines/{sid}.png';draw_timeline(before[sid]['analysis'],result,n,output/t)
            rows.append({'sample_id':sid,'video_id':sample['video_id'],'source_group':sample['source_group'],
                'original_split':'train','frame_count':n,'source_identity':identity,'v1':before[sid]['analysis'],'v2':result,
                'original_preview':original,'v1_preview':v1preview,'scene_previews':previews,'timeline':t,
                'contact_pages':['contacts/'+x for x in sheets],'native_inspection_frames':native,
                'preview_is_lossy_not_training_input':True})
            events.append({'sample_id':sid,'video_id':sample['video_id'],
                'target_event_identity':'later_free_throw_after_transition' if sid=='Y012' else None,
                'target_event_identity_human_confirmed':sid=='Y012','target_event_source':'inputs/Y012_human_target_clarification.json' if sid=='Y012' else None,
                'AI_proposed_target_interval':None,'release_frame':None,'ball_at_rim_frame':None,'result_visible_frame':None,
                'exact_boundaries_human_confirmed':False,'event_completeness_human_confirmed':False,
                'training_use_approved':False})
            save(output/'run_state.json',{'status':'running','videos_completed':len(rows)})
            print(sid,{'cuts':result['hard_cut_candidates'],'flash':[(x['start_frame'],x['end_frame_exclusive']) for x in result['exposure_flash_candidates']],
                'gradual':[(x['start_frame'],x['end_frame_exclusive']) for x in result['gradual_change_regions']],
                'segments':[(x['start_frame'],x['end_frame_exclusive']) for x in result['contiguous_scene_candidates']]},flush=True)
        access=ledger.statistics();require(access['train']['opens']==7 and access['validation']['opens']==access['test']['opens']==0,'Unexpected source access')
        require(all(sha(p)==h for p,h in protected.items()),'Protected source/baseline changed')
        report={'schema':'scene_segmentation_v2_seven_train_development_cases','status':'completed','videos':rows,'policy':asdict(policy),
                'original_video_access':access,'protected_input_sha256':protected,'training_started':False,
                'feature_cache_modified':False,'original_labels_modified':False,'automatic_training_use_videos':0,
                'semantic_target_event_selection_implemented':False,'input_class_metadata_used_by_algorithm':False,
                'known_development_examples_not_blind_evaluation':True,
                'display_inspection_windows_are_case_specific_not_algorithm_parameters':DISPLAY_RANGES,
                'AI_review_status':'pending'}
        save(output/'report.json',report);save(output/'event_reference_draft.json',{'origin':'pending_reference_not_full_human_truth','videos':events})
        render(output,rows);save(output/'run_state.json',{'status':'completed','videos_completed':7});return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','videos_completed':len(rows),'error':f'{type(exc).__name__}: {exc}'});raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent-run',default=str(DEFAULT_PARENT));p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    print(execute(a.parent_run,a.dry_run))


if __name__=='__main__':main()
