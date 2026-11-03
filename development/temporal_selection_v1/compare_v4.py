"""Refine19 cached v3 diagnostics; native re-decode only changed cases for pixel verification."""
from dataclasses import asdict
from pathlib import Path
import argparse
import html
import shutil
import tempfile

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger, SplitGuard
from development.yolo_train_probe_v1.probe import read, save, sha, require
from .probe import FOLDER, ARTIFACTS, encode, frame_pages
from .frozen_probe_v2 import decode_new, boundary_indices
from .compare_v3 import signatures
from .selector_v3 import analyze_v3, ScenePolicyV3
from .selector_v4 import BoundaryPolicy, refine_analysis, analyze_v4

PARENT=ARTIFACTS/'exposure_tail_v3_r7cwrftz'


def execute(dry_run=False):
    parent=read(PARENT/'report.json');require(parent['status']=='completed','Incomplete parent')
    for name in ('selector.py','selector_v2.py','selector_v3.py'):
        require(sha(FOLDER/name)==sha(PARENT/'inputs'/name),'Parent method changed: '+name)
    require(asdict(ScenePolicyV3())==read(PARENT/'policy_v3.json'),'Parent policy changed')
    ledger=AccessLedger();guard=SplitGuard('train',ledger);policy=BoundaryPolicy();plan=[]
    for old in parent['videos']:
        require(old['original_split']=='train' and old['video_id'] in guard.allowed,'Nontrain before open')
        require(guard.identity(old['video_id'])==old['source_identity']['before'],'Parent source changed')
        new=refine_analysis(old['v3'],policy)
        plan.append((old,new,signatures(old['v3'])!=signatures(new)))
    changed=[r['sample_id'] for r,_,c in plan if c]
    if dry_run:
        return {'cached_cases':len(plan),'changed_cases':changed,'source_opens':0,'writes':False,'training_started':False}
    output=Path(tempfile.mkdtemp(prefix='boundary_refine_v4_',dir=ARTIFACTS))
    for folder in ('inputs','previews','contacts','native_frames'):(output/folder).mkdir()
    protected={**parent['protected_sha256'],str(PARENT/'report.json'):sha(PARENT/'report.json')}
    for name in ('selector.py','selector_v2.py','selector_v3.py','selector_v4.py','compare_v4.py'):
        source=FOLDER/name;protected[str(source)]=sha(source);shutil.copy2(source,output/'inputs'/name)
    save(output/'policy.json',asdict(policy))
    save(output/'sampling_plan.json',{'parent_run':str(PARENT),'all_cases':[r['sample_id'] for r,_,_ in plan],
        'changed_cases_to_redecode':changed,'reason':'All19 cached pixel-metric records refined; only changed records need fresh native preview and composition replay',
        'not_blind_evaluation':True,'labels_or_event_references_not_inputs':True,'training_started':False})
    save(output/'run_state.json',{'status':'running','records_completed':0})
    print('OUTPUT:',output,flush=True);rows=[];cards=[];cv2.setNumThreads(1)
    try:
        for old,new,change in plan:
            sid=old['sample_id'];n=old['frame_count'];fps=old['fps_rational']
            original=f'previews/{sid}_original.mp4';src=PARENT/old['original_preview']
            protected[str(src)]=sha(src);shutil.copy2(src,output/original)
            frames=None;identity=old['source_identity'];contacts=[];native=[]
            if change:
                frames,actual_fps,identity=decode_new({**guard.allowed[old['video_id']],'sample_id':sid},guard)
                require(len(frames)==n and actual_fps==fps and identity==old['source_identity'],'Native parent changed')
                require(analyze_v3(frames)==old['v3'],'Native v3 replay changed')
                require(analyze_v4(frames,policy)==new,'Cached refinement differs from native composition')
                points=set(boundary_indices(n,old['v3'])+boundary_indices(n,new))
                for d in new['boundary_refinement']:
                    for p in ([d['cut_frame']] if 'cut_frame' in d else (d.get('proposed_change_span') or [])):
                        points.update(range(max(0,p-3),min(n,p+4)))
                points=sorted(points|set(np.linspace(0,n-1,8,dtype=int).tolist()))
                (output/'native_frames'/sid).mkdir()
                for i in points:
                    rel=f'native_frames/{sid}/f{i:06d}.png';require(cv2.imwrite(str(output/rel),frames[i]),'PNG failed')
                    require(np.array_equal(cv2.imread(str(output/rel)),frames[i]),'PNG changed pixels')
                    native.append({'frame_index':i,'path':rel,'sha256':sha(output/rel)})
                contacts=['contacts/'+p for p in frame_pages(frames,points,output/'contacts',sid,set(new['hard_cut_candidates']))]
            previews=[]
            for scene in new['contiguous_scene_candidates']:
                if not scene['length_eligible_for_preview']:continue
                a,b=scene['start_frame'],scene['end_frame_exclusive'];rel=f'previews/{sid}_v4_{a}_{b}.mp4'
                if frames is not None:encode(frames[a:b],output/rel,fps)
                else:
                    prior=next(p for p in old['scene_previews'] if p['start_frame']==a and p['end_frame_exclusive']==b)
                    src=PARENT/prior['path'];protected[str(src)]=sha(src);shutil.copy2(src,output/rel)
                previews.append({**scene,'path':rel})
            row={'sample_id':sid,'video_id':old['video_id'],'source_group':old['source_group'],'original_split':'train',
                 'frame_count':n,'fps_rational':fps,'v3':old['v3'],'v4':new,'changed':change,
                 'native_redecoded':change,'native_composition_matches_cached_refinement':True if change else None,
                 'source_identity':identity,'original_preview':original,'scene_previews':previews,
                 'contact_pages':contacts,'native_inspection_frames':native}
            rows.append(row)
            details=''.join(f'<img src="{p}" loading="lazy">' for p in contacts)
            videos=''.join(f'<label>v4候選 [{p["start_frame"]},{p["end_frame_exclusive"]})<br><video controls preload="metadata" src="{p["path"]}"></video></label>' for p in previews)
            cards.append(f'<section id="{sid}"><h2>{sid}（{"變更" if change else "不變"}）</h2><p>v3：{html.escape(str(signatures(old["v3"])))}<br>v4：{html.escape(str(signatures(new)))}</p><div><label>原片（有損預覽）<br><video controls preload="metadata" src="{original}"></video></label>{videos}</div><details><summary>原生邊界影格（僅變更案例重新解碼）</summary>{details}</details></section>')
            save(output/'run_state.json',{'status':'running','records_completed':len(rows)})
            print(sid,'CHANGED' if change else 'same',signatures(new),flush=True)
        require(all(sha(p)==h for p,h in protected.items()),'Protected input changed')
        access=ledger.statistics();require(access['train']['opens']==len(changed) and access['validation']['opens']==access['test']['opens']==0,'Unexpected source access')
        by={r['sample_id']:r for r in rows}
        checks={'flashes_identical_all19':all(r['v4']['exposure_flash_candidates']==r['v3']['exposure_flash_candidates'] for r in rows),
                'T004_full_scene_still_kept':signatures(by['T004']['v4'])['scenes']==[(0,146)],
                'four_prior_verified_cuts_retained':all(c in by[s]['v4']['hard_cut_candidates'] for s,c in [('Y010',162),('Y011',44),('Y012',142),('T012',10)]),
                'all_short_spans_accounted':all(sum(s['length'] for s in r['v4']['all_segments_including_short'])==r['frame_count'] for r in rows),
                'no_automatic_target_or_training':all(not r['v4']['target_event_selected'] and not r['v4']['automatic_training_use_enabled'] for r in rows)}
        report={'status':'completed','parent_run':str(PARENT),'videos':rows,'protected_sha256':protected,
                'summary':{'cached_records_compared':len(rows),'changed_samples':changed,'unchanged_records':sum(not r['changed'] for r in rows)},
                'regression_checks':checks,'original_access':access,'sequential_frames_decoded':sum(r['frame_count'] for r in rows if r['native_redecoded']),
                'not_blind_evaluation':True,'classification_metrics_used':False,'parameter_sweep_used':False,
                'training_started':False,'feature_cache_modified':False,'semantic_graphic_mask_implemented':False,
                'target_event_selection_implemented':False,'automatic_training_use_videos':0}
        save(output/'report.json',report)
        page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v4邊界細修：仍為場景候選</title><style>body{font:18px system-ui;line-height:1.8;max-width:1400px;margin:30px auto;padding:0 20px}img{max-width:100%}video{width:400px;max-width:100%}section{border-top:1px solid #ccd;padding:20px 0}section div{display:flex;gap:16px;flex-wrap:wrap}</style><h1>v4邊界細修：寬搜尋區不再直接冒稱圖卡</h1><p>固定19筆已看過的v3像素分數細修；僅變更案例重開原train影片，重播v3及v4確認與快取分析一致。不是新盲測、不用投籃類別或人工時間選段，也不是完整圖卡分類器。</p><p>獨立突變可提出單一切鏡；多幀變化裁掉滑動視窗前後的低活動邊緣。兩塊間隔只有持續變化才合併，不因片段短就合併。局部變化也可能是動作／移鏡，恢復幀只是假設；模糊情況保留未驗證，不接訓練。所有短場景／變化區也完整列在JSON，不藉省略短片段宣稱排除成功。</p><p><a href="policy.json">未校準工程預設</a> · <a href="report.json">全部場景／回歸／SHA</a> · <a href="AI_review.html">AI初審（若有）</a></p>'''+''.join(cards)+'</html>'
        (output/'index.html').write_text(page,encoding='utf-8')
        save(output/'run_state.json',{'status':'completed','records_completed':len(rows),'regression_checks':checks})
        print(checks,flush=True);return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','records_completed':len(rows),'error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();print(execute(args.dry_run))
