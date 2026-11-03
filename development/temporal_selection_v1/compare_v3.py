"""v2/v3 regression comparison on19 already-inspected train cases; no new blind-test claim."""
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
from .probe import ROOT, FOLDER, ARTIFACTS, encode, frame_pages
from .frozen_probe_v2 import decode_new, boundary_indices
from .selector_v2 import analyze_v2, ScenePolicy
from .selector_v3 import analyze_v3, ScenePolicyV3

OLD7 = ARTIFACTS/'temporal_probe_v2_xafv9o3e'
NEW12 = ARTIFACTS/'frozen_v2_new12_pst3tp7_'


def signatures(result):
    return {'cuts': result['hard_cut_candidates'],
            'gradual': [(r['start_frame'],r['end_frame_exclusive']) for r in result['gradual_change_regions']],
            'flashes': [(r['start_frame'],r['end_frame_exclusive']) for r in result['exposure_flash_candidates']],
            'scenes': [(r['start_frame'],r['end_frame_exclusive']) for r in result['contiguous_scene_candidates']]}


def load_inputs():
    old = read(OLD7/'report.json'); new = read(NEW12/'manifest.json')
    require(old['status'] == new['status'] == 'completed', 'Incomplete parent regression data')
    cases = [(OLD7,r,r['v2']) for r in old['videos']] + [(NEW12,r,r['analysis']) for r in new['videos']]
    require(len(cases) == 19 and len({r['video_id'] for _,r,_ in cases}) == 19, 'Wrong/duplicate cases')
    require(sha(FOLDER/'selector_v2.py') == sha(OLD7/'inputs/selector_v2.py'), 'v2 source changed')
    require(sha(FOLDER/'selector.py') == sha(OLD7/'inputs/selector.py'), 'Pixel descriptor changed')
    old_policy = asdict(ScenePolicy()); new_policy = asdict(ScenePolicyV3())
    require(all(new_policy[k] == v for k,v in old_policy.items() if k != 'version'),
            'Existing v2 thresholds changed')
    protected = {**old['protected_input_sha256'], **new['protected_sha256']}
    for p in (OLD7/'report.json', NEW12/'manifest.json', OLD7/'event_reference_draft.json',
              OLD7/'human_Y019_target_confirmation.json',
              ARTIFACTS/'event_retention_v1_vqoqjui0/human_Y019_start_uncertainty.json'):
        protected[str(p)] = sha(p)
    return cases, protected


def execute(dry_run=False):
    cases, protected = load_inputs(); ledger = AccessLedger(); guard = SplitGuard('train',ledger)
    for _,row,_ in cases:
        require(row['original_split'] == 'train' and row['video_id'] in guard.allowed, 'Nontrain ID before open')
        require(guard.identity(row['video_id']) == row['source_identity']['before'], 'Parent source identity changed')
    if dry_run:
        return {'videos':19,'source_opens':0,'writes':False,'training_started':False,
                'already_seen_regression_cases_not_blind_test':True}
    output = Path(tempfile.mkdtemp(prefix='exposure_tail_v3_', dir=ARTIFACTS))
    for name in ('inputs','previews','contacts','native_frames'):
        (output/name).mkdir()
    for name in ('selector.py','selector_v2.py','selector_v3.py','compare_v3.py'):
        source = FOLDER/name; protected[str(source)] = sha(source)
        shutil.copy2(source,output/'inputs'/name)
    save(output/'policy_v2.json',asdict(ScenePolicy())); save(output/'policy_v3.json',asdict(ScenePolicyV3()))
    save(output/'sampling_plan.json',{'cases':[{k:r[k] for k in ('sample_id','video_id','source_group')} for _,r,_ in cases],
         'selection':'All7 earlier cases plus all12 frozen-v2 cases, no replacement or outcome selection',
         'already_inspected_development_cases_not_blind':True,'training_started':False})
    save(output/'run_state.json',{'status':'running','videos_completed':0})
    print('OUTPUT:',output,flush=True); cv2.setNumThreads(1); rows=[]; cards=[]
    try:
        for parent,old,expected_v2 in cases:
            sid=old['sample_id']; sample={**guard.allowed[old['video_id']],'sample_id':sid}
            frames,fps,identity=decode_new(sample,guard)
            require(len(frames)==old['frame_count'] and identity==old['source_identity'],'Parent native geometry/identity changed')
            v2=analyze_v2(frames); require(v2==expected_v2,'v2 replay differs from frozen parent: '+sid)
            v3=analyze_v3(frames); s2=signatures(v2); s3=signatures(v3)
            original=f'previews/{sid}_original.mp4'; src=parent/old['original_preview']
            protected[str(src)]=sha(src); shutil.copy2(src,output/original)
            previews=[]
            for scene in v3['contiguous_scene_candidates']:
                if not scene['length_eligible_for_preview']:continue
                a,b=scene['start_frame'],scene['end_frame_exclusive'];rel=f'previews/{sid}_v3_{a}_{b}.mp4'
                encode(frames[a:b],output/rel,fps);previews.append({**scene,'path':rel})
            n=len(frames); points=sorted(set(boundary_indices(n,v2)+boundary_indices(n,v3)))
            overview_points=sorted(set(np.linspace(0,n-1,12,dtype=int).tolist()))
            all_points=sorted(set(points+overview_points));(output/'native_frames'/sid).mkdir();native=[]
            for i in all_points:
                rel=f'native_frames/{sid}/f{i:06d}.png';require(cv2.imwrite(str(output/rel),frames[i]),'Native PNG failed')
                require(np.array_equal(cv2.imread(str(output/rel)),frames[i]),'Native PNG round-trip mismatch')
                native.append({'frame_index':i,'path':rel,'sha256':sha(output/rel)})
            marks=set(v2['hard_cut_candidates']+v3['hard_cut_candidates'])
            over=['contacts/'+p for p in frame_pages(frames,overview_points,output/'contacts',sid+'_o',marks)]
            bounds=['contacts/'+p for p in frame_pages(frames,points,output/'contacts',sid+'_b',marks)] if points else []
            row={'sample_id':sid,'video_id':old['video_id'],'source_group':old['source_group'],'original_split':'train',
                 'frame_count':n,'fps_rational':fps,'source_identity':identity,'parent_run':str(parent),
                 'v2_replay_matches_parent':True,'v2':v2,'v3':v3,'v2_signature':s2,'v3_signature':s3,
                 'scene_or_boundary_changed':s2!=s3,'original_preview':original,'scene_previews':previews,
                 'overview_pages':over,'boundary_pages':bounds,'native_inspection_frames':native}
            rows.append(row)
            shot_html=''.join(f'<label>v3場景 [{p["start_frame"]},{p["end_frame_exclusive"]})<br><video controls preload="metadata" src="{p["path"]}"></video></label>' for p in previews)
            cards.append(f'<section id="{sid}"><h2>{sid}</h2><p>v2：{html.escape(str(s2))}<br>v3：{html.escape(str(s3))}</p><div><label>原片（有損檢視，非訓練輸入）<br><video controls preload="metadata" src="{original}"></video></label>{shot_html}</div>'
                         '<details><summary>固定抽幀</summary>'+''.join(f'<img src="{p}" loading="lazy">' for p in over)+'</details>'
                         '<details><summary>兩版提出的邊界前後（CUT?任一版候選，非v3已確認）</summary>'+''.join(f'<img src="{p}" loading="lazy">' for p in bounds)+'</details></section>')
            save(output/'run_state.json',{'status':'running','videos_completed':len(rows)})
            print(sid,'CHANGED' if s2!=s3 else 'same', 'v2=',s2,'v3=',s3,flush=True)
        stats=ledger.statistics()
        require(stats['train']['opens']==19 and stats['validation']['opens']==stats['test']['opens']==0,'Unexpected access')
        require(all(sha(p)==h for p,h in protected.items()),'Protected source/baseline changed')
        by_id={r['sample_id']:r for r in rows}
        checks={'T004_no_false_cut10':10 not in by_id['T004']['v3']['hard_cut_candidates'],
                'T004_continuous_full_scene':by_id['T004']['v3_signature']['scenes']==[(0,146)],
                'Y011_real_cut44_kept':44 in by_id['Y011']['v3']['hard_cut_candidates'],
                'Y012_real_cut142_kept':142 in by_id['Y012']['v3']['hard_cut_candidates'],
                'T012_real_cut10_kept':10 in by_id['T012']['v3']['hard_cut_candidates'],
                'Y010_real_cut162_kept':162 in by_id['Y010']['v3']['hard_cut_candidates'],
                'Y013_no_new_exclusion':by_id['Y013']['v3_signature']['scenes']==[(0,146)],
                'unchanged_scene_signatures_except_target_T004':all(
                    r['v2_signature']['scenes']==r['v3_signature']['scenes'] and
                    r['v2_signature']['cuts']==r['v3_signature']['cuts'] and
                    r['v2_signature']['gradual']==r['v3_signature']['gradual']
                    for r in rows if r['sample_id']!='T004')}
        report={'status':'completed','videos':rows,'targeted_regression_checks':checks,'protected_sha256':protected,
                'original_access':stats,'sequential_frames_decoded':sum(r['frame_count'] for r in rows),
                'summary':{'videos':19,'source_groups':len({r['source_group'] for r in rows}),
                           'changed_samples':[r['sample_id'] for r in rows if r['scene_or_boundary_changed']]},
                'not_blind_evaluation':True,'classification_metrics_used':False,'parameter_sweep_used':False,
                'training_started':False,'feature_cache_modified':False,'automatic_training_use_videos':0,
                'target_event_selection_implemented':False,'AI_review_status':'pending'}
        save(output/'report.json',report)
        page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v3曝光尾巴：19支已知案例回歸</title><style>body{font:18px system-ui;line-height:1.8;max-width:1400px;margin:30px auto;padding:0 20px}img{max-width:100%}video{width:400px;max-width:100%}section{border-top:1px solid #ccd;padding:20px 0}section div{display:flex;gap:15px;flex-wrap:wrap}</style><h1>v3：曝光尾巴修正與v2回歸對照</h1><p>19支全部是已看過的train開發案例，不是新盲測。先重現v2逐項輸出，再用v3比較；未按分類準確率挑參數。v3只增加有限4幀內恢復搜尋、2幀恢復持續與亮度接近檢查，並檢查中間影格外觀相關。相似場景仍可能混淆。</p><p>原片像素不刪，不跨場景串接，不選最早／最長場景當目標；不改原三窗口或特徵，不訓練。Y019開頭不確定維持未知，不借此次修正改成通過。</p><p><a href="report.json">回歸檢查、逐幀分數與SHA</a> · <a href="AI_review.html">AI初審（若有）</a></p>'''+''.join(cards)+'</html>'
        (output/'index.html').write_text(page,encoding='utf-8')
        save(output/'run_state.json',{'status':'completed','videos_completed':19,'targeted_regressions':checks})
        print(checks,flush=True);return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','videos_completed':len(rows),'error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();print(execute(args.dry_run))
