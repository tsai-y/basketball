"""Fixed100 train-only v4 proposals. No automatic event choice, features, or training."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import html
import json
import shutil
import tempfile
import time

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger, SplitGuard
from development.data_pipeline.multiwindow_v1 import select_covering_windows
from development.yolo_train_probe_v1.probe import read, save, sha, require
from .probe import ROOT, FOLDER, ARTIFACTS, encode
from .frozen_probe_v2 import prior_inventory, decode_new, boundary_indices
from .selector_v4 import BoundaryPolicy, analyze_v4

SEED = 20261028
COUNT = 100
PARENT = ARTIFACTS / 'boundary_refine_v4_se8f2bo6'
SMOKE = ARTIFACTS / 'adapter_smoke_v1_yngsk1h0'
CACHE = ROOT / 'development/mixnet_i3d/artifacts/feature_cache_v1/67b6400e0982bb6d'


def rank(value):
    return hashlib.sha256(f'fixed-v4-balanced100:{SEED}:{value}'.encode()).hexdigest()


def choose_samples(allowed, known, count=COUNT):
    """Round-robin all source groups; no class, outcome, pixel, or length input."""
    require(type(count) is int and count > 0 and allowed, 'Invalid requested sample size')
    require(all(k == r['video_id'] and r['split'] == 'train' and str(r['excluded']) == '0'
                for k, r in allowed.items()), 'Canonical non-excluded train only')
    groups = sorted({r['source_group'] for r in allowed.values()}, key=rank)
    require(count >= len(groups), 'Must cover every source group; no silent subset')
    pools = {g: sorted((r for v, r in allowed.items() if v not in known and r['source_group'] == g),
                       key=lambda r: rank(r['video_id'])) for g in groups}
    require(all(pools.values()), 'A source has no unlisted videos; do not relax exclusion')
    require(sum(map(len, pools.values())) >= count, 'Insufficient unlisted videos')
    rows = []; depth = 0
    while len(rows) < count:
        for group in groups:
            if depth < len(pools[group]):
                rows.append(dict(pools[group][depth]))
                if len(rows) == count:
                    break
        depth += 1
    require(len({r['video_id'] for r in rows}) == count, 'Duplicate sample')
    return [{**r, 'sample_id': f'S{i:03d}'} for i, r in enumerate(rows, 1)]


def geometry(a, b):
    length = b-a
    if length < 66:
        return {'eligible': False, 'reason': 'shorter_than_66', 'windows': None}
    try:
        windows = tuple((a+x, a+y) for x, y in select_covering_windows(length))
    except ValueError:
        return {'eligible': False, 'reason': 'three_windows_cannot_fully_cover', 'windows': None}
    require(all(a <= x < y <= b and y-x == 64 for x, y in windows), 'Invalid geometry')
    require({f for x, y in windows for f in range(x, y)} == set(range(a, b)), 'Incomplete coverage')
    return {'eligible': True, 'reason': 'geometry_only_not_event_confirmation', 'windows': windows}


def plan_input(n, analysis):
    """No semantic inference from duration. Even a single crop stays unapproved.

    Count ALL clean spans, including <8-frame spans omitted by the preview list.
    A second short scene must not disappear merely because another scene is long.
    """
    spans = analysis['all_segments_including_short']
    require(spans and spans[0]['start_frame'] == 0 and spans[-1]['end_frame_exclusive'] == n,
            'Missing scene partition endpoints')
    require(all(s['end_frame_exclusive'] > s['start_frame'] and
                s['length'] == s['end_frame_exclusive']-s['start_frame'] for s in spans), 'Invalid span')
    require(all(x['end_frame_exclusive'] == y['start_frame'] for x, y in zip(spans, spans[1:])),
            'Scene partition gap or overlap')
    clean = [s for s in spans if not s['inside_proposed_change_span']]
    candidates = [{**s, 'geometry': geometry(s['start_frame'], s['end_frame_exclusive']),
                   'event_identity_verified': False, 'event_completeness_verified': False}
                  for s in clean]
    try:
        original = select_covering_windows(n)
    except ValueError:
        original = None
    result = {'mode': 'fallback_original', 'reason': None, 'all_clean_scene_candidates': candidates,
              'original_windows': original, 'effective_windows': original, 'effective_interval': [0,n],
              'original_policy_can_decode': original is not None, 'single_crop_proposal': None,
              'potential_removed_frames_geometry_only': 0, 'actual_removed_frames': 0,
              'training_use_approved': False, 'semantic_target_automatically_selected': False,
              'frames_concatenated_across_scenes': False, 'temporal_resampling_used': False}
    if original is None:
        result['reason'] = 'original_policy_rejects_source_no_silent_padding'
    elif len(clean) == 0:
        result['reason'] = 'no_clean_scene_proposal'
    elif len(clean) > 1:
        result['reason'] = 'multiple_scene_proposals_target_unconfirmed'
    elif not candidates[0]['geometry']['eligible']:
        result['reason'] = candidates[0]['geometry']['reason']
    elif (clean[0]['start_frame'], clean[0]['end_frame_exclusive']) == (0,n):
        result.update(mode='identity_original', reason='full_source_no_temporal_change')
    else:
        candidate = candidates[0]
        result.update(reason='single_crop_unverified_removed_content_and_event',
                      single_crop_proposal=candidate,
                      potential_removed_frames_geometry_only=n-candidate['length'])
    return result


def cache_inventory():
    entries = []
    for p in sorted(CACHE.rglob('*')):
        if p.is_file():
            s = p.stat(); entries.append((str(p.relative_to(CACHE)), s.st_size, s.st_mtime_ns))
    require(entries, 'Missing original feature cache')
    return {'files': len(entries), 'name_size_mtime_sha256': hashlib.sha256(json.dumps(entries).encode()).hexdigest()}


def fixed_plan(guard, plan_path=None):
    require(asdict(BoundaryPolicy()) == read(PARENT/'policy.json'), 'Frozen v4 settings changed')
    for name in ('selector.py','selector_v2.py','selector_v3.py','selector_v4.py'):
        require(sha(FOLDER/name) == sha(PARENT/'inputs'/name), 'Frozen source changed: '+name)
    if plan_path is not None:
        plan = read(plan_path)
        require(plan['schema'] == 'fixed_v4_train100_before_pixels_v1' and plan['seed'] == SEED,
                'Foreign sampling plan')
        require(plan['policy'] == asdict(BoundaryPolicy()), 'Saved policy mismatch')
        require(len(plan['samples']) == COUNT and len({s['video_id'] for s in plan['samples']}) == COUNT,
                'Saved plan must contain100 unique videos')
        for sample in plan['samples']:
            require(sample['video_id'] in guard.allowed, 'Saved nontrain ID')
            row = guard.allowed[sample['video_id']]
            require(all(sample[k] == v for k,v in row.items()), 'Saved canonical row mismatch')
            require(guard.identity(sample['video_id']) == plan['source_identity_before_pixels'][sample['video_id']],
                    'Saved source identity changed')
        require(all(sha(s['path']) == s['sha256'] for s in plan['prior_inventory']), 'Prior inventory changed')
        return plan
    # Reproduction must explicitly reuse the persisted plan, not silently draw100 more.
    require(not list(ARTIFACTS.glob('batch100_v4_*/sampling_plan.json')),
            'A fixed100 plan already exists. Reuse its path with --plan; do not redraw after viewing.')
    known, inventory = prior_inventory(guard.allowed)
    samples = choose_samples(guard.allowed, known)
    return {'schema': 'fixed_v4_train100_before_pixels_v1', 'seed': SEED, 'samples': samples,
            'policy': asdict(BoundaryPolicy()), 'prior_inventory': inventory,
            'excluded_prior_listed_ids': sorted(known),
            'sampling': 'SHA256-ranked all source groups, round-robin SHA256-ranked unlisted IDs; no labels/pixels/lengths',
            'source_group_counts': dict(sorted(Counter(s['source_group'] for s in samples).items())),
            'source_identity_before_pixels': {s['video_id']: guard.identity(s['video_id']) for s in samples},
            'novelty_scope': 'New IDs relative to listed YOLO/temporal manifests/plans only; previously used sources, not source-blind, not exhaustive human viewing history',
            'previous_rgb_training_may_have_used_these_train_videos': True,
            'failed_uncertain_samples_not_replaced': True, 'training_started': False,
            'classification_validation_test_used': False, 'automatic_training_use_videos': 0}


def contact(frames, indices, path, title, columns=4, scale=1.):
    indices = sorted(set(indices)); require(indices, 'Empty contact')
    w,h = round(320*scale),round(240*scale); tiles = []
    for i in indices:
        require(0 <= i < len(frames), 'Contact index outside source')
        tile = np.zeros((h+24,w,3),np.uint8)
        tile[24:] = frames[i] if scale == 1. else cv2.resize(frames[i],(w,h),interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f'{title} f{i}', (4,17), cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    canvas = np.vstack([np.hstack(tiles[i:i+columns]) for i in range(0,len(tiles),columns)])
    require(cv2.imwrite(str(path),canvas), 'Contact write failed')


def overview_strip(frames, sid):
    indices = sorted(set(np.linspace(0,len(frames)-1,8,dtype=int).tolist()))
    tiles = []
    for i in indices:
        tile = np.zeros((144,160,3),np.uint8)
        tile[24:] = cv2.resize(frames[i],(160,120),interpolation=cv2.INTER_AREA)
        cv2.putText(tile,f'{sid} f{i}',(3,16),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1)
        tiles.append(tile)
    return np.hstack(tiles), indices


def render(output, rows, summary, pages):
    cards=[]
    for r in rows:
        sid=r['sample_id']
        if r['status']!='completed':
            cards.append(f'<section id="{sid}"><h2>{sid}: failed → no replacement</h2><pre>{html.escape(r["error"])}</pre></section>'); continue
        p=r['input_plan'];candidate=p['single_crop_proposal']
        desc='無單一裁段候選' if candidate is None else str((candidate['start_frame'],candidate['end_frame_exclusive']))
        videos='<label>原片（有損檢視預覽）<br><video controls preload="none" src="'+r['original_preview']+'"></video></label>'
        videos+=''.join(f'<label>場景候選 [{v["start_frame"]},{v["end_frame_exclusive"]})<br><video controls preload="none" src="{v["path"]}"></video></label>' for v in r['scene_previews'])
        contacts=''.join(f'<a href="{v}"><img loading="lazy" src="{v}"></a>' for v in r['contact_pages'])
        cards.append(f'<section id="{sid}"><h2>{sid} / {html.escape(r["source_group"])}</h2><p>{html.escape(r["video_id"])}<br>原長度 {r["frame_count"]}；單一裁段候選 {desc}；目前 {p["mode"]}：{p["reason"]}<br>原／有效窗口相同：{p["effective_windows"]}；沒有把候選自動採用。</p><div>{videos}</div><details><summary>原生16位置及候選邊界接觸圖</summary>{contacts}</details></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定v4：100支train選段試跑</title><style>body{font:17px system-ui;line-height:1.7;max-width:1400px;margin:30px auto;padding:0 18px}video{width:400px;max-width:100%}img{max-width:100%}section{border-top:1px solid #ccd;padding:20px 0}section div{display:flex;gap:14px;flex-wrap:wrap}.warn{background:#fff1cd;padding:15px}pre{white-space:pre-wrap}</style><h1>固定v4：100支train選段試跑，不訓練</h1><p class="warn">新影片ID、不是新來源盲測。抽樣不依標籤／像素／長度／分類結果；固定v4、不調參、不換掉失敗樣本。只產生場景與窗口幾何提案，不因只有一段長片段就認定為目標投籃；所有裁段先回退原政策，AI視覺初審也不當成人工真值或訓練批准。</p><p>原生BGR以既有train guard只讀解碼；三窗口座標均為原片半開區間。不開分類validation/test、不讀寫特徵內容、不載入模型、不新增訓練。TANet保持不動。預覽不是特徵輸入。</p><p><a href="sampling_plan.json">看像素前固定抽樣</a> · <a href="manifest.json">全部逐幀分析與回退計畫</a> · <a href="summary.json">統計</a> · <a href="AI_review.html">AI初審（產生後）</a></p><pre>'''+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><h2>100支概覽（每支固定8幀，160×120縮圖，不是逐球判定）</h2>'+''.join(f'<a href="{p}"><img loading="lazy" src="{p}"></a>' for p in pages)+''.join(cards)+'</html>'
    (output/'index.html').write_text(page,encoding='utf-8')


def execute(dry_run=False, plan_path=None):
    ledger=AccessLedger(); guard=SplitGuard('train',ledger); plan=fixed_plan(guard,plan_path)
    if dry_run:
        return {'videos':len(plan['samples']), 'source_groups':plan['source_group_counts'],
                'prior_excluded_ids':len(plan['excluded_prior_listed_ids']),
                'source_opens':0, 'files_written':False, 'training_started':False}
    output=Path(tempfile.mkdtemp(prefix='batch100_v4_',dir=ARTIFACTS))
    for name in ('inputs','contacts','overview','previews'):(output/name).mkdir()
    protected=dict(read(SMOKE/'report.json')['protected_sha256'])
    protected.update({str(PARENT/'report.json'):sha(PARENT/'report.json'),str(SMOKE/'report.json'):sha(SMOKE/'report.json')})
    protected.update({s['path']:s['sha256'] for s in plan['prior_inventory']})
    for name in ('selector.py','selector_v2.py','selector_v3.py','selector_v4.py','batch_probe_v1.py','window_adapter_v1.py','frozen_probe_v2.py','probe.py'):
        p=FOLDER/name; protected[str(p)]=sha(p); shutil.copy2(p,output/'inputs'/name)
    save(output/'sampling_plan.json',plan); save(output/'policy.json',plan['policy'])
    cache_before=cache_inventory(); save(output/'inputs/cache_inventory_before.json',cache_before)
    save(output/'run_state.json',{'status':'running','videos_completed':0})
    print('OUTPUT:',output,flush=True); cv2.setNumThreads(1); started=time.monotonic()
    rows=[]; strips=[]; pages=[]
    try:
        for sample in plan['samples']:
            sid=sample['sample_id']; tick=time.monotonic()
            try:
                frames,fps,identity=decode_new(sample,guard)
                require(identity['before']==plan['source_identity_before_pixels'][sample['video_id']], 'Source changed since plan')
                n=len(frames);analysis=analyze_v4(frames,BoundaryPolicy());decision=plan_input(n,analysis)
                original=f'previews/{sid}_original.mp4';encode(frames,output/original,fps)
                previews=[]
                for scene in decision['all_clean_scene_candidates']:
                    if not scene['geometry']['eligible']:continue
                    a,b=scene['start_frame'],scene['end_frame_exclusive']
                    if (a,b)==(0,n):continue
                    rel=f'previews/{sid}_scene_{a}_{b}.mp4';encode(frames[a:b],output/rel,fps)
                    previews.append({'start_frame':a,'end_frame_exclusive':b,'path':rel,'not_selected_target':True})
                overview=sorted(set(np.linspace(0,n-1,16,dtype=int).tolist()))
                rel=f'contacts/{sid}_native16.png';contact(frames,overview,output/rel,sid);contacts=[rel]
                edges=boundary_indices(n,analysis)
                if edges:
                    rel=f'contacts/{sid}_boundaries.png';contact(frames,edges,output/rel,sid);contacts.append(rel)
                strip,points=overview_strip(frames,sid);strips.append(strip)
                if len(strips)==8:
                    rel=f'overview/batch_{len(pages)+1:02d}.png'
                    require(cv2.imwrite(str(output/rel),np.vstack(strips)), 'Overview failed');pages.append(rel);strips=[]
                row={'sample_id':sid,'video_id':sample['video_id'],'source_group':sample['source_group'],
                     'original_split':'train','status':'completed','frame_count':n,'fps_rational':fps,
                     'source_identity':identity,'v4':analysis,'input_plan':decision,
                     'original_preview':original,'scene_previews':previews,'contact_pages':contacts,
                     'overview_frame_indices':points,'AI_review_status':'pending','seconds':time.monotonic()-tick}
                del frames
            except (ValueError,OSError,cv2.error) as exc:
                row={'sample_id':sid,'video_id':sample['video_id'],'source_group':sample['source_group'],
                     'original_split':'train','status':'failed','error':f'{type(exc).__name__}: {exc}',
                     'replaced':False,'training_use_approved':False}
            rows.append(row)
            save(output/'run_state.json',{'status':'running','videos_processed':len(rows),'last_sample':sid})
            print(f'{len(rows)}/100 {sid}: '+(row['input_plan']['reason'] if row['status']=='completed' else row['error']),flush=True)
        if strips:
            rel=f'overview/batch_{len(pages)+1:02d}.png'
            require(cv2.imwrite(str(output/rel),np.vstack(strips)), 'Overview failed');pages.append(rel)
        good=[r for r in rows if r['status']=='completed'];access=ledger.statistics()
        require(access['validation']['opens']==access['test']['opens']==0,'Nontrain source access')
        require(len(rows)==COUNT and access['train']['opens']<=COUNT,'Wrong sample/access count')
        require(all(sha(p)==h for p,h in protected.items()),'Protected implementation/history changed')
        cache_after=cache_inventory();require(cache_before==cache_after,'Original feature-cache inventory changed')
        summary={'videos_requested':COUNT,'videos_completed':len(good),'failures_not_replaced':COUNT-len(good),
                 'source_groups':len(plan['source_group_counts']),
                 'reason_counts':dict(Counter(r['input_plan']['reason'] for r in good)),
                 'original_frame_count_distribution':dict(sorted(Counter(r['frame_count'] for r in good).items())),
                 'single_crop_proposal_lengths':dict(sorted(Counter(r['input_plan']['single_crop_proposal']['length'] for r in good if r['input_plan']['single_crop_proposal']).items())),
                 'geometry_only_potential_removed_frames':sum(r['input_plan']['potential_removed_frames_geometry_only'] for r in good),
                 'actual_removed_frames':0,'automatic_training_use_videos':0,
                 'original_access':access,'sequential_frames_decoded':sum(r['frame_count'] for r in good),
                 'elapsed_seconds':time.monotonic()-started,'threshold_tuning_used':False,
                 'class_counts_after_sampling_not_selection_inputs':dict(Counter(s['original_class'] for s in plan['samples']))}
        report={'schema':'fixed_v4_train100_scene_diagnostic_v1','status':'completed' if len(good)==COUNT else 'completed_with_failures',
                'plan_path':'sampling_plan.json','summary':summary,'videos':rows,'overview_pages':pages,
                'protected_sha256':protected,'cache_inventory_before':cache_before,'cache_inventory_after':cache_after,
                'training_started':False,'classification_accuracy_computed':False,'models_loaded':False,
                'feature_cache_read_or_written':False,'TANet_read_or_modified':False,
                'adapter_tensors_extracted_for_this_batch':False,
                'effective_windows_are_coordinate_plans_not_new_feature_cache':True,
                'selection_policy_automatically_approved':False,'AI_review_status':'pending'}
        save(output/'manifest.json',report);save(output/'summary.json',summary);render(output,rows,summary,pages)
        save(output/'run_state.json',{'status':report['status'],'videos_processed':len(rows)})
        print(json.dumps(summary,ensure_ascii=False),flush=True);return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','videos_processed':len(rows),'error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true');p.add_argument('--plan',type=Path)
    args=p.parse_args();print(execute(args.dry_run,args.plan))
