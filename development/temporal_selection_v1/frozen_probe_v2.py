"""Fixed v2 on12 new diagnostic train videos from2 previously unlisted visual-audit sources."""
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
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

from development.data_pipeline.pipeline import AccessLedger, SplitGuard
from development.yolo_train_probe_v1.probe import read, save, sha, require
from .probe import ROOT, FOLDER, ARTIFACTS, encode, frame_pages
from .selector_v2 import ScenePolicy, analyze_v2

SEED = 20261028
PARENT = ARTIFACTS / 'temporal_probe_v2_xafv9o3e'
AUDIT_ROOTS = (ROOT / 'development/yolo_audit_v1/artifacts',
               ROOT / 'development/yolo_train_probe_v1/artifacts', ARTIFACTS)


def nested_video_ids(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get('video_id'), str):
            yield obj['video_id']
        for value in obj.values():
            yield from nested_video_ids(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from nested_video_ids(value)


def prior_inventory(allowed):
    paths = sorted({p for root in AUDIT_ROOTS for p in root.rglob('*.json')
                    if 'manifest' in p.name or p.name == 'sampling_plan.json'})
    require(paths, 'No existing audit inventory')
    known = set(); sources = []
    for path in paths:
        ids = set(nested_video_ids(read(path))) & allowed.keys()
        known.update(ids)
        sources.append({'path': str(path), 'sha256': sha(path), 'canonical_train_ids': sorted(ids)})
    return known, sources


def rank(value):
    return hashlib.sha256(f'temporal-fixed-v2:{SEED}:{value}'.encode()).hexdigest()


def choose_samples(allowed, known, groups_required=2, per_group=6):
    require(type(groups_required) is int and groups_required > 0 and
            type(per_group) is int and per_group > 0, 'Invalid sample size')
    require(allowed and all(r['video_id'] == k and r['split'] == 'train' and str(r['excluded']) == '0'
                           for k, r in allowed.items()), 'Canonical non-excluded train rows only')
    seen_groups = {allowed[v]['source_group'] for v in known if v in allowed}
    fresh = sorted({r['source_group'] for r in allowed.values()} - seen_groups, key=rank)
    require(len(fresh) >= groups_required, 'Not enough unlisted source groups; do not silently loosen novelty')
    rows = []
    for group in fresh[:groups_required]:
        pool = sorted((r for r in allowed.values() if r['source_group'] == group),
                      key=lambda r: rank(r['video_id']))
        require(len(pool) >= per_group, 'Insufficient videos in fixed source; do not replace after inspection')
        rows.extend(dict(row) for row in pool[:per_group])
    require(not set(r['video_id'] for r in rows) & known, 'Previously listed video selected')
    require(len({r['video_id'] for r in rows}) == groups_required * per_group, 'Duplicate selection')
    return [{**r, 'sample_id': f'T{i:03d}'} for i, r in enumerate(rows, 1)], sorted(seen_groups)


def decode_new(sample, guard):
    # Authorize before any path access/open. No current class/label or event evidence enters analyze_v2.
    require(sample['split'] == 'train' and str(sample['excluded']) == '0', 'Nontrain/excluded rejected')
    require(all(sample[k] == v for k, v in guard.allowed[sample['video_id']].items()), 'Canonical row mismatch')
    before = guard.identity(sample['video_id'])
    fd = guard.open(sample['video_id'], sample['video_path'], 'frozen_v2_new_train_visual_diagnostic')
    cap = None; frames = []
    try:
        cap = cv2.VideoCapture(f'/proc/self/fd/{fd}', cv2.CAP_FFMPEG)
        require(cap.isOpened(), 'Cannot decode guarded native source')
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT); fps = cap.get(cv2.CAP_PROP_FPS)
        require(math.isfinite(count) and abs(count-round(count)) < 1e-6 and 2 <= count <= 1000,
                'Invalid/oversized native frame count')
        require(math.isfinite(fps) and fps > 0, 'Invalid FPS')
        rational = str(Fraction(fps).limit_denominator(1001))
        require(abs(float(Fraction(rational))-fps) < 1e-5, 'FPS rational reconstruction mismatch')
        for _ in range(round(count)):
            ok, frame = cap.read()
            require(ok and frame.shape == (240,320,3) and frame.dtype == np.uint8, 'Bad/truncated native frame')
            frames.append(frame)
        require(not cap.read()[0], 'Extra frames')
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)
    after = guard.identity(sample['video_id']); require(before == after, 'Source changed')
    return frames, rational, {'before': before, 'after': after}


def boundary_indices(n, result):
    edges = set(result['hard_cut_candidates'])
    for region in result['exposure_flash_candidates'] + result['gradual_change_regions']:
        edges.update((region['start_frame'], region['end_frame_exclusive']))
    return sorted({i for edge in edges for i in range(max(0,edge-2), min(n,edge+3))})


def execute(dry_run=False):
    # Freeze both pixel descriptor and v2 implementation against the already-run snapshot before sampling.
    for name in ('selector.py', 'selector_v2.py'):
        require(sha(FOLDER/name) == sha(PARENT/'inputs'/name), 'Frozen implementation changed: '+name)
    policy = ScenePolicy()
    require(asdict(policy) == read(PARENT/'policy.json'), 'Frozen thresholds changed')
    ledger = AccessLedger(); guard = SplitGuard('train', ledger)
    known, inventory = prior_inventory(guard.allowed)
    samples, seen_groups = choose_samples(guard.allowed, known)
    plan = {'seed': SEED, 'samples': samples, 'prior_inventory': inventory,
            'known_train_video_ids': sorted(known), 'known_source_groups': seen_groups,
            'sampling': 'Before pixels: SHA256-ranked fresh source groups,6 SHA256-ranked videos each; no class or outcome stratification',
            'novelty_scope': 'Not listed in inventoried YOLO/temporal manifests or sampling plans; not a claim about all human viewing or quality-pilot viewing',
            'previous_rgb_training_may_have_used_these_train_videos': True,
            'source_groups': sorted({s['source_group'] for s in samples}),
            'independent_source_groups': 2, 'videos': 12, 'policy': asdict(policy),
            'missing_failed_or_uncertain_samples_not_replaced': True,
            'not_classification_validation_or_test': True, 'training_started': False}
    if dry_run:
        return {**plan, 'files_written': False, 'source_opens': 0}
    output = Path(tempfile.mkdtemp(prefix='frozen_v2_new12_', dir=ARTIFACTS))
    for name in ('inputs', 'previews', 'contacts', 'native_frames'):
        (output/name).mkdir()
    protected = {item['path']: item['sha256'] for item in inventory}
    protected.update(read(PARENT/'report.json')['protected_input_sha256'])
    for name in ('selector.py','selector_v2.py','frozen_probe_v2.py'):
        source = FOLDER/name; protected[str(source)] = sha(source)
        shutil.copy2(source, output/'inputs'/name)
    protected[str(PARENT/'policy.json')] = sha(PARENT/'policy.json')
    # This immutable plan is saved before the first native video decode.
    save(output/'sampling_plan.json', plan)
    save(output/'run_state.json', {'status': 'running', 'videos_completed': 0})
    print('OUTPUT:', output, flush=True); rows = []; cards = []; cv2.setNumThreads(1)
    try:
        for sample in samples:
            sid = sample['sample_id']; frames, fps, identity = decode_new(sample, guard)
            result = analyze_v2(frames, policy); n = len(frames)
            original = f'previews/{sid}_original.mp4'; encode(frames, output/original, fps)
            previews = []
            for scene in result['contiguous_scene_candidates']:
                if not scene['length_eligible_for_preview']:
                    continue
                a,b = scene['start_frame'],scene['end_frame_exclusive']
                rel = f'previews/{sid}_{a}_{b}.mp4'; encode(frames[a:b],output/rel,fps)
                previews.append({**scene,'path':rel})
            overview_points = sorted(set(np.linspace(0,n-1,16,dtype=int).tolist()))
            edges = boundary_indices(n,result); points = sorted(set(overview_points+edges))
            (output/'native_frames'/sid).mkdir(); native = []
            for i in points:
                rel = f'native_frames/{sid}/f{i:06d}.png'
                require(cv2.imwrite(str(output/rel),frames[i]),'Native PNG failed')
                require(np.array_equal(cv2.imread(str(output/rel)),frames[i]),'Native PNG changed pixels')
                native.append({'frame_index':i,'path':rel,'sha256':sha(output/rel)})
            overview = ['contacts/'+p for p in frame_pages(frames,overview_points,output/'contacts',sid+'_o',set(result['hard_cut_candidates']))]
            boundaries = (['contacts/'+p for p in frame_pages(frames,edges,output/'contacts',sid+'_b',set(result['hard_cut_candidates']))]
                          if edges else [])
            row = {'sample_id':sid,'video_id':sample['video_id'],'source_group':sample['source_group'],
                   'original_split':'train','frame_count':n,'fps_rational':fps,'source_identity':identity,
                   'analysis':result,'original_preview':original,'scene_previews':previews,
                   'overview_points':overview_points,'overview_pages':overview,'boundary_pages':boundaries,
                   'native_inspection_frames':native,'AI_review_status':'pending'}
            rows.append(row)
            intervals = [(s['start_frame'],s['end_frame_exclusive']) for s in result['contiguous_scene_candidates']]
            gradual = [(r['start_frame'],r['end_frame_exclusive']) for r in result['gradual_change_regions']]
            shots = ''.join(f'<label>場景 [{p["start_frame"]},{p["end_frame_exclusive"]})，非目標確認<br><video controls preload="metadata" src="{p["path"]}"></video></label>' for p in previews)
            cards.append(f'<section id="{sid}"><h2>{sid} / {html.escape(sample["source_group"])}</h2><p>場景 {html.escape(str(intervals))}；切鏡 {result["hard_cut_candidates"]}；漸變 {html.escape(str(gradual))}。</p>'
                         f'<div><label>原片（有損預覽，非訓練輸入）<br><video controls preload="metadata" src="{original}"></video></label>{shots}</div>'
                         '<details><summary>固定16位置原生抽幀</summary>'+''.join(f'<img src="{p}" loading="lazy">' for p in overview)+'</details>'
                         '<details><summary>提出的邊界前後原生抽幀</summary>'+''.join(f'<img src="{p}" loading="lazy">' for p in boundaries)+'</details></section>')
            save(output/'run_state.json',{'status':'running','videos_completed':len(rows)})
            print(sid,sample['source_group'], 'cuts=',result['hard_cut_candidates'],'gradual=',[(r['start_frame'],r['end_frame_exclusive']) for r in result['gradual_change_regions']],'scenes=',intervals,flush=True)
        require(all(sha(p)==h for p,h in protected.items()),'Protected input changed')
        access=ledger.statistics()
        require(access['train']['opens']==12 and access['validation']['opens']==access['test']['opens']==0,'Unexpected source access')
        summary={'videos':12,'independent_source_groups':2,'sample_class_counts_not_used_for_selection':dict(Counter(s['original_class'] for s in samples)),
                 'videos_with_hard_cuts':sum(bool(r['analysis']['hard_cut_candidates']) for r in rows),
                 'videos_with_gradual_flags':sum(bool(r['analysis']['gradual_change_regions']) for r in rows),
                 'videos_with_exposure_flags':sum(bool(r['analysis']['exposure_flash_candidates']) for r in rows),
                 'no_length_eligible_scene_videos':sum(not r['scene_previews'] for r in rows),
                 'automatic_training_use_videos':0}
        report={'status':'completed','schema':'frozen_v2_new12_train_diagnostic','plan':plan,'summary':summary,
                'videos':rows,'original_access':access,'sequential_frames_decoded':sum(r['frame_count'] for r in rows),
                'protected_sha256':protected,'frozen_implementation_verified':True,'threshold_tuning_used':False,
                'test_used':False,'training_started':False,'feature_cache_modified':False,'target_event_selection_implemented':False}
        save(output/'manifest.json',report)
        page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定v2：新12支train影片診斷</title><style>body{font:18px system-ui;line-height:1.8;max-width:1400px;margin:30px auto;padding:0 20px}img{max-width:100%}video{width:400px;max-width:100%}section{border-top:1px solid #ccd;padding:20px 0}section div{display:flex;gap:15px;flex-wrap:wrap}</style><h1>不調參的v2：新12支train影片，僅2個獨立來源</h1><p>抽樣前依雜湊固定來源與影片，排除既有YOLO／選段manifest與sampling plan列出的影片及其來源。這不是完整的人類觀看紀錄、正式分類validation/test或12個獨立來源；原RGB訓練可能已使用這些train影片。類別／Make-Miss不進演算法，也未按球可見或模型準確率挑片。</p><p>只提出場景，不選第一段／最長段當目標，不跨鏡頭拼接；短場景仍在JSON。未知不算通過。不改政策、特徵或訓練。</p><p><a href="sampling_plan.json">看像素前固定抽樣</a> · <a href="manifest.json">來源、全部分數與SHA</a> · <a href="AI_review.html">AI初審（若有）</a></p>'''+''.join(cards)+'</html>'
        (output/'index.html').write_text(page,encoding='utf-8')
        save(output/'run_state.json',{'status':'completed','videos_completed':12})
        print(summary,flush=True);return output
    except BaseException as exc:
        save(output/'run_state.json',{'status':'failed','videos_completed':len(rows),'error':f'{type(exc).__name__}: {exc}'})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();print(execute(args.dry_run))
