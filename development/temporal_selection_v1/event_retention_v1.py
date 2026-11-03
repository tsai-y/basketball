"""Native-frame inspection of two human-identified targets; no selector or training change."""
from pathlib import Path
import argparse
import html
import shutil
import tempfile

import cv2
import numpy as np

from development.data_pipeline.pipeline import AccessLedger, SplitGuard
from development.yolo_train_probe_v1.probe import read, save, sha, require
from .probe import ARTIFACTS, FOLDER, select_samples, decode, frame_pages

DEFAULT_PARENT = ARTIFACTS / 'temporal_probe_v2_xafv9o3e'
CASES = ('Y012', 'Y019')
# Display-only ROI and dense review ranges, never input selection or training cropping.
DISPLAY = {'Y012': ((0, 0, 225, 170), (44, 96)),
           'Y019': ((65, 0, 260, 170), (0, 52))}


def bracket_retention(candidate, phases, frame_count):
    """Containment of supplied evidence brackets, NOT independent event verification."""
    def valid(pair):
        return (isinstance(pair, (list, tuple)) and len(pair) == 2
                and all(type(x) is int for x in pair)
                and 0 <= pair[0] < pair[1] <= frame_count)
    require(type(frame_count) is int and frame_count > 0 and valid(candidate), 'Invalid candidate')
    require(set(phases) == {'release', 'rim', 'result'}, 'Three phase keys required')
    checks = {}
    for name, bounds in phases.items():
        if bounds is None:
            checks[name] = None
        else:
            require(valid(bounds), 'Invalid evidence bracket')
            checks[name] = candidate[0] <= bounds[0] and bounds[1] <= candidate[1]
    status = ('evidence_outside_candidate' if False in checks.values() else
              'unresolved' if None in checks.values() else 'all_supplied_brackets_retained')
    return {'phase_checks': checks, 'status': status, 'independent_event_verification': False,
            'human_temporal_truth': False, 'training_use_approved': False}


def execute(parent=DEFAULT_PARENT, dry_run=False):
    parent = Path(parent).resolve()
    require(parent.is_relative_to(ARTIFACTS.resolve()), 'Foreign parent')
    report = read(parent / 'report.json')
    require(report['status'] == 'completed', 'Incomplete parent')
    refs = read(parent / 'event_reference_draft.json')
    for sid in CASES:
        ref = next(x for x in refs['videos'] if x['sample_id'] == sid)
        require(ref['target_event_identity_human_confirmed'], 'Unconfirmed target identity')
    manifest = read(parent / 'inputs/parent_report.json')
    audit_path = next(Path(p) for p in manifest['protected_input_sha256'] if p.endswith('/manifest.json'))
    ledger = AccessLedger(); guard = SplitGuard('train', ledger)
    samples = [s for s in select_samples(read(audit_path), guard.allowed) if s['sample_id'] in CASES]
    require(tuple(s['sample_id'] for s in samples) == CASES, 'Unexpected diagnostic samples')
    if dry_run:
        return {'videos': 2, 'source_opens': 0, 'writes': False, 'training_started': False}
    protected = {str(parent / p): sha(parent / p) for p in
                 ('report.json', 'event_reference_draft.json', 'AI_event_reference.json',
                  'human_Y019_target_confirmation.json', 'inputs/Y012_human_target_clarification.json')}
    protected.update(report['protected_input_sha256'])
    output = Path(tempfile.mkdtemp(prefix='event_retention_v1_', dir=ARTIFACTS))
    for folder in ('native', 'contacts', 'detail', 'inputs'):
        (output / folder).mkdir()
    shutil.copy2(Path(__file__), output / 'inputs/event_retention_v1.py')
    for name in ('event_reference_draft.json', 'human_Y019_target_confirmation.json'):
        shutil.copy2(parent / name, output / 'inputs' / name)
    rows = []; cards = []; cv2.setNumThreads(1)
    print('OUTPUT:', output, flush=True)
    for sample in samples:
        sid = sample['sample_id']; frames, identity = decode(sample, guard)
        (output / 'native' / sid).mkdir()
        native = []
        for i, frame in enumerate(frames):
            rel = f'native/{sid}/f{i:06d}.png'
            require(cv2.imwrite(str(output / rel), frame), 'PNG write failed')
            require(np.array_equal(cv2.imread(str(output / rel)), frame), 'Native PNG round-trip mismatch')
            native.append({'frame_index': i, 'path': rel, 'sha256': sha(output / rel)})
        overview = frame_pages(frames, range(0, len(frames), 4), output / 'contacts', sid, set())
        box, bounds = DISPLAY[sid]; x1, y1, x2, y2 = box
        # Nearest-neighbor magnification: no restoration, inferred ball pixels or smoothing.
        crops = [cv2.resize(f[y1:y2, x1:x2], (320, 240), interpolation=cv2.INTER_NEAREST) for f in frames]
        details = frame_pages(crops, range(*bounds), output / 'detail', sid, set())
        previous = next(v for v in report['videos'] if v['sample_id'] == sid)
        candidates = previous['v2']['contiguous_scene_candidates']
        rows.append({'sample_id': sid, 'video_id': sample['video_id'], 'frame_count': len(frames),
                     'source_identity': identity, 'native_frames': native, 'v2_scene_candidates': candidates,
                     'display_only_roi_xyxy': box, 'dense_display_range': bounds,
                     'overview_pages': ['contacts/' + p for p in overview],
                     'detail_pages': ['detail/' + p for p in details]})
        scene_text = html.escape(str([(c['start_frame'], c['end_frame_exclusive']) for c in candidates]))
        cards.append(f'<section><h2>{sid}</h2><p>原v2場景：{scene_text}；目標身分已確認，時間仍待AI檢查。</p>'
                     f'<input type="range" min="0" max="{len(frames)-1}" value="0" data-sid="{sid}">'
                     f'<span id="label{sid}">frame 0</span><br><img class="native" id="img{sid}" src="native/{sid}/f000000.png">'
                     '<details><summary>每4幀完整畫面總覽</summary>' +
                     ''.join(f'<img src="contacts/{p}" loading="lazy">' for p in overview) + '</details>'
                     '<details open><summary>連續逐幀局部放大（僅供檢視）</summary>' +
                     ''.join(f'<img src="detail/{p}" loading="lazy">' for p in details) + '</details></section>')
        print(sid, 'native frames:', len(frames), 'detail pages:', len(details), flush=True)
    require(all(sha(p) == h for p, h in protected.items()), 'Protected inputs changed')
    access = ledger.statistics()
    require(access['train']['opens'] == 2 and access['validation']['opens'] == access['test']['opens'] == 0,
            'Unexpected original source access')
    save(output / 'report.json', {'status': 'completed', 'videos': rows, 'protected_sha256': protected,
         'parent_run': str(parent), 'source_access': access, 'sequential_frames_decoded': sum(r['frame_count'] for r in rows),
         'training_started': False, 'selector_changed': False, 'feature_cache_modified': False,
         'AI_event_assessment': 'pending', 'display_crops_used_by_algorithm': False})
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>出手／到框／結果保留檢查</title>
<style>body{font:18px system-ui;line-height:1.8;max-width:1300px;margin:30px auto;padding:0 20px}img{max-width:100%}.native{width:640px;image-rendering:pixelated}input{width:65%}section{border-top:1px solid #ccc;margin-top:24px}</style>
<h1>兩支人工確認目標：事件保留檢查</h1><p>只作診斷；全原生影格逐格可查。局部圖採nearest放大、非超解析或訓練裁切。AI檢查不是人類真值，不用計分板／Make-Miss標籤反推球的位置。不改v2或訓練輸入。</p><p><a href="AI_review.html">AI檢查結論（若有）</a> · <a href="report.json">原生PNG SHA與來源</a></p>''' + ''.join(cards) + '''
<script>document.querySelectorAll('input').forEach(e=>e.addEventListener('input',()=>{let s=e.dataset.sid,i=Number(e.value);document.getElementById('img'+s).src='native/'+s+'/f'+String(i).padStart(6,'0')+'.png';document.getElementById('label'+s).textContent='frame '+i;}));</script></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', default=str(DEFAULT_PARENT)); p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(); print(execute(args.parent_run, args.dry_run))
