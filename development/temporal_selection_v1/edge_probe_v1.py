"""One fixed8 PNG-only comparison: frozen v5, raw edges, motion-compensated edges.

No video decoder, model, Dataset adaptation, threshold sweep or training entry.
"""
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
from .batch_probe_v1 import cache_inventory
from .edge_structure_v1 import EdgePolicy, analyze_edges, gray_frame, pair_metrics
from .probe import ARTIFACTS, FOLDER, ROOT
from .selector_v5 import ExposurePolicyV5, analyze_v5

PARENT = ARTIFACTS/'event_review_v5_bi_g35eh'
BASELINE = ARTIFACTS/'local_exposure_v5_of7zq9p2'
CASE_IDS = ('Y012', 'Y019', 'S002', 'S018', 'S025', 'S026', 'S030', 'S052')
REFERENCES = {
    'edge_motion_inspiration_not_exact_reproduction': 'https://www.cs.cornell.edu/~rdz/Papers/ZMM-JMS99.pdf',
    'opencv_lk': 'https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html',
    'opencv_similarity_ransac': 'https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html',
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fixed_plan():
    """Canonical train membership and frozen PNG list checked before pixels."""
    ledger = AccessLedger()
    guard = SplitGuard('train', ledger)
    parent, baseline = read(PARENT/'report.json'), read(BASELINE/'report.json')
    require(read(PARENT/'AI_review.json')['status'] == 'all8_AI_preliminary_reviews_complete', 'Need completed prior review')
    require(asdict(ExposurePolicyV5()) == read(BASELINE/'policy_v5.json'), 'Frozen v5 policy changed')
    for name in ('selector.py', 'selector_v2.py', 'selector_v3.py', 'selector_v4.py', 'selector_v5.py'):
        require(sha(FOLDER/name) == sha(BASELINE/'inputs'/name), 'Frozen selector changed')
    by = {r['sample_id']: r for r in parent['videos']}
    old = {r['sample_id']: r for r in baseline['videos']}
    require(set(by) == set(CASE_IDS), 'Do not redraw or replace the fixed8')
    cases = []
    for sid in CASE_IDS:
        row, prior = by[sid], old[sid]
        require(row['video_id'] == prior['video_id'] and prior['original_split'] == 'train', 'Wrong source mapping')
        require(row['video_id'] in guard.allowed, 'Nontrain source rejected before PNG access')
        require(len(row['native_frames']) == row['frame_count'], 'Incomplete native sequence')
        require([f['frame_index'] for f in row['native_frames']] == list(range(row['frame_count'])), 'Nonsequential source')
        for i, entry in enumerate(row['native_frames']):
            require(entry['path'] == f'native/{sid}/f{i:06d}.png', 'Noncanonical PNG path')
        cases.append({'sample_id': sid, 'video_id': row['video_id'], 'frame_count': row['frame_count'],
                      'native_frames': row['native_frames'], 'split': 'train'})
    plan = {'schema': 'fixed8_png_edge_probe_v1_before_pixels', 'cases': cases,
            'edge_policy': asdict(EdgePolicy()), 'v5_policy': asdict(ExposurePolicyV5()),
            'parent_report_sha256': sha(PARENT/'report.json'), 'baseline_report_sha256': sha(BASELINE/'report.json'),
            'sampling': 'Reuse exactly the existing purposeful8; no replacement, labels or classifier predictions in selector',
            'evaluation': 'Known development examples, not blind or representative; no boundary ground truth',
            'rule': 'Single frozen policy; no threshold adjustments after real-case outputs',
            'references': REFERENCES, 'video_opens': 0, 'production_crop_adoptions': 0,
            'training_started': False, 'new_human_annotations_requested': False}
    return plan, by, old, ledger


def load_native(row):
    require(row['split'] == 'train' and row['sample_id'] in CASE_IDS, 'Reject outside fixed train plan')
    frames = []
    for entry in row['native_frames']:
        path = PARENT/entry['path']
        require(path.resolve().is_relative_to((PARENT/'native'/row['sample_id']).resolve()), 'Foreign PNG')
        require(sha(path) == entry['sha256'], 'Native PNG changed')
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        require(image is not None and image.shape == (240, 320, 3) and image.dtype == np.uint8, 'Invalid native PNG')
        frames.append(image)
    return frames


def risk_checks(baseline, hybrid, prior):
    """Prior AI brackets are observations, not boundary/event ground truth."""
    result = []
    old_regions = [(r['start_frame'], r['end_frame_exclusive']) for r in baseline['gradual_change_regions']]
    for phase, bounds in prior['source_phase_evidence_brackets'].items():
        if bounds is None:
            continue
        a, b = bounds
        for version, cuts, regions in (
            ('v5', baseline['hard_cut_candidates'], old_regions),
            ('hybrid', hybrid['hard_cut_candidates'], hybrid['gradual_change_regions'])):
            result.append({'phase': phase, 'bracket': bounds, 'version': version,
                           'cuts_inside_bracket': [i for i in cuts if a < i < b],
                           'change_span_overlap_frames': sum(max(0, min(b, y)-max(a, x)) for x, y in regions)})
    return result


def boundary_panel(frames, index, path, sid):
    """Show source, raw edge overlay and aligned edge overlay for one boundary."""
    require(1 <= index < len(frames), 'Not an adjacent boundary')
    m, (a, b, warped) = pair_metrics(gray_frame(frames[index-1]), gray_frame(frames[index]), return_maps=True)
    def tile(image, label):
        out = np.zeros((image.shape[0]+30, image.shape[1], 3), np.uint8)
        out[30:] = image
        cv2.putText(out, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
        return out
    top = []
    for i in (index-1, index, min(index+1, len(frames)-1)):
        top.append(tile(frames[i], f'{sid} original frame {i}'))
    def overlay(old):
        out = np.zeros((*a.shape, 3), np.uint8)
        out[old] = (0, 180, 0)
        out[b] = (0, 0, 230)
        out[old & b] = (255, 255, 255)
        return cv2.resize(out, (320, 240), interpolation=cv2.INTER_NEAREST)
    g = cv2.cvtColor(gray_frame(frames[index]), cv2.COLOR_GRAY2BGR)
    bottom = [tile(overlay(a), 'Raw edges: old green / new red'),
              tile(overlay(warped), 'Aligned edges: white = overlap'),
              tile(cv2.resize(g, (320, 240)), f'ROI only; compensation={m["compensation_used"]}')]
    require(cv2.imwrite(str(path), np.vstack([np.hstack(top), np.hstack(bottom)])), 'Panel write failed')
    return m


def plot_scores(row, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = row['edge']['per_frame']
    x = np.arange(len(m))
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, constrained_layout=True)
    for key, label, color in [('raw', 'Raw full-frame ECR', '#aaaaaa'),
                              ('raw_common_overlap', 'Raw ECR in same overlap', '#cc9340'),
                              ('aligned', 'Aligned ECR (accepted fits only)', '#2765b0')]:
        axes[0].plot(x, [r[key]['ecr'] if r[key] and r[key].get('valid') else np.nan for r in m],
                     label=label, color=color, linewidth=1.)
    axes[0].axhline(EdgePolicy().abrupt_ecr, color='#bc3030', linestyle=':', label='Fixed abrupt threshold')
    axes[0].set(ylabel='Unmatched edge fraction', ylim=(-.03, 1.03), title=row['sample_id']+' / development diagnostics, not event truth')
    axes[0].legend(loc='upper left', fontsize=8)
    for y, name, cuts, spans, color in (
        (1, 'Frozen v5', row['v5']['hard_cut_candidates'], row['v5_regions'], '#c56b12'),
        (0, 'v5 + edge proposals', row['edge']['hybrid']['hard_cut_candidates'], row['edge']['hybrid']['gradual_change_regions'], '#2765b0')):
        for a, b in spans:
            axes[1].broken_barh([(a, b-a)], (y-.22, .44), facecolors=color, alpha=.35)
        axes[1].scatter(cuts, [y]*len(cuts), marker='|', s=180, color=color)
    axes[1].set(yticks=[0, 1], yticklabels=['v5 + edge', 'Frozen v5'], xlabel='Original frame index (edge i is between i-1 and i)', ylim=(-.5, 1.5))
    axes[1].grid(axis='x', alpha=.2)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render(output, report):
    cards, table = [], []
    for row in report['videos']:
        sid = row['sample_id']
        v, h = row['v5'], row['edge']['hybrid']
        old, new = str(v['hard_cut_candidates']), str(h['hard_cut_candidates'])
        table.append(f'<tr><td><a href="#{sid}">{sid}</a></td><td>{old}</td><td>{new}</td><td>{row["v5_regions"]} → {h["gradual_change_regions"]}</td><td>{row["prior_event_status"]}</td></tr>')
        imgs = ''.join(f'<figure><figcaption>{html.escape(p["purpose"])}, edge {p["edge_index"]}</figcaption><a href="{p["path"]}"><img loading="lazy" src="{p["path"]}"></a></figure>' for p in row['evidence_panels'])
        cards.append(f'<section id="{sid}"><h2>{sid}</h2><p>運動補償可用 {row["compensation_used_pairs"]}/{row["frame_count"]-1} 對；新增切點 {h["added_cuts_vs_v5"]}，移除 {h["removed_cuts_vs_v5"]}。既有事件初審：{row["prior_event_status"]}，本輪未升級為完整事件。</p><img src="plots/{sid}.png">{imgs}<details><summary>準備／出手區間風險對照（沿用舊AI觀察，非人工真值）</summary><pre>{html.escape(json.dumps(row["known_phase_risk_checks"],ensure_ascii=False,indent=2))}</pre></details></section>')
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>邊緣結構＋運動補償固定八例比較</title><style>body{font:17px system-ui;line-height:1.8;max-width:1200px;margin:30px auto;padding:0 20px;color:#192333}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #ccd}section{border-top:1px solid #ccc;margin-top:30px}pre{white-space:pre-wrap}.notice{background:#fff1cc;padding:16px}figure{margin:20px 0}</style><h1>v5＋邊緣結構：固定八例診斷</h1><p class="notice">只使用既存train原生PNG。單一參數版本，未看輸出後調整閾值。這是已知開發案例，不是盲測；沒有完整場景邊界真值，不能計算精確率或宣稱選段改善。原三窗口、標籤、快取、TANet均保持不動。</p><p>方法：Canny輪廓 → 前後向LK特徵追蹤 → RANSAC相似變換 → 共用有效範圍內的雙向邊緣變化比率。拉普拉斯變異數只記錄作診斷，沒有作為刪片／選段條件。參數、來源與限制見JSON與說明。</p><p><a href="sampling_plan.json">看像素前固定的計畫</a> · <a href="report.json">全部數據</a> · <a href="review.html">AI視覺初審（完成後）</a></p><table><tr><th>案例</th><th>v5切點</th><th>v5＋edge切點</th><th>疑似變化區間</th><th>沿用事件初審</th></tr>'''+''.join(table)+'</table>'+''.join(cards)+'</html>'
    (output/'index.html').write_text(page, encoding='utf-8')


def execute(dry_run=False):
    started = time.monotonic()
    plan, parent_rows, baselines, ledger = fixed_plan()
    if dry_run:
        return {'case_ids': list(CASE_IDS), 'PNG_frames': sum(c['frame_count'] for c in plan['cases']),
                'writes': False, 'video_opens': 0, 'edge_policy': plan['edge_policy']}
    cv2.setNumThreads(1)
    cv2.setRNGSeed(EdgePolicy().seed)
    output = Path(tempfile.mkdtemp(prefix='edge_probe_v1_', dir=ARTIFACTS))
    for name in ('inputs', 'panels', 'plots'):
        (output/name).mkdir()
    # Save the plan before any PNG is opened; failures remain visible in state.
    save(output/'sampling_plan.json', plan)
    protected_paths = [PARENT/'report.json', PARENT/'AI_review.json', PARENT/'sampling_plan.json',
                       PARENT/'run_state.json', BASELINE/'report.json', BASELINE/'policy_v5.json']
    protected_paths += [p for folder in (FOLDER, ROOT/'development/tanet_v1', ROOT/'development/data_pipeline')
                        for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py', '.md', '.json')
                        and 'artifacts' not in p.relative_to(folder).parts and '__pycache__' not in p.parts]
    protected_paths += [p for p in (PARENT/'reviews').rglob('*') if p.is_file()]
    protected = {str(p): sha(p) for p in protected_paths}
    before = cache_inventory()
    prior = {a['sample_id']: a for a in read(PARENT/'AI_review.json')['assessments']}
    for name in ('edge_structure_v1.py', 'edge_probe_v1.py', 'selector.py', 'selector_v2.py', 'selector_v3.py', 'selector_v4.py', 'selector_v5.py'):
        shutil.copy2(FOLDER/name, output/'inputs'/name)
    save(output/'run_state.json', {'status': 'running', 'completed': 0})
    print('OUTPUT:', output, flush=True)
    rows = []
    try:
        for case in plan['cases']:
            tick = time.monotonic()
            sid = case['sample_id']
            frames = load_native(case)
            baseline = analyze_v5(frames)
            require(canonical_hash(baseline) == canonical_hash(baselines[sid]['v5']), 'Frozen v5 replay mismatch: '+sid)
            edge = analyze_edges(frames, baseline)
            hybrid = edge['hybrid']
            # Show every changed proposal endpoint, all old cut boundaries, and
            # one strongest motion correction. No manual cherry-picked frame.
            choices = {}
            for i in baseline['hard_cut_candidates']:
                choices[i] = 'Existing v5 cut control'
            for i in hybrid['added_cuts_vs_v5']+hybrid['removed_cuts_vs_v5']:
                choices[i] = 'Changed hard-cut proposal'
            for check in hybrid['gradual_checks']:
                for i in set(check['v5_span']+check['proposed_span']):
                    if 1 <= i < len(frames):
                        choices[i] = 'Gradual span endpoint comparison'
            eligible = [m for m in edge['per_frame'][1:] if m['raw_minus_aligned'] is not None]
            if eligible:
                top = max(eligible, key=lambda m: (m['raw_minus_aligned'], -m['frame_index']))
                choices.setdefault(top['frame_index'], 'Largest measured motion correction (selected mechanically)')
            panels = []
            for i, purpose in sorted(choices.items()):
                rel = f'panels/{sid}_edge{i:03d}.png'
                boundary_panel(frames, i, output/rel, sid)
                panels.append({'edge_index': i, 'path': rel, 'purpose': purpose})
            metrics = edge['per_frame'][1:]
            row = {**{k: case[k] for k in ('sample_id', 'video_id', 'frame_count', 'split')},
                   'v5': baseline, 'v5_replay_identical': True,
                   'v5_regions': [[r['start_frame'], r['end_frame_exclusive']] for r in baseline['gradual_change_regions']],
                   'edge': edge, 'evidence_panels': panels,
                   'compensation_used_pairs': sum(m['compensation_used'] for m in metrics),
                   'motion_rejection_reasons': dict(Counter(m['motion']['reason'] for m in metrics if not m['motion']['accepted'])),
                   'prior_event_status': prior[sid]['status'],
                   'prior_event_review_sha256': sha(PARENT/'AI_review.json'),
                   'known_phase_risk_checks': risk_checks(baseline, hybrid, prior[sid]),
                   'effective_windows': select_covering_windows(len(frames)),
                   'production_crop_adopted': False, 'training_use_approved': False,
                   'seconds': time.monotonic()-tick}
            plot_scores(row, output/'plots'/f'{sid}.png')
            rows.append(row)
            save(output/'run_state.json', {'status': 'running', 'completed': len(rows)})
            print(sid, 'cuts', baseline['hard_cut_candidates'], '->', hybrid['hard_cut_candidates'],
                  'spans', row['v5_regions'], '->', hybrid['gradual_change_regions'],
                  f'{row["seconds"]:.2f}s', flush=True)
        require(all(sha(p) == h for p, h in protected.items()), 'Existing source/history changed')
        require(all(sha(PARENT/f['path']) == f['sha256'] for c in plan['cases'] for f in c['native_frames']), 'PNG changed during run')
        after = cache_inventory()
        require(before == after, 'Original cache metadata inventory changed')
        access = ledger.statistics()
        require(all(v['opens'] == 0 for v in access.values()), 'Unexpected original video access')
        report = {'status': 'comparison_complete_visual_review_pending', 'sampling_plan_sha256': sha(output/'sampling_plan.json'),
                  'videos': rows, 'protected_sha256': protected, 'cache_inventory_before': before,
                  'cache_inventory_after': after, 'original_video_access': access,
                  'PNG_frames_read': sum(c['frame_count'] for c in plan['cases']),
                  'runtime_seconds': time.monotonic()-started,
                  'environment': {'opencv': cv2.__version__, 'numpy': np.__version__, 'opencv_threads': 1},
                  'validation_test_pixels_read': False, 'feature_tensors_read_or_written': False,
                  'training_started': False, 'TANet_modified': False, 'production_crop_adoptions': 0,
                  'classification_accuracy_computed': False, 'boundary_precision_recall_computed': False,
                  'limitations': ['Known purposeful8, no held-out/source-blind evaluation',
                    'No boundary truth; proposals and edge scores cannot establish improved accuracy',
                    'Similarity fit cannot fully model zoom/parallax/nonrigid players; low texture becomes unknown',
                    'Motion gain compares unaligned/aligned edges on the identical valid overlap; raw full-frame scores also retained',
                    'Reuses frozen v5 flash suppression; not an independent flash detector',
                    'New gradual endpoints remain appearance proposals, not certified clean frames',
                    'No target event selected; all existing event uncertainty/rejection retained']}
        save(output/'report.json', report)
        render(output, report)
        save(output/'run_state.json', {'status': 'comparison_complete_visual_review_pending', 'completed': 8})
        return output
    except BaseException as exc:
        save(output/'run_state.json', {'status': 'failed', 'completed': len(rows), 'error': f'{type(exc).__name__}: {exc}'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    print(execute(parser.parse_args().dry_run))
