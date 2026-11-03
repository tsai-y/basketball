"""Frozen prediction comparison + bounded16 validation visual audit. No model forward.

Uses TANet best.pt's saved validation logits and the mainline's saved prediction
CSV, joined by ID. Never trains, recalibrates a threshold, relabels, or opens test.
"""
from collections import Counter
import argparse
import csv
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import torch

from development.data_pipeline.pipeline import AccessLedger, SplitGuard
from development.data_pipeline.multiwindow_v1 import select_covering_windows
from development.temporal_selection_v1.batch_probe_v1 import cache_inventory
from development.temporal_selection_v1.probe import encode
from .config import FOLDER, InputConfig, need, sha256
from .data import normalize_rgb, uniform_indices
from .training import atomic_save, contract

ROOT = FOLDER.parents[1]
TANET = FOLDER/'artifacts/formal_train_ji1xnlg_'
MAIN = ROOT/'development/temporal_baseline_v1/artifacts/temporal_seed20261025_cwac4gwz'
SEED = 20261029
PATTERNS = ('both_correct', 'tanet_only_correct', 'main_only_correct', 'both_wrong')
PATTERN_ZH = {'both_correct': '兩模型皆正確', 'tanet_only_correct': '只有TANet正確',
              'main_only_correct': '只有主線正確', 'both_wrong': '兩模型皆錯誤'}


def read(path):
    return json.loads(Path(path).read_text())


def aggregate(rows, prediction):
    need(rows, 'Empty predictions')
    cm = [[0, 0], [0, 0]]
    for row in rows:
        y, p = row['true_label'], row[prediction]
        need(type(y) is int and type(p) is int and y in (0, 1) and p in (0, 1), 'Invalid binary prediction')
        cm[y][p] += 1
    per_class = {}
    for k, name in enumerate(('Miss', 'Make')):
        tp, actual, predicted = cm[k][k], sum(cm[k]), sum(cm[j][k] for j in (0, 1))
        per_class[name] = {'support': actual, 'precision': tp/predicted if predicted else 0.,
            'recall': tp/actual if actual else None, 'f1': 2*tp/(actual+predicted) if actual+predicted else 0.}
    return {'videos': len(rows), 'accuracy': (cm[0][0]+cm[1][1])/len(rows),
            'macro_f1': sum(p['f1'] for p in per_class.values())/2,
            'confusion_matrix': cm, 'per_class': per_class,
            'predicted_Make_count': sum(cm[j][1] for j in (0, 1))}


def align_predictions(tanet_rows, main_rows, allowed):
    def index(rows):
        result = {r['video_id']: r for r in rows}
        need(len(result) == len(rows), 'Duplicate prediction ID')
        return result
    ta, ma = index(tanet_rows), index(main_rows)
    need(set(ta) == set(ma) == set(allowed), 'Prediction IDs differ from complete canonical validation')
    result = []
    for vid in sorted(allowed):
        row = allowed[vid]
        need(row['split'] == 'validation' and str(row['excluded']) == '0', 'Nonvalidation/excluded metadata')
        y = int(row['binary_label'])
        need(ta[vid]['true_label'] == ma[vid]['true_label'] == y, 'Saved/canonical label mismatch')
        a, b = ta[vid]['prediction'], ma[vid]['prediction']
        need(type(a) is int and type(b) is int and a in (0, 1) and b in (0, 1), 'Invalid predictions')
        for p in (ta[vid]['p_make'], ma[vid]['p_make']):
            need(math.isfinite(p) and 0 <= p <= 1, 'Invalid probability')
        pattern = ('both_correct' if a == y and b == y else
                   'tanet_only_correct' if a == y else 'main_only_correct' if b == y else 'both_wrong')
        result.append({'video_id': vid, 'true_label': y, 'tanet_prediction': a, 'main_prediction': b,
            'tanet_p_make': ta[vid]['p_make'], 'main_p_make': ma[vid]['p_make'], 'pattern': pattern,
            'split': 'validation', 'source_group': row['source_group'], 'shot_type': row['original_class'][:-1]})
    return result


def rank(value):
    return hashlib.sha256(f'tanet-main-fixed16:{SEED}:{value}'.encode()).hexdigest()


def select_cases(rows, per_stratum=2):
    """2 per truth x agreement cell, spread sources before pixels; no probability rank."""
    need(type(per_stratum) is int and per_stratum > 0, 'Invalid case budget')
    need(len({r['video_id'] for r in rows}) == len(rows) and
         all(r['split'] == 'validation' for r in rows), 'Canonical validation only')
    selected, counts, used_groups = [], [], Counter()
    for y in (0, 1):
        for pattern in PATTERNS:
            pool = [r for r in rows if r['true_label'] == y and r['pattern'] == pattern]
            local_groups = set()
            chosen = []
            for _ in range(min(per_stratum, len(pool))):
                candidate = min(pool, key=lambda r: (r['source_group'] in local_groups,
                    used_groups[r['source_group']], rank(r['source_group']), rank(r['video_id'])))
                pool.remove(candidate)
                used_groups[candidate['source_group']] += 1
                local_groups.add(candidate['source_group'])
                chosen.append(candidate)
                selected.append({**candidate, 'sample_id': f'A{len(selected)+1:03d}'})
            counts.append({'true_label': y, 'pattern': pattern, 'available': len(pool)+len(chosen),
                           'requested': per_stratum, 'selected': len(chosen)})
    return selected, counts


def best_epoch(history):
    need(history and all(math.isfinite(r['validation']['CE']) for r in history), 'Invalid history')
    return min(history, key=lambda r: r['validation']['CE'])['epoch']


def load_saved_predictions(guard):
    torch.set_num_threads(2)
    saved = torch.load(TANET/'best.pt', map_location='cpu', weights_only=True)
    plan, _, _ = contract()
    need(saved['kind'] == 'formal' and saved['contract_sha256'] == plan['contract_sha256'], 'Wrong TANet checkpoint contract')
    need(read(TANET/'run_state.json')['status'] == 'completed', 'Training has not completed')
    history = read(TANET/'history.json')
    need(saved['epoch'] == best_epoch(history) == 10, 'Selection rule changed')
    v = saved['validation']
    ids, logits, labels = v['video_ids'], v['logits'].cpu(), v['labels'].cpu()
    need(logits.shape == (1547, 2) and labels.shape == (1547,) and labels.dtype == torch.int64,
         'Saved validation shapes changed')
    need(torch.isfinite(logits).all().item(), 'Nonfinite saved logits')
    need(abs(float(torch.nn.functional.cross_entropy(logits, labels))-v['metrics']['CE']) < 1e-7, 'TANet CE mismatch')
    probability = logits.softmax(1)[:, 1].tolist()
    predictions = logits.argmax(1).tolist()
    ta = [{'video_id': vid, 'true_label': int(y), 'prediction': int(p), 'p_make': float(prob)}
          for vid, y, p, prob in zip(ids, labels.tolist(), predictions, probability)]
    with (MAIN/'validation_predictions.csv').open(newline='') as stream:
        ma = [{'video_id': r['video_id'], 'true_label': int(r['true_label']),
               'prediction': int(r['predicted_label']), 'p_make': float(r['p_make'])} for r in csv.DictReader(stream)]
    cfg = read(MAIN/'config.json')
    need(cfg['seed'] == plan['recipe']['seed'] == 20261025 and cfg['validation_count'] == 1547,
         'Seed/full validation mismatch')
    need(cfg['selection'] == 'minimum_validation_loss' and cfg['threshold'] == .5, 'Mainline selection changed')
    rows = align_predictions(ta, ma, guard.allowed)
    baseline = read(MAIN/'summary.json')
    for prediction, expected in [('tanet_prediction', v['metrics']), ('main_prediction', baseline['validation'])]:
        actual = aggregate(rows, prediction)
        cm_key = 'confusion_matrix_Miss_Make' if prediction.startswith('tanet') else 'confusion_matrix'
        need(actual['confusion_matrix'] == expected[cm_key], 'Saved confusion matrix mismatch')
        need(abs(actual['accuracy']-expected['accuracy']) < 1e-7 and
             abs(actual['macro_f1']-expected['macro_f1']) < 1e-10, 'Saved accuracy/F1 mismatch')
    del saved
    return rows, history, plan, baseline


def decode_native(case, guard):
    need(case['split'] == 'validation' and guard.split == 'validation', 'Validation only before source access')
    vid = case['video_id']
    need(vid in guard.allowed, 'Noncanonical source rejected')
    before = guard.identity(vid)
    fd = guard.open(vid, guard.allowed[vid]['video_path'], operation='decode_contiguous_rgb_clip')
    import os
    cap, frames = None, []
    try:
        cap = cv2.VideoCapture(f'/proc/self/fd/{fd}', cv2.CAP_FFMPEG)
        need(cap.isOpened(), 'Cannot decode authorized source')
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        need(math.isfinite(n) and n == round(n) and 50 <= n <= 192 and math.isfinite(fps) and fps > 0,
             'Unexpected source length/fps; keep failure, do not replace case')
        for _ in range(round(n)):
            ok, bgr = cap.read()
            need(ok and bgr.dtype == np.uint8 and bgr.shape == (240, 320, 3), 'Invalid/truncated native frame')
            frames.append(bgr)
        need(not cap.read()[0], 'Extra native frames')
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)
    need(guard.identity(vid) == before, 'Source changed during review decode')
    return frames, str(Fraction(fps).limit_denominator(1001)), before


def contact(frames, indices, path, *, title, size=(320, 240), columns=4):
    tiles = []
    for i in indices:
        tile = np.zeros((size[1]+25, size[0], 3), np.uint8)
        tile[25:] = cv2.resize(frames[i], size, interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f'{title} f{i}', (3, 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    need(cv2.imwrite(str(path), np.vstack([np.hstack(tiles[j:j+columns]) for j in range(0, len(tiles), columns)])), 'Contact sheet write failed')


def plot_reports(output, history, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with (MAIN/'history.csv').open(newline='') as f:
        main_history = list(csv.DictReader(f))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    epochs = [r['epoch'] for r in history]
    ax[0].plot(epochs, [r['train_CE'] for r in history], label='TANet train CE')
    ax[0].plot(epochs, [r['validation']['CE'] for r in history], label='TANet validation CE')
    ax[0].axvline(best_epoch(history), color='#555', linestyle=':', label='Selected epoch10')
    ax[0].set(title='TANet: fixed loss-based checkpoint selection', xlabel='Epoch', ylabel='Cross entropy')
    ax[1].plot(epochs, [r['validation']['accuracy'] for r in history], label='TANet validation accuracy')
    ax[1].plot(epochs, [r['validation']['macro_f1'] for r in history], label='TANet validation macro-F1')
    ax[1].axvline(10, color='#555', linestyle=':')
    ax[1].set(title='Accuracy/F1 do not reselect checkpoint', xlabel='Epoch', ylabel='Score')
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=.2)
    fig.savefig(output/'plots/learning_curves.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for a, key, title in zip(axes, ['main_prediction', 'tanet_prediction'], ['MixNet-I3D + Transformer', 'TANet']):
        cm = np.array(aggregate(rows, key)['confusion_matrix'])
        a.imshow(cm, cmap='Blues', vmin=0, vmax=807)
        for i in (0, 1):
            for j in (0, 1):
                a.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > 500 else 'black')
        a.set(xticks=[0, 1], xticklabels=['Miss', 'Make'], yticks=[0, 1], yticklabels=['Miss', 'Make'],
              xlabel='Predicted', ylabel='Dataset label', title=title)
    fig.savefig(output/'plots/confusion_matrices.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    x = np.arange(4)
    labels = ['Miss recall', 'Make recall', 'Accuracy', 'Macro-F1']
    for shift, key, name in [(-.18, 'main_prediction', 'Mainline'), (.18, 'tanet_prediction', 'TANet')]:
        m = aggregate(rows, key)
        values = [m['per_class']['Miss']['recall'], m['per_class']['Make']['recall'], m['accuracy'], m['macro_f1']]
        ax.bar(x+shift, values, width=.36, label=name)
    ax.set(xticks=x, xticklabels=labels, ylim=(0, 1), title='Same1547 validation videos / seed20261025')
    ax.legend()
    fig.savefig(output/'plots/class_comparison.png', dpi=140)
    plt.close(fig)
    # Keep the mainline history as evidence, without aligning epoch budgets as
    # though the frozen-feature and RGB optimizers had equivalent update costs.
    atomic_save(output/'mainline_history.json', main_history, json_format=True)


def render(output, report):
    m, t = report['metrics']['mainline'], report['metrics']['tanet']
    def pct(x):
        return f'{100*x:.2f}%'
    measures = [('Accuracy', pct(m['accuracy']), pct(t['accuracy'])), ('Macro-F1', f'{m["macro_f1"]:.4f}', f'{t["macro_f1"]:.4f}')]
    for name in ('Miss', 'Make'):
        measures.append((name+' recall', pct(m['per_class'][name]['recall']), pct(t['per_class'][name]['recall'])))
    table = ''.join(f'<tr><th>{a}</th><td>{b}</td><td>{c}</td></tr>' for a, b, c in measures)
    paired = ''.join(f'<tr><th>{"Miss" if y == 0 else "Make"}</th>'+''.join(f'<td>{report["paired_counts_by_label"][str(y)][k]}</td>' for k in PATTERNS)+'</tr>' for y in (0, 1))
    groups = []
    for row in report['cases']:
        sid = row['sample_id']
        p = row['pattern']
        groups.append(f'<section class="case" data-pattern="{p}" id="{sid}"><h2>{sid}：{PATTERN_ZH[p]}</h2><p>資料集標籤：{"Make" if row["true_label"] else "Miss"}；TANet P(Make)={row["tanet_p_make"]:.3f}，主線={row["main_p_make"]:.3f}。這些是未校準的模型分數，不能代替人工結果。<br>來源 {html.escape(row["source_group"])}，類型 {html.escape(row["shot_type"])}；{html.escape(row["video_id"])}</p><video controls preload="none" src="previews/{sid}.mp4"></video><p>原片12個固定等距影格總覽（非完整逐幀初審）</p><a href="contacts/{sid}_overview.png"><img loading="lazy" src="contacts/{sid}_overview.png"></a><details><summary>全部原生影格、實際輸入與時間座標</summary><p>原生滑桿：<input data-sid="{sid}" type="range" min="0" max="{row["frame_count"]-1}" value="0"><span id="label{sid}">f0</span></p><img id="native{sid}" class="native" src="native/{sid}/f000000.png"><p>TANet全部50張實際縮放輸入；標示原片frame index。</p><img loading="lazy" src="contacts/{sid}_tanet50.png"><p>主線仍為三個連續64幀窗口，完整窗口如下；重複是窗口重疊，不代表影片有重複幀。</p>'+''.join(f'<img loading="lazy" src="contacts/{sid}_main{j}.png">' for j in range(3))+'</details></section>')
    page='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TANet基準分析與主線錯誤對照</title><style>body{font:18px system-ui;line-height:1.8;max-width:1250px;margin:30px auto;padding:0 20px;color:#192333}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:10px}img{max-width:100%;height:auto}.native{width:640px;image-rendering:pixelated}video{width:640px;max-width:100%}.notice{padding:16px;background:#fff1cc}section{border-top:1px solid #ccc;margin:30px 0;padding-top:10px}input{width:60%}button{font-size:16px;padding:8px;margin:4px}</style><h1>TANet基準分析與主線錯誤對照</h1><p class="notice">同seed20261025、完整1547支validation。TANet取全片50幀且端到端訓練；主線取三個連續64幀窗口、骨幹凍結。這是兩個既有系統的描述比較，不是純架構消融或穩定優勢證明。未重訓、未改標籤／門檻、未執行test。</p><p><a href="review.html">AI初審與結論（完成後）</a> · <a href="report.json">完整數據</a> · <a href="paired_predictions.csv">逐片對照</a> · <a href="sampling_plan.json">看影格前固定的16例</a></p><table><tr><th>指標</th><th>MixNet-I3D＋Transformer</th><th>TANet</th></tr>'''+table+'''</table><img src="plots/learning_curves.png"><img src="plots/confusion_matrices.png"><img src="plots/class_comparison.png"><h2>同一影片的判對／判錯組合</h2><table><tr><th>資料集標籤</th>'''+''.join('<th>'+PATTERN_ZH[k]+'</th>' for k in PATTERNS)+'</tr>'+paired+'''</table><p>下面刻意每類、每種組合抽2例（共16例），包含判對對照。樣本比例不代表整體錯誤分布；選取不使用信心高低、不看像素後換片、不將AI觀察視為人工真值。</p><div><button onclick="filterCases('all')">全部</button>'''+''.join(f'<button onclick="filterCases(\'{k}\')">{PATTERN_ZH[k]}</button>' for k in PATTERNS)+'</div>'+''.join(groups)+'''<script>function filterCases(k){document.querySelectorAll('.case').forEach(e=>e.hidden=k!=='all'&&e.dataset.pattern!==k)}document.querySelectorAll('input').forEach(e=>e.addEventListener('input',()=>{let sid=e.dataset.sid,i=Number(e.value);document.getElementById('native'+sid).src='native/'+sid+'/f'+String(i).padStart(6,'0')+'.png';document.getElementById('label'+sid).textContent='f'+i;}));</script></html>'''
    (output/'index.html').write_text(page, encoding='utf-8')


def execute(dry_run=False):
    ledger = AccessLedger()
    guard = SplitGuard('validation', ledger)
    rows, history, training_plan, baseline = load_saved_predictions(guard)
    cases, strata = select_cases(rows)
    need(len(cases) == 16, 'Expected16 fixed examples; do not fill missing strata after inspection')
    protected_paths = [TANET/p for p in ('best.pt', 'last.pt', 'history.json', 'contract.json', 'run_state.json')]
    protected_paths += [MAIN/p for p in ('best.pt', 'history.csv', 'summary.json', 'config.json', 'validation_predictions.csv')]
    protected_paths += [FOLDER/p for p in ('model.py', 'data.py', 'config.py', 'config.json', 'training.py', 'train.py', 'readiness.py', 'train_config.json', 'upstream/temporal_module.py', 'upstream/PROVENANCE.json')]
    protected_paths += [ROOT/'data/processed/dataset_manifest.csv', ROOT/'data/processed/split_summary.json', Path(__file__)]
    protected = {str(p): sha256(p) for p in protected_paths}
    plan = {'schema': 'tanet_main_validation_fixed16_before_pixels_v1', 'seed': SEED,
            'cases': cases, 'strata': strata, 'selection': '2 per dataset label x correctness-pattern cell, source-spread then SHA rank; no confidence/pixel ranking',
            'labels_used_only_for_post_training_analysis': True, 'no_case_replacement_after_pixels': True,
            'validation_not_test': True, 'new_human_labels': False, 'protected_sha256': protected,
            'training_contract_sha256': training_plan['contract_sha256']}
    if dry_run:
        return {'cases': cases, 'strata': strata, 'video_opens': 0, 'model_forwards': 0, 'writes': False}
    output = Path(tempfile.mkdtemp(prefix='analysis_v1_', dir=FOLDER/'artifacts'))
    for name in ('plots', 'contacts', 'native', 'previews', 'inputs'):
        (output/name).mkdir()
    atomic_save(output/'sampling_plan.json', plan, json_format=True)
    shutil.copy2(Path(__file__), output/'inputs/analyze_v1.py')
    before = cache_inventory()
    atomic_save(output/'run_state.json', {'status': 'preparing', 'completed_cases': 0}, json_format=True)
    print('OUTPUT:', output, flush=True)
    try:
        plot_reports(output, history, rows)
        with (output/'paired_predictions.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        completed = []
        cv2.setNumThreads(1)
        for case in cases:
            sid = case['sample_id']
            frames, fps, identity = decode_native(case, guard)
            n = len(frames)
            selected = uniform_indices(n)
            windows = select_covering_windows(n)
            rgb = np.stack([cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), (256, 192), interpolation=cv2.INTER_AREA) for i in selected])
            tensor = normalize_rgb(rgb, InputConfig())
            need(tensor.shape == (3, 50, 192, 256) and torch.isfinite(tensor).all().item(), 'Input reconstruction failed')
            inputs = [cv2.cvtColor(r, cv2.COLOR_RGB2BGR) for r in rgb]
            # Use the exact selected original indices as contact labels.
            actual = [None]*n
            for i, image in zip(selected, inputs):
                actual[i] = image
            (output/'native'/sid).mkdir()
            for i, frame in enumerate(frames):
                need(cv2.imwrite(str(output/'native'/sid/f'f{i:06d}.png'), frame), 'Native PNG write failed')
            overview = sorted({i*(n-1)//11 for i in range(12)})
            contact(frames, overview, output/'contacts'/f'{sid}_overview.png', title=sid)
            contact(actual, selected, output/'contacts'/f'{sid}_tanet50.png', title='TANet', size=(256, 192), columns=8)
            for j, (a, b) in enumerate(windows):
                scaled = [cv2.resize(f, (256, 192), interpolation=cv2.INTER_AREA) for f in frames]
                contact(scaled, list(range(a, b)), output/'contacts'/f'{sid}_main{j}.png', title=f'Main W{j}', size=(160, 120), columns=8)
            encode(frames, output/'previews'/f'{sid}.mp4', fps)
            completed.append({**case, 'frame_count': n, 'fps_rational': fps, 'source_identity': identity,
                'overview_indices': overview, 'tanet_indices': list(selected), 'main_windows': windows,
                'main_windows_cover_full_source': {i for a, b in windows for i in range(a, b)} == set(range(n)),
                'input_reconstruction': 'Same pinned selection/resize/normalization functions; no independent model-forward or full loader reread',
                'input_tensor_sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                'AI_visual_review': 'pending', 'original_label_changed': False})
            atomic_save(output/'run_state.json', {'status': 'preparing', 'completed_cases': len(completed)}, json_format=True)
            print(sid, case['pattern'], 'label', case['true_label'], 'frames', n, flush=True)
        need(all(sha256(k) == h for k, h in protected.items()), 'Protected checkpoint/data/source changed')
        after = cache_inventory()
        need(before == after, 'Original cache inventory changed')
        access = ledger.statistics()
        need(access['validation']['opens'] == 16 and access['train']['opens'] == access['test']['opens'] == 0,
             'Unexpected source access')
        main_metrics, tanet_metrics = aggregate(rows, 'main_prediction'), aggregate(rows, 'tanet_prediction')
        report = {'status': 'numeric_and_case_materials_complete_AI_review_pending',
            'comparison_seed': 20261025, 'validation_videos': len(rows),
            'metrics': {'mainline': main_metrics, 'tanet': tanet_metrics},
            'losses': {'mainline_saved_BCE': baseline['validation']['loss'], 'tanet_saved_CE': min(history, key=lambda r: r['validation']['CE'])['validation']['CE']},
            'best_epochs': {'mainline': baseline['best_epoch'], 'tanet': best_epoch(history)},
            'paired_counts': dict(Counter(r['pattern'] for r in rows)),
            'paired_counts_by_label': {str(y): {k: sum(r['true_label'] == y and r['pattern'] == k for r in rows) for k in PATTERNS} for y in (0, 1)},
            'correct_count_difference_TANet_minus_mainline': round((tanet_metrics['accuracy']-main_metrics['accuracy'])*len(rows)),
            'per_source': {g: {'mainline': aggregate([r for r in rows if r['source_group'] == g], 'main_prediction'),
                              'tanet': aggregate([r for r in rows if r['source_group'] == g], 'tanet_prediction')} for g in sorted({r['source_group'] for r in rows})},
            'per_shot_type': {g: {'mainline': aggregate([r for r in rows if r['shot_type'] == g], 'main_prediction'),
                                 'tanet': aggregate([r for r in rows if r['shot_type'] == g], 'tanet_prediction')} for g in sorted({r['shot_type'] for r in rows})},
            'cases': completed, 'source_access': access, 'model_forward_calls': 0, 'optimizer_steps': 0,
            'cache_inventory_before': before, 'cache_inventory_after': after, 'protected_sha256': protected,
            'labels_thresholds_weights_changed': False, 'test_used': False, 'new_training_started': False,
            'limitations': ['Single TANet seed; systems differ in input, trainable backbone, optimizer and training budget',
                'Validation examples already used for model selection; not a new held-out evaluation',
                'Probabilities are uncalibrated; sampled cases stratified by dataset labels and saved correctness',
                'Visual observations do not establish causes, prove label truth or authorize exclusions',
                'No new predictions or automatic fusion; no model-forward during this analysis']}
        atomic_save(output/'report.json', report, json_format=True)
        render(output, report)
        atomic_save(output/'run_state.json', {'status': 'numeric_and_case_materials_complete_AI_review_pending', 'completed_cases': 16}, json_format=True)
        return output
    except BaseException as exc:
        atomic_save(output/'run_state.json', {'status': 'failed', 'completed_cases': len(locals().get('completed', [])),
                    'error': f'{type(exc).__name__}: {exc}'}, json_format=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    print(execute(parser.parse_args().dry_run))
