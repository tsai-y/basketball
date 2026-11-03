"""Frozen three-seed validation error distributions and fixed24 visual audit.

No model construction/forward, optimizer, test access, relabeling or tuning.
Sampling is committed before pixels; previews and AI observations are diagnostic.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
from fractions import Fraction
import hashlib
import html
import importlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch

from development.data_pipeline.pipeline import AccessLedger, SplitGuard
from development.tanet_v1.analyze_v1 import aggregate, align_predictions, contact, decode_native
from development.temporal_selection_v1.probe import encode
from . import campaign
from .config import FOLDER, InputConfig, need, sha256
from .data import normalize_rgb, uniform_indices
from .training import atomic_save, metrics

CAMPAIGN = FOLDER/'artifacts/three_seed_cew8zqhi'
SEEDS = (20261024, 20261025, 20261026)
SAMPLE_SEED = 20261031
SHOTS = ('2p', '3p', 'ft', 'mp')
BUCKETS = ('common_wrong', 'seed_disagreement', 'all_correct')
BUCKET_ZH = {'common_wrong': '三 seed 共同誤判', 'seed_disagreement': 'seed 判斷分歧', 'all_correct': '三 seed 全對對照'}


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat()


def join_seeds(paired, allowed):
    """Complete canonical ID join, never row-order joins or majority relabels."""
    need(set(paired) == set(SEEDS), 'Require all three fixed seeds')
    indexed = {}
    for seed, rows in paired.items():
        indexed[seed] = {r['video_id']: r for r in rows}
        need(len(indexed[seed]) == len(rows) and set(indexed[seed]) == set(allowed), 'Duplicate/missing/foreign ID')
    merged = []
    for vid in sorted(allowed):
        meta = allowed[vid]
        need(meta['split'] == 'validation' and str(meta['excluded']) == '0', 'Validation only')
        y = int(meta['binary_label'])
        row = {'video_id': vid, 'split': 'validation', 'true_label': y,
               'source_group': meta['source_group'], 'original_class': meta['original_class'],
               'shot_type': meta['original_class'][:-1]}
        need(y in (0, 1) and row['shot_type'] in SHOTS, 'Invalid label/shot type')
        for seed in SEEDS:
            saved = indexed[seed][vid]
            need(saved['split'] == 'validation' and saved['true_label'] == y and
                 saved['source_group'] == row['source_group'] and saved['shot_type'] == row['shot_type'],
                 'Prediction/canonical metadata mismatch')
            for model, prefix in (('tanet', 'L'), ('main', 'main')):
                pred, prob = saved[model+'_prediction'], saved[model+'_p_make']
                need(type(pred) is int and pred in (0, 1) and math.isfinite(prob) and 0 <= prob <= 1,
                     'Invalid saved prediction')
                row[f'{prefix}_{seed}_prediction'] = pred
                row[f'{prefix}_{seed}_p_make'] = prob
        count = sum(row[f'L_{s}_prediction'] == y for s in SEEDS)
        row.update(L_correct_seeds=count,
                   main_correct_seeds=sum(row[f'main_{s}_prediction'] == y for s in SEEDS),
                   bucket='common_wrong' if count == 0 else 'all_correct' if count == 3 else 'seed_disagreement')
        merged.append(row)
    return merged


def load_frozen(guard):
    summary, plan, state = (read(CAMPAIGN/name) for name in ('summary.json', 'plan.json', 'state.json'))
    need(summary['status'] == state['status'] == 'completed' and summary['test_used'] is False and
         summary['seeds'] == list(SEEDS), 'Incomplete/foreign campaign')
    need(sha256(CAMPAIGN/'plan.json') == state['plan_sha256'], 'Campaign plan changed')
    campaign.verify_sources(plan)
    protected = dict(plan['protected_sha256'])
    for name in ('plan.json', 'state.json', 'launcher.json', 'summary.json', 'index.html'):
        protected[str(CAMPAIGN/name)] = sha256(CAMPAIGN/name)
    paired = {}
    for detail in summary['details']:
        seed, run = detail['seed'], Path(detail['TANet_run'])
        need(seed in SEEDS and read(run/'run_state.json')['status'] == 'completed', 'Unfinished seed')
        item = next(x for x in plan['queue'] if x['seed'] == seed)
        contract = importlib.import_module(item['module']+'.training').contract()[0]
        need(contract == read(run/'contract.json') and contract['contract_sha256'] == item['contract_sha256'],
             'Pinned formal contract changed')
        saved = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        history = read(run/'history.json')
        need(saved['kind'] == 'formal' and saved['contract_sha256'] == contract['contract_sha256'] and
             saved['epoch'] == min(history, key=lambda h: h['validation']['CE'])['epoch'] == detail['TANet_best_epoch'],
             'Wrong checkpoint selection')
        val = saved['validation']
        need(val['logits'].shape == (1547, 2) and val['labels'].shape == (1547,) and
             len(val['video_ids']) == 1547, 'Incomplete validation tensors')
        actual = metrics(val['logits'], val['labels'])
        need(actual['confusion_matrix_Miss_Make'] == val['metrics']['confusion_matrix_Miss_Make'] == detail['TANet']['confusion_matrix'] and
             abs(actual['CE']-val['metrics']['CE']) < 1e-7 and
             abs(actual['accuracy']-val['metrics']['accuracy']) < 1e-7 and
             abs(actual['macro_f1']-val['metrics']['macro_f1']) < 1e-10,
             'Saved metric mismatch')
        ta = [{'video_id': vid, 'true_label': int(y), 'prediction': int(pred), 'p_make': float(prob)}
              for vid, y, pred, prob in zip(val['video_ids'], val['labels'].tolist(),
                    val['logits'].argmax(1).tolist(), val['logits'].softmax(1)[:, 1].tolist())]
        main = Path(detail['mainline_run'])
        with (main/'validation_predictions.csv').open() as f:
            ma = [{'video_id': r['video_id'], 'true_label': int(r['true_label']),
                   'prediction': int(r['predicted_label']), 'p_make': float(r['p_make'])} for r in csv.DictReader(f)]
        aligned = align_predictions(ta, ma, guard.allowed)
        with (CAMPAIGN/f'paired_seed{seed}.csv').open() as f:
            evidence = list(csv.DictReader(f))
        need(len(evidence) == len(aligned), 'Paired CSV length mismatch')
        evidence = {r['video_id']: r for r in evidence}
        need(len(evidence) == len(aligned), 'Duplicate paired CSV ID')
        for row in aligned:
            old = evidence[row['video_id']]
            for key, value in row.items():
                if key.endswith('_p_make'):
                    need(abs(float(old[key])-value) < 1e-12, 'Paired score mismatch')
                else:
                    need(str(value) == old[key], 'Paired metadata/prediction mismatch')
        paired[seed] = aligned
        for name in ('best.pt', 'last.pt', 'contract.json', 'history.json', 'run_state.json', 'progress.log'):
            protected[str(run/name)] = sha256(run/name)
        for name in ('best.pt', 'config.json', 'summary.json', 'history.csv', 'validation_predictions.csv'):
            protected[str(main/name)] = sha256(main/name)
        protected[str(CAMPAIGN/f'paired_seed{seed}.csv')] = sha256(CAMPAIGN/f'paired_seed{seed}.csv')
    for path in (Path(__file__), FOLDER/'tests/test_diagnose_errors_v1.py', FOLDER/'ERROR_AUDIT_V1.md',
                 campaign.ROOT/'development/tanet_v1/analyze_v1.py',
                 campaign.ROOT/'development/temporal_selection_v1/probe.py'):
        protected[str(path)] = sha256(path)
    return join_seeds(paired, guard.allowed), protected, summary


def distribution(rows):
    need(rows, 'Empty stratum')
    count = Counter(r['L_correct_seeds'] for r in rows)
    n = len(rows)
    by_seed = {str(s): {prefix: aggregate(rows, f'{prefix}_{s}_prediction') for prefix in ('L', 'main')}
               for s in SEEDS}
    return {'support': n, 'correct_seed_counts': {str(k): count[k] for k in range(4)},
            'common_wrong': count[0], 'common_wrong_rate': count[0]/n,
            'seed_disagreement': count[1]+count[2], 'seed_disagreement_rate': (count[1]+count[2])/n,
            'all_correct': count[3], 'all_correct_rate': count[3]/n,
            'mean_L_seed_error_rate': sum(3-r['L_correct_seeds'] for r in rows)/(3*n),
            'mean_main_seed_error_rate': sum(3-r['main_correct_seeds'] for r in rows)/(3*n),
            'both_systems_all_seeds_wrong': sum(r['L_correct_seeds'] == r['main_correct_seeds'] == 0 for r in rows),
            'by_seed': by_seed}


def summarize_groups(rows):
    result = {'overall': distribution(rows)}
    for name, keys in [('source', ('source_group',)), ('shot_type', ('shot_type',)),
                       ('label', ('true_label',)), ('source_label', ('source_group', 'true_label')),
                       ('shot_label', ('shot_type', 'true_label')), ('source_shot', ('source_group', 'shot_type'))]:
        groups = {}
        for row in rows:
            value = tuple(row[k] for k in keys)
            groups.setdefault(value, []).append(row)
        result[name] = [{**dict(zip(keys, key)), **distribution(groups[key])} for key in sorted(groups)]
    return result


def rank(value):
    return hashlib.sha256(f'L-v2-fixed24:{SAMPLE_SEED}:{value}'.encode()).hexdigest()


def select_cases(rows):
    """12 wrong,8 disagreement,4 controls; fixed class quotas/source spread."""
    need(len({r['video_id'] for r in rows}) == len(rows) and
         all(r['split'] == 'validation' for r in rows), 'Unique validation rows only')
    requests = []
    for bucket in ('common_wrong', 'seed_disagreement'):
        for y in (0, 1):
            for shot in SHOTS:
                requests.append((bucket, y, shot, 2 if bucket == 'common_wrong' and shot in ('2p', 'ft') else 1))
    requests += [('all_correct', i % 2, shot, 1) for i, shot in enumerate(SHOTS)]
    selected, strata, used = [], [], Counter()
    for bucket, y, shot, requested in requests:
        pool = [r for r in rows if (r['bucket'], r['true_label'], r['shot_type']) == (bucket, y, shot)]
        need(len(pool) >= requested, 'Missing fixed stratum; stop before pixels, never substitute silently')
        local = set()
        for _ in range(requested):
            row = min(pool, key=lambda r: (r['source_group'] in local, used[r['source_group']],
                                          rank(r['source_group']), rank(r['video_id'])))
            pool.remove(row)
            selected.append({**row, 'sample_id': f'E{len(selected)+1:03d}'})
            local.add(row['source_group'])
            used[row['source_group']] += 1
        strata.append({'bucket': bucket, 'true_label': y, 'shot_type': shot,
                       'available': len(pool)+requested, 'selected': requested})
    need(len(selected) == 24 and len({r['video_id'] for r in selected}) == 24, 'Fixed24 budget mismatch')
    return selected, strata


def temporal_gaps(indices, fps):
    need(len(indices) == 50 and len(set(indices)) == 50 and indices == sorted(indices) and fps > 0,
         'Invalid sampling indices/FPS')
    gaps = [b-a for a, b in zip(indices, indices[1:])]
    return {'min_sample_step_frames': min(gaps), 'max_sample_step_frames': max(gaps),
            'max_omitted_consecutive_frames': max(gaps)-1,
            'max_time_between_samples_seconds': max(gaps)/fps}


def write_csv(path, rows):
    need(rows, 'Empty export')
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot_groups(output, groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, name, key in ((axes[0], 'shot_label', 'shot_type'), (axes[1], 'source', 'source_group')):
        rows = groups[name]
        labels = [r[key]+('/'+('Make' if r['true_label'] else 'Miss') if name == 'shot_label' else '') for r in rows]
        ax.bar(range(len(rows)), [r['common_wrong_rate']*100 for r in rows], color='#aa413c')
        for i, r in enumerate(rows):
            ax.text(i, r['common_wrong_rate']*100+.8, f'{r["common_wrong"]}/{r["support"]}', ha='center', fontsize=8)
        ax.set(xticks=range(len(rows)), xticklabels=labels, ylabel='All-three-seed wrong / stratum videos (%)',
               ylim=(0, max(r['common_wrong_rate']*100 for r in rows)+12), title=name.replace('_', ' ').title())
        ax.tick_params(axis='x', labelrotation=45)
    fig.savefig(output/'plots/common_errors.png', dpi=150)
    plt.close(fig)


def render(output, report):
    def table(rows, key):
        return '<table><tr><th>分組</th><th>總數</th><th>三seed共同誤判</th><th>共同誤判率</th><th>L平均錯誤率</th><th>主線平均錯誤率</th></tr>'+''.join(
            f'<tr><td>{html.escape(str(r[key]))}</td><td>{r["support"]}</td><td>{r["common_wrong"]}</td><td>{100*r["common_wrong_rate"]:.2f}%</td><td>{100*r["mean_L_seed_error_rate"]:.2f}%</td><td>{100*r["mean_main_seed_error_rate"]:.2f}%</td></tr>' for r in rows)+'</table>'
    groups = report['distributions']
    cards = []
    for c in report['cases']:
        sid = c['sample_id']
        scores = ''.join(f'<tr><td>{seed}</td><td>{"Make" if c[f"L_{seed}_prediction"] else "Miss"}</td><td>{c[f"L_{seed}_p_make"]:.4f}</td><td>{"Make" if c[f"main_{seed}_prediction"] else "Miss"}</td><td>{c[f"main_{seed}_p_make"]:.4f}</td></tr>' for seed in SEEDS)
        cards.append(f'''<section class="case" id="{sid}" data-bucket="{c['bucket']}">
<h2>{sid} · {html.escape(c['source_group'])} · {html.escape(c['shot_type'])}</h2>
<p>{html.escape(c['video_id'])}；原片 {c['frame_count']} 幀／{c['frame_count']/c['fps']:.2f} 秒。
先看影片，再展開資料集標籤與模型分數。</p>
<video controls preload="none" src="previews/{sid}.mp4"></video>
<details><summary>顯示分組、資料集標籤與分數（非人工真值）</summary>
<p>{BUCKET_ZH[c['bucket']]}；資料集標籤：{'Make' if c['true_label'] else 'Miss'}</p>
<table><tr><th>seed</th><th>L判斷</th><th>L P(Make)</th><th>主線判斷</th><th>主線 P(Make)</th></tr>{scores}</table></details>
<p>固定12幀總覽（不能代替完整逐幀檢查）</p><img loading="lazy" src="contacts/{sid}_overview.png">
<details><summary>原生逐幀與50幀抽樣對照</summary>
<div class="controls"><button data-step="-1" data-sid="{sid}">前一幀</button>
<input aria-label="{sid} 原生影格" data-sid="{sid}" type="range" min="0" max="{c['frame_count']-1}" value="0">
<button data-step="1" data-sid="{sid}">後一幀</button><span id="position{sid}"></span></div>
<div class="pair"><figure><figcaption>原生320×240（原片逐幀解碼）</figcaption><img id="native{sid}" src="native/{sid}/f000000.png"></figure>
<figure><figcaption>最近的TANet抽樣影格，RGB縮放256×192</figcaption><img id="input{sid}" src="inputs50/{sid}/f000000.png"></figure></div>
<p>取樣覆蓋首尾，全片固定50幀；最大相鄰抽樣距離 {c['sampling_gaps']['max_sample_step_frames']} 幀，
最多連續略過 {c['sampling_gaps']['max_omitted_consecutive_frames']} 幀。未標記出手／結果真值，不能據此自動宣稱保留或漏掉事件。</p>
<a href="contacts/{sid}_tanet50.png"><img loading="lazy" src="contacts/{sid}_tanet50.png"></a>
<p><a href="contacts/{sid}_native_all.png">全部原生影格總覽</a> · <a href="timelines/{sid}.json">影格與秒數座標</a></p></details>
<label>人工回看筆記（只存在此瀏覽器，可匯出；不回寫資料集）<br>
<textarea data-note="{sid}" rows="3" placeholder="出手可見？結果可見？切鏡／遮擋？證據frame範圍？不確定之處？"></textarea></label></section>''')
    data = {c['sample_id']: {'indices': c['tanet_indices'], 'frames': c['frame_count'], 'fps': c['fps']} for c in report['cases']}
    css = 'body{font:17px system-ui;line-height:1.75;max-width:1250px;margin:28px auto;padding:0 20px;color:#233047}table{border-collapse:collapse}td,th{border:1px solid #cbd5df;padding:8px}img{max-width:100%;height:auto}video{width:640px;max-width:100%;background:#111}section{border-top:1px solid #ccd;margin-top:30px;padding-top:10px}button{font-size:16px;padding:6px;margin:5px}textarea{width:95%}.notice{padding:16px;background:#fff0cf}.pair{display:flex;flex-wrap:wrap;gap:18px}figure{margin:5px}figure img{width:480px}input[type=range]{width:50%}details{padding:12px;background:#f3f6fa;margin:10px 0}'
    js = '''const cases=DATA, storeKey="tanet-L-fixed24:"+location.pathname;
function show(sid,i){let c=cases[sid],j=c.indices.reduce((a,b)=>Math.abs(b-i)<Math.abs(a-i)?b:a),pad=n=>String(n).padStart(6,'0');
document.getElementById('native'+sid).src='native/'+sid+'/f'+pad(i)+'.png';document.getElementById('input'+sid).src='inputs50/'+sid+'/f'+pad(j)+'.png';
document.getElementById('position'+sid).textContent='原生 f'+i+' ('+(i/c.fps).toFixed(3)+'s)，最近抽樣 f'+j+(c.indices.includes(i)?'；此幀有被抽取':'；此幀未抽取');}
document.querySelectorAll('input[type=range]').forEach(e=>{show(e.dataset.sid,0);e.addEventListener('input',()=>show(e.dataset.sid,Number(e.value)))});
document.querySelectorAll('[data-step]').forEach(b=>b.addEventListener('click',()=>{let e=document.querySelector('input[data-sid="'+b.dataset.sid+'"]');e.value=String(Math.max(0,Math.min(Number(e.max),Number(e.value)+Number(b.dataset.step))));show(b.dataset.sid,Number(e.value))}));
document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>document.querySelectorAll('.case').forEach(e=>e.hidden=b.dataset.filter!=='all'&&e.dataset.bucket!==b.dataset.filter)));
let notes={};try{notes=JSON.parse(localStorage.getItem(storeKey)||'{}')}catch(e){}
document.querySelectorAll('textarea[data-note]').forEach(e=>{e.value=notes[e.dataset.note]||'';e.addEventListener('input',()=>{notes[e.dataset.note]=e.value;try{localStorage.setItem(storeKey,JSON.stringify(notes))}catch(err){}})});
document.getElementById('export').addEventListener('click',()=>{let blob=new Blob([JSON.stringify({scope:'fixed24 exploratory review; no canonical changes',notes},null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human_review_notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
'''.replace('DATA', json.dumps(data, ensure_ascii=False).replace('<', '\\u003c'))
    page = '<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TANet L v2 錯誤診斷與24片回看</title><style>'+css+'</style><h1>TANet L v2：三 seed 錯誤診斷與固定24片回看</h1>'
    page += '<p class="notice">完整1547支既有validation，依各seed最低CE選模。分組率使用各組影片總數作分母；三seed重複預測不是4641支獨立影片。24片是刻意分層診斷樣本，不能用其命中率代表整體表現。僅檢視既有輸出與原片；沒有模型forward、訓練、改標籤／門檻或test。</p>'
    page += '<p><a href="report.json">完整數據</a> · <a href="sampling_plan.json">讀像素前固定的抽樣計畫</a> · <a href="review.html">AI初審與診斷解讀（完成後）</a> · <a href="predictions.csv">1547片預測</a> · <a href="common_wrong.csv">143片共同誤判</a></p>'
    page += '<img src="plots/common_errors.png"><h2>按來源</h2>'+table(groups['source'], 'source_group')
    page += '<h2>按出手類型</h2>'+table(groups['shot_type'], 'shot_type')
    labeled = [{**r, 'group': r['shot_type']+'/'+('Make' if r['true_label'] else 'Miss')} for r in groups['shot_label']]
    page += '<h2>按出手類型 × 資料集標籤</h2>'+table(labeled, 'group')
    page += '<h2>固定24支回看</h2><p>12支共同誤判、8支seed分歧、4支全對對照；類別配額與來源分散先固定，不依信心分數挑選。影片為有損轉碼預覽；逐幀PNG保留原生解碼與實際縮放影格。</p><button data-filter="all">全部</button>'
    page += ''.join(f'<button data-filter="{b}">{BUCKET_ZH[b]}</button>' for b in BUCKETS)+'<button id="export">匯出人工筆記</button>'
    page += ''.join(cards)+'<script>'+js+'</script></html>'
    (output/'index.html').write_text(page)


def prepare():
    need(not torch.cuda.is_initialized(), 'CPU diagnostic only')
    torch.set_num_threads(2)
    ledger = AccessLedger()
    guard = SplitGuard('validation', ledger)
    rows, protected, summary = load_frozen(guard)
    cases, strata = select_cases(rows)
    groups = summarize_groups(rows)
    need(len(rows) == 1547 and groups['overall']['common_wrong'] == 143 and
         groups['overall']['seed_disagreement'] == 330 and groups['overall']['all_correct'] == 1074,
         'Frozen three-seed totals do not reproduce')
    output = Path(tempfile.mkdtemp(prefix='error_audit_v1_', dir=FOLDER/'artifacts'))
    for name in ('inputs', 'contacts', 'native', 'inputs50', 'previews', 'timelines', 'plots'):
        (output/name).mkdir()
    for name in ('diagnose_errors_v1.py', 'tests/test_diagnose_errors_v1.py', 'ERROR_AUDIT_V1.md'):
        target = output/'inputs'/Path(name).name
        shutil.copy2(FOLDER/name, target)
    plan = {'schema': 'TANet_L_v2_frozen_validation_fixed24_v1', 'created_at_taiwan': stamp(),
            'campaign': str(CAMPAIGN), 'seeds': list(SEEDS), 'sample_seed': SAMPLE_SEED,
            'cases': cases, 'strata': strata, 'protected_sha256': protected,
            'selection': '12 common-wrong (6 per label;2 each 2p/ft,1 each 3p/mp),8 disagreement (1 per label x shot),4 all-correct (1 per shot, alternating label); source spread then SHA ranking',
            'no_probability_ranking': True, 'no_case_replacement_after_pixels': True,
            'original_video_budget': {'train': 0, 'validation': 24, 'test': 0},
            'model_forwards': 0, 'parameter_updates': 0, 'labels_or_thresholds_changed': False,
            'review_scope': 'Exploratory AI observations, not new human truth, event ground truth or test',
            'data_identity': guard.scope, 'selected_best_epochs': {str(d['seed']): d['TANet_best_epoch'] for d in summary['details']}}
    atomic_save(output/'sampling_plan.json', plan, json_format=True)
    atomic_save(output/'distributions.json', groups, json_format=True)
    write_csv(output/'predictions.csv', rows)
    write_csv(output/'common_wrong.csv', [r for r in rows if r['bucket'] == 'common_wrong'])
    for name in ('source', 'shot_type', 'label', 'source_label', 'shot_label', 'source_shot'):
        write_csv(output/f'{name}_summary.csv', [{k: v for k, v in r.items() if k not in ('by_seed', 'correct_seed_counts')} for r in groups[name]])
    atomic_save(output/'run_state.json', {'status': 'prepared_before_pixels', 'completed_cases': 0}, json_format=True)
    print('OUTPUT:', output, flush=True)
    return output


def execute(output):
    output = Path(output).resolve()
    need(output.parent == (FOLDER/'artifacts').resolve() and output.name.startswith('error_audit_v1_'), 'Foreign audit output')
    plan = read(output/'sampling_plan.json')
    need(read(output/'run_state.json')['status'] == 'prepared_before_pixels', 'Already started; no retry or substitution')
    need(all(sha256(p) == h for p, h in plan['protected_sha256'].items()), 'Frozen input changed')
    need(not torch.cuda.is_initialized(), 'No GPU/model forward')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    ledger, completed = AccessLedger(), []
    guard = SplitGuard('validation', ledger)
    print('OUTPUT:', output, flush=True)
    atomic_save(output/'run_state.json', {'status': 'running', 'completed_cases': 0}, json_format=True)
    try:
        for case in plan['cases']:
            sid = case['sample_id']
            frames, fps_rational, identity = decode_native(case, guard)
            n, fps = len(frames), float(Fraction(fps_rational))
            indices = list(uniform_indices(n))
            rgb = np.stack([cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), (256, 192), interpolation=cv2.INTER_AREA) for i in indices])
            tensor = normalize_rgb(rgb, InputConfig())
            need(tensor.shape == (3, 50, 192, 256) and torch.isfinite(tensor).all().item(), 'Input reconstruction failed')
            (output/'native'/sid).mkdir()
            (output/'inputs50'/sid).mkdir()
            for i, frame in enumerate(frames):
                need(cv2.imwrite(str(output/'native'/sid/f'f{i:06d}.png'), frame), 'Native PNG write failed')
            actual = [None]*n
            for i, frame in zip(indices, rgb):
                actual[i] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                need(cv2.imwrite(str(output/'inputs50'/sid/f'f{i:06d}.png'), actual[i]), 'Input PNG write failed')
            overview = sorted({i*(n-1)//11 for i in range(12)})
            contact(frames, overview, output/'contacts'/f'{sid}_overview.png', title=sid)
            contact(frames, list(range(n)), output/'contacts'/f'{sid}_native_all.png', title=sid, size=(160, 120), columns=12)
            contact(actual, indices, output/'contacts'/f'{sid}_tanet50.png', title='Input', size=(256, 192), columns=8)
            encode(frames, output/'previews'/f'{sid}.mp4', fps_rational)
            timing = {'frame_count': n, 'fps_rational': fps_rational,
                      'selected_indices': indices, 'selected_times_seconds': [i/fps for i in indices],
                      'sampling_gaps': temporal_gaps(indices, fps)}
            atomic_save(output/'timelines'/f'{sid}.json', timing, json_format=True)
            completed.append({**case, 'frame_count': n, 'fps': fps, 'fps_rational': fps_rational,
                'source_identity': identity, 'overview_indices': overview, 'tanet_indices': indices,
                'sampling_gaps': timing['sampling_gaps'],
                'normalized_input_sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                'input_reconstruction': 'Exact pinned uniform_indices/resize/normalize on same native decoded frames; no model forward',
                'event_annotations': None, 'AI_review': 'pending', 'canonical_label_changed': False})
            atomic_save(output/'run_state.json', {'status': 'running', 'completed_cases': len(completed)}, json_format=True)
            print(f'{sid} {case["bucket"]} {case["original_class"]} {case["source_group"]} frames={n}', flush=True)
        stats = ledger.statistics()
        need(stats['validation']['opens'] == stats['validation']['decodes'] == 24 and
             stats['train']['opens'] == stats['test']['opens'] == 0, 'Unexpected video access')
        need(all(sha256(p) == h for p, h in plan['protected_sha256'].items()), 'Protected source/weights/data changed')
        report = {'status': 'numeric_and_case_materials_complete_AI_review_pending',
            'completed_at_taiwan': stamp(), 'plan_sha256': sha256(output/'sampling_plan.json'),
            'validation_videos': 1547, 'seeds': list(SEEDS), 'distributions': read(output/'distributions.json'),
            'cases': completed, 'source_access': stats, 'original_access_events': ledger.events,
            'model_forward_calls': 0, 'optimizer_updates': 0, 'test_used': False,
            'source_weights_labels_thresholds_unchanged': True, 'protected_files_verified': len(plan['protected_sha256']),
            'limitations': ['Existing validation already used for policy exploration and checkpoint selection',
                'Fixed24 deliberately stratified; its visual observations are not population error rates',
                'AI review does not establish human truth, automatically relabel or exclude videos',
                'No event annotations yet; exact frame spacing alone cannot prove an event was missed',
                'Per-source/per-shot rates are descriptive; some strata are small',
                'Reconstructed RGB inputs are not a new model-forward reproducibility test']}
        atomic_save(output/'report.json', report, json_format=True)
        template = {'status': 'unreviewed_human_template', 'canonical_labels_changed': False,
                    'cases': [{'sample_id': c['sample_id'], 'video_id': c['video_id'],
                        'release_frame': None, 'result_frame': None, 'outcome': None,
                        'reviewer_confirmed': False, 'notes': ''} for c in completed]}
        atomic_save(output/'human_review_template.json', template, json_format=True)
        plot_groups(output, report['distributions'])
        render(output, report)
        atomic_save(output/'run_state.json', {'status': report['status'], 'completed_cases': 24}, json_format=True)
        return output
    except BaseException as exc:
        atomic_save(output/'run_state.json', {'status': 'failed', 'completed_cases': len(completed),
                    'error': f'{type(exc).__name__}: {exc}'}, json_format=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', type=Path)
    args = parser.parse_args()
    prepare() if args.prepare else execute(args.run)
