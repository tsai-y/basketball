"""CPU-only diagnosis of saved best/last logits and BatchNorm state, without forward.

No model construction, raw video/feature access, BN recalibration, fitting,
threshold change or optimizer. Original checkpoints remain read-only.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime
import html
import importlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
import torch

from .config import FOLDER, need, sha256
from .data import VideoScope
from .training import atomic_save, json_hash
from .repeat_seeds_v1 import read, verify_sources
from .analyze_v1 import aggregate

ROOT = FOLDER.parents[1]
CAMPAIGN = FOLDER/'artifacts/multiseed_c0qf_w7l'
EPS = 1e-5  # Author constructs BatchNorm1d with PyTorch's default eps.


def finite_summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return None
    need(np.isfinite(a).all(), 'Nonfinite diagnostic values')
    return {'count': int(a.size), 'min': float(a.min()), 'p01': float(np.quantile(a, .01)),
            'median': float(np.median(a)), 'p95': float(np.quantile(a, .95)),
            'p99': float(np.quantile(a, .99)), 'max': float(a.max()), 'mean': float(a.mean())}


def validate_ids(ids, labels, allowed):
    need(len(ids) == len(set(ids)) and set(ids) == set(allowed), 'Incomplete/duplicate canonical validation IDs')
    need(len(labels) == len(ids), 'Label length differs')
    for vid, y in zip(ids, labels):
        row = allowed[vid]
        need(row['split'] == 'validation' and str(row['excluded']) == '0', 'Validation only')
        need(type(y) is int and y in (0, 1) and y == int(row['binary_label']), 'Canonical label mismatch')


def rows_from_logits(saved, allowed):
    v = saved['validation']
    logits, labels = v['logits'], v['labels']
    n = len(allowed)
    need(logits.shape == (n, 2) and labels.shape == (n,) and labels.dtype == torch.int64,
         'Invalid saved validation shapes')
    need(torch.isfinite(logits).all().item(), 'Nonfinite saved logits')
    ids, ys = v['video_ids'], labels.tolist()
    validate_ids(ids, ys, allowed)
    # Validate the original float32 CE, then use float64 log_softmax for per-video
    # NLL: never infer true-class log probability from a rounded/saturated softmax.
    need(abs(float(torch.nn.functional.cross_entropy(logits.float(), labels))-v['metrics']['CE']) < 1e-7,
         'Original saved CE mismatch')
    logp = logits.double().log_softmax(1)
    p = logp.exp()
    pred = logits.argmax(1)
    result = []
    for i, (vid, y) in enumerate(zip(ids, ys)):
        prediction = int(pred[i])
        result.append({'video_id': vid, 'true_label': y, 'prediction': prediction,
                       'p_make': float(p[i, 1]), 'confidence': float(p[i, prediction]),
                       'nll': float(-logp[i, y]), 'entropy': float(-(p[i]*logp[i]).sum()),
                       'nll_method': 'float64_log_softmax_of_saved_float32_logits',
                       'source_group': allowed[vid]['source_group'],
                       'shot_type': allowed[vid]['original_class'][:-1]})
    result.sort(key=lambda r: r['video_id'])
    actual = aggregate(result, 'prediction')
    need(actual['confusion_matrix'] == v['metrics']['confusion_matrix_Miss_Make'] and
         abs(actual['macro_f1']-v['metrics']['macro_f1']) < 1e-10, 'Saved metrics mismatch')
    return result


def mainline_rows(folder, allowed):
    with (folder/'validation_predictions.csv').open() as f:
        saved = list(csv.DictReader(f))
    validate_ids([r['video_id'] for r in saved], [int(r['true_label']) for r in saved], allowed)
    result = []
    for r in saved:
        vid, y, pred, q = r['video_id'], int(r['true_label']), int(r['predicted_label']), float(r['p_make'])
        need(math.isfinite(q) and 0 <= q <= 1 and pred in (0, 1) and pred == int(q >= .5), 'Invalid mainline prediction')
        # Original mainline logits were not saved; expose probability clipping
        # and retain the original saved BCE as the authoritative run loss.
        true_p = q if y else 1-q
        clipped = min(max(true_p, 1e-15), 1-1e-15)
        result.append({'video_id': vid, 'true_label': y, 'prediction': pred, 'p_make': q,
            'confidence': q if pred else 1-q, 'nll': -math.log(clipped),
            'entropy': -sum(p*math.log(p) for p in (q, 1-q) if p > 0),
            'nll_method': 'approximate_from_saved_probability_clipped_1e-15',
            'probability_endpoint': q in (0., 1.), 'source_group': allowed[vid]['source_group'],
            'shot_type': allowed[vid]['original_class'][:-1]})
    result.sort(key=lambda r: r['video_id'])
    need(aggregate(result, 'prediction')['confusion_matrix'] == read(folder/'summary.json')['validation']['confusion_matrix'],
         'Mainline saved metrics mismatch')
    return result


def confidence_metrics(rows):
    need(rows, 'Empty diagnostic subset')
    n = len(rows)
    correct = [r for r in rows if r['prediction'] == r['true_label']]
    wrong = [r for r in rows if r['prediction'] != r['true_label']]
    bins = []
    for i in range(10):
        # Ten fixed [0,.1),...,[.9,1] bins, no tuning to observed data.
        subset = [r for r in rows if min(int(r['confidence']*10), 9) == i]
        bins.append({'lower': i/10, 'upper': (i+1)/10, 'count': len(subset),
            'accuracy': sum(r['prediction'] == r['true_label'] for r in subset)/len(subset) if subset else None,
            'mean_confidence': sum(r['confidence'] for r in subset)/len(subset) if subset else None})
    ece = sum(b['count']/n*abs(b['accuracy']-b['mean_confidence']) for b in bins if b['count'])
    total_nll = sum(r['nll'] for r in rows)
    top_n = max(1, math.ceil(.01*n))
    top = sorted(rows, key=lambda r: (-r['nll'], r['video_id']))[:top_n]
    high = {}
    for threshold in (.90, .95):
        selected = [r for r in rows if r['confidence'] >= threshold]
        errors = sum(r['prediction'] != r['true_label'] for r in selected)
        high[str(threshold)] = {'count': len(selected), 'errors': errors,
            'error_rate_among_selected': errors/len(selected) if selected else None,
            'share_of_all_errors': errors/len(wrong) if wrong else None}
    return {'metrics': aggregate(rows, 'prediction'), 'mean_nll': total_nll/n,
        'binary_brier': sum((r['p_make']-r['true_label'])**2 for r in rows)/n,
        'mean_entropy': sum(r['entropy'] for r in rows)/n, 'ece10_toplabel': ece,
        'reliability_bins': bins, 'wrong_confidence': finite_summary([r['confidence'] for r in wrong]),
        'correct_confidence': finite_summary([r['confidence'] for r in correct]),
        'wrong_nll': finite_summary([r['nll'] for r in wrong]),
        'correct_nll': finite_summary([r['nll'] for r in correct]),
        'wrong_share_of_nll': sum(r['nll'] for r in wrong)/total_nll if total_nll else 0.,
        'high_confidence': high, 'top_one_percent_count': top_n,
        'top_one_percent_share_of_nll': sum(r['nll'] for r in top)/total_nll if total_nll else 0.,
        'probability_endpoint_count': sum(r.get('probability_endpoint', False) for r in rows)}


def grouped_metrics(rows):
    return {'overall': confidence_metrics(rows),
            'by_label': {str(y): confidence_metrics([r for r in rows if r['true_label'] == y]) for y in (0, 1)},
            'by_source': {g: confidence_metrics([r for r in rows if r['source_group'] == g])
                          for g in sorted({r['source_group'] for r in rows})},
            'by_shot_type': {g: confidence_metrics([r for r in rows if r['shot_type'] == g])
                             for g in sorted({r['shot_type'] for r in rows})}}


def bn_statistics(state, expected_batches):
    """Inspect all32 author TAM BN1d layers; statistics are not a causal test."""
    prefixes = sorted(k.removesuffix('.running_var') for k in state if '.tam.' in k and k.endswith('.running_var'))
    need(len(prefixes) == 32, 'Expected16 G and16 L BatchNorm1d layers')
    result, raw = [], {}
    for name in prefixes:
        mean, var, gamma, beta = [state[name+'.'+k].detach().double().cpu().numpy()
                                  for k in ('running_mean', 'running_var', 'weight', 'bias')]
        need(mean.shape == var.shape == gamma.shape == beta.shape and mean.ndim == 1, 'BN shape mismatch')
        need(all(np.isfinite(v).all() for v in (mean, var, gamma, beta)) and (var >= 0).all(), 'Invalid BN buffer/affine')
        batches = int(state[name+'.num_batches_tracked'])
        need(batches == expected_batches, 'TAM BN counter differs from complete training forwards')
        gain = np.abs(gamma)/np.sqrt(var+EPS)
        row = {'layer': name, 'branch': 'G' if '.G.' in name else 'L', 'features': len(var),
               'num_batches_tracked': batches, 'expected_training_forwards': expected_batches,
               'running_mean_abs': finite_summary(np.abs(mean)), 'running_var': finite_summary(var),
               'gamma_abs': finite_summary(np.abs(gamma)), 'beta_abs': finite_summary(np.abs(beta)),
               'eval_gain_abs': finite_summary(gain),
               'variance_le_1e-8_count': int((var <= 1e-8).sum()),
               'variance_le_1e-5_count': int((var <= 1e-5).sum())}
        result.append(row)
        raw[name] = (mean.copy(), var.copy(), gain.copy())
    need(Counter(r['branch'] for r in result) == {'G': 16, 'L': 16}, 'Wrong TAM BN branches')
    return result, raw


def frozen_bn_check(state, imagenet):
    keys = [k for k in state if '.tam.' not in k and
            k.endswith(('.running_mean', '.running_var', '.num_batches_tracked'))]
    need(len(keys) == 159, 'Expected53 pretrained BN2d modules, three buffers each')
    different = []
    for k in keys:
        source = k.removeprefix('backbone.').replace('.net.', '.')
        # The pinned legacy ImageNet file has267 tensors and no BN counters.
        # torchvision's standard loader creates those counters as zero, just
        # as it did during the original initialized_model construction.
        if source.endswith('.num_batches_tracked') and source not in imagenet:
            expected = torch.zeros_like(state[k])
        else:
            need(source in imagenet, 'Missing ImageNet BN buffer: '+source)
            expected = imagenet[source]
        if not torch.equal(state[k], expected):
            different.append(k)
    return {'checked_buffer_tensors': len(keys), 'different': different, 'all_bitwise_equal': not different,
            'legacy_missing_ImageNet_BN_counters_initialized_to_zero':
                sum(k.endswith('.num_batches_tracked') and k.removeprefix('backbone.').replace('.net.', '.') not in imagenet for k in keys)}


def write_csv(path, rows):
    need(rows, 'No CSV rows')
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def prepare_inputs():
    plan, campaign_state = read(CAMPAIGN/'plan.json'), read(CAMPAIGN/'state.json')
    summary = read(CAMPAIGN/'summary.json')
    need(campaign_state['status'] == summary['status'] == 'completed' and summary['seeds'] == [20261024, 20261025, 20261026],
         'Three seed campaign incomplete')
    verify_sources(plan)
    protected = dict(plan['protected_sha256'])
    for name in ('plan.json', 'state.json', 'summary.json'):
        protected[str(CAMPAIGN/name)] = sha256(CAMPAIGN/name)
    packages = {item['seed']: item['module'] for item in plan['queue']}
    packages[20261025] = 'development.tanet_v1'
    for d in summary['details']:
        folder = Path(d['TANet_run'])
        current, _, _ = importlib.import_module(packages[d['seed']]+'.training').contract()
        need(current == read(folder/'contract.json'), 'Run source/recipe/data contract changed')
        need(read(folder/'run_state.json')['status'] == 'completed', 'TANet run incomplete')
        for name in ('best.pt', 'last.pt', 'contract.json', 'history.json', 'run_state.json', 'progress.log'):
            protected[str(folder/name)] = sha256(folder/name)
    protected[str(Path(__file__).resolve())] = sha256(__file__)
    cfg = read(Path(summary['details'][0]['TANet_run'])/'contract.json')['initialization']
    protected[cfg['imagenet_checkpoint']] = cfg['imagenet_sha256']
    need(all(sha256(p) == h for p, h in protected.items()), 'Protected source/checkpoint changed')
    return summary, protected, cfg


def execute():
    torch.set_num_threads(2)
    need(not torch.cuda.is_initialized(), 'This diagnosis must not initialize CUDA')
    summary, protected, cfg = prepare_inputs()
    allowed = VideoScope('validation').allowed
    need(len(allowed) == 1547, 'Expected full1547 validation metadata')
    out = Path(tempfile.mkdtemp(prefix='stability_v1_', dir=FOLDER/'artifacts'))
    (out/'plots').mkdir(); (out/'inputs').mkdir()
    shutil.copy2(__file__, out/'inputs/diagnose_stability_v1.py')
    policy = {'confidence_thresholds': [.9, .95], 'reliability_bins': '10 equal width bins over[0,1], final bin includes1',
              'top_nll_fraction': .01, 'top_case_list_count': 20, 'BN_eps': EPS,
              'small_variance_descriptive_thresholds': [1e-8, 1e-5],
              'scope': 'saved best/last validation predictions and BN buffers only',
              'no_threshold_or_temperature_fitting': True, 'original_video_opens': 0,
              'model_forwards': 0, 'optimizer_steps': 0, 'test_used': False,
              'protected_sha256': protected}
    atomic_save(out/'plan.json', policy, json_format=True)
    atomic_save(out/'run_state.json', {'status': 'running'}, json_format=True)
    print('OUTPUT:', out, flush=True)
    try:
        imagenet = torch.load(cfg['imagenet_checkpoint'], map_location='cpu', weights_only=True)
        details, all_rows, bn_csv, drift_csv, best_rows, histories = [], [], [], [], {}, {}
        for d in summary['details']:
            seed, run = d['seed'], Path(d['TANet_run'])
            hist, contract = read(run/'history.json'), read(run/'contract.json')
            histories[str(seed)] = hist
            detail = {'seed': seed, 'run': str(run), 'snapshots': {}}
            previous = None
            for tag in ('best', 'last'):
                saved = torch.load(run/f'{tag}.pt', map_location='cpu', weights_only=True)
                expected = min(hist, key=lambda e: e['validation']['CE']) if tag == 'best' else hist[-1]
                need(saved['kind'] == 'formal' and saved['epoch'] == expected['epoch'] and
                     saved['contract_sha256'] == contract['contract_sha256'] and saved['history'] == hist[:saved['epoch']],
                     'Checkpoint purpose/selection/history mismatch')
                rows = rows_from_logits(saved, allowed)
                bn, raw = bn_statistics(saved['model'], saved['epoch']*7193)
                frozen = frozen_bn_check(saved['model'], imagenet)
                need(frozen['all_bitwise_equal'], 'Pretrained BN2d running buffers changed despite frozen policy')
                metrics = grouped_metrics(rows)
                detail['snapshots'][tag] = {'epoch': saved['epoch'], 'global_step': saved['global_step'],
                    'saved_CE': expected['validation']['CE'], **metrics, 'TAM_BN': bn, 'pretrained_BN2d': frozen}
                all_rows.extend({'seed': seed, 'snapshot': tag, **r} for r in rows)
                if tag == 'best': best_rows[seed] = rows
                for b in bn:
                    bn_csv.append({'seed': seed, 'snapshot': tag, 'epoch': saved['epoch'], 'layer': b['layer'],
                        'branch': b['branch'], 'features': b['features'], 'batches': b['num_batches_tracked'],
                        'variance_min': b['running_var']['min'], 'variance_median': b['running_var']['median'],
                        'variance_max': b['running_var']['max'], 'gain_p95': b['eval_gain_abs']['p95'],
                        'gain_max': b['eval_gain_abs']['max'], 'variance_le_1e-8_count': b['variance_le_1e-8_count'],
                        'variance_le_1e-5_count': b['variance_le_1e-5_count']})
                if previous is not None:
                    drift = []
                    for name, (mu, var, gain) in raw.items():
                        old_mu, old_var, old_gain = previous[name]
                        stat = {'layer': name,
                            'abs_mean_shift_in_best_running_sd': finite_summary(np.abs(mu-old_mu)/np.sqrt(old_var+EPS)),
                            'log10_variance_ratio_last_vs_best': finite_summary(np.log10((var+EPS)/(old_var+EPS)))}
                        drift.append(stat)
                        drift_csv.append({'seed': seed, 'layer': name,
                            'mean_shift_standardized_median': stat['abs_mean_shift_in_best_running_sd']['median'],
                            'mean_shift_standardized_p95': stat['abs_mean_shift_in_best_running_sd']['p95'],
                            'log10_variance_ratio_median': stat['log10_variance_ratio_last_vs_best']['median']})
                    detail['BN_best_to_last_drift'] = drift
                previous = raw
                del saved
                print(seed, tag, 'epoch', expected['epoch'], 'NLL', metrics['overall']['mean_nll'],
                      'wrong>=.9', metrics['overall']['high_confidence']['0.9']['errors'], flush=True)
            main = Path(d['mainline_run'])
            rows = mainline_rows(main, allowed)
            detail['mainline'] = {'run': str(main), 'saved_BCE': read(main/'summary.json')['validation']['loss'],
                                  **grouped_metrics(rows)}
            all_rows.extend({'seed': seed, 'snapshot': 'mainline', **r} for r in rows)
            details.append(detail)
        # All seeds are evaluated on the same validation videos, not independent datasets.
        cross = []
        for triple in zip(*(best_rows[seed] for seed in summary['seeds'])):
            need(len({r['video_id'] for r in triple}) == len({r['true_label'] for r in triple}) == 1, 'Cross seed ID mismatch')
            cross.append({'video_id': triple[0]['video_id'], 'true_label': triple[0]['true_label'],
                'source_group': triple[0]['source_group'],
                **{f'prediction_{seed}': row['prediction'] for seed, row in zip(summary['seeds'], triple)},
                'correct_seeds': sum(r['prediction'] == r['true_label'] for r in triple)})
        # Union CSV has an explicit endpoint field on every row.
        for r in all_rows: r.setdefault('probability_endpoint', False)
        fieldnames = list(all_rows[0])
        all_rows = [{k: r[k] for k in fieldnames} for r in all_rows]
        write_csv(out/'per_video_predictions.csv', all_rows)
        write_csv(out/'TAM_BN_layers.csv', bn_csv)
        write_csv(out/'BN_best_last_drift.csv', drift_csv)
        write_csv(out/'cross_seed_predictions.csv', cross)
        top = []
        for d in details:
            for tag in ('best', 'last', 'mainline'):
                subset = [r for r in all_rows if r['seed'] == d['seed'] and r['snapshot'] == tag]
                top.extend(sorted(subset, key=lambda r: (-r['nll'], r['video_id']))[:20])
        write_csv(out/'top20_nll_per_snapshot.csv', top)
        need(all(sha256(p) == h for p, h in protected.items()), 'Original artifacts changed during diagnosis')
        need(not torch.cuda.is_initialized(), 'Unexpected CUDA context')
        report = {'status': 'completed', 'completed_at_taiwan': datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
            'unique_validation_videos': len(allowed), 'prediction_rows_including_repeated_snapshots': len(all_rows),
            'policy': policy, 'details': details, 'histories': histories,
            'cross_seed_correct_count_distribution': dict(Counter(r['correct_seeds'] for r in cross)),
            'original_artifacts_sha256_unchanged': True, 'cuda_initialized': False,
            'source_video_opens': {'train': 0, 'validation': 0, 'test': 0},
            'model_forward_calls': 0, 'optimizer_steps': 0, 'weights_or_buffers_modified': False,
            'limitations': [
                'Validation has already been used for checkpoint selection; these are descriptive diagnostics, not test.',
                'Only best and last per-video logits/BN states were retained. Intermediate epochs have aggregate history only.',
                'ECE uses fixed10 bins and finite selected validation data; it is not a calibrated model or a universal quality ranking.',
                'Mainline NLL is approximate from stored probabilities, with endpoints clipped; saved BCE is authoritative.',
                'Small variance or large eval gain alone is not proof of a broken BatchNorm layer.',
                'Best-to-last BN drift also includes changed learned weights; no causal or train/eval activation comparison is performed.',
                'Randomly initialized TAM channels are not pointwise aligned across seeds; only distribution summaries are compared.',
                'Microbatch1 still has channel/time samples inside TAM BN. Accumulation16 does not pool BN statistics across16 videos.',
                'The snapshot analysis cannot determine whether BN, optimization, input sampling, or source shift caused instability.']}
        atomic_save(out/'report.json', report, json_format=True)
        plots(out, report, bn_csv)
        render(out, report)
        atomic_save(out/'run_state.json', {'status': 'completed', 'unique_validation_videos': len(allowed),
                    'report_sha256': sha256(out/'report.json'), 'model_forwards': 0, 'test_used': False}, json_format=True)
        return out
    except BaseException as exc:
        atomic_save(out/'run_state.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}, json_format=True)
        raise


def plots(out, report, bn_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#1675b8', '#d56a00', '#287d48']
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for j, d in enumerate(report['details']):
        hist = report['histories'][str(d['seed'])]
        x = [e['epoch'] for e in hist]
        axes[0,j].plot(x, [e['train_CE'] for e in hist], label='Train CE')
        axes[0,j].plot(x, [e['validation']['CE'] for e in hist], label='Validation CE')
        axes[0,j].axvline(d['snapshots']['best']['epoch'], linestyle=':', color='#555')
        axes[0,j].set(title=str(d['seed']), xlabel='Epoch', ylabel='CE'); axes[0,j].legend()
        for tag, style in [('best','-'), ('last','--')]:
            bins = d['snapshots'][tag]['overall']['reliability_bins']
            a = [b for b in bins if b['count']]
            axes[1,j].plot([b['mean_confidence'] for b in a], [b['accuracy'] for b in a], style+'o', label=f'TANet {tag}')
        b = [b for b in d['mainline']['overall']['reliability_bins'] if b['count']]
        axes[1,j].plot([v['mean_confidence'] for v in b], [v['accuracy'] for v in b], ':o', color='#555', label='Mainline')
        axes[1,j].plot([.5,1],[.5,1], color='#aaa', linewidth=1)
        axes[1,j].set(xlim=(.48,1.02), ylim=(0,1.02), xlabel='Mean predicted confidence', ylabel='Accuracy in fixed bin')
        axes[1,j].legend(fontsize=8)
    fig.savefig(out/'plots/curves_reliability.png', dpi=140); plt.close(fig)
    sources = sorted(report['details'][0]['snapshots']['best']['by_source'])
    labels, values = [], []
    for d in report['details']:
        for tag in ('best','last','mainline'):
            m = d['mainline'] if tag=='mainline' else d['snapshots'][tag]
            labels.append(f'{d["seed"]} {tag}')
            values.append([100*m['by_source'][g]['metrics']['accuracy'] for g in sources])
    fig, ax = plt.subplots(figsize=(11,6), constrained_layout=True)
    ax.imshow(values, vmin=40, vmax=100, cmap='YlGnBu', aspect='auto')
    for i,row in enumerate(values):
        for j,v in enumerate(row):ax.text(j,i,f'{v:.1f}',ha='center',va='center',color='white' if v>80 else 'black')
    supports = [report['details'][0]['snapshots']['best']['by_source'][g]['metrics']['videos'] for g in sources]
    ax.set(xticks=range(len(sources)),xticklabels=[f'{g}\nn={n}' for g,n in zip(sources,supports)],
           yticks=range(len(labels)),yticklabels=labels,title='Validation accuracy (%) by source; same videos each row')
    fig.savefig(out/'plots/source_accuracy.png',dpi=140);plt.close(fig)
    fig, axes = plt.subplots(2,1,figsize=(14,8),constrained_layout=True)
    layers = sorted({b['layer'] for b in bn_rows})
    for j,d in enumerate(report['details']):
        for tag,style in [('best','-'),('last','--')]:
            subset = {b['layer']: b for b in bn_rows if b['seed']==d['seed'] and b['snapshot']==tag}
            for ax,key in zip(axes,('variance_median','gain_p95')):
                ax.plot(range(32),[max(subset[k][key],1e-12) for k in layers],style,color=colors[j],label=f'{d["seed"]} {tag}')
    for ax in axes:
        ax.set_yscale('log');ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2)
        ax.set_xticks(range(32),[k.removeprefix('backbone.').removesuffix('.1') for k in layers],rotation=90,fontsize=7)
    axes[0].set(title='TAM BN running variance (median across features)',ylabel='Running variance')
    axes[1].set(title='TAM BN evaluation gain |gamma| / sqrt(var + eps), p95 across features',ylabel='Absolute eval gain')
    fig.savefig(out/'plots/BN_distributions.png',dpi=140);plt.close(fig)


def render(out, report):
    esc = html.escape
    sections = []
    for d in report['details']:
        rows = []
        for tag in ('best','last','mainline'):
            s = d['mainline'] if tag=='mainline' else d['snapshots'][tag]
            o = s['overall'];m = o['metrics'];h = o['high_confidence']['0.9']
            loss = s['saved_BCE'] if tag=='mainline' else s['saved_CE']
            rows.append(f'<tr><td>{tag}</td><td>{s.get("epoch","—")}</td><td>{loss:.4f}</td><td>{100*m["accuracy"]:.2f}%</td><td>{o["ece10_toplabel"]:.4f}</td><td>{o["binary_brier"]:.4f}</td><td>{h["errors"]}/{h["count"]}</td><td>{m["predicted_Make_count"]}</td></tr>')
        sections.append(f'<h2>Seed {d["seed"]}</h2><table><tr><th>Snapshot</th><th>Epoch</th><th>保存CE/BCE</th><th>Accuracy</th><th>ECE10</th><th>Binary Brier</th><th>信心≥90%中的錯誤／總數</th><th>預測Make數</th></tr>'+''.join(rows)+'</table>')
    body = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TANet訓練穩定性診斷V1</title><style>body{font:18px system-ui;line-height:1.8;max-width:1250px;margin:32px auto;padding:0 20px;color:#192333}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:9px}img{max-width:100%}.notice{background:#fff1cc;padding:18px}li{margin:8px 0}</style><h1>TANet訓練穩定性診斷V1</h1><p class="notice">三seed各自的best／last，共6份checkpoint；每份均為同一批1547支validation。僅讀已保存logits／BN buffers；未讀原片、未做模型forward、未重訓／校正BN／改門檻或權重，未執行test。</p><p><a href="report.json">完整數據與限制</a> · <a href="per_video_predictions.csv">逐片預測與NLL</a> · <a href="TAM_BN_layers.csv">192列TAM BN摘要</a> · <a href="BN_best_last_drift.csv">同seed best→last BN變化</a> · <a href="cross_seed_predictions.csv">三seed判對一致性</a> · <a href="top20_nll_per_snapshot.csv">每份預測的最高20筆NLL</a></p><p>信心固定門檻90%／95%、ECE固定10等寬箱，未調參。主線逐片NLL從保存機率近似計算（端點裁為1e-15）；保存的BCE才是原run loss。TANet逐片NLL以保存logits做float64 log_softmax，避免機率飽和造成無限值。</p>'''
    body += ''.join(sections)
    body += '<h2>曲線與信心可靠度</h2><img src="plots/curves_reliability.png"><h2>來源差異</h2><img src="plots/source_accuracy.png"><h2>BatchNorm快照</h2><p>所有6份checkpoint的32個TAM BN1d counters，皆等於所屬epoch×7193。53個預訓練BN2d的159個running buffers，皆與ImageNet初始化逐值一致。TAM的G分支按B×C、L分支按B×T提供統計樣本；micro-batch1不代表每層只有一個scalar，但梯度累積16不會把16支影片的BN統計合併。</p><img src="plots/BN_distributions.png"><p>上述圖只比較每層分布摘要，不把不同seed的隨機TAM通道當成一一對應特徵。方差小、gain大或best→last變動本身不證明BN故障；沒有實測train/eval activation落差。</p><h2>可判斷的範圍</h2><ul>'
    body += ''.join(f'<li>{esc(s)}</li>' for s in report['limitations'])+'</ul><p><a href="https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm1d.html">PyTorch2.11 BatchNorm1d官方定義</a></p></html>'
    (out/'index.html').write_text(body)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    print(execute())
