"""Two sequential TANet repeats using byte-identical engines in isolated packages.

Only the seed in each copied config.json/train_config.json changes. Each seed
gets a fresh GPU readiness probe and ImageNet initialization. Never opens test,
changes the original run, retries failed jobs, or selects a winning seed.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

from .config import FOLDER, need, sha256
from .training import atomic_save, contract, json_hash

ROOT = FOLDER.parents[1]
SEEDS = (20261024, 20261025, 20261026)
NEW_SEEDS = (20261024, 20261026)
FILES = ('__init__.py', 'config.py', 'config.json', 'data.py', 'model.py',
         'prepare.py', 'training.py', 'train.py', 'readiness.py', 'train_config.json',
         'upstream/__init__.py', 'upstream/temporal_module.py', 'upstream/PROVENANCE.json',
         'upstream/LICENSE', 'upstream/reference_models.py.txt')
MAIN_NAMES = {20261024: 'temporal_seed20261024_y7wv4h0p',
              20261025: 'temporal_seed20261025_cwac4gwz',
              20261026: 'temporal_seed20261026_59bm89qx'}
ORIGINAL = FOLDER/'artifacts/formal_train_ji1xnlg_'


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now().astimezone().isoformat()


def seed_config(cfg, seed):
    need(type(seed) is int and seed in NEW_SEEDS, 'Only the two declared repeat seeds are authorized')
    result = copy.deepcopy(cfg)
    result['seed'] = seed
    return result


def check_clone(source, destination, seed):
    for name in FILES:
        target = destination/name
        need(target.is_file() and not target.is_symlink(), 'Missing/symlinked repeat source: '+name)
        if name in ('config.json', 'train_config.json'):
            need(read(target) == seed_config(read(source/name), seed), 'Only seed may change: '+name)
        else:
            need(sha256(source/name) == sha256(target), 'Repeat engine changed: '+name)
    return {name: sha256(destination/name) for name in FILES}


def clone_seed(source, destination, seed):
    """Exclusive creation; an existing package/run is never overwritten."""
    need(not destination.exists(), 'Repeat package already exists; inspect before a new launch')
    destination.mkdir()
    for name in FILES:
        target = destination/name
        target.parent.mkdir(exist_ok=True)
        if name in ('config.json', 'train_config.json'):
            target.write_text(json.dumps(seed_config(read(source/name), seed), indent=2)+'\n')
        else:
            shutil.copy2(source/name, target)
    (destination/'artifacts').mkdir()
    return check_clone(source, destination, seed)


def validate_mainlines():
    result, reference = {}, None
    for seed in SEEDS:
        folder = ROOT/'development/temporal_baseline_v1/artifacts'/MAIN_NAMES[seed]
        cfg, summary = read(folder/'config.json'), read(folder/'summary.json')
        need(cfg['seed'] == summary['seed'] == seed and cfg['train_count'] == 7193 and
             cfg['validation_count'] == 1547 and cfg['selection'] == 'minimum_validation_loss',
             'Mainline seed/scope/selection mismatch')
        comparable = {k: v for k, v in cfg.items() if k != 'seed'}
        reference = comparable if reference is None else reference
        need(comparable == reference, 'Mainline recipes differ beyond seed')
        need(summary['cache_run_id'] == cfg['cache_run_id'], 'Mainline cache mismatch')
        result[str(seed)] = str(folder)
    return result


def prepare():
    base, _, _ = contract()
    need(base == read(ORIGINAL/'contract.json'), 'Original formal contract changed')
    need(read(ORIGINAL/'run_state.json')['status'] == 'completed', 'Original run not complete')
    old_report = read(FOLDER/'artifacts/analysis_v1_cfh3n8r2/report.json')
    protected = dict(old_report['protected_sha256'])
    mainlines = validate_mainlines()
    for folder in map(Path, mainlines.values()):
        for name in ('best.pt', 'config.json', 'summary.json', 'history.csv', 'validation_predictions.csv'):
            protected[str(folder/name)] = sha256(folder/name)
    protected[str(Path(__file__).resolve())] = sha256(__file__)
    need(all(sha256(p) == h for p, h in protected.items()), 'Protected original changed')
    # Validate destinations before creating either package.
    destinations = {seed: ROOT/'development'/f'tanet_seed{seed}_v1' for seed in NEW_SEEDS}
    need(all(not p.exists() for p in destinations.values()), 'Repeat packages already exist')
    output = Path(tempfile.mkdtemp(prefix='multiseed_', dir=FOLDER/'artifacts'))
    entries = []
    for seed, destination in destinations.items():
        hashes = clone_seed(FOLDER, destination, seed)
        module = f'development.{destination.name}'
        plan, _, _ = importlib.import_module(module+'.training').contract()
        expected = copy.deepcopy(base)
        expected['recipe']['seed'] = seed
        expected['initialization']['seed'] = seed
        expected['source_sha256']['config.json'] = hashes['config.json']
        expected['source_sha256']['train_config.json'] = hashes['train_config.json']
        expected.pop('contract_sha256')
        need(plan == {**expected, 'contract_sha256': json_hash(expected)}, 'Repeat contract changed beyond seed')
        atomic_save(destination/'REPEAT_PROVENANCE.json', {
            'original_package': str(FOLDER), 'seed': seed, 'only_config_changes': ['seed'],
            'original_formal_contract_sha256': base['contract_sha256'],
            'repeat_contract_sha256': plan['contract_sha256'], 'files_sha256': hashes,
            'author_code_modified': False, 'test_access': False}, json_format=True)
        entries.append({'seed': seed, 'package': str(destination), 'module': module,
                        'contract_sha256': plan['contract_sha256'], 'source_sha256': hashes})
    plan = {'schema': 'tanet_fixed_recipe_three_seed_campaign_v1', 'created_at': stamp(),
            'seeds': list(SEEDS), 'new_seeds': list(NEW_SEEDS), 'queue': entries,
            'original_seed20261025': str(ORIGINAL), 'mainline_runs': mainlines,
            'recipe': base['recipe'], 'protected_sha256': protected,
            'formal_training_explicitly_authorized': True, 'serial_gpu_jobs': True,
            'failure_policy': 'stop_without_retry_or_automatic_resume',
            'selection': 'minimum_validation_CE_within_each_seed; report_all_three_seeds',
            'test_access': False, 'hyperparameter_search': False, 'original_runs_modified': False}
    atomic_save(output/'plan.json', plan, json_format=True)
    atomic_save(output/'state.json', {'status': 'prepared', 'jobs': [], 'test_used': False}, json_format=True)
    print(output, flush=True)
    return output


def process_identity(pid):
    proc = Path('/proc')/str(pid)
    parts = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'process_start_ticks': int(parts[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'cmdline': (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()}


def output_path(log, prefix):
    lines = Path(log).read_text().splitlines()
    found = [line[len(prefix):].strip() for line in lines if line.startswith(prefix)]
    need(len(found) == 1, 'Missing/ambiguous child output path')
    return Path(found[0]).resolve()


def verify_sources(plan):
    need(all(sha256(p) == h for p, h in plan['protected_sha256'].items()), 'Protected original changed')
    for item in plan['queue']:
        hashes = check_clone(FOLDER, Path(item['package']), item['seed'])
        need(hashes == item['source_sha256'], 'Pinned repeat source changed')


def command_for(item, stage, report=None):
    common = [sys.executable, '-B', '-m', item['module']]
    if stage == 'readiness':
        return common[:-1]+[common[-1]+'.readiness', '--run']
    need(stage == 'training' and report is not None, 'Training requires a fresh readiness report')
    return common[:-1]+[common[-1]+'.train', '--start', '--readiness-report', str(report)]


def run_child(output, state, item, stage, command):
    log = output/f'seed{item["seed"]}_{stage}.log'
    env = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONDONTWRITEBYTECODE': '1'}
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        job = {'seed': item['seed'], 'stage': stage, 'command': command,
               'stdout_log': str(log), 'started_at': stamp(), **process_identity(child.pid)}
        state['jobs'].append(job)
        state.update(status='running', active_seed=item['seed'], active_stage=stage)
        atomic_save(output/'state.json', state, json_format=True)
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
    job.update(returncode=code, ended_at=stamp())
    atomic_save(output/'state.json', state, json_format=True)
    need(code == 0, f'Seed {item["seed"]} {stage} failed with code {code}; see {log}')
    path = output_path(log, 'OUTPUT: ')
    need(path.is_relative_to((Path(item['package'])/'artifacts').resolve()), 'Foreign child output')
    job['output'] = str(path)
    atomic_save(output/'state.json', state, json_format=True)
    return path


def distribution(values):
    need(len(values) == 3, 'Report exactly three seeds, not a selected subset')
    return {'values': values, 'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values), 'min': min(values), 'max': max(values)}


def summarize(output, plan, runs):
    """Read saved validation outputs only; no model forward/test/threshold tuning."""
    import csv
    import torch
    from .analyze_v1 import align_predictions, aggregate
    from .data import VideoScope
    canonical = VideoScope('validation').allowed
    details = []
    for seed in SEEDS:
        run = Path(runs[str(seed)])
        cfg = read(run/'contract.json')
        need(cfg['recipe']['seed'] == seed and read(run/'run_state.json')['status'] == 'completed', 'Incomplete seed')
        saved = torch.load(run/'best.pt', map_location='cpu', weights_only=True)
        hist = read(run/'history.json')
        best = min(hist, key=lambda x: x['validation']['CE'])
        need(saved['kind'] == 'formal' and saved['epoch'] == best['epoch'] and
             saved['contract_sha256'] == cfg['contract_sha256'], 'Wrong seed checkpoint/selection')
        val = saved['validation']
        logits, labels = val['logits'], val['labels']
        need(logits.shape == (1547, 2) and labels.shape == (1547,) and
             len(val['video_ids']) == 1547 and torch.isfinite(logits).all().item(), 'Invalid saved validation tensors')
        probs, pred = logits.softmax(1)[:, 1].tolist(), logits.argmax(1).tolist()
        ta = [{'video_id': vid, 'true_label': int(y), 'prediction': int(p), 'p_make': float(q)}
              for vid, y, p, q in zip(val['video_ids'], labels.tolist(), pred, probs)]
        base = Path(plan['mainline_runs'][str(seed)])
        with (base/'validation_predictions.csv').open() as f:
            ma = [{'video_id': row['video_id'], 'true_label': int(row['true_label']),
                   'prediction': int(row['predicted_label']), 'p_make': float(row['p_make'])} for row in csv.DictReader(f)]
        rows = align_predictions(ta, ma, canonical)
        need(len(rows) == 1547, 'Incomplete validation pairing')
        m, t = aggregate(rows, 'main_prediction'), aggregate(rows, 'tanet_prediction')
        need(m['confusion_matrix'] == read(base/'summary.json')['validation']['confusion_matrix'],
             'Saved mainline validation does not reproduce')
        need(t['confusion_matrix'] == best['validation']['confusion_matrix_Miss_Make'] and
             abs(float(torch.nn.functional.cross_entropy(logits, labels))-best['validation']['CE']) < 1e-7,
             'Saved TANet validation does not reproduce')
        with (output/f'paired_seed{seed}.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        details.append({'seed': seed, 'TANet_run': str(run), 'mainline_run': str(base),
                        'TANet_best_epoch': saved['epoch'], 'mainline_best_epoch': read(base/'summary.json')['best_epoch'],
                        'TANet_validation_CE': best['validation']['CE'],
                        'mainline_validation_BCE': read(base/'summary.json')['validation']['loss'],
                        'TANet': t, 'mainline': m, 'paired_counts': dict(Counter(row['pattern'] for row in rows)),
                        'accuracy_difference_TANet_minus_mainline': t['accuracy']-m['accuracy']})
    aggregates = {}
    for model in ('TANet', 'mainline'):
        aggregates[model] = {metric: distribution([d[model][metric] for d in details])
                             for metric in ('accuracy', 'macro_f1')}
        for cls in ('Miss', 'Make'):
            aggregates[model][cls+'_recall'] = distribution([d[model]['per_class'][cls]['recall'] for d in details])
    result = {'status': 'completed', 'seeds': list(SEEDS), 'validation_videos_each_seed': 1547,
              'details': details, 'aggregate': aggregates,
              'paired_accuracy_difference': distribution([d['accuracy_difference_TANet_minus_mainline'] for d in details]),
              'sd_definition': 'Sample standard deviation across three training seeds; not a confidence interval over sources/videos',
              'limitations': ['Same validation used for checkpoint selection', 'Systems differ in input and training settings',
                              'Three seeds are descriptive variability, not proof of superiority', 'No seed selected for deployment'],
              'new_predictions_computed': False, 'test_used': False, 'completed_at': stamp()}
    atomic_save(output/'summary.json', result, json_format=True)
    rows = ''.join(f'<tr><td>{d["seed"]}</td><td>{100*d["mainline"]["accuracy"]:.2f}%</td><td>{100*d["TANet"]["accuracy"]:.2f}%</td><td>{d["mainline"]["macro_f1"]:.4f}</td><td>{d["TANet"]["macro_f1"]:.4f}</td><td>{d["TANet_best_epoch"]}</td></tr>' for d in details)
    stats = ''.join(f'<tr><td>{model}</td><td>{100*aggregates[model]["accuracy"]["mean"]:.2f}% ± {100*aggregates[model]["accuracy"]["sample_sd"]:.2f} pp</td><td>{aggregates[model]["macro_f1"]["mean"]:.4f} ± {aggregates[model]["macro_f1"]["sample_sd"]:.4f}</td></tr>' for model in ('mainline', 'TANet'))
    (output/'index.html').write_text('<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>TANet三seed對照</title><style>body{font:18px system-ui;line-height:1.8;max-width:1100px;margin:32px auto;padding:20px}td,th{border:1px solid #ccc;padding:10px}table{border-collapse:collapse}</style><h1>TANet三seed validation對照</h1><p>1547支相同validation，每個seed以最低validation loss選模。以下±為三seed樣本標準差，非信賴區間。未執行test。</p><table><tr><th>Seed</th><th>主線accuracy</th><th>TANet accuracy</th><th>主線macro-F1</th><th>TANet macro-F1</th><th>TANet最佳epoch</th></tr>'+rows+'</table><h2>三seed描述統計</h2><table><tr><th>系統</th><th>Accuracy mean ± SD</th><th>Macro-F1 mean ± SD</th></tr>'+stats+'</table><p>系統的輸入與訓練設定不同；三seed結果不構成純架構消融或穩定優勢證明。類別召回率、各seed逐片對照與細項請見<a href="summary.json">完整數據</a>。</p></html>')
    return result


def execute(output):
    output = Path(output).resolve()
    need(output.parent == (FOLDER/'artifacts').resolve() and output.name.startswith('multiseed_'), 'Foreign campaign')
    plan = read(output/'plan.json')
    need(plan['new_seeds'] == list(NEW_SEEDS) and plan['seeds'] == list(SEEDS) and
         [item['seed'] for item in plan['queue']] == list(NEW_SEEDS) and
         plan['formal_training_explicitly_authorized'] is True and plan['test_access'] is False, 'Campaign scope changed')
    with (FOLDER/'artifacts/multiseed_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(output/'state.json')
        need(state['status'] == 'prepared' and not state['jobs'], 'Campaign already started; no automatic restart')
        state.update(status='running', supervisor=process_identity(os.getpid()), started_at=stamp())
        atomic_save(output/'state.json', state, json_format=True)
        runs = {'20261025': plan['original_seed20261025']}
        try:
            for item in plan['queue']:
                verify_sources(plan)
                ready = run_child(output, state, item, 'readiness', command_for(item, 'readiness'))
                passed = read(ready/'report.json')
                need(passed['contract_sha256'] == item['contract_sha256'] and
                     passed['status'] == 'GPU_readiness_passed_formal_start_pending' and
                     all(passed['checks'].values()), 'Fresh repeat readiness failed')
                verify_sources(plan)
                run = run_child(output, state, item, 'training', command_for(item, 'training', ready/'report.json'))
                need(read(run/'run_state.json')['status'] == 'completed', 'Training process exited before completion')
                runs[str(item['seed'])] = str(run)
                state['formal_runs'] = runs
                atomic_save(output/'state.json', state, json_format=True)
            verify_sources(plan)
            summarize(output, plan, runs)
            state.update(status='completed', completed_at=stamp(), active_seed=None, active_stage=None,
                         summary='summary.json', test_used=False)
            atomic_save(output/'state.json', state, json_format=True)
        except BaseException as exc:
            state.update(status='failed_or_interrupted', error=f'{type(exc).__name__}: {exc}', failed_at=stamp())
            atomic_save(output/'state.json', state, json_format=True)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', type=Path)
    args = parser.parse_args()
    return prepare() if args.prepare else execute(args.run)


if __name__ == '__main__':
    main()
