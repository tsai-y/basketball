"""Explicit three-seed L-policy campaign; sequential fresh readiness and training.

Existing engines, checkpoints, dataset and diagnostic artifacts stay immutable.
No retries, automatic resumes, test evaluation or winning-seed selection.
"""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from development.tanet_v1 import repeat_seeds_v1 as old
from .config import FOLDER, need, sha256
from .eval_policy import POLICY
from .training import atomic_save, contract

ROOT = FOLDER.parents[1]
SEEDS = (20261024, 20261025, 20261026)
FILES = (*old.FILES, 'eval_policy.py')
OLD_RUNS = {
    20261024: ROOT/'development/tanet_seed20261024_v1/artifacts/formal_train_3auu0x0k',
    20261025: ROOT/'development/tanet_v1/artifacts/formal_train_ji1xnlg_',
    20261026: ROOT/'development/tanet_seed20261026_v1/artifacts/formal_train_y6xhlt7x',
}
read, stamp = old.read, old.stamp


def expected_json(name, seed):
    need(type(seed) is int and seed in SEEDS, 'Only the three declared seeds are authorized')
    cfg = read(FOLDER/name)
    cfg['seed'] = seed
    return cfg


def check_package(folder, seed):
    folder = Path(folder)
    for name in FILES:
        path = folder/name
        need(path.is_file() and not path.is_symlink(), 'Missing/symlinked L source: '+str(path))
        if name in ('config.json', 'train_config.json'):
            need(read(path) == expected_json(name, seed), 'Variant changed beyond seed: '+name)
        else:
            need(sha256(path) == sha256(FOLDER/name), 'Variant engines differ: '+name)
    return {name: sha256(folder/name) for name in FILES}


def clone(folder, seed):
    need(not folder.exists(), 'Existing package must not be overwritten')
    folder.mkdir()
    for name in FILES:
        target = folder/name
        target.parent.mkdir(exist_ok=True)
        if name in ('config.json', 'train_config.json'):
            target.write_text(json.dumps(expected_json(name, seed), indent=2)+'\n')
        else:
            shutil.copy2(FOLDER/name, target)
    (folder/'artifacts').mkdir()
    return check_package(folder, seed)


def verify_sources(plan):
    need(all(sha256(p) == h for p, h in plan['protected_sha256'].items()), 'Pinned original/campaign source changed')
    for item in plan['queue']:
        need(check_package(item['package'], item['seed']) == item['source_sha256'], 'L package changed')


def prepare():
    base, _, _ = contract()
    original_recipe = read(old.FOLDER/'train_config.json')
    expected = {**original_recipe, 'schema': 'tanet_uniform50_L_eval_recipe_v2', 'evaluation_policy': POLICY}
    need(base['recipe'] == expected, 'Only the declared evaluation policy may change')
    # Include the complete historical protection chain, plus all old seed engines.
    previous = old.FOLDER/'artifacts/validation_L_v1_40caaq7i'
    protected = dict(read(previous/'plan.json')['protected_sha256'])
    for name in ('plan.json', 'report.json', 'qa.json', 'interpretation.json'):
        protected[str(previous/name)] = sha256(previous/name)
    historical = old.FOLDER/'artifacts/multiseed_c0qf_w7l/summary.json'
    protected[str(historical)] = sha256(historical)
    for seed, run in OLD_RUNS.items():
        need(read(run/'run_state.json')['status'] == 'completed', 'Historical seed incomplete')
        package = run.parent.parent
        original_contract = importlib.import_module('development.'+package.name+'.training').contract()[0]
        need(original_contract == read(run/'contract.json'), 'Historical contract changed')
        for name in old.FILES:
            protected[str(package/name)] = sha256(package/name)
        for name in ('contract.json', 'history.json', 'run_state.json', 'best.pt', 'last.pt'):
            protected[str(run/name)] = sha256(run/name)
    for name in (*FILES, 'campaign.py', 'README.md', 'tests/test_L_v2.py'):
        protected[str(FOLDER/name)] = sha256(FOLDER/name)
    for name in ('repeat_seeds_v1.py', 'analyze_v1.py', 'probe_bn_behavior_v1.py'):
        protected[str(old.FOLDER/name)] = sha256(old.FOLDER/name)
    need(all(sha256(p) == h for p, h in protected.items()), 'Original protection audit failed')
    destinations = {seed: FOLDER if seed == 20261025 else
                    ROOT/'development'/f'tanet_l_seed{seed}_v2' for seed in SEEDS}
    need(all(not folder.exists() for seed, folder in destinations.items() if seed != 20261025),
         'L repeat packages already exist; no overwrite/restart')
    output = Path(tempfile.mkdtemp(prefix='three_seed_', dir=FOLDER/'artifacts'))
    queue = []
    for seed in SEEDS:
        folder = destinations[seed]
        hashes = check_package(folder, seed) if folder == FOLDER else clone(folder, seed)
        module = 'development.'+folder.name
        variant = importlib.import_module(module+'.training').contract()[0]
        queue.append({'seed': seed, 'package': str(folder), 'module': module,
                      'contract_sha256': variant['contract_sha256'], 'source_sha256': hashes})
    plan = {'schema': 'tanet_L_per_video_validation_three_seed_v2', 'created_at': stamp(),
            'seeds': list(SEEDS), 'queue': queue, 'evaluation_policy': POLICY,
            'recipe': base['recipe'], 'mainline_runs': old.validate_mainlines(),
            'historical_TANet_summary': str(historical),
            'historical_TANet_runs': {str(k): str(v) for k, v in OLD_RUNS.items()},
            'protected_sha256': protected, 'fresh_ImageNet_initialization_each_seed': True,
            'selection': 'actual_minimum_full_validation_CE_with_fixed_L_policy_each_epoch',
            'formal_training_explicitly_authorized': True, 'serial_gpu_jobs': True,
            'failure_policy': 'stop_without_retry_or_automatic_resume',
            'test_access': False, 'old_runs_modified': False, 'max_epochs': 15,
            'limitations': ['Policy chosen after inspecting the same validation split; exploratory comparison',
                            'No untouched test estimate and no deployment seed selection']}
    atomic_save(output/'plan.json', plan, json_format=True)
    atomic_save(output/'state.json', {'status': 'prepared', 'jobs': [], 'test_used': False,
                'plan_sha256': sha256(output/'plan.json')}, json_format=True)
    print(output, flush=True)
    return output


def summarize(output, plan, runs):
    # Reuse the validated ID-aligned saved-logit aggregator, never a new forward.
    for item in plan['queue']:
        saved_contract = read(Path(runs[str(item['seed'])])/'contract.json')
        need(saved_contract['contract_sha256'] == item['contract_sha256'] and
             saved_contract['recipe']['evaluation_policy'] == POLICY, 'Summary policy/contract mismatch')
    result = old.summarize(output, plan, runs)
    result.update(variant='TANet L per-video statistics v2', evaluation_policy=POLICY,
                  historical_TANet=read(plan['historical_TANet_summary']),
                  same_validation_used_to_choose_policy=True,
                  training_recipe_changes=['validation/inference L statistics policy only'])
    result['limitations'] += plan['limitations']
    atomic_save(output/'summary.json', result, json_format=True)
    page = output/'index.html'
    html = page.read_text().replace('TANet三seed', 'TANet L 當片統計 v2 三seed')
    html = html.replace('<h2>三seed描述統計</h2>',
        '<p>每輪 validation 與選模固定使用 L 當片統計；G/BN2d 使用已存統計，dropout 關閉。'
        '此規則是在觀察同一 validation 後提出，結果屬探索性比較。舊版三 seed 與類別召回率保留於完整數據。</p>'
        '<h2>三seed描述統計</h2>')
    page.write_text(html)
    return result


def execute(output):
    output = Path(output).resolve()
    need(output.parent == (FOLDER/'artifacts').resolve() and output.name.startswith('three_seed_'), 'Foreign L campaign')
    plan, state = read(output/'plan.json'), read(output/'state.json')
    need(state['plan_sha256'] == sha256(output/'plan.json'), 'Campaign plan changed')
    need(plan['seeds'] == list(SEEDS) and [x['seed'] for x in plan['queue']] == list(SEEDS) and
         plan['evaluation_policy'] == POLICY and plan['max_epochs'] == 15 and
         plan['formal_training_explicitly_authorized'] is True and plan['test_access'] is False and
         plan['fresh_ImageNet_initialization_each_seed'] is True, 'Campaign scope changed')
    # Share the original multi-seed lock to avoid overlapping TANet campaigns.
    with (old.FOLDER/'artifacts/multiseed_gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        need(state['status'] == 'prepared' and not state['jobs'], 'Campaign already started; no auto-restart')
        state.update(status='running', supervisor=old.process_identity(os.getpid()), started_at=stamp())
        atomic_save(output/'state.json', state, json_format=True)
        runs = {}
        try:
            for item in plan['queue']:
                verify_sources(plan)
                ready = old.run_child(output, state, item, 'readiness', old.command_for(item, 'readiness'))
                passed = read(ready/'report.json')
                need(passed['contract_sha256'] == item['contract_sha256'] and
                     passed['recipe']['evaluation_policy'] == POLICY and
                     passed['status'] == 'GPU_readiness_passed_formal_start_pending' and
                     all(passed['checks'].values()), 'Fresh L readiness failed')
                verify_sources(plan)
                run = old.run_child(output, state, item, 'training', old.command_for(item, 'training', ready/'report.json'))
                need(read(run/'run_state.json')['status'] == 'completed', 'Training exited before completion')
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
