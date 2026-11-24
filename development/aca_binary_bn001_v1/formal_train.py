"""One fresh binary ACA run, only TSCI BN momentum changed; no auto-restart."""
import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from development.aca_binary_v1 import formal_train as base, core, runtime
from development.aca_binary_v1.data import RevisedDataset, EpochStream
from .model import initialized, validate_momentum

ROOT = base.ROOT
ORIGINAL = base.original
PARENT = base.RUN
PARENT_SHA = 'd44e193fe8e12c51ed6800e50c884e6473222a866af9361e0d3ba549555034aa'
RUN = base.CURRENT/'aca_binary_bn001_seed20261024_labelrev1_v1'
QA = base.CURRENT/'aca_binary_bn001_readiness_20261122_v1'
PREFIX = 'development/aca_binary_bn001_v1/'
need = core.need


def parent_result(plan):
    summary = ORIGINAL.read(PARENT/'summary.json')
    history = ORIGINAL.read(PARENT/'history.json')
    identity = ORIGINAL.read(PARENT/'run.json')
    need(ORIGINAL.read(PARENT/'plan.json') == plan and core.plan_digest(plan) == PARENT_SHA,
         'Original binary plan changed')
    need(summary['status'] == 'completed' and summary['test_run'] is False and
         identity['run_id'] == summary['run_id'] and
         identity['plan_sha256'] == summary['plan_sha256'] == PARENT_SHA,
         'Original binary completion identity mismatch')
    need(len(history) == summary['completed_epochs'] > 0, 'Incomplete parent history')
    best = min(history, key=lambda r: r['validation']['ce'])
    bad = len(history)-best['epoch']
    state = dict(epoch=len(history), cursor=0, updates=summary['updates'],
                 train_metrics=core.metrics_state(core.BinaryMetrics()), history=history,
                 best_epoch=best['epoch'], best_ce=best['validation']['ce'], bad_epochs=bad)
    runtime.validate_state(state, plan, {})
    need(summary['selected_validation'] == best['validation'] and summary['best_epoch'] == best['epoch'] and
         (bad >= plan['patience'] or len(history) == plan['max_epochs']) and
         summary['stop_reason'] == ('early_stop' if bad >= plan['patience'] else 'epoch_cap'),
         'Original binary selection mismatch')
    return dict(directory=str(PARENT), run_id=identity['run_id'], plan_sha256=PARENT_SHA,
                summary_sha256=ORIGINAL.sha(PARENT/'summary.json'),
                history_sha256=ORIGINAL.sha(PARENT/'history.json'))


def draft_plan():
    parent = base.build_plan()  # Revalidate unchanged sources, runtime, data and old acceptance.
    reference = parent_result(parent)
    plan = deepcopy(parent)
    plan['purpose'] = 'single_seed_direct_binary_TSCI_BN_momentum_0.1_vs_0.01_controlled_comparison'
    plan['allowed_run_directory'] = str(RUN)
    plan['normalization_policy']['new_tsci_bn_momentum'] = 0.01
    plan['momentum_experiment'] = dict(
        schema='tsci_bn_momentum_single_change_v1', baseline=reference,
        changed_field='TSCI_BatchNorm2d_momentum', before=0.1, after=0.01, module_count=48,
        unchanged_pooling='mean_then_amax', initialization='same_fresh_parameters_buffers_and_RNG',
        reference_commit='2cf02fb91196408ac6352f7984a0336aa4f7677a',
        author_exact_configuration_verified=False)
    for path in sorted(Path(__file__).parent.glob('*.py')):
        plan['code_sha256'][str(path.relative_to(ROOT))] = ORIGINAL.sha(path)
    changed = {k for k in plan if plan[k] != parent.get(k)}
    need(changed == {'purpose','allowed_run_directory','normalization_policy','momentum_experiment','code_sha256'},
         'Unexpected protocol change')
    return plan


def build_plan():
    plan = draft_plan()
    for name in ('cpu_report.json','gpu_report.json'):
        evidence = ORIGINAL.read(QA/name)
        need(evidence['status'] == 'passed' and evidence['test_video_decodes'] == 0 and
             evidence['draft_plan_sha256'] == core.plan_digest(plan) and
             evidence['code_sha256'] == {k:v for k,v in plan['code_sha256'].items() if k.startswith(PREFIX)},
             'Momentum acceptance or implementation mismatch')
    plan['momentum_readiness'] = {name: ORIGINAL.sha(QA/name) for name in ('cpu_report.json','gpu_report.json')}
    return plan


def verify_plan(plan, expected_hash):
    need(core.plan_digest(plan) == expected_hash and plan == build_plan(), 'Momentum plan mismatch')


class TrainingLoop(base.TrainingLoop):
    def __init__(self, model, plan, *args, **kwargs):
        validate_momentum(model)
        need(plan['momentum_experiment']['after'] == 0.01 and
             plan['normalization_policy']['new_tsci_bn_momentum'] == 0.01, 'Wrong momentum plan')
        super().__init__(model, plan, *args, **kwargs)


def run_path(variant):
    need(variant == 'aca', 'Only ACA seed24')
    return RUN


def require_predecessor(plan):
    need(parent_result(base.build_plan()) == plan['momentum_experiment']['baseline'], 'Parent changed')


def execute(plan_path, digest):
    ORIGINAL.verify_plan = verify_plan
    ORIGINAL.run_path = run_path
    ORIGINAL.require_predecessor = require_predecessor
    ORIGINAL.initialized = initialized
    ORIGINAL.rt = SimpleNamespace(**{**vars(runtime), 'VerifiedDataset': RevisedDataset,
                                     'EpochStream': EpochStream, 'TrainingLoop': TrainingLoop})
    return ORIGINAL.execute(plan_path, digest, RUN, resume=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare'); q.add_argument('--output', type=Path, required=True)
    q = sub.add_parser('start'); q.add_argument('--plan', type=Path, required=True); q.add_argument('--plan-sha256', required=True)
    a = p.parse_args()
    if a.command == 'prepare':
        plan = build_plan(); processes = ORIGINAL.assert_idle_gpu(plan['gpu_policy'])
        out = ORIGINAL.safe_new_directory(a.output)
        runtime.atomic_json(out/'plan.json', plan)
        runtime.atomic_json(out/'review.json', dict(status='prepared_not_started', plan_sha256=core.plan_digest(plan),
                            run_directory=str(RUN), gpu_processes=processes, only_numerical_change='TSCI BN momentum 0.1 -> 0.01'))
        print(core.plan_digest(plan), flush=True)
    else:
        execute(a.plan, a.plan_sha256)


if __name__ == '__main__': main()
