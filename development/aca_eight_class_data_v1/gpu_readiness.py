"""Fixed 24-train/8-validation CUDA engineering probe; no formal training CLI.

Reuses the verified loop's _run state machine unchanged. Four real optimizer
steps maximum across a continuous run and an interrupted/resumed comparison.
Probe checkpoints live only in TemporaryDirectory and cannot initialize a run.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import loop_core as core
from .adapter import CLASSES, EightClassDataset, collate_eight_class
from .foundation import ROOT, CONTRACT_SHA256, load_contract, audit_rows, baseline
from .model_adapter import SPEC_SHA256, initialized_baseline8, load_model_contract, baseline_training_mode
from development.aca_four_arm_campaign_v1.gpu_guard import make_policy, assert_idle_gpu

SCHEMA = 'baseline8_cuda_readiness_only_v1'
SEED = 20261024
need = core.need


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def configure():
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Set deterministic workspace before launch')
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Need one CUDA device; no CPU fallback')
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    reset_seed()


def reset_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def worker_init(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def choose_ids():
    rows, _ = audit_rows(load_contract())
    selected = {}
    for split, per_class in (('train', 3), ('validation', 1)):
        selected[split] = sorted(vid for cls in CLASSES for vid in
            sorted(r['video_id'] for r in rows if r['audit_split'] == split and r['original_class'] == cls)[:per_class])
    need(len(selected['train']) == 24 and len(selected['validation']) == 8, 'Probe cohort mismatch')
    return selected


class OrderedStream:
    """Two spawn workers, separate RNG, exact predeclared subset (no extra prefetch IDs)."""
    def __init__(self, datasets, reference, *, cursor=0, train_stop=24):
        self.datasets, self.reference = datasets, reference
        self.orders = {'train': torch.randperm(24, generator=torch.Generator().manual_seed(SEED)).tolist()[cursor:train_stop],
                       'validation': list(range(8))}
        self.iterators, self.positions = {}, {'train': 0, 'validation': 0}
        self.observed = []

    def __call__(self, split, index):
        position = self.positions[split]
        need(position < len(self.orders[split]) and self.orders[split][position] == index, 'Stream index mismatch')
        if split not in self.iterators:
            before = core.capture_rng()
            loader = DataLoader(Subset(self.datasets[split], self.orders[split]), batch_size=1,
                                num_workers=2, multiprocessing_context='spawn', persistent_workers=False,
                                prefetch_factor=2, collate_fn=collate_eight_class, worker_init_fn=worker_init,
                                generator=torch.Generator().manual_seed(SEED + 100000), pin_memory=False)
            self.iterators[split] = iter(loader)
            need(equal(before, core.capture_rng()), 'Worker startup consumed model RNG')
        batch = next(self.iterators[split])
        record = batch['records'][0]
        expected = self.reference[record['video_id']]
        need(record['split'] == split and batch['input_contract_sha256'] == CONTRACT_SHA256 and
             tuple(batch['tensor'].shape) == (1, 3, 50, 120, 160) and
             batch['targets'].item() == expected['class_id'] and
             batch['binary_targets'].item() == expected['binary_label'] and
             tensor_sha(batch['tensor'][0]) == expected['tensor_sha256'], 'Worker/reference mismatch')
        self.positions[split] += 1
        self.observed.append((split, index, record['video_id']))
        if self.positions[split] == len(self.orders[split]):
            # Exhaustion joins workers. No private DataLoader lifecycle APIs.
            try:
                next(self.iterators[split])
            except StopIteration:
                del self.iterators[split]
            else:
                raise ValueError('Unexpected extra batch')
        return batch

    def close(self):
        self.iterators.clear()
        gc.collect()


class ReadinessLoop(core.FixtureLoop):
    """CUDA transport/checkpoint adapter around the exact existing _run implementation.

    Only the single prescribed 24+8 probe is accepted. Resume deliberately accepts
    only its epoch0/cursor16 checkpoint, never arbitrary/formal/old task weights.
    """
    def __init__(self, model, plan, fetch, checkpoint):
        need(plan['schema'] == SCHEMA and plan['max_epochs'] == 1 and plan['effective'] == 16 and
             plan['patience'] == 5 and plan['checkpoint_every'] == 25 and plan['seed'] == SEED and
             plan['spec_sha256'] == SPEC_SHA256 and plan['device'] == 'cuda' and
             len(plan['train_ids']) == 24 and len(plan['validation_ids']) == 8 and
             plan['formal_training_authorized'] is False, 'Not a bounded readiness plan')
        need(all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in model.parameters()), 'Need CUDA FP32 model')
        baseline_training_mode(model)
        self.model, self.plan, self.fetch, self.checkpoint = model, copy.deepcopy(plan), fetch, Path(checkpoint)
        self.optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                         lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0.,
                                         amsgrad=False, foreach=False, fused=False)
        self.state = {'epoch': 0, 'cursor': 0, 'updates': 0, 'train_metrics': core.metrics_state(core.EightClassMetrics()),
                      'history': [], 'best_epoch': None, 'best_ce': None, 'bad_epochs': 0}
        self.best_model, self.valid, self.started = None, True, False

    def _batch(self, split, index):
        x, y = super()._batch(split, index)
        return x.cuda(), y.cuda()

    def _payload(self, state=None, best=None):
        return {'schema': SCHEMA, 'plan_sha256': core.plan_digest(self.plan), 'plan': copy.deepcopy(self.plan),
                'model': cpu_tree(self.model.state_dict()), 'optimizer': cpu_tree(self.optimizer.state_dict()),
                'rng': core.capture_rng(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'state': copy.deepcopy(self.state if state is None else state),
                'best_model': cpu_tree(self.best_model if best is None else best)}

    def resume(self):
        need(not self.started, 'Explicit resume before run only')
        payload = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        need(payload['schema'] == SCHEMA and payload['plan'] == self.plan and
             payload['plan_sha256'] == core.plan_digest(self.plan), 'Checkpoint identity mismatch')
        core.finite_tree(payload)
        state = payload['state']
        metric = core.restore_metrics(state['train_metrics'])
        need(state == {'epoch': 0, 'cursor': 16, 'updates': 1, 'train_metrics': core.metrics_state(metric),
                       'history': [], 'best_epoch': None, 'best_ce': None, 'bad_epochs': 0} and
             metric._count == 16 and payload['best_model'] is None, 'Not the authorized probe restart point')
        reference = self.model.state_dict()
        need(payload['model'].keys() == reference.keys() and all(
            payload['model'][k].shape == reference[k].shape and payload['model'][k].dtype == reference[k].dtype
            for k in reference), 'Checkpoint model signature mismatch')
        reference_groups = self.optimizer.state_dict()['param_groups']
        need(payload['optimizer']['param_groups'] == reference_groups, 'Adam configuration mismatch')
        params = self.optimizer.param_groups[0]['params']
        indices = reference_groups[0]['params']
        need(set(payload['optimizer']['state']) == set(indices), 'Adam state coverage mismatch')
        for index, param in zip(indices, params):
            value = payload['optimizer']['state'][index]
            need(set(value) == {'step', 'exp_avg', 'exp_avg_sq'} and value['step'].numel() == 1 and
                 value['step'].item() == 1 and all(value[k].shape == param.shape and value[k].dtype == param.dtype
                 for k in ('exp_avg', 'exp_avg_sq')), 'Adam shape/step mismatch')
        original = core.capture_rng()
        try:
            core.restore_rng(payload['rng'])
        finally:
            core.restore_rng(original)
        need(len(payload['cuda_rng']) == 1, 'CUDA RNG count mismatch')
        torch.Generator(device='cuda').set_state(payload['cuda_rng'][0])
        self.model.load_state_dict(payload['model'], strict=True)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.state, self.best_model = copy.deepcopy(state), None
        self.optimizer.zero_grad(set_to_none=True)
        core.restore_rng(payload['rng'])
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
        self.started = True


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def run(source_root, output):
    need(not output.exists(), 'Do not overwrite a previous readiness report')
    output.mkdir(parents=True)
    started = time.monotonic()
    json_write(output / 'status.json', {'status': 'checking_resources', 'formal_training_started': False})
    temp_path = None
    try:
        spec = load_model_contract()
        policy = make_policy()
        observed = assert_idle_gpu(policy)
        need(shutil.disk_usage('/tmp').free >= 4 * 1024**3, 'Need 4GiB temporary checkpoint space')
        configure()
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total',
                                        '--format=csv,noheader,nounits'], text=True).strip()
        selected = choose_ids()
        datasets = {split: EightClassDataset(source_root=source_root, split=split, video_ids=ids)
                    for split, ids in selected.items()}
        reference, sources = {}, {}
        print('STAGE: serial input reference, exactly 24 train + 8 validation', flush=True)
        for split, dataset in datasets.items():
            for index, vid in enumerate(selected[split]):
                sources[vid] = baseline.guarded_source_sha256(dataset.foundation.scope, vid)
                item = dataset[index]
                reference[vid] = {'class_id': item['class_id'], 'binary_label': item['binary_label'],
                                  'tensor_sha256': tensor_sha(item['tensor'])}
        del item
        plan = {'schema': SCHEMA, 'purpose': 'engineering_only_not_accuracy_evidence',
                'formal_training_authorized': False, 'test_enabled': False,
                'seed': SEED, 'effective': 16, 'patience': 5, 'max_epochs': 1, 'checkpoint_every': 25,
                'device': 'cuda', 'spec_sha256': SPEC_SHA256, 'input_contract_sha256': CONTRACT_SHA256,
                'train_ids': selected['train'], 'validation_ids': selected['validation'],
                'reference': reference, 'source_sha256': sources, 'source_root': str(source_root),
                'gpu_policy': policy,
                'bounds': {'unique_train': 24, 'unique_validation': 8, 'total_real_optimizer_steps': 4,
                           'total_real_forward_backward_samples': 48, 'total_real_eval_samples': 16,
                           'total_video_decodes': 96, 'wall_timeout_seconds_external': 900,
                           'checkpoint_initialization_for_formal_training': 'prohibited'},
                'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'numpy': np.__version__, 'opencv': cv2.__version__, 'cuda': torch.version.cuda,
                            'cudnn': torch.backends.cudnn.version(), 'gpu': gpu, 'workers': 2,
                            'spawn': True, 'threads': 4, 'amp': False, 'tf32': False,
                            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']},
                'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (
                    Path(__file__), Path(core.__file__), ROOT/'development/aca_eight_class_data_v1/adapter.py',
                    ROOT/'development/aca_eight_class_data_v1/model_adapter.py',
                    ROOT/'development/aca_eight_class_data_v1/foundation.py',
                    ROOT/'development/aca_four_arm_campaign_v1/gpu_guard.py')}}
        json_write(output/'plan.json', plan)
        json_write(output/'status.json', {'status': 'running_bounded_probe', 'formal_training_started': False})
        # Verify saved CUDA RNG actually reproduces a subsequent draw, then restore it.
        rng = torch.cuda.get_rng_state_all()
        expected_draw = torch.rand(32, device='cuda')
        torch.cuda.set_rng_state_all(rng)
        need(torch.equal(expected_draw, torch.rand(32, device='cuda')), 'CUDA RNG roundtrip mismatch')
        torch.cuda.set_rng_state_all(rng)
        del expected_draw
        totals = {'updates': 0}
        observed_runs, gradient_checks, saved_sizes = {}, [], []
        with tempfile.TemporaryDirectory(prefix='baseline8_cuda_readiness_') as scratch:
            temp_path = scratch
            def perform(name, *, cursor=0, stop_at_16=False, resume=False):
                assert_idle_gpu(policy)
                reset_seed()
                model, init = initialized_baseline8()
                baseline_training_mode(model)
                frozen = {k: v.clone() for k, v in model.state_dict().items()
                          if k not in {n for n, p in model.named_parameters() if p.requires_grad}}
                initial = {k: model.state_dict()[k].clone() for k in ('conv1.weight', 'layer4.2.conv3.weight', 'fc.weight')}
                model.cuda()
                stream = OrderedStream(datasets, reference, cursor=cursor, train_stop=16 if stop_at_16 else 24)
                path = Path(scratch) / ('interrupted.pt' if resume or stop_at_16 else 'continuous.pt')
                loop = ReadinessLoop(model, plan, stream, path)
                if resume:
                    loop.resume()
                original_step = loop.optimizer.step
                def bounded_step(*args, **kwargs):
                    need(totals['updates'] < 4 and time.monotonic()-started < 900, 'Probe execution cap reached')
                    groups = {'backbone': 0., 'head': 0.}
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            need(param.grad is not None and torch.isfinite(param.grad).all().item(), 'Nonfinite/missing gradient')
                            groups['head' if name.startswith('fc.') else 'backbone'] += param.grad.abs().sum().item()
                        else:
                            need(param.grad is None, 'Frozen BN received gradient')
                    need(all(v > 0 for v in groups.values()), 'Missing backbone/head gradients')
                    result = original_step(*args, **kwargs)
                    totals['updates'] += 1
                    gradient_checks.append(groups)
                    print(f'STAGE: {name_of_run}, engineering update {totals["updates"]}/4', flush=True)
                    return result
                name_of_run = name
                loop.optimizer.step = bounded_step
                torch.cuda.reset_peak_memory_stats()
                phase_start = time.monotonic()
                try:
                    status = loop.run(lambda phase, state, seen: stop_at_16 and phase == 'train' and state['updates'] == 1)
                    torch.cuda.synchronize()
                    need(status == ('paused' if stop_at_16 else 'completed'), 'Unexpected loop status')
                    need(all(torch.equal(model.state_dict()[k].cpu(), v) for k, v in frozen.items()), 'Frozen BN state changed')
                    need(all(not torch.equal(model.state_dict()[k].cpu(), v) for k, v in initial.items()), 'Backbone/head did not update')
                    payload = loop._payload()
                    need(equal(payload, torch.load(path, map_location='cpu', weights_only=True)), 'Disk checkpoint differs from committed live state')
                    observed_runs[name] = {'status': status, 'stream': stream.observed,
                                          'updates': loop.state['updates'], 'cursor': loop.state['cursor'],
                                          'history_epochs': len(loop.state['history']),
                                          'seconds': time.monotonic()-phase_start,
                                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                          'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                                          'frozen_bn_unchanged': True, 'backbone_and_head_updated': True}
                    saved_sizes.append(path.stat().st_size)
                    return payload
                finally:
                    stream.close()
                    loop.optimizer.step = original_step
                    del loop, model, original_step, bounded_step
                    gc.collect()
                    torch.cuda.empty_cache()
            continuous = perform('continuous')
            paused = perform('interrupted', stop_at_16=True)
            need(paused['state']['cursor'] == 16 and paused['state']['updates'] == 1 and
                 not paused['state']['history'] and paused['best_model'] is None, 'Wrong pause checkpoint')
            del paused
            resumed = perform('resumed', cursor=16, resume=True)
            comparisons = {key: equal(continuous[key], resumed[key]) for key in continuous}
            need(all(comparisons.values()), 'Continuous/resumed trajectory mismatch: ' + repr(comparisons))
            need(totals['updates'] == 4 and resumed['state']['updates'] == 2 and
                 resumed['state']['history'][0]['train']['count'] == 24 and
                 resumed['state']['history'][0]['validation']['count'] == 8, 'Wrong final coverage')
            del continuous, resumed
        need(not Path(temp_path).exists(), 'Temporary checkpoints were not removed')
        for split, dataset in datasets.items():
            for vid in selected[split]:
                need(baseline.guarded_source_sha256(dataset.foundation.scope, vid) == sources[vid], 'Original video changed')
        need(all(sha(ROOT/p) == v for p, v in plan['code_sha256'].items()), 'Readiness code changed during execution')
        load_model_contract()
        report = {'schema': SCHEMA, 'status': 'passed', 'plan_sha256': core.plan_digest(plan),
                  'formal_training_started': False, 'formal_training_ready': False, 'test_videos_opened': 0,
                  'real_optimizer_updates_engineering_only': totals['updates'], 'gpu_rng_draw_roundtrip_equal': True,
                  'continuous_vs_resumed_equal': comparisons, 'workers_equal_serial_tensor_sha': True,
                  'worker_rng_isolated': True, 'original_videos_unchanged': True,
                  'runs': observed_runs, 'gradient_checks': gradient_checks, 'gpu_processes_before_start': observed,
                  'checkpoint_sizes_bytes': saved_sizes, 'temporary_checkpoints_removed': True,
                  'elapsed_seconds': time.monotonic()-started,
                  'scope_note': '32 unique real train/validation clips, 96 bounded decodes, 4 temporary engineering updates; no scores are research results.',
                  'remaining': ['Formal run prelaunch plan/guard and all-7192/1547 loader identity binding.',
                                'General CUDA resume including interrupted validation and later epochs; readiness adapter only accepts cursor16.',
                                'No author-exact ACA claim or accuracy conclusion.']}
        json_write(output/'report.json', report)
        json_write(output/'status.json', {'status': 'passed', 'formal_training_started': False})
        print('PASSED: bounded real input/CUDA/worker/checkpoint comparison; no formal run started', flush=True)
    except BaseException as exc:
        json_write(output/'failure.json', {'error': repr(exc), 'formal_training_started': False,
                                           'elapsed_seconds': time.monotonic()-started})
        json_write(output/'status.json', {'status': 'failed_stopped', 'formal_training_started': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source_root, args.output)


if __name__ == '__main__':
    main()
