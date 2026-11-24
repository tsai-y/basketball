"""Bounded GPU engineering readiness check for aca_spatial_v1.

Uses only 2 fixed train clips (04 and 22), 0 validation video decodes, 0 test
access, at most 4 real CUDA engineering updates, temporary checkpoints deleted.
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import tempfile
import time
import traceback

import torch
from torch.utils.data import DataLoader, Subset

from development.aca_binary_v1 import core, runtime
from development.aca_binary_bn001_v1.model import initialized, validate_momentum
from development.aca_eight_class_data_v1.gpu_readiness import configure, reset_seed, worker_init, equal
from . import augment, formal_train as f
from .data import SpatialDataset, SpatialCollator

FIXTURES = ('04', '22')
DIAGNOSTIC = f.CURRENT / 'input_diagnostic32_20261119_v1'
need = core.need


def fixture_batches(plan):
    ids = [json.loads((DIAGNOSTIC / x / 'evidence.json').read_text())['video_id'] for x in FIXTURES]
    ds = SpatialDataset(plan['source_root'], 'train', plan['train_ids'], plan['reference'])
    ds.set_epoch(0)
    indices = [plan['train_ids'].index(vid) for vid in ids]
    serial_items = [ds[i] for i in indices]

    # Verify 2-worker spawn DataLoader matches serial decode bitwise & preserves global RNG.
    rng_before = core.capture_rng()
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        num_workers=2,
        collate_fn=SpatialCollator(),
        worker_init_fn=worker_init,
        multiprocessing_context='spawn',
        prefetch_factor=2,
        generator=torch.Generator().manual_seed(20261024),
        persistent_workers=False,
    )
    worker_batches = list(loader)
    need(equal(rng_before, core.capture_rng()), 'Spawn workers consumed model RNG')
    for item, batch in zip(serial_items, worker_batches):
        need(torch.equal(item['tensor'], batch['tensor'][0]), 'Worker tensor differs from serial')
        need(batch['targets'].tolist() == [item['binary_label']], 'Worker target differs')
        need(batch.get('spatial_policy_sha256') == augment.POLICY_SHA, 'Missing spatial policy SHA')
    return worker_batches, ids


def check_gpu(plan):
    observed = f.assert_idle_gpu(plan['gpu_policy'])
    with f.shell.run_lock(f.CURRENT.parent / '00_admin', 'baseline8_gpu.lock'):
        f.assert_idle_gpu(plan['gpu_policy'])
        configure()
        batches, ids = fixture_batches(plan)

        probe = copy.deepcopy(plan)
        probe.update(
            train_ids=[ids[i % 2] for i in range(24)],
            validation_ids=ids,
            max_epochs=1,
            checkpoint_every=1,
        )

        def fetch(split, index):
            need(split == 'train', 'Engineering validation data access forbidden')
            return batches[index % 2]

        torch.cuda.reset_peak_memory_stats()
        with tempfile.TemporaryDirectory(prefix='aca_spatial_gpu_probe_', dir=f.QA) as temp:
            folder = Path(temp)

            def new_loop(path):
                reset_seed()
                m, _ = initialized('aca', 20261024)
                validate_momentum(m)
                m.cuda()
                return f.TrainingLoop(m, probe, fetch, path, 'engineering_only')

            continuous = new_loop(folder / 'continuous.pt')
            frozen_backbone_bn = {
                n: b.detach().cpu().clone()
                for n, b in continuous.model.named_buffers()
                if '.aca.' not in n
            }
            need(continuous.run(lambda phase, state, seen: phase == 'validation') == 'paused',
                 'Probe failed to pause before validation')
            expected = torch.load(folder / 'continuous.pt', map_location='cpu', weights_only=True)
            need(expected['state']['updates'] == 2 and expected['state']['cursor'] == 24,
                 'Effective16/tail8 update count wrong')
            need(all(torch.equal(b, dict(continuous.model.named_buffers())[n].cpu())
                     for n, b in frozen_backbone_bn.items()),
                 'Frozen backbone BN buffers changed during train')
            need(all(p.grad is None for p in continuous.model.parameters()),
                 'Gradients retained after update')
            del continuous
            gc.collect()
            torch.cuda.empty_cache()

            partial = new_loop(folder / 'resume.pt')
            need(partial.run(lambda phase, state, seen: state['updates'] == 1) == 'paused',
                 'Pause at update 1 failed')
            need(partial.committed['cursor'] == 16, 'Pause not at effective16 boundary')
            del partial
            gc.collect()
            torch.cuda.empty_cache()

            resumed = new_loop(folder / 'resume.pt')
            resumed.resume()
            need(resumed.run(lambda phase, state, seen: phase == 'validation') == 'paused',
                 'Resume pause before validation failed')
            actual = torch.load(folder / 'resume.pt', map_location='cpu', weights_only=True)
            matches = {
                k: equal(expected[k], actual[k])
                for k in ('model', 'optimizer', 'rng', 'cuda_rng', 'state', 'best_model')
            }
            need(all(matches.values()), f'Continuous vs resumed mismatch: {matches}')

            # Verify eval forward does not change any BN buffer (using train fixtures only).
            resumed.model.eval()
            all_bn_before = {n: b.detach().clone() for n, b in resumed.model.named_buffers()}
            metric = core.BinaryMetrics()
            with torch.no_grad():
                for batch in batches:
                    logits = resumed.model(batch['tensor'].cuda())
                    target = batch['targets'].cuda()
                    core.summed_ce(logits, target)
                    metric.add(logits.cpu(), target.cpu())
            need(metric.result()['count'] == 2, 'Metric accumulation failed')
            need(all(torch.equal(b, dict(resumed.model.named_buffers())[n])
                     for n, b in all_bn_before.items()),
                 'Eval forward mutated buffers')
            peak = {
                'allocated_bytes': torch.cuda.max_memory_allocated(),
                'reserved_bytes': torch.cuda.max_memory_reserved(),
            }
            del resumed
            gc.collect()
            torch.cuda.empty_cache()

        need(not folder.exists(), 'Temporary engineering directory still exists')
        return {
            'gpu_processes': observed,
            'train_video_ids': ids,
            'worker_vs_serial_bitwise_equal': True,
            'global_rng_unchanged_by_workers': True,
            'real_optimizer_updates_engineering_only': 4,
            'effective': 16,
            'tail': 8,
            'microbatch': 1,
            'input_shape': [1, 3, 50, 120, 160],
            'continuous_vs_resumed_equal': matches,
            'backbone_frozen_bn_unchanged': True,
            'eval_all_buffers_unchanged': True,
            'tsci_bn_momentum': 0.01,
            'temporary_checkpoints_removed': True,
            'source_train_decodes': 4,
            'validation_video_decodes': 0,
            'test_video_decodes': 0,
            'accuracy_reported': False,
            'peak_memory': peak,
            'formal_weights_from_engineering': False,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['gpu'])
    p.parse_args()
    f.QA.mkdir(exist_ok=True)
    output = f.QA / 'gpu_report.json'
    need(not output.exists(), 'Refusing to overwrite existing gpu_report.json')
    t0 = time.monotonic()
    report = {'status': 'running', 'phase': 'gpu', 'at_taiwan': runtime.now()}
    try:
        plan = f.draft_plan()
        report['draft_plan_sha256'] = core.plan_digest(plan)
        report.update(check_gpu(plan))
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        report['elapsed_seconds'] = round(time.monotonic() - t0, 4)
        runtime.atomic_json(output, report)
        raise
    report['elapsed_seconds'] = round(time.monotonic() - t0, 4)
    runtime.atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
