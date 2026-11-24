"""Bounded CPU + GPU readiness checks for aca_spatial_pair_v1 (seed=20261025).

Verifies:
- CPU: seed20261025 wiring across model init, temporal sampling, spatial/color
  sampling, and epoch read order (distinct from seed20261024), 2-worker spawn
  bitwise identity, global RNG preservation, rejection of seed24 metadata, and
  exact validation delegation (0 validation/test video decodes).
- GPU: both (a) train-boundary effective16/tail8 continuous vs resumed AND
  (b) post-selection (best_model is not None) epoch-boundary continuous vs
  resumed, using repeated train fixtures only (0 validation/test video decodes).
"""
import argparse
import copy
import gc
import json
from pathlib import Path
import tempfile
import time
import traceback
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Subset

from development.aca_binary_v1 import core, runtime
from development.aca_binary_v1.data import RevisedDataset, BinaryCollator
from development.aca_binary_bn001_v1.model import validate_momentum
from development.aca_eight_class_data_v1.gpu_readiness import worker_init, equal
from development.aca_temporal_v1 import sampling
from development.aca_temporal_pair_v1.model import initialized as pair_initialized
from development.aca_spatial_v1 import augment

from . import formal_train as f
from .data import SEED, SpatialDataset25, SpatialCollator25, EpochStream25

FIXTURES = ('04', '22')
DIAGNOSTIC = f.CURRENT / 'input_diagnostic32_20261119_v1'
need = core.need


def _rejects(fn):
    try:
        fn()
    except (ValueError, KeyError):
        return True
    raise AssertionError('Expected rejection')


def fixture_batches(plan):
    ids = [json.loads((DIAGNOSTIC / x / 'evidence.json').read_text())['video_id'] for x in FIXTURES]
    ds = SpatialDataset25(plan['source_root'], 'train', plan['train_ids'], plan['reference'])
    ds.set_epoch(0)
    indices = [plan['train_ids'].index(vid) for vid in ids]
    serial_items = [ds[i] for i in indices]

    rng_before = core.capture_rng()
    loader = DataLoader(
        Subset(ds, indices),
        batch_size=1,
        num_workers=2,
        collate_fn=SpatialCollator25(),
        worker_init_fn=worker_init,
        multiprocessing_context='spawn',
        prefetch_factor=2,
        generator=torch.Generator().manual_seed(SEED + 100000),
        persistent_workers=False,
    )
    worker_batches = list(loader)
    need(equal(rng_before, core.capture_rng()), 'Spawn workers consumed model RNG')
    for item, batch in zip(serial_items, worker_batches):
        need(torch.equal(item['tensor'], batch['tensor'][0]), 'Worker tensor differs from serial')
        need(batch['targets'].tolist() == [item['binary_label']], 'Worker target differs')
        need(batch.get('spatial_policy_sha256') == augment.POLICY_SHA, 'Missing spatial policy SHA')
        need(batch.get('spatial_seed') == SEED, 'Wrong spatial seed')
    return serial_items, worker_batches, ids


def check_cpu(plan):
    f.configure()
    # 1. Model init seed25 determinism & distinction from seed24
    m25_a, ev25_a = pair_initialized('aca', 20261025)
    m25_b, ev25_b = pair_initialized('aca', 20261025)
    m24, _ = pair_initialized('aca', 20261024)
    sd_a, sd_b, sd_24 = m25_a.state_dict(), m25_b.state_dict(), m24.state_dict()
    need(all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a), 'Seed25 model init not deterministic')
    need(not torch.equal(sd_a['fc.weight'], sd_24['fc.weight']), 'Seed25 head equals seed24')
    need(any(not torch.equal(sd_a[k], sd_24[k]) for k in sd_a if '.aca.' in k), 'Seed25 ACA equals seed24')
    need(all(torch.equal(sd_a[k], sd_24[k]) for k in sd_a if '.aca.' not in k and not k.startswith('fc.')),
         'Pretrained ResNet50 backbone differs across seeds')
    need(ev25_a['head_initialization_seed'] == 20264025 and ev25_a['tsci_bn_count'] == 48 and
         ev25_a['tsci_bn_momentum'] == 0.01, 'Unexpected seed25 model init metadata')
    del m25_a, m25_b, m24

    # 2. Temporal & spatial sampling use seed25 and differ from seed24, without touching global RNG
    ids = [json.loads((DIAGNOSTIC / x / 'evidence.json').read_text())['video_id'] for x in FIXTURES]
    rng0 = core.capture_rng()
    idx25 = sampling.indices(146, SEED, 0, ids[0])
    idx24 = sampling.indices(146, 20261024, 0, ids[0])
    need(idx25 != idx24 and idx25[0] == 0 and idx25[-1] == 145 and len(idx25) == 50,
         'Seed25 temporal sampling invalid')
    sp25 = augment.parameters(SEED, 0, ids[0])
    sp24 = augment.parameters(20261024, 0, ids[0])
    need(sp25 != sp24, 'Seed25 spatial params identical to seed24')
    augment.validate_params(sp25)
    need(equal(rng0, core.capture_rng()), 'Sampling mutated global RNG')

    # 3. Read order uses seed25 (distinct from seed24)
    ord25 = torch.randperm(7192, generator=torch.Generator().manual_seed(SEED)).tolist()
    ord24 = torch.randperm(7192, generator=torch.Generator().manual_seed(20261024)).tolist()
    need(ord25 != ord24, 'Seed25 read order equals seed24')

    # 4. Production dataset + 2-worker spawn collator + EpochStream25 on 2 train clips
    serial_items, worker_batches, ids = fixture_batches(plan)
    tiny_plan = {**plan, 'train_ids': ids}
    tiny_ds = SpatialDataset25(plan['source_root'], 'train', ids, plan['reference'])
    state = {'epoch': 0, 'cursor': 0}
    stream = EpochStream25({'train': tiny_ds}, tiny_plan, lambda: state)
    try:
        tiny_ds.set_epoch(0)
        order = torch.randperm(2, generator=torch.Generator().manual_seed(SEED)).tolist()
        before = core.capture_rng()
        for i in order:
            b = stream('train', i)
            need(torch.equal(b['tensor'][0], serial_items[i]['tensor']), 'EpochStream25 tensor mismatch')
            state['cursor'] += 1
        need(equal(before, core.capture_rng()), 'EpochStream25 consumed global RNG')
    finally:
        stream.close()

    # 5. Reject wrong seed24 metadata & verify exact validation delegation (0 val decodes)
    bad = copy.deepcopy(serial_items[0])
    bad['metadata']['spatial_augmentation']['seed'] = 20261024
    _rejects(lambda: SpatialCollator25()([bad]))
    bad2 = copy.deepcopy(serial_items[0])
    bad2['metadata']['temporal_seed'] = 20261024
    _rejects(lambda: SpatialCollator25()([bad2]))

    val_ds = SpatialDataset25(plan['source_root'], 'validation', plan['validation_ids'][:1], plan['reference'])
    sentinel = object()
    for ep in (0, 29):
        val_ds.set_epoch(ep)
        with patch.object(RevisedDataset, '__getitem__', return_value=sentinel) as m_get:
            need(val_ds[0] is sentinel, 'Validation did not delegate to RevisedDataset')
            m_get.assert_called_once_with(0)
        with patch.object(BinaryCollator, '__call__', return_value=sentinel) as m_col:
            need(SpatialCollator25()([{'split': 'validation', 'metadata': {}}]) is sentinel,
                 'Validation did not delegate to BinaryCollator')
            m_col.assert_called_once()

    return {
        'seed': SEED,
        'model_init_seed25_deterministic_and_distinct_from_seed24': True,
        'head_initialization_seed': 20264025,
        'temporal_sampling_seed25_distinct_from_seed24': True,
        'spatial_sampling_seed25_distinct_from_seed24': True,
        'read_order_seed25_distinct_from_seed24': True,
        'worker_vs_serial_bitwise_equal': True,
        'epoch_stream_seed25_verified': True,
        'wrong_seed24_metadata_rejected': True,
        'validation_exact_frozen_delegation': True,
        'global_rng_unchanged': True,
        'train_video_ids': ids,
        'source_train_decodes': 6,
        'validation_video_decodes': 0,
        'test_video_decodes': 0,
        'optimizer_updates': 0,
    }


def check_gpu(plan):
    observed = f.assert_idle_gpu(plan['gpu_policy'])
    with f.shell.run_lock(f.CURRENT.parent / '00_admin', 'baseline8_gpu.lock'):
        f.assert_idle_gpu(plan['gpu_policy'])
        f.configure()
        _, batches, ids = fixture_batches(plan)

        # Engineering validation batches built cleanly from the same 2 train clips
        # (stripping train-only aug keys to satisfy validation-no-aug guard without opening real val videos).
        val_eng_batches = []
        for b in batches:
            vb = copy.deepcopy(b)
            vb.pop('temporal_sampling_version', None)
            vb.pop('spatial_policy_sha256', None)
            vb.pop('spatial_seed', None)
            for r in vb['records']:
                r['split'] = 'validation'
                r['metadata'].pop('spatial_augmentation', None)
            val_eng_batches.append(vb)

        # Probe 1: train-boundary continuous vs resumed (effective16 / tail8, pause at update 1)
        probe1 = copy.deepcopy(plan)
        probe1.update(
            train_ids=[ids[i % 2] for i in range(24)],
            validation_ids=ids,
            max_epochs=1,
            checkpoint_every=1,
        )

        def fetch1(split, index):
            need(split == 'train', 'Probe 1 must pause before validation')
            return batches[index % 2]

        # Probe 2: post-selection (best_model is not None) continuous vs resumed across epoch 1 -> epoch 2
        probe2 = copy.deepcopy(plan)
        probe2.update(
            train_ids=[ids[i % 2] for i in range(24)],
            validation_ids=ids,
            max_epochs=2,
            checkpoint_every=2,
        )
        # Map engineering validation_ids into probe2 reference with split='validation' for guard_fetch
        # (strictly in-memory engineering probe on the 2 train clips; 0 real validation videos opened).
        probe2['reference'] = copy.deepcopy(plan['reference'])

        # Prepare epoch 1 train batches (with epoch_index=1 metadata) so epoch 2 guard passes cleanly.
        ds_e1 = SpatialDataset25(plan['source_root'], 'train', plan['train_ids'], plan['reference'])
        ds_e1.set_epoch(1)
        batches_e1 = [SpatialCollator25()([ds_e1[plan['train_ids'].index(vid)]]) for vid in ids]

        torch.cuda.reset_peak_memory_stats()
        with tempfile.TemporaryDirectory(prefix='aca_spatial25_gpu_probe_', dir=f.QA) as temp:
            folder = Path(temp)

            def new_loop(probe_cfg, fetch_fn, path):
                f.configure()
                m, _ = f.initialized('aca', SEED)
                validate_momentum(m)
                m.cuda()
                return f.TrainingLoop25(m, probe_cfg, fetch_fn, path, 'engineering_only')

            # --- Probe 1: train boundary ---
            cont1 = new_loop(probe1, fetch1, folder / 'cont1.pt')
            frozen_bn = {
                n: b.detach().cpu().clone()
                for n, b in cont1.model.named_buffers()
                if '.aca.' not in n
            }
            need(cont1.run(lambda phase, state, seen: phase == 'validation') == 'paused',
                 'Probe 1 failed to pause before validation')
            exp1 = torch.load(folder / 'cont1.pt', map_location='cpu', weights_only=True)
            need(exp1['state']['updates'] == 2 and exp1['state']['cursor'] == 24, 'Probe 1 update count wrong')
            need(all(torch.equal(b, dict(cont1.model.named_buffers())[n].cpu())
                     for n, b in frozen_bn.items()), 'Frozen backbone BN changed during train')
            need(all(p.grad is None for p in cont1.model.parameters()), 'Gradients retained')
            del cont1
            gc.collect()
            torch.cuda.empty_cache()

            part1 = new_loop(probe1, fetch1, folder / 'res1.pt')
            need(part1.run(lambda phase, state, seen: state['updates'] == 1) == 'paused', 'Probe 1 pause failed')
            need(part1.committed['cursor'] == 16, 'Probe 1 not at effective16 boundary')
            del part1
            gc.collect()
            torch.cuda.empty_cache()

            res1 = new_loop(probe1, fetch1, folder / 'res1.pt')
            res1.resume()
            need(res1.run(lambda phase, state, seen: phase == 'validation') == 'paused', 'Probe 1 resume failed')
            act1 = torch.load(folder / 'res1.pt', map_location='cpu', weights_only=True)
            matches_train = {
                k: equal(exp1[k], act1[k])
                for k in ('model', 'optimizer', 'rng', 'cuda_rng', 'state', 'best_model')
            }
            need(all(matches_train.values()), f'Train-boundary resume mismatch: {matches_train}')
            del res1
            gc.collect()
            torch.cuda.empty_cache()

            # --- Probe 2: post-selection (best_model is not None) resume across epoch 1 -> epoch 2 ---
            def make_fetch2(loop_box):
                def fetch2(split, index):
                    if split == 'train':
                        ep = loop_box[0].state['epoch']
                        return (batches if ep == 0 else batches_e1)[index % 2]
                    return val_eng_batches[index % 2]
                return fetch2

            box_c = [None]
            cont2 = new_loop(probe2, make_fetch2(box_c), folder / 'cont2.pt')
            box_c[0] = cont2
            need(cont2.run(lambda phase, state, seen: False) == 'completed', 'Probe 2 continuous failed')
            exp2 = torch.load(folder / 'cont2.pt', map_location='cpu', weights_only=True)
            need(exp2['best_model'] is not None and exp2['state']['epoch'] == 2 and
                 exp2['state']['best_epoch'] in (1, 2), 'Probe 2 selection did not commit best_model')
            del cont2
            gc.collect()
            torch.cuda.empty_cache()

            box_p = [None]
            part2 = new_loop(probe2, make_fetch2(box_p), folder / 'res2.pt')
            box_p[0] = part2
            # Pause immediately once epoch 1 (including engineering validation & best_model commit) completes.
            need(part2.run(lambda phase, state, seen: phase == 'train' and state['epoch'] == 1) == 'paused',
                 'Probe 2 pause after epoch 1 selection failed')
            mid2 = torch.load(folder / 'res2.pt', map_location='cpu', weights_only=True)
            need(mid2['best_model'] is not None and mid2['state']['epoch'] == 1 and
                 mid2['state']['best_epoch'] == 1, 'Epoch 1 best_model not saved before resume')
            del part2
            gc.collect()
            torch.cuda.empty_cache()

            box_r = [None]
            res2 = new_loop(probe2, make_fetch2(box_r), folder / 'res2.pt')
            box_r[0] = res2
            res2.resume()
            need(res2.run(lambda phase, state, seen: False) == 'completed', 'Probe 2 resume failed')
            act2 = torch.load(folder / 'res2.pt', map_location='cpu', weights_only=True)
            matches_post_sel = {
                k: equal(exp2[k], act2[k])
                for k in ('model', 'optimizer', 'rng', 'cuda_rng', 'state', 'best_model')
            }
            need(all(matches_post_sel.values()), f'Post-selection resume mismatch: {matches_post_sel}')

            # Verify eval forward mutates zero buffers
            res2.model.eval()
            all_bn_before = {n: b.detach().clone() for n, b in res2.model.named_buffers()}
            with torch.no_grad():
                for b in val_eng_batches:
                    core.summed_ce(res2.model(b['tensor'].cuda()), b['targets'].cuda())
            need(all(torch.equal(b, dict(res2.model.named_buffers())[n]) for n, b in all_bn_before.items()),
                 'Eval forward mutated buffers')
            peak = {
                'allocated_bytes': torch.cuda.max_memory_allocated(),
                'reserved_bytes': torch.cuda.max_memory_reserved(),
            }
            del res2
            gc.collect()
            torch.cuda.empty_cache()

        need(not folder.exists(), 'Temporary engineering directory still exists')
        return {
            'seed': SEED,
            'gpu_processes': observed,
            'train_video_ids': ids,
            'effective': 16,
            'tail': 8,
            'microbatch': 1,
            'input_shape': [1, 3, 50, 120, 160],
            'train_boundary_continuous_vs_resumed_equal': matches_train,
            'post_selection_continuous_vs_resumed_equal': matches_post_sel,
            'post_selection_best_model_non_null_verified': True,
            'backbone_frozen_bn_unchanged': True,
            'eval_all_buffers_unchanged': True,
            'tsci_bn_momentum': 0.01,
            'temporary_checkpoints_removed': True,
            'source_train_decodes': 6,
            'validation_video_decodes': 0,
            'test_video_decodes': 0,
            'accuracy_reported': False,
            'peak_memory': peak,
            'formal_weights_from_engineering': False,
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['cpu', 'gpu'])
    a = p.parse_args()
    f.QA.mkdir(exist_ok=True)
    output = f.QA / f'{a.phase}_report.json'
    need(not output.exists(), f'Refusing to overwrite existing {output}')
    t0 = time.monotonic()
    report = {'status': 'running', 'phase': a.phase, 'at_taiwan': runtime.now()}
    try:
        plan = f.draft_plan()
        report['draft_plan_sha256'] = core.plan_digest(plan)
        if a.phase == 'gpu':
            cpu_ev = f.read(f.QA / 'cpu_report.json')
            need(cpu_ev['status'] == 'passed' and cpu_ev['draft_plan_sha256'] == report['draft_plan_sha256'],
                 'CPU gate missing or mismatched')
        report.update(check_cpu(plan) if a.phase == 'cpu' else check_gpu(plan))
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
