"""Bounded engineering readiness; never calls the formal-training entry."""
import copy
import gc
import hashlib
import json
from pathlib import Path
import signal
import shutil
import tempfile
from unittest.mock import patch

import torch
from torch import nn
from development.aca_candidate_v1.experiment import initialized_candidate
from development.aca_net_v1 import train as base
from development.aca_net_v1.baseline_preflight import check_gradients, check_state, frozen_snapshot, training_mode
from development.aca_net_v1.data import VideoScope, guarded_source_sha256, require, select_smoke_ids
from development.aca_net_v1.readiness import ToyDataset, nested_equal, toy_model
from development.aca_validation_v1.analyze import tensor_digest
from . import engine, experiment as exp


def tree_digest(value):
    digest = hashlib.sha256()
    def walk(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(str((str(item.dtype), tuple(item.shape))).encode())
            digest.update(item.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); walk(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item: walk(child)
        else:
            digest.update(repr(item).encode())
    walk(value)
    return digest.hexdigest()


def synthetic_resume(output, device='cpu'):
    cfg = exp.HighResConfig(epochs=2, patience=2, workers=0, checkpoint_every=1)
    plan = {'config': vars(cfg), 'train_count': 19, 'validation_count': 5, 'engineering_only_synthetic_toy': True}
    train, validation = ToyDataset(19), ToyDataset(5)
    continuous, interrupted = output / 'continuous', output / 'interrupted'
    continuous.mkdir(); interrupted.mkdir()
    base.configure_runtime(cfg.seed, cuda=device == 'cuda')
    model = toy_model(device); optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    engine.fit(model, optimizer, train, validation, cfg, plan, continuous, base.initial_progress(), device=device)
    expected = torch.load(continuous / 'last.pt', map_location='cpu', weights_only=True)
    base.configure_runtime(cfg.seed, cuda=device == 'cuda')
    model = toy_model(device); optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    step, calls = optimizer.step, 0
    def interrupted_step(*args, **kwargs):
        nonlocal calls
        result = step(*args, **kwargs); calls += 1
        if calls == 1: signal.raise_signal(signal.SIGINT)
        return result
    with patch.object(optimizer, 'step', side_effect=interrupted_step):
        stopped = engine.fit(model, optimizer, train, validation, cfg, plan, interrupted, base.initial_progress(), device=device)
    require(stopped['status'] == 'stopped_at_optimizer_boundary' and stopped['cursor'] == 16, 'Bad interrupt boundary')
    model = toy_model(device); optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    progress = engine.load_checkpoint(interrupted / 'last.pt', model, optimizer, plan)
    engine.fit(model, optimizer, train, validation, cfg, plan, interrupted, progress, device=device)
    actual = torch.load(interrupted / 'last.pt', map_location='cpu', weights_only=True)
    require(all(nested_equal(expected[k], actual[k]) for k in ('model', 'optimizer', 'rng', 'progress')),
            'High-resolution engine resume trajectory mismatch')
    require(actual['progress']['optimizer_steps'] == 4 and all(r['train']['count'] == 19 for r in actual['progress']['history']),
            'Partial effective batch omitted')
    return {'bitwise_equal': True, 'device': device, 'last_effective_batch': 3,
            'synthetic_optimizer_updates_executed': 8, 'formal_training_started': False}


def run(source_root, preparation):
    preparation = Path(preparation).resolve()
    require(preparation.parent == exp.artifact_parent().resolve() and preparation.name.startswith('cpu_prepare_'), 'Wrong CPU preparation family')
    cpu = json.loads((preparation / 'report.json').read_text())
    require(cpu['status'] == 'cpu_prepared_gpu_readiness_pending' and cpu['formal_training_started'] is False, 'Bad CPU preparation')
    plan = exp.build_plan(source_root)
    require(cpu['plan_sha256'] == base.canonical_hash(plan) == base.canonical_hash(json.loads((preparation / 'plan.json').read_text())),
            'CPU preparation stale; do not bypass fingerprint check')
    exp.gpu_idle()
    require(shutil.disk_usage(exp.FOLDER).free >= 8 * 1024**3, 'Insufficient free disk')
    require(torch.cuda.is_available(), 'Host CUDA required')
    free, _ = torch.cuda.mem_get_info()
    require(free >= 11 * 1024**3, 'Less than 11 GiB free GPU memory')
    output = Path(tempfile.mkdtemp(prefix='gpu_readiness_', dir=exp.artifact_parent()))
    print(f'ENGINEERING_READINESS: {output}; NOT formal training', flush=True)
    base.atomic_json(output / 'plan.json', plan)
    exp.snapshot(output, plan)
    try:
        toy = synthetic_resume(output, 'cuda')
        base.configure_runtime(exp.config_from().seed)
        scope = VideoScope(source_root=source_root, split='train')
        ids = select_smoke_ids(scope)
        hashes = {i: guarded_source_sha256(scope, i) for i in ids}
        data = exp.dataset(source_root, 'train', ids)
        items = list(engine.checked_loader(base.make_loader(data, list(range(4)), exp.config_from(), 0), plan, 'train'))
        require([b['video_id'][0] for b in items] == ids, 'Production loader order mismatch')
        # Existing cost cache contains inputs only, never model weights.
        cache_plan = json.loads((exp.PREPARATION / 'plan.json').read_text())
        cache_path = exp.PREPARATION / 'engineering_inputs.pt'
        exp.verify_files({str(cache_path): cache_plan['engineering_inputs_sha256']})
        cache = torch.load(cache_path, map_location='cpu', weights_only=True)
        require(cache['train_ids'] == ids and cache_plan['source_video_sha256'] == hashes, 'Train source identity mismatch')
        require(all(torch.equal(item['tensor'][0], cached['256x192'])
                    for item, cached in zip(items, cache['inputs'])), 'Formal loader differs from original high-resolution cost inputs')
        del cache
        model, initialization = initialized_candidate(exp.config_from().seed)
        require(tensor_digest(model.state_dict()) == cpu['initial_model_state_sha256'], 'Fresh initialization mismatch')
        frozen = frozen_snapshot(model)
        model = model.cuda(); training_mode(model)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        metrics = base.Metrics()
        optimizer.zero_grad(set_to_none=True)
        for index in range(16):
            item = items[index % 4]
            x, y = item['tensor'].cuda(), item['binary_label'].cuda()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y, reduction='sum')
            require(torch.isfinite(loss).item(), 'Nonfinite engineering loss')
            (loss / 16).backward()
            metrics.add(logits, y, loss.item(), item['original_class'])
        gradients = check_gradients(model)
        branches = {branch: sum(p.grad.abs().sum().item() for name, p in model.named_parameters()
                               if f'.{branch}.' in name and p.grad is not None)
                    for branch in ('gate_value', 'gate_control', 'gate_output', 'short_term', 'kernel_generator', 'tsci')}
        require(all(v > 0 for v in branches.values()), 'Missing ACA gradient branch')
        require(tensor_digest(model.state_dict()) == cpu['initial_model_state_sha256'], 'Forward/backward changed model or BN')
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()  # Exactly ONE discarded engineering update, not formal training.
        engine.check_adam(optimizer, 1)
        check_state(model, frozen)
        peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        optimizer.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad(): expected = model(items[0]['tensor'].cuda()).cpu()
        progress = base.initial_progress()
        progress.update(cursor=16, optimizer_steps=1, train_metrics=metrics.state)
        engineering_plan = {**plan, 'engineering_only_real_probe': True, 'probe_kind': 'real_highres_engineering_not_formal'}
        checkpoint = output / 'engineering_only.pt'
        engine.save_checkpoint(checkpoint, model, optimizer, progress, engineering_plan)
        saved_rng = base.capture_rng()
        model_digest, adam_digest = tensor_digest(model.state_dict()), tree_digest(optimizer.state_dict())
        del model, optimizer, logits, x, y, loss
        gc.collect(); torch.cuda.empty_cache()
        model, _ = initialized_candidate(exp.config_from().seed)
        model = model.cuda()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        # Formal schema refuses this engineering checkpoint before model mutation.
        try:
            engine.load_checkpoint(checkpoint, model, optimizer, plan)
        except ValueError:
            pass
        else:
            raise ValueError('Engineering checkpoint incorrectly accepted as formal')
        restored = engine.load_checkpoint(checkpoint, model, optimizer, engineering_plan)
        require(restored == progress and nested_equal(saved_rng, base.capture_rng())
                and tensor_digest(model.state_dict()) == model_digest and tree_digest(optimizer.state_dict()) == adam_digest,
                'Real model/Adam/RNG/progress roundtrip mismatch')
        model.eval()
        with torch.no_grad(): actual = model(items[0]['tensor'].cuda()).cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        val_ids = select_smoke_ids(VideoScope(source_root=source_root, split='validation'))
        val = exp.dataset(source_root, 'validation', val_ids)
        count = 0
        with torch.no_grad():
            for batch in engine.checked_loader(base.make_loader(val, list(range(4)), exp.config_from(), 0), plan, 'validation'):
                logits = model(batch['tensor'].cuda())
                require(tuple(logits.shape) == (1, 2) and torch.isfinite(logits).all().item(), 'Validation engineering forward failed')
                count += 1
        require(count == 4 and tensor_digest(model.state_dict()) == model_digest, 'Validation loader/state mismatch')
        check_state(model, frozen)
        del model, optimizer, logits, frozen
        gc.collect(); torch.cuda.empty_cache()
        require(all(guarded_source_sha256(scope, i) == h for i, h in hashes.items()), 'Source video changed')
        exp.verify_plan_sources(plan)
        require(base.canonical_hash(exp.build_plan(source_root)) == base.canonical_hash(plan), 'Plan changed during readiness')
        report = {'schema': exp.PREFLIGHT_SCHEMA, 'status': 'passed', 'plan_sha256': base.canonical_hash(plan),
                  'engineering_only': True, 'formal_training_started': False,
                  'engineering_checkpoint_may_initialize_training': False, 'real_optimizer_updates': 1,
                  'engineering_effective_batch': 16, 'unique_train_videos_repeated_four_times': 4,
                  'synthetic_cuda_resume_bitwise_equal': toy['bitwise_equal'], 'synthetic_resume': toy,
                  'real_checkpoint_roundtrip_bitwise_equal': True, 'real_model_Adam_RNG_progress_preserved': True,
                  'engineering_as_formal_rejected': True, 'all_BN_unchanged': True,
                  'loader_workers_2_verified': True, 'input_tensor_parity_with_prior_highres_probe': True,
                  'decoded_train_videos': 4, 'decoded_validation_videos': 4, 'test_videos_opened': 0,
                  'validation_accuracy_computed': False, 'gradients': gradients, 'branch_gradients': branches,
                  'peak_allocated_bytes': peak, 'peak_reserved_bytes': reserved,
                  'initialization': initialization, 'ready_for_separately_authorized_formal_start': True,
                  'automatic_start': False, 'protected_history_unchanged': True}
        base.atomic_json(output / 'report.json', report)
        return {'output': str(output), **report}
    except BaseException as exc:
        base.atomic_json(output / 'failure.json', {'status': 'failed_no_automatic_formal_launch', 'error': repr(exc),
                                                  'formal_training_started': False, 'test_videos_opened': 0})
        raise
