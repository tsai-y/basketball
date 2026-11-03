"""Exactly two train-video GPU engineering checks; no formal/validation/test run."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch
from torch import nn

from .config import FOLDER, InputConfig, need, sha256
from .data import decode
from .prepare import select_smoke_ids
from .training import (atomic_save, checkpoint, collate, configure_runtime, contract, eval_batch,
    initialized_model, load_checkpoint, optimizer_for, state_digest, train_group)


def bn2d_buffers(model):
    return {f'{name}.{key}': value.detach().cpu().clone()
            for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)
            for key, value in module.named_buffers(recurse=False)}


def tensor_sha(x):
    return state_digest(x)


def execute():
    started = time.monotonic()
    plan, scope, _ = contract()
    cfg = plan['recipe']
    device = configure_runtime(cfg)
    free, total = torch.cuda.mem_get_info()
    need(free >= 12*1024**3, 'Need at least12GiB free for bounded GPU probe; do not interrupt other jobs')
    parent = FOLDER/'artifacts'
    output = Path(tempfile.mkdtemp(prefix='gpu_readiness_', dir=parent))
    atomic_save(output/'plan.json', {**plan, 'train_ids': select_smoke_ids(scope),
        'engineering_updates_executed': 3, 'logical_updates_with_resume_replay': 2,
        'microsteps_per_update': cfg['effective_batch'], 'formal_training_started': False}, json_format=True)
    atomic_save(output/'run_state.json', {'status': 'running', 'formal_training_started': False}, json_format=True)
    print('OUTPUT:', output, flush=True)
    try:
        samples, decode_seconds = [], []
        for video_id in select_smoke_ids(scope):
            tick = time.monotonic()
            sample = decode(scope, video_id, InputConfig())
            decode_seconds.append(time.monotonic()-tick)
            samples.append(sample)
        batches = [collate([s]) for s in samples]
        model = initialized_model(cfg, device)
        optimizer = optimizer_for(model, cfg)
        initial_hash = state_digest(model.state_dict())
        frozen = bn2d_buffers(model)
        probes = {name: tensor_sha(p) for name, p in model.named_parameters()
                  if name in ('backbone.conv1.weight', 'classifier.weight', 'backbone.layer1.0.tam.L.0.weight')}
        torch.cuda.reset_peak_memory_stats()
        def perform(label):
            print(label+' begin (16 accumulated microsteps)', flush=True)
            def progress(seen, total):
                if seen % 4 == 0:
                    print(f'{label}: {seen}/{total}', flush=True)
            result = train_group(model, optimizer, [batches[i % 2] for i in range(cfg['effective_batch'])],
                device=device, group_videos=cfg['effective_batch'], cfg=cfg, progress=progress)
            print(label+' passed CE='+str(result['CE']), flush=True)
            return result
        step1 = perform('step1')
        need(state_digest(bn2d_buffers(model)) == state_digest(frozen), 'Pretrained BN2d buffers changed')
        need(all(tensor_sha(dict(model.named_parameters())[k]) != h for k, h in probes.items()),
             'Backbone/TAM/head did not all update')
        payload = checkpoint(model, optimizer, contract_sha256=plan['contract_sha256'], kind='engineering_only',
                             epoch=0, global_step=1, stop_state={}, history=[])
        ckpt = output/'engineering_step1.pt'
        atomic_save(ckpt, payload)
        del payload
        reference = perform('continuous_step2')
        expected_model = state_digest(model.state_dict())
        expected_optimizer = state_digest(optimizer.state_dict())
        need(state_digest(bn2d_buffers(model)) == state_digest(frozen), 'BN2d changed in continuous branch')
        peak_allocated = torch.cuda.max_memory_allocated()/1024**3
        peak_reserved = torch.cuda.max_memory_reserved()/1024**3
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        model = initialized_model(cfg, device)
        optimizer = optimizer_for(model, cfg)
        loaded = load_checkpoint(ckpt, model, optimizer, contract_sha256=plan['contract_sha256'], kind='engineering_only')
        need(loaded['global_step'] == 1, 'Wrong resume step')
        del loaded
        resumed = perform('resumed_step2')
        need(resumed['last_logits'] == reference['last_logits'] and resumed['CE'] == reference['CE'],
             'Resumed logits/loss differ from uninterrupted branch')
        need(state_digest(model.state_dict()) == expected_model, 'Resumed model/BN state differs')
        need(state_digest(optimizer.state_dict()) == expected_optimizer, 'Resumed Adam state differs')
        need(state_digest(bn2d_buffers(model)) == state_digest(frozen), 'Resumed BN2d changed')
        before_eval = state_digest(model.state_dict())
        eval_seconds = []
        for batch in batches:
            torch.cuda.synchronize()
            tick = time.monotonic()
            out, y = eval_batch(model, batch, device)
            torch.cuda.synchronize()
            eval_seconds.append(time.monotonic()-tick)
            need(out.shape == (1, 2) and y.dtype == torch.int64, 'Eval interface mismatch')
        need(state_digest(model.state_dict()) == before_eval, 'Eval changed parameters/buffers')
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated()/1024**3)
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved()/1024**3)
        need(contract()[0]['contract_sha256'] == plan['contract_sha256'], 'Source/data/recipe changed during probe')
        # Inputs are cached for GPU timing. Sequential decode is estimated
        # separately from exactly two samples, never called full-epoch evidence.
        micro = statistics.median(step1['microstep_seconds'][1:]+reference['microstep_seconds'])
        io = statistics.mean(decode_seconds)
        forward = statistics.mean(eval_seconds)
        estimated = plan['train_count']*(micro+io) + plan['validation_count']*(forward+io)
        estimated += plan['optimizer_steps_per_epoch']*reference['optimizer_seconds']
        report = {'status': 'GPU_readiness_passed_formal_start_pending', 'contract_sha256': plan['contract_sha256'],
            'recipe': cfg, 'source_sha256': plan['source_sha256'], 'data_identity': plan['data_identity'],
            'environment': {'gpu': torch.cuda.get_device_name(), 'total_GiB': total/1024**3,
                'free_GiB_before': free/1024**3, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(), 'precision': cfg['precision'],
                'deterministic_algorithms': True, 'TF32': False},
            'checks': {'full_input_and_output_shapes': True, 'finite_loss_gradients_updates': True,
                'backbone_TAM_head_updated': True, 'pretrained_BN2d_buffers_bitwise_unchanged': True,
                'checkpoint_resume_logits_loss_bitwise_equal': True, 'checkpoint_resume_model_bitwise_equal': True,
                'checkpoint_resume_Adam_state_bitwise_equal': True, 'eval_all_model_state_unchanged': True,
                'source_data_recipe_unchanged': True},
            'samples': [{k: v for k, v in s.items() if k != 'tensor'} for s in samples],
            'input_shape': [1, 3, 50, 192, 256], 'output_shape': [1, 2],
            'original_video_opens': {'train': 2, 'validation': 0, 'test': 0},
            'initial_model_sha256': initial_hash, 'continuous_step2_model_sha256': expected_model,
            'continuous_step2_optimizer_sha256': expected_optimizer,
            'engineering_checkpoint': {'path': ckpt.name, 'sha256': sha256(ckpt), 'formal_initialization_allowed': False},
            'steps': {'step1': step1, 'continuous_step2': reference, 'resumed_step2': resumed},
            'optimizer_updates_executed': 3, 'microsteps_executed': 3*cfg['effective_batch'],
            'logical_updates': 2, 'classification_accuracy_computed': False,
            'peak_allocated_GiB': peak_allocated, 'peak_reserved_GiB': peak_reserved,
            'timing': {'decode_seconds_two_samples': decode_seconds, 'median_train_microstep_seconds': micro,
                'eval_seconds_two_samples': eval_seconds, 'estimated_epoch_seconds': estimated,
                'estimated_15_epochs_hours': estimated*cfg['max_epochs']/3600,
                'scope': 'Engineering extrapolation from two cached train videos, excludes full data I/O variability, checkpoint/validation aggregation and other workloads; not a complete epoch measurement'},
            'formal_training_started': False, 'feature_cache_used_or_modified': False,
            'temporal_selector_integrated': False, 'elapsed_seconds': time.monotonic()-started}
        atomic_save(output/'report.json', report, json_format=True)
        atomic_save(output/'run_state.json', {'status': 'completed', 'formal_training_started': False}, json_format=True)
        print(json.dumps({k: report[k] for k in ['status', 'peak_allocated_GiB', 'peak_reserved_GiB', 'timing']}, indent=2), flush=True)
        return output
    except BaseException as exc:
        atomic_save(output/'run_state.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                    'formal_training_started': False}, json_format=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True, help='Explicitly perform bounded engineering updates')
    parser.parse_args()
    print(execute())
