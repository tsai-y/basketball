"""Explicit TANet formal start/resume entry; --dry-run is metadata-only."""
import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import torch
from torch.utils.data import DataLoader

from .config import FOLDER, InputConfig, need, sha256
from .data import BasketballTANetDataset
from .training import (atomic_save, checkpoint, collate, configure_runtime, contract, eval_batch,
    initialized_model, load_checkpoint, metrics, optimizer_for, stopping, train_group)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--start', action='store_true', help='Explicit full train/validation start authorization')
    parser.add_argument('--readiness-report', type=Path)
    parser.add_argument('--resume', type=Path, help='Resume last.pt at an epoch boundary in its original run')
    args = parser.parse_args(argv)
    if args.start and args.readiness_report is None:
        parser.error('--start requires --readiness-report')
    if args.resume is not None and not args.start:
        parser.error('--resume requires --start')
    return args


def readiness_gate(path, plan):
    path = Path(path).resolve()
    need(path.is_relative_to((FOLDER/'artifacts').resolve()) and path.name == 'report.json', 'Foreign readiness report')
    report = json.loads(path.read_text())
    need(report['status'] == 'GPU_readiness_passed_formal_start_pending', 'GPU readiness not passed')
    need(report['contract_sha256'] == plan['contract_sha256'], 'Readiness source/data/recipe changed; rerun probe')
    need(report['environment']['torch'] == str(torch.__version__) and
         report['environment']['cuda'] == torch.version.cuda, 'Readiness runtime changed; rerun probe')
    need(all(report['checks'].values()), 'Incomplete readiness checks')
    return report


def make_loader(dataset, cfg, *, epoch=None):
    return DataLoader(dataset, batch_size=cfg['micro_batch'], num_workers=0, drop_last=False,
        shuffle=epoch is not None,
        generator=torch.Generator().manual_seed(cfg['seed']+(epoch or 0)), collate_fn=collate)


def run_training(args, plan):
    cfg = plan['recipe']
    passed = readiness_gate(args.readiness_report, plan)
    device = configure_runtime(cfg)
    need(passed['environment']['gpu'] == torch.cuda.get_device_name(), 'GPU differs from readiness')
    parent = FOLDER/'artifacts'
    if args.resume is None:
        output = Path(tempfile.mkdtemp(prefix='formal_train_', dir=parent))
        atomic_save(output/'contract.json', plan, json_format=True)
    else:
        path = args.resume.resolve()
        need(path.name == 'last.pt' and path.parent.name.startswith('formal_train_') and
             path.parent.is_relative_to(parent.resolve()) and not args.resume.is_symlink(), 'Resume only a formal run last.pt')
        output = path.parent
        need(json.loads((output/'contract.json').read_text()) == plan, 'Run contract changed')
        need((output/'best.pt').is_file(), 'Resume requires preserved best.pt')
    lock = output/'active.lock'
    with lock.open('x') as stream:
        stream.write('TANet formal process active\n')
    try:
        model = initialized_model(cfg, device)
        optimizer = optimizer_for(model, cfg)
        start_epoch, global_step, history = 1, 0, []
        stop = {'best_ce': float('inf'), 'early_anchor': float('inf'), 'wait': 0}
        if args.resume is not None:
            saved = load_checkpoint(args.resume, model, optimizer, contract_sha256=plan['contract_sha256'], kind='formal')
            start_epoch, global_step = saved['epoch']+1, saved['global_step']
            stop, history = saved['stop_state'], saved['history']
            need(start_epoch <= cfg['max_epochs'] and stop['wait'] < cfg['patience'], 'Completed/early-stopped recipe cannot silently extend')
        train = BasketballTANetDataset(split='train', config=InputConfig())
        validation = BasketballTANetDataset(split='validation', config=InputConfig())
        expected_train = {r['video_id'] for r in train.scope.rows}
        expected_val = [r['video_id'] for r in validation.scope.rows]
        with (output/'progress.log').open('a', buffering=1) as logfile:
            def log(message):
                print(message, flush=True)
                logfile.write(message+'\n')
            log('OUTPUT: '+str(output))
            atomic_save(output/'run_state.json', {'status': 'running', 'next_epoch': start_epoch,
                'formal_training_started': True, 'test_used': False}, json_format=True)
            for epoch in range(start_epoch, cfg['max_epochs']+1):
                tick = time.monotonic()
                loader = make_loader(train, cfg, epoch=epoch)
                iterator = iter(loader)
                observed, seen, epoch_updates, total_loss = set(), 0, 0, 0.
                while seen < len(train):
                    count = min(cfg['effective_batch'], len(train)-seen)
                    def group_batches():
                        remaining = count
                        while remaining:
                            batch = next(iterator)
                            ids = batch['video_id']
                            need(observed.isdisjoint(ids) and len(set(ids)) == len(ids), 'Duplicate training ID')
                            observed.update(ids)
                            remaining -= len(ids)
                            need(remaining >= 0, 'Unexpected group boundary')
                            yield batch
                    result = train_group(model, optimizer, group_batches(), device=device,
                                         group_videos=count, cfg=cfg)
                    total_loss += result['CE']*count
                    previous = seen
                    seen += count
                    global_step += 1
                    epoch_updates += 1
                    if previous == 0 or seen//cfg['progress_every_videos'] != previous//cfg['progress_every_videos'] or seen == len(train):
                        log(f'epoch {epoch}/{cfg["max_epochs"]}: train={seen}/{len(train)} updates={global_step} CE={total_loss/seen:.6f}')
                need(observed == expected_train and epoch_updates == math.ceil(len(train)/cfg['effective_batch']), 'Incomplete train pass')
                outputs, labels, ids = [], [], []
                for batch in make_loader(validation, cfg):
                    out, y = eval_batch(model, batch, device)
                    outputs.append(out)
                    labels.append(y)
                    ids.extend(batch['video_id'])
                    if len(ids) % cfg['progress_every_videos'] == 0 or len(ids) == len(validation):
                        log(f'epoch {epoch}: validation={len(ids)}/{len(validation)}')
                need(ids == expected_val, 'Incomplete/reordered validation pass')
                logits, y = torch.cat(outputs), torch.cat(labels)
                scores = metrics(logits, y)
                stop, selected, finished = stopping(scores['CE'], stop, cfg)
                entry = {'epoch': epoch, 'train_CE': total_loss/len(train), 'validation': scores,
                         'seconds': time.monotonic()-tick, 'global_step': global_step,
                         'best_selected': selected, 'early_wait': stop['wait']}
                history.append(entry)
                payload = checkpoint(model, optimizer, contract_sha256=plan['contract_sha256'], kind='formal',
                    epoch=epoch, global_step=global_step, stop_state=stop, history=history)
                payload['validation'] = {'metrics': scores, 'logits': logits, 'labels': y, 'video_ids': ids}
                if selected:
                    atomic_save(output/'best.pt', payload)
                atomic_save(output/'last.pt', payload)
                del payload
                atomic_save(output/'history.json', history, json_format=True)
                atomic_save(output/'run_state.json', {'status': 'completed' if finished or epoch == cfg['max_epochs'] else 'running',
                    'epoch': epoch, 'global_step': global_step, 'early_wait': stop['wait'],
                    'best_validation_CE': stop['best_ce'], 'formal_training_started': True, 'test_used': False}, json_format=True)
                log(json.dumps(entry, ensure_ascii=False))
                if finished:
                    log('Early stopping: fixed validation-CE rule reached.')
                    break
        return output
    except BaseException as exc:
        atomic_save(output/'failure.json', {'status': 'interrupted_or_failed', 'error': str(exc),
                    'resume': 'Use preserved last.pt to restart its next epoch; an incomplete epoch is replayed.'}, json_format=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    plan, _, _ = contract()
    if args.dry_run:
        result = {'status': 'metadata_plan_ready_formal_start_pending', 'plan': plan,
                  'video_opens': 0, 'cuda_initialized': torch.cuda.is_initialized(),
                  'optimizer_steps': 0, 'output_files_written': False, 'formal_training_started': False}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return result
    return run_training(args, plan)


if __name__ == '__main__':
    main()
