"""Isolated binary checkpoint runtime adapted from the audited eight-class runtime.

Numerical epoch/selection loop remains FixtureLoop._run, without copying or
changing its loss, accumulation, clipping, selection or patience rules.
"""
import copy
from datetime import datetime
import gc
import math
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo
import json

import torch
from torch.utils.data import DataLoader, Subset, Dataset

from . import core
from development.aca_eight_class_data_v1.adapter import CLASSES, CLASS_TO_BINARY, EightClassDataset, collate_eight_class
from development.aca_eight_class_data_v1.foundation import CONTRACT_SHA256, baseline
from development.aca_eight_class_data_v1.gpu_readiness import cpu_tree, equal, tensor_sha, worker_init

SCHEMA = 'aca_direct_binary_formal_checkpoint_v1'
need = core.need
from .core import restore_metrics, metrics_state, summed_ce, finite_tree, BinaryMetrics, training_mode


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat()


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class EpochStream:
    """Lazy, epoch-aware worker stream. Recreate validation from zero on resume."""
    def __init__(self, datasets, plan, state_getter, *, collate=collate_eight_class, workers=2):
        self.datasets, self.plan, self.state_getter = datasets, plan, state_getter
        self.collate, self.workers = collate, workers
        self.iterator, self.key, self.order, self.position = None, None, [], 0
        self.observed = 0

    def __call__(self, split, index):
        state = self.state_getter()
        key = (state['epoch'], split)
        if key != self.key:
            need(self.iterator is None, 'Unexhausted prior stream')
            n = len(self.datasets[split])
            self.order = (torch.randperm(n, generator=torch.Generator().manual_seed(
                self.plan['seed'] + state['epoch'])).tolist()[state['cursor']:]
                if split == 'train' else list(range(n)))
            before = core.capture_rng()
            loader = DataLoader(Subset(self.datasets[split], self.order), batch_size=1, shuffle=False,
                                num_workers=self.workers, collate_fn=self.collate, worker_init_fn=worker_init,
                                generator=torch.Generator().manual_seed(self.plan['seed'] + state['epoch'] + 100000),
                                persistent_workers=False, pin_memory=False, drop_last=False,
                                **({'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if self.workers else {}))
            self.iterator = iter(loader)
            need(equal(before, core.capture_rng()), 'Loader consumed model RNG')
            self.key, self.position = key, 0
        need(self.position < len(self.order) and self.order[self.position] == index, 'Epoch/cursor order mismatch')
        batch = next(self.iterator)
        self.position += 1
        self.observed += 1
        if self.position == len(self.order):
            try:
                next(self.iterator)
            except StopIteration:
                self.iterator = None
            else:
                raise ValueError('Unexpected extra loader batch')
        return batch

    def close(self):
        self.iterator = None
        gc.collect()


def validate_state(state, plan, best):
    n, b = len(plan['train_ids']), plan['effective']
    epoch, cursor, updates = (state[k] for k in ('epoch', 'cursor', 'updates'))
    need(all(type(x) is int for x in (epoch, cursor, updates, state['bad_epochs'])) and
         0 <= epoch <= plan['max_epochs'] and 0 <= cursor <= n and
         (cursor % b == 0 or cursor == n) and updates == epoch * math.ceil(n / b) + math.ceil(cursor / b),
         'Invalid optimizer boundary')
    need(epoch < plan['max_epochs'] or cursor == 0, 'Cursor beyond final epoch')
    metric = core.restore_metrics(state['train_metrics'])
    need(metric._count == cursor and len(state['history']) == epoch, 'History/cursor mismatch')
    best_ce, best_epoch, bad = None, None, 0
    for i, row in enumerate(state['history']):
        need(type(row['epoch']) is int and row['epoch'] == i + 1, 'History epoch mismatch')
        for split, count in (('train', n), ('validation', len(plan['validation_ids']))):
            m = core.restore_metrics(row[split + '_state'])
            need(m._count == count and m.result() == row[split], 'Incomplete/inconsistent history')
        ce = row['validation']['ce']
        if best_ce is None or ce < best_ce:
            best_ce, best_epoch, bad = ce, i + 1, 0
        else:
            bad += 1
        need(bad < plan['patience'] or i == epoch - 1, 'History beyond early stop')
    need((state['best_ce'], state['best_epoch'], state['bad_epochs']) == (best_ce, best_epoch, bad),
         'Best selection mismatch')
    need((best is None) == (best_epoch is None) and (bad < plan['patience'] or cursor == 0), 'Best/terminal state mismatch')


class TrainingLoop(core.FixtureLoop):
    """General device adapter; the CLI validates the complete formal plan first.

    Unit fixtures may supply tiny models/data directly; there is no fixture option
    in the formal CLI. Every checkpoint is bound to both plan hash and run ID.
    """
    def __init__(self, model, plan, fetch, checkpoint, run_id, *, publish=None):
        need(plan.get('task') == 'basketball51_direct_binary' and plan.get('class_order') == ['Miss','Make'], 'Direct binary plan required')
        self.device = next(model.parameters()).device
        need(self.device.type in ('cpu', 'cuda') and all(p.dtype == torch.float32 and p.device == self.device
                                                       for p in model.parameters()), 'Need single FP32 device')
        need(bool(run_id), 'Missing run ID')
        self.run_id, self.publish = run_id, publish
        self.model, self.plan, self.fetch, self.checkpoint = model, copy.deepcopy(plan), fetch, Path(checkpoint)
        core.training_mode(model)
        self.optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                         lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0.,
                                         amsgrad=False, foreach=False, fused=False)
        self.state = {'epoch': 0, 'cursor': 0, 'updates': 0, 'train_metrics': core.metrics_state(core.BinaryMetrics()),
                      'history': [], 'best_epoch': None, 'best_ce': None, 'bad_epochs': 0}
        self.best_model, self.valid, self.started = None, True, False
        self.committed = copy.deepcopy(self.state)

    def _batch(self, split, index):
        x, y = super()._batch(split, index)
        return x.to(self.device), y.to(self.device)

    def _payload(self, state=None, best=None):
        return {'schema': SCHEMA, 'run_id': self.run_id, 'plan_sha256': core.plan_digest(self.plan),
                'plan': copy.deepcopy(self.plan), 'model': cpu_tree(self.model.state_dict()),
                'optimizer': cpu_tree(self.optimizer.state_dict()), 'rng': core.capture_rng(),
                'cuda_rng': torch.cuda.get_rng_state_all() if self.device.type == 'cuda' else [],
                'state': copy.deepcopy(self.state if state is None else state),
                'best_model': cpu_tree(self.best_model if best is None else best)}

    def _save(self, state=None, best=None):
        payload = self._payload(state, best)
        core.finite_tree(payload)
        validate_state(payload['state'], self.plan, payload['best_model'])
        core.atomic_save(payload, self.checkpoint)
        self.committed = copy.deepcopy(payload['state'])
        if self.publish:
            self.publish('checkpoint_committed', self.committed)

    def resume(self):
        need(not self.started, 'Resume before run only')
        payload = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        need(payload['schema'] == SCHEMA and payload['run_id'] == self.run_id and
             payload['plan'] == self.plan and payload['plan_sha256'] == core.plan_digest(self.plan), 'Resume identity mismatch')
        core.finite_tree(payload)
        validate_state(payload['state'], self.plan, payload['best_model'])
        reference = self.model.state_dict()
        for weights in (payload['model'], payload['best_model']):
            if weights is not None:
                need(weights.keys() == reference.keys() and all(
                    isinstance(weights[k], torch.Tensor) and weights[k].shape == reference[k].shape and
                    weights[k].dtype == reference[k].dtype for k in reference), 'Model signature mismatch')
        groups = self.optimizer.state_dict()['param_groups']
        need(payload['optimizer']['param_groups'] == groups, 'Adam configuration mismatch')
        steps = payload['state']['updates']
        indices = groups[0]['params']
        states = payload['optimizer']['state']
        need(set(states) == (set(indices) if steps else set()), 'Adam coverage mismatch')
        for index, param in zip(indices, self.optimizer.param_groups[0]['params']):
            if steps:
                value = states[index]
                need(set(value) == {'step', 'exp_avg', 'exp_avg_sq'} and value['step'].numel() == 1 and
                     value['step'].item() == steps and all(value[k].dtype == param.dtype and value[k].shape == param.shape
                     for k in ('exp_avg', 'exp_avg_sq')) and (value['exp_avg_sq'] >= 0).all().item(), 'Adam state mismatch')
        original_rng = core.capture_rng()
        try:
            core.restore_rng(payload['rng'])
        finally:
            core.restore_rng(original_rng)
        cuda_count = torch.cuda.device_count() if self.device.type == 'cuda' else 0
        need(len(payload['cuda_rng']) == cuda_count, 'CUDA RNG device mismatch')
        for index, rng in enumerate(payload['cuda_rng']):
            torch.Generator(device=f'cuda:{index}').set_state(rng)
        # All validation completes before mutating the live model/optimizer/RNG.
        self.model.load_state_dict(payload['model'], strict=True)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.state = copy.deepcopy(payload['state'])
        self.committed = copy.deepcopy(self.state)
        self.best_model = cpu_tree(payload['best_model'])
        self.optimizer.zero_grad(set_to_none=True)
        core.restore_rng(payload['rng'])
        if cuda_count:
            torch.cuda.set_rng_state_all(payload['cuda_rng'])
        self.started = True
        if self.publish:
            self.publish('restored_from_checkpoint', self.committed)

    def _run(self, stop):
            n, b = len(self.plan['train_ids']), self.plan['effective']
            while self.state['epoch'] < self.plan['max_epochs'] and self.state['bad_epochs'] < self.plan['patience']:
                training_mode(self.model)
                order = torch.randperm(n, generator=torch.Generator().manual_seed(
                    self.plan['seed'] + self.state['epoch'])).tolist()
                metric = restore_metrics(self.state['train_metrics'])
                while self.state['cursor'] < n:
                    if stop('train', copy.deepcopy(self.state), self.state['cursor']):
                        self._save()
                        return 'paused'
                    start = self.state['cursor']
                    indices = order[start:start + b]
                    self.optimizer.zero_grad(set_to_none=True)
                    for index in indices:
                        x, y = self._batch('train', index)
                        logits = self.model(x)
                        (summed_ce(logits, y) / len(indices)).backward()
                        metric.add(logits.detach().cpu(), y.cpu())
                    torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                                  1., norm_type=2., error_if_nonfinite=True)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    finite_tree(self.model.state_dict())
                    finite_tree(self.optimizer.state_dict())
                    self.state['cursor'] += len(indices)
                    self.state['updates'] += 1
                    self.state['train_metrics'] = metrics_state(metric)
                    if self.state['updates'] % self.plan['checkpoint_every'] == 0 or self.state['cursor'] == n:
                        self._save()
                # Train-end checkpoint is the validation restart point, including RNG.
                if stop('validation', copy.deepcopy(self.state), 0):
                    return 'paused'
                self.model.eval()
                validation = BinaryMetrics()
                with torch.no_grad():
                    for index in range(len(self.plan['validation_ids'])):
                        x, y = self._batch('validation', index)
                        logits = self.model(x)
                        summed_ce(logits, y)
                        validation.add(logits.detach().cpu(), y.cpu())
                        if stop('validation', copy.deepcopy(self.state), index + 1):
                            # Do not persist partial validation or advance RNG checkpoint.
                            return 'paused'
                candidate = copy.deepcopy(self.state)
                candidate['epoch'] += 1
                candidate['history'].append({'epoch': candidate['epoch'], 'train': metric.result(),
                                             'train_state': metrics_state(metric), 'validation': validation.result(),
                                             'validation_state': metrics_state(validation)})
                ce = validation.result()['ce']
                best = self.best_model
                if candidate['best_ce'] is None or ce < candidate['best_ce']:
                    candidate.update(best_ce=ce, best_epoch=candidate['epoch'], bad_epochs=0)
                    best = copy.deepcopy(self.model.state_dict())
                else:
                    candidate['bad_epochs'] += 1
                candidate.update(cursor=0, train_metrics=metrics_state(BinaryMetrics()))
                self._save(candidate, best)  # Only this atomic replacement commits selection.
                self.state, self.best_model = candidate, best
            return 'completed'
