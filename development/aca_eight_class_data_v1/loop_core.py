"""Eight-class loop core, currently gated to CPU synthetic engineering fixtures.

No CLI, real-data loader, model initializer, CUDA launcher or automatic resume.
The authoritative checkpoint embeds both current and selected weights. Its atomic
replacement is the validation commit; derived reports must be regenerated from it.
"""
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from .adapter import CLASSES, EightClassMetrics
from .model_adapter import SPEC_SHA256, load_model_contract

SCHEMA = 'eight_class_loop_cpu_fixture_v1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def metrics_state(metric):
    return {'schema': 'eight_class_metrics_v1', 'classes': list(CLASSES),
            'matrix': copy.deepcopy(metric._matrix), 'loss_sum': metric._loss_sum,
            'count': metric._count}


def restore_metrics(state):
    need(state['schema'] == 'eight_class_metrics_v1' and state['classes'] == list(CLASSES),
         'Metric schema/classes mismatch')
    matrix, count, loss = state['matrix'], state['count'], state['loss_sum']
    need(type(count) is int and count >= 0 and len(matrix) == 8 and
         all(len(row) == 8 and all(type(v) is int and v >= 0 for v in row) for row in matrix),
         'Invalid confusion/count')
    need(sum(map(sum, matrix)) == count and type(loss) in (float, int) and
         math.isfinite(loss) and loss >= 0 and (count > 0 or loss == 0), 'Invalid metric totals')
    metric = EightClassMetrics()
    metric._matrix, metric._count, metric._loss_sum = copy.deepcopy(matrix), count, float(loss)
    return metric


def capture_rng():
    ns = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
            'torch_cpu': torch.get_rng_state().clone()}


def restore_rng(state):
    # Validate on isolated generators before mutating process-wide RNG.
    random.Random().setstate(state['python'])
    ns = state['numpy']
    numpy_state = (ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4])
    np.random.RandomState().set_state(numpy_state)
    torch.Generator().set_state(state['torch_cpu'])
    random.setstate(state['python'])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state['torch_cpu'])


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        need(not value.is_floating_point() or torch.isfinite(value).all().item(), 'Nonfinite state')
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        need(math.isfinite(value), 'Nonfinite state')


def atomic_save(payload, path):
    """Same-directory temp + fsync + replace. Failure before replace preserves last commit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def fixture_plan(train_ids, validation_ids, *, model_identity, max_epochs=30):
    """Explicit synthetic-only plan. Code/runtime/data identity binds resume."""
    load_model_contract()
    for ids in (train_ids, validation_ids):
        need(bool(ids) and list(ids) == sorted(set(ids)) and all(type(x) is str for x in ids),
             'Require nonempty sorted unique IDs')
    need(not set(train_ids) & set(validation_ids), 'Overlapping split IDs')
    need(type(max_epochs) is int and 1 <= max_epochs <= 30, 'Invalid epoch cap')
    need(isinstance(model_identity, str) and bool(model_identity), 'Missing fixture identity')
    here = Path(__file__)
    return {'schema': SCHEMA, 'purpose': 'synthetic_cpu_only_not_formal_training',
            'spec_sha256': SPEC_SHA256, 'train_ids': list(train_ids),
            'validation_ids': list(validation_ids), 'model_identity': model_identity,
            'seed': 20261024, 'effective': 16, 'patience': 5, 'max_epochs': max_epochs,
            'checkpoint_every': 25, 'device': 'cpu',
            'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (here, here.with_name('adapter.py'), here.with_name('model_adapter.py'))},
            'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                        'numpy': str(np.__version__), 'threads': torch.get_num_threads(),
                        'deterministic': torch.are_deterministic_algorithms_enabled()}}


def plan_digest(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def training_mode(model):
    model.train()
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            layer.eval()
            for parameter in layer.parameters():
                parameter.requires_grad_(False)


def summed_ce(logits, targets):
    """Device-local FP32 eight-class loss; metrics remain detached on CPU."""
    need(logits.dtype == torch.float32 and logits.ndim == 2 and logits.shape[1] == 8 and
         targets.dtype == torch.long and targets.shape == (len(logits),) and
         targets.device == logits.device and len(logits) > 0, 'Invalid logits/targets')
    need(torch.isfinite(logits).all().item() and ((targets >= 0) & (targets < 8)).all().item(),
         'Nonfinite logits or invalid targets')
    value = F.cross_entropy(logits, targets, reduction='sum')
    need(torch.isfinite(value).item(), 'Nonfinite loss')
    return value


class FixtureLoop:
    """Reusable optimizer-boundary state machine; real-data execution is not exposed.

    fetch(split, index) must be deterministic and return a CPU microbatch dict
    with tensor, targets and one records item (video_id, split). Identity includes
    the fixture implementation/data digest supplied as model_identity.
    """
    def __init__(self, model, plan, fetch, checkpoint):
        expected = fixture_plan(plan['train_ids'], plan['validation_ids'],
                                model_identity=plan['model_identity'], max_epochs=plan['max_epochs'])
        need(plan == expected, 'Plan mismatch / unsupported execution mode')
        need(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in model.parameters()),
             'CPU FP32 fixture only')
        need(all(b.device.type == 'cpu' for b in model.buffers()), 'CPU buffers only')
        self.model, self.plan, self.fetch, self.checkpoint = model, copy.deepcopy(plan), fetch, Path(checkpoint)
        training_mode(model)
        self.optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                         lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0.,
                                         amsgrad=False, foreach=False, fused=False)
        self.state = {'epoch': 0, 'cursor': 0, 'updates': 0, 'train_metrics': metrics_state(EightClassMetrics()),
                      'history': [], 'best_epoch': None, 'best_ce': None, 'bad_epochs': 0}
        self.best_model = None
        self.valid = True
        self.started = False

    def _payload(self, state=None, best=None):
        return {'schema': SCHEMA, 'plan_sha256': plan_digest(self.plan), 'plan': copy.deepcopy(self.plan),
                'model': copy.deepcopy(self.model.state_dict()), 'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                'rng': capture_rng(), 'state': copy.deepcopy(self.state if state is None else state),
                'best_model': copy.deepcopy(self.best_model if best is None else best)}

    def _save(self, state=None, best=None):
        payload = self._payload(state, best)
        finite_tree(payload)
        atomic_save(payload, self.checkpoint)

    def resume(self):
        """Explicit call only; failed loads leave live model/optimizer/RNG/state untouched."""
        need(not self.started, 'Resume must precede run')
        payload = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        need(payload['schema'] == SCHEMA and payload['plan'] == self.plan and
             payload['plan_sha256'] == plan_digest(self.plan), 'Resume identity mismatch')
        finite_tree(payload)
        state = payload['state']
        n, b = len(self.plan['train_ids']), self.plan['effective']
        epoch, cursor, updates = (state[k] for k in ('epoch', 'cursor', 'updates'))
        need(all(type(v) is int for v in (epoch, cursor, updates)) and
             0 <= epoch <= self.plan['max_epochs'] and 0 <= cursor <= n and
             (cursor % b == 0 or cursor == n) and
             updates == epoch * math.ceil(n / b) + math.ceil(cursor / b), 'Invalid optimizer boundary')
        need(epoch < self.plan['max_epochs'] or cursor == 0, 'Cursor beyond epoch cap')
        metric = restore_metrics(state['train_metrics'])
        need(metric._count == cursor and len(state['history']) == epoch, 'History/cursor mismatch')
        best_ce, best_epoch, bad = None, None, 0
        for i, row in enumerate(state['history']):
            need(row['epoch'] == i + 1, 'History epoch mismatch')
            for split, count in (('train', n), ('validation', len(self.plan['validation_ids']))):
                m = restore_metrics(row[split + '_state'])
                need(m._count == count and m.result() == row[split], 'Incomplete/inconsistent history')
            ce = row['validation']['ce']
            if best_ce is None or ce < best_ce:
                best_ce, best_epoch, bad = ce, i + 1, 0
            else:
                bad += 1
            need(bad < self.plan['patience'] or i == epoch - 1, 'History beyond early stop')
        need((best_ce, best_epoch, bad) == (state['best_ce'], state['best_epoch'], state['bad_epochs']),
             'Selection mismatch')
        need((payload['best_model'] is None) == (best_epoch is None), 'Missing/unexpected best weights')
        # Stage loads on copied objects, checking architecture and optimizer shapes.
        reference = self.model.state_dict()
        for weights in (payload['model'], payload['best_model']):
            if weights is not None:
                need(weights.keys() == reference.keys() and all(
                    isinstance(weights[k], torch.Tensor) and weights[k].shape == reference[k].shape and
                    weights[k].dtype == reference[k].dtype for k in reference), 'Model signature mismatch')
        staged_model, staged_optimizer = copy.deepcopy((self.model, self.optimizer))
        staged_model.load_state_dict(payload['model'], strict=True)
        if payload['best_model'] is not None:
            staged_model.load_state_dict(payload['best_model'], strict=True)
        reference_groups = self.optimizer.state_dict()['param_groups']
        need(payload['optimizer']['param_groups'] == reference_groups, 'Optimizer configuration mismatch')
        staged_optimizer.load_state_dict(payload['optimizer'])
        for p in staged_optimizer.param_groups[0]['params']:
            opt = staged_optimizer.state.get(p, {})
            need((updates == 0 and not opt) or
                 (set(opt) == {'step', 'exp_avg', 'exp_avg_sq'} and
                  opt['step'].numel() == 1 and opt['step'].item() == updates and
                  opt['exp_avg'].shape == p.shape and opt['exp_avg_sq'].shape == p.shape),
                 'Invalid Adam state')
        # RNG validation is performed without retaining any global change.
        original_rng = capture_rng()
        try:
            restore_rng(payload['rng'])
        finally:
            restore_rng(original_rng)
        self.model.load_state_dict(payload['model'], strict=True)
        self.optimizer.load_state_dict(payload['optimizer'])
        self.state, self.best_model = copy.deepcopy(state), copy.deepcopy(payload['best_model'])
        self.optimizer.zero_grad(set_to_none=True)
        restore_rng(payload['rng'])
        self.started = True

    def _batch(self, split, index):
        batch = self.fetch(split, index)
        records, x, y = batch['records'], batch['tensor'], batch['targets']
        need(len(records) == 1 and records[0]['split'] == split and
             records[0]['video_id'] == self.plan[split + '_ids'][index], 'Data order/split mismatch')
        need(x.device.type == 'cpu' and x.dtype == torch.float32 and x.shape[0] == 1 and
             torch.isfinite(x).all().item() and y.device.type == 'cpu' and
             y.dtype == torch.long and y.shape == (1,) and 0 <= y.item() < 8, 'Invalid microbatch')
        return x, y

    def run(self, should_stop=lambda phase, state, seen: False):
        """Stop callback is checked only at optimizer boundaries or in validation.

        Any failure poisons this in-memory object. Resume explicitly into a fresh
        object from the last committed checkpoint; partial mutations are not reused.
        """
        need(self.valid, 'Discard paused/failed loop; explicitly resume a fresh instance')
        try:
            if not self.started:
                need(not self.checkpoint.exists(), 'Existing run requires explicit resume')
                self._save()
                self.started = True
            result = self._run(should_stop)
            if result == 'paused':
                self.valid = False
            return result
        except BaseException:
            self.valid = False
            raise

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
            validation = EightClassMetrics()
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
            candidate.update(cursor=0, train_metrics=metrics_state(EightClassMetrics()))
            self._save(candidate, best)  # Only this atomic replacement commits selection.
            self.state, self.best_model = candidate, best
        return 'completed'
