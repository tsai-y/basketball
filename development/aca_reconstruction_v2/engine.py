"""Bounded engineering adapter; no formal run CLI or general campaign launch.

_run is copied verbatim from the frozen core. Its only changed dependency is
training_mode; AST equality is checked. Checkpoints have a separate schema.
"""
import copy
from pathlib import Path
import torch
from development.aca_eight_class_data_v1 import loop_core as core
from development.aca_eight_class_data_v1.loop_core import (restore_metrics, metrics_state, summed_ce,
                                                          finite_tree, EightClassMetrics)
from development.aca_eight_class_data_v1.training_runtime import validate_state
from development.aca_eight_class_data_v1.gpu_readiness import cpu_tree
from .model import training_mode
SCHEMA = 'aca_reconstruction_v2_engineering_only'
need = core.need

class EngineeringLoop(core.FixtureLoop):
    """General device adapter; the CLI validates the complete formal plan first.

    Unit fixtures may supply tiny models/data directly; there is no fixture option
    in the formal CLI. Every checkpoint is bound to both plan hash and run ID.
    """
    def __init__(self, model, plan, fetch, checkpoint, run_id, *, publish=None):
        need(plan.get('schema') == SCHEMA and plan.get('formal_training_authorized') is False and
             plan.get('test_enabled') is False and 1 <= len(plan['train_ids']) <= 24 and
             1 <= len(plan['validation_ids']) <= 8 and 1 <= plan['max_epochs'] <= 2 and
             plan['effective'] == 16 and plan['patience'] == 5 and plan['checkpoint_every'] == 25 and
             plan['seed'] == 20261024, 'Engineering scope only')
        self.device = next(model.parameters()).device
        need(self.device.type in ('cpu', 'cuda') and all(p.dtype == torch.float32 and p.device == self.device
                                                       for p in model.parameters()), 'Need single FP32 device')
        need(bool(run_id), 'Missing run ID')
        self.run_id, self.publish = run_id, publish
        self.model, self.plan, self.fetch, self.checkpoint = model, copy.deepcopy(plan), fetch, Path(checkpoint)
        training_mode(model)
        self.optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                         lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0.,
                                         amsgrad=False, foreach=False, fused=False)
        self.state = {'epoch': 0, 'cursor': 0, 'updates': 0, 'train_metrics': core.metrics_state(core.EightClassMetrics()),
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
