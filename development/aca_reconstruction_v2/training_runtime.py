"""Formal checkpoint boundary; numerical loop is the GPU-validated engineering loop.

Full source/data/runtime identity is checked by the CLI before creating a model.
No changes to the frozen baseline or bounded engineering adapters.
"""
import copy
from pathlib import Path
import torch
from development.aca_eight_class_data_v1 import loop_core as core
from development.aca_eight_class_data_v1 import training_runtime as base
from development.aca_eight_class_data_v1.training_runtime import (
    atomic_json, now, validate_state, VerifiedDataset, EpochStream)
from development.aca_eight_class_data_v1.gpu_readiness import cpu_tree
from .model import training_mode
from .engine import EngineeringLoop
SCHEMA='aca_reconstruction_v2_formal_checkpoint_v1'
PLAN_SCHEMA='aca_reconstruction_v2_single_seed_formal_plan_v1'
need=core.need

class TrainingLoop(base.TrainingLoop):
    _run = EngineeringLoop._run
    def __init__(self, model, plan, fetch, checkpoint, run_id, *, publish=None):
        need(plan.get('schema') == PLAN_SCHEMA and plan.get('variant') in ('lsta','tsci','aca') and
             plan.get('test_enabled') is False and plan['seed'] == 20261024 and
             plan['max_epochs'] == 30 and plan['patience'] == 5 and plan['effective'] == 16 and
             plan['checkpoint_every'] == 25 and len(plan['train_ids']) == 7192 and
             len(plan['validation_ids']) == 1547, 'Fixed formal single-seed scope required')
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

    def _payload(self, state=None, best=None):
        payload = super()._payload(state, best)
        payload['schema'] = SCHEMA
        return payload

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
