"""Bounded acceptance: CPU identity/BN formula and two existing train clips on CUDA."""
import argparse
import copy
import gc
import json
import torch
from torch import nn
from development.aca_binary_v1 import readiness as previous
from development.aca_binary_v1.model import initialized as baseline_initialized
from development.aca_eight_class_data_v1.gpu_readiness import equal
from . import formal_train as formal
from .model import initialized, apply_momentum, tsci_bn_names, validate_momentum


def report(plan, fields, name):
    formal.QA.mkdir(exist_ok=True)
    path = formal.QA/name
    formal.need(not path.exists(), 'Readiness already exists')
    value = dict(status='passed', at_taiwan=formal.runtime.now(), draft_plan_sha256=formal.core.plan_digest(plan),
                 code_sha256={k:v for k,v in plan['code_sha256'].items() if k.startswith(formal.PREFIX)},
                 test_video_decodes=0, formal_training_started=False, **fields)
    formal.runtime.atomic_json(path, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def cpu():
    # Plan creation queries host runtime, but numerical work here stays on CPU.
    plan = formal.draft_plan(); torch.set_num_threads(4)
    before_rng = formal.core.capture_rng()
    baseline, _ = baseline_initialized()
    original_state = copy.deepcopy(baseline.state_dict())
    attrs = {n:(m.momentum,m.eps,m.training,m.affine,m.track_running_stats,
                tuple(p.requires_grad for p in m.parameters()))
             for n,m in baseline.named_modules() if isinstance(m,nn.BatchNorm2d)}
    rng_before_apply = formal.core.capture_rng()
    names = apply_momentum(baseline)
    assert equal(rng_before_apply, formal.core.capture_rng())
    assert equal(original_state, baseline.state_dict())
    for n,m in baseline.named_modules():
        if isinstance(m,nn.BatchNorm2d):
            expected = (0.01 if n in names else attrs[n][0],) + attrs[n][1:]
            assert (m.momentum,m.eps,m.training,m.affine,m.track_running_stats,
                    tuple(p.requires_grad for p in m.parameters())) == expected
    after_rng = formal.core.capture_rng()
    formal.core.restore_rng(before_rng)
    candidate, info = initialized()
    assert equal(after_rng, formal.core.capture_rng()) and equal(original_state, candidate.state_dict())
    assert candidate.fc.out_features == 2 and info['head_shape'] == [2,2048]
    validate_momentum(candidate)
    # Explicit exponential moving mean/variance formula, independent of module construction.
    bn = copy.deepcopy(candidate.get_submodule(names[0])).double().train()
    x = torch.arange(24,dtype=torch.float64).reshape(2,1,3,4)/7
    oldmean,oldvar=bn.running_mean.clone(),bn.running_var.clone()
    bn(x)
    assert torch.allclose(bn.running_mean, .99*oldmean+.01*x.mean((0,2,3)),atol=1e-12,rtol=0)
    assert torch.allclose(bn.running_var, .99*oldvar+.01*x.var((0,2,3),unbiased=True),atol=1e-12,rtol=0)
    bn.eval();saved=copy.deepcopy(bn.state_dict());bn(x)
    assert equal(saved,bn.state_dict())
    del baseline,candidate,original_state;gc.collect()
    report(plan, dict(checks=['only_48_TSCI_momentum_attributes_changed','all_parameters_buffers_and_RNG_equal',
              'backbone_BN_policy_unchanged','binary_head_identical','running_mean_variance_formula_0.01',
              'evaluation_does_not_update_buffers','plan_diff_allowlist'], tsci_bn_count=48,
              real_video_decodes=0, engineering_optimizer_updates=0), 'cpu_report.json')


def gpu():
    # Reuse unchanged, bounded two-train-video probe. Only dependencies are redirected.
    # Mutations are process-local; no old source, evidence or numerical loop is changed.
    previous.formal = formal
    previous.initialized = initialized
    previous.report = report
    formal.original = formal.ORIGINAL
    previous.gpu()


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);a=p.parse_args()
    cpu() if a.mode=='cpu' else gpu()
