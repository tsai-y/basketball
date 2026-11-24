"""One ACA arm, fixed 24+8 clips, four temporary updates; no formal training.

Reuse the frozen input/worker helpers and reviewed numerical loop. All temporary
weights are removed after the continuous versus resume comparison.
"""
import copy,gc,json,os,platform,shutil,subprocess,tempfile,time
from pathlib import Path
import cv2,numpy as np,torch
from development.aca_eight_class_data_v1 import loop_core as core
from development.aca_eight_class_data_v1.gpu_readiness import (
    sha,cpu_tree,equal,configure,reset_seed,tensor_sha,choose_ids,OrderedStream,json_write,SEED)
from development.aca_eight_class_data_v1.adapter import EightClassDataset
from development.aca_eight_class_data_v1.foundation import ROOT,CONTRACT_SHA256,baseline
from development.aca_eight_class_data_v1.model_adapter import SPEC_SHA256,load_model_contract
from development.aca_eight_class_data_v1.formal_train import audit_reference,run_lock,runtime_identity
from development.aca_four_arm_campaign_v1.gpu_guard import make_policy,assert_idle_gpu
from .model import initialized,training_mode,contract,PROPOSAL_SHA
from .engine import EngineeringLoop,SCHEMA
need=core.need
SOURCE_RUN=Path('/home/tsai/projects/basketball/outputs/01_current/baseline8_formal_seed20261024_v1')

def run(source_root, output):
    need(not output.exists(), 'Do not overwrite a previous readiness report')
    output.mkdir(parents=True)
    started = time.monotonic()
    json_write(output / 'status.json', {'status': 'checking_resources', 'formal_training_started': False})
    temp_path = None
    try:
        spec = load_model_contract()
        contract()
        audited = audit_reference()
        expected_runtime = json.loads((SOURCE_RUN/"plan.json").read_text())["runtime"]
        need(runtime_identity() == expected_runtime, "Baseline runtime changed; stop for compatibility review")
        policy = make_policy()
        observed = assert_idle_gpu(policy)
        need(shutil.disk_usage('/tmp').free >= 4 * 1024**3, 'Need 4GiB temporary checkpoint space')
        configure()
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total',
                                        '--format=csv,noheader,nounits'], text=True).strip()
        selected = choose_ids()
        datasets = {split: EightClassDataset(source_root=source_root, split=split, video_ids=ids)
                    for split, ids in selected.items()}
        reference, sources = {}, {}
        print('STAGE: serial input reference, exactly 24 train + 8 validation', flush=True)
        for split, dataset in datasets.items():
            for index, vid in enumerate(selected[split]):
                sources[vid] = baseline.guarded_source_sha256(dataset.foundation.scope, vid)
                item = dataset[index]
                reference[vid] = {'class_id': item['class_id'], 'binary_label': item['binary_label'],
                                  'tensor_sha256': tensor_sha(item['tensor'])}
                expected = audited[vid]
                need(sources[vid] == expected['source_sha256'] and
                     reference[vid] == {k:expected[k] for k in ('class_id','binary_label','tensor_sha256')},
                     'Source or tensor differs from frozen full audit')
        del item
        plan = {'schema': SCHEMA, 'purpose': 'engineering_only_not_accuracy_evidence',
                'formal_training_authorized': False, 'test_enabled': False,
                'arm':'aca','proposal_sha256':PROPOSAL_SHA,
                'seed': SEED, 'effective': 16, 'patience': 5, 'max_epochs': 1, 'checkpoint_every': 25,
                'device': 'cuda', 'spec_sha256': SPEC_SHA256, 'input_contract_sha256': CONTRACT_SHA256,
                'train_ids': selected['train'], 'validation_ids': selected['validation'],
                'reference': reference, 'source_sha256': sources, 'source_root': str(source_root),
                'gpu_policy': policy,
                'bounds': {'unique_train': 24, 'unique_validation': 8, 'total_real_optimizer_steps': 4,
                           'total_real_forward_backward_samples': 48, 'total_real_eval_samples': 16,
                           'total_video_decodes': 96, 'wall_timeout_seconds_external': 900,
                           'checkpoint_initialization_for_formal_training': 'prohibited'},
                'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'numpy': np.__version__, 'opencv': cv2.__version__, 'cuda': torch.version.cuda,
                            'cudnn': torch.backends.cudnn.version(), 'gpu': gpu, 'workers': 2,
                            'spawn': True, 'threads': 4, 'amp': False, 'tf32': False,
                            'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']},
                'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (
                    Path(__file__), Path(core.__file__), ROOT/'development/aca_eight_class_data_v1/adapter.py',
                    ROOT/'development/aca_eight_class_data_v1/model_adapter.py',
                    ROOT/'development/aca_eight_class_data_v1/foundation.py',
                    ROOT/'development/aca_four_arm_campaign_v1/gpu_guard.py',
                    ROOT/'development/aca_reconstruction_v2/model.py',
                    ROOT/'development/aca_reconstruction_v2/engine.py')}}
        json_write(output/'plan.json', plan)
        json_write(output/'status.json', {'status': 'running_bounded_probe', 'formal_training_started': False})
        # Verify saved CUDA RNG actually reproduces a subsequent draw, then restore it.
        rng = torch.cuda.get_rng_state_all()
        expected_draw = torch.rand(32, device='cuda')
        torch.cuda.set_rng_state_all(rng)
        need(torch.equal(expected_draw, torch.rand(32, device='cuda')), 'CUDA RNG roundtrip mismatch')
        torch.cuda.set_rng_state_all(rng)
        del expected_draw
        totals = {'updates': 0}
        observed_runs, gradient_checks, saved_sizes = {}, [], []
        with tempfile.TemporaryDirectory(prefix='aca_v2_cuda_readiness_') as scratch:
            temp_path = scratch
            def perform(name, *, cursor=0, stop_at_16=False, resume=False):
                assert_idle_gpu(policy)
                reset_seed()
                model, init = initialized('aca')
                training_mode(model)
                frozen = {k: v.clone() for k, v in model.state_dict().items()
                          if '.aca.' not in k and k not in {n for n, p in model.named_parameters() if p.requires_grad}}
                initial = {k: model.state_dict()[k].clone() for k in ('conv1.weight', 'layer4.2.conv3.weight', 'fc.weight')}
                model.cuda()
                stream = OrderedStream(datasets, reference, cursor=cursor, train_stop=16 if stop_at_16 else 24)
                path = Path(scratch) / ('interrupted.pt' if resume or stop_at_16 else 'continuous.pt')
                loop = EngineeringLoop(model, plan, stream, path, 'aca-v2-bounded-engineering')
                if resume:
                    loop.resume()
                original_step = loop.optimizer.step
                def bounded_step(*args, **kwargs):
                    need(totals['updates'] < 4 and time.monotonic()-started < 900, 'Probe execution cap reached')
                    groups = {'backbone': 0., 'head': 0., 'lsta':0., 'tsci':0.}
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            need(param.grad is not None and torch.isfinite(param.grad).all().item(), 'Nonfinite/missing gradient')
                            group = 'lsta' if '.aca.lsta.' in name else 'tsci' if '.aca.tsci.' in name else 'head' if name.startswith('fc.') else 'backbone'
                            groups[group] += param.grad.abs().sum().item()
                        else:
                            need(param.grad is None, 'Frozen BN received gradient')
                    need(all(v > 0 for v in groups.values()), 'Missing backbone/head gradients')
                    result = original_step(*args, **kwargs)
                    totals['updates'] += 1
                    gradient_checks.append(groups)
                    print(f'STAGE: {name_of_run}, engineering update {totals["updates"]}/4', flush=True)
                    return result
                name_of_run = name
                loop.optimizer.step = bounded_step
                eval_bn_checks = []
                def before_eval(mod, args):
                    if not mod.training:
                        eval_bn_checks.append({n:v.detach().clone() for n,v in mod.named_buffers()})
                def after_eval(mod, args, result):
                    if not mod.training:
                        need(all(torch.equal(dict(mod.named_buffers())[n],v) for n,v in eval_bn_checks[-1].items()),
                             'Validation changed normalization buffers')
                        eval_bn_checks[-1] = True
                hooks=[model.register_forward_pre_hook(before_eval),model.register_forward_hook(after_eval)]
                torch.cuda.reset_peak_memory_stats()
                phase_start = time.monotonic()
                try:
                    status = loop.run(lambda phase, state, seen: stop_at_16 and phase == 'train' and state['updates'] == 1)
                    torch.cuda.synchronize()
                    need(status == ('paused' if stop_at_16 else 'completed'), 'Unexpected loop status')
                    need(all(torch.equal(model.state_dict()[k].cpu(), v) for k, v in frozen.items()), 'Frozen BN state changed')
                    need(all(not torch.equal(model.state_dict()[k].cpu(), v) for k, v in initial.items()), 'Backbone/head did not update')
                    counters={n:v.item() for n,v in model.named_buffers() if '.aca.tsci.' in n and n.endswith('num_batches_tracked')}
                    need(len(counters)==48 and set(counters.values())=={16 if stop_at_16 else 24},
                         'New BN counter not restored/trained exactly once per training clip')
                    need(len(eval_bn_checks)==(0 if stop_at_16 else 8) and all(eval_bn_checks), 'Validation buffer check missing')
                    payload = loop._payload()
                    need(equal(payload, torch.load(path, map_location='cpu', weights_only=True)), 'Disk checkpoint differs from committed live state')
                    observed_runs[name] = {'status': status, 'stream': stream.observed,
                                          'updates': loop.state['updates'], 'cursor': loop.state['cursor'],
                                          'history_epochs': len(loop.state['history']),
                                          'seconds': time.monotonic()-phase_start,
                                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                          'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                                          'frozen_backbone_bn_unchanged': True, 'backbone_and_head_updated': True,
                                          'new_tsci_bn_train_counters':counters,'validation_bn_buffers_unchanged':True}
                    saved_sizes.append(path.stat().st_size)
                    return payload
                finally:
                    for h in hooks:h.remove()
                    stream.close()
                    loop.optimizer.step = original_step
                    del loop, model, original_step, bounded_step
                    gc.collect()
                    torch.cuda.empty_cache()
            continuous = perform('continuous')
            paused = perform('interrupted', stop_at_16=True)
            need(paused['state']['cursor'] == 16 and paused['state']['updates'] == 1 and
                 not paused['state']['history'] and paused['best_model'] is None, 'Wrong pause checkpoint')
            del paused
            resumed = perform('resumed', cursor=16, resume=True)
            comparisons = {key: equal(continuous[key], resumed[key]) for key in continuous}
            need(all(comparisons.values()), 'Continuous/resumed trajectory mismatch: ' + repr(comparisons))
            need(totals['updates'] == 4 and resumed['state']['updates'] == 2 and
                 resumed['state']['history'][0]['train']['count'] == 24 and
                 resumed['state']['history'][0]['validation']['count'] == 8, 'Wrong final coverage')
            del continuous, resumed
        need(not Path(temp_path).exists(), 'Temporary checkpoints were not removed')
        for split, dataset in datasets.items():
            for vid in selected[split]:
                need(baseline.guarded_source_sha256(dataset.foundation.scope, vid) == sources[vid], 'Original video changed')
        need(all(sha(ROOT/p) == v for p, v in plan['code_sha256'].items()), 'Readiness code changed during execution')
        load_model_contract()
        report = {'schema': SCHEMA, 'status': 'passed', 'plan_sha256': core.plan_digest(plan),
                  'formal_training_started': False, 'formal_training_ready': False, 'test_videos_opened': 0,
                  'real_optimizer_updates_engineering_only': totals['updates'], 'gpu_rng_draw_roundtrip_equal': True,
                  'continuous_vs_resumed_equal': comparisons, 'workers_equal_serial_tensor_sha': True,
                  'worker_rng_isolated': True, 'original_videos_unchanged': True,
                  'runs': observed_runs, 'gradient_checks': gradient_checks, 'gpu_processes_before_start': observed,
                  'checkpoint_sizes_bytes': saved_sizes, 'temporary_checkpoints_removed': True,
                  'elapsed_seconds': time.monotonic()-started,
                  'scope_note': '32 unique real train/validation clips, 96 bounded decodes, 4 temporary engineering updates; no scores are research results.',
                  'remaining': ['Formal four-arm launch interface not implemented or authorized.',
                                'This real CUDA probe covers ACA only; other arms have CPU/meta engineering evidence.',
                                'No author-exact ACA claim or accuracy conclusion.']}
        json_write(output/'report.json', report)
        json_write(output/'status.json', {'status': 'passed', 'formal_training_started': False})
        print('PASSED: bounded real input/CUDA/worker/checkpoint comparison; no formal run started', flush=True)
    except BaseException as exc:
        json_write(output/'failure.json', {'error': repr(exc), 'formal_training_started': False,
                                           'elapsed_seconds': time.monotonic()-started})
        json_write(output/'status.json', {'status': 'failed_stopped', 'formal_training_started': False})
        raise

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with run_lock(Path('/home/tsai/projects/basketball/outputs/00_admin'),'baseline8_gpu.lock'):
        run(Path('/home/tsai/projects/basketball'),args.output)

if __name__=='__main__':main()
