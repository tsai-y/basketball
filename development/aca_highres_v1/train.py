"""Explicit high-resolution entry: default plan only; never auto-chains training."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import tempfile

import torch
from development.aca_net_v1 import train as base
from development.aca_net_v1.data import require
from development.aca_candidate_v1.experiment import initialized_candidate
from . import experiment as exp


def run_training(source_root, readiness, *, accepted, confirmed, resume=None):
    exp.launch_prerequisites(accepted, confirmed)
    cfg = exp.config_from()
    plan = exp.build_plan(source_root, cfg)
    exp.validate_readiness(readiness, plan)
    if resume:
        output = Path(resume).resolve()
        require(output.parent == exp.artifact_parent().resolve() and output.name.startswith('aca_highres_train_'),
                'Resume only a formal 256x192 run; old low-resolution/engineering checkpoints refused')
        require(base.canonical_hash(json.loads((output / 'config.json').read_text())) == base.canonical_hash(plan),
                'Resume plan mismatch')
    exp.artifact_parent().mkdir(exist_ok=True)
    # OS lock releases on exit; never stop another process or delete a claim.
    with (exp.artifact_parent() / 'formal_launch.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        exp.gpu_idle()  # Recheck after acquiring the single-worker lock.
        require(torch.cuda.is_available(), 'CUDA required; no CPU fallback')
        free, _ = torch.cuda.mem_get_info()
        require(free >= 11 * 1024**3, 'Less than 11 GiB free GPU memory; no automatic parameter changes')
        base.configure_runtime(cfg.seed)
        model, _ = initialized_candidate(cfg.seed)  # Never initialize from readiness or low-resolution weights.
        model = model.cuda()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
        from . import engine
        if resume:
            progress = engine.load_checkpoint(output / 'last.pt', model, optimizer, plan)
        else:
            output = Path(tempfile.mkdtemp(prefix='aca_highres_train_', dir=exp.artifact_parent()))
            base.atomic_json(output / 'config.json', plan)
            shutil.copy2(readiness, output / 'readiness.json')
            exp.snapshot(output, plan)
            progress = base.initial_progress()
            engine.save_checkpoint(output / 'last.pt', model, optimizer, progress, plan)
        print(f'RUN: {output}', flush=True)
        return engine.fit(model, optimizer, exp.dataset(source_root, 'train'), exp.dataset(source_root, 'validation'),
                          cfg, plan, output, progress, device='cuda')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--start-training', action='store_true')
    parser.add_argument('--source-root', type=Path, default=exp.SOURCE)
    parser.add_argument('--preparation', type=Path)
    parser.add_argument('--readiness-report', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--accept-candidate-architecture', action='store_true')
    parser.add_argument('--confirm-formal-training', action='store_true')
    args = parser.parse_args(argv)
    require(not args.resume or args.start_training, '--resume requires an explicit formal start')
    require(not args.confirm_formal_training or args.start_training, 'Confirmation alone cannot start training')
    if args.start_training:
        require(args.readiness_report is not None, 'High-resolution readiness report required')
        result = run_training(args.source_root, args.readiness_report, accepted=args.accept_candidate_architecture,
                              confirmed=args.confirm_formal_training, resume=args.resume)
    elif args.preflight:
        require(args.preparation is not None and args.accept_candidate_architecture, 'Preparation and candidate acceptance required')
        from .preflight import run
        result = run(args.source_root, args.preparation)
    elif args.prepare:
        result = exp.prepare(args.source_root)
    else:
        result = exp.build_plan(args.source_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    main()
