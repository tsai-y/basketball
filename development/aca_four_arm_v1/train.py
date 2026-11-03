"""Explicit four-arm commands; default plan, never an automatic queue."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import torch

from development.aca_net_v1.data import Basketball50Dataset, require
from development.aca_net_v1.train import atomic_json, canonical_hash, configure_runtime, initial_progress
from . import engine
from .experiment import (ArmConfig, FOLDER, VARIANTS, arm_plan, assert_idle_gpu, campaign_plan,
                         initialized, load_imagenet, prepare, verify_preflight)
from development.aca_net_v1.data import ROOT, file_sha256


def start_training(source_root, config, report, *, accepted=False, resume=None):
    require(accepted, "Explicitly acknowledge the unverified candidate architecture")
    assert_idle_gpu()
    campaign = campaign_plan(source_root)
    verify_preflight(report, campaign, config)
    configure_runtime(config.seed)
    state, _ = load_imagenet()
    model, _ = initialized(config.variant, config.seed, state)
    del state
    model = model.cuda()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config.lr, weight_decay=config.weight_decay)
    plan = arm_plan(campaign, config)
    if resume:
        output = Path(resume).resolve()
        require(output.parent == (FOLDER/'artifacts').resolve() and output.name.startswith('formal_'),
                "Resume must use a formal four-arm output, never an engineering run")
        require(canonical_hash(json.loads((output/'plan.json').read_text())) == canonical_hash(plan),
                "Saved arm plan mismatch")
        progress = engine.load_checkpoint(output/'last.pt', model, optimizer, plan)
    else:
        (FOLDER/'artifacts').mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f"formal_{config.variant}_seed{config.seed}_", dir=FOLDER/'artifacts'))
        atomic_json(output/'plan.json', plan); atomic_json(output/'campaign.json', campaign)
        shutil.copy2(report, output/'readiness.json')
        for name, expected in campaign['source_code_sha256'].items():
            require(file_sha256(ROOT/name) == expected, 'Source changed before formal snapshot')
            target=output/'source_snapshot'/name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        progress = initial_progress()
        engine.save_checkpoint(output/'last.pt', model, optimizer, progress, plan)
    print(f"OUTPUT: {output}", flush=True)
    train = Basketball50Dataset(source_root=source_root, split='train')
    validation = Basketball50Dataset(source_root=source_root, split='validation')
    return engine.fit(model, optimizer, train, validation, config, plan, output, progress, device='cuda')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--start-training', action='store_true')
    parser.add_argument('--preparation', type=Path)
    parser.add_argument('--readiness-report', type=Path)
    parser.add_argument('--variant', choices=VARIANTS, default='baseline')
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--accept-candidate-architecture', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    require(not args.resume or args.start_training, 'Resume requires explicit training command')
    if args.start_training:
        require(args.readiness_report is not None, 'Matching GPU readiness report is required')
        value = start_training(args.source_root, ArmConfig(variant=args.variant, seed=args.seed),
                               args.readiness_report, accepted=args.accept_candidate_architecture,
                               resume=args.resume)
    elif args.prepare:
        value = prepare(args.source_root)
    elif args.preflight:
        require(args.preparation is not None, 'Matching CPU preparation is required')
        from .preflight import run
        value = run(args.source_root, args.preparation, seed=args.seed)
    else:
        value = campaign_plan(args.source_root)
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
