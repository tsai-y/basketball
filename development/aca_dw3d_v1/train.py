"""ACA-DW3D candidate CLI: default prints a plan; no automatic chaining or test path."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import torch

from development.aca_net_v1.data import Basketball50Dataset, require
from development.aca_net_v1.train import atomic_json, canonical_hash, configure_runtime, initial_progress
from . import engine
from .experiment import (FOLDER, assert_launch_prerequisites, build_plan, config_from,
                         initialized_candidate, prepare, snapshot_sources, verify_gpu_report)


def run_training(source_root, config, report_path, *, accepted, resume=None, on_created=None):
    assert_launch_prerequisites(accepted)
    require(torch.cuda.is_available(), "CUDA required; no CPU fallback")
    configure_runtime(config.seed)
    plan = build_plan(source_root, config)
    verify_gpu_report(report_path, plan)
    model, _ = initialized_candidate(config.seed)
    model = model.cuda()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config.lr, weight_decay=config.weight_decay)
    if resume:
        output = Path(resume).resolve()
        require(output.parent == (FOLDER / "artifacts").resolve() and output.name.startswith("aca_dw3d_train_"),
                "Resume must use a formal candidate run, not baseline or engineering artifacts")
        require(canonical_hash(json.loads((output / "config.json").read_text())) == canonical_hash(plan),
                "Saved candidate plan mismatch")
        progress = engine.load_checkpoint(output / "last.pt", model, optimizer, plan)
    else:
        (FOLDER / "artifacts").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f"aca_dw3d_train_seed{config.seed}_", dir=FOLDER / "artifacts"))
        atomic_json(output / "config.json", plan)
        shutil.copy2(report_path, output / "gpu_preflight.json")
        snapshot_sources(output, plan)
        progress = initial_progress()
        engine.save_checkpoint(output / "last.pt", model, optimizer, progress, plan)
    print(f"RUN: {output}", flush=True)
    if on_created is not None:
        on_created(output)
    train_data = Basketball50Dataset(source_root=source_root, split="train")
    validation_data = Basketball50Dataset(source_root=source_root, split="validation")
    return engine.fit(model, optimizer, train_data, validation_data, config, plan, output, progress, device="cuda")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=FOLDER / "config.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="CPU only; no video decoding or optimizer")
    mode.add_argument("--preflight", action="store_true", help="After baseline finishes: bounded GPU engineering update")
    mode.add_argument("--start-training", action="store_true")
    parser.add_argument("--accept-candidate-architecture", action="store_true")
    parser.add_argument("--preflight-report", type=Path, default=FOLDER / "artifacts/latest_gpu_preflight.json")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = config_from(args.config)
    require(not args.resume or args.start_training, "Resume requires --start-training")
    if args.start_training:
        require(args.preflight_report.is_file(), "Candidate GPU preflight report missing; run --preflight after baseline finishes")
        result = run_training(args.source_root, config, args.preflight_report,
                              accepted=args.accept_candidate_architecture, resume=args.resume)
    elif args.preflight:
        from .preflight import run
        result = run(args.source_root, config, accepted=args.accept_candidate_architecture)
    elif args.prepare:
        result = prepare(args.source_root, config)
    else:
        result = build_plan(args.source_root, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
