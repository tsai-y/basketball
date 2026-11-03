"""New high-resolution protocol; frozen modules/runs are imported read-only."""
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import tempfile

import torch
from development.aca_candidate_v1 import experiment as dense
from development.aca_net_v1 import train as base
from development.aca_net_v1.data import ROOT, POLICY, VideoScope, file_sha256, require
from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.modules import ReconstructionConfig
from development.aca_net_v1.weights import pin, verify_payload, FOLDER as WEIGHTS
from development.aca_resolution_v1.data import ResolutionDataset
from development.aca_resolution_v1.probe import gpu_idle

FOLDER = Path(__file__).resolve().parent
SCHEMA = 'aca_dense_highres_resume_v1'
PREFLIGHT_SCHEMA = 'aca_dense_highres_readiness_v1'
SOURCE = Path('/home/tsai/projects/basketball')
COST = ROOT / 'development/aca_resolution_v1/artifacts/gpu_cost_ba0juxok'
PREPARATION = ROOT / 'development/aca_resolution_v1/artifacts/cpu_prepare_vygk24ls'


@dataclass(frozen=True)
class HighResConfig:
    variant: str = 'aca_dense_highres_candidate'
    seed: int = 20261025
    epochs: int = 15
    patience: int = 5
    effective_batch: int = 16
    micro_batch: int = 1
    workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 0.
    clip_norm: float = 1.
    checkpoint_every: int = 25

    def __post_init__(self):
        require(self.variant == 'aca_dense_highres_candidate', 'Wrong high-resolution variant')
        base.TrainConfig(**{**asdict(self), 'variant': 'baseline'})
        require(self.micro_batch == 1 and self.seed == 20261025, 'Unreviewed micro-batch/seed change')


def config_from(path=FOLDER / 'config.json'):
    cfg = HighResConfig(**json.loads(Path(path).read_text()))
    require(cfg.epochs == 15 and cfg.patience == 5, 'Prepared budget is total 15, patience 5; re-review budget changes')
    return cfg


def dataset(source_root, split, ids=None):
    require(split in ('train', 'validation'), 'No high-resolution test interface')
    return ResolutionDataset(source_root=source_root, split=split, resolution=(192, 256), video_ids=ids)


def verify_files(expected):
    for path, digest in expected.items():
        require(file_sha256(Path(path)) == digest, f'Protected file changed: {path}')


def historical_protection():
    cost_plan = json.loads((COST / 'plan.json').read_text())
    protected = dict(cost_plan['protected_sha256'])
    # The low-resolution test is historical evidence only: no labels/predictions
    # or scores are read by this training interface.
    test_folder = ROOT / 'development/aca_test_v1/artifacts/three_equal_test_g62sv35q'
    for path in [*COST.glob('*.json'), PREPARATION / 'plan.json', PREPARATION / 'report.json',
                 *test_folder.glob('*')]:
        if path.is_file():
            protected[str(path)] = file_sha256(path)
    verify_files(protected)
    return protected


def source_hashes():
    result = dense.source_hashes()
    for folder in (FOLDER, ROOT / 'development/aca_resolution_v1'):
        for path in folder.glob('*'):
            if path.is_file() and path.suffix in ('.py', '.json'):
                result[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return result


def build_plan(source_root=SOURCE, cfg=None):
    cfg = cfg or config_from()
    require(Path(source_root).resolve() == SOURCE, 'Only the pinned original source root is accepted')
    train = VideoScope(source_root=SOURCE, split='train')
    validation = VideoScope(source_root=SOURCE, split='validation')
    weights = pin()
    verify_payload(WEIGHTS / weights['local_relative_path'], weights)
    policy = copy.deepcopy(POLICY)
    policy.update(version='aca_full_video_uniform50_256x192_v1',
                  resize='full_frame_320x240_to_256x192_INTER_AREA_no_crop',
                  resolution_height_width=[192, 256])
    return {'schema': SCHEMA, 'config': asdict(cfg), 'architecture': asdict(ReconstructionConfig()),
            'model_class': 'VideoResNet50(variant=aca,num_classes=2)', 'expected_parameters': dense.PARAMETERS,
            'source_root': str(SOURCE), 'input_shape_per_video': [3, 50, 192, 256],
            'input_hashes': train.hashes, 'train_count': len(train.rows), 'validation_count': len(validation.rows),
            'train_ordered_ids_sha256': base.canonical_hash([r['video_id'] for r in train.rows]),
            'validation_ordered_ids_sha256': base.canonical_hash([r['video_id'] for r in validation.rows]),
            'original_source_inventory': base.source_inventory((train, validation)),
            'policy': policy, 'weight_identity': weights, 'runtime': dense.environment(),
            'source_code_sha256': source_hashes(), 'protected_historical_sha256': historical_protection(),
            'initialization': 'fresh_pinned_ImageNet_V1_plus_new_ACA_and_same_seed_paired_binary_head',
            'trained_low_resolution_weights_loaded': False, 'engineering_weights_loaded': False,
            'bn_policy': 'all_BN_affine_and_running_buffers_frozen_including_new_TSCI_BN',
            'optimizer': 'Adam_all_non_BN_parameters_lr1e-4_wd0', 'scheduler': 'none',
            'selection': 'strict_minimum_validation_CE_ties_keep_earlier_epoch',
            'early_stop': 'patience_5_consecutive_nonimproving_validation_CE_epochs',
            'optimizer_steps_per_full_epoch': 450, 'last_effective_batch_videos': 9,
            'resume_policy': 'same_plan_optimizer_boundary_model_Adam_RNG_metrics_cursor_early_stop',
            'paper_exact_reproduction': False, 'author_architecture_verified': False,
            'test_policy': 'reject_before_path_access_no_test_entry',
            'development_after_low_resolution_test_exposure': True,
            'new_test_assessment_requires_separate_user_decision': True,
            'budget_note': '15/5 follows previous project budget, not paper 30; unstarted for user review',
            'formal_training_started': False}


def verify_plan_sources(plan):
    verify_files(plan['protected_historical_sha256'])
    verify_files({str(ROOT / p): digest for p, digest in plan['source_code_sha256'].items()})


def snapshot(output, plan):
    dense.snapshot_sources(output, plan)


def prepare(source_root=SOURCE):
    require(not torch.cuda.is_initialized(), 'CPU prepare must not initialize CUDA')
    base.configure_runtime(config_from().seed, cuda=False)
    plan = build_plan(source_root)
    model, initialization = dense.initialized_candidate(plan['config']['seed'])
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'Nonfinite fresh initialization')
    from development.aca_validation_v1.analyze import tensor_digest
    initial_sha = tensor_digest(model.state_dict())
    del model
    shapes = {}
    with torch.device('meta'):
        model = VideoResNet50(variant='aca', num_classes=2)
        for name in ('conv1', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'):
            getattr(model, name).register_forward_hook(lambda m, x, y, key=name: shapes.__setitem__(key, list(y.shape)))
        logits = model(torch.empty(1, 3, 50, 192, 256))
        require(tuple(logits.shape) == (1, 2) and sum(p.numel() for p in model.parameters()) == dense.PARAMETERS,
                'High-resolution shape/parameter mismatch')
    del model
    artifact_parent().mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='cpu_prepare_', dir=artifact_parent()))
    base.atomic_json(output / 'plan.json', plan)
    snapshot(output, plan)
    report = {'status': 'cpu_prepared_gpu_readiness_pending', 'plan_sha256': base.canonical_hash(plan),
              'initialization': initialization, 'initial_model_state_sha256': initial_sha,
              'meta_shapes': shapes, 'output_shape': [1, 2], 'parameters': dense.PARAMETERS,
              'video_decodes': 0, 'test_videos_opened': 0, 'real_optimizer_updates': 0,
              'formal_training_started': False, 'cuda_initialized': torch.cuda.is_initialized()}
    verify_plan_sources(plan)
    base.atomic_json(output / 'report.json', report)
    return {'output': str(output), **report}


def artifact_parent():
    return FOLDER / 'artifacts'


def validate_readiness(path, plan):
    path = Path(path).resolve()
    require(path.parent.parent == artifact_parent().resolve() and path.parent.name.startswith('gpu_readiness_')
            and path.name == 'report.json', 'Only this module immutable per-run readiness accepted')
    report = json.loads(path.read_text())
    require(report.get('schema') == PREFLIGHT_SCHEMA and report.get('status') == 'passed'
            and report.get('plan_sha256') == base.canonical_hash(plan)
            and report.get('formal_training_started') is False and report.get('engineering_only') is True
            and report.get('engineering_checkpoint_may_initialize_training') is False
            and report.get('synthetic_cuda_resume_bitwise_equal') is True
            and report.get('real_checkpoint_roundtrip_bitwise_equal') is True
            and report.get('all_BN_unchanged') is True and report.get('loader_workers_2_verified') is True
            and report.get('test_videos_opened') == 0, 'Missing/stale/incompatible high-resolution readiness')
    require(base.canonical_hash(json.loads((path.parent / 'plan.json').read_text())) == base.canonical_hash(plan),
            'Readiness plan mirror changed')
    return report


def launch_prerequisites(accepted, confirmed):
    require(accepted and confirmed, 'Formal training needs BOTH --accept-candidate-architecture and --confirm-formal-training')
    gpu_idle()
    require(shutil.disk_usage(FOLDER).free >= 8 * 1024**3, 'At least 8 GiB free disk required')
