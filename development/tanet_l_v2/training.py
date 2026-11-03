"""Shared formal/preflight engine. Author TAM and original preparation stay unchanged."""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import cv2
import numpy as np
import torch
from torch import nn

from .config import FOLDER, InputConfig, load_config, need, sha256, verify_upstream
from .data import VideoScope
from .model import TANetBinary
from .eval_policy import POLICY, per_video_L_stats


def recipe():
    cfg = json.loads((FOLDER/'train_config.json').read_text())
    need(cfg['schema'] == 'tanet_uniform50_L_eval_recipe_v2', 'Unknown training recipe')
    need((cfg['frames'], cfg['height'], cfg['width']) == (50, 192, 256), 'Fixed input required')
    need(cfg['micro_batch'] == 1 and cfg['effective_batch'] == 16, 'Fixed batch recipe required')
    need(cfg['optimizer'] == 'Adam' and cfg['learning_rate'] == 1e-4 and cfg['weight_decay'] == 1e-4,
         'Fixed optimizer recipe required')
    need(cfg['max_epochs'] == 15 and cfg['patience'] == 5 and cfg['min_delta'] == 1e-4,
         'Fixed stopping recipe required')
    need(cfg['precision'] == 'cuda_bfloat16_autocast_FP32_parameters_and_CE' and
         cfg['bn_policy'] == 'pretrained_BN2d_eval_stats_affine_trainable_TAM_BN1d_train', 'Unknown numeric policy')
    need(cfg['test_access'] is False and cfg['automatic_formal_start'] is False, 'No automatic test/start')
    need(cfg['evaluation_policy'] == POLICY, 'Fixed L evaluation policy required')
    return cfg


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def contract():
    cfg, initial = recipe(), load_config()
    need(cfg['seed'] == initial['seed'] and cfg['dropout'] == initial['dropout'], 'Preparation identity changed')
    train, val = VideoScope('train'), VideoScope('validation')
    need(len(train.rows) == 7193 and len(val.rows) == 1547, 'Fixed split counts changed')
    need(not ({r['source_group'] for r in train.rows} & {r['source_group'] for r in val.rows}), 'Source overlap')
    files = ['__init__.py', 'config.py', 'config.json', 'data.py', 'model.py', 'prepare.py', 'training.py', 'train.py',
             'readiness.py', 'eval_policy.py', 'train_config.json', 'upstream/temporal_module.py', 'upstream/PROVENANCE.json',
             'upstream/LICENSE', 'upstream/reference_models.py.txt', 'upstream/__init__.py']
    root = FOLDER.parents[1]
    source = {p: sha256(FOLDER/p) for p in files}
    for p in ['development/data_pipeline/pipeline.py', 'scripts/audit_i3d_temporal_feasibility.py',
              'development/tanet_v1/probe_bn_behavior_v1.py']:
        source['project/'+p] = sha256(root/p)
    identity = {'recipe': cfg, 'initialization': initial, 'data_identity': train.identity,
                'train_ids_sha256': json_hash([r['video_id'] for r in train.rows]),
                'validation_ids_sha256': json_hash([r['video_id'] for r in val.rows]),
                'train_count': len(train.rows), 'validation_count': len(val.rows),
                'source_sha256': source, 'upstream': verify_upstream(),
                'optimizer_steps_per_epoch': math.ceil(len(train.rows)/cfg['effective_batch']),
                'last_optimizer_group_videos': len(train.rows) % cfg['effective_batch']}
    return {**identity, 'contract_sha256': json_hash(identity)}, train, val


def configure_runtime(cfg, *, cuda=True):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    need(os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8', 'Need pinned CUBLAS workspace')
    torch.set_num_threads(cfg['cpu_threads'])
    cv2.setNumThreads(cfg['opencv_threads'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if cuda:
        need(torch.cuda.is_available(), 'CUDA unavailable; no CPU substitute for GPU readiness')
        need(torch.cuda.is_bf16_supported(), 'Native BF16 support required; no silent precision switch')
        torch.cuda.manual_seed_all(cfg['seed'])
    return torch.device('cuda' if cuda else 'cpu')


def initialized_model(cfg, device):
    initial = load_config()
    return TANetBinary(frames=cfg['frames'], dropout=cfg['dropout'],
        checkpoint=initial['imagenet_checkpoint'], checkpoint_sha256=initial['imagenet_sha256'],
        checkpoint_size=initial['imagenet_size_bytes']).to(device)


def training_mode(model):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()  # Running statistics frozen; affine remains trainable.


def optimizer_for(model, cfg):
    return torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])


def autocast(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def collate(samples):
    return {'tensor': torch.stack([s['tensor'] for s in samples]),
            'binary_label': torch.tensor([s['binary_label'] for s in samples], dtype=torch.int64),
            'video_id': [s['video_id'] for s in samples]}


def move_batch(batch, device):
    x, y = batch['tensor'], batch['binary_label']
    need(x.dtype == torch.float32 and torch.isfinite(x).all().item(), 'Invalid normalized RGB tensor')
    need(y.dtype == torch.int64 and y.shape == (len(x),) and set(y.tolist()) <= {0, 1}, 'Invalid CE labels')
    return x.to(device), y.to(device)


def train_group(model, optimizer, batches, *, device, group_videos, cfg, progress=None):
    """One update, sum CE / ACTUAL group size. Also used by GPU resume probe."""
    need(type(group_videos) is int and group_videos > 0, 'Invalid update group')
    training_mode(model)
    optimizer.zero_grad(set_to_none=True)
    seen, loss_sum, last_logits = 0, 0., None
    micro_times = []
    for batch in batches:
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.monotonic()
        x, y = move_batch(batch, device)
        need(seen+len(y) <= group_videos, 'Accumulation group overflow')
        with autocast(device):
            logits = model(x)
        need(logits.shape == (len(y), 2) and torch.isfinite(logits).all().item(), 'Invalid training output')
        loss = nn.functional.cross_entropy(logits.float(), y, reduction='sum')
        need(torch.isfinite(loss).item(), 'Nonfinite training loss')
        (loss/group_videos).backward()
        seen += len(y)
        loss_sum += float(loss.detach())
        last_logits = logits.detach().cpu().float().tolist()
        del x, y, logits, loss
        if device.type == 'cuda':
            torch.cuda.synchronize()
        micro_times.append(time.monotonic()-start)
        if progress is not None:
            progress(seen, group_videos)
    need(seen == group_videos, 'Incomplete accumulation group')
    parameters = [p for p in model.parameters() if p.requires_grad]
    need(all(p.grad is not None for p in parameters), 'Missing trainable gradient')
    need(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all().item(), 'Nonfinite gradient')
    norm = nn.utils.clip_grad_norm_(parameters, cfg['max_grad_norm'], error_if_nonfinite=True)
    start = time.monotonic()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    need(torch.stack([torch.isfinite(p).all() for p in parameters]).all().item(), 'Nonfinite updated parameter')
    return {'CE': loss_sum/seen, 'videos': seen, 'gradient_norm_before_clipping': float(norm),
            'microstep_seconds': micro_times, 'optimizer_seconds': time.monotonic()-start,
            'last_logits': last_logits}


@torch.no_grad()
def eval_batch(model, batch, device):
    x, y = move_batch(batch, device)
    need(len(y) == 1, 'L evaluation requires exactly one video per batch')
    model.eval()
    with per_video_L_stats(model), autocast(device):
        logits = model(x)
    need(logits.shape == (len(y), 2) and torch.isfinite(logits).all().item(), 'Invalid evaluation logits')
    return logits.detach().cpu().float(), y.cpu()


def metrics(logits, labels):
    need(logits.ndim == 2 and logits.shape == (len(labels), 2) and len(labels) > 0, 'Invalid metric shapes')
    need(torch.isfinite(logits).all().item() and labels.dtype == torch.int64 and
         set(labels.tolist()) <= {0, 1}, 'Invalid metric values')
    prediction = logits.argmax(1)
    cm = torch.bincount(2*labels+prediction, minlength=4).reshape(2, 2)
    f1 = []
    for k in (0, 1):
        tp = int(cm[k, k])
        denominator = int(cm[k, :].sum()+cm[:, k].sum())
        f1.append(2*tp/denominator if denominator else 0.)
    return {'CE': float(nn.functional.cross_entropy(logits.float(), labels)),
            'accuracy': float((prediction == labels).float().mean()), 'macro_f1': sum(f1)/2,
            'per_class_f1': f1, 'confusion_matrix_Miss_Make': cm.tolist(), 'videos': len(labels)}


def stopping(validation_ce, state, cfg):
    """Actual minimum picks best.pt; significant improvement resets patience."""
    need(math.isfinite(validation_ce), 'Nonfinite validation CE')
    result = dict(state)
    selected = validation_ce < state['best_ce']
    if selected:
        result['best_ce'] = validation_ce
    if validation_ce < state['early_anchor']-cfg['min_delta']:
        result.update(early_anchor=validation_ce, wait=0)
    else:
        result['wait'] += 1
    return result, selected, result['wait'] >= cfg['patience']


def capture_rng():
    n = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'python': random.getstate(),
            'numpy': [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])]}


def restore_rng(state):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def state_digest(value):
    digest = hashlib.sha256()
    def walk(v):
        if torch.is_tensor(v):
            t = v.detach().cpu().contiguous()
            digest.update(f'tensor:{t.dtype}:{list(t.shape)}:'.encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=str):
                digest.update(repr(key).encode())
                walk(v[key])
        elif isinstance(v, (list, tuple)):
            digest.update(f'{type(v).__name__}:{len(v)}:'.encode())
            for item in v:
                walk(item)
        else:
            digest.update(repr(v).encode())
    walk(value)
    return digest.hexdigest()


def atomic_save(path, value, *, json_format=False):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.pending_', dir=path.parent)
    try:
        with os.fdopen(fd, 'w' if json_format else 'wb') as stream:
            if json_format:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n')
            else:
                torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(model, optimizer, *, contract_sha256, kind, epoch, global_step, stop_state, history):
    need(kind in ('engineering_only', 'formal'), 'Unknown checkpoint purpose')
    return {'schema': 'tanet_training_checkpoint_v1', 'kind': kind,
            'contract_sha256': contract_sha256, 'epoch': epoch, 'global_step': global_step,
            'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
            'rng': capture_rng(), 'stop_state': stop_state, 'history': history,
            'runtime': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name() if torch.cuda.is_initialized() else None}}


def load_checkpoint(path, model, optimizer, *, contract_sha256, kind):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    need(payload.get('schema') == 'tanet_training_checkpoint_v1' and payload.get('kind') == kind,
         'Checkpoint purpose mismatch; engineering weights cannot initialize formal training')
    need(payload['contract_sha256'] == contract_sha256, 'Checkpoint source/data/recipe mismatch')
    need(payload['runtime']['torch'] == str(torch.__version__) and payload['runtime']['cuda'] == torch.version.cuda,
         'Resume runtime mismatch')
    if torch.cuda.is_initialized():
        need(payload['runtime']['gpu'] == torch.cuda.get_device_name(), 'Resume GPU mismatch')
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    restore_rng(payload['rng'])
    return payload
