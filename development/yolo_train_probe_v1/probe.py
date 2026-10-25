"""Train-only, human-complete YOLO small-overfit probe, not an efficacy experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import traceback

import cv2
import yaml

from development.yolo_audit_v1.target_label_pilot import (
    CLASS_NAMES, PROTOCOL, validate_box, validate_review,
)

FOLDER = Path(__file__).resolve().parent
REPO = FOLDER.parents[1]
AUDITS = REPO / 'development/yolo_audit_v1/artifacts'
ARTIFACTS = FOLDER / 'artifacts'
SEED = 20261025


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within(root, relative):
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(), 'Missing/outside source file')
    return path


def xyxy_to_yolo(class_id, box, width=320, height=240):
    require(type(class_id) is int and class_id in range(3), 'Invalid class ID')
    validate_box(box)
    x1, y1, x2, y2 = box
    values = ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)
    require(all(math.isfinite(v) and 0 < v <= 1 for v in values), 'Invalid normalized label')
    return str(class_id) + ' ' + ' '.join(f'{v:.9f}' for v in values)


def select_human_complete(manifest, draft, review):
    result = validate_review(draft, review)
    require(manifest['protocol'] == PROTOCOL and manifest['classes'] == CLASS_NAMES, 'Protocol mismatch')
    require(manifest['pilot_id'] == review['pilot_id'], 'Foreign pilot')
    videos = {s['sample_id']: s for s in manifest['source_videos']}
    require(videos and all(s['split'] == 'train' and str(s['excluded']) == '0'
                           for s in videos.values()), 'Only non-excluded original train sources allowed')
    metas = {m['image_id']: m for m in manifest['images']}
    selected = []
    for iid in result['training_ready_ids']:
        row = next(r for r in review['images'] if r['image_id'] == iid)
        meta = metas[iid]
        source = videos[meta['audit_sample_id']]
        require(meta['image_sha256'] == row['image_sha256'], 'Image SHA mismatch')
        events = [e for e in review.get('human_full_image_confirmation_events', []) if e['image_id'] == iid]
        require(bool(events) and events[-1].get('whole_image_confirmed') is True,
                'A reviewed row without explicit whole-image human approval cannot be exported')
        actual = Counter(a['class_id'] for a in row['annotations'])
        require(events[-1]['accepted_counts'] == {name: actual[i] for i, name in enumerate(CLASS_NAMES)},
                'Latest whole-image confirmation disagrees with current objects')
        selected.append({'row': deepcopy(row), 'meta': deepcopy(meta), 'source': deepcopy(source),
                         'human_confirmation': deepcopy(events[-1])})
    require(len(selected) >= 2, 'Need at least two fully human-confirmed images for the engineering probe')
    require(all(any(a['class_id'] == c for s in selected for a in s['row']['annotations']) for c in range(3)),
            'All three object classes are required')
    return selected


def prepare(pilot, weights):
    pilot, weights = Path(pilot).resolve(), Path(weights).resolve()
    require(pilot.is_relative_to(AUDITS.resolve()), 'Pilot must be an existing local audit artifact')
    require(weights.is_file() and weights.is_relative_to(REPO), 'Use an existing local repository checkpoint')
    manifest_path = within(pilot, 'manifest.json')
    draft_path = within(pilot, 'draft_review.json')
    review_path = within(pilot, 'guided/guided_review.json')
    manifest, draft, review = map(read, (manifest_path, draft_path, review_path))
    selected = select_human_complete(manifest, draft, review)
    protected = {str(p): sha(p) for p in (manifest_path, draft_path, review_path, weights)}
    for meta in manifest['images']:
        image = within(pilot, meta['image_path'])
        require(sha(image) == meta['image_sha256'], 'Original image changed')
        protected[str(image)] = sha(image)
    for record in selected:
        e = record['human_confirmation']
        p = within(pilot / 'guided', e['draft_path'])
        require(sha(p) == e['draft_sha256'], 'Human-confirmed draft changed')
        protected[str(p)] = sha(p)
    ARTIFACTS.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='human5_overfit_dataset_', dir=ARTIFACTS))
    for name in ('images/train', 'labels/train', 'inputs'):
        (output / name).mkdir(parents=True)
    # Immutable snapshots: future annotation work cannot silently change this run's labels.
    for name, value in [('review_snapshot.json', review), ('pilot_manifest_snapshot.json', manifest),
                        ('draft_snapshot.json', draft)]:
        save(output / 'inputs' / name, value)
    records, counts, files = [], Counter(), {}
    for item in selected:
        row, meta, source = item['row'], item['meta'], item['source']
        iid = row['image_id']; original = within(pilot, meta['image_path'])
        pixels = cv2.imread(str(original))
        require(pixels is not None and pixels.shape == (240, 320, 3), 'Unexpected native image')
        image = output / f'images/train/{iid}.png'
        label = output / f'labels/train/{iid}.txt'
        shutil.copy2(original, image)
        lines = [xyxy_to_yolo(a['class_id'], a['xyxy']) for a in row['annotations']]
        require(len(lines) == len(set(lines)), 'Duplicate class/geometry labels')
        label.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        require(sha(image) == meta['image_sha256'], 'Image copy must be lossless/bitwise-identical')
        for a in row['annotations']:
            counts[a['class_id']] += 1
        records.append({'image_id': iid, 'image': str(image.relative_to(output)),
            'label': str(label.relative_to(output)), 'image_sha256': sha(image), 'label_sha256': sha(label),
            'source_split': 'train', 'source_group': source['source_group'], 'video_id': source['video_id'],
            'audit_sample_id': meta['audit_sample_id'], 'frame_index': meta['frame_index'],
            'annotations': row['annotations'], 'human_full_image_confirmation': item['human_confirmation']})
        files[str(image.relative_to(output))] = sha(image)
        files[str(label.relative_to(output))] = sha(label)
    data = {'path': str(output), 'train': 'images/train', 'val': 'images/train',
            'names': {i: name for i, name in enumerate(CLASS_NAMES)}}
    (output / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    files['data.yaml'] = sha(output / 'data.yaml')
    for p in (output / 'inputs').iterdir():
        files[str(p.relative_to(output))] = sha(p)
    report = {'schema': 'human_complete_yolo_overfit_dataset_v1', 'pilot_id': review['pilot_id'],
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol': PROTOCOL,
        'experiment_scope': 'small_overfit_engineering_only', 'independent_validation': False,
        'train_val_overlap_expected': True, 'validation_is_training_image_resubstitution': True,
        'generalization_metrics_valid': False, 'test_used': False, 'original_split_changed': False,
        'source_split': 'train', 'image_count': len(records), 'records': records,
        'object_counts': {name: counts[i] for i, name in enumerate(CLASS_NAMES)},
        'source_groups': sorted({r['source_group'] for r in records}),
        'excluded_pending_ids': [r['image_id'] for r in review['images'] if r['status'] != 'reviewed'],
        'optional_holding_ball_roles_exported': False, 'AI_only_candidates_exported': False,
        'weights': str(weights), 'weights_sha256': sha(weights), 'protected_source_sha256': protected,
        'dataset_file_sha256': files, 'training_started': False,
        'authorization': 'User requested verification followed by YOLO training; this exports only complete human-reviewed frames for an engineering overfit probe',
        'prior_supplemental_frame_question': review.get('current_question')}
    save(output / 'dataset_manifest.json', report)
    check_dataset(output)
    return output


def check_dataset(output):
    output = Path(output).resolve()
    require(output.is_relative_to(ARTIFACTS.resolve()), 'Not a local probe dataset')
    m = read(output / 'dataset_manifest.json')
    require(m['schema'] == 'human_complete_yolo_overfit_dataset_v1'
            and m['experiment_scope'] == 'small_overfit_engineering_only', 'Not an overfit-only dataset')
    require(m['source_split'] == 'train' and m['test_used'] is False and m['AI_only_candidates_exported'] is False,
            'Original-train/human-only guard failed')
    for name, expected in m['dataset_file_sha256'].items():
        require(sha(within(output, name)) == expected, 'Dataset file changed: ' + name)
    for p, expected in m['protected_source_sha256'].items():
        require(sha(p) == expected, 'Protected source changed: ' + p)
    data = yaml.safe_load((output / 'data.yaml').read_text())
    require(data == {'path': str(output), 'train': 'images/train', 'val': 'images/train',
                     'names': {i: name for i, name in enumerate(CLASS_NAMES)}}, 'Unexpected data config')
    expected_ids = {r['image_id'] for r in m['records']}
    require({p.stem for p in (output / 'images/train').glob('*.png')} == expected_ids, 'Unexpected images')
    require({p.stem for p in (output / 'labels/train').glob('*.txt')} == expected_ids, 'Unexpected labels')
    for r in m['records']:
        require(r['source_split'] == 'train', 'Non-train record')
        lines = (output / r['label']).read_text().splitlines()
        require(len(lines) == len(r['annotations']), 'Lost objects')
        for line, a in zip(lines, r['annotations']):
            require(line == xyxy_to_yolo(a['class_id'], a['xyxy']), 'Changed class or geometry')
    return m


def train(output, *, epochs=150, smoke=False):
    output = Path(output).resolve();m = check_dataset(output)
    require(type(epochs) is int and 1 <= epochs <= 300, 'Bounded engineering run: 1..300 epochs')
    require(not smoke or epochs == 1, 'Smoke stage must be exactly one epoch')
    run = Path(tempfile.mkdtemp(prefix='smoke_' if smoke else 'overfit_', dir=output))
    runtime = run / 'runtime';runtime.mkdir()
    os.environ['YOLO_CONFIG_DIR'] = str(runtime)
    os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['COMET_MODE'] = 'DISABLED'
    os.environ['CLEARML_OFFLINE_MODE'] = '1'
    # Local ASCII fallback avoids a font download; plots are disabled for training.
    font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    require(font.is_file(), 'Need an existing local fallback font')
    shutil.copy2(font, runtime / 'Arial.ttf')
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS, USER_CONFIG_DIR
    # Installed 8.4.58 appends /Ultralytics to YOLO_CONFIG_DIR.
    Path(USER_CONFIG_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy2(font, Path(USER_CONFIG_DIR) / 'Arial.ttf')
    require(torch.cuda.is_available(), 'GPU unavailable: do not silently start slow CPU training')
    torch.set_num_threads(8)
    SETTINGS.update({k: False for k in ('sync', 'hub', 'wandb', 'mlflow', 'clearml', 'comet',
        'neptune', 'tensorboard', 'dvc', 'raytune') if k in SETTINGS})
    args = dict(data=str(output / 'data.yaml'), epochs=epochs, imgsz=640, batch=2, nbs=2,
        device=0, workers=0, seed=SEED, deterministic=True, optimizer='AdamW', lr0=.001,
        lrf=.1, weight_decay=0., warmup_epochs=0., freeze=0, patience=0,
        amp=False, compile=False, pretrained=True, val=True, plots=False, cache=False,
        mosaic=0., mixup=0., cutmix=0., copy_paste=0., degrees=0., translate=0., scale=0.,
        shear=0., perspective=0., flipud=0., fliplr=0., hsv_h=0., hsv_s=0., hsv_v=0.,
        bgr=0., multi_scale=0., close_mosaic=0, save=True, save_period=50 if not smoke else -1,
        project=str(run), name='fit', exist_ok=False, verbose=False)
    log = {'schema': 'YOLO_small_overfit_training_v1', 'status': 'starting',
        'stage': 'smoke' if smoke else 'small_overfit', 'run_dir': str(run),
        'experiment_scope': 'engineering_only_not_generalization', 'dataset': str(output),
        'image_count': m['image_count'], 'object_counts': m['object_counts'],
        'independent_validation': False, 'validation_is_training_image_resubstitution': True,
        'generalization_metrics_valid': False, 'test_used': False,
        'ultralytics': ultralytics.__version__, 'torch': torch.__version__,
        'GPU': torch.cuda.get_device_name(0), 'training_args': args,
        'weights_sha256': m['weights_sha256'], 'dataset_manifest_sha256': sha(output / 'dataset_manifest.json'),
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'epochs_completed': 0,
        'metric_warning': 'All YOLO validation metrics are computed on the same five training frames; NOT held-out performance',
        'local_font_fallback': str(font)}
    save(run / 'run_state.json', log)
    print('RUN_DIR:', run, flush=True)
    print('ENGINEERING ONLY: train == val; no independent accuracy/mAP claim.', flush=True)
    initial = {};model = YOLO(m['weights'])

    def on_start(trainer):
        expected = {str((output / r['image']).resolve()) for r in m['records']}
        for loader in (trainer.train_loader, trainer.test_loader):
            require({str(Path(p).resolve()) for p in loader.dataset.im_files} == expected,
                    'Loader escaped the fixed human-only five-image allowlist')
        require(trainer.data['names'] == {i: name for i, name in enumerate(CLASS_NAMES)}, 'Wrong classes')
        parameters = [(n,p) for n,p in trainer.model.named_parameters() if p.requires_grad]
        for name, parameter in (parameters[0], parameters[-1]):
            initial[name] = parameter.detach().cpu().clone()
        log.update(status='training', first_trainable_parameter=parameters[0][0],
                   last_trainable_parameter=parameters[-1][0])
        save(run / 'run_state.json', log)

    def batch_end(trainer):
        require(torch.isfinite(trainer.loss_items).all().item(), 'Non-finite training loss')

    def epoch_end(trainer):
        # Ultralytics invokes on_fit_epoch_end again during final checkpoint validation.
        if log['epochs_completed'] >= epochs:
            return
        params = dict(trainer.model.named_parameters())
        deltas = {n: float((params[n].detach().cpu()-v).abs().max()) for n,v in initial.items()}
        values = [float(v) for v in trainer.tloss.detach().cpu()]
        require(all(math.isfinite(v) for v in values), 'Non-finite epoch loss')
        log.update(epochs_completed=trainer.epoch+1, latest_training_losses=values,
                   parameter_max_abs_updates=deltas,
                   peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
        save(run / 'run_state.json', log)
        print('PROBE_EPOCH:', trainer.epoch+1, 'losses:', values, flush=True)

    model.add_callback('on_train_start', on_start)
    model.add_callback('on_train_batch_end', batch_end)
    model.add_callback('on_fit_epoch_end', epoch_end)
    try:
        model.train(**args)
        require(any(v>0 for v in log['parameter_max_abs_updates'].values()), 'No parameter update')
        last = run / 'fit/weights/last.pt'
        require(last.is_file(), 'Training did not save last.pt')
        reloaded = YOLO(str(last))
        require(reloaded.names == {i:name for i,name in enumerate(CLASS_NAMES)}, 'Checkpoint class map wrong')
        predictions = reloaded.predict(source=[str(output/r['image']) for r in m['records']],
            imgsz=640, device=0, conf=.10, save=False, verbose=False)
        require(len(predictions)==m['image_count'], 'Checkpoint inference count mismatch')
        for r in predictions:
            require(torch.isfinite(r.boxes.xyxy).all().item(), 'Non-finite checkpoint predictions')
        log.update(status='completed', checkpoint_round_trip=True, checkpoint=str(last),
            checkpoint_sha256=sha(last), completed_at_utc=datetime.now(timezone.utc).isoformat())
        save(run / 'run_state.json', log)
    except BaseException as exc:
        log.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        save(run / 'run_state.json', log)
        raise
    finally:
        # Hash-check protected inputs, not caches created by the framework.
        check_dataset(output)
    print(json.dumps({'run_dir':str(run), 'status':log['status'], 'epochs':log['epochs_completed'],
        'checkpoint':log['checkpoint'], 'generalization_metrics_valid':False}, indent=2), flush=True)
    return run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true');mode.add_argument('--check', action='store_true')
    mode.add_argument('--train', action='store_true')
    parser.add_argument('--pilot-dir');parser.add_argument('--dataset-dir')
    parser.add_argument('--weights',default=str(REPO/'yolo11m.pt'))
    parser.add_argument('--epochs',type=int,default=150);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        require(args.pilot_dir is not None, '--pilot-dir required')
        print(prepare(args.pilot_dir,args.weights))
    elif args.check:
        m=check_dataset(args.dataset_dir)
        print(json.dumps({k:m[k] for k in ('image_count','object_counts','source_groups','generalization_metrics_valid')},indent=2))
    else:
        train(args.dataset_dir,epochs=args.epochs,smoke=args.smoke)


if __name__=='__main__':
    main()
