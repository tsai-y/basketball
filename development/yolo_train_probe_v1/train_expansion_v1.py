"""Human-only, source-separated tiny detector development pilot; not formal evaluation."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import math
import os
import shutil
import tempfile
import traceback

import cv2
import yaml

from .probe import (ARTIFACTS, REPO, SEED, check_dataset as check_old,
                    read, require, save, sha, within, xyxy_to_yolo)
from development.yolo_audit_v1.target_label_pilot import CLASS_NAMES, PROTOCOL, validate_review

SCHEMA = 'human_source_separated_yolo_tiny_pilot_v1'


def check_partitions(records, assignments):
    require(records and len({r['image_id'] for r in records}) == len(records), 'Duplicate image IDs')
    for r in records:
        require(r['source_split'] == 'train', 'Original classification train only')
        require(r['partition'] in ('train', 'val'), 'Invalid detector partition')
        expected = 'detector_train' if r['partition'] == 'train' else 'detector_dev_validation'
        require(assignments.get(r['source_group']) == expected, 'Reserved source assignment violated')
        require(r['annotations'] and r['human_full_image_confirmation']['whole_image_confirmed'] is True,
                'Complete explicit human approval required')
        actual = Counter(a['class_id'] for a in r['annotations'])
        require(r['human_full_image_confirmation']['accepted_counts'] ==
                {n: actual[i] for i, n in enumerate(CLASS_NAMES)}, 'Human confirmation count mismatch')
    for field in ('source_group', 'video_id', 'image_sha256'):
        train = {r[field] for r in records if r['partition'] == 'train'}
        val = {r[field] for r in records if r['partition'] == 'val'}
        require(train and val and train.isdisjoint(val), 'Train/val overlap: '+field)
    for split in ('train', 'val'):
        require(all(any(a['class_id'] == c for r in records if r['partition'] == split
                        for a in r['annotations']) for c in range(3)), 'Each split requires all three classes')


def collect_expansion(batch):
    batch = Path(batch).resolve()
    require(batch.is_relative_to(ARTIFACTS.resolve()), 'Expansion must be local artifact')
    paths = [within(batch, x) for x in ('manifest.json', 'draft_review.json',
             'guided/guided_review.json', 'split_plan.json')]
    manifest, draft, review, plan = map(read, paths)
    require(manifest['schema'] == 'yolo_annotation_expansion_v1' and
            manifest['protocol'] == PROTOCOL and manifest['classes'] == CLASS_NAMES, 'Wrong annotation protocol')
    require(plan['schema'] == 'yolo_expansion_source_partition_v1', 'Wrong split plan')
    status = validate_review(draft, review)
    protected = {str(p): sha(p) for p in paths}
    records = []
    for iid in status['training_ready_ids']:
        row = next(r for r in review['images'] if r['image_id'] == iid)
        meta = next(r for r in manifest['images'] if r['image_id'] == iid)
        source = next(r for r in plan['records'] if r['image_id'] == iid)
        require(source['original_split'] == 'train' and source['image_sha256'] == meta['image_sha256'],
                'Non-train source or foreign source identity')
        expected = plan['source_assignments'].get(source['source_group'])
        require(expected == source['detector_partition'], 'Source reservation changed')
        require(expected in ('detector_train', 'detector_dev_validation'), 'Unknown source assignment')
        events = [e for e in review.get('human_full_image_confirmation_events', []) if e['image_id'] == iid]
        require(events and events[-1].get('whole_image_confirmed') is True and
                bool(events[-1].get('user_reply')), 'No explicit human whole-image confirmation')
        event = events[-1]
        snapshot = within(batch/'guided', event['draft_path'])
        overlay = within(batch, event['overlay_path'])
        require(sha(snapshot) == event['draft_sha256'] and sha(overlay) == event['overlay_sha256'],
                'Confirmed annotation/display evidence changed')
        proposed = read(snapshot)['annotations']
        rejected = set(event.get('rejected_annotation_ids', []))
        require({(a['id'], a['class_id'], tuple(a['xyxy'])) for a in proposed if a['id'] not in rejected} ==
                {(a['id'], a['class_id'], tuple(a['xyxy'])) for a in row['annotations']},
                'Geometry differs from human-confirmed immutable draft')
        image = within(batch, meta['image_path'])
        require(sha(image) == meta['image_sha256'], 'Native original image changed')
        for p in (snapshot, overlay, image): protected[str(p)] = sha(p)
        records.append({'image_id': iid, 'original_image': str(image), 'image_sha256': sha(image),
            'source_split': 'train', 'source_group': source['source_group'], 'video_id': source['source_video_id'],
            'original_class': source['original_class'], 'frame_index': source['frame_index'],
            'partition': 'train' if expected == 'detector_train' else 'val',
            'annotations': deepcopy(row['annotations']), 'human_full_image_confirmation': deepcopy(event)})
    return records, protected, dict(zip(('manifest', 'draft', 'review', 'split_plan'), (manifest, draft, review, plan)))


def prepare(batch, previous):
    previous = Path(previous).resolve();old = check_old(previous)
    records, protected, inputs = collect_expansion(batch)
    weights = Path(old['weights']).resolve()
    require(weights.is_relative_to(REPO) and weights.is_file() and sha(weights) == old['weights_sha256'],
            'Original local COCO initializer missing/changed')
    protected.update(old['protected_source_sha256'])
    protected[str(previous/'dataset_manifest.json')] = sha(previous/'dataset_manifest.json')
    for old_record in old['records']:
        record = deepcopy(old_record)
        image = within(previous, record['image'])
        record.update(partition='train', original_image=str(image))
        records.append(record)
        protected[str(image)] = sha(image)
    assignments = inputs['split_plan']['source_assignments']
    check_partitions(records, assignments)
    require(Counter(r['partition'] for r in records) == {'train': 11, 'val': 3},
            'This frozen pilot requires exactly 11 train and 3 development-validation images')
    output = Path(tempfile.mkdtemp(prefix='human14_sourcepilot_', dir=ARTIFACTS))
    for name in ('images/train', 'images/val', 'labels/train', 'labels/val', 'inputs'):
        (output/name).mkdir(parents=True)
    files = {}
    for name, value in {**inputs, 'previous_dataset': old}.items():
        save(output/'inputs'/f'{name}.json', value)
    for r in records:
        pixels = cv2.imread(r['original_image'])
        require(pixels is not None and pixels.shape == (240, 320, 3), 'Wrong native image shape')
        image = output/f'images/{r["partition"]}/{r["image_id"]}.png'
        label = output/f'labels/{r["partition"]}/{r["image_id"]}.txt'
        shutil.copy2(r['original_image'], image)
        lines = [xyxy_to_yolo(a['class_id'], a['xyxy']) for a in r['annotations']]
        require(len(lines) == len(set(lines)), 'Duplicate labels')
        label.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        require(sha(image) == r['image_sha256'], 'Lossless image copy failed')
        r.update(image=str(image.relative_to(output)), label=str(label.relative_to(output)))
        for p in (image, label): files[str(p.relative_to(output))] = sha(p)
    config = {'path': str(output), 'train': 'images/train', 'val': 'images/val',
              'names': dict(enumerate(CLASS_NAMES))}
    (output/'data.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    for p in [output/'data.yaml', *(output/'inputs').iterdir()]: files[str(p.relative_to(output))] = sha(p)
    counts = {split: dict(Counter(CLASS_NAMES[a['class_id']] for r in records if r['partition'] == split
                                 for a in r['annotations'])) for split in ('train', 'val')}
    report = {'schema': SCHEMA, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': PROTOCOL, 'classes': CLASS_NAMES, 'records': records, 'image_counts': {'train': 11, 'val': 3},
        'object_counts': counts, 'source_assignments': assignments, 'dataset_file_sha256': files,
        'protected_source_sha256': protected, 'weights': str(weights), 'weights_sha256': sha(weights),
        'initializer': 'original_local_COCO_yolo11m_not_five_image_overfit',
        'train_val_source_disjoint': True, 'test_used': False, 'original_split_changed': False,
        'AI_only_candidates_exported': False, 'uncertain_images_exported': False,
        'generalization_metrics_valid': False, 'validation_is_training_image_resubstitution': False,
        'scope': 'tiny_source_separated_development_pilot_not_formal_evaluation',
        'limitations': ['Only 3 development images from 2 source groups', 'Selection and labels were development-reviewed',
                       'Missing new 2p/3p training additions and reserved 2p/3p development sources',
                       'No automatic full-video ROI/filtering authorization'],
        'training_started': False}
    save(output/'dataset_manifest.json', report)
    check_dataset(output)
    return output


def check_dataset(output):
    output = Path(output).resolve()
    require(output.is_relative_to(ARTIFACTS.resolve()), 'Not a local pilot dataset')
    m = read(output/'dataset_manifest.json')
    require(m['schema'] == SCHEMA and m['protocol'] == PROTOCOL and m['classes'] == CLASS_NAMES, 'Wrong pilot schema')
    require(m['test_used'] is False and m['AI_only_candidates_exported'] is False and
            m['uncertain_images_exported'] is False, 'Unsafe export flags')
    check_partitions(m['records'], m['source_assignments'])
    for p, expected in m['dataset_file_sha256'].items():
        require(sha(within(output, p)) == expected, 'Dataset changed: '+p)
    for p, expected in m['protected_source_sha256'].items():
        require(sha(p) == expected, 'Protected source changed: '+p)
    require(sha(m['weights']) == m['weights_sha256'], 'Initializer changed')
    config = yaml.safe_load((output/'data.yaml').read_text())
    require(config == {'path': str(output), 'train': 'images/train', 'val': 'images/val',
                       'names': dict(enumerate(CLASS_NAMES))}, 'Changed data config')
    for split in ('train', 'val'):
        expected_ids = {r['image_id'] for r in m['records'] if r['partition'] == split}
        require(len(expected_ids) == m['image_counts'][split], 'Wrong image count')
        require({p.name for p in (output/f'images/{split}').iterdir()} == {i+'.png' for i in expected_ids}, 'Unexpected images')
        require({p.name for p in (output/f'labels/{split}').iterdir()} == {i+'.txt' for i in expected_ids}, 'Unexpected labels')
    for r in m['records']:
        require(r['image'] == f'images/{r["partition"]}/{r["image_id"]}.png' and
                r['label'] == f'labels/{r["partition"]}/{r["image_id"]}.txt', 'Record escaped split')
        require(sha(within(output, r['image'])) == r['image_sha256'], 'Image identity mismatch')
        require((output/r['label']).read_text().splitlines() ==
                [xyxy_to_yolo(a['class_id'], a['xyxy']) for a in r['annotations']], 'Labels differ from approved geometry')
    return m


def training_args(output, run, epochs, smoke):
    require(type(epochs) is int and 1 <= epochs <= 100, 'Bounded run: 1..100 epochs')
    require(not smoke or epochs == 1, 'Smoke must be one epoch')
    return dict(data=str(Path(output)/'data.yaml'), epochs=epochs, imgsz=640, batch=2, nbs=2,
        device=0, workers=0, seed=SEED, deterministic=True, optimizer='AdamW', lr0=.0001,
        lrf=.1, weight_decay=.0005, warmup_epochs=3., warmup_bias_lr=.0001,
        freeze=0, patience=0 if smoke else 20,
        amp=False, compile=False, pretrained=True, val=True, plots=False, cache=False,
        mosaic=0., mixup=0., cutmix=0., copy_paste=0., degrees=0., translate=0., scale=0.,
        shear=0., perspective=0., flipud=0., fliplr=0., hsv_h=0., hsv_s=0., hsv_v=0.,
        bgr=0., multi_scale=0., close_mosaic=0, save=True, save_period=-1,
        project=str(run), name='fit', exist_ok=False, verbose=False)


def new_training_epoch(epoch, completed, observed_training_epochs):
    """Final best-checkpoint validation increments trainer.epoch without training."""
    return epoch in observed_training_epochs and epoch+1 > completed


def train(output, *, epochs=100, smoke=False):
    output = Path(output).resolve();m = check_dataset(output)
    # Validate bounds before creating a run or loading a model.
    training_args(output, output/'unused', epochs, smoke)
    import torch
    require(torch.cuda.is_available(), 'CUDA required; do not silently train on CPU')
    run = Path(tempfile.mkdtemp(prefix='smoke_' if smoke else 'pilot_', dir=output))
    runtime = run/'runtime';runtime.mkdir()
    os.environ.update(YOLO_CONFIG_DIR=str(runtime), MPLCONFIGDIR=str(runtime/'matplotlib'),
        YOLO_AUTOINSTALL='false', HF_HUB_OFFLINE='1', YOLO_OFFLINE='true', WANDB_MODE='disabled',
        COMET_MODE='DISABLED', CLEARML_OFFLINE_MODE='1')
    font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');require(font.is_file(), 'Local font missing')
    shutil.copy2(font, runtime/'Arial.ttf')
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS, USER_CONFIG_DIR
    Path(USER_CONFIG_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy2(font, Path(USER_CONFIG_DIR)/'Arial.ttf')
    SETTINGS.update({k: False for k in ('sync', 'hub', 'wandb', 'mlflow', 'clearml', 'comet',
        'neptune', 'tensorboard', 'dvc', 'raytune') if k in SETTINGS})
    torch.set_num_threads(8);torch.cuda.reset_peak_memory_stats()
    args = training_args(output, run, epochs, smoke)
    log = {'schema': SCHEMA, 'status': 'starting', 'stage': 'smoke' if smoke else 'tiny_development_training',
        'run_dir': str(run), 'dataset': str(output), 'training_args': args,
        'image_counts': m['image_counts'], 'object_counts': m['object_counts'],
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__, 'GPU': torch.cuda.get_device_name(0),
        'initializer_sha256': m['weights_sha256'], 'dataset_manifest_sha256': sha(output/'dataset_manifest.json'),
        'source_code_sha256': sha(__file__), 'train_val_source_disjoint': True, 'test_used': False,
        'generalization_metrics_valid': False, 'epochs_completed': 0,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint_selection': 'Ultralytics best validation fitness on 3 development images; also retain last',
        'metric_warning': 'Tiny development validation, 3 images/2 sources; not reliable generalization or deployment evidence'}
    save(run/'run_state.json', log)
    print('RUN_DIR:', run, flush=True)
    print('11 train / 3 development images; source-disjoint, engineering pilot only.', flush=True)
    initial = {};observed_training_epochs = set();model = YOLO(m['weights'])

    def on_start(trainer):
        for split, loader in [('train', trainer.train_loader), ('val', trainer.test_loader)]:
            expected = {str((output/r['image']).resolve()) for r in m['records'] if r['partition'] == split}
            require({str(Path(p).resolve()) for p in loader.dataset.im_files} == expected,
                    'Actual loader differs from human-only split allowlist: '+split)
        require(trainer.data['names'] == dict(enumerate(CLASS_NAMES)), 'Wrong detector class map')
        params = [(n,p) for n,p in trainer.model.named_parameters() if p.requires_grad]
        for name, parameter in (params[0], params[-1]): initial[name] = parameter.detach().cpu().clone()
        log.update(status='training', actual_loader_allowlists_checked=True)
        save(run/'run_state.json', log)

    def batch_end(trainer):
        require(torch.isfinite(trainer.loss_items).all().item(), 'Non-finite training loss')

    def train_epoch_end(trainer):
        observed_training_epochs.add(trainer.epoch)

    def epoch_end(trainer):
        if not new_training_epoch(trainer.epoch, log['epochs_completed'], observed_training_epochs): return
        params = dict(trainer.model.named_parameters())
        deltas = {n:float((params[n].detach().cpu()-v).abs().max()) for n,v in initial.items()}
        values = [float(v) for v in trainer.tloss.detach().cpu()]
        require(all(math.isfinite(v) for v in values), 'Non-finite epoch loss')
        log.update(epochs_completed=trainer.epoch+1, latest_training_losses=values,
            parameter_max_abs_updates=deltas, latest_development_metrics={k:float(v) for k,v in trainer.metrics.items()},
            peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3)
        save(run/'run_state.json', log)
        print('PILOT_EPOCH:', trainer.epoch+1, 'losses:', values, flush=True)

    model.add_callback('on_train_start', on_start)
    model.add_callback('on_train_batch_end', batch_end)
    model.add_callback('on_train_epoch_end', train_epoch_end)
    model.add_callback('on_fit_epoch_end', epoch_end)
    try:
        model.train(**args)
        require(any(v > 0 for v in log['parameter_max_abs_updates'].values()), 'No parameter update')
        checkpoints = {}
        for name in ('last', 'best'):
            path = run/f'fit/weights/{name}.pt';require(path.is_file(), 'Missing '+name+' checkpoint')
            loaded = YOLO(str(path));require(loaded.names == dict(enumerate(CLASS_NAMES)), 'Reload class mismatch')
            samples = [str(output/r['image']) for r in m['records'] if r['partition'] == 'val']
            results = loaded.predict(source=samples, imgsz=640, device=0, conf=.1, save=False, verbose=False)
            require(len(results) == len(samples), 'Reload inference count mismatch')
            require(all(torch.isfinite(x.boxes.xyxy).all().item() for x in results), 'Non-finite reload output')
            checkpoints[name] = {'path': str(path), 'sha256': sha(path)}
        log.update(status='completed', checkpoint_round_trip=True, checkpoints=checkpoints,
            stop_reason='early_stopping' if log['epochs_completed'] < epochs else 'epoch_limit',
            completed_at_utc=datetime.now(timezone.utc).isoformat())
        save(run/'run_state.json', log)
    except BaseException as exc:
        log.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        save(run/'run_state.json', log)
        raise
    finally: check_dataset(output)
    print('COMPLETED:', run, flush=True)
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'check', 'train'): modes.add_argument('--'+name, action='store_true')
    p.add_argument('--batch');p.add_argument('--previous-dataset');p.add_argument('--dataset-dir')
    p.add_argument('--epochs', type=int, default=100);p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.prepare:
        require(a.batch and a.previous_dataset, 'Need batch and previous dataset')
        print(prepare(a.batch, a.previous_dataset))
    elif a.check:
        m = check_dataset(a.dataset_dir);print(m['image_counts'], m['object_counts'])
    else: train(a.dataset_dir, epochs=a.epochs, smoke=a.smoke)


if __name__ == '__main__': main()
