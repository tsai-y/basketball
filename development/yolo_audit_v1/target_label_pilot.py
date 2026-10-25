"""Prepare 24 train-only object-annotation images; never run YOLO or train.

Uses the unannotated left half of existing lossless audit PNGs. Person detections
are editable proposals, never automatically converted to player ground truth.
"""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from tempfile import mkdtemp

import cv2
import numpy as np

from .role_metadata import ROLE_PROTOCOL, validate_role_metadata


FOLDER = Path(__file__).resolve().parent
AUDIT_ROOT = FOLDER / 'artifacts'
EXCLUDE = {'Y004', 'Y012', 'Y018', 'Y019'}
SEED = 20261024
CLASS_NAMES = ['active_player', 'ball', 'rim']
PROTOCOL = 'active-player-ball-visible-rim-v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_samples(manifest):
    require(manifest.get('split') == 'train', 'Only train audit allowed')
    rows = manifest.get('samples', [])
    require(rows and all(r.get('split') == 'train' and r.get('excluded') == '0' for r in rows),
            'Non-train/excluded sample')
    require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate sample IDs')
    chosen, sources = [], set()
    for shot in ('2p', '3p', 'ft', 'mp'):
        for label in ('0', '1'):
            pool = [r for r in rows if r['sample_id'] not in EXCLUDE
                    and r['original_class'] == shot + label and r['source_group'] not in sources]
            require(bool(pool), 'Missing pilot stratum: ' + shot + label)
            row = min(pool, key=lambda r: hashlib.sha256(
                f'target_label_pilot_v1:{SEED}:{r["video_id"]}'.encode()).hexdigest())
            sources.add(row['source_group'])
            chosen.append(row)
    return chosen


def validate_box(box):
    require(isinstance(box, list) and len(box) == 4, 'Expected xyxy list')
    require(all(type(v) in (int, float) and math.isfinite(v) for v in box), 'Non-numeric/non-finite box')
    x1, y1, x2, y2 = box
    require(0 <= x1 < x2 <= 320 and 0 <= y1 < y2 <= 240, 'Box outside image/degenerate')


def validate_review(draft, review):
    """Validate import and readiness; this does not export a training dataset."""
    require(review.get('schema') == draft['schema'] and review.get('pilot_id') == draft['pilot_id'],
            'Review schema/pilot mismatch')
    require(review.get('classes') == CLASS_NAMES and review.get('protocol') == PROTOCOL,
            'Class mapping/protocol mismatch')
    # Optional roles do not change object classes or detection-completeness rules.
    reference = {r['image_id']: r for r in draft['images']}
    rows = review.get('images', [])
    require(len(rows) == len(reference) and len({r['image_id'] for r in rows}) == len(rows)
            and {r['image_id'] for r in rows} == set(reference), 'Missing/duplicate/unexpected image IDs')
    counts = Counter()
    ready_ids = []
    for row in rows:
        original = reference[row['image_id']]
        require(row.get('image_sha256') == original['image_sha256'], 'Image identity changed')
        require(row.get('status') in ('pending', 'reviewed', 'excluded'), 'Invalid review status')
        require(row.get('scene') in ('unreviewed', 'court_play', 'closeup', 'transition', 'other', 'uncertain')
                and type(row.get('annotation_complete')) is bool and isinstance(row.get('notes'), str),
                'Invalid scene/completeness/notes')
        counts[row['status']] += 1
        annotations = row.get('annotations', [])
        require(isinstance(annotations, list), 'Invalid annotations')
        require(len({a['id'] for a in annotations}) == len(annotations), 'Duplicate annotation IDs')
        for annotation in annotations:
            require(type(annotation['class_id']) is int and 0 <= annotation['class_id'] < 3,
                    'Invalid class ID')
            validate_box(annotation['xyxy'])
        proposed = {p['id']: p for p in original['proposals']}
        states = row.get('proposal_states', {})
        require(set(states) == set(proposed) and all(v in ('pending', 'accepted', 'rejected', 'uncertain')
                                                  for v in states.values()), 'Invalid proposal states')
        for pid, state in states.items():
            linked = [a for a in annotations if a.get('source_proposal_id') == pid]
            require(len(linked) == (1 if state == 'accepted' else 0), 'Proposal/annotation mismatch')
            require(not linked or linked[0]['class_id'] == 0, 'Person proposal may only become player')
        require(all(a.get('source_proposal_id') is None or a['source_proposal_id'] in proposed
                    for a in annotations), 'Unknown proposal reference')
        visibility = row.get('class_visibility', {})
        require(set(visibility) == set(CLASS_NAMES) and all(v in ('unreviewed', 'present', 'absent', 'uncertain')
                                                         for v in visibility.values()), 'Invalid visibility state')
        if row['status'] == 'excluded':
            require(isinstance(row.get('notes'), str) and row['notes'].strip(), 'Excluded image needs reason')
        if row['status'] == 'reviewed':
            require(row.get('scene') == 'court_play', 'Only court-play images can be training-ready')
            require(row.get('annotation_complete') is True, 'Completeness confirmation missing')
            require(all(v in ('accepted', 'rejected') for v in states.values()), 'Unresolved person proposals')
            for class_id, name in enumerate(CLASS_NAMES):
                visible = visibility[name]
                require(visible in ('present', 'absent'), 'Unreviewed/uncertain class blocks readiness')
                amount = sum(a['class_id'] == class_id for a in annotations)
                require((visible == 'present' and amount > 0) or (visible == 'absent' and amount == 0),
                        'Class visibility/box mismatch')
            ready_ids.append(row['image_id'])
    validate_role_metadata(review)
    return {'schema_valid': True, 'images': len(rows), 'status_counts': dict(counts),
            'training_ready_images': len(ready_ids), 'training_ready_ids': ready_ids,
            'yolo_training_labels_exported': False, 'training_started': False}


def prepare(audit_run, *, dry_run=False):
    source = Path(audit_run).resolve()
    require(source.is_relative_to(AUDIT_ROOT.resolve()) and source.is_dir(), 'Not a local YOLO audit run')
    manifest_path = source / 'manifest.json'
    samples = choose_samples(read_json(manifest_path))
    plans, immutable = [], {str(manifest_path): sha(manifest_path)}
    for sample in samples:
        sid = sample['sample_id']
        data_path = source / 'detections' / f'{sid}.json'
        require(data_path.resolve().is_relative_to(source), 'Detection path escapes source')
        data = read_json(data_path)
        require(data['sample_id'] == sid and data['video_id'] == sample['video_id'], 'Detection identity mismatch')
        require(len(data['frames']) == sample['frame_count'], 'Frame metadata mismatch')
        immutable[str(data_path)] = sha(data_path)
        for q in (3, 5, 7):
            index = sample['frame_count'] * q // 10
            png = source / 'frames' / f'{sid}_f{index:06d}.png'
            require(png.resolve().is_relative_to(source) and png.is_file(), 'Missing/outside original audit PNG')
            immutable[str(png)] = sha(png)
            frame = data['frames'][index]
            require(frame['frame_index'] == index, 'Frame index mismatch')
            for det in frame['detections']:
                require(det['class_id'] == 0, 'Non-person proposal')
                validate_box(det['xyxy'])
            plans.append((sample, frame, png))
    if dry_run:
        return {'dry_run': True, 'videos': 8, 'images': len(plans),
                'source_groups': len({s['source_group'] for s in samples}),
                'excluded_development_samples': sorted(EXCLUDE),
                'selected_audit_ids': [s['sample_id'] for s in samples],
                'image_decodes': 0, 'output_files_written': False, 'training_started': False}
    output = Path(mkdtemp(prefix='target_label_pilot_v1_', dir=source / 'outputs'))
    (output / 'images').mkdir()
    images, records = [], []
    for number, (sample, frame, png) in enumerate(plans, 1):
        pair = cv2.imread(str(png), cv2.IMREAD_COLOR)
        require(pair is not None and pair.dtype == np.uint8 and pair.shape == (264, 640, 3), 'Invalid audit PNG')
        image = pair[24:, :320].copy()
        image_id = f'L{number:03d}'
        image_path = output / 'images' / f'{image_id}.png'
        require(cv2.imwrite(str(image_path), image), 'PNG export failed')
        require(np.array_equal(cv2.imread(str(image_path)), image), 'Lossless image round-trip failed')
        proposals = [{'id': f'D{d["detection_id"]}', 'xyxy': d['xyxy'],
                      'confidence': d['confidence'], 'origin': 'official_person_prediction_not_ground_truth'}
                     for d in frame['detections']]
        row = {'image_id': image_id, 'image_path': f'images/{image_id}.png', 'image_sha256': sha(image_path),
               'width': 320, 'height': 240, 'audit_sample_id': sample['sample_id'],
               'frame_index': frame['frame_index'], 'time_seconds': frame['time_seconds'],
               'proposals': proposals}
        images.append(row)
        records.append({'image_id': image_id, 'image_sha256': row['image_sha256'], 'status': 'pending',
                        'scene': 'unreviewed', 'annotations': [],
                        'proposals': proposals,
                        'proposal_states': {p['id']: 'pending' for p in proposals},
                        'class_visibility': {name: 'unreviewed' for name in CLASS_NAMES},
                        'annotation_complete': False, 'notes': ''})
    pilot_id = hashlib.sha256(json.dumps({'protocol': PROTOCOL, 'images': images},
                                        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    draft = {'schema': 'yolo_target_label_review_v1', 'pilot_id': pilot_id, 'protocol': PROTOCOL,
             'annotation_origin': 'human_review_required_with_official_person_proposals',
             'classes': CLASS_NAMES, 'images': records, 'role_metadata_protocol': ROLE_PROTOCOL}
    metadata = {'schema': 'yolo_target_label_pilot_v1', 'pilot_id': pilot_id, 'protocol': PROTOCOL,
                'seed': SEED, 'classes': CLASS_NAMES, 'images': images, 'draft': draft,
                'source_videos': samples, 'source_sha256': immutable,
                'source_pixels': 'lossless PNG left image from initial audit; no diagnostic H264 re-decoding',
                'sampling': 'one per shot-type x Make/Miss from existing train audit, excluding four ROI-development clips; 30/50/70% frames',
                'limitations': '24-frame development/annotation feasibility pilot, not independent evaluation or enough data for robust detector claims',
                'training_started': False, 'original_video_opens': 0, 'validation_test_opens': 0,
                'yolo_rerun': False, 'training_export_implemented': False,
                'script_sha256': sha(Path(__file__)),
                'ui_sha256': {name: sha(FOLDER / name) for name in ('label_editor.html', 'label_core.js', 'label_editor.js')}}
    write_json(output / 'manifest.json', metadata)
    write_json(output / 'draft_review.json', draft)
    for name in ('label_core.js', 'label_editor.js'):
        (output / name).write_bytes((FOLDER / name).read_bytes())
    # The UI never receives Make/Miss labels, source filenames or source-group metadata.
    ui_data = {'pilot_id': pilot_id, 'classes': CLASS_NAMES, 'images': images, 'draft': draft}
    template = (FOLDER / 'label_editor.html').read_text(encoding='utf-8')
    require(template.count('__PILOT_DATA__') == 1, 'Missing/duplicate UI data placeholder')
    inline = json.dumps(ui_data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    (output / 'index.html').write_text(template.replace('__PILOT_DATA__', inline), encoding='utf-8')
    require(all(sha(Path(p)) == digest for p, digest in immutable.items()), 'Source audit changed')
    status = validate_review(draft, draft)
    write_json(output / 'initial_status.json', status)
    print('Saved:', output / 'index.html', flush=True)
    return {'output': str(output), **status, 'original_inputs_unchanged': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--check-review', metavar='REVIEW_JSON')
    parser.add_argument('--audit-run')
    parser.add_argument('--pilot-dir')
    args = parser.parse_args(argv)
    if args.check_review:
        require(args.pilot_dir is not None, '--pilot-dir required for review check')
        result = validate_review(read_json(Path(args.pilot_dir) / 'draft_review.json'), read_json(args.check_review))
    else:
        require(args.audit_run is not None, '--audit-run required')
        result = prepare(args.audit_run, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
