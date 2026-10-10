#!/usr/bin/env python3
"""Bounded train-only visual pilot. Dry-run/manifest creation never access videos."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import PIL
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'development/data_quality'))
sys.path.insert(0, str(Path(__file__).parent))
import audit_quality_aware_sampler_v1 as qa
import audit_blur_train_v3 as scanner
from transforms.video_clip_transforms import ClipTransform, STRATEGIES, derived_seed, validate_config

BASE = ROOT / 'development/data_quality'
CONFIG = BASE / 'configs/visual_preprocessing_pilot_v1.yaml'
MANIFEST = BASE / 'manifests/visual_preprocessing_pilot_manifest_v1.csv'
OUTPUT = BASE / 'artifacts/visual_preprocessing_pilot_v1'
need = qa.need
sha256 = qa.sha256


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+'\n'


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(content)


def load_config(path=CONFIG):
    cfg = yaml.safe_load(path.read_text())
    need(isinstance(cfg, dict) and set(cfg) == {'version', 'seed', 'sampler_seed', 'epoch',
         'sample_size', 'clip_length', 'selection_rule', 'transform', 'strategies'}, 'Invalid pilot keys')
    need(cfg['version'] == 'visual_preprocessing_pilot_v1', 'Invalid version')
    need(type(cfg['sample_size']) is int and cfg['sample_size'] == 48, 'This pilot is bounded to 48 videos')
    need(cfg['clip_length'] == 64 and cfg['transform']['clip_length'] == 64, 'Exactly 64 continuous frames required')
    need(cfg['selection_rule'] == 'balanced_label_fps_category_source_sha256_v1', 'Invalid selection rule')
    for key in ('seed', 'sampler_seed', 'epoch'):
        need(type(cfg[key]) is int and 0 <= cfg[key] < 2**64, f'Invalid {key}')
    need(cfg['strategies'] == list(STRATEGIES), 'Unexpected candidate set')
    validate_config(cfg['transform'])
    sampler = qa.load_config(qa.DEFAULT_CONFIG)
    need(cfg['sampler_seed'] == sampler['global_seed'] and sampler['clip_length'] == 64,
         'Sampler config mismatch')
    return cfg


def category(row):
    return ('fallback' if row['has_clean_64'] == 'false' else
            'affected_clean' if int(row['candidate_frame_count']) else 'unaffected')


def build_manifest(cfg):
    summaries, segments, sidecar = qa.load_sidecars(qa.DEFAULT_MANIFEST, qa.DEFAULT_SEGMENTS, qa.DEFAULT_SUMMARY)
    rows = list(summaries.values())
    fps_values = sorted({r['fps_rational'] for r in rows})
    need(fps_values == ['2997/100', '2997/125'], 'Unexpected FPS distribution; review design')
    def rank(row):
        return hashlib.sha256(f"{cfg['seed']}\0{row['video_id']}\0{cfg['selection_rule']}".encode()).hexdigest()
    chosen = sorted((r for r in rows if category(r) == 'fallback'), key=rank)
    need(len(chosen) == 5, 'Expected all five fallback videos')
    cells = [(label, fps) for label in ('Make', 'Miss') for fps in fps_values]
    for slot in range(5, 48):
        counts = Counter((r['label'], r['fps_rational']) for r in chosen)
        wanted = min(cells, key=lambda cell: (counts[cell], cell))
        target_category = 'affected_clean' if (slot-5) % 2 == 0 else 'unaffected'
        used = {r['video_id'] for r in chosen}
        sources = Counter(r['source_group'] for r in chosen)
        pool = [r for r in rows if r['video_id'] not in used and
                (r['label'], r['fps_rational']) == wanted and category(r) == target_category]
        need(bool(pool), 'Cannot satisfy specified cohort; no silent substitution')
        chosen.append(min(pool, key=lambda r: (sources[r['source_group']], rank(r), r['video_id'])))
    records = []
    for r in sorted(chosen, key=lambda r: r['video_id']):
        s = qa.select_clips(video_id=r['video_id'], frame_count=int(r['frame_count']),
            candidate_segments=segments.get(r['video_id'], []), clip_length=64,
            seed=cfg['sampler_seed'], epoch=cfg['epoch'])[0].to_dict()
        records.append(dict(split='train', label=r['label'],
            source_group=r['source_group'], fps_rational=r['fps_rational'], category=category(r),
            frame_count=int(r['frame_count']), **s, seed=cfg['sampler_seed'], epoch=cfg['epoch'],
            pilot_seed=cfg['seed'], seed_provenance='sampler_config.global_seed; SHA256(seed,video_id,epoch,clip_index,policy,start)'))
    return records, sidecar


def manifest_text(records):
    return qa.csv_text(records, list(records[0]))


def validate_manifest(path, cfg):
    # Validate every string against immutable metadata before any video filesystem operation.
    qa.load_sidecars(path, qa.DEFAULT_SEGMENTS, qa.DEFAULT_SUMMARY)
    expected, sidecar = build_manifest(cfg)
    need(path.read_text() == manifest_text(expected), 'Pilot manifest differs from reproducible selection')
    return expected, sidecar


class PilotGuard(scanner.Guard):
    def __init__(self, records):
        super().__init__()
        ids = [r['video_id'] for r in records]
        need(len(ids) == 48 and len(set(ids)) == 48 and set(ids) <= self.allowed.keys(), 'Invalid pilot allowlist')
        self.pilot_ids = set(ids)
        self.events = []

    def authorize(self, video_id, video_path=None):
        need(video_id in self.pilot_ids, 'ID outside pilot rejected before path access')
        return super().authorize(video_id, video_path)

    def open(self, video_id, video_path, operation):
        fd = super().open(video_id, video_path, operation)
        self.events.append(dict(video_id=video_id, split='train', operation=operation, read_only=True))
        return fd


def decode_clip(record, guard):
    """Bounded RGB decoder: sequential indexing avoids imprecise compressed-video seeks.

    Existing decoders return full-video grayscale or embed frozen C3D preprocessing,
    so this new adapter reuses the existing guarded-open contract instead.
    Prefix frames must be decoded to locate the exact index; only [start,end) is retained.
    """
    vid = record['video_id']
    before = guard.identity(vid)
    fd = guard.open(vid, guard.allowed[vid]['video_path'], 'decode_contiguous_rgb_clip')
    cap, frames, timestamps = None, [], []
    try:
        cap = cv2.VideoCapture(f'/proc/self/fd/{fd}', cv2.CAP_FFMPEG)
        need(cap.isOpened(), 'Decoder failed')
        for index in range(record['end_frame_exclusive']):
            ok, bgr = cap.read()
            need(ok, f'Truncated clip at frame {index}: {vid}')
            if index >= record['start_frame']:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                timestamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)
    need(guard.identity(vid) == before, 'Source file identity changed')
    clip = np.stack(frames)
    need(len(clip) == 64, 'Noncontinuous/short clip')
    fps = float(record['fps_rational'].split('/')[0])/float(record['fps_rational'].split('/')[1])
    expected = np.arange(record['start_frame'], record['end_frame_exclusive']) * 1000/fps
    need(np.max(np.abs(np.array(timestamps)-expected)) < 500/fps, 'Frame timestamps disagree with contiguous indices')
    return clip, dict(retained_frames=64, decoder_read_frames=record['end_frame_exclusive'],
        prefix_frames=record['start_frame'], timestamps_match_indices=True,
        source_unchanged=True, source_identity=before, input_shape=list(clip.shape))


def diagnostics(tensor, transform):
    rgb = tensor.transpose(1, 2, 3, 0)
    luma = np.sum(rgb*np.array([0.2126, 0.7152, 0.0722], np.float32), axis=-1)
    brightness = luma.mean(axis=(1, 2))
    lap, ten, saturation = [], [], []
    for f, g in zip(rgb, luma):
        gray = np.rint(g*255).astype(np.float64)
        lap.append(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var())
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ten.append(np.mean(gx*gx+gy*gy))
        saturation.append(cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[:, :, 1].mean())
    result = dict(value_min=float(rgb.min()), value_max=float(rgb.max()),
        mean=float(rgb.mean()), std=float(rgb.std()),
        brightness=float(brightness.mean()), contrast=float(luma.std(axis=(1, 2)).mean()),
        saturation=float(np.mean(saturation)),
        endpoint_channel_fraction=float(np.mean((rgb <= 0) | (rgb >= 1))),
        newly_clipped_channel_fraction=transform['newly_clipped_channel_fraction'],
        laplacian_variance=float(np.mean(lap)), tenengrad=float(np.mean(ten)),
        frame_brightness_abs_delta=float(np.abs(np.diff(brightness)).mean()),
        retained_area_fraction=transform['retained_area_fraction'],
        aspect_ratio_relative_error=transform['aspect_ratio_relative_error'])
    return result, brightness


def aggregate(samples):
    result = {}
    for strategy in STRATEGIES:
        rows = [s['candidates'][strategy]['diagnostics'] for s in samples]
        result[strategy] = {key: dict(mean=float(np.mean([r[key] for r in rows])),
            min=float(min(r[key] for r in rows)), max=float(max(r[key] for r in rows))) for key in rows[0]}
    return result


def contact_sheet(path, record, clip, previews):
    indices = [0, 21, 42, 63]
    width, height = 256, 192
    canvas = Image.new('RGB', (4*width+180, 6*(height+24)+50), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{record['video_id']} | {record['category']} | overlap={record['candidate_overlap']}", fill='black')
    for row_index, name in enumerate(['source_context', *STRATEGIES]):
        y = 42+row_index*(height+24)
        draw.text((6, y+18), name, fill='black')
        for column, index in enumerate(indices):
            raw = clip[index] if name == 'source_context' else previews[name][column]
            im = Image.fromarray(raw)
            im.thumbnail((width, height))
            x = 180+column*width
            canvas.paste(im, (x+(width-im.width)//2, y+(height-im.height)//2))
            draw.text((x+4, y+height+2), f"frame {record['start_frame']+index}", fill='black')
    canvas.save(path)


def run(cfg, config_path, manifest_path, out):
    records, sidecar = validate_manifest(manifest_path, cfg)
    base = BASE/'artifacts'
    need(out.absolute().is_relative_to(base.absolute()) and out.absolute() != base.absolute(), 'Output must be under ignored artifacts')
    need(not any(p.is_symlink() for p in [out, *out.parents]), 'Symlink output rejected')
    need(not out.exists(), 'Refusing existing output directory')
    out.mkdir(parents=True)
    (out/'contact_sheets').mkdir()
    guard = PilotGuard(records)
    # All five fallback + two examples of each other category, selected before seeing pixels.
    review_ids = {r['video_id'] for r in records if r['fallback']}
    for cat in ('unaffected', 'affected_clean'):
        review_ids.update(r['video_id'] for r in [r for r in records if r['category'] == cat][:2])
    samples = []
    cv2.setNumThreads(1)
    for record in records:
        clip, decoding = decode_clip(record, guard)
        sample = dict(selection=record, decoding=decoding, candidates={})
        previews = {}
        ref_brightness = None
        for strategy in STRATEGIES:
            seed = derived_seed(cfg['seed'], record['video_id'], cfg['epoch'], 0, strategy)
            transform = ClipTransform(cfg['transform'], strategy)
            tensor, meta = transform(clip, seed=seed, sample_metadata=record)
            # End-to-end equality in the same decoder/library environment, no re-open.
            repeat, repeat_meta = transform(clip, seed=seed, sample_metadata=record)
            need(np.array_equal(tensor, repeat) and meta == repeat_meta, 'Same-seed mismatch')
            del repeat
            stats, brightness = diagnostics(tensor, meta['transform'])
            if strategy == 'reference':
                ref_brightness = brightness
            residual = np.diff(brightness)-np.diff(ref_brightness)
            stats['brightness_delta_residual_mae_vs_reference'] = float(np.abs(residual).mean())
            stats['brightness_delta_residual_max_vs_reference'] = float(np.abs(residual).max())
            sample['candidates'][strategy] = dict(metadata=meta, diagnostics=stats,
                same_seed_equal=True, tensor_sha256=hashlib.sha256(tensor.tobytes()).hexdigest())
            previews[strategy] = np.rint(tensor[:, [0, 21, 42, 63]].transpose(1, 2, 3, 0)*255).astype(np.uint8)
            del tensor
        if record['video_id'] in review_ids:
            name = record['video_id'].replace('/', '__').replace('.mp4', '.png')
            contact_sheet(out/'contact_sheets'/name, record, clip, previews)
            sample['contact_sheet'] = f'contact_sheets/{name}'
        samples.append(sample)
        print(f"{len(samples)}/48 {record['video_id']}", flush=True)
    # Original file identities are evidence, but only portable relative paths enter tracked reports.
    code_paths = [Path(__file__), BASE/'transforms/video_clip_transforms.py',
        Path(qa.__file__), Path(scanner.__file__), Path(scanner.p.__file__), Path(scanner.p.gate.__file__),
        BASE/'samplers/quality_aware_clip_sampler.py', qa.DEFAULT_CONFIG]
    inputs = [config_path, manifest_path, qa.DEFAULT_MANIFEST, ROOT/'data/processed/split_summary.json',
        qa.DEFAULT_SEGMENTS, qa.DEFAULT_SUMMARY, qa.DEFAULT_SEGMENTS.parent/qa.METADATA_NAME]
    summary = dict(schema_version='basketball.visual-preprocessing-pilot.v1', status='PILOT_COMPLETE',
        config=cfg, videos=len(samples), clips=len(samples), retained_frames=64*len(samples),
        decoder_read_frames=sum(s['decoding']['decoder_read_frames'] for s in samples),
        prefix_frames=sum(s['decoding']['prefix_frames'] for s in samples),
        labels=dict(Counter(r['label'] for r in records)), categories=dict(Counter(r['category'] for r in records)),
        fps=dict(Counter(r['fps_rational'] for r in records)),
        label_fps=dict(Counter(r['label']+' / '+r['fps_rational'] for r in records)),
        sources=len({r['source_group'] for r in records}),
        source_counts=dict(Counter(r['source_group'] for r in records)),
        fallback_clips=sum(r['fallback'] for r in records),
        candidate_overlap=sum(r['candidate_overlap'] for r in records),
        same_seed_checks=len(samples)*len(STRATEGIES), all_same_seed_equal=True,
        all_timestamps_match_indices=True, all_sources_unchanged=True,
        contact_sheets=[s['contact_sheet'] for s in samples if 'contact_sheet' in s],
        diagnostics=aggregate(samples), sidecar_run_identity=sidecar['run_identity'],
        provenance={str(p.relative_to(ROOT)): sha256(p) for p in inputs+code_paths},
        environment=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
                         pillow=PIL.__version__, pyyaml=yaml.__version__),
        command=sys.argv, access_events=guard.events,
        limitations=['Quota cohort is not representative of full train prevalence.',
            'Source groups are not verified game IDs.', 'Frame deltas include real scene motion/cuts.',
            'Contact sheets are sparse spatial review, not continuous human playback.',
            'Higher sharpness is not proof of quality or classification benefit.'])
    write_new(out/'samples.json', dumps(samples))
    summary['samples_sha256'] = sha256(out/'samples.json')
    write_new(out/'summary.json', dumps(summary))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=CONFIG)
    ap.add_argument('--manifest', type=Path, default=MANIFEST)
    ap.add_argument('--output-dir', type=Path, default=OUTPUT)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--create-manifest', action='store_true')
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.create_manifest:
        need(args.manifest.absolute() == MANIFEST, 'Only the dedicated new pilot manifest may be created')
        records, _ = build_manifest(cfg)
        write_new(args.manifest, manifest_text(records))
        print('MANIFEST_CREATED: 48 train IDs; video access=0')
    elif args.dry_run:
        records, _ = validate_manifest(args.manifest, cfg)
        print(dumps(dict(status='DRY_RUN_PASS', videos=len(records), video_access=0,
            labels=dict(Counter(r['label'] for r in records)),
            categories=dict(Counter(r['category'] for r in records)))))
    else:
        summary = run(cfg, args.config, args.manifest, args.output_dir)
        print(dumps({k: summary[k] for k in ('status', 'videos', 'clips', 'retained_frames', 'decoder_read_frames')}))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
