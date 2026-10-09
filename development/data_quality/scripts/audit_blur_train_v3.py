#!/usr/bin/env python3
"""Offline train-only candidate scanner. Default: fixed 20-video smoke; never cleans data."""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import stat
import subprocess
import sys
import time
import uuid

import cv2
import numpy as np
import yaml
import audit_blur_pilot_v2 as v2

p = v2.p
ROOT = p.ROOT
VERSION = 'blur_train_v3.1'
DEFAULT_CONFIG = ROOT / 'development/data_quality/configs/blur_train_v3.yaml'
ARTIFACTS = ROOT / 'development/data_quality/artifacts'
SMOKE_LIMIT = 20
need = p.need
digest = p.gate.digest


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def atomic_text(path, text):
    """Exclusive final name, fsynced temp + rename under the run's advisory lock."""
    need(not path.exists() and not path.is_symlink(), f'Refusing overwrite: {path}')
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('x') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    # Never remove a crashed temp file; resume records and preserves it.
    need(not path.exists() and not path.is_symlink(), 'Destination appeared during write')
    os.rename(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    atomic_text(path, p.dumps(value))


def csv_text(rows, fields=None):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields or list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def append_event(ledger_path, **event):
    # A crash can truncate the last line; readers must fail closed on malformed ledgers.
    with ledger_path.open('a') as f:
        f.write(json.dumps(dict(timestamp=utc(), **event), sort_keys=True, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def load_config(path):
    cfg = yaml.safe_load(path.read_text())
    expected = dict(version='blur_train_v3', seed=20261009, smoke_size=20,
        smoke_manifest='development/data_quality/smoke_manifest_v3.csv',
        primary_metric='laplacian_variance', rolling_seconds=3.0, mad_scale_floor=.05,
        log_floor=1e-12, z_threshold=-2., minimum_seconds=.16, windows=[16, 32, 64],
        max_frames_per_video=10000)
    for key, value in expected.items():
        need(cfg.get(key) == value, f'Frozen v3 contract mismatch: {key}; stop, do not retune')
    required = [v2.DEFAULT_CONFIG, ROOT/'docs/data_quality/blur_screening_pilot_v2.json',
                ROOT/'docs/data_quality/blur_screening_method_v2.md', Path(v2.__file__)]
    need(set(cfg['frozen_evidence']) == {str(x.relative_to(ROOT)) for x in required}, 'Missing frozen evidence')
    for name, sha in cfg['frozen_evidence'].items():
        data = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
        need(hashlib.sha256(data).hexdigest() == sha == digest(ROOT/name),
             f'Uncommitted/changed v2 evidence: {name}; stop')
    old = yaml.safe_load(v2.DEFAULT_CONFIG.read_text())
    report = read_json(required[1])
    t = report['temporal_thresholds']['laplacian']
    need(report['provenance']['config_sha256'] == digest(v2.DEFAULT_CONFIG), 'v2 provenance mismatch')
    actual = (old['temporal']['rolling_seconds'], old['temporal']['mad_epsilon'],
              old['score_epsilon'], t['z_threshold'], t['minimum_seconds'])
    need(actual == (cfg['rolling_seconds'], cfg['mad_scale_floor'], cfg['log_floor'],
                    cfg['z_threshold'], cfg['minimum_seconds']), f'Prompt/v2 mismatch: {actual}; stop')
    return cfg


def choose_smoke(guard, seed):
    """Alternate labels, least-used source, SHA tie break; reserve observed train FPS groups.

    Only source/FPS columns from prior train metadata are used. No quality scores.
    """
    hints = {}
    for name in ('pilot_manifest_v1.csv', 'pilot_manifest_v2.csv'):
        for row in p.read_csv(ROOT/'development/data_quality'/name):
            need(row['video_id'] in guard.allowed and row['split'] == 'train', 'Non-train FPS hint')
            hints.setdefault(row['source_group'], set()).add(row['fps_rational'])
    pool = list(guard.allowed.values())
    selected, used = [], Counter()
    fps_groups = sorted({fps for values in hints.values() for fps in values})
    for i in range(SMOKE_LIMIT):
        label = str(i % 2)
        eligible = [r for r in pool if r['binary_label'] == label]
        if i < 2 * len(fps_groups):
            fps = fps_groups[i//2]
            eligible = [r for r in eligible if hints.get(r['source_group']) == {fps}]
        row = min(eligible, key=lambda r: (used[r['source_group']], v2.keyed(seed, r['video_id']), r['video_id']))
        selected.append(dict(video_id=row['video_id'], video_path=row['video_path'], split='train',
            binary_label=label, label='Make' if label == '1' else 'Miss', source_group=row['source_group'],
            fps_hint=';'.join(sorted(hints.get(row['source_group'], []))), selection_seed=str(seed)))
        used[row['source_group']] += 1
        pool.remove(row)
    return sorted(selected, key=lambda r: r['video_id'])


class Guard(p.TrainGuard):
    def __init__(self, ledger=None):
        super().__init__()
        self.ledger = ledger

    def identity(self, video_id):
        path = self.authorize(video_id)
        # No arbitrary path accepted, no symlink/hardlink aliases to held-out files.
        need(not any(x.is_symlink() for x in [path, *path.parents]), 'Symlink input rejected')
        need(path.resolve() == path.absolute(), 'Resolved path differs from fixed train manifest')
        info = path.stat()
        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Nonregular/hardlink input rejected')
        return dict(video_id=video_id, resolved_path=str(path), size=info.st_size,
            mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns, device=info.st_dev, inode=info.st_ino,
            manifest_sha256=self.scope['input_hashes']['manifest'])

    def open(self, video_id, video_path, operation):
        path = self.authorize(video_id, video_path)
        before = self.identity(video_id)
        event = dict(video_id=video_id, path=str(path), split='train', operation=operation, read_only=True)
        if self.ledger:
            append_event(self.ledger, phase='authorized_attempt', **event)
        fd = p.gate.open_guarded(path)
        try:
            after = os.fstat(fd)
            need((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
                 (before['device'], before['inode'], before['size'], before['mtime_ns']), 'Input changed on open')
            if self.ledger:
                append_event(self.ledger, phase='opened', **event)
        except BaseException:
            os.close(fd)
            raise
        return fd


def code_identity():
    files = [Path(__file__), Path(v2.__file__), Path(p.__file__), Path(p.gate.__file__)]
    hashes = {str(f.relative_to(ROOT)): digest(f) for f in files}
    return dict(version=VERSION, files=hashes, sha256=p.gate.objhash(hashes))


def context(cfg, config_path, guard, selected, scope, seed):
    return dict(config_hash=digest(config_path), config=cfg, scanner=code_identity(),
        inputs=guard.scope['input_hashes'], selection=selected, scope=scope, seed=seed,
        environment=dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__,
            ffprobe=subprocess.check_output(['ffprobe', '-version'], text=True).splitlines()[0]))


def output_path(path):
    path = path.absolute()
    need(not any(x.is_symlink() for x in [path, *path.parents]), 'Symlink output rejected')
    need(path.parent == ARTIFACTS and path.name.startswith('blur_train_v3_'),
         'Output must be a new blur_train_v3_* directory directly below ignored artifacts')
    if path.exists():
        need(path.is_dir(), 'Output is not a directory')
        need(not any(f.is_symlink() for f in path.rglob('*')), 'Symlink within outputs rejected')
    return path


@contextmanager
def locked_run(out, ctx, resume):
    if out.exists():
        need(resume, 'Output exists; use --resume with identical context')
    else:
        need(not resume, 'Resume requires an existing run')
        out.mkdir(parents=True)
    fd = os.open(out/'run.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            try:
                need(read_json(out/'run.json')['context'] == ctx, 'Resume context/config/code/selection mismatch')
            except (ValueError, OSError) as exc:
                append_event(out/'errors.jsonl', kind='resume_context_rejected', reason=str(exc))
                raise
        else:
            atomic_json(out/'run.json', dict(context=ctx, created_utc=utc(),
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                command=sys.argv))
            (out/'videos').mkdir()
            (out/'invocations').mkdir()
            (out/'aggregates').mkdir()
        yield
    finally:
        os.close(fd)


def key(video_id):
    return hashlib.sha256(video_id.encode()).hexdigest()


def latest_attempt(out, video_id):
    directory = out/'videos'/key(video_id)
    attempts = sorted(directory.glob('attempt-*')) if directory.exists() else []
    return attempts[-1] if attempts else None


def inspect_attempt(attempt, row, ctx, identity=None):
    """Fail closed, including incomplete records, unexpected files and wrong row counts."""
    if attempt is None:
        return 'pending', None
    try:
        record = read_json(attempt/'completion.json')
        need(record['video_id'] == row['video_id'], 'Completion ID mismatch')
        need(record['config_hash'] == ctx['config_hash'] and record['scanner'] == ctx['scanner'], 'Completion context mismatch')
        need(record['status'] in ('ok', 'failed'), 'Unknown completion status')
        need(record['input_identity']['video_id'] == row['video_id'], 'Input identity mismatch')
        need(record['input_identity']['resolved_path'] == str(p.gate.DATA/row['video_path'])
             and record['input_identity']['manifest_sha256'] == ctx['inputs']['manifest'], 'Input provenance mismatch')
        if identity is not None:
            need(record['input_identity'] == identity, 'Input stat identity changed')
        need(set(record['output_hashes']) == {'frames.csv', 'summary.json'}, 'Unexpected output inventory')
        need({f.name for f in attempt.iterdir()} == {'frames.csv', 'summary.json', 'completion.json'}, 'Partial/unexpected output files')
        for name, sha in record['output_hashes'].items():
            need(digest(attempt/name) == sha, 'Output hash mismatch')
        summary = read_json(attempt/'summary.json')
        need(summary['video_id'] == row['video_id'] and summary['decode_status'] == record['status'], 'Summary identity/status mismatch')
        need(summary['split'] == 'train' and summary['source_group'] == row['source_group']
             and summary['label'] == ('Make' if row['binary_label']=='1' else 'Miss'), 'Summary label/split mismatch')
        need(summary['config_hash'] == ctx['config_hash'] and summary['scanner_version'] == VERSION, 'Summary context mismatch')
        with (attempt/'frames.csv').open(newline='') as f:
            count = 0
            for frame in csv.DictReader(f):
                need(frame['video_id'] == row['video_id'] and int(frame['frame_index']) == count, 'Frame identity/order mismatch')
                count += 1
        need(count == record['processed_frame_count'] == summary['processed_frame_count'], 'Processed count mismatch')
        need(summary['expected_frame_count'] == record['expected_frame_count'], 'Expected count mismatch')
        if record['status'] == 'ok':
            need(count > 0 and count == record['expected_frame_count'], 'Incomplete decode')
        return record['status'], summary
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return 'suspect', dict(reason=str(exc))


def segments(mask, fps):
    return [dict(start_frame=lo, end_frame_exclusive=hi+1, start_seconds=lo/fps,
        end_seconds_exclusive=(hi+1)/fps, duration_seconds=(hi-lo+1)/fps, frames=hi-lo+1)
        for lo, hi in p.runs(mask)]


def summarize_scores(scores, fps, cfg):
    old = dict(score_epsilon=cfg['log_floor'], temporal=dict(rolling_seconds=cfg['rolling_seconds'], mad_epsilon=cfg['mad_scale_floor']))
    z, _, _, width = v2.local_normalize(scores, fps, old)
    raw = z[:, 0] <= cfg['z_threshold']
    keep = raw.copy()
    for lo, hi in p.runs(raw):
        if (hi-lo+1)/fps < cfg['minimum_seconds']:
            keep[lo:hi+1] = False
    bad, good = segments(keep, fps), segments(~keep, fps)
    # Fixed v2 calibration P10 is descriptive only. Never fit to this run.
    thresholds = read_json(ROOT/'docs/data_quality/blur_screening_pilot_v2.json')['static']['absolute_thresholds']
    diagnostics = {}
    for j, metric in enumerate(p.METRICS):
        mask = scores[:, j] <= thresholds[metric]
        spans = segments(mask, fps)
        diagnostics[metric] = dict(reference_v2_calibration_p10=thresholds[metric],
            below_reference_ratio=float(mask.mean()), entire_video_below_reference=bool(mask.all()),
            longest_below_reference_seconds=max((s['duration_seconds'] for s in spans), default=0),
            diagnostic_only=True)
    summary = dict(candidate_frames=int(keep.sum()), candidate_ratio=float(keep.mean()),
        candidate_segments=bad, noncandidate_segments=good,
        longest_candidate_seconds=max((s['duration_seconds'] for s in bad), default=0),
        noncandidate_windows={str(w):dict(exists=any(s['frames'] >= w for s in good),
            count=sum(max(0, s['frames']-w+1) for s in good)) for w in cfg['windows']},
        quantiles={m:{name:float(np.quantile(scores[:, j], q)) for name,q in
            [('p00',0),('p10',.1),('p50',.5),('p90',.9),('p100',1)]} for j,m in enumerate(p.METRICS)},
        persistent_absolute_low_diagnostics=diagnostics, rolling_window_frames=width)
    return z, raw, keep, summary


FRAME_FIELDS = ['video_id','label','source_group','split','fps','expected_frame_count','duration_seconds',
    'frame_index','timestamp_seconds','estimated_timestamp_seconds','timestamp_status',
    'laplacian_variance','tenengrad','laplacian_local_z','tenengrad_local_z','primary_candidate',
    'candidate_reason','decode_status','config_hash','scanner_version']


def process_video(row, guard, cfg, ctx, attempt):
    start = time.perf_counter()
    identity = guard.identity(row['video_id'])
    rec, scores, pts = None, [], []
    status, reason = 'ok', None
    try:
        rec = p.metadata(row, guard, {})
        need(0 < rec['frame_count'] <= cfg['max_frames_per_video'], 'Engineering frame cap exceeded')
        need(rec['resolution'] == '320x240' and 0 < rec['fps'] <= 120, 'Unsupported metadata bounds')
        fd = guard.open(row['video_id'], row['video_path'], 'decode')
        cap = None
        try:
            cap = cv2.VideoCapture(f'/proc/self/fd/{fd}', cv2.CAP_FFMPEG)
            need(cap.isOpened(), 'Decoder open failed')
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                need(len(scores) < cfg['max_frames_per_video'], 'Decode frame cap exceeded')
                need(bgr.shape == (240, 320, 3), 'Decoded resolution changed')
                scores.append(v2.score(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)))
                pts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000)
        finally:
            if cap is not None:
                cap.release()
            os.close(fd)
        need(len(scores) == rec['frame_count'], 'Decode frame count mismatch')
        need(guard.identity(row['video_id']) == identity, 'Input changed during processing')
    except (OSError, ValueError, subprocess.SubprocessError, cv2.error) as exc:
        status, reason = 'failed', str(exc)
    n = len(scores)
    fps = rec['fps'] if rec else None
    reliable = bool(n and np.isfinite(pts).all() and (np.diff(pts) > 0).all()
        and np.max(np.abs(np.asarray(pts)-np.arange(n)/fps)) <= .5/fps
        and rec['fps_rational'] == rec['r_frame_rate'])
    summary = dict(video_id=row['video_id'], label='Make' if row['binary_label']=='1' else 'Miss',
        source_group=row['source_group'], split='train', fps=fps,
        fps_rational=rec['fps_rational'] if rec else None, expected_frame_count=rec['frame_count'] if rec else None,
        processed_frame_count=n, duration_seconds=rec['duration_seconds'] if rec else None,
        decode_status=status, failure_reason=reason, timestamp_verified=reliable,
        config_hash=ctx['config_hash'], scanner_version=VERSION)
    if status == 'ok':
        z, raw, keep, quality = summarize_scores(np.asarray(scores), fps, cfg)
        summary.update(quality)
    else:
        z = np.zeros((n, 2)); raw = keep = np.zeros(n, dtype=bool)
        summary.update(candidate_frames=None, candidate_ratio=None, candidate_segments=None,
            noncandidate_segments=None, quality_unavailable=True)
    frames = []
    for i in range(n):
        why = ('decode_failure_not_quality' if status != 'ok' else 'laplacian_local_z_le_-2_duration_ge_0.16'
               if keep[i] else 'below_z_but_short_run' if raw[i] else 'not_candidate')
        frames.append(dict(video_id=row['video_id'], label=summary['label'], source_group=row['source_group'],
            split='train', fps=fps, expected_frame_count=summary['expected_frame_count'],
            duration_seconds=summary['duration_seconds'], frame_index=i,
            timestamp_seconds=pts[i] if reliable else None, estimated_timestamp_seconds=i/fps,
            timestamp_status='decoder_cfr_consistent' if reliable else 'unverified_use_frame_index',
            laplacian_variance=float(scores[i][0]), tenengrad=float(scores[i][1]),
            laplacian_local_z=float(z[i,0]) if status=='ok' else None,
            tenengrad_local_z=float(z[i,1]) if status=='ok' else None,
            primary_candidate=int(keep[i]) if status=='ok' else None, candidate_reason=why,
            decode_status=status, config_hash=ctx['config_hash'], scanner_version=VERSION))
    atomic_text(attempt/'frames.csv', csv_text(frames, FRAME_FIELDS))
    summary['processing_seconds'] = time.perf_counter()-start
    atomic_json(attempt/'summary.json', summary)
    record = dict(video_id=row['video_id'], input_identity=identity, config_hash=ctx['config_hash'],
        scanner=ctx['scanner'], expected_frame_count=summary['expected_frame_count'], processed_frame_count=n,
        output_hashes={name:digest(attempt/name) for name in ('frames.csv','summary.json')},
        completion_timestamp=utc(), status=status)
    atomic_json(attempt/'completion.json', record)  # commit marker LAST
    if status != 'ok':
        append_event(attempt.parents[2]/'decode_failures.jsonl', video_id=row['video_id'],
            attempt=str(attempt.relative_to(attempt.parents[2])), reason=reason,
            expected_frame_count=summary['expected_frame_count'], processed_frame_count=n)
    return summary


def aggregate(out, ctx):
    """Read one per-video summary/CSV at a time; no dataset-wide frame arrays."""
    result = dict(selected=len(ctx['selection']), ok=0, failed=0, pending=0, suspect=0,
        processed_frames=0, candidate_frames=0, candidate_segments=0, affected_videos=0,
        groups={dim:{} for dim in ('label','source_group','fps_rational')}, videos=[])
    for row in ctx['selection']:
        attempt = latest_attempt(out, row['video_id'])
        status, summary = inspect_attempt(attempt, row, ctx)
        result[status] += 1
        info = dict(video_id=row['video_id'], status=status, attempt=str(attempt.relative_to(out)) if attempt else None)
        if status == 'ok':
            n, candidates = summary['processed_frame_count'], summary['candidate_frames']
            result['processed_frames'] += n
            result['candidate_frames'] += candidates
            result['candidate_segments'] += len(summary['candidate_segments'])
            result['affected_videos'] += int(candidates > 0)
            for dim, groups in result['groups'].items():
                group = groups.setdefault(str(summary[dim]), dict(videos=0, frames=0, candidate_frames=0))
                group['videos'] += 1; group['frames'] += n; group['candidate_frames'] += candidates
            info.update({k:summary[k] for k in ('label','source_group','fps_rational','processed_frame_count',
                'candidate_frames','candidate_ratio','processing_seconds','longest_candidate_seconds')})
            info['candidate_segments'] = len(summary['candidate_segments'])
        result['videos'].append(info)
    result['candidate_ratio'] = result['candidate_frames']/result['processed_frames'] if result['processed_frames'] else None
    result['status'] = 'COMPLETE' if result['ok'] == result['selected'] else 'INCOMPLETE'
    return result


def save_aggregate(out, ctx):
    result = aggregate(out, ctx)
    path = out/'aggregates'/('summary-'+str(time.time_ns())+'.json')
    atomic_json(path, result)
    print(f'AGGREGATE {path} {result["status"]} ok={result["ok"]}/{result["selected"]}', flush=True)
    return result


def run(out, ctx, cfg, resume=False, retry=False, rebuild=False):
    start = time.perf_counter()
    stats = dict(started_utc=utc(), command=sys.argv, skipped=0, attempted=0, failed=0, suspect=0, interrupted=False)
    with locked_run(out, ctx, resume):
        guard = Guard(out/'access.jsonl')
        cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); cv2.setUseOptimized(False)
        if not rebuild:
            try:
                for ordinal, row in enumerate(ctx['selection'], 1):
                    attempt = latest_attempt(out, row['video_id'])
                    try:
                        identity = guard.identity(row['video_id'])
                        state, detail = inspect_attempt(attempt, row, ctx, identity)
                        if state == 'ok':
                            stats['skipped'] += 1
                            print(f'[{ordinal}/{len(ctx["selection"])}] SKIP {row["video_id"]}', flush=True)
                            continue
                        if state == 'suspect':
                            stats['suspect'] += 1
                            append_event(out/'errors.jsonl', kind='suspect_output', video_id=row['video_id'],
                                attempt=str(attempt.relative_to(out)), reason=detail['reason'])
                        if state in ('failed','suspect') and not retry:
                            continue
                        directory = out/'videos'/key(row['video_id'])
                        directory.mkdir(exist_ok=True)
                        number = len(list(directory.glob('attempt-*'))) + 1
                        attempt = directory/f'attempt-{number:06d}'
                        attempt.mkdir()
                        stats['attempted'] += 1
                        print(f'[{ordinal}/{len(ctx["selection"])}] SCAN {row["video_id"]}', flush=True)
                        summary = process_video(row, guard, cfg, ctx, attempt)
                        stats['failed'] += int(summary['decode_status'] != 'ok')
                    except (OSError, ValueError, subprocess.SubprocessError, cv2.error) as exc:
                        stats['failed'] += 1
                        append_event(out/'errors.jsonl', kind='video_failure', video_id=row['video_id'], reason=str(exc))
            except KeyboardInterrupt:
                stats['interrupted'] = True
                append_event(out/'errors.jsonl', kind='interrupted', reason='SIGINT/SIGTERM; partial attempt preserved')
        result = save_aggregate(out, ctx)
        stats.update(elapsed_seconds=time.perf_counter()-start, finished_utc=utc(),
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            status=result['status'])
        atomic_json(out/'invocations'/f'{time.time_ns()}.json', stats)
    unresolved = stats['failed'] or (stats['suspect'] and not retry)
    return 130 if stats['interrupted'] else 0 if result['status']=='COMPLETE' and not unresolved else 2


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    ap.add_argument('--manifest', type=Path, default=p.gate.MANIFEST, help='Only the fixed dataset_manifest.csv is accepted')
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--scope', choices=('smoke','full-train'), default='smoke')
    ap.add_argument('--max-videos', type=int)
    ap.add_argument('--video-id', action='append', help='Subset of fixed smoke IDs, or train IDs in explicitly confirmed full-train scope')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--retry-failures', action='store_true', help='With resume: preserve failed/suspect attempts; write a NEW attempt')
    ap.add_argument('--rebuild-summary', action='store_true', help='With resume: derived outputs only, no video stat/open/probe/decode')
    ap.add_argument('--dry-run', action='store_true', help='Metadata only: no video stat/open/probe/decode, no outputs')
    ap.add_argument('--seed', type=int, default=20261009)
    ap.add_argument('--confirm-full-train', action='store_true', help='Required for full-train, including dry-run')
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    need(args.scope != 'full-train' or args.confirm_full_train, 'full-train requires explicit --confirm-full-train')
    need(not args.confirm_full_train or args.scope == 'full-train', 'Confirmation requires full-train scope')
    need(not (args.retry_failures or args.rebuild_summary) or args.resume, 'Recovery/rebuild requires --resume')
    cfg = load_config(args.config)
    need(args.seed == cfg['seed'], 'Fixed selection seed required')
    need(args.manifest.absolute() == p.gate.MANIFEST and not args.manifest.is_symlink(), 'Arbitrary manifest rejected')
    guard = Guard()
    if args.scope == 'smoke':
        selected = choose_smoke(guard, args.seed)
        need(p.read_csv(ROOT/cfg['smoke_manifest']) == selected, 'Fixed smoke manifest differs from reproducible selection')
    else:
        selected = sorted(guard.allowed.values(), key=lambda r:r['video_id'])
    if args.video_id:
        need(len(args.video_id) == len(set(args.video_id)), 'Duplicate video ID')
        need(set(args.video_id) <= {r['video_id'] for r in selected}, 'Requested ID outside fixed scope')
        selected = [r for r in selected if r['video_id'] in args.video_id]
    if args.max_videos is not None:
        need(1 <= args.max_videos <= (SMOKE_LIMIT if args.scope=='smoke' else 7193), 'Invalid video limit')
        selected = selected[:args.max_videos]
    need(bool(selected), 'Empty selection')
    if args.dry_run:
        print(p.dumps(dict(status='DRY_RUN_PASS', scope=args.scope, selected=len(selected),
            ids=[r['video_id'] for r in selected], config_hash=digest(args.config),
            frozen_evidence=cfg['frozen_evidence'], input_hashes=guard.scope['input_hashes'],
            video_opens=0, video_probes=0, video_decodes=0)))
        return 0
    need(args.scope != 'full-train' or args.output_dir is not None, 'Full-train requires explicit output directory')
    out = output_path(args.output_dir or ROOT/cfg['output_dir'])
    ctx = context(cfg, args.config, guard, selected, args.scope, args.seed)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    return run(out, ctx, cfg, args.resume, args.retry_failures, args.rebuild_summary)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
