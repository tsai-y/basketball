#!/usr/bin/env python3
"""Train-only, read-only video quality pilot. Candidate scores are not blur truth."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import PIL
import yaml

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location('gate1a', ROOT / 'scripts/audit_i3d_temporal_feasibility.py')
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)
METHODS = ('laplacian_low', 'tenengrad_low', 'both_low', 'smoothed_both_low')
METRICS = ('laplacian_variance', 'tenengrad')
DEFAULT_CONFIG = ROOT / 'development/data_quality/configs/blur_pilot_v1.yaml'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'


def write_json(path, obj):
    with path.open('x') as f:
        f.write(dumps(obj))


def write_csv(path, rows):
    need(bool(rows), 'Empty CSV output')
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def load_config(path):
    cfg = yaml.safe_load(path.read_text())
    need(cfg['version'] == 'blur_pilot_v1', 'Unsupported version')
    need(cfg['selection_rule'] == 'balanced_label_unique_source_sha256_v1', 'Unknown selection rule')
    need(2 <= cfg['sample_size'] <= 40 and cfg['sample_size'] % 2 == 0,
         'Pilot requires an even sample size 2..40; full-train execution is disabled')
    need(cfg['dataset_root'] == str(gate.DATA.relative_to(ROOT)), 'Dataset root is fixed')
    need(cfg['scoring'] == dict(decoder='opencv_CAP_FFMPEG', resolution='original',
         grayscale='COLOR_BGR2GRAY_uint8_then_float64', border='BORDER_REFLECT_101',
         laplacian_ksize=1, sobel_ksize=3, scale=1.0, delta=0.0), 'Unsupported scoring contract')
    t = cfg['thresholds']
    need(0 < t['quantile'] < 0.5 and t['quantile_method'] == 'linear'
         and t['comparison'] == 'less_equal', 'Invalid quantile specification')
    need(t['smoothing'] == 'centered_median_truncated_edges'
         and 0 < t['smoothing_seconds'] <= 1 and 0 < t['minimum_segment_seconds'] <= 1,
         'Invalid smoothing specification')
    need(t['primary_candidate'] in METHODS, 'Invalid primary candidate')
    need(all(isinstance(w, int) and w > 0 for w in cfg['windows']), 'Invalid windows')
    cs = cfg['contact_sheets']
    need(1 <= cs['examples_per_band'] <= 6 and 1 <= cs['preview_jpeg_quality'] <= 100,
         'Invalid contact-sheet limits')
    need(set(cs['bands']) == {'low', 'middle', 'high'}
         and all(0 <= q <= 1 for q in cs['bands'].values()), 'Invalid contact bands')
    return cfg


class TrainGuard:
    """The immutable Gate 1A manifest hash and exact train row authorize every open."""
    def __init__(self):
        rows, denied, self.scope = gate.load_scope(['train'])
        self.allowed = {r['video_id']: r for r in rows}
        self.denied = denied | {r['video_path'] for r in read_csv(gate.MANIFEST)
                                if r['split'] != 'train'}
        self.events = []

    def authorize(self, video_id, video_path=None):
        # Check strings BEFORE resolving/statting/opening any supplied path.
        need(video_id in self.allowed, 'ID is not in the fixed train allowlist')
        row = self.allowed[video_id]
        need(video_path is None or video_path == row['video_path'], 'Path does not match train ID')
        need(row['split'] == 'train', 'Unknown/non-train split')
        return gate.authorize(row, self.allowed, self.denied)

    def open(self, video_id, video_path, operation):
        path = self.authorize(video_id, video_path)
        fd = gate.open_guarded(path)
        self.events.append(dict(video_id=video_id, split='train', operation=operation,
                                read_only=True, fd=fd))
        return fd


def choose(guard, cfg, seed):
    """Alternate labels; prefer least-used source; seeded SHA256 breaks ties."""
    pool = sorted(guard.allowed.values(), key=lambda r: r['video_id'])
    selected, source_counts = [], Counter()
    for i in range(cfg['sample_size']):
        label = str(i % 2)
        candidates = [r for r in pool if r['binary_label'] == label]
        row = min(candidates, key=lambda r: (source_counts[r['source_group']],
                  hashlib.sha256(f"{seed}:{r['video_id']}".encode()).hexdigest(), r['video_id']))
        selected.append(dict(row))
        source_counts[row['source_group']] += 1
        pool.remove(row)
    return sorted(selected, key=lambda r: r['video_id'])


def cached_train_metadata(cfg, guard):
    path = Path(cfg['gate1a_cache'])
    if not path.exists():
        return {}, 'Gate 1A per-video cache absent; probe selected pilot train only'
    report = json.loads((ROOT / 'docs/i3d_temporal_feasibility_audit.json').read_text())
    need(not path.is_symlink() and gate.digest(path) == report['provenance']['cache_sha256'],
         'Gate 1A cache identity mismatch')
    payload = json.loads(path.read_text())
    need(payload['scope']['input_hashes'] == guard.scope['input_hashes'], 'Stale metadata cache')
    cached = {}
    for r in payload['records']:
        if r['split'] != 'train':
            continue  # never derive or use held-out metadata
        guard.authorize(r['video_id'])
        need(r['binary_label'] == guard.allowed[r['video_id']]['binary_label'], 'Cache label mismatch')
        cached[r['video_id']] = gate.derive(r)
    return cached, 'Gate 1A hash-verified train records reused'


def metadata(row, guard, cached):
    if row['video_id'] in cached:
        rec = cached[row['video_id']]
    else:
        fd = guard.open(row['video_id'], row['video_path'], 'probe')
        try:
            result = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
                    '-show_entries', gate.ENTRIES, '-of', 'json', f'/proc/self/fd/{fd}'],
                    pass_fds=(fd,), capture_output=True, text=True, timeout=30)
        finally:
            os.close(fd)
        need(result.returncode == 0, 'Selected train metadata probe failed')
        rec = gate.derive(dict(video_id=row['video_id'], split='train', probe=json.loads(result.stdout)))
    need(rec['valid'], 'Selected train metadata invalid')
    return rec


def score_frame(bgr, cfg):
    # No resize/crop/normalization. Original uint8 BGR -> rounded uint8 gray -> float64.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    s = cfg['scoring']
    kw = dict(scale=s['scale'], delta=s['delta'], borderType=cv2.BORDER_REFLECT_101)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=s['laplacian_ksize'], **kw)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=s['sobel_ksize'], **kw)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=s['sobel_ksize'], **kw)
    return float(lap.var(ddof=0)), float(np.mean(gx * gx + gy * gy))


def decode(row, rec, guard, cfg, previews):
    fd = guard.open(row['video_id'], row['video_path'], 'decode')
    capture = cv2.VideoCapture(f'/proc/self/fd/{fd}', cv2.CAP_FFMPEG)
    scores = []
    try:
        need(capture.isOpened(), 'Selected train decoder failed to open')
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            i = len(scores)
            need(f"{bgr.shape[1]}x{bgr.shape[0]}" == rec['resolution'], 'Resolution changed')
            lap, ten = score_frame(bgr, cfg)
            scores.append(dict(video_id=row['video_id'], frame_index=i,
                timestamp_seconds=float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000,
                timestamp_status='pending', estimated_timestamp_seconds=i / rec['fps'],
                laplacian_variance=lap, tenengrad=ten, decode_status='ok'))
            ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY,
                                               cfg['contact_sheets']['preview_jpeg_quality']])
            need(ok, 'Preview encoding failed')
            previews[(row['video_id'], i)] = jpg.tobytes()
    finally:
        capture.release()
        os.close(fd)
    count_ok = len(scores) == rec['frame_count']
    pts = np.array([r['timestamp_seconds'] for r in scores])
    expected = np.arange(len(scores)) / rec['fps']
    reliable = bool(len(pts) and np.isfinite(pts).all() and (np.diff(pts) > 0).all()
                    and np.max(np.abs(pts - expected)) <= 0.5 / rec['fps']
                    and rec['fps_rational'] == rec['r_frame_rate'])
    for r in scores:
        r['timestamp_status'] = 'decoder_monotonic_cfr_consistent' if reliable else 'unverified_use_frame_index'
        if not reliable:
            r['timestamp_seconds'] = None
    # A short read is an uncertainty/failure, never a low-quality score or guessed missing index.
    return scores, dict(video_id=row['video_id'], decoded_frames=len(scores),
        metadata_frames=rec['frame_count'], decode_status='ok' if count_ok else 'frame_count_mismatch',
        timestamp_verified=reliable, terminal_read='false_at_expected_count' if count_ok else 'unexpected_stop')


def runs(mask):
    result, start = [], None
    for i, val in enumerate(list(mask) + [False]):
        if val and start is None:
            start = i
        if not val and start is not None:
            result.append((start, i - 1))
            start = None
    return result


def smooth(values, width):
    half = width // 2
    return np.array([np.median(values[max(0, i-half):min(len(values), i+half+1)])
                     for i in range(len(values))])


def fitted_thresholds(frames, cfg):
    t = cfg['thresholds']
    return {key: float(np.quantile([r[key] for r in frames], t['quantile'], method=t['quantile_method']))
            for key in METRICS}


def flags_for(frames, fps, thresholds, cfg):
    t = cfg['thresholds']
    lap, ten = (np.array([r[k] for r in frames]) for k in METRICS)
    a, b = lap <= thresholds[METRICS[0]], ten <= thresholds[METRICS[1]]
    # Smallest odd number of frames spanning at least the configured nominal duration.
    width = math.ceil(t['smoothing_seconds'] * fps)
    width += int(width % 2 == 0)
    sm = ((smooth(lap, width) <= thresholds[METRICS[0]]) &
          (smooth(ten, width) <= thresholds[METRICS[1]]))
    keep = np.zeros(len(sm), dtype=bool)
    for lo, hi in runs(sm):
        if (hi-lo+1) / fps >= t['minimum_segment_seconds']:
            keep[lo:hi+1] = True
    return dict(zip(METHODS, (a, b, a & b, keep))), width


def summarize_video(row, frames, flags, cfg):
    fps = float(row['fps'])
    result = dict(video_id=row['video_id'], label=row['label'], source_group=row['source_group'],
                  fps=row['fps'], frame_count=len(frames), strategies={})
    segment_rows = []
    for method, mask in flags.items():
        bad, usable = runs(mask), runs(~mask)
        longest = max((hi-lo+1 for lo, hi in bad), default=0)
        good_lengths = [hi-lo+1 for lo, hi in usable]
        result['strategies'][method] = dict(candidate_frames=int(mask.sum()),
            candidate_ratio=float(mask.mean()), candidate_segments=len(bad),
            longest_candidate_frames=longest, longest_candidate_seconds=longest / fps,
            usable_segments=len(usable), usable_frames=int((~mask).sum()),
            longest_usable_frames=max(good_lengths, default=0),
            usable_windows={str(w):sum(max(0, n-w+1) for n in good_lengths) for w in cfg['windows']})
        for kind, spans in [('candidate', bad), ('unflagged', usable)]:
            for lo, hi in spans:
                segment_rows.append(dict(video_id=row['video_id'], strategy=method, kind=kind,
                    start_frame=lo, end_frame=hi, start_timestamp=frames[lo]['timestamp_seconds'],
                    end_timestamp=frames[hi]['timestamp_seconds'], frame_count=hi-lo+1,
                    nominal_duration_seconds=(hi-lo+1)/fps))
    return result, segment_rows


def describe(values):
    arr = np.array(values, dtype=float)
    return dict(n=len(arr), min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()),
        **{key:float(np.quantile(arr, q, method='linear')) for key,q in
           [('p05',.05),('p10',.10),('p25',.25),('median',.5),('p75',.75),('p90',.9),('p95',.95)]})


def average_ranks(values):
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def aggregate(videos, cfg):
    total = sum(v['frame_count'] for v in videos)
    return {method:dict(candidate_frames=sum(v['strategies'][method]['candidate_frames'] for v in videos),
        candidate_ratio=sum(v['strategies'][method]['candidate_frames'] for v in videos) / total,
        candidate_segments=sum(v['strategies'][method]['candidate_segments'] for v in videos),
        affected_videos=sum(v['strategies'][method]['candidate_frames'] > 0 for v in videos),
        total_videos=len(videos), total_frames=total,
        videos_with_usable_window={str(w):sum(v['strategies'][method]['usable_windows'][str(w)] > 0
                                           for v in videos) for w in cfg['windows']}) for method in METHODS}


def make_sheets(frames, previews, cfg, out):
    entries = []
    font = ImageFont.load_default(size=14)
    for metric in METRICS:
        ordered = sorted(frames, key=lambda r: (r[metric], r['video_id'], r['frame_index']))
        tiles = []
        for band, q in cfg['contact_sheets']['bands'].items():
            # Nearest quantile rank, distinct videos to avoid near-identical adjacent frames.
            target = round(q * (len(ordered)-1))
            used = set()
            for idx in sorted(range(len(ordered)), key=lambda i: (abs(i-target), i)):
                r = ordered[idx]
                if r['video_id'] in used:
                    continue
                used.add(r['video_id'])
                tile = Image.new('RGB', (390, 345), '#151b26')
                im = Image.open(io.BytesIO(previews[(r['video_id'], r['frame_index'])])).convert('RGB')
                im.thumbnail((380, 230))
                tile.paste(im, ((390-im.width)//2, 4))
                draw = ImageDraw.Draw(tile)
                ts = 'unverified' if r['timestamp_seconds'] is None else f"{r['timestamp_seconds']:.4f}s"
                label = (f"{band} | {metric}\n{r['video_id']}\nf={r['frame_index']} t={ts}\n"
                         f"L={r['laplacian_variance']:.3f} T={r['tenengrad']:.3f}\n"
                         f"candidate={r['candidate_flag']} ({cfg['thresholds']['primary_candidate']})\n"
                         f"{cfg['threshold_version']}")
                draw.multiline_text((7, 236), label, fill='white', font=font, spacing=2)
                tiles.append(tile)
                entries.append(dict(metric=metric, band=band, **r))
                if len(used) == cfg['contact_sheets']['examples_per_band']:
                    break
        cols = cfg['contact_sheets']['examples_per_band']
        sheet = Image.new('RGB', (390 * cols, 345 * 3), '#151b26')
        for i, tile in enumerate(tiles):
            sheet.paste(tile, ((i % cols)*390, (i // cols)*345))
        sheet.save(out / 'contact_sheets' / f'{metric}_low_middle_high.png')
    write_csv(out / 'contact_sheets/index.csv', entries)


def markdown(report):
    lines = ['# Basketball-51 自動模糊品質篩檢 Pilot v1', '',
        f"狀態：**{report['status']}**；train-only；候選標記不是人工模糊真值，沒有正式清理。", '',
        f"開始時 branch `{report['provenance']['branch']}`，HEAD `{report['provenance']['head']}`；工作樹乾淨。", '',
        f"固定 seed {report['selection']['seed']}；{report['selection']['rule']}。交替 Miss/Make，優先尚未使用的來源群組，SHA-256(seed:video_id) 排序破同分，最後按 video_id 排序。", '',
        f"影片 {len(report['videos'])}；成功評分影格 {report['frame_count']}；來源群組 {report['selection']['source_groups']}。", '',
        f"Metadata：{report['metadata_source']}。沒有因分數或 probe 結果更換樣本。", '',
        '## 指標分布及相關性', '',
        '| 指標 | 最小 | P10 | 中位數 | P90 | 最大 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for metric, dist in report['distributions'].items():
        lines.append(f"| {metric} | {dist['min']:.6f} | {dist['p10']:.6f} | {dist['median']:.6f} | {dist['p90']:.6f} | {dist['max']:.6f} |")
    lines += ['', f"逐影格 Pearson = {report['correlation']['pearson']:.6f}；Spearman = {report['correlation']['spearman']:.6f}。影格有時間相關，不能視為獨立樣本推論。", '',
        '## 探索性門檻及影響', '',
        f"版本 `{report['threshold_version']}`，Pilot train 全影格等權 P{report['config']['thresholds']['quantile']*100:g}，linear quantile；比較使用 ≤。", '',
        f"Laplacian ≤ {report['thresholds']['laplacian_variance']:.9f}；Tenengrad ≤ {report['thresholds']['tenengrad']:.9f}。只適用本 Pilot 尺寸、解碼、數值定義，不能當正式清理規則。", '',
        '| 策略 | 候選影格 | 比例 | 連續區段 | 涉及影片 | 有16/32/64未標記連續窗的影片 |', '| --- | ---: | ---: | ---: | ---: | --- |']
    for method, a in report['impact'].items():
        lines.append(f"| {method} | {a['candidate_frames']} | {a['candidate_ratio']:.2%} | {a['candidate_segments']} | {a['affected_videos']}/{a['total_videos']} | " + '/'.join(str(a['videos_with_usable_window'][str(w)]) for w in report['config']['windows']) + ' |')
    lines += ['', '區段端點 frame index 為零起始、兩端包含；timestamp 為解碼器時間且需通過單調及 CFR 一致性檢查。區段名義長度是 n/FPS，首末 timestamp 差為 (n−1)/FPS。未標記不代表內容可用；窗口只提供可行性資訊，不選定模型輸入長度。', '', '## 類別、來源與 FPS', '']
    for dimension, groups in report['groups'].items():
        lines += [f'### {dimension}', '', '| 群組 | 影片/影格 | Lap低 | Ten低 | 雙低 | 平滑雙低 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        for group, impacts in groups.items():
            first = impacts[METHODS[0]]
            lines.append(f"| {group} | {first['total_videos']}/{first['total_frames']} | " + ' | '.join(f"{impacts[m]['candidate_ratio']:.2%}" for m in METHODS) + ' |')
    lines += ['', '來源代碼尚未核實為 game ID。每組僅少量影片；刻意增加來源覆蓋與類別平衡，因此比例不是完整 train 盛行率估計，類別/FPS/來源差異互有混雜。', '',
        '## 存取與驗證證據', '', '```json', dumps(report['access']).rstrip(), '```', '',
        '所有影片入口先經固定 train ID/path allowlist；probe 與 decode ledger 保留於本機 runtime_logs。非 train ID/path 拒絕測試不開啟影片。外部 syscall trace 與独立重算結果見 verification.json。', '',
        '## 本機產物', '', f"根目錄：`{report['output_dir']}`。", '',
        '- `frame_scores/`：每影片逐影格 CSV、四策略 flags/reasons、decode status。',
        '- `segments.csv`、`video_summary.json`：候選與未標記區段、最長區段、连续窗口摘要。',
        '- `contact_sheets/laplacian_variance_low_middle_high.png`、`tenengrad_low_middle_high.png` 與 index.csv：各低/中/高三張不同影片畫面，共 18 tiles，僅檢查演算法反應。',
        '- `runtime_logs/`：probe metadata、decode status、存取 ledger、版本、syscall trace 與驗證。', '',
        '## 能回答、不能回答與下一步', '',
        '可回答原始解析度全畫面局部強度變化的相對強弱、探索性門檻會標記多少影格及時間區段，以及各群組受到的影響。不能證明真實模糊、球或投籃者清晰度、模型效果或正式資料可用性。低紋理可能低分；運動模糊與失焦不能可靠區分；高紋理背景可能掩蓋主體模糊；全畫面清晰不等於球或投籃者清晰；兩指標分數不是人工模糊真值。', '',
        'Pilot 提供後續完整 train 評分的技術基礎；現行 CLI 上限 40 支，仍需另行授權與容量規劃，沒有自動擴展。正式清理政策仍待決定：品質定義、主體區域、跨來源/解析度校準、門檻與錯誤成本、持續時間、短片處理、保留或排除的審核規則。不得直接將候選轉為排除清單。', '',
        '原始影片、固定 manifest/split、A1/A2/B/C frozen artifacts 未修改；沒有刪格、拼接、生成清理後影片、增強、特徵擷取、融合、訓練、評估或 checkpoint loading。Gate 1C/原 D 維持暫停。', '']
    return '\n'.join(lines)


def run(cfg, config_path, seed, out, guard, selected):
    # Confine every output to ignored local artifacts; no force/overwrite facility.
    base = ROOT / 'development/data_quality/artifacts'
    need(out.resolve().is_relative_to(base.resolve()) and out.resolve() != base.resolve(),
         'Output must be a new child of development/data_quality/artifacts')
    need(not out.exists() and not out.is_symlink(), 'Output exists; refusing overwrite')
    out.mkdir(parents=True, exist_ok=False)
    for sub in ('frame_scores', 'contact_sheets', 'runtime_logs'):
        (out / sub).mkdir()
    write_json(out / 'runtime_logs/effective_config.json', cfg)
    write_json(out / 'runtime_logs/claim.json', dict(seed=seed, selected_ids=[r['video_id'] for r in selected]))
    cached, source = cached_train_metadata(cfg, guard)
    manifest, all_frames, decoding, previews, raw_meta = [], [], [], {}, []
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    cv2.setUseOptimized(False)
    try:
        for row in selected:
            rec = metadata(row, guard, cached)
            raw_meta.append(rec)
            entry = dict(video_id=row['video_id'], label='Make' if row['binary_label']=='1' else 'Miss',
                binary_label=row['binary_label'], source_group=row['source_group'], fps=rec['fps'],
                fps_rational=rec['fps_rational'], frame_count=rec['frame_count'],
                duration_seconds=rec['duration_seconds'], relative_video_path=row['video_path'],
                split='train', selection_seed=seed, selection_rule=cfg['selection_rule'])
            manifest.append(entry)
            frames, status = decode(row, rec, guard, cfg, previews)
            all_frames.extend(frames)
            decoding.append(status)
            print(f"{len(manifest):02}/{len(selected)} {row['video_id']} {len(frames)} frames {status['decode_status']}", flush=True)
    finally:
        write_json(out / 'runtime_logs/access_ledger.json', guard.events)
        write_json(out / 'runtime_logs/metadata.json', raw_meta)
        write_json(out / 'runtime_logs/decode_status.json', decoding)
    write_csv(out / 'pilot_manifest_v1.csv', manifest)
    if any(r['decode_status'] != 'ok' for r in decoding):
        if all_frames:
            write_csv(out / 'frame_scores/partial_scores.csv', all_frames)
        raise ValueError('Decode/count failure separate from quality; pilot incomplete, see runtime_logs')
    thresholds = fitted_thresholds(all_frames, cfg)
    videos, segments = [], []
    for row in manifest:
        frames = [r for r in all_frames if r['video_id'] == row['video_id']]
        flags, width = flags_for(frames, row['fps'], thresholds, cfg)
        for i, frame in enumerate(frames):
            for method in METHODS:
                frame[method] = int(flags[method][i])
            frame['candidate_flag'] = frame[cfg['thresholds']['primary_candidate']]
            frame['flag_reason'] = ';'.join(m for m in METHODS if frame[m]) or 'none'
            frame['threshold_version'] = cfg['threshold_version']
        summary, spans = summarize_video(row, frames, flags, cfg)
        summary['smoothing_window_frames'] = width
        videos.append(summary)
        segments.extend(spans)
        write_csv(out / 'frame_scores' / (Path(row['video_id']).stem + '.csv'), frames)
    write_csv(out / 'segments.csv', segments)
    write_json(out / 'video_summary.json', videos)
    make_sheets(all_frames, previews, cfg, out)
    a, b = (np.array([r[k] for r in all_frames]) for k in METRICS)
    versions = dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
        pyyaml=yaml.__version__, pillow=PIL.__version__,
        ffprobe=subprocess.check_output(['ffprobe', '-version'], text=True).splitlines()[0])
    with (out / 'runtime_logs/opencv_build.txt').open('x') as f:
        f.write(cv2.getBuildInformation())
    write_json(out / 'runtime_logs/versions.json', versions)
    known_fps = set(json.loads((ROOT / 'docs/i3d_temporal_feasibility_audit.json').read_text())['statistics']['splits']['train']['fps_rational'])
    observed_fps = {r['fps_rational'] for r in manifest}
    access = {split:{'opens':sum(e['split']==split for e in guard.events),
                     'probes':sum(e['split']==split and e['operation']=='probe' for e in guard.events),
                     'decodes':sum(e['split']==split and e['operation']=='decode' for e in guard.events)}
              for split in ('train','validation','test')}
    report = dict(schema_version=1, status='COMPLETE' if known_fps <= observed_fps else 'INCOMPLETE_FPS_COVERAGE',
        provenance=dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            started_utc=datetime.now(timezone.utc).isoformat(), initial_worktree='clean (session preflight)',
            input_hashes=guard.scope['input_hashes'], script_sha256=gate.digest(Path(__file__)),
            config_sha256=gate.digest(config_path), versions=versions),
        config=cfg, selection=dict(seed=seed, rule=cfg['selection_rule'], source_groups=len({r['source_group'] for r in manifest}),
            counts=dict(Counter(r['label'] for r in manifest)), fps_counts=dict(Counter(str(r['fps']) for r in manifest)),
            known_train_fps=sorted(known_fps), observed_fps=sorted(observed_fps),
            id_sha256=gate.objhash([r['video_id'] for r in manifest])),
        metadata_source=source, frame_count=len(all_frames), decode_failures=0,
        distributions={k:describe([r[k] for r in all_frames]) for k in METRICS},
        correlation=dict(pearson=float(np.corrcoef(a,b)[0,1]),
                         spearman=float(np.corrcoef(average_ranks(a),average_ranks(b))[0,1])),
        threshold_version=cfg['threshold_version'], thresholds=thresholds, impact=aggregate(videos,cfg),
        groups={dim:{key:aggregate([v for v in videos if str(v[dim])==key],cfg)
                     for key in sorted({str(v[dim]) for v in videos})}
                for dim in ('label','source_group','fps')},
        access=access, output_dir=str(out.relative_to(ROOT)), videos=videos)
    write_json(out / 'blur_screening_pilot_v1.json', report)
    with (out / 'blur_screening_pilot_v1.md').open('x') as f:
        f.write(markdown(report))
    print(dumps(dict(status=report['status'],frames=len(all_frames),access=access)))
    return 0 if report['status']=='COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--dry-run', action='store_true', help='Metadata strings only; no video stat/open/probe/decode or writes')
    parser.add_argument('--output-dir', type=Path, help='New child of ignored artifacts directory; existing path rejected')
    parser.add_argument('--seed', type=int, help='Override deterministic selection seed')
    parser.add_argument('--check-video-id', help='Guard-only check; optionally pair with --check-video-path; never opens video')
    parser.add_argument('--check-video-path', help='Exact manifest-relative path for guard-only check')
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg['seed'] if args.seed is None else args.seed
    guard = TrainGuard()
    if args.check_video_id is not None:
        guard.authorize(args.check_video_id, args.check_video_path)
        print(dumps(dict(status='AUTHORIZED_TRAIN_METADATA_ONLY',video_opens=0)))
        return 0
    need(args.check_video_path is None, '--check-video-path requires --check-video-id')
    selected = choose(guard, cfg, seed)
    if args.dry_run:
        for r in selected:
            guard.authorize(r['video_id'],r['video_path'])
        print(dumps(dict(status='DRY_RUN_PASS',seed=seed, selected_ids=[r['video_id'] for r in selected],
              labels=dict(Counter(r['binary_label'] for r in selected)),
              source_groups=len({r['source_group'] for r in selected}),
              video_opens=0, probes=0, decodes=0, scope=guard.scope)))
        return 0
    out = args.output_dir or ROOT / cfg['output_dir']
    if not out.is_absolute():
        out = ROOT / out
    return run(cfg, args.config, seed, out, guard, selected)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
