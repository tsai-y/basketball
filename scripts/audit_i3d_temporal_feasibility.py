#!/usr/bin/env python3
"""Gate 1A metadata-only audit. Default --mode check never opens any video."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'archive/Basketball_51 dataset'
MANIFEST = ROOT / 'data/processed/dataset_manifest.csv'
SPLIT = ROOT / 'data/processed/split_summary.json'
EXPECTED = {'train': 7193, 'validation': 1547}
HASHES = {'manifest': 'e7bba1acd6028cf67cdf28251e69852e82cff201e13519ed05970ac483cce6b6',
          'split_summary': '953da31313976763c8f5d597da5f0fe5a6a45e10f5d5edcd42fb0c120af2f7c5'}
THRESHOLDS = (16, 32, 64, 96, 128)
ENTRIES = 'stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,time_base,duration_ts:format=format_name,duration'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def objhash(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def safe_relative(value):
    p = PurePosixPath(value)
    need(not p.is_absolute() and '..' not in p.parts and str(p) == value
         and len(p.parts) == 2 and p.suffix == '.mp4', 'Unsafe manifest path (redacted)')
    return value


def load_scope(splits):
    need((ROOT / 'docs/research_baseline_v1.3.md').is_file(), 'Missing research baseline; stop')
    need(set(splits) <= EXPECTED.keys() and len(splits) == len(set(splits)), 'Forbidden or duplicate split')
    actual = {'manifest': digest(MANIFEST), 'split_summary': digest(SPLIT)}
    need(actual == HASHES, 'Frozen manifest/split hash mismatch; stop before video access')
    with MANIFEST.open(newline='') as f:
        rows = list(csv.DictReader(f))
    # No filesystem lookup of test/excluded paths: denylist is metadata strings only.
    denied = {safe_relative(r['video_path']) for r in rows if r['split'] not in EXPECTED}
    denied_ids = {r['video_id'] for r in rows if r['split'] not in EXPECTED}
    counts = Counter(r['split'] for r in rows)
    need(counts == {'train': 7193, 'validation': 1547, 'test': 1539, 'excluded': 32}, 'Unexpected manifest counts')
    need(len({r['video_id'] for r in rows}) == len(rows), 'Duplicate manifest video ID; stop')
    need(len({r['video_path'] for r in rows}) == len(rows), 'Duplicate manifest path; stop')
    summary = json.loads(SPLIT.read_text())
    selected = sorted((r for r in rows if r['split'] in splits), key=lambda r: r['video_id'])
    for r in selected:
        safe_relative(r['video_path'])
        need(r['excluded'] == '0' and r['video_id'] == r['video_path'], 'Unexpected selected row')
        need(r['video_path'] not in denied and r['video_id'] not in denied_ids, 'Forbidden path/ID; stop')
        need(r['original_class'] in ('2p0','2p1','3p0','3p1','ft0','ft1','mp0','mp1'), 'Invalid class')
        need(r['binary_label'] == r['original_class'][-1], 'Invalid label')
        need(summary['group_assignment'][r['source_group']] == r['split'], 'Group split mismatch')
    for s in splits:
        need(dict(Counter(r['binary_label'] for r in selected if r['split'] == s)) == summary['label_counts'][s], 'Label counts mismatch')
    return selected, denied, {'input_hashes': actual, 'manifest_counts': dict(sorted(counts.items())),
                             'duplicate_rows': 0, 'duplicate_ids': 0, 'duplicate_paths': 0}


def authorize(row, allowed, denied):
    # Exact allowlist equality prevents injected test rows even if relabelled train.
    need(row.get('split') in EXPECTED and allowed.get(row.get('video_id')) == row,
         'Unauthorized row; stop before any filesystem/video access')
    need(row['video_path'] not in denied, 'Forbidden path; stop before access')
    rel = safe_relative(row['video_path'])
    return DATA / rel


def open_guarded(path):
    # Reject symlinks at every component, including dataset parents; never follow a test alias.
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        need(not current.is_symlink(), 'Symlink rejected before video open')
    info = path.stat()
    need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, 'Nonregular/hardlinked video rejected')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    after = os.fstat(fd)
    if (info.st_dev, info.st_ino) != (after.st_dev, after.st_ino) or after.st_nlink != 1:
        os.close(fd)
        raise ValueError('Video changed during open')
    return fd


def probe(row, allowed, denied):
    path = authorize(row, allowed, denied)
    rec = {k: row[k] for k in ('video_id','split','original_class','binary_label')}
    rec.update(found=False, video_open_count=0, probe_count=0, decode_count=0)
    try:
        fd = open_guarded(path)
        rec.update(found=True, video_open_count=1, probe_count=1)
        try:
            # Pass an already authorized descriptor, so ffprobe cannot race a path replacement.
            proc = subprocess.run(['ffprobe','-v','error','-protocol_whitelist','file,pipe',
                                   '-show_entries',ENTRIES,'-of','json',f'/proc/self/fd/{fd}'],
                                  pass_fds=(fd,), capture_output=True, text=True, timeout=30)
        finally:
            os.close(fd)
        if proc.returncode:
            rec['error'] = f'ffprobe_exit_{proc.returncode}'
        else:
            rec['probe'] = json.loads(proc.stdout)
    except FileNotFoundError:
        rec['error'] = 'missing_path_or_tool'
    except subprocess.TimeoutExpired:
        rec['error'] = 'probe_timeout'
    except (OSError, ValueError) as exc:
        # Do not emit exception paths or arbitrary ffprobe stderr.
        rec['error'] = type(exc).__name__
    return rec


def positive(value):
    try:
        q = Fraction(str(value))
        return q if q > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def derive(rec):
    r = dict(rec)
    issues = []
    streams = [s for s in r.get('probe',{}).get('streams',[]) if s.get('codec_type') == 'video']
    if len(streams) != 1:
        r.update(issues=['probe_failed' if 'error' in r else 'video_stream_count'], valid=False)
        return r
    s = streams[0]
    fmt = r['probe'].get('format',{})
    fps = positive(s.get('avg_frame_rate'))
    frames = positive(s.get('nb_frames'))
    if frames is not None and frames.denominator != 1:
        frames = None
    duration = positive(s.get('duration'))
    fallback = duration is None
    if fallback:
        duration = positive(fmt.get('duration'))
    if fps is None: issues.append('invalid_fps')
    if frames is None: issues.append('invalid_frame_count')
    if duration is None: issues.append('invalid_duration')
    if (s.get('width'),s.get('height')) != (320,240): issues.append('unexpected_resolution')
    if s.get('codec_name') != 'h264': issues.append('unexpected_codec')
    if 'mp4' not in fmt.get('format_name','').split(','): issues.append('unexpected_container')
    r.update(fps_rational=s.get('avg_frame_rate'), r_frame_rate=s.get('r_frame_rate'),
             fps=float(fps) if fps else None, frame_count=int(frames) if frames else None,
             duration_seconds=float(duration) if duration else None,
             duration_exact=str(duration) if duration else None,
             duration_source='format_fallback' if fallback else 'video_stream',
             stream_duration=s.get('duration'), container_duration=fmt.get('duration'),
             resolution=f"{s.get('width')}x{s.get('height')}", codec=s.get('codec_name'),
             container=fmt.get('format_name'))
    r['valid'] = fps is not None and frames is not None and duration is not None
    if r['valid']:
        eq = duration * 25
        available = math.floor(eq)
        r.update(equivalent_frames_25fps=float(eq), equivalent_frames_25fps_exact=str(eq),
                 equivalent_floor=available, equivalent_ceil=math.ceil(eq),
                 equivalent_round_half_up=math.floor(eq + Fraction(1,2)))
        delta = frames/fps - duration
        limit = max(Fraction(1,10), 1/fps)
        r.update(frame_count_over_fps_seconds=float(frames/fps), delta_seconds=float(delta),
                 absolute_delta_seconds=float(abs(delta)), inconsistency_limit_seconds=float(limit))
        if abs(delta) > limit: issues.append('duration_frame_fps_mismatch')
        container_duration = positive(fmt.get('duration'))
        r['container_minus_stream_seconds'] = float(container_duration-duration) if container_duration else None
        if positive(s.get('r_frame_rate')) != fps: issues.append('avg_vs_r_frame_rate_mismatch')
        if eq < 64:
            r['looping'] = {'available_frames': available, 'unrounded_frames': float(eq),
                            'repeat_factor': math.ceil(Fraction(64,available)) if available > 0 else None,
                            'duplicate_positions': 64-available if available > 0 else None,
                            'duplicate_exposure_ratio': (64-available)/64 if available > 0 else None}
            if available == 0: issues.append('looping_undefined_zero_available')
    r['issues'] = issues
    return r


def describe(values):
    v = sorted(float(x) for x in values)
    if not v: return {'n': 0, **{k: None for k in ('min','mean','median','p05','p10','p25','p75','p90','p95','max')}}
    def percentile(p):
        x = (len(v)-1)*p
        lo = math.floor(x)
        hi = math.ceil(x)
        return v[lo] + (v[hi]-v[lo])*(x-lo)
    return {'n':len(v),'min':v[0],'mean':math.fsum(v)/len(v),'median':percentile(.5),
            **{f'p{p:02}':percentile(p/100) for p in (5,10,25,75,90,95)},'max':v[-1]}


def ratio(n,d):
    return {'count':n,'denominator':d,'ratio':n/d if d else None}


def thresholds(values, limits):
    return {str(t):ratio(sum(x < Fraction(str(t)) for x in values),len(values)) for t in limits}


def distribution(values):
    return dict(sorted(Counter(str(x) for x in values).items()))


def aggregate(records):
    good = [r for r in records if r['valid']]
    eq = [Fraction(r['equivalent_frames_25fps_exact']) for r in good]
    durations = [Fraction(r['duration_exact']) for r in good]
    frames = [r['frame_count'] for r in good]
    short = [r for r in good if Fraction(r['equivalent_frames_25fps_exact']) < 64]
    loop = [r['looping'] for r in short if r['looping']['repeat_factor'] is not None]
    groups = []
    for r in good:
        v = r['fps']
        groups.append(next((str(x) for x in (23.976,25,29.97) if abs(v-x)<.001), 'other'))
    return {'n':len(records), 'valid_n':len(good),
            'fps_rational':distribution(r['fps_rational'] for r in good),
            'r_frame_rate':distribution(r['r_frame_rate'] for r in good),
            'fps_float':distribution(r['fps'] for r in good), 'fps_groups':distribution(groups),
            'fps_other_video_ids':[r['video_id'] for r,g in zip(good,groups) if g=='other'],
            'frame_count':describe(frames), 'duration_seconds':describe(durations),
            'equivalent_frames_25fps':describe(eq),
            'raw_below':thresholds(frames,THRESHOLDS),
            'duration_below':thresholds(durations,('.64','1.28','2.56','3.84','5.12')),
            'equivalent_unrounded_below':thresholds(eq,THRESHOLDS),
            'rounding_sensitivity':{key:thresholds([r[key] for r in good],THRESHOLDS)
                                    for key in ('equivalent_floor','equivalent_round_half_up','equivalent_ceil')},
            'codec':distribution(r['codec'] for r in good),
            'resolution':distribution(r['resolution'] for r in good),
            'container':distribution(r['container'] for r in good),
            'duration_source':distribution(r['duration_source'] for r in good),
            'metadata_signed_delta_seconds':describe(r['delta_seconds'] for r in good),
            'metadata_absolute_delta_seconds':describe(r['absolute_delta_seconds'] for r in good),
            'container_minus_stream_seconds':describe(r['container_minus_stream_seconds'] for r in good if r['container_minus_stream_seconds'] is not None),
            'looping':{'short_n':len(short), 'defined_n':len(loop), 'undefined_n':len(short)-len(loop),
                       'available_frames':describe(x['available_frames'] for x in loop),
                       'repeat_factor':describe(x['repeat_factor'] for x in loop),
                       'repeat_factor_counts':distribution(x['repeat_factor'] for x in loop),
                       'duplicate_positions':describe(x['duplicate_positions'] for x in loop),
                       'duplicate_exposure_ratio':describe(x['duplicate_exposure_ratio'] for x in loop),
                       'cycles_2':sum(x['repeat_factor']==2 for x in loop),
                       'cycles_3':sum(x['repeat_factor']==3 for x in loop),
                       'cycles_4_or_more':sum(x['repeat_factor']>=4 for x in loop),
                       'worst_cases':sorted([{'video_id':r['video_id'],**r['looping']} for r in short],
                                            key=lambda x:(-(x['duplicate_exposure_ratio'] or 0),x['video_id']))[:10]}}


def summarize(records, scope, splits):
    groups = {}
    for name in (*splits,'combined'):
        subset = records if name=='combined' else [r for r in records if r['split']==name]
        groups[name] = aggregate(subset)
    grouped = {}
    for scope_name in (*splits,'combined'):
        subset = records if scope_name=='combined' else [r for r in records if r['split']==scope_name]
        grouped[scope_name] = {}
        for field in ('outcome','shot_type','original_class'):
            def key(r):
                return ('Make' if r['binary_label']=='1' else 'Miss') if field=='outcome' else (r['original_class'][:-1].upper() if field=='shot_type' else r['original_class'])
            grouped[scope_name][field] = {g:aggregate([r for r in subset if key(r)==g]) for g in sorted({key(r) for r in subset})}
    issues = [{'video_id':r['video_id'],'split':r['split'],'issues':r['issues'],
               **({k:r[k] for k in ('frame_count','fps','duration_seconds','delta_seconds') if k in r})}
              for r in records if r['issues']]
    complete = len(records)==sum(EXPECTED[s] for s in splits) and all(r['valid'] for r in records)
    return {'scope':scope,'status':'COMPLETE' if complete else 'INCOMPLETE',
            'integrity':{s:{'expected':EXPECTED[s], 'selected':sum(r['split']==s for r in records),
                            'found':sum(r['split']==s and r['found'] for r in records),
                            'probe_attempts':sum(r['probe_count'] for r in records if r['split']==s),
                            'probe_failures':sum('error' in r for r in records if r['split']==s),
                            'invalid_metadata':sum(not r['valid'] for r in records if r['split']==s)} for s in splits},
            'access':{'actual_splits':sorted({r['split'] for r in records if r['probe_count']}),
                      'video_open_count':sum(r['video_open_count'] for r in records),
                      'probe_count':sum(r['probe_count'] for r in records), 'decode_count':0,
                      'test_video_open_count':0,'test_probe_count':0,'test_decode_count':0,
                      'selected_id_sha256':objhash([r['video_id'] for r in records]),
                      'ledger_sha256':objhash([{k:r[k] for k in ('video_id','split','video_open_count','probe_count','decode_count')} for r in records])},
            'missing_paths':[r['video_id'] for r in records if not r['found']],
            'probe_errors':[{'video_id':r['video_id'],'error':r['error']} for r in records if 'error' in r],
            'issue_counts':distribution(i for r in records for i in r['issues']), 'issues':issues,
            'splits':groups, 'groups':grouped}


def validate(summary):
    def walk(x):
        if isinstance(x,dict):
            if set(x)=={'count','denominator','ratio'}:
                need(0<=x['count']<=x['denominator'], 'Invalid numerator')
                need(x['ratio']==(x['count']/x['denominator'] if x['denominator'] else None), 'Invalid proportion')
            for v in x.values(): walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(summary)
    for g in summary['splits'].values():
        need(g['duration_below']['2.56']==g['equivalent_unrounded_below']['64'], 'Duration/equivalent disagreement')
    need(summary['access']['actual_splits'] == sorted(k for k,v in summary['integrity'].items() if v['probe_attempts']), 'Access split mismatch')


def markdown(report):
    s = report['statistics']
    lines = ['# Experiment D Gate 1A — Temporal Feasibility Audit','',
             f"Status: **{s['status']}**. UTC: {report['provenance']['started_utc']}.", '',
             '本報告只描述 train／validation 的時間 metadata；不選擇 Experiment D protocol，也不是 pre-release 研究。', '',
             '## 範圍、方法與可重現性','',
             '- 既有 dataset_summary.json 僅含全資料集彙總，沒有可重用的逐影片、分 split metadata；未用其全體分布制定本報告。',
             '- 僅由 frozen manifest allowlist 開啟 train／validation；不遍歷資料目錄、不 probe test。拒絕 test row/path、symlink 與 hardlink。',
             '- ffprobe 讀取 metadata，未使用 -count_frames 或完整解碼；raw frame count 為 nb_frames，不是本輪逐格解碼計數。',
             '- duration 優先使用 video stream duration，缺失時才用 format duration；兩者原值均保留。FPS 主統計用 avg_frame_rate，同時保存 r_frame_rate。',
             '- 百分位採線性插值：h=(n−1)p，介於兩個排序值時線性插值。比例分母為該組有效 metadata 數；失敗另列，不當作零。',
             '- 25 FPS 等效值 E=duration×25，用 Fraction 精確計算並保存未取整值。主 short 判定用 E<T；另列 floor、round-half-up=floor(E+0.5)、ceil 敏感度。',
             '- Looping 僅為反事實分析：E<64 時 A=floor(E)，repeat_factor=ceil(64/A)（包含首輪），重複位置=64−A，duplicate exposure ratio=(64−A)/64；A=0 則未定義，不強制補成1。',
             '- 16/32/64/96/128 與 0.64/1.28/2.56/3.84/5.12 秒只是描述閾值，不代表選定長度或25 FPS。',
             '- Metadata 明顯不一致定義：|nb_frames/avg_fps − duration| > max(0.1 秒, 1/fps)。小差值仍完整統計；沒有修改或排除資料。',
             '- 逐影片 probe、未取整值、取整值與存取 ledger 存在 /tmp 的 cache（不進 Git）；JSON 報告只保留彙總及異常。', '',
             '```json',json.dumps(report['provenance'],ensure_ascii=False,indent=2),'```','',
             '## 完整性與零 test 存取','', '| Split | 預期 | 列數 | 找到 | Probe | Probe失敗 | 無效metadata |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for k,v in s['integrity'].items():
        lines.append(f"| {k} | {v['expected']} | {v['selected']} | {v['found']} | {v['probe_attempts']} | {v['probe_failures']} | {v['invalid_metadata']} |")
    lines += ['', '```json',json.dumps({'scope':s['scope'],'access':s['access'],'issue_counts':s['issue_counts'],'missing_paths':s['missing_paths'],'probe_errors':s['probe_errors']},indent=2),'```','']
    for name,g in s['splits'].items():
        lines += [f'## {name}（n={g["n"]}，有效={g["valid_n"]}）','',
                  '### FPS / 格式分布','', '```json',json.dumps({k:g[k] for k in ('fps_rational','r_frame_rate','fps_float','fps_groups','fps_other_video_ids','codec','resolution','container','duration_source')},indent=2),'```','',
                  '### 長度統計','', '| 量 | Min | Mean | Median | P5 | P10 | P25 | P75 | P90 | P95 | Max |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        def num(v): return 'N/A' if v is None else f'{v:.9f}'.rstrip('0').rstrip('.')
        for key in ('frame_count','duration_seconds','equivalent_frames_25fps','metadata_signed_delta_seconds','metadata_absolute_delta_seconds','container_minus_stream_seconds'):
            lines.append('| '+key+' | '+' | '.join(num(g[key][k]) for k in ('min','mean','median','p05','p10','p25','p75','p90','p95','max'))+' |')
        lines += ['', '### 閾值（嚴格少於；count/denominator, %）','', '| Frames閾值 | Raw | 秒閾值 | Duration | E未取整 | Floor | Half-up | Ceil |', '| ---: | --- | ---: | --- | --- | --- | --- | --- |']
        def cell(v): return f"{v['count']}/{v['denominator']} ({100*v['ratio']:.6f}%)" if v['ratio'] is not None else '0/0 (N/A)'
        for t,d in zip(THRESHOLDS,('.64','1.28','2.56','3.84','5.12')):
            vals=[g['equivalent_unrounded_below'][str(t)]]+[g['rounding_sensitivity'][k][str(t)] for k in ('equivalent_floor','equivalent_round_half_up','equivalent_ceil')]
            lines.append(f"| {t} | {cell(g['raw_below'][str(t)])} | {d} | {cell(g['duration_below'][d])} | "+' | '.join(cell(v) for v in vals)+' |')
        lines += ['', '### Looping exposure（只對 E<64 的子集）','', '```json',json.dumps(g['looping'],indent=2),'```','']
    lines += ['## 分組比較','', '只描述分布；完整分位數、閾值與 FPS 分布見 JSON groups。','', '| Scope | 維度 | Group | n | Raw min/mean/median/max | Duration min/mean/median/max (s) | Raw<64 | E<64 |', '| --- | --- | --- | ---: | --- | --- | --- | --- |']
    for scope,fields in s['groups'].items():
        for field,groups in fields.items():
            for name,g in groups.items():
                def compact(key): return '/'.join(num(g[key][k]) for k in ('min','mean','median','max'))
                lines.append(f"| {scope} | {field} | {name} | {g['n']} | {compact('frame_count')} | {compact('duration_seconds')} | {cell(g['raw_below']['64'])} | {cell(g['equivalent_unrounded_below']['64'])} |")
    lines += ['', '## 異常與待人工檢查項目','', '```json',json.dumps(s['issues'],ensure_ascii=False,indent=2),'```','',
              '## 解讀與未決問題','']
    g=s['splits']['combined']
    lines += [f"- 原始 <64 frames：{cell(g['raw_below']['64'])}；25 FPS 等效 <64 frames／<2.56秒：{cell(g['equivalent_unrounded_below']['64'])}。",
              '- 若 short 子集為空，looping 分布與最嚴重案例為空／N/A，而非聲稱一個未觀察到的曝光比率。',
              '- 長度充足僅表示 metadata 層面的64-frame／2.56秒窗口相容；未保證實際重採樣端點、內容覆蓋、逐格時間戳或模型可行性。',
              '- 不同原生 FPS 下固定 frame count 對應不同秒數；後續 protocol 須明定時間基準、duration來源、取整／端點及短片處理，但本輪不選擇任何政策。',
              '- 本輪無完整解碼，不能重新證明逐格可解碼性或排除 VFR／PTS 問題；小 metadata 差值不等於影像損壞。',
              '- Gate 1A 只涵蓋時間資料稽核；權重 provenance、shape、VRAM、throughput 與其他 Gate 未在本輪執行。',
              '- 未實作 I3D、未下載 checkpoint、未訓練／評估、未讀取 test 影片、未修改 A1/A2/B/C、未 commit 或 push。','',
              '## 重現命令','', '```bash',
              'python3 -B scripts/audit_i3d_temporal_feasibility.py --help',
              'python3 -B scripts/audit_i3d_temporal_feasibility.py --mode check',
              'python3 -B scripts/audit_i3d_temporal_feasibility.py --mode audit',
              'python3 -B scripts/audit_i3d_temporal_feasibility.py --mode replay', '```','',
              'audit 重做 allowlist metadata probe；replay 僅讀 /tmp cache、核對 hash／身份並重算，不開影片。重跑統計 hash 應相同；執行時間與來源 provenance 可不同。', '',
              f"Statistics SHA-256: `{report['statistics_sha256']}`", '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check','audit','replay'), default='check')
    parser.add_argument('--splits', nargs='+', choices=tuple(EXPECTED), default=list(EXPECTED))
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    need(1<=args.workers<=32,'workers must be 1..32')
    started=datetime.now(timezone.utc).isoformat()
    tick=time.monotonic()
    rows,denied,scope=load_scope(args.splits)
    if args.mode=='check':
        print(json.dumps({'status':'PASS','scope':scope,'selected':len(rows),'video_opens':0,'probe_count':0,'test_video_accesses':0},indent=2))
        return 0
    need(set(args.splits)==set(EXPECTED),'Full report requires both train and validation')
    args.splits=list(EXPECTED)
    cache=Path('/tmp/basketball51_gate1a_metadata.json')
    need(not cache.is_symlink(),'Cache symlink rejected')
    allowed={r['video_id']:r for r in rows}
    if args.mode=='audit':
        version=subprocess.run(['ffprobe','-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records=list(pool.map(lambda r:probe(r,allowed,denied),rows))
        records=[derive(r) for r in records]
        provenance={'started_utc':started,'elapsed_seconds':round(time.monotonic()-tick,3),
                    'python':platform.python_version(),'ffprobe':version,
                    'command':['python3','-B','scripts/audit_i3d_temporal_feasibility.py',*sys.argv[1:]],
                    'effective_arguments':vars(args),'script_sha256':digest(Path(__file__)),
                    'ffprobe_template':['ffprobe','-v','error','-protocol_whitelist','file,pipe','-show_entries',ENTRIES,'-of','json','<authorized_fd>'],
                    'cache_basename':cache.name,'metadata_source':'fresh_ffprobe_no_full_decode'}
        payload={'scope':scope,'records':records,'provenance':provenance}
        cache.write_text(canonical(payload)+'\n')
    else:
        payload=json.loads(cache.read_text())
        need(payload['scope']==scope,'Cache input hashes/scope mismatch')
        records=payload['records']
        need([r['video_id'] for r in records]==[r['video_id'] for r in rows], 'Cache IDs mismatch (redacted)')
        for r,row in zip(records,rows):
            need(all(r[k]==row[k] for k in ('video_id','split','original_class','binary_label')), 'Cache row mismatch (redacted)')
        records=[derive({k:v for k,v in r.items() if k in ('video_id','split','original_class','binary_label','found','video_open_count','probe_count','decode_count','probe','error')}) for r in records]
        provenance=payload['provenance']
    result=summarize(records,scope,args.splits)
    validate(result)
    # Replay using reversed raw records, restored ID ordering: deterministic aggregation check.
    again=summarize(sorted(records[::-1],key=lambda r:r['video_id']),scope,args.splits)
    need(canonical(result)==canonical(again),'Nondeterministic statistics')
    report={'schema_version':1,'provenance':provenance,'statistics_sha256':objhash(result),'statistics':result}
    report['provenance']['cache_sha256']=digest(cache)
    target=ROOT/'docs/i3d_temporal_feasibility_audit.json'
    md=ROOT/'docs/i3d_temporal_feasibility_audit.md'
    need(not target.is_symlink() and not md.is_symlink(),'Report symlink rejected')
    if args.mode=='replay':
        previous=json.loads(target.read_text())
        need(previous['statistics_sha256']==report['statistics_sha256'],'Replay statistics changed')
        need(md.read_text()==markdown(previous),'Markdown differs from JSON renderer')
        print(json.dumps({'status':'REPLAY_PASS','statistics_sha256':report['statistics_sha256'],'current_video_opens':0,'current_probes':0,'json_markdown_consistent':True}))
    else:
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        md.write_text(markdown(report))
        print(json.dumps({'status':result['status'],'statistics_sha256':report['statistics_sha256'],'access':result['access'],'issues':result['issue_counts']}))
    return 0 if result['status']=='COMPLETE' else 2


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,OSError,KeyError) as exc:
        print(f'AUDIT STOPPED: {type(exc).__name__}: {exc}',file=sys.stderr)
        sys.exit(2)
