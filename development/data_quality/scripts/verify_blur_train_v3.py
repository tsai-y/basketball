#!/usr/bin/env python3
"""Independent derived-score and access verification. Never opens dataset videos."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys
import numpy as np
import audit_blur_train_v3 as s


def spans(mask, fps):
    result, start = [], 0
    for flag, group in itertools.groupby(mask):
        length = sum(1 for _ in group)
        if flag:
            result.append(dict(start_frame=start, end_frame_exclusive=start+length,
                start_seconds=start/fps, end_seconds_exclusive=(start+length)/fps,
                duration_seconds=length/fps, frames=length))
        start += length
    return result


def reference_z(values, fps, cfg):
    width = math.ceil(cfg['rolling_seconds']*fps) | 1
    logs = [math.log(max(x, cfg['log_floor'])) for x in values]
    result = []
    for i, value in enumerate(logs):
        window = logs[max(0,i-width//2):i+width//2+1]
        median = statistics.median(window)
        mad = statistics.median(abs(x-median) for x in window)
        result.append((value-median)/max(1.4826*mad,cfg['mad_scale_floor']))
    return np.asarray(result)


def trace_audit(trace, selected):
    prefix = str(s.p.gate.DATA)+'/'
    opened, probes = [], 0
    for line in trace.read_text().splitlines():
        if prefix in line:
            ids = set(re.findall(re.escape(prefix)+r'([^"<>]+\.mp4)', line))
            if not ids:
                continue
            s.need('openat(' in line and 'O_RDONLY' in line, 'Unexpected dataset syscall')
            s.need(not any(flag in line for flag in ('O_WRONLY','O_RDWR','O_CREAT','O_TRUNC')), 'Dataset write syscall')
            s.need(len(ids)==1 and ids <= selected, 'Out-of-scope video syscall')
            opened.extend(ids)
        if 'execve(' in line and '= 0' in line:
            s.need('/ffmpeg"' not in line, 'Unexpected encoder execution')
            probes += int('/ffprobe"' in line and '-show_entries' in line)
    return dict(video_open_syscalls=len(opened), unique_video_ids=len(set(opened)),
        per_video_open_syscalls=dict(sorted(Counter(opened).items())), probes=probes,
        validation_opens_probes_decodes=0, test_opens_probes_decodes=0)


def verify(out, trace=None, compare=None, initial=None):
    run = s.read_json(out/'run.json'); ctx=run['context'];cfg=ctx['config']
    s.need(ctx['scanner']==s.code_identity(), 'Scanner code changed since run')
    s.need(ctx['config_hash']==s.digest(s.DEFAULT_CONFIG), 'Current config differs from run')
    s.load_config(s.DEFAULT_CONFIG)
    guard=s.Guard()
    selected=ctx['selection']; ids={r['video_id'] for r in selected}
    if ctx['scope']=='smoke':
        allowed={r['video_id'] for r in s.choose_smoke(guard,ctx['seed'])}
        s.need(ids <= allowed and len(ids)<=20, 'Smoke scope exceeded')
    totals=Counter(); groups={dim:{} for dim in ('label','source_group','fps_rational')}
    artifact_hashes={}
    for row in selected:
        guard.authorize(row['video_id'],row['video_path'])  # strings only
        attempt=s.latest_attempt(out,row['video_id'])
        status, summary=s.inspect_attempt(attempt,row,ctx)
        s.need(status=='ok', f'Incomplete/suspect/failed output: {row["video_id"]}')
        frames=s.p.read_csv(attempt/'frames.csv'); n=len(frames);fps=summary['fps']
        scores=np.asarray([[float(r[m]) for m in s.p.METRICS] for r in frames])
        z=np.column_stack([reference_z(scores[:,j],fps,cfg) for j in range(2)])
        np.testing.assert_allclose(z,[[float(r['laplacian_local_z']),float(r['tenengrad_local_z'])] for r in frames],rtol=1e-12,atol=1e-12)
        keep=z[:,0]<=cfg['z_threshold']
        for segment in spans(keep,fps):
            if segment['frames']/fps < cfg['minimum_seconds']:
                keep[segment['start_frame']:segment['end_frame_exclusive']]=False
        np.testing.assert_array_equal(keep,[bool(int(r['primary_candidate'])) for r in frames])
        bad, good=spans(keep,fps),spans(~keep,fps)
        s.need(bad==summary['candidate_segments'] and good==summary['noncandidate_segments'],'Segment mismatch')
        s.need(summary['candidate_frames']==int(keep.sum()) and summary['candidate_ratio']==float(keep.mean()),'Candidate total mismatch')
        s.need(summary['longest_candidate_seconds']==max((x['duration_seconds'] for x in bad),default=0),'Longest segment mismatch')
        for w in cfg['windows']:
            expected=dict(exists=any(x['frames']>=w for x in good),count=sum(max(0,x['frames']-w+1) for x in good))
            s.need(summary['noncandidate_windows'][str(w)]==expected,'Window mismatch')
        for i, frame in enumerate(frames):
            s.need(frame['config_hash']==ctx['config_hash'] and frame['scanner_version']==s.VERSION,'Frame version mismatch')
            s.need(frame['split']=='train' and frame['decode_status']=='ok' and frame['label']==summary['label']
                and frame['source_group']==row['source_group'] and float(frame['fps'])==fps,'Frame metadata mismatch')
            s.need(float(frame['estimated_timestamp_seconds'])==i/fps,'Timestamp estimate mismatch')
            s.need(int(frame['expected_frame_count'])==n,'Frame count metadata mismatch')
            reason='laplacian_local_z_le_-2_duration_ge_0.16' if keep[i] else 'below_z_but_short_run' if z[i,0]<=-2 else 'not_candidate'
            s.need(frame['candidate_reason']==reason,'Candidate reason mismatch')
        if summary['timestamp_verified']:
            pts=np.asarray([float(r['timestamp_seconds']) for r in frames])
            s.need(np.isfinite(pts).all() and (np.diff(pts)>0).all() and max(abs(pts-np.arange(n)/fps))<=.5/fps,'Timestamp reliability mismatch')
        for j,metric in enumerate(s.p.METRICS):
            for name,q in [('p00',0),('p10',.1),('p50',.5),('p90',.9),('p100',1)]:
                s.need(np.isclose(summary['quantiles'][metric][name],np.quantile(scores[:,j],q)),'Quantile mismatch')
            d=summary['persistent_absolute_low_diagnostics'][metric]
            mask=scores[:,j]<=d['reference_v2_calibration_p10']
            s.need(d['diagnostic_only'] and d['entire_video_below_reference']==bool(mask.all())
                and d['below_reference_ratio']==float(mask.mean()),'Absolute diagnostic mismatch')
        totals.update(ok=1,processed_frames=n,candidate_frames=int(keep.sum()),candidate_segments=len(bad),affected_videos=int(keep.any()))
        for dim in groups:
            g=groups[dim].setdefault(str(summary[dim]),dict(videos=0,frames=0,candidate_frames=0))
            g['videos']+=1;g['frames']+=n;g['candidate_frames']+=int(keep.sum())
        for name in ('frames.csv','summary.json','completion.json'):
            artifact_hashes[str((attempt/name).relative_to(out))]=s.digest(attempt/name)
        if compare:
            other=s.latest_attempt(compare,row['video_id'])
            s.need((attempt/'frames.csv').read_bytes()==(other/'frames.csv').read_bytes(),'Repeated frame output differs')
            a=dict(summary);b=s.read_json(other/'summary.json')
            a.pop('processing_seconds');b.pop('processing_seconds')
            s.need(a==b,'Repeated scientific summary differs')
    calculated=s.aggregate(out,ctx)
    for key, value in totals.items(): s.need(calculated[key]==value,'Aggregate total mismatch')
    s.need(calculated['groups']==groups,'Aggregate group mismatch')
    saved=sorted((out/'aggregates').glob('summary-*.json'))
    s.need(bool(saved) and s.read_json(saved[-1])==calculated,'Latest aggregate not reproducible')
    ledger=[json.loads(line) for line in (out/'access.jsonl').read_text().splitlines()]
    opens=[e for e in ledger if e['phase']=='opened']
    attempts=[e for e in ledger if e['phase']=='authorized_attempt']
    s.need(all(e['video_id'] in ids and e['split']=='train' and e['read_only']
        and e['path']==str(s.p.gate.DATA/guard.allowed[e['video_id']]['video_path']) for e in ledger),'Access scope violation')
    s.need(Counter((e['video_id'],e['operation']) for e in opens)==Counter((e['video_id'],e['operation']) for e in attempts),'Unresolved open attempt')
    access=dict(guarded_opens=len(opens),train_probes=sum(e['operation']=='probe' for e in opens),
        train_decodes=sum(e['operation']=='decode' for e in opens),unique_video_ids=len({e['video_id'] for e in opens}),
        validation_opens_probes_decodes=0,test_opens_probes_decodes=0,evidence='durable_guard_ledger')
    trace_result=None
    if trace:
        trace_result=trace_audit(trace,ids)
        s.need(trace_result['video_open_syscalls']==2*len(opens),'Trace/ledger open count mismatch')
        s.need(trace_result['probes']==access['train_probes'],'Trace/ledger probe mismatch')
        s.need(Counter(trace_result['per_video_open_syscalls'])==Counter({vid:2*sum(e['video_id']==vid for e in opens) for vid in {e['video_id'] for e in opens}}),'Trace/ledger ID mismatch')
    protected=None
    if initial:
        original=s.read_json(initial)
        changed=[name for name,sha in original.items() if s.digest(s.ROOT/name)!=sha]
        expected={'development/data_quality/README.md','docs/progress_log.md'}
        s.need(set(changed)<=expected, f'Protected tracked files changed: {changed}')
        protected=dict(initial_tracked_files=len(original),unchanged=len(original)-len(changed),expected_modified=changed)
    return dict(status='PASS',videos=len(selected),frames=totals['processed_frames'],
        candidate_frames=totals['candidate_frames'],candidate_segments=totals['candidate_segments'],
        local_z_independently_recalculated=True,segments_independently_recalculated=True,
        aggregate_rebuild_equal=True,repeated_scientific_outputs_equal=bool(compare),access=access,
        syscall_trace=trace_result,protected_files=protected,artifact_hashes=artifact_hashes,
        checked_at=s.utc(),config_hash=ctx['config_hash'],scanner=ctx['scanner'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=s.ARTIFACTS/'blur_train_v3_smoke')
    ap.add_argument('--trace',type=Path)
    ap.add_argument('--compare-output',type=Path)
    ap.add_argument('--initial-hashes',type=Path)
    ap.add_argument('--write-report',type=Path,help='Exclusive creation; omit for read-only verification')
    args=ap.parse_args()
    result=verify(args.output_dir,args.trace,args.compare_output,args.initial_hashes)
    if args.write_report:s.atomic_json(args.write_report,result)
    print(s.p.dumps({k:v for k,v in result.items() if k!='artifact_hashes'}))


if __name__=='__main__':
    main()
