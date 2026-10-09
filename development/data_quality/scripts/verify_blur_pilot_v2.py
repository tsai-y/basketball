#!/usr/bin/env python3
"""Independent derived-artifact checks and syscall audit; no dataset video reads."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import audit_blur_pilot_v2 as v
p=v.p


def spans(mask):
    start=0;result=[]
    for flag,group in itertools.groupby(mask):
        n=len(list(group))
        if flag:result.append((start,start+n))
        start+=n
    return result


def reference_mask(z,method,threshold,minimum,fps):
    a=z[:,0]<=threshold;b=z[:,1]<=threshold
    mask={'laplacian':a,'tenengrad':b,'both':np.logical_and(a,b),'either':np.logical_or(a,b)}[method].copy()
    for lo,hi in spans(mask):
        if hi-lo<minimum*fps:mask[lo:hi]=False
    return mask


def verify(out,trace,check_only=False):
    report=json.loads((out/'analysis/report.json').read_text());cfg=report['config']
    initial=json.loads((out/'runtime_logs/initial_tracked_hashes.json').read_text())
    changed=[path for path,h in initial.items() if not (p.ROOT/path).is_file() or p.gate.digest(p.ROOT/path)!=h]
    assert not changed,changed  # Includes all v1 and frozen tracked evidence.
    assert p.gate.digest(Path(v.__file__))==report['provenance']['script_sha256']
    assert p.gate.digest(v.DEFAULT_CONFIG)==report['provenance']['config_sha256']
    assert p.gate.digest(Path(p.__file__))==report['provenance']['v1_script_sha256']
    assert p.gate.digest(Path(__file__).with_name('analyze_blur_pilot_v2.py'))==report['analysis_script_sha256']
    guard=p.TrainGuard();selected,inventory,_=v.choose(guard,cfg)
    m=pd.read_csv(out/'pilot_manifest_v2.csv');by_id=m.set_index('video_id')
    assert list(m.video_id)==[r['video_id'] for r in selected]
    assert len(m)==48 and m.source_group.nunique()==12
    assert not set(m[m.subset=='calibration'].source_group)&set(m[m.subset=='verification'].source_group)
    for row in m.to_dict('records'):guard.authorize(row['video_id'],row['relative_video_path'])
    assert not guard.events
    statuses=json.loads((out/'runtime_logs/decode_status.json').read_text())
    assert len(statuses)==48 and all(s['decode_status']=='ok' and s['timestamp_verified'] for s in statuses)
    ledger=json.loads((out/'runtime_logs/access_ledger.json').read_text())
    assert Counter((e['video_id'],e['operation']) for e in ledger)==Counter((vid,op) for vid in m.video_id for op in ('probe','decode'))
    assert all(e['split']=='train' and e['read_only'] for e in ledger)
    prefix=str(p.gate.DATA)+'/'
    opened=[];lines=trace.read_text().splitlines()
    for line in lines:
        if prefix not in line:continue
        assert 'openat(' in line and 'O_RDONLY' in line,line
        assert not any(flag in line for flag in ['O_WRONLY','O_RDWR','O_TRUNC','O_CREAT'])
        ids=set(re.findall(re.escape(prefix)+r'([^"<>]+\.mp4)',line))
        assert len(ids)==1,line
        vid=next(iter(ids));assert vid in by_id.index
        opened.append(vid)
    assert Counter(opened)==Counter({vid:4 for vid in m.video_id})
    executions=[line for line in lines if 'execve(' in line and '= 0' in line]
    assert sum('/ffprobe"' in l and '-show_entries' in l for l in executions)==48
    assert not any('/ffmpeg"' in l for l in executions)
    df=pd.read_csv(out/'synthetic_scores.csv');anchors=pd.read_csv(out/'calibration_frames.csv')
    assert len(df)==5280 and len(anchors)==240
    for vid,rows in anchors.groupby('video_id'):
        info=by_id.loc[vid];n=int(info.frame_count)
        expected=[math.floor(x*(n-1)+.5) for x in cfg['frame_positions']]
        assert rows.frame_index.tolist()==expected
        np.testing.assert_allclose(rows.estimated_timestamp_seconds,rows.frame_index/info.fps)
        np.testing.assert_allclose(rows.timestamp_seconds,rows.frame_index/info.fps,atol=.5/info.fps)
    for metric in v.METRICS:
        assert (df[metric]>0).all()
        np.testing.assert_allclose(df[metric+'_relative'],df[metric]/df[metric+'_original'],rtol=1e-12)
        np.testing.assert_allclose(df[metric+'_log_drop'],np.log(df[metric])-np.log(df[metric+'_original']),rtol=1e-12,atol=1e-12)
        base=df[df.severity==0]
        np.testing.assert_array_equal(base[metric],base[metric+'_original'])
        cal=base[base.subset=='calibration']
        assert np.isclose(np.quantile(cal[metric],.1),report['static']['absolute_thresholds'][metric])
    curves=pd.read_csv(out/'analysis/sensitivity_curves.csv')
    groups=df.groupby(['video_id','frame_index','transformation_family','angle'],dropna=False)
    for row in curves.to_dict('records'):
        g=df[(df.video_id==row['video_id'])&(df.frame_index==row['frame_index'])&
             (df.transformation_family==row['transformation_family'])]
        if not pd.isna(row['angle']):g=g[g.angle==row['angle']]
        g=g.sort_values('severity');metric=row['metric']
        vals=np.sqrt(g[v.METRICS[0]+'_relative']*g[v.METRICS[1]+'_relative']) if metric=='geometric_relative' else g[metric]
        correlation=np.corrcoef(rankdata(g.severity),rankdata(vals))[0,1]
        assert np.isclose(correlation,row['spearman'])
        tol=cfg['monotonic_relative_tolerance']*max(1,abs(vals.iloc[0]))
        assert int((np.diff(vals)>tol).sum())==row['increasing_steps']
    scenarios=pd.read_csv(out/'temporal_scenarios.csv');det=pd.read_csv(out/'analysis/temporal_detection.csv')
    segment_csv=pd.read_csv(out/'analysis/temporal_detected_segments.csv')
    thresholds=json.loads((out/'analysis/frozen_thresholds.json').read_text())
    assert thresholds==report['temporal_thresholds']
    expected_segments=[];z_checked=0;calibration_grid=[]
    for row in m.to_dict('records'):
        vid=row['video_id'];fps=row['fps']
        sc=scenarios[scenarios.video_id==vid].sort_values('scenario_id')
        detections=det[det.video_id==vid].set_index(['scenario_id','method'])
        with np.load(out/'streams'/f'{Path(vid).stem}.npz') as f:
            scores=f['scores'];z=f['local_z'];assert scores.shape==z.shape==(67,int(row['frame_count']),2)
            original=scores[0];width=math.ceil(cfg['temporal']['rolling_seconds']*fps);width+=int(width%2==0)
            # Independent centered-window implementation; same-window MAD, no oracle baseline.
            log=np.log(np.maximum(scores,cfg['score_epsilon']))
            for i in range(scores.shape[1]):
                window=log[:,max(0,i-width//2):i+width//2+1,:]
                med=np.median(window,axis=1);mad=np.median(abs(window-med[:,None,:]),axis=1)
                zz=(log[:,i,:]-med)/np.maximum(1.4826*mad,cfg['temporal']['mad_epsilon'])
                np.testing.assert_allclose(z[:,i,:],zz,rtol=1e-12,atol=1e-12)
            z_checked+=len(z)
            for s in sc.to_dict('records'):
                sid=s['scenario_id'];lo=int(s['start_frame']);hi=int(s['end_frame_exclusive'])
                np.testing.assert_array_equal(scores[sid,:lo],original[:lo])
                np.testing.assert_array_equal(scores[sid,hi:],original[hi:])
                if sid:
                    assert hi-lo==math.ceil(s['duration_seconds']*fps)
                    assert np.isclose(s['actual_duration_seconds'],(hi-lo)/fps)
                    assert (lo,hi)==v.injection_span(len(original),fps,s['duration_seconds'],s['center_position'])
                for method,choice in thresholds.items():
                    mask=reference_mask(z[sid],method,choice['z_threshold'],choice['minimum_seconds'],fps)
                    r=detections.loc[(sid,method)]
                    assert r.z_threshold==choice['z_threshold'] and r.minimum_seconds==choice['minimum_seconds']
                    tp=int(mask[lo:hi].sum());fp=int(mask.sum())-tp;fn=hi-lo-tp
                    assert (tp,fp,fn)==(r.tp,r.fp,r.fn)
                    assert np.isclose(r.f1,2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
                    spans_here=spans(mask)
                    ious=[max(0,min(hi,b)-max(lo,a))/(hi-lo+b-a-max(0,min(hi,b)-max(lo,a))) for a,b in spans_here]
                    best=max(ious,default=0);assert np.isclose(r.best_segment_iou,best)
                    assert bool(r.event_detected)==(best>=cfg['temporal']['event_iou_threshold'])
                    expected_segments.extend((vid,sid,method,a,b) for a,b in spans_here)
            # Independently recompute every candidate using calibration only.
            if row['subset']=='calibration':
                for method in v.METHODS:
                    for threshold in cfg['temporal']['z_thresholds']:
                        for minimum in cfg['temporal']['minimum_seconds_grid']:
                            f1s=[];control=None
                            for s in sc.to_dict('records'):
                                sid=s['scenario_id'];lo=int(s['start_frame']);hi=int(s['end_frame_exclusive'])
                                mask=reference_mask(z[sid],method,threshold,minimum,fps)
                                if sid==0:control=float(mask.mean());continue
                                tp=int(mask[lo:hi].sum());fp=int(mask.sum())-tp;fn=hi-lo-tp
                                f1s.append(2*tp/(2*tp+fp+fn))
                            calibration_grid.append(dict(source=row['source_group'],method=method,z_threshold=threshold,
                                minimum_seconds=minimum,f1=float(np.mean(f1s)),control=control))
    actual_segments=list(segment_csv[['video_id','scenario_id','method','start_frame','end_frame_exclusive']].itertuples(index=False,name=None))
    assert sorted(expected_segments)==sorted(actual_segments)
    grid=pd.DataFrame(calibration_grid).groupby(['source','method','z_threshold','minimum_seconds'])[['f1','control']].mean().groupby(['method','z_threshold','minimum_seconds']).mean().reset_index()
    saved_grid=pd.read_csv(out/'analysis/threshold_search_calibration_only.csv')
    for r in saved_grid.to_dict('records'):
        g=grid[(grid.method==r['method'])&(grid.z_threshold==r['z_threshold'])&(grid.minimum_seconds==r['minimum_seconds'])].iloc[0]
        assert np.isclose(g.f1,r['calibration_source_macro_f1']) and np.isclose(g.control,r['calibration_control_flag_rate'])
    for method in v.METHODS:
        feasible=[r for r in saved_grid.to_dict('records') if r['method']==method and r['calibration_control_flag_rate']<=cfg['temporal']['max_calibration_control_flag_rate']]
        best=max(feasible,key=lambda r:(r['calibration_source_macro_f1'],-r['calibration_control_flag_rate'],-r['z_threshold'],r['minimum_seconds']))
        assert best['z_threshold']==thresholds[method]['z_threshold'] and best['minimum_seconds']==thresholds[method]['minimum_seconds']
    for s in report['temporal_summary']:
        g=det[(det.subset==s['subset'])&(det.method==s['method'])&(det.transformation_family!='control')]
        if s['dimension']!='all':g=g[g[s['dimension']].astype(str)==s['group']]
        f1=g.groupby(['source_group','video_id']).f1.mean().groupby('source_group').mean().mean()
        assert np.isclose(f1,s['source_macro_f1'])
    subprocess.run(['git','diff','--check'],cwd=p.ROOT,check=True,capture_output=True)
    artifact_hashes={str(f.relative_to(out)):p.gate.digest(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='verification.json'}
    result=dict(status='PASS',unchanged_preexisting_tracked_files=len(initial),selected_videos=48,
        decoded_frames=int(m.frame_count.sum()),anchor_frames=240,synthetic_score_rows=5280,
        independently_verified_normalized_streams=z_checked,independently_recomputed_threshold_candidates=len(saved_grid),
        independent_detection_rows=len(det),independent_segments=len(actual_segments),
        train_guard_opens=96,video_open_syscalls=len(opened),unique_video_ids_opened=len(set(opened)),
        train_probes=48,train_decodes=48,validation_opens_probes_decodes=0,test_opens_probes_decodes=0,
        all_video_opens_read_only=True,encoded_videos=0,source_disjoint=True,
        thresholds_calibration_only=True,verification_thresholds_unchanged=True,
        tests='10 v2 + 6 v1 PASS (logs); no videos opened in guard tests',
        artifact_hashes=artifact_hashes)
    if not check_only:
        p.write_json(out/'runtime_logs/verification.json',result)
    print(p.dumps({k:val for k,val in result.items() if k!='artifact_hashes'}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=p.ROOT/'development/data_quality/artifacts/blur_pilot_v2')
    ap.add_argument('--trace',type=Path)
    ap.add_argument('--check-only',action='store_true',help='Recompute checks without creating or overwriting any artifact')
    args=ap.parse_args();verify(args.output_dir,args.trace or args.output_dir/'runtime_logs/syscalls.log',args.check_only)
