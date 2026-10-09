#!/usr/bin/env python3
"""Analyze derived Pilot v2 scores only. Threshold fitting accepts calibration sources only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import audit_blur_pilot_v2 as v
p=v.p


def rho(a,b):
    return float(spearmanr(a,b).statistic) if len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0 else None


def describe(a):
    a=np.asarray(a,dtype=float);a=a[np.isfinite(a)]
    return dict(n=len(a),p10=float(np.quantile(a,.1)),median=float(np.median(a)),p90=float(np.quantile(a,.9))) if len(a) else dict(n=0,p10=None,median=None,p90=None)


def eta_squared(values, groups):
    values=np.asarray(values);groups=np.asarray(groups)
    total=np.sum((values-values.mean())**2)
    return float(sum(np.sum(groups==g)*(values[groups==g].mean()-values.mean())**2 for g in np.unique(groups))/total) if total>0 else None


def static_analysis(df,manifest,cfg,out):
    curves=[];distributions=[];complement=[];absolute=[];confounds=[]
    base=df[(df.transformation_family=='gaussian')&(df.severity==0)].copy()
    control=base[base.subset=='calibration']
    thresholds={m:float(control[m].quantile(cfg['absolute_control_quantile'])) for m in v.METRICS}
    for subset in ('calibration','verification'):
        ss=df[df.subset==subset]
        for (vid,idx,family,angle),rows in ss.groupby(['video_id','frame_index','transformation_family','angle'],dropna=False):
            rows=rows.sort_values('severity')
            for metric in (*v.METRICS,'geometric_relative'):
                values=(np.sqrt(rows[v.METRICS[0]+'_relative']*rows[v.METRICS[1]+'_relative']).to_numpy()
                        if metric=='geometric_relative' else rows[metric].to_numpy())
                tol=cfg['monotonic_relative_tolerance']*max(1.,abs(values[0]))
                deltas=np.diff(values)
                r=rows.iloc[0]
                curves.append(dict(subset=subset,video_id=vid,frame_index=int(idx),source_group=r.source_group,
                    label=r.label,fps=r.fps,transformation_family=family,angle=None if pd.isna(angle) else angle,
                    metric=metric,spearman=rho(rows.severity,values),monotonic_nonincrease=bool((deltas<=tol).all()),
                    strictly_decreasing=bool((deltas < -tol).all()),decreasing_step_fraction=float((deltas < -tol).mean()),
                    increasing_steps=int((deltas>tol).sum())))
        for (family,severity,angle),rows in ss.groupby(['transformation_family','severity','angle'],dropna=False):
            ident=dict(subset=subset,transformation_family=family,severity=severity,angle=None if pd.isna(angle) else angle)
            for dimension in ('all','source_group','fps','label'):
                grouped=[('all',rows)] if dimension=='all' else rows.groupby(dimension)
                for group,g in grouped:
                    for metric in v.METRICS:
                        for representation in ('absolute','relative','log_drop'):
                            col=metric if representation=='absolute' else metric+'_'+representation
                            distributions.append(dict(**ident,dimension=dimension,group=str(group),metric=metric,
                                representation=representation,**describe(g[col])))
            a=rows[v.METRICS[0]]<=thresholds[v.METRICS[0]]
            b=rows[v.METRICS[1]]<=thresholds[v.METRICS[1]]
            for method,mask in zip(v.METHODS,(a,b,a&b,a|b)):
                absolute.append(dict(**ident,method=method,n=len(rows),flag_fraction=float(mask.mean())))
            complement.append(dict(**ident,paired_log_drop_spearman=rho(rows[v.METRICS[0]+'_log_drop'],rows[v.METRICS[1]+'_log_drop']),
                absolute_flag_disagreement=float((a!=b).mean()),laplacian_only=int((a&~b).sum()),tenengrad_only=int((b&~a).sum())))
        # One row per video avoids treating correlated neighboring frames as independent videos.
        b=base[base.subset==subset]
        video=b.groupby(['video_id','source_group','fps','label'],as_index=False)[list(v.METRICS)].median()
        for metric in v.METRICS:
            y=np.log(video[metric].to_numpy())
            for dim in ('source_group','fps','label'):
                confounds.append(dict(subset=subset,metric=metric,representation='log_video_median_original',
                    transformation_family='original',severity=0,angle=None,dimension=dim,eta_squared=eta_squared(y,video[dim]),videos=len(video)))
        for (family,severity,angle),rows in ss[ss.severity>0].groupby(['transformation_family','severity','angle'],dropna=False):
            video=rows.groupby(['video_id','source_group','fps','label'],as_index=False)[[m+'_log_drop' for m in v.METRICS]].median()
            for metric in v.METRICS:
                for dim in ('source_group','fps','label'):
                    confounds.append(dict(subset=subset,metric=metric,representation='video_median_paired_log_drop',
                        transformation_family=family,severity=severity,angle=None if pd.isna(angle) else angle,
                        dimension=dim,eta_squared=eta_squared(video[metric+'_log_drop'].to_numpy(),video[dim]),videos=len(video)))
    curves=pd.DataFrame(curves)
    summary=[]
    for (subset,family,metric),g in curves.groupby(['subset','transformation_family','metric']):
        summary.append(dict(subset=subset,transformation_family=family,metric=metric,curves=len(g),
            monotonic_nonincrease_fraction=float(g.monotonic_nonincrease.mean()),
            strictly_decreasing_fraction=float(g.strictly_decreasing.mean()),nonmonotonic_cases=int((~g.monotonic_nonincrease).sum()),
            mean_decreasing_step_fraction=float(g.decreasing_step_fraction.mean()),spearman=describe(g.spearman)))
    for name,rows in [('sensitivity_curves',curves),('sensitivity_distributions',distributions),
                      ('metric_complementarity',complement),('absolute_threshold_response',absolute),('source_confounding',confounds)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    # Source-conditioned label contrast, descriptive only; FPS effect may be non-identifiable.
    within=[];design=[]
    for subset in ('calibration','verification'):
        b=base[base.subset==subset]
        med=b.groupby(['video_id','source_group','fps','label'],as_index=False)[list(v.METRICS)].median()
        for source,g in med.groupby('source_group'):
            row=dict(subset=subset,source_group=source,videos=len(g),fps_values=sorted(g.fps.unique().tolist()))
            for m in v.METRICS:
                row[m+'_make_minus_miss_log_median']=float(np.log(g[g.label=='Make'][m]).mean()-np.log(g[g.label=='Miss'][m]).mean())
            within.append(row)
        source_dummies=pd.get_dummies(med.source_group,drop_first=True,dtype=float).to_numpy()
        fps_dummies=pd.get_dummies(med.fps,drop_first=True,dtype=float).to_numpy()
        label=(med.label=='Make').to_numpy(dtype=float)[:,None]
        X=np.column_stack([np.ones(len(med)),source_dummies,label]);XF=np.column_stack([X,fps_dummies])
        design.append(dict(subset=subset,source_label_design_rank=int(np.linalg.matrix_rank(X)),
            with_fps_design_rank=int(np.linalg.matrix_rank(XF)),with_fps_columns=XF.shape[1],
            fps_adds_identifiable_dimension=bool(np.linalg.matrix_rank(XF)>np.linalg.matrix_rank(X))))
    return dict(sensitivity=summary,absolute_thresholds=thresholds,original_confounding=[r for r in confounds if r['severity']==0],
                within_source_label_contrasts=within,fps_source_design=design)


def detect(z,method,threshold,minimum_seconds,fps):
    a,b=z[:,0]<=threshold,z[:,1]<=threshold
    mask={'laplacian':a,'tenengrad':b,'both':a&b,'either':a|b}[method].copy()
    for lo,hi in p.runs(mask):
        if (hi-lo+1)/fps < minimum_seconds:mask[lo:hi+1]=False
    return mask


def metrics(mask,lo,hi,fps,iou_threshold):
    truth=np.zeros(len(mask),dtype=bool);truth[lo:hi]=True
    tp=int((mask&truth).sum());fp=int((mask&~truth).sum());fn=int((~mask&truth).sum())
    spans=p.runs(mask)
    best=None;best_iou=0
    for start,end in spans:
        end+=1;intersection=max(0,min(hi,end)-max(lo,start))
        union=(hi-lo)+(end-start)-intersection
        iou=intersection/union if union else 0
        if iou>best_iou:best_iou=iou;best=(start,end)
    matched=best_iou>=iou_threshold
    return dict(tp=tp,fp=fp,fn=fn,tn=int((~mask&~truth).sum()),
        precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        outside_flag_fraction=fp/max(1,len(mask)-(hi-lo)),best_segment_iou=best_iou,
        event_detected=bool(matched),predicted_segments=len(spans),
        false_positive_segments=len(spans)-int(matched),
        onset_error_seconds=(best[0]-lo)/fps if matched else None,
        offset_error_seconds=(best[1]-hi)/fps if matched else None,
        onset_abs_error_seconds=abs(best[0]-lo)/fps if matched else None,
        offset_abs_error_seconds=abs(best[1]-hi)/fps if matched else None)


def load_streams(out,manifest,scenarios,subset):
    result=[]
    for r in manifest[manifest.subset==subset].to_dict('records'):
        with np.load(out/'streams'/f'{Path(r["video_id"]).stem}.npz') as payload:
            zs=payload['local_z'].copy()
        rows=scenarios[scenarios.video_id==r['video_id']].sort_values('scenario_id').to_dict('records')
        p.need(len(rows)==len(zs),'Scenario/stream count mismatch')
        for row,z in zip(rows,zs):result.append((row,z))
    return result


def macro(frame,column):
    # Scenarios -> videos -> sources; held-out videos/frames are not independent replicates.
    return float(frame.groupby(['source_group','video_id'])[column].mean().groupby('source_group').mean().mean())


def fit_temporal(calibration,cfg):
    p.need(all(r['subset']=='calibration' for r,z in calibration),'Threshold fit refuses verification data')
    t=cfg['temporal'];grid=[];chosen={}
    for method in v.METHODS:
        for threshold in t['z_thresholds']:
            for minimum in t['minimum_seconds_grid']:
                outcomes=[];controls=[]
                for row,z in calibration:
                    mask=detect(z,method,threshold,minimum,row['fps'])
                    identity={k:row[k] for k in ('video_id','source_group')}
                    if row['transformation_family']=='control':
                        controls.append(dict(**identity,flag_rate=float(mask.mean())))
                    else:
                        lo=int(row['start_frame']);hi=int(row['end_frame_exclusive'])
                        tp=int(mask[lo:hi].sum());fp=int(mask.sum())-tp;fn=hi-lo-tp
                        outcomes.append(dict(**identity,f1=2*tp/(2*tp+fp+fn)))
                f1=macro(pd.DataFrame(outcomes),'f1');fpr=macro(pd.DataFrame(controls),'flag_rate')
                grid.append(dict(method=method,z_threshold=threshold,minimum_seconds=minimum,
                    calibration_source_macro_f1=f1,calibration_control_flag_rate=fpr,
                    feasible=fpr<=t['max_calibration_control_flag_rate']))
        candidates=[r for r in grid if r['method']==method and r['feasible']]
        p.need(bool(candidates),'No threshold meets predeclared control flag budget; do not relax using verification')
        # Highest F1, lower control flag rate, stricter z, longer duration; exact stable tie order.
        chosen[method]=max(candidates,key=lambda r:(r['calibration_source_macro_f1'],-r['calibration_control_flag_rate'],
                                                   -r['z_threshold'],r['minimum_seconds']))
    return chosen,grid


def evaluate(streams,thresholds,cfg):
    result=[];segments=[]
    for row,z in streams:
        for method,choice in thresholds.items():
            mask=detect(z,method,choice['z_threshold'],choice['minimum_seconds'],row['fps'])
            stat=metrics(mask,int(row['start_frame']),int(row['end_frame_exclusive']),row['fps'],cfg['temporal']['event_iou_threshold'])
            result.append(dict(**row,method=method,z_threshold=choice['z_threshold'],minimum_seconds=choice['minimum_seconds'],
                               flag_fraction=float(mask.mean()),**stat))
            for lo,hi in p.runs(mask):
                segments.append(dict(video_id=row['video_id'],subset=row['subset'],scenario_id=row['scenario_id'],method=method,
                    start_frame=lo,end_frame_exclusive=hi+1,start_estimated_seconds=lo/row['fps'],
                    end_estimated_seconds=(hi+1)/row['fps'],duration_seconds=(hi-lo+1)/row['fps']))
    return result,segments


def summarize_temporal(rows):
    df=pd.DataFrame(rows);results=[]
    for (subset,method),sub in df.groupby(['subset','method']):
        controls=sub[sub.transformation_family=='control'];ss=sub[sub.transformation_family!='control']
        for dim in ('all','transformation_family','duration_seconds','source_group','fps','label','severity','angle','center_position'):
            grouped=[('all',ss)] if dim=='all' else ss.groupby(dim,dropna=False)
            for group,g in grouped:
                results.append(dict(subset=subset,method=method,dimension=dim,group=str(group),scenarios=len(g),
                    source_macro_f1=macro(g,'f1'),source_macro_precision=macro(g,'precision'),source_macro_recall=macro(g,'recall'),
                    source_macro_event_recall=macro(g,'event_detected'),source_macro_iou=macro(g,'best_segment_iou'),
                    source_macro_outside_flag_rate=macro(g,'outside_flag_fraction'),
                    matched_onset_abs_error=describe(g.onset_abs_error_seconds),matched_offset_abs_error=describe(g.offset_abs_error_seconds),
                    calibration_or_verification_control_flag_rate=macro(controls,'flag_fraction')))
    return results


def analyze(out):
    cfg=json.loads((out/'runtime_logs/config.json').read_text())
    dest=out/'analysis';p.need(not dest.exists(),'Refusing existing analysis');dest.mkdir()
    manifest=pd.read_csv(out/'pilot_manifest_v2.csv');df=pd.read_csv(out/'synthetic_scores.csv')
    scenarios=pd.read_csv(out/'temporal_scenarios.csv')
    source_sets=[set(manifest[manifest.subset==s].source_group) for s in ('calibration','verification')]
    p.need(not source_sets[0]&source_sets[1],'Source leakage')
    static=static_analysis(df,manifest,cfg,dest)
    calibration=load_streams(out,manifest,scenarios,'calibration')
    thresholds,grid=fit_temporal(calibration,cfg)
    p.write_json(dest/'frozen_thresholds.json',thresholds)
    pd.DataFrame(grid).to_csv(dest/'threshold_search_calibration_only.csv',index=False)
    # Verification arrays are first loaded only after threshold persistence.
    rows,segments=evaluate(calibration,thresholds,cfg)
    verification=load_streams(out,manifest,scenarios,'verification')
    rr,sg=evaluate(verification,thresholds,cfg);rows+=rr;segments+=sg
    pd.DataFrame(rows).to_csv(dest/'temporal_detection.csv',index=False)
    pd.DataFrame(segments).to_csv(dest/'temporal_detected_segments.csv',index=False)
    summary=summarize_temporal(rows)
    p.write_json(dest/'temporal_group_summary.json',summary)
    selected=json.loads((out/'runtime_logs/selection.json').read_text())
    known=set(json.loads((p.ROOT/'docs/i3d_temporal_feasibility_audit.json').read_text())['statistics']['splits']['train']['fps_rational'])
    fps_coverage={s:sorted(manifest[manifest.subset==s].fps_rational.unique().tolist()) for s in ('calibration','verification')}
    report=dict(schema_version=2,status='COMPLETE' if all(known<=set(fs) for fs in fps_coverage.values()) else 'INCOMPLETE_FPS_COVERAGE',
        config=cfg,output_dir=str(out.relative_to(p.ROOT)),provenance=json.loads((out/'runtime_logs/provenance.json').read_text()),
        analysis_script_sha256=p.gate.digest(Path(__file__)),selection=dict(seed=cfg['seed'],videos=len(manifest),
        source_groups=len(set(manifest.source_group)),subsets={s:sorted(gs) for s,gs in zip(('calibration','verification'),source_sets)},
        fps_coverage=fps_coverage,label_counts=manifest.groupby(['subset','label']).size().to_dict() if False else
            {s:dict(Counter(manifest[manifest.subset==s].label)) for s in ('calibration','verification')},limitations=selected['limitations']),
        counts=dict(decoded_frames=int(manifest.frame_count.sum()),anchor_frames=len(df[df.severity==0]),synthetic_rows=len(df),
                    injected_scenarios=int((scenarios.transformation_family!='control').sum()),control_streams=len(manifest)),
        static=static,temporal_thresholds=thresholds,temporal_summary=[r for r in summary if r['dimension'] in ('all','transformation_family','duration_seconds')])
    p.write_json(dest/'report.json',report)
    print(p.dumps(dict(status=report['status'],counts=report['counts'],thresholds=thresholds,
                      overall=[r for r in summary if r['dimension']=='all'])))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,default=p.ROOT/'development/data_quality/artifacts/blur_pilot_v2')
    args=ap.parse_args();analyze(args.output_dir)
