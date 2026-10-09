#!/usr/bin/env python3
"""Train-only paired synthetic calibration; never emits images or encoded videos."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import cv2
import numpy as np
import yaml
import audit_blur_pilot as p

ROOT = p.ROOT
DEFAULT_CONFIG = ROOT / 'development/data_quality/configs/blur_pilot_v2.yaml'
METHODS = ('laplacian', 'tenengrad', 'both', 'either')
METRICS = p.METRICS


def load_config(path):
    cfg = yaml.safe_load(path.read_text())
    # A bounded experiment, deliberately no arbitrary expansion/full-train switch.
    p.need(cfg['version'] == 'blur_pilot_v2' and cfg['seed'] == 20261009, 'Fixed v2 version/seed')
    p.need(cfg['sources_per_band'] == 4 and cfg['videos_per_source'] == 4, 'Fixed 12 x 4 pilot')
    p.need(cfg['selection_rule'] == 'v1_band_fps_source_disjoint_sha256_v2', 'Unknown selection')
    p.need(cfg['frame_positions'] == [.1,.3,.5,.7,.9], 'Fixed calibration positions')
    p.need(cfg['gaussian_sigmas'] == [0,.5,1,1.5,2,3] and cfg['motion_lengths'] == [3,5,9,13]
           and cfg['motion_angles'] == [0,45,90,135], 'Fixed synthetic grid')
    t = cfg['temporal']
    p.need(t['durations_seconds'] == [.25,.5,1.] and t['center_positions'] == [.3,.7], 'Fixed temporal grid')
    p.need(t['rolling_seconds'] > 2 * max(t['durations_seconds']) and t['mad_epsilon'] > 0
           and cfg['score_epsilon'] > 0, 'Invalid robust normalization')
    p.need(all(s in cfg['gaussian_sigmas'][1:] for s in t['gaussian_sigmas'])
           and all(k in cfg['motion_lengths'] for k in t['motion_lengths'])
           and all(a in cfg['motion_angles'] for a in t['motion_angles']), 'Invalid temporal transforms')
    p.need(all(z < 0 for z in t['z_thresholds']) and all(0 <= s <= .25 for s in t['minimum_seconds_grid']),
           'Invalid temporal thresholds')
    return cfg


def keyed(seed, text):
    return hashlib.sha256(f'{seed}:{text}'.encode()).hexdigest()


def choose(guard, cfg):
    """Use only fixed train strings and v1 train metadata; no filesystem video access."""
    seed = cfg['seed']
    old = json.loads((ROOT / 'docs/data_quality/blur_screening_pilot_v1.json').read_text())
    old_m = p.read_csv(ROOT / 'development/data_quality/pilot_manifest_v1.csv')
    hint = {r['source_group']:r['fps_rational'] for r in old_m}
    rates = {s:v['both_low']['candidate_ratio'] for s,v in old['groups']['source_group'].items()}
    positive_median = float(np.median([v for v in rates.values() if v > 0]))
    pool = defaultdict(list)
    for r in guard.allowed.values():
        pool[r['source_group']].append(r)
    inventory = []
    for source, rows in sorted(pool.items()):
        rate = rates.get(source)
        band = 'unobserved' if rate is None else 'low' if rate == 0 else 'middle' if rate <= positive_median else 'high'
        inventory.append(dict(source_group=source, available_train_videos=len(rows),
            available_make=sum(r['binary_label']=='1' for r in rows),
            available_miss=sum(r['binary_label']=='0' for r in rows),
            v1_candidate_ratio=rate, v1_band=band, v1_fps_hint=hint.get(source)))
    picked, limitations = [], []
    for band in ('high','middle','low'):
        candidates = [r for r in inventory if r['v1_band']==band]
        candidates.sort(key=lambda r: keyed(seed, 'source:'+r['source_group']))
        take = []
        if band == 'high':
            take = [r for r in candidates if r['source_group']==cfg['required_source']]
        # Reserve two sources per observed v1 FPS in the zero-candidate stratum.
        if band == 'low':
            for fps in sorted({r['v1_fps_hint'] for r in candidates}):
                take.extend([r for r in candidates if r['v1_fps_hint']==fps][:2])
        take = (take + [r for r in candidates if r not in take])[:cfg['sources_per_band']]
        if len(take)<cfg['sources_per_band']: limitations.append(f'{band}: only {len(take)} sources')
        # Minimize per-band FPS imbalance across source-disjoint halves using deterministic exhaustive search.
        import itertools
        ncal = (len(take)+1)//2
        choices = list(itertools.combinations(range(len(take)), ncal))
        def split_key(indices):
            a = Counter(take[i]['v1_fps_hint'] for i in indices)
            b = Counter(r['v1_fps_hint'] for i,r in enumerate(take) if i not in indices)
            return (sum(abs(a[f]-b[f]) for f in a.keys()|b.keys()),
                    keyed(seed, 'subset:'+','.join(sorted(take[i]['source_group'] for i in indices))))
        cal = min(choices, key=split_key)
        for i, info in enumerate(take):
            group = info['source_group']
            available = sorted(pool[group], key=lambda r:(keyed(seed, r['video_id']),r['video_id']))
            chosen = []
            for label in ('0','1'):
                chosen += [r for r in available if r['binary_label']==label][:2]
            chosen += [r for r in available if r not in chosen][:max(0,4-len(chosen))]
            if len(chosen)<4: limitations.append(f'{group}: only {len(chosen)} videos')
            for r in chosen:
                picked.append(dict(r, subset='calibration' if i in cal else 'verification',
                    label='Make' if r['binary_label']=='1' else 'Miss', v1_band=band,
                    v1_candidate_ratio=info['v1_candidate_ratio'], selection_seed=seed,
                    selection_rule=cfg['selection_rule']))
    picked.sort(key=lambda r:(r['subset'],r['source_group'],r['video_id']))
    p.need(len(picked)<=60 and cfg['required_source'] in {r['source_group'] for r in picked}, 'Invalid sample')
    return picked, inventory, limitations


def score(gray):
    kw = dict(borderType=cv2.BORDER_REFLECT_101)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=1, **kw)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, **kw)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, **kw)
    return np.array([lap.var(ddof=0), np.mean(gx*gx+gy*gy)])


def motion_kernel(length, angle):
    # Symmetric line PSF: length equally weighted samples at unit pixel distances,
    # bilinear splatting onto an odd square grid. Image y points downward.
    kernel = np.zeros((length,length), dtype=np.float64)
    mid = (length-1)/2
    rad = math.radians(angle)
    for offset in np.arange(length)-mid:
        x = np.clip(mid+offset*math.cos(rad),0,length-1)
        y = np.clip(mid+offset*math.sin(rad),0,length-1)
        for xx, wx in ((math.floor(x),1-(x%1)), (math.floor(x)+1,x%1)):
            for yy, wy in ((math.floor(y),1-(y%1)), (math.floor(y)+1,y%1)):
                if 0<=xx<length and 0<=yy<length: kernel[yy,xx] += wx*wy
    return kernel/kernel.sum()


def transform(gray, family, severity, angle):
    if family=='gaussian':
        if severity==0: return gray  # identity, no numeric modification
        radius = math.ceil(3*severity)
        return cv2.GaussianBlur(gray,(2*radius+1,2*radius+1),sigmaX=severity,sigmaY=severity,
                                borderType=cv2.BORDER_REFLECT_101)
    return cv2.filter2D(gray,cv2.CV_64F,motion_kernel(int(severity),angle),
                        borderType=cv2.BORDER_REFLECT_101)


def variants(cfg, temporal=False):
    c = cfg['temporal'] if temporal else cfg
    return [('gaussian',s,None) for s in c['gaussian_sigmas']] + [
        ('motion',k,a) for k in c['motion_lengths'] for a in c['motion_angles']]


def frame_indices(n, positions):
    return [math.floor(pos*(n-1)+.5) for pos in positions]


def injection_span(n, fps, duration, center):
    length = math.ceil(duration*fps)
    p.need(length<=n, 'Video shorter than requested injection; do not replace it')
    start = max(0,min(n-length, math.floor(center*(n-1)+.5)-length//2))
    return start,start+length  # half-open


def local_normalize(scores, fps, cfg):
    t = cfg['temporal']
    width = math.ceil(t['rolling_seconds']*fps)
    width += int(width%2==0)
    logs = np.log(np.maximum(scores,cfg['score_epsilon']))
    med = np.empty_like(logs); mad = np.empty_like(logs)
    for i in range(len(logs)):
        window = logs[max(0,i-width//2):min(len(logs),i+width//2+1)]
        med[i] = np.median(window,axis=0)
        mad[i] = np.median(np.abs(window-med[i]),axis=0)
    z = (logs-med)/np.maximum(1.4826*mad,t['mad_epsilon'])
    return z, med, mad, width


def decode(row, rec, guard):
    fd = guard.open(row['video_id'],row['video_path'],'decode')
    cap = None; frames=[]; pts=[]
    try:
        cap = cv2.VideoCapture(f'/proc/self/fd/{fd}',cv2.CAP_FFMPEG)
        p.need(cap.isOpened(),'Decoder failed to open')
        while True:
            ok,bgr = cap.read()
            if not ok: break
            p.need(f'{bgr.shape[1]}x{bgr.shape[0]}'==rec['resolution'],'Resolution changed')
            frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float64))
            pts.append(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000)
    finally:
        if cap is not None: cap.release()
        os.close(fd)
    count_ok = len(frames)==rec['frame_count']
    reliable = bool(len(pts) and np.isfinite(pts).all() and (np.diff(pts)>0).all()
        and np.max(np.abs(np.array(pts)-np.arange(len(pts))/rec['fps']))<=.5/rec['fps']
        and rec['fps_rational']==rec['r_frame_rate'])
    status = dict(video_id=row['video_id'],decoded_frames=len(frames),metadata_frames=rec['frame_count'],
                  decode_status='ok' if count_ok else 'frame_count_mismatch', timestamp_verified=reliable)
    return frames,pts,status


def paired_values(value, original, epsilon):
    # Ratios with a zero denominator are undefined; do not manufacture sensitivity.
    ratio = value/original if original>0 else None
    drop = math.log(value)-math.log(original) if min(value,original)>0 else None
    return ratio,drop


def run(cfg, config_path, out, guard, selected, inventory, limitations):
    base = ROOT/'development/data_quality/artifacts'
    p.need(out.resolve().is_relative_to(base.resolve()) and out.resolve()!=base.resolve(), 'Output must be below ignored artifacts')
    p.need(not out.exists() and not out.is_symlink(),'Refusing existing output')
    # Reject any symlink ancestor rather than following it into historical outputs.
    p.need(not any(x.is_symlink() for x in [out,*out.parents]),'Symlink output rejected')
    out.mkdir(parents=True)
    for d in ('runtime_logs','streams'): (out/d).mkdir()
    p.write_json(out/'runtime_logs/config.json',cfg)
    p.write_csv(out/'source_inventory.csv',inventory)
    p.write_json(out/'runtime_logs/selection.json',dict(selected=selected,limitations=limitations))
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); cv2.setUseOptimized(False)
    p.write_json(out/'runtime_logs/provenance.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_hashes=guard.scope['input_hashes'], script_sha256=p.gate.digest(Path(__file__)),
        config_sha256=p.gate.digest(config_path), v1_script_sha256=p.gate.digest(Path(p.__file__)),
        command=sys.argv, python=platform.python_version(), numpy=np.__version__,opencv=cv2.__version__,
        ffprobe=subprocess.check_output(['ffprobe','-version'],text=True).splitlines()[0]))
    (out/'runtime_logs/opencv_build.txt').write_text(cv2.getBuildInformation())
    manifest=[]; static=[]; anchors=[]; statuses=[]; scenarios=[]; metadata=[]
    for ordinal,row in enumerate(selected):
        print(f"[{ordinal+1}/{len(selected)}] {row['subset']} {row['video_id']}",flush=True)
        rec=None
        try:
            rec=p.metadata(row,guard,{})
            metadata.append(rec)
            gray,pts,status=decode(row,rec,guard)
            statuses.append(status)
            p.need(status['decode_status']=='ok','Frame count mismatch; no sample replacement')
            fps=rec['fps']; n=len(gray)
            m={k:row[k] for k in ('video_id','label','binary_label','source_group','subset','split',
                                  'v1_band','v1_candidate_ratio','selection_seed','selection_rule')}
            m.update(relative_video_path=row['video_path'],fps=fps,fps_rational=rec['fps_rational'],
                     frame_count=n,duration_seconds=rec['duration_seconds'],decode_status='ok')
            manifest.append(m)
            common={k:m[k] for k in ('video_id','source_group','subset','label','fps')}
            original=np.array([score(g) for g in gray])
            idx=frame_indices(n,cfg['frame_positions'])
            for position,i in zip(cfg['frame_positions'],idx):
                stamp=dict(frame_index=i,position=position,timestamp_seconds=pts[i] if status['timestamp_verified'] else None,
                    estimated_timestamp_seconds=i/fps,timestamp_status='decoder_cfr_consistent' if status['timestamp_verified'] else 'unverified_use_frame_index')
                anchors.append(dict(**common,**stamp))
                for family,severity,angle in variants(cfg):
                    scores=original[i] if severity==0 else score(transform(gray[i],family,severity,angle))
                    r=dict(**common,**stamp,transformation_family=family,severity=severity,angle=angle)
                    for j,metric in enumerate(METRICS):
                        ratio,drop=paired_values(scores[j],original[i,j],cfg['score_epsilon'])
                        r.update({metric:float(scores[j]),metric+'_original':float(original[i,j]),
                                  metric+'_relative':ratio,metric+'_log_drop':drop})
                    static.append(r)
            # Only scores are cached/persisted; at most one decoded grayscale video in RAM.
            stream_scores=[original]; stream_z=[]
            z,med,mad,width=local_normalize(original,fps,cfg)
            stream_z.append(z)
            controls=dict(**common,scenario_id=0,transformation_family='control',severity=0,angle=None,
                duration_seconds=0,center_position=None,start_frame=0,end_frame_exclusive=0,
                actual_duration_seconds=0,rolling_window_frames=width)
            scenarios.append(controls)
            spans=[(d,c,*injection_span(n,fps,d,c)) for d in cfg['temporal']['durations_seconds']
                   for c in cfg['temporal']['center_positions']]
            union=sorted({i for _,_,lo,hi in spans for i in range(lo,hi)})
            sid=0
            for family,severity,angle in variants(cfg,True):
                cache={i:score(transform(gray[i],family,severity,angle)) for i in union}
                for duration,center,lo,hi in spans:
                    sid+=1; injected=original.copy()
                    injected[lo:hi]=[cache[i] for i in range(lo,hi)]
                    z,_,_,width=local_normalize(injected,fps,cfg)
                    stream_scores.append(injected);stream_z.append(z)
                    scenarios.append(dict(**common,scenario_id=sid,transformation_family=family,severity=severity,
                        angle=angle,duration_seconds=duration,center_position=center,start_frame=lo,
                        end_frame_exclusive=hi,actual_duration_seconds=(hi-lo)/fps,rolling_window_frames=width))
            np.savez_compressed(out/'streams'/f'{Path(row["video_id"]).stem}.npz',
                scores=np.array(stream_scores),local_z=np.array(stream_z),
                original_rolling_median=med,original_rolling_mad=mad,
                decoder_timestamps=np.array(pts),estimated_timestamps=np.arange(n)/fps)
        except (ValueError,OSError,subprocess.SubprocessError,cv2.error) as exc:
            statuses.append(dict(video_id=row['video_id'],decode_status='failed',reason=str(exc)))
            # Stop fitting on any failure. Selection file preserves every planned ID; never replace it.
            p.write_json(out/'runtime_logs/failure.json',dict(status='INCOMPLETE',video_id=row['video_id'],reason=str(exc)))
            break
    p.write_json(out/'runtime_logs/access_ledger.json',guard.events)
    p.write_json(out/'runtime_logs/decode_status.json',statuses)
    p.write_json(out/'runtime_logs/metadata.json',metadata)
    if manifest:p.write_csv(out/'pilot_manifest_v2.csv',manifest)
    if anchors:p.write_csv(out/'calibration_frames.csv',anchors)
    if static:p.write_csv(out/'synthetic_scores.csv',static)
    if scenarios:p.write_csv(out/'temporal_scenarios.csv',scenarios)
    p.need(len(manifest)==len(selected) and all(s['decode_status']=='ok' for s in statuses),
           'Incomplete pilot: no fitting or replacement permitted')
    print(p.dumps(dict(status='SCORED',videos=len(manifest),frames=len(anchors),synthetic_rows=len(static),scenarios=len(scenarios))))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--check-video-id');ap.add_argument('--check-video-path')
    args=ap.parse_args();cfg=load_config(args.config);guard=p.TrainGuard()
    if args.check_video_id:
        guard.authorize(args.check_video_id,args.check_video_path)
        print('AUTHORIZED_TRAIN_METADATA_ONLY');return
    p.need(args.check_video_path is None,'Path requires ID')
    selected,inventory,limitations=choose(guard,cfg)
    if args.dry_run:
        print(p.dumps(dict(status='DRY_RUN_PASS',selected=selected,source_inventory=inventory,
                          limitations=limitations,video_opens=0)));return
    out=args.output_dir or ROOT/cfg['output_dir']
    if not out.is_absolute():out=ROOT/out
    run(cfg,args.config,out,guard,selected,inventory,limitations)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(2)
