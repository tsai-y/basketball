"""Unvalidated camera-cut / court-appearance proposals, NOT a shot-event detector."""
from dataclasses import dataclass,asdict
import math

import cv2
import numpy as np


def need(condition,message):
    if not condition:raise ValueError(message)


@dataclass(frozen=True)
class Policy:
    version:str='court_scene_proposal_v1_not_event_localization'
    min_clip_frames:int=64
    min_other_court_run:int=8
    histogram_cut:float=.48
    gray_cut:float=.14
    strong_histogram_cut:float=.70
    flash_recovery_histogram:float=.12
    flash_recovery_gray:float=.06
    floor_area_min:float=.20
    floor_column_min:float=.55
    smoothing_radius:int=2


def validate_policy(p):
    need(type(p.min_clip_frames) is int and p.min_clip_frames>=2,'Invalid clip length')
    need(type(p.min_other_court_run) is int and 1<=p.min_other_court_run<=p.min_clip_frames,'Invalid ambiguity length')
    need(type(p.smoothing_radius) is int and 0<=p.smoothing_radius<=5,'Invalid smoothing radius')
    for k,v in asdict(p).items():
        if type(v) is float:need(math.isfinite(v) and 0<v<1,'Invalid threshold: '+k)


def descriptor(bgr):
    need(isinstance(bgr,np.ndarray) and bgr.dtype==np.uint8 and bgr.shape==(240,320,3),'Native320x240 BGR required')
    # Ignore fixed top/bottom overlays in descriptors, without cropping actual clips.
    small=cv2.resize(bgr[24:204],(96,54),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV);gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    hist=cv2.calcHist([hsv],[0,1],None,[24,8],[0,180,0,256]);hist/=max(float(hist.sum()),1.)
    # Broad wood-like color is only an appearance cue: skin/graphics may also match.
    mask=cv2.inRange(hsv,np.array([8,20,75],np.uint8),np.array([38,220,255],np.uint8))>0
    area=float(mask.mean());columns=float((mask.mean(axis=0)>=.20).mean())
    return {'hist':hist,'gray':gray,'floor_area':area,'floor_columns':columns}


def distances(a,b):
    return float(cv2.compareHist(a['hist'],b['hist'],cv2.HISTCMP_BHATTACHARYYA)),float(np.abs(a['gray']-b['gray']).mean())


def cut_candidates(features,policy=Policy()):
    validate_policy(policy);need(len(features)>=2,'Need at least2 frames')
    metrics=[{'frame_index':0,'histogram_distance':0.,'gray_difference':0.}]
    flags=[False]*len(features);flashes=[]
    for i in range(1,len(features)):
        h,g=distances(features[i-1],features[i])
        need(math.isfinite(h) and math.isfinite(g),'Non-finite descriptor')
        flags[i]=(h>=policy.histogram_cut and g>=policy.gray_cut) or h>=policy.strong_histogram_cut
        metrics.append({'frame_index':i,'histogram_distance':h,'gray_difference':g})
    # A one-frame exposure spike with near-identical recovery is not two camera cuts.
    for i in range(1,len(features)-1):
        if not(flags[i] and flags[i+1]):continue
        h,g=distances(features[i-1],features[i+1])
        if h<=policy.flash_recovery_histogram and g<=policy.flash_recovery_gray:
            flags[i]=flags[i+1]=False;flashes.append(i)
    return [i for i,v in enumerate(flags) if v],flashes,metrics


def runs(flags):
    result=[];start=None
    for i,value in enumerate([*flags,False]):
        if value and start is None:start=i
        elif not value and start is not None:result.append((start,i));start=None
    return result


def propose_intervals(court_flags,cuts,policy=Policy()):
    validate_policy(policy);n=len(court_flags)
    need(n>=2 and all(type(v) is bool for v in court_flags),'Boolean per-frame appearance required')
    need(cuts==sorted(set(cuts)) and all(type(i) is int and 0<i<n for i in cuts),'Invalid cut boundaries')
    boundaries=[0,*cuts,n];candidates=[];shots=[]
    for left,right in zip(boundaries,boundaries[1:]):
        local=court_flags[left:right];smoothed=[];r=policy.smoothing_radius
        # Never smooth or bridge across a camera cut.
        for i in range(len(local)):
            values=local[max(0,i-r):min(len(local),i+r+1)]
            smoothed.append(sum(values)*2>=len(values)+1)
        shots.append({'start_frame':left,'end_frame_exclusive':right,
                      'court_appearance_fraction':sum(local)/len(local)})
        for a,b in runs(smoothed):
            if b-a>=policy.min_other_court_run:
                candidates.append({'start_frame':left+a,'end_frame_exclusive':left+b,'length':b-a,
                                   'camera_segment':[left,right],
                                   'eligible_length':b-a>=policy.min_clip_frames})
    eligible=[c for c in candidates if c['eligible_length']]
    reasons=[];proposal=None
    if not eligible:reasons.append('no_single_contiguous_court_candidate_at_least64_frames')
    elif len(candidates)!=1:reasons.append('multiple_court_intervals_cannot_identify_shot_and_result')
    else:proposal=eligible[0]
    return {'camera_segments':shots,'court_interval_candidates':candidates,'proposed_contiguous_interval':proposal,
            'candidate_fallback_reasons':reasons,'interval_convention':'zero-based[start,end)',
            'event_semantics_verified':False,'automatic_training_use_enabled':False,
            'effective_training_policy':'original_three_window_coverage_unchanged_pending_event_review',
            'frames_concatenated_across_cuts':False,'frames_deleted':False}


def analyze(frames,policy=Policy()):
    validate_policy(policy);need(len(frames)>=2,'Need frames')
    features=[descriptor(f) for f in frames];cuts,flashes,metrics=cut_candidates(features,policy)
    flags=[bool(f['floor_area']>=policy.floor_area_min and f['floor_columns']>=policy.floor_column_min) for f in features]
    result=propose_intervals(flags,cuts,policy)
    result.update(policy=asdict(policy),cut_frame_candidates=cuts,flash_frame_candidates=flashes,
                  per_frame=[{**m,'floor_area':f['floor_area'],'floor_columns':f['floor_columns'],'court_like':flag}
                             for m,f,flag in zip(metrics,features,flags)])
    return result
