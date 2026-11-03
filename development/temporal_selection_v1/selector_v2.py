"""Exploratory scene segmentation v2; pixels only, no event/class labels or training use."""
from dataclasses import dataclass,asdict
import math

import numpy as np

from .selector import descriptor,distances,runs,need


@dataclass(frozen=True)
class ScenePolicy:
    version:str='scene_segmentation_v2_exploratory_not_target_event_selector'
    min_preview_frames:int=64
    min_segment_frames:int=8
    hard_hist:float=.48
    hard_gray:float=.14
    strong_hist:float=.70
    luma_window:int=5
    luma_peak:float=.10
    flash_max_frames:int=4
    recovery_hist:float=.30
    recovery_correlation:float=.75
    gradual_radius:int=5
    gradual_window_hist:float=.38
    gradual_adjacent_hist:float=.055
    gradual_active_edges:int=3
    gradual_bridge_frames:int=3
    gradual_pad_frames:int=2


def validate(policy):
    need(2<=policy.min_preview_frames and 1<=policy.min_segment_frames<=policy.min_preview_frames,'Invalid segment lengths')
    for k,v in asdict(policy).items():
        if type(v) is float:need(math.isfinite(v) and 0<v<1,'Invalid threshold: '+k)
        elif type(v) is int:need(v>=0,'Invalid integer setting: '+k)
    need(policy.luma_window>=1 and policy.gradual_radius>=2 and policy.flash_max_frames>=1,'Invalid temporal radius')


def merge_regions(regions,n,gap=0,pad=0):
    need(type(n) is int and n>0 and type(gap) is int and gap>=0 and type(pad) is int and pad>=0,'Invalid bounds')
    checked=[]
    for a,b in regions:
        need(type(a) is int and type(b) is int and 0<=a<b<=n,'Invalid half-open region')
        checked.append((max(0,a-pad),min(n,b+pad)))
    result=[]
    for a,b in sorted(checked):
        if result and a<=result[-1][1]+gap:result[-1]=(result[-1][0],max(b,result[-1][1]))
        else:result.append((a,b))
    return result


def correlation(a,b):
    a=a.ravel().astype(np.float64);b=b.ravel().astype(np.float64)
    ac=a-a.mean();bc=b-b.mean();scale=float(np.linalg.norm(ac)*np.linalg.norm(bc))
    if scale<1e-8:return float(np.abs(a-b).mean()<.02)
    return float(np.dot(ac,bc)/scale)


def exposure_events(features,policy=ScenePolicy()):
    validate(policy);n=len(features);means=[float(f['gray'].mean()) for f in features]
    peaks=[means[i]-float(np.median(means[max(0,i-policy.luma_window):min(n,i+policy.luma_window+1)])) for i in range(n)]
    regions=merge_regions(runs([p>=policy.luma_peak for p in peaks]),n,gap=1)
    accepted=[]
    for a,b in regions:
        if a==0 or b>=n or b-a>policy.flash_max_frames:continue
        h,_=distances(features[a-1],features[b]);c=correlation(features[a-1]['gray'],features[b]['gray'])
        if h<=policy.recovery_hist and c>=policy.recovery_correlation:
            accepted.append({'start_frame':a,'end_frame_exclusive':b,'recovery_histogram':h,'recovery_correlation':c,
                             'pixels_removed':False})
    return accepted,peaks


def clean_segments(n,cuts,regions,policy=ScenePolicy()):
    validate(policy)
    need(cuts==sorted(set(cuts)) and all(type(c) is int and 0<c<n for c in cuts),'Invalid cuts')
    regions=merge_regions(regions,n)
    boundaries=sorted({0,n,*cuts,*[x for pair in regions for x in pair]})
    segments=[]
    for a,b in zip(boundaries,boundaries[1:]):
        if any(a<y and b>x for x,y in regions):continue
        if b-a<policy.min_segment_frames:continue
        segments.append({'start_frame':a,'end_frame_exclusive':b,'length':b-a,
                         'length_eligible_for_preview':b-a>=policy.min_preview_frames,
                         'event_identity_verified':False})
    long=[s for s in segments if s['length_eligible_for_preview']]
    return {'contiguous_scene_candidates':segments,'unique_length_eligible_candidate':long[0] if len(long)==1 else None,
            'target_event_selected':False,'automatic_training_use_enabled':False,
            'effective_training_policy':'original_three_window_coverage_unchanged',
            'short_other_scenes_present':any(not s['length_eligible_for_preview'] for s in segments),
            'frames_concatenated_across_boundaries':False,
            'limitation':'Clean means outside heuristic transition flags, not proof of no graphics or correct event. Length alone does not select target.'}


def distinguish_hard_regions(regions,hard,policy):
    """A localized single-cut response has window width; do not call that a gradual event."""
    result=[]
    support=2*policy.gradual_radius+2*policy.gradual_pad_frames+1
    for a,b in regions:
        inside=[c for c in hard if a<=c<b]
        if len(inside)==1 and b-a<=support:continue
        result.append((a,b))
    return result


def analyze_v2(frames,policy=ScenePolicy()):
    validate(policy);need(len(frames)>=2,'Need continuous native frames')
    features=[descriptor(frame) for frame in frames];n=len(features)
    flashes,luma_peaks=exposure_events(features,policy)
    suppressed={i for f in flashes for i in range(f['start_frame'],min(n,f['end_frame_exclusive']+1))}
    adjacent=[(0.,0.)]+[distances(features[i-1],features[i]) for i in range(1,n)]
    hard=[i for i,(h,g) in enumerate(adjacent) if i and i not in suppressed and
          ((h>=policy.hard_hist and g>=policy.hard_gray) or h>=policy.strong_hist)]
    gradual_flags=[];window_scores=[];active_counts=[]
    for i in range(n):
        a=max(0,i-policy.gradual_radius);b=min(n-1,i+policy.gradual_radius)
        h,_=distances(features[a],features[b]);window_scores.append(h)
        active=sum(adjacent[j][0]>=policy.gradual_adjacent_hist for j in range(a+1,b+1)
                   if j not in suppressed and j not in hard)
        active_counts.append(active)
        # A recovered flash contaminates the whole temporal window, not only its center.
        # Exclude such windows from the gradual detector rather than creating artificial wings.
        exposure_in_window=any(j in suppressed for j in range(a,b+1))
        gradual_flags.append(bool(h>=policy.gradual_window_hist and active>=policy.gradual_active_edges and not exposure_in_window))
    regions=merge_regions(runs(gradual_flags),n,gap=policy.gradual_bridge_frames,pad=policy.gradual_pad_frames)
    regions=distinguish_hard_regions(regions,hard,policy)
    # Hard cuts inside a gradual-activity region are explained by that region, not extra isolated cuts.
    cuts=[c for c in hard if not any(a<=c<b for a,b in regions)]
    result=clean_segments(n,cuts,regions,policy)
    result.update(policy=asdict(policy),hard_cut_candidates=cuts,raw_hard_cut_candidates=hard,
        exposure_flash_candidates=flashes,gradual_change_regions=[{'start_frame':a,'end_frame_exclusive':b,
             'interpretation':'possible_transition_or_fast_appearance_change_NOT_confirmed_graphic'} for a,b in regions],
        floor_appearance_used_as_exclusion_rule=False,input_metadata_used_for_selection=False,
        per_frame=[{'frame_index':i,'histogram_distance':adjacent[i][0],'gray_difference':adjacent[i][1],
                    'luma':float(features[i]['gray'].mean()),'luma_peak':luma_peaks[i],
                    'window_histogram_distance':window_scores[i],'active_edges':active_counts[i],
                    'flash_boundary_suppressed':i in suppressed,'floor_area_diagnostic_only':features[i]['floor_area']}
                   for i in range(n)])
    return result
