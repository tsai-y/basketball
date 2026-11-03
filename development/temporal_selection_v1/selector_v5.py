"""Supplement v4 with spatially local, short recovered exposure proposals only.

No global threshold relaxation, semantic event selection, pixel deletion, or
training approval. v1-v4 remain immutable. Similar-looking short cuts and
transient moving lights/objects can still resemble an exposure pulse.
"""
from dataclasses import dataclass, asdict

import numpy as np

from .selector import descriptor, distances, runs, need
from .selector_v2 import merge_regions, correlation, clean_segments, distinguish_hard_regions
from .selector_v3 import ScenePolicyV3, exposure_events_v3
from .selector_v4 import BoundaryPolicy, validate_boundary, refine_analysis


@dataclass(frozen=True)
class ExposurePolicyV5(BoundaryPolicy):
    version: str = 'scene_v5_local_exposure_diagnostic_not_target_event_selection'
    local_grid_rows: int = 3
    local_grid_columns: int = 4
    local_min_connected_tiles: int = 2


def base_policy(policy):
    return BoundaryPolicy(**{k:getattr(policy,k) for k in asdict(BoundaryPolicy()) if k!='version'})


def validate_v5(policy):
    need(isinstance(policy,ExposurePolicyV5),'Need explicit v5 policy')
    validate_boundary(base_policy(policy))
    need(type(policy.local_grid_rows) is int and 1<=policy.local_grid_rows<=54 and
         54%policy.local_grid_rows==0,'Local rows must divide the frozen54-pixel descriptor')
    need(type(policy.local_grid_columns) is int and 1<=policy.local_grid_columns<=96 and
         96%policy.local_grid_columns==0,'Local columns must divide the frozen96-pixel descriptor')
    need(type(policy.local_min_connected_tiles) is int and
         2<=policy.local_min_connected_tiles<=policy.local_grid_rows*policy.local_grid_columns,
         'Require at least two spatially connected bright tiles')


def largest_component(mask):
    """Four-neighbor connectivity; two unrelated diagonal bright tiles are not enough."""
    need(mask.ndim==2 and mask.dtype==np.bool_,'Boolean tile mask required')
    remaining={tuple(x) for x in np.argwhere(mask)};best=0
    while remaining:
        seed=remaining.pop();stack=[seed];size=0
        while stack:
            r,c=stack.pop();size+=1
            for v in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if v in remaining:remaining.remove(v);stack.append(v)
        best=max(best,size)
    return best


def local_seeds(features,policy):
    validate_v5(policy);n=len(features);rr=policy.local_grid_rows;cc=policy.local_grid_columns
    need(n>=2,'Need continuous frames')
    grays=np.stack([f['gray'] for f in features])
    need(grays.shape==(n,54,96) and np.isfinite(grays).all(),'Invalid frozen gray descriptor')
    tiles=grays.reshape(n,rr,54//rr,cc,96//cc).mean(axis=(2,4))
    peaks=np.stack([tiles[i]-np.median(tiles[max(0,i-policy.luma_window):min(n,i+policy.luma_window+1)],axis=0)
                    for i in range(n)])
    components=[largest_component(m>=policy.luma_peak) for m in peaks]
    flags=[c>=policy.local_min_connected_tiles for c in components]
    return flags,peaks,components


def confirm_local(features,a,peak_end,policy):
    """Same v3 bounded recovery/correlation/histogram/luma/hold requirements."""
    n=len(features);means=[float(f['gray'].mean()) for f in features]
    diagnostic={'start_frame':a,'peak_end_frame_exclusive':peak_end,
                'accepted':False,'reason':'no_persistent_recovery','return_checks':[]}
    if a==0 or peak_end-a>policy.flash_max_frames:
        diagnostic['reason']='source_start_or_seed_longer_than_existing_flash_limit';return None,diagnostic
    before=features[a-1];baseline=means[a-1]
    latest=min(a+policy.flash_max_frames,n-policy.recovery_hold_frames)
    for end in range(peak_end,latest+1):
        intermediate=all(means[j]>=baseline-policy.recovery_luma_tolerance and
                         correlation(before['gray'],features[j]['gray'])>=policy.recovery_correlation
                         for j in range(a,end))
        if not intermediate:
            diagnostic['reason']='intermediate_scene_or_structure_not_preserved';break
        checks=[]
        for j in range(end,end+policy.recovery_hold_frames):
            h,_=distances(before,features[j]);anchor=before if j==end else features[end]
            checks.append({'frame_index':j,'histogram_distance':h,
                           'correlation':correlation(anchor['gray'],features[j]['gray']),
                           'luma_delta':abs(means[j]-baseline)})
        diagnostic['return_checks'].append({'candidate_return_frame':end,'hold_checks':checks})
        if all(c['histogram_distance']<=policy.recovery_hist and
               c['correlation']>=policy.recovery_correlation and
               c['luma_delta']<=policy.recovery_luma_tolerance for c in checks):
            first=checks[0];diagnostic.update(accepted=True,reason='bounded_recovery_confirmed',return_frame=end)
            event={'start_frame':a,'end_frame_exclusive':end,'peak_end_frame_exclusive':peak_end,
                   'recovery_histogram':first['histogram_distance'],'recovery_correlation':first['correlation'],
                   'recovery_luma_delta':first['luma_delta'],'recovery_hold_frames':policy.recovery_hold_frames,
                   'pixels_removed':False,'detection_basis':'local_connected_tile_pulse_with_v3_recovery'}
            return event,diagnostic
    return None,diagnostic


def exposure_events_v5(features,policy=ExposurePolicyV5()):
    validate_v5(policy)
    old_policy=ScenePolicyV3(**{k:getattr(policy,k) for k in asdict(ScenePolicyV3()) if k!='version'})
    old,peaks=exposure_events_v3(features,old_policy)
    flags,tile_peaks,components=local_seeds(features,policy)
    events=[{**e,'detection_basis':'unchanged_v3_global_luma_rule'} for e in old]
    diagnostics=[]
    for a,b in merge_regions(runs(flags),len(features),gap=1):
        d={'start_frame':a,'peak_end_frame_exclusive':b,'accepted':False,
           'max_tile_luma_peak':float(tile_peaks[a:b].max()),
           'max_connected_bright_tiles':max(components[a:b])}
        if any(a<e['end_frame_exclusive'] and b>e['start_frame'] for e in events):
            d['reason']='overlaps_already_confirmed_exposure_do_not_extend';diagnostics.append(d);continue
        event,checks=confirm_local(features,a,b,policy);d.update(checks)
        if event is not None:events.append(event)
        diagnostics.append(d)
    events.sort(key=lambda e:(e['start_frame'],e['end_frame_exclusive']))
    return events,peaks,diagnostics,tile_peaks,components


def analyze_v5(frames,policy=ExposurePolicyV5()):
    validate_v5(policy);need(len(frames)>=2,'Need continuous native frames')
    features=[descriptor(f) for f in frames];n=len(features)
    flashes,peaks,diagnostics,tile_peaks,components=exposure_events_v5(features,policy)
    suppressed={i for f in flashes for i in range(f['start_frame'],min(n,f['end_frame_exclusive']+1))}
    adjacent=[(0.,0.)]+[distances(features[i-1],features[i]) for i in range(1,n)]
    hard=[i for i,(h,g) in enumerate(adjacent) if i and i not in suppressed and
          ((h>=policy.hard_hist and g>=policy.hard_gray) or h>=policy.strong_hist)]
    flags=[];scores=[];counts=[]
    for i in range(n):
        a=max(0,i-policy.gradual_radius);b=min(n-1,i+policy.gradual_radius)
        h,_=distances(features[a],features[b]);scores.append(h)
        active=sum(adjacent[j][0]>=policy.gradual_adjacent_hist for j in range(a+1,b+1)
                   if j not in suppressed and j not in hard)
        counts.append(active)
        flags.append(bool(h>=policy.gradual_window_hist and active>=policy.gradual_active_edges and
                          not any(j in suppressed for j in range(a,b+1))))
    regions=merge_regions(runs(flags),n,gap=policy.gradual_bridge_frames,pad=policy.gradual_pad_frames)
    regions=distinguish_hard_regions(regions,hard,policy)
    cuts=[c for c in hard if not any(a<=c<b for a,b in regions)]
    # The older probability-only validator must not see v4's dominance ratio.
    scene_policy=ScenePolicyV3(**{k:getattr(policy,k) for k in asdict(ScenePolicyV3()) if k!='version'})
    base=clean_segments(n,cuts,regions,scene_policy)
    base.update(hard_cut_candidates=cuts,raw_hard_cut_candidates=hard,exposure_flash_candidates=flashes,
                gradual_change_regions=[{'start_frame':a,'end_frame_exclusive':b} for a,b in regions],
                per_frame=[{'frame_index':i,'histogram_distance':adjacent[i][0],'gray_difference':adjacent[i][1],
                            'luma':float(features[i]['gray'].mean()),'luma_peak':peaks[i],
                            'window_histogram_distance':scores[i],'active_edges':counts[i],
                            'flash_boundary_suppressed':i in suppressed,
                            'floor_area_diagnostic_only':features[i]['floor_area'],
                            'local_max_tile_luma_peak':float(tile_peaks[i].max()),
                            'local_connected_bright_tiles':components[i]} for i in range(n)])
    result=refine_analysis(base,base_policy(policy))
    result.update(policy=asdict(policy),local_exposure_diagnostics=diagnostics,
                  v3_global_exposure_and_numeric_thresholds_unchanged=True,
                  patch_scope='spatially_local_seed_supplement_only; not camera-cut sensitivity or semantic event localization',
                  original_pixels_modified=False,training_use_approved=False)
    return result
