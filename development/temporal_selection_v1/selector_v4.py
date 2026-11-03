"""Refine padded v3 search regions; appearance proposals, not semantic graphic masks."""
from dataclasses import dataclass, asdict

from .selector import need
from .selector_v2 import clean_segments
from .selector_v3 import ScenePolicyV3, analyze_v3, validate_v3


@dataclass(frozen=True)
class BoundaryPolicy(ScenePolicyV3):
    version: str = 'scene_boundary_v4_diagnostic_not_semantic_event_selection'
    local_cut_hist: float = .40
    local_cut_gray: float = .14
    local_peak_dominance: float = 2.5
    boundary_activity_hist: float = .15
    boundary_activity_gray: float = .07
    gap_active_fraction: float = .5


def validate_boundary(policy):
    # v3 validates float thresholds as probabilities; peak dominance is a ratio.
    need(isinstance(policy, BoundaryPolicy), 'Need explicit boundary policy')
    validate_v3(ScenePolicyV3(**{k: getattr(policy,k) for k in asdict(ScenePolicyV3())}))
    for key in ('local_cut_hist','local_cut_gray','boundary_activity_hist','boundary_activity_gray','gap_active_fraction'):
        value=getattr(policy,key)
        need(type(value) is float and 0 < value < 1, 'Invalid refinement threshold')
    need(type(policy.local_peak_dominance) is float and policy.local_peak_dominance > 1,
         'Invalid localized-peak dominance')


def refine_regions(base, policy=BoundaryPolicy()):
    """Use adjacent-change edges inside coarse search episodes, without frame IDs.

    Edge i is between i-1 and i. A gradual episode [first,last) excludes the
    changing span, not the first returned frame at last. At video end, no return
    can be certified; preserve right-censoring instead of treating last as clean.
    """
    validate_boundary(policy)
    metrics=base['per_frame'];n=len(metrics)
    need(n>=2 and [m['frame_index'] for m in metrics]==list(range(n)), 'Invalid sequential metrics')
    active=lambda i: (not metrics[i]['flash_boundary_suppressed'] and
        (metrics[i]['histogram_distance']>=policy.boundary_activity_hist or
         metrics[i]['gray_difference']>=policy.boundary_activity_gray))
    episodes=[]
    for r in base['gradual_change_regions']:
        a,b=r['start_frame'],r['end_frame_exclusive']
        need(0<=a<b<=n,'Invalid coarse region')
        # Merge only when measured change persists through a small gap, not just
        # because a scene is short. This remains appearance evidence, not a logo class.
        if episodes and a-episodes[-1][1]<=2*policy.gradual_radius:
            gap=list(range(episodes[-1][1],a))
            if not gap or sum(active(i) for i in gap)/len(gap)>=policy.gap_active_fraction:
                episodes[-1]=(episodes[-1][0],b);continue
        episodes.append((a,b))
    regions=[];cuts=set(base['hard_cut_candidates']);diagnostics=[]
    for a,b in episodes:
        edges=[i for i in range(max(1,a),min(n-1,b)+1) if not metrics[i]['flash_boundary_suppressed']]
        ranked=sorted(edges,key=lambda i:metrics[i]['histogram_distance'],reverse=True)
        peak=ranked[0] if ranked else None
        second=metrics[ranked[1]]['histogram_distance'] if len(ranked)>1 else 0.
        dominance=(metrics[peak]['histogram_distance']/max(second,1e-8)) if peak is not None else 0.
        abrupt=(peak is not None and metrics[peak]['histogram_distance']>=policy.local_cut_hist
                and metrics[peak]['gray_difference']>=policy.local_cut_gray
                and dominance>=policy.local_peak_dominance)
        evidence=[i for i in edges if active(i)]
        d={'coarse_search_span':[a,b],'dominant_edge':peak,'dominance_ratio':dominance,
           'semantic_graphic_verified':False,'training_use_approved':False}
        if abrupt:
            cuts.add(peak);d.update(kind='localized_cut_candidate',cut_frame=peak,
                proposed_change_span=None,limitation='A localized appearance step is not semantic proof of a true cut.')
        elif len(evidence)>=2:
            left,right=evidence[0],evidence[-1]
            censored=b==n and right==n-1
            if censored:right=n
            if right>left:
                regions.append((left,right));d.update(kind='trimmed_appearance_change_candidate',
                    proposed_change_span=[left,right],right_censored=censored,
                    limitation='Activity may be camera/player motion, not a graphic; last active edge is a return-frame proposal, not certified clean.')
            else:
                regions.append((a,b));d.update(kind='unresolved_keep_coarse',proposed_change_span=[a,b])
        else:
            regions.append((a,b));d.update(kind='unresolved_keep_coarse',proposed_change_span=[a,b],
                limitation='Insufficient local evidence: keep old region, do not invent clean frames.')
        diagnostics.append(d)
    cuts=sorted(c for c in cuts if not any(a<=c<b for a,b in regions))
    return cuts,regions,diagnostics


def all_segments(n,cuts,regions):
    bounds=sorted({0,n,*cuts,*[p for r in regions for p in r]})
    return [{'start_frame':a,'end_frame_exclusive':b,'length':b-a,
             'inside_proposed_change_span':any(a<y and b>x for x,y in regions)}
            for a,b in zip(bounds,bounds[1:])]


def refine_analysis(base,policy=BoundaryPolicy()):
    cuts,regions,diagnostics=refine_regions(base,policy);n=len(base['per_frame'])
    common=ScenePolicyV3(**{k:getattr(policy,k) for k in asdict(ScenePolicyV3())})
    result=clean_segments(n,cuts,regions,common)
    result.update(policy=asdict(policy),hard_cut_candidates=cuts,
        raw_hard_cut_candidates=base['raw_hard_cut_candidates'],
        exposure_flash_candidates=base['exposure_flash_candidates'],
        gradual_change_regions=[{'start_frame':a,'end_frame_exclusive':b,
            'interpretation':'refined_appearance_change_NOT_confirmed_graphic'} for a,b in regions],
        coarse_v3_search_regions=base['gradual_change_regions'],boundary_refinement=diagnostics,
        all_segments_including_short=all_segments(n,cuts,regions),per_frame=base['per_frame'],
        input_metadata_used_for_selection=False,floor_appearance_used_as_exclusion_rule=False,
        semantic_graphic_mask_implemented=False,boundaries_human_confirmed=False,
        ambiguous_case_fallback='original_three_window_policy_unchanged; never deploy these candidates automatically')
    return result


def analyze_v4(frames,policy=BoundaryPolicy()):
    validate_boundary(policy)
    # Exposure recovery and its numeric settings remain the exact v3 method.
    common={k:getattr(policy,k) for k in asdict(ScenePolicyV3()) if k!='version'}
    base=analyze_v3(frames,ScenePolicyV3(**common))
    return refine_analysis(base,policy)
