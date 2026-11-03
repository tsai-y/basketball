"""Versioned exposure-tail recovery; diagnostic scenes only, never event selection."""
from dataclasses import dataclass, asdict

import numpy as np

from .selector import descriptor, distances, runs, need
from .selector_v2 import (ScenePolicy, validate, merge_regions, correlation,
                          clean_segments, distinguish_hard_regions)


@dataclass(frozen=True)
class ScenePolicyV3(ScenePolicy):
    version: str = 'scene_segmentation_v3_exposure_tail_not_target_event_selector'
    recovery_hold_frames: int = 2
    recovery_luma_tolerance: float = .04


def validate_v3(policy):
    need(isinstance(policy, ScenePolicyV3), 'Need an explicit v3 policy')
    validate(policy)
    need(policy.recovery_hold_frames >= 1, 'Recovery must persist for at least one frame')


def exposure_events_v3(features, policy=ScenePolicyV3()):
    """Bounded forward recovery search, with shape/brightness and persistence checks.

    A peak is a seed, not an exposure endpoint. Search only up to the existing
    flash_max_frames limit. Every intermediate frame must still resemble the
    preceding scene; a brief unrelated bright scene must not count as a flash.
    This remains a heuristic: similar-looking cuts can be indistinguishable.
    """
    validate_v3(policy); n = len(features); need(n >= 2, 'Need at least two frames')
    means = [float(f['gray'].mean()) for f in features]
    peaks = [means[i] - float(np.median(means[max(0, i-policy.luma_window):
                                              min(n, i+policy.luma_window+1)])) for i in range(n)]
    seeds = merge_regions(runs([p >= policy.luma_peak for p in peaks]), n, gap=1)
    accepted = []; last_end = -1
    for a, peak_end in seeds:
        if a == 0 or a < last_end or peak_end-a > policy.flash_max_frames:
            continue
        before = features[a-1]; baseline = means[a-1]
        latest = min(a+policy.flash_max_frames, n-policy.recovery_hold_frames)
        for end in range(peak_end, latest+1):
            intermediate_ok = all(
                means[j] >= baseline-policy.recovery_luma_tolerance
                and correlation(before['gray'], features[j]['gray']) >= policy.recovery_correlation
                for j in range(a, end))
            if not intermediate_ok:
                # Extending farther cannot remove an unrelated intermediate scene.
                break
            checks = []
            for j in range(end, end+policy.recovery_hold_frames):
                h, _ = distances(before, features[j])
                # First return must match the pre-exposure scene. Persistence
                # then compares against that returned frame, not an ever-older
                # spatial anchor: ordinary camera motion must not fail the hold.
                anchor = before if j == end else features[end]
                c = correlation(anchor['gray'], features[j]['gray'])
                delta = abs(means[j]-baseline)
                checks.append((h, c, delta))
            if all(h <= policy.recovery_hist and c >= policy.recovery_correlation
                   and delta <= policy.recovery_luma_tolerance for h, c, delta in checks):
                accepted.append({'start_frame': a, 'end_frame_exclusive': end,
                                 'peak_end_frame_exclusive': peak_end,
                                 'recovery_histogram': checks[0][0], 'recovery_correlation': checks[0][1],
                                 'recovery_luma_delta': checks[0][2],
                                 'recovery_hold_frames': policy.recovery_hold_frames,
                                 'pixels_removed': False})
                last_end = end
                break
    return accepted, peaks


def analyze_v3(frames, policy=ScenePolicyV3()):
    validate_v3(policy); need(len(frames) >= 2, 'Need continuous native frames')
    features = [descriptor(frame) for frame in frames]; n = len(features)
    flashes, luma_peaks = exposure_events_v3(features, policy)
    suppressed = {i for f in flashes for i in range(f['start_frame'], min(n, f['end_frame_exclusive']+1))}
    adjacent = [(0., 0.)] + [distances(features[i-1], features[i]) for i in range(1, n)]
    hard = [i for i, (h, g) in enumerate(adjacent) if i and i not in suppressed and
            ((h >= policy.hard_hist and g >= policy.hard_gray) or h >= policy.strong_hist)]
    flags = []; scores = []; counts = []
    for i in range(n):
        a = max(0, i-policy.gradual_radius); b = min(n-1, i+policy.gradual_radius)
        h, _ = distances(features[a], features[b]); scores.append(h)
        active = sum(adjacent[j][0] >= policy.gradual_adjacent_hist for j in range(a+1, b+1)
                     if j not in suppressed and j not in hard)
        counts.append(active)
        flags.append(bool(h >= policy.gradual_window_hist and active >= policy.gradual_active_edges
                          and not any(j in suppressed for j in range(a, b+1))))
    regions = merge_regions(runs(flags), n, gap=policy.gradual_bridge_frames, pad=policy.gradual_pad_frames)
    regions = distinguish_hard_regions(regions, hard, policy)
    cuts = [c for c in hard if not any(a <= c < b for a, b in regions)]
    result = clean_segments(n, cuts, regions, policy)
    result.update(policy=asdict(policy), hard_cut_candidates=cuts, raw_hard_cut_candidates=hard,
                  exposure_flash_candidates=flashes,
                  gradual_change_regions=[{'start_frame': a, 'end_frame_exclusive': b,
                    'interpretation': 'possible_transition_or_fast_appearance_change_NOT_confirmed_graphic'} for a,b in regions],
                  floor_appearance_used_as_exclusion_rule=False, input_metadata_used_for_selection=False,
                  per_frame=[{'frame_index': i, 'histogram_distance': adjacent[i][0],
                    'gray_difference': adjacent[i][1], 'luma': float(features[i]['gray'].mean()),
                    'luma_peak': luma_peaks[i], 'window_histogram_distance': scores[i], 'active_edges': counts[i],
                    'flash_boundary_suppressed': i in suppressed,
                    'floor_area_diagnostic_only': features[i]['floor_area']} for i in range(n)])
    return result
