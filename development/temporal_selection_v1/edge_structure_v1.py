"""Frozen development-only edge/motion supplement. Never selects a target event.

Inspired by edge change ratio, not a reproduction of Zabih et al. (1999).
Analysis coordinates exclude overlays; no transformed pixels reach a Dataset.
"""
from dataclasses import asdict, dataclass
import math

import cv2
import numpy as np

from .selector import need
from .selector_v4 import all_segments


@dataclass(frozen=True)
class EdgePolicy:
    version: str = 'edge_structure_v1_fixed8_diagnostic_only'
    width: int = 256
    height: int = 144
    roi_top: int = 24
    roi_bottom: int = 204
    gaussian_size: int = 5
    canny_low: int = 50
    canny_high: int = 150
    match_radius: int = 2
    min_edge_pixels: int = 60
    max_corners: int = 300
    corner_quality: float = .01
    corner_distance: int = 5
    lk_window: int = 21
    lk_levels: int = 3
    forward_backward_px: float = 1.5
    min_tracks: int = 12
    ransac_px: float = 2.
    min_inlier_ratio: float = .6
    min_quadrants: int = 3
    min_overlap: float = .75
    min_scale: float = .9
    max_scale: float = 1.1
    max_rotation_degrees: float = 10.
    max_translation_fraction: float = .2
    abrupt_ecr: float = .65
    abrupt_hist: float = .12
    motion_raw_ecr: float = .45
    motion_residual_ecr: float = .25
    motion_min_gain: float = .2
    strong_hist_no_veto: float = .7
    gradual_ecr: float = .35
    gradual_hist: float = .055
    seed: int = 20261028


def validate_policy(p):
    # A single frozen prototype, not a threshold sweep interface.
    need(isinstance(p, EdgePolicy) and p == EdgePolicy(), 'Only frozen EdgePolicy v1 supported')


def gray_frame(frame, p=EdgePolicy()):
    validate_policy(p)
    need(isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and
         frame.shape == (240, 320, 3), 'Need native 320x240 uint8 BGR')
    small = cv2.resize(frame[p.roi_top:p.roi_bottom], (p.width, p.height), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
                            (p.gaussian_size, p.gaussian_size), 0)


def edges(gray, p=EdgePolicy()):
    return cv2.Canny(gray, p.canny_low, p.canny_high, L2gradient=True) > 0


def edge_change(before, after, valid=None, p=EdgePolicy()):
    """Bidirectional unmatched-edge fraction, tolerant within radius2 pixels.

    Both denominators and matches use the common valid interior. Low texture is
    unknown, never zero evidence of a cut and never evidence of no basketball.
    """
    need(before.shape == after.shape and before.dtype == after.dtype == np.bool_, 'Boolean edge maps required')
    if valid is None:
        valid = np.ones(before.shape, np.uint8)
    else:
        need(valid.shape == before.shape, 'Invalid overlap mask')
    radius = p.match_radius
    kernel = np.ones((2*radius+1, 2*radius+1), np.uint8)
    interior = cv2.erode(valid.astype(np.uint8), kernel,
                         borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    a, b = before & interior, after & interior
    na, nb = int(a.sum()), int(b.sum())
    if min(na, nb) < p.min_edge_pixels:
        return {'valid': False, 'ecr': None, 'outgoing': None, 'incoming': None,
                'before_edges': na, 'after_edges': nb, 'reason': 'insufficient_edges'}
    da = cv2.dilate(a.astype(np.uint8), kernel) > 0
    db = cv2.dilate(b.astype(np.uint8), kernel) > 0
    outgoing = float((a & ~db).sum()/na)
    incoming = float((b & ~da).sum()/nb)
    return {'valid': True, 'ecr': max(outgoing, incoming), 'outgoing': outgoing,
            'incoming': incoming, 'before_edges': na, 'after_edges': nb, 'reason': 'measured'}


def estimate_motion(before, after, p=EdgePolicy()):
    """Forward/backward LK + RANSAC similarity; reject unreliable fits.

    A passing fit is only geometric evidence: independently moving players,
    parallax, repeated textures and similar-looking cuts remain limitations.
    """
    result = {'accepted': False, 'reason': 'insufficient_corners', 'corners': 0,
              'tracks': 0, 'inliers': 0, 'inlier_ratio': 0., 'quadrants': 0,
              'overlap': None, 'matrix': None}
    points = cv2.goodFeaturesToTrack(before, maxCorners=p.max_corners,
        qualityLevel=p.corner_quality, minDistance=p.corner_distance, blockSize=7)
    if points is None or len(points) < p.min_tracks:
        return result
    result['corners'] = len(points)
    kwargs = dict(winSize=(p.lk_window, p.lk_window), maxLevel=p.lk_levels,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    forward, status, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None, **kwargs)
    if forward is None or status is None:
        result['reason'] = 'forward_flow_failed'
        return result
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(after, before, forward, None, **kwargs)
    if backward is None or back_status is None:
        result['reason'] = 'backward_flow_failed'
        return result
    a, b, back = points[:, 0], forward[:, 0], backward[:, 0]
    h, w = before.shape
    good = ((status[:, 0] == 1) & (back_status[:, 0] == 1) &
            np.isfinite(b).all(axis=1) & np.isfinite(back).all(axis=1) &
            (np.linalg.norm(a-back, axis=1) <= p.forward_backward_px) &
            (b[:, 0] >= 0) & (b[:, 0] < w) & (b[:, 1] >= 0) & (b[:, 1] < h))
    a, b = a[good], b[good]
    result.update(tracks=len(a), reason='insufficient_consistent_tracks')
    if len(a) < p.min_tracks:
        return result
    cv2.setRNGSeed(p.seed)
    matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
        ransacReprojThreshold=p.ransac_px, maxIters=2000, confidence=.99, refineIters=10)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        result['reason'] = 'ransac_failed'
        return result
    selected = inliers[:, 0] > 0
    count = int(selected.sum())
    # Require spatial support in both frames, not a fit to one player alone.
    def quadrants(points):
        return len(set((points[:, 0] >= w/2).astype(int) + 2*(points[:, 1] >= h/2).astype(int)))
    quadrants_count = min(quadrants(a[selected]), quadrants(b[selected])) if count else 0
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    rotation = float(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
    mask = cv2.warpAffine(np.ones((h, w), np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
    overlap = float(mask.mean())
    result.update(inliers=count, inlier_ratio=count/len(a), quadrants=quadrants_count,
                  scale=scale, rotation_degrees=rotation, overlap=overlap, matrix=matrix.tolist())
    if count < p.min_tracks or count/len(a) < p.min_inlier_ratio or quadrants_count < p.min_quadrants:
        result['reason'] = 'insufficient_distributed_inliers'
    elif not p.min_scale <= scale <= p.max_scale or abs(rotation) > p.max_rotation_degrees:
        result['reason'] = 'transform_out_of_bounds'
    elif abs(matrix[0, 2]) > w*p.max_translation_fraction or abs(matrix[1, 2]) > h*p.max_translation_fraction:
        result['reason'] = 'translation_out_of_bounds'
    elif overlap < p.min_overlap:
        result['reason'] = 'insufficient_common_area'
    else:
        result.update(accepted=True, reason='distributed_similarity_fit')
    return result


def pair_metrics(before, after, p=EdgePolicy(), *, return_maps=False):
    a, b = edges(before, p), edges(after, p)
    raw = edge_change(a, b, p=p)
    motion = estimate_motion(before, after, p)
    aligned = None
    raw_common = None
    warped = a
    if motion['accepted']:
        matrix = np.asarray(motion['matrix'])
        h, w = a.shape
        warped = cv2.warpAffine(a.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST) > 0
        valid = cv2.warpAffine(np.ones(a.shape, np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
        # Compare on the same output footprint. Otherwise the apparent gain
        # could partly come from discarding newly exposed border content.
        raw_common = edge_change(a, b, valid, p)
        aligned = edge_change(warped, b, valid, p)
    # Do not silently choose the smaller of two scores. Accepted common-area
    # geometry determines which score is used; both scores remain in the report.
    used = bool(aligned and aligned['valid'])
    effective = aligned if used else raw
    result = {'raw': raw, 'raw_common_overlap': raw_common, 'aligned': aligned, 'motion': motion,
              'compensation_used': used, 'effective_ecr': effective['ecr'],
              'score_valid': effective['valid'],
              'raw_minus_aligned': raw_common['ecr']-aligned['ecr']
              if used and raw_common['valid'] else None}
    if return_maps:
        return result, (a, b, warped)
    return result


def propose(metrics, baseline, p=EdgePolicy()):
    """Counterfactual scene proposals; never crop, choose longest, or approve.

    Strong existing color cuts are protected from a motion veto. Gradual spans
    can only narrow inside the frozen v5 search span with >=2 structural edges;
    insufficient structure retains the old span. End censoring is preserved.
    """
    n = len(metrics)
    need(n == len(baseline['per_frame']) and
         [m['frame_index'] for m in metrics] == list(range(n)), 'Sequential metrics required')
    cuts = set(baseline['hard_cut_candidates'])
    suppressed, vetoed, additions, activity = set(), [], [], []
    for e in baseline['exposure_flash_candidates']:
        suppressed.update(range(e['start_frame'], min(n, e['end_frame_exclusive']+1)))
    for i in range(1, n):
        m, v = metrics[i], baseline['per_frame'][i]
        score = m['effective_ecr']
        if i in suppressed or score is None:
            continue
        hist = v['histogram_distance']
        gain = m['raw_minus_aligned']
        if (i in cuts and m['compensation_used'] and gain is not None and
            hist < p.strong_hist_no_veto and m['raw']['ecr'] >= p.motion_raw_ecr and
            score <= p.motion_residual_ecr and gain >= p.motion_min_gain):
            cuts.remove(i)
            vetoed.append(i)
        if score >= p.abrupt_ecr and hist >= p.abrupt_hist:
            additions.append(i)
        if score >= p.gradual_ecr and hist >= p.gradual_hist:
            activity.append(i)
    # Keep one strongest edge per consecutive response cluster for new cuts.
    clusters = []
    for i in additions:
        if clusters and i == clusters[-1][-1]+1:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    peaks = [max(group, key=lambda i: (metrics[i]['effective_ecr'], -i)) for group in clusters]
    cuts.update(peaks)
    regions, region_checks = [], []
    for region in baseline['gradual_change_regions']:
        a, b = region['start_frame'], region['end_frame_exclusive']
        support = [i for i in activity if max(1, a) <= i <= min(n-1, b)]
        proposed = [a, b]
        if len(support) >= 2:
            left, right = support[0], support[-1]
            if b == n and right == n-1:
                right = n
            if left < right:
                proposed = [left, right]
        regions.append(proposed)
        region_checks.append({'v5_span': [a, b], 'proposed_span': proposed,
                              'support_edges': support, 'changed': proposed != [a, b],
                              'semantic_transition_verified': False})
    final = sorted(i for i in cuts if not any(a <= i < b for a, b in regions))
    return {'hard_cut_candidates': final, 'gradual_change_regions': regions,
            'motion_vetoed_v5_cuts': vetoed, 'new_edge_cut_peaks_before_region_filter': peaks,
            'added_cuts_vs_v5': sorted(set(final)-set(baseline['hard_cut_candidates'])),
            'removed_cuts_vs_v5': sorted(set(baseline['hard_cut_candidates'])-set(final)),
            'gradual_checks': region_checks, 'all_segments_including_short': all_segments(n, final, regions),
            'target_event_selected': False, 'training_use_approved': False,
            'production_crop_adoptions': 0, 'effective_training_policy': 'original_three_windows_unchanged'}


def analyze_edges(frames, baseline, p=EdgePolicy()):
    validate_policy(p)
    need(len(frames) >= 2 and len(frames) == len(baseline['per_frame']), 'Full continuous sequence required')
    gray = [gray_frame(f, p) for f in frames]
    metrics = [{'frame_index': 0, 'raw': {'valid': False, 'ecr': None},
                'raw_common_overlap': None, 'aligned': None,
                'motion': {'accepted': False, 'reason': 'source_start'},
                'compensation_used': False, 'effective_ecr': None, 'score_valid': False,
                'raw_minus_aligned': None, 'laplacian_variance_diagnostic_only':
                float(cv2.Laplacian(gray[0], cv2.CV_32F).var())}]
    for i in range(1, len(frames)):
        row = pair_metrics(gray[i-1], gray[i], p)
        row.update(frame_index=i, laplacian_variance_diagnostic_only=
                   float(cv2.Laplacian(gray[i], cv2.CV_32F).var()))
        metrics.append(row)
    return {'policy': asdict(p), 'per_frame': metrics, 'hybrid': propose(metrics, baseline, p),
            'original_pixels_modified': False, 'laplacian_used_for_selection': False,
            'hough_lines_used_for_selection': False, 'labels_used_for_selection': False}
