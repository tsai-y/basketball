"""Deterministic continuous clip selection from candidate-segment metadata.

The module never reads video data.  Every selected clip is a half-open,
contiguous interval in the original frame index space.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable, Mapping, Sequence


POLICY_VERSION = "quality_aware_sampler_v1"


class SamplerError(ValueError):
    """Raised when sidecar metadata or a sampling request is invalid."""


@dataclass(frozen=True, order=True)
class CandidateSegment:
    start_frame: int
    end_frame_exclusive: int


@dataclass(frozen=True)
class ClipSelection:
    video_id: str
    clip_index: int
    start_frame: int
    end_frame_exclusive: int
    candidate_overlap: int
    fallback: bool
    fallback_reason: str | None
    policy_version: str
    selection_mode: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _coerce_segment(value: CandidateSegment | Mapping[str, object] | Sequence[int]) -> CandidateSegment:
    if isinstance(value, CandidateSegment):
        return value
    if isinstance(value, Mapping):
        return CandidateSegment(int(value["start_frame"]), int(value["end_frame_exclusive"]))
    if len(value) != 2:
        raise SamplerError("A candidate segment must have exactly two endpoints")
    return CandidateSegment(int(value[0]), int(value[1]))


def validate_segments(
    segments: Iterable[CandidateSegment | Mapping[str, object] | Sequence[int]],
    frame_count: int,
) -> tuple[CandidateSegment, ...]:
    """Return validated segments, rejecting unsorted, overlapping metadata."""
    if frame_count <= 0:
        raise SamplerError("frame_count must be positive")
    result = tuple(_coerce_segment(value) for value in segments)
    previous_end = 0
    for index, segment in enumerate(result):
        if segment.start_frame < 0 or segment.end_frame_exclusive <= segment.start_frame:
            raise SamplerError("Candidate segments must be positive half-open intervals")
        if segment.end_frame_exclusive > frame_count:
            raise SamplerError("Candidate segment exceeds frame_count")
        if index and segment.start_frame < previous_end:
            raise SamplerError("Candidate segments must be sorted and non-overlapping")
        previous_end = segment.end_frame_exclusive
    return result


def candidate_overlap(start: int, end_frame_exclusive: int, segments: Iterable[CandidateSegment]) -> int:
    """Independently calculate the number of candidate frames in a window."""
    if start < 0 or end_frame_exclusive <= start:
        raise SamplerError("Clip must be a positive half-open interval")
    return sum(max(0, min(end_frame_exclusive, item.end_frame_exclusive) - max(start, item.start_frame))
               for item in segments)


def window_costs(frame_count: int, clip_length: int, segments: Iterable[CandidateSegment]) -> list[int]:
    """Calculate candidate overlap for every legal contiguous clip start."""
    if clip_length <= 0:
        raise SamplerError("clip_length must be positive")
    if clip_length > frame_count:
        raise SamplerError("clip_length exceeds frame_count")
    checked = validate_segments(segments, frame_count)
    candidates = [0] * (frame_count + 1)
    for segment in checked:
        candidates[segment.start_frame] += 1
        candidates[segment.end_frame_exclusive] -= 1
    for index in range(1, frame_count):
        candidates[index] += candidates[index - 1]
    width_cost = sum(candidates[:clip_length])
    costs = [width_cost]
    for start in range(1, frame_count - clip_length + 1):
        width_cost += candidates[start + clip_length - 1] - candidates[start - 1]
        costs.append(width_cost)
    return costs


def _rank(seed: int, video_id: str, epoch: int, clip_index: int, start: int, policy_version: str) -> bytes:
    material = f"{seed}\0{video_id}\0{epoch}\0{clip_index}\0{policy_version}\0{start}".encode("utf-8")
    return hashlib.sha256(material).digest()


def select_clips(
    *,
    video_id: str,
    frame_count: int,
    candidate_segments: Iterable[CandidateSegment | Mapping[str, object] | Sequence[int]],
    clip_length: int,
    num_clips: int = 1,
    seed: int = 0,
    epoch: int = 0,
    selection_mode: str = "seeded_random",
    policy_version: str = POLICY_VERSION,
    candidate_margin_frames: int = 0,
) -> list[ClipSelection]:
    """Select reproducible clips, preferring zero candidate overlap.

    Multiple clips use unique starts while possible.  After the first choice,
    maximum distance from already selected starts improves temporal coverage;
    SHA-256 ranks break ties reproducibly.  If requests exceed the pool size,
    starts repeat in deterministic per-clip hash order.
    """
    if not video_id:
        raise SamplerError("video_id is required")
    if num_clips < 1:
        raise SamplerError("num_clips must be at least 1")
    if candidate_margin_frames != 0:
        raise SamplerError("quality_aware_sampler_v1 requires candidate_margin_frames=0")
    if selection_mode not in {"seeded_random", "earliest"}:
        raise SamplerError(f"Unsupported selection_mode: {selection_mode}")
    checked = validate_segments(candidate_segments, frame_count)
    costs = window_costs(frame_count, clip_length, checked)
    minimum = min(costs)
    fallback = minimum != 0
    pool = [start for start, cost in enumerate(costs) if cost == minimum]
    selected: list[int] = []
    records: list[ClipSelection] = []
    for clip_index in range(num_clips):
        unused = [start for start in pool if start not in selected]
        choices = unused or pool
        if selection_mode == "earliest":
            chosen = min(choices)
        else:
            if unused and selected:
                max_distance = max(min(abs(start - prior) for prior in selected) for start in choices)
                choices = [start for start in choices
                           if min(abs(start - prior) for prior in selected) == max_distance]
            chosen = min(choices, key=lambda start: (_rank(seed, video_id, epoch, clip_index, start,
                                                            policy_version), start))
        selected.append(chosen)
        end = chosen + clip_length
        overlap = candidate_overlap(chosen, end, checked)
        if overlap != costs[chosen]:
            raise AssertionError("Internal candidate-overlap mismatch")
        records.append(ClipSelection(
            video_id=video_id, clip_index=clip_index, start_frame=chosen,
            end_frame_exclusive=end, candidate_overlap=overlap, fallback=fallback,
            fallback_reason="no_zero_candidate_overlap_window" if fallback else None,
            policy_version=policy_version, selection_mode=selection_mode))
    return records

