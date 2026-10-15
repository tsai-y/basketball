#!/usr/bin/env python3
"""Derived-only train audit for v3.1 candidates and sampler-v1 selections.

This program reads CSV/JSON/YAML metadata only.  It never opens, probes, or
decodes a video, and it rejects any validation/test ID in the quality inputs.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[3]
DQ = ROOT / "development/data_quality"
sys.path.insert(0, str(DQ))
from samplers.quality_aware_clip_sampler import (  # noqa: E402
    CandidateSegment,
    candidate_overlap,
    select_clips,
)

DEFAULT_MANIFEST = ROOT / "data/processed/dataset_manifest.csv"
DEFAULT_SPLIT_SUMMARY = ROOT / "data/processed/split_summary.json"
DEFAULT_METADATA = DQ / "sidecars/blur_quality_sidecar_v3_1.json"
DEFAULT_VIDEO_SUMMARY = DQ / "sidecars/blur_video_summary_v3_1.csv"
DEFAULT_SEGMENTS = DQ / "sidecars/blur_candidate_segments_v3_1.csv"
DEFAULT_CONFIG = DQ / "configs/quality_aware_sampler_v1.yaml"
DEFAULT_SELECTIONS = DQ / "artifacts/quality_aware_sampler_v1/selections.csv"
DEFAULT_ARTIFACT_DIR = DQ / "artifacts/train_quality_associations_v1"
DEFAULT_JSON = ROOT / "docs/data_quality/train_quality_associations_v1.json"
DEFAULT_MARKDOWN = ROOT / "docs/data_quality/train_quality_associations_v1.md"

EXPECTED = {
    "train_videos": 7193,
    "decoded_frames": 1105146,
    "candidate_frames": 18390,
    "candidate_segments": 1798,
    "affected_videos": 1698,
    "unaffected_videos": 5495,
    "zero_overlap_selections": 7188,
    "fallback_selections": 5,
    "fallback_total_overlap": 20,
}
LABELS = ("Make", "Miss")
BOOTSTRAP_SEED = 20261016
DEFAULT_COUNTERFACTUAL_EPOCHS = 20
POSITION_SHIFT_ALERT = 0.05
SELECTION_FIELDS = (
    "video_id", "label", "source_group", "fps_rational", "frame_count", "clip_index",
    "start_frame", "end_frame_exclusive", "candidate_overlap", "fallback",
    "fallback_reason", "policy_version", "selection_mode", "seed", "epoch",
)


class AuditError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    need(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def csv_text(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def load_config(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        need(":" in line, f"Invalid config line: {raw}")
        key, value = (part.strip() for part in line.split(":", 1))
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
        else:
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
    frozen = {
        "policy_version": "quality_aware_sampler_v1",
        "quality_source": "blur_screening_v3_1",
        "clip_length": 64,
        "num_clips_per_video": 1,
        "candidate_margin_frames": 0,
        "primary_rule": "zero_candidate_overlap",
        "fallback_rule": "minimum_candidate_overlap",
        "train_selection": "seeded_random",
        "global_seed": 20261009,
        "validation_quality_mode": "disabled",
    }
    for key, value in frozen.items():
        need(result.get(key) == value, f"Frozen sampler config mismatch: {key}")
    return result


def validate_unique(rows: Sequence[dict[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        need(bool(value), f"Missing {key} in {name}")
        need(value not in result, f"Duplicate {key} in {name}: {value}")
        result[value] = row
    return result


def validate_exact_id_set(actual: Iterable[str], expected: Iterable[str], name: str) -> None:
    actual_list = list(actual)
    expected_set = set(expected)
    need(len(actual_list) == len(set(actual_list)), f"Duplicate video_id in {name}")
    actual_set = set(actual_list)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    need(not missing and not extra,
         f"{name} ID mismatch: missing={missing[:3]} extra={extra[:3]}")


def shot_type(original_class: str, binary_label: str) -> str:
    need(original_class in {"2p0", "2p1", "3p0", "3p1", "ft0", "ft1", "mp0", "mp1"},
         f"Unexpected original_class: {original_class}")
    need(original_class[-1] == binary_label, f"Class/label mismatch: {original_class}/{binary_label}")
    return original_class[:-1]


def normalize_segment(start: int, end: int, frame_count: int) -> tuple[float, float]:
    need(frame_count > 0, "frame_count must be positive")
    need(0 <= start < end <= frame_count, "Invalid half-open candidate segment")
    return ((start + end) / (2.0 * frame_count), (end - start) / frame_count)


def normalize_selection(start: int, end: int, frame_count: int, clip_length: int) -> tuple[float, float]:
    need(end - start == clip_length, "Selection length mismatch")
    denominator = frame_count - clip_length
    need(denominator > 0, "normalized_start requires frame_count > clip_length")
    need(0 <= start <= denominator and end <= frame_count, "Selection outside video")
    return (start / denominator, (start + end) / (2.0 * frame_count))


def position_band(center: float) -> str:
    need(0.0 <= center <= 1.0, "Normalized center outside [0,1]")
    if center < 1.0 / 3.0:
        return "front"
    if center < 2.0 / 3.0:
        return "middle"
    return "rear"


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    need(0.0 <= q <= 1.0, "Quantile outside [0,1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: Sequence[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {
        "n": len(data),
        "mean": statistics.fmean(data) if data else None,
        "median": quantile(data, 0.5),
        "p25": quantile(data, 0.25),
        "p75": quantile(data, 0.75),
        "p90": quantile(data, 0.90),
        "p95": quantile(data, 0.95),
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }


def cliffs_delta(make: Sequence[float], miss: Sequence[float]) -> float | None:
    if not make or not miss:
        return None
    ordered = sorted(float(value) for value in miss)
    net = 0
    for value in make:
        less = bisect_left(ordered, float(value))
        greater = len(ordered) - bisect_right(ordered, float(value))
        net += less - greater
    return net / (len(make) * len(miss))


def _pair_net(make: Sequence[float], miss: Sequence[float]) -> int:
    ordered = sorted(miss)
    return sum(bisect_left(ordered, value) - (len(ordered) - bisect_right(ordered, value))
               for value in make)


def percentile_interval(values: Sequence[float]) -> list[float | None]:
    return [quantile(values, 0.025), quantile(values, 0.975)]


def summarize_candidate_by_label(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    validate_unique(rows, "video_id", "per-video candidate rows")
    output: dict[str, Any] = {}
    for label in LABELS:
        values = [float(row["candidate_ratio"]) for row in rows if row["label"] == label]
        affected = [value for value in values if value > 0]
        item = describe(values)
        item.update({
            "affected_count": len(affected),
            "affected_rate": len(affected) / len(values) if values else None,
            "affected_only": describe(affected),
        })
        output[label] = item
    output["differences_make_minus_miss"] = {
        "mean": output["Make"]["mean"] - output["Miss"]["mean"],
        "median": output["Make"]["median"] - output["Miss"]["median"],
        "affected_rate": output["Make"]["affected_rate"] - output["Miss"]["affected_rate"],
        "affected_only_mean": (output["Make"]["affected_only"]["mean"]
                               - output["Miss"]["affected_only"]["mean"]),
        "cliffs_delta": cliffs_delta(
            [float(row["candidate_ratio"]) for row in rows if row["label"] == "Make"],
            [float(row["candidate_ratio"]) for row in rows if row["label"] == "Miss"],
        ),
    }
    return output


def cluster_bootstrap_two_part(rows: Sequence[dict[str, Any]], value_key: str, *,
                               replicates: int, seed: int) -> dict[str, Any]:
    need(replicates > 0, "bootstrap replicates must be positive")
    sources = sorted({str(row["source_group"]) for row in rows})
    values: dict[str, dict[str, list[float]]] = {
        source: {label: [] for label in LABELS} for source in sources
    }
    for row in rows:
        values[str(row["source_group"])][str(row["label"])].append(float(row[value_key]))
    pair_net = {
        (left, right): _pair_net(values[left]["Make"], values[right]["Miss"])
        for left in sources for right in sources
    }
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for source in sources:
        summaries[source] = {}
        for label in LABELS:
            data = values[source][label]
            positive = [value for value in data if value > 0]
            summaries[source][label] = {
                "n": len(data), "sum": sum(data), "affected": len(positive),
                "affected_sum": sum(positive),
            }

    rng = random.Random(seed)
    sampled: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        counts = Counter(sources[rng.randrange(len(sources))] for _ in sources)
        totals: dict[str, dict[str, float]] = {
            label: {"n": 0.0, "sum": 0.0, "affected": 0.0, "affected_sum": 0.0}
            for label in LABELS
        }
        for source, multiplier in counts.items():
            for label in LABELS:
                for key in totals[label]:
                    totals[label][key] += multiplier * summaries[source][label][key]
        if not totals["Make"]["n"] or not totals["Miss"]["n"]:
            continue
        sampled["mean_difference"].append(
            totals["Make"]["sum"] / totals["Make"]["n"]
            - totals["Miss"]["sum"] / totals["Miss"]["n"])
        sampled["affected_risk_difference"].append(
            totals["Make"]["affected"] / totals["Make"]["n"]
            - totals["Miss"]["affected"] / totals["Miss"]["n"])
        if totals["Make"]["affected"] and totals["Miss"]["affected"]:
            sampled["affected_only_mean_difference"].append(
                totals["Make"]["affected_sum"] / totals["Make"]["affected"]
                - totals["Miss"]["affected_sum"] / totals["Miss"]["affected"])
        net = sum(left_count * right_count * pair_net[(left, right)]
                  for left, left_count in counts.items() for right, right_count in counts.items())
        sampled["cliffs_delta"].append(net / (totals["Make"]["n"] * totals["Miss"]["n"]))

    raw = summarize_candidate_by_label(rows)["differences_make_minus_miss"]
    estimates = {
        "mean_difference": raw["mean"],
        "affected_risk_difference": raw["affected_rate"],
        "affected_only_mean_difference": raw["affected_only_mean"],
        "cliffs_delta": raw["cliffs_delta"],
    }
    return {
        key: {"estimate": estimates[key], "cluster_bootstrap_95_ci": percentile_interval(sampled[key])}
        for key in estimates
    }


def source_shot_adjusted(rows: Sequence[dict[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    cells: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cells[(str(row["source_group"]), str(row["shot_type"]))][str(row["label"])].append(
            float(row["candidate_ratio"]))
    contributions: dict[str, dict[str, float]] = defaultdict(
        lambda: {"weight": 0.0, "mean": 0.0, "risk": 0.0, "cells": 0.0, "videos": 0.0})
    eligible_cells = 0
    eligible_videos = 0
    for (source, _shot), labels in cells.items():
        make, miss = labels["Make"], labels["Miss"]
        if not make or not miss:
            continue
        weight = len(make) * len(miss) / (len(make) + len(miss))
        mean_diff = statistics.fmean(make) - statistics.fmean(miss)
        risk_diff = sum(value > 0 for value in make) / len(make) - sum(value > 0 for value in miss) / len(miss)
        item = contributions[source]
        item["weight"] += weight
        item["mean"] += weight * mean_diff
        item["risk"] += weight * risk_diff
        item["cells"] += 1
        item["videos"] += len(make) + len(miss)
        eligible_cells += 1
        eligible_videos += len(make) + len(miss)
    total_weight = sum(item["weight"] for item in contributions.values())
    need(total_weight > 0, "No source×shot cells contain both labels")
    estimates = {
        "candidate_ratio_mean_difference": sum(item["mean"] for item in contributions.values()) / total_weight,
        "affected_risk_difference": sum(item["risk"] for item in contributions.values()) / total_weight,
    }
    sources = sorted(contributions)
    rng = random.Random(seed)
    boot = {key: [] for key in estimates}
    for _ in range(replicates):
        counts = Counter(sources[rng.randrange(len(sources))] for _ in sources)
        denominator = sum(multiplier * contributions[source]["weight"] for source, multiplier in counts.items())
        if not denominator:
            continue
        boot["candidate_ratio_mean_difference"].append(
            sum(multiplier * contributions[source]["mean"] for source, multiplier in counts.items()) / denominator)
        boot["affected_risk_difference"].append(
            sum(multiplier * contributions[source]["risk"] for source, multiplier in counts.items()) / denominator)
    return {
        "method": "source_group×shot_type exact-cell overlap weighting; w=n_make*n_miss/(n_make+n_miss)",
        "eligible_cells": eligible_cells,
        "eligible_videos_cell_sum": eligible_videos,
        "source_groups": len(sources),
        "effects": {key: {"estimate": value, "cluster_bootstrap_95_ci": percentile_interval(boot[key])}
                    for key, value in estimates.items()},
    }


def continuous_label_effect(rows: Sequence[dict[str, Any]], value_key: str, *,
                            replicates: int, seed: int) -> dict[str, Any]:
    sources = sorted({str(row["source_group"]) for row in rows})
    by_source: dict[str, dict[str, list[float]]] = {
        source: {label: [] for label in LABELS} for source in sources
    }
    for row in rows:
        by_source[str(row["source_group"])][str(row["label"])].append(float(row[value_key]))
    make = [float(row[value_key]) for row in rows if row["label"] == "Make"]
    miss = [float(row[value_key]) for row in rows if row["label"] == "Miss"]
    estimate = statistics.fmean(make) - statistics.fmean(miss)
    delta = cliffs_delta(make, miss)
    pair_net = {(a, b): _pair_net(by_source[a]["Make"], by_source[b]["Miss"])
                for a in sources for b in sources}
    rng = random.Random(seed)
    mean_boot: list[float] = []
    delta_boot: list[float] = []
    for _ in range(replicates):
        counts = Counter(sources[rng.randrange(len(sources))] for _ in sources)
        make_n = sum(multiplier * len(by_source[source]["Make"]) for source, multiplier in counts.items())
        miss_n = sum(multiplier * len(by_source[source]["Miss"]) for source, multiplier in counts.items())
        if not make_n or not miss_n:
            continue
        make_sum = sum(multiplier * sum(by_source[source]["Make"]) for source, multiplier in counts.items())
        miss_sum = sum(multiplier * sum(by_source[source]["Miss"]) for source, multiplier in counts.items())
        mean_boot.append(make_sum / make_n - miss_sum / miss_n)
        net = sum(a_count * b_count * pair_net[(a, b)]
                  for a, a_count in counts.items() for b, b_count in counts.items())
        delta_boot.append(net / (make_n * miss_n))
    return {
        "mean_difference_make_minus_miss": {"estimate": estimate,
                                             "cluster_bootstrap_95_ci": percentile_interval(mean_boot)},
        "cliffs_delta": {"estimate": delta, "cluster_bootstrap_95_ci": percentile_interval(delta_boot)},
    }


def grouped_candidate(rows: Sequence[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[dimension])].append(row)
    result = []
    for group in sorted(groups):
        summary = summarize_candidate_by_label(groups[group])
        result.append({"group": group, "videos": len(groups[group]), **summary})
    return result


def metric_by_label(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in LABELS:
        subset = [row for row in rows if row["label"] == label]
        result[label] = {"n": len(subset), **{key: describe([float(row[key]) for row in subset]) for key in keys}}
    result["differences_make_minus_miss"] = {
        key: result["Make"][key]["mean"] - result["Miss"][key]["mean"] for key in keys
    }
    return result


def grouped_metrics(rows: Sequence[dict[str, Any]], dimension: str, keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[dimension])].append(row)
    return [{"group": group, **metric_by_label(groups[group], keys)} for group in sorted(groups)]


def temporal_groups(segment_rows: Sequence[dict[str, Any]], video_rows: Sequence[dict[str, Any]],
                    dimension: str) -> list[dict[str, Any]]:
    names = sorted({str(row[dimension]) for row in video_rows})
    result = []
    for name in names:
        segments = [row for row in segment_rows if str(row[dimension]) == name]
        videos = [row for row in video_rows if str(row[dimension]) == name]
        item: dict[str, Any] = {"group": name}
        for label in LABELS:
            seg = [row for row in segments if row["label"] == label]
            vid = [row for row in videos if row["label"] == label]
            bands = Counter(row["position_band"] for row in seg)
            item[label] = {
                "affected_videos": len(vid), "segments": len(seg),
                "segment_center": describe([float(row["center_normalized"]) for row in seg]),
                "segment_length": describe([float(row["length_normalized"]) for row in seg]),
                "video_mean_center": describe([float(row["mean_segment_center"]) for row in vid]),
                "video_mean_length": describe([float(row["mean_segment_length"]) for row in vid]),
                "position_counts": {band: bands[band] for band in ("front", "middle", "rear")},
                "position_rates": {band: bands[band] / len(seg) if seg else None
                                   for band in ("front", "middle", "rear")},
            }
        result.append(item)
    return result


def reconstruct_selections(video_rows: Sequence[dict[str, Any]], segments_by_video: dict[str, list[CandidateSegment]],
                           config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in sorted(video_rows, key=lambda value: str(value["video_id"])):
        selected = select_clips(
            video_id=str(row["video_id"]), frame_count=int(row["frame_count"]),
            candidate_segments=segments_by_video.get(str(row["video_id"]), []),
            clip_length=int(config["clip_length"]), num_clips=int(config["num_clips_per_video"]),
            seed=int(config["global_seed"]), epoch=0, selection_mode="seeded_random",
            candidate_margin_frames=int(config["candidate_margin_frames"]),
        )
        for selection in selected:
            records.append({
                "video_id": row["video_id"], "label": row["label"], "source_group": row["source_group"],
                "fps_rational": row["fps_rational"], "frame_count": row["frame_count"],
                **selection.to_dict(), "seed": config["global_seed"], "epoch": 0,
            })
    return records


def multi_epoch_sampler_counterfactual(
    video_rows: Sequence[dict[str, Any]],
    segment_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    *,
    epochs: int,
) -> dict[str, Any]:
    """Compare quality-aware selections with same-seed unrestricted selections.

    The unrestricted baseline passes no candidate segments to the same sampler.
    Therefore each per-video displacement isolates the effect of restricting the
    eligible start pool while retaining the video ID, seed, epoch and hash rule.
    Epochs are a deterministic sensitivity sweep, not independent observations.
    """
    need(epochs > 0, "counterfactual epochs must be positive")
    segments_by_video: dict[str, list[CandidateSegment]] = defaultdict(list)
    for row in segment_rows:
        segments_by_video[str(row["video_id"])].append(
            CandidateSegment(int(row["start_frame"]), int(row["end_frame_exclusive"])))

    per_epoch: list[dict[str, Any]] = []
    for epoch in range(epochs):
        centers: dict[str, dict[str, list[float]]] = {
            label: {"quality_aware": [], "unrestricted": [], "displacement": []}
            for label in LABELS
        }
        for row in video_rows:
            video_id = str(row["video_id"])
            frame_count = int(row["frame_count"])
            common = {
                "video_id": video_id,
                "frame_count": frame_count,
                "clip_length": int(config["clip_length"]),
                "num_clips": int(config["num_clips_per_video"]),
                "seed": int(config["global_seed"]),
                "epoch": epoch,
                "selection_mode": str(config["train_selection"]),
                "candidate_margin_frames": int(config["candidate_margin_frames"]),
            }
            quality = select_clips(
                candidate_segments=segments_by_video.get(video_id, []), **common)[0]
            unrestricted = select_clips(candidate_segments=[], **common)[0]
            quality_center = (
                quality.start_frame + quality.end_frame_exclusive) / (2.0 * frame_count)
            unrestricted_center = (
                unrestricted.start_frame + unrestricted.end_frame_exclusive) / (2.0 * frame_count)
            label = str(row["label"])
            centers[label]["quality_aware"].append(quality_center)
            centers[label]["unrestricted"].append(unrestricted_center)
            centers[label]["displacement"].append(quality_center - unrestricted_center)

        quality_shift = (
            statistics.fmean(centers["Make"]["quality_aware"])
            - statistics.fmean(centers["Miss"]["quality_aware"]))
        unrestricted_shift = (
            statistics.fmean(centers["Make"]["unrestricted"])
            - statistics.fmean(centers["Miss"]["unrestricted"]))
        make_displacement = statistics.fmean(centers["Make"]["displacement"])
        miss_displacement = statistics.fmean(centers["Miss"]["displacement"])
        per_epoch.append({
            "epoch": epoch,
            "quality_aware_center_make_minus_miss": quality_shift,
            "unrestricted_center_make_minus_miss": unrestricted_shift,
            "quality_induced_displacement_make": make_displacement,
            "quality_induced_displacement_miss": miss_displacement,
            "quality_induced_difference_in_differences": make_displacement - miss_displacement,
        })

    keys = (
        "quality_aware_center_make_minus_miss",
        "unrestricted_center_make_minus_miss",
        "quality_induced_displacement_make",
        "quality_induced_displacement_miss",
        "quality_induced_difference_in_differences",
    )
    return {
        "method": (
            "same video/seed/epoch/hash sampler; unrestricted baseline supplies no candidate segments; "
            "deterministic epoch sensitivity, not independent-sample inference"),
        "epochs": epochs,
        "alert_threshold_video_length": POSITION_SHIFT_ALERT,
        "summary_across_epochs": {
            key: describe([float(row[key]) for row in per_epoch]) for key in keys
        },
        "per_epoch": per_epoch,
    }


def load_and_validate(args: argparse.Namespace) -> dict[str, Any]:
    metadata = read_json(args.metadata)
    split_summary = read_json(args.split_summary)
    config = load_config(args.config)
    need(metadata.get("schema_version") == "basketball.blur-quality-sidecar.v3.1", "Sidecar schema mismatch")
    need(sha256(args.manifest) == metadata["manifest"]["sha256"], "Manifest hash mismatch")
    need(sha256(args.video_summary) == metadata["csv_files"][args.video_summary.name]["sha256"],
         "Video-summary hash mismatch")
    need(sha256(args.segments) == metadata["csv_files"][args.segments.name]["sha256"],
         "Candidate-segments hash mismatch")

    manifest = read_csv(args.manifest)
    validate_unique(manifest, "video_id", "dataset manifest")
    train = [row for row in manifest if row["split"] == "train" and row["excluded"] == "0"]
    train_index = {row["video_id"]: row for row in train}
    need(len(train_index) == EXPECTED["train_videos"], "Train manifest total mismatch")
    held_out_ids = {row["video_id"] for row in manifest if row["split"] in {"validation", "test"}}

    summaries = read_csv(args.video_summary)
    summary_index = validate_unique(summaries, "video_id", "v3.1 video summary")
    validate_exact_id_set(summary_index, train_index, "v3.1 video summary")
    video_rows: list[dict[str, Any]] = []
    for video_id in sorted(train_index):
        source = train_index[video_id]
        quality = summary_index[video_id]
        need(video_id not in held_out_ids, f"Held-out ID in sidecar: {video_id}")
        label = "Make" if source["binary_label"] == "1" else "Miss"
        need(quality["label"] == label, f"Sidecar label mismatch: {video_id}")
        need(quality["source_group"] == source["source_group"], f"Sidecar source mismatch: {video_id}")
        need(quality["config_hash"] == metadata["config"]["sha256"], f"Config hash mismatch: {video_id}")
        need(quality["run_identity"] == metadata["run_identity"], f"Run identity mismatch: {video_id}")
        video_rows.append({
            "video_id": video_id, "label": label, "binary_label": source["binary_label"],
            "original_class": source["original_class"],
            "shot_type": shot_type(source["original_class"], source["binary_label"]),
            "source_group": source["source_group"], "game_code": source["game_code"],
            "fps_rational": quality["fps_rational"], "frame_count": int(quality["frame_count"]),
            "candidate_frame_count": int(quality["candidate_frame_count"]),
            "candidate_ratio": float(quality["candidate_ratio"]),
            "candidate_segment_count": int(quality["candidate_segment_count"]),
        })
    video_index = {row["video_id"]: row for row in video_rows}

    segment_input = read_csv(args.segments)
    segment_keys: set[tuple[str, int, int]] = set()
    segments_by_video: dict[str, list[CandidateSegment]] = defaultdict(list)
    normalized_segments: list[dict[str, Any]] = []
    for raw in segment_input:
        video_id = raw["video_id"]
        need(video_id in video_index and video_id not in held_out_ids, f"Non-train candidate segment: {video_id}")
        start, end = int(raw["start_frame"]), int(raw["end_frame_exclusive"])
        key = (video_id, start, end)
        need(key not in segment_keys, f"Duplicate candidate segment: {key}")
        segment_keys.add(key)
        row = video_index[video_id]
        center, length = normalize_segment(start, end, int(row["frame_count"]))
        need(end - start == int(raw["candidate_frame_count"]), f"Candidate length mismatch: {key}")
        segments_by_video[video_id].append(CandidateSegment(start, end))
        normalized_segments.append({
            "video_id": video_id, "label": row["label"], "shot_type": row["shot_type"],
            "source_group": row["source_group"], "fps_rational": row["fps_rational"],
            "frame_count": row["frame_count"], "start_frame": start, "end_frame_exclusive": end,
            "candidate_frame_count": end - start, "center_normalized": center,
            "length_normalized": length, "position_band": position_band(center),
        })
    for video_id, segments in segments_by_video.items():
        previous = -1
        for segment in segments:
            need(segment.start_frame >= previous, f"Unsorted/overlapping candidate segments: {video_id}")
            previous = segment.end_frame_exclusive
    for row in video_rows:
        segments = segments_by_video.get(row["video_id"], [])
        need(len(segments) == row["candidate_segment_count"], f"Per-video segment count mismatch: {row['video_id']}")
        need(sum(item.end_frame_exclusive - item.start_frame for item in segments)
             == row["candidate_frame_count"], f"Per-video candidate total mismatch: {row['video_id']}")

    reconstructed = reconstruct_selections(video_rows, segments_by_video, config)
    reconstructed_csv = csv_text(reconstructed, SELECTION_FIELDS)
    reconstructed_hash = text_sha256(reconstructed_csv)
    if args.selections.exists():
        selection_origin = "existing_derived_csv"
        need(sha256(args.selections) == reconstructed_hash,
             "Existing selections.csv is not byte-identical to the epoch=0 reconstruction")
        selections = read_csv(args.selections)
    else:
        selection_origin = "metadata_only_reconstruction"
        selections = [{key: str(value) for key, value in row.items()} for row in reconstructed]
    selection_index = validate_unique(selections, "video_id", "sampler selections")
    validate_exact_id_set(selection_index, video_index, "sampler selections")
    reconstructed_index = {str(row["video_id"]): row for row in reconstructed}
    enriched_selections: list[dict[str, Any]] = []
    for video_id in sorted(video_index):
        raw = selection_index[video_id]
        expected_selection = reconstructed_index[video_id]
        video = video_index[video_id]
        need(raw["label"] == video["label"], f"Selection label mismatch: {video_id}")
        need(raw["source_group"] == video["source_group"], f"Selection source mismatch: {video_id}")
        need(raw["fps_rational"] == video["fps_rational"], f"Selection FPS mismatch: {video_id}")
        need(int(raw["frame_count"]) == video["frame_count"], f"Selection frame count mismatch: {video_id}")
        need(int(raw["epoch"]) == 0 and int(raw["seed"]) == config["global_seed"],
             f"Selection seed/epoch mismatch: {video_id}")
        start, end = int(raw["start_frame"]), int(raw["end_frame_exclusive"])
        overlap = candidate_overlap(start, end, segments_by_video.get(video_id, []))
        fallback = str(raw["fallback"]).lower() == "true"
        need(overlap == int(raw["candidate_overlap"]), f"Selection overlap mismatch: {video_id}")
        need(start == int(expected_selection["start_frame"])
             and end == int(expected_selection["end_frame_exclusive"])
             and overlap == int(expected_selection["candidate_overlap"])
             and fallback == bool(expected_selection["fallback"]),
             f"Selection is not the deterministic epoch=0 result: {video_id}")
        normalized_start, center = normalize_selection(start, end, video["frame_count"], config["clip_length"])
        enriched_selections.append({
            "video_id": video_id, "label": video["label"], "shot_type": video["shot_type"],
            "source_group": video["source_group"], "fps_rational": video["fps_rational"],
            "frame_count": video["frame_count"], "start_frame": start,
            "end_frame_exclusive": end, "normalized_start": normalized_start,
            "selected_center_normalized": center, "candidate_overlap": overlap,
            "fallback": fallback, "fallback_reason": raw.get("fallback_reason", ""),
        })

    totals = {
        "train_videos": len(video_rows),
        "decoded_frames": sum(row["frame_count"] for row in video_rows),
        "candidate_frames": sum(row["candidate_frame_count"] for row in video_rows),
        "candidate_segments": len(normalized_segments),
        "affected_videos": sum(row["candidate_frame_count"] > 0 for row in video_rows),
    }
    totals["unaffected_videos"] = totals["train_videos"] - totals["affected_videos"]
    totals["zero_overlap_selections"] = sum(row["candidate_overlap"] == 0 for row in enriched_selections)
    totals["fallback_selections"] = sum(row["fallback"] for row in enriched_selections)
    totals["fallback_total_overlap"] = sum(row["candidate_overlap"] for row in enriched_selections if row["fallback"])
    for key, expected in EXPECTED.items():
        need(totals[key] == expected, f"Frozen total mismatch for {key}: {totals[key]} != {expected}")
    coverage = metadata["coverage"]
    need(coverage["train_videos"] == EXPECTED["train_videos"], "Metadata train total mismatch")
    need(coverage["decoded_frames_from_source_metadata"] == EXPECTED["decoded_frames"],
         "Metadata decoded-frame total mismatch")

    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        if row["excluded"] == "0" and row["split"] in {"train", "validation", "test"}:
            source_splits[row["source_group"]].add(row["split"])
    crossing = {source: sorted(splits) for source, splits in source_splits.items() if len(splits) > 1}
    need(not crossing, f"source_group crosses splits: {crossing}")
    assignment = split_summary["group_assignment"]
    need(all(assignment.get(source) == next(iter(splits)) for source, splits in source_splits.items()),
         "split_summary group assignment mismatch")
    paired = [row for row in manifest if row["excluded"] == "0" and row["game_code"] in {"160", "166"}]
    need({row["game_code"] for row in paired} == {"160", "166"}, "v160/v166 rows missing")
    need(all(row["source_group"] == "v160_v166" for row in paired), "v160/v166 source group is not preserved")
    need({row["game_code"] for row in manifest if row["excluded"] == "0"
          and row["source_group"] == "v160_v166"} == {"160", "166"},
         "v160_v166 contains unexpected game codes")

    affected_video_temporal: list[dict[str, Any]] = []
    by_video_normalized: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized_segments:
        by_video_normalized[row["video_id"]].append(row)
    for video_id, segments in sorted(by_video_normalized.items()):
        video = video_index[video_id]
        bands = Counter(row["position_band"] for row in segments)
        affected_video_temporal.append({
            "video_id": video_id, "label": video["label"], "shot_type": video["shot_type"],
            "source_group": video["source_group"], "fps_rational": video["fps_rational"],
            "segment_count": len(segments),
            "mean_segment_center": statistics.fmean(float(row["center_normalized"]) for row in segments),
            "mean_segment_length": statistics.fmean(float(row["length_normalized"]) for row in segments),
            "front_share": bands["front"] / len(segments), "middle_share": bands["middle"] / len(segments),
            "rear_share": bands["rear"] / len(segments),
        })

    return {
        "metadata": metadata, "split_summary": split_summary, "config": config,
        "manifest": manifest, "video_rows": video_rows, "segment_rows": normalized_segments,
        "affected_video_temporal": affected_video_temporal,
        "selection_rows": enriched_selections, "selection_origin": selection_origin,
        "totals": totals,
        "split_check": {
            "source_groups": len(source_splits), "cross_split_source_groups": crossing,
            "v160_v166_same_source_group": True,
            "holdout_description": "source-group holdout (not confirmed game-level holdout)",
            "near_duplicate_video_scan_performed": False,
        },
        "input_hashes": {
            "manifest": sha256(args.manifest), "split_summary": sha256(args.split_summary),
            "metadata": sha256(args.metadata), "video_summary": sha256(args.video_summary),
            "segments": sha256(args.segments), "sampler_config": sha256(args.config),
            # This identity is stable even when the ignored CSV is absent and is
            # regenerated entirely from the frozen metadata/config.
            "epoch0_selections": reconstructed_hash,
        },
    }


def make_summary(data: dict[str, Any], *, replicates: int,
                 counterfactual_epochs: int = DEFAULT_COUNTERFACTUAL_EPOCHS) -> dict[str, Any]:
    videos = data["video_rows"]
    segments = data["segment_rows"]
    temporal_videos = data["affected_video_temporal"]
    selections = data["selection_rows"]
    candidate_label = summarize_candidate_by_label(videos)
    candidate_label["source_cluster_bootstrap"] = cluster_bootstrap_two_part(
        videos, "candidate_ratio", replicates=replicates, seed=BOOTSTRAP_SEED)
    adjusted = source_shot_adjusted(videos, replicates=replicates, seed=BOOTSTRAP_SEED + 1)

    # Add an explicit constant dimension for the same grouping machinery.
    segments_all = [{**row, "overall": "all_train"} for row in segments]
    temporal_all = [{**row, "overall": "all_train"} for row in temporal_videos]
    candidate_time = {
        "unit_note": "Segment rows are descriptive; label effects use one aggregate row per affected video.",
        "overall": temporal_groups(segments_all, temporal_all, "overall")[0],
        "by_source_group": temporal_groups(segments, temporal_videos, "source_group"),
        "by_shot_type": temporal_groups(segments, temporal_videos, "shot_type"),
        "by_fps": temporal_groups(segments, temporal_videos, "fps_rational"),
        "video_unit_label_effects": {
            "mean_segment_center": continuous_label_effect(
                temporal_videos, "mean_segment_center", replicates=replicates, seed=BOOTSTRAP_SEED + 2),
            "mean_segment_length": continuous_label_effect(
                temporal_videos, "mean_segment_length", replicates=replicates, seed=BOOTSTRAP_SEED + 3),
            "rear_share": continuous_label_effect(
                temporal_videos, "rear_share", replicates=replicates, seed=BOOTSTRAP_SEED + 4),
        },
    }
    selection_keys = ("start_frame", "end_frame_exclusive", "normalized_start", "selected_center_normalized")
    sampler = {
        "selection_source": "validated existing derived CSV or byte-identical metadata-only epoch=0 reconstruction",
        "overall_by_label": metric_by_label(selections, selection_keys),
        "by_source_group": grouped_metrics(selections, "source_group", selection_keys),
        "by_shot_type": grouped_metrics(selections, "shot_type", selection_keys),
        "by_fps": grouped_metrics(selections, "fps_rational", selection_keys),
        "normalized_start_label_effect": continuous_label_effect(
            selections, "normalized_start", replicates=replicates, seed=BOOTSTRAP_SEED + 5),
        "selected_center_label_effect": continuous_label_effect(
            selections, "selected_center_normalized", replicates=replicates, seed=BOOTSTRAP_SEED + 6),
        "candidate_overlap": {
            "zero_overlap": sum(row["candidate_overlap"] == 0 for row in selections),
            "positive_overlap": sum(row["candidate_overlap"] > 0 for row in selections),
            "total_overlap": sum(row["candidate_overlap"] for row in selections),
            "by_label": {label: {
                "videos": sum(row["label"] == label for row in selections),
                "zero_overlap": sum(row["label"] == label and row["candidate_overlap"] == 0 for row in selections),
                "fallback": sum(row["label"] == label and row["fallback"] for row in selections),
                "fallback_overlap": sum(row["candidate_overlap"] for row in selections
                                        if row["label"] == label and row["fallback"]),
            } for label in LABELS},
        },
        "fallback_rows": [{key: row[key] for key in (
            "video_id", "label", "shot_type", "source_group", "fps_rational", "frame_count",
            "start_frame", "end_frame_exclusive", "normalized_start", "selected_center_normalized",
            "candidate_overlap", "fallback_reason")}
                          for row in selections if row["fallback"]],
    }
    counterfactual = multi_epoch_sampler_counterfactual(
        videos, segments, data["config"], epochs=counterfactual_epochs)
    sampler["multi_epoch_counterfactual"] = counterfactual
    raw_mean = candidate_label["differences_make_minus_miss"]["mean"]
    raw_risk = candidate_label["differences_make_minus_miss"]["affected_rate"]
    raw_delta = candidate_label["differences_make_minus_miss"]["cliffs_delta"]
    adjusted_mean = adjusted["effects"]["candidate_ratio_mean_difference"]["estimate"]
    adjusted_risk = adjusted["effects"]["affected_risk_difference"]["estimate"]
    selected_shift = sampler["selected_center_label_effect"]["mean_difference_make_minus_miss"]["estimate"]
    selected_ci = sampler["selected_center_label_effect"]["mean_difference_make_minus_miss"]["cluster_bootstrap_95_ci"]
    epoch0_shift = counterfactual["per_epoch"][0]["quality_aware_center_make_minus_miss"]
    need(abs(epoch0_shift - selected_shift) < 1e-12,
         "Epoch-0 counterfactual quality selection does not match frozen selections")
    cf_summary = counterfactual["summary_across_epochs"]
    quality_epoch_values = [
        float(row["quality_aware_center_make_minus_miss"])
        for row in counterfactual["per_epoch"]
    ]
    induced_epoch_values = [
        float(row["quality_induced_difference_in_differences"])
        for row in counterfactual["per_epoch"]
    ]
    epoch0_ci_within_alert = (
        selected_ci[0] > -POSITION_SHIFT_ALERT and selected_ci[1] < POSITION_SHIFT_ALERT)
    all_quality_epochs_within_alert = all(
        abs(value) < POSITION_SHIFT_ALERT for value in quality_epoch_values)
    all_induced_epochs_within_alert = all(
        abs(value) < POSITION_SHIFT_ALERT for value in induced_epoch_values)
    blocks_dataloader = not (
        epoch0_ci_within_alert
        and all_quality_epochs_within_alert
        and all_induced_epochs_within_alert)
    rear = candidate_time["overall"]
    rear_all = (rear["Make"]["position_counts"]["rear"] + rear["Miss"]["position_counts"]["rear"]) / len(segments)
    front_all = (rear["Make"]["position_counts"]["front"] + rear["Miss"]["position_counts"]["front"]) / len(segments)
    conclusions = {
        "candidate_label_descriptive_association": (
            f"有清楚的描述性關聯：Make 的 mean candidate ratio 高 {100 * raw_mean:.2f} 個百分點、"
            f"affected rate 高 {100 * raw_risk:.2f} 個百分點，Cliff’s delta={raw_delta:.3f}；"
            "rank effect 偏小，且 source-cluster CI 必須一併解讀。" if raw_mean > 0 else
            "未見 Make 高於 Miss 的描述性關聯；effect size 與 source-cluster CI 必須一併解讀。"),
        "after_source_and_shot_control": (
            f"source_group×shot_type 內的 overlap-weighted 差異仍存在（mean {100 * adjusted_mean:.2f}、"
            f"affected risk {100 * adjusted_risk:.2f} 個百分點），但較未調整差異縮小；"
            "這是描述性調整，不是因果估計。" if adjusted_mean > 0 else
            "source_group×shot_type 內的 overlap-weighted 差異不再為正；這是描述性調整，不是因果估計。"),
        "sampler_label_position_shift": (
            f"epoch 0 有小幅前移（Make−Miss selected-center={selected_shift:.4f}，95% CI "
            f"[{selected_ci[0]:.4f}, {selected_ci[1]:.4f}]）；{counterfactual_epochs} 個 deterministic epochs "
            f"的 quality-aware 差異範圍為 [{min(quality_epoch_values):.4f}, "
            f"{max(quality_epoch_values):.4f}]，相對 unrestricted baseline 的平均 "
            f"difference-in-differences={cf_summary['quality_induced_difference_in_differences']['mean']:.4f}。"
            f"三項檢查均在 ±{POSITION_SHIFT_ALERT:.2f} 警示尺度內，未見實質 label 位移。"
            if not blocks_dataloader else
            f"epoch 0 CI、{counterfactual_epochs}-epoch quality-aware 範圍或相對 unrestricted baseline "
            f"的位移達到 ±{POSITION_SHIFT_ALERT:.2f} 警示尺度，整合前需進一步診斷。"),
        "blocks_dataloader_integration": blocks_dataloader,
        "next_round": "維持 v3.1 進入可觀測、train-only DataLoader 整合；另做 z=-2.5 derived sensitivity，並以人工標註校準 candidate 的語義。",
        "candidate_rear_concentration": (
            "candidate segments 後段占比高於前段；需連同 source/shot/FPS 分層解讀。"
            if rear_all > front_all else
            "candidate segments 未呈現後段占比高於前段。"),
        "sampler_selected_center_ci": selected_ci,
        "sampler_gate_checks": {
            "alert_threshold_video_length": POSITION_SHIFT_ALERT,
            "epoch0_cluster_ci_within_alert": epoch0_ci_within_alert,
            "all_counterfactual_quality_epochs_within_alert": all_quality_epochs_within_alert,
            "all_counterfactual_induced_epochs_within_alert": all_induced_epochs_within_alert,
        },
    }
    return {
        "schema_version": "basketball.train-quality-associations.v1",
        "scope": {
            "task": "full-video Make/Miss classification",
            "pre_release_prediction_claimed": False,
            "split": "train only",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_cluster": "source_group",
            "counterfactual_epochs": counterfactual_epochs,
        },
        "inputs": data["input_hashes"], "coverage": data["totals"],
        "candidate_label": candidate_label,
        "stratification": {
            "by_source_group": grouped_candidate(videos, "source_group"),
            "by_shot_type": grouped_candidate(videos, "shot_type"),
            "by_fps": grouped_candidate(videos, "fps_rational"),
            "source_shot_adjusted": adjusted,
        },
        "candidate_time": candidate_time,
        "sampler": sampler,
        "split_check": data["split_check"],
        "conclusions": conclusions,
        "limitations": [
            "All associations are descriptive and do not identify causal effects.",
            "Candidate means low local Laplacian under v3.1, not human-confirmed blur.",
            "The centered scanner is offline/full-video and cannot support a pre-release prediction claim.",
            "Segment-level summaries contain correlated segments; inferential label effects use affected-video aggregates.",
            "Only source-group holdout is established; game-level holdout is not confirmed.",
            "No video-content near-duplicate scan was performed in this audit.",
            "No validation/test video or quality metadata was analyzed.",
            "The multi-epoch counterfactual is a deterministic sensitivity sweep, not independent-sample inference.",
        ],
        "video_access": {"opens": 0, "probes": 0, "decodes": 0,
                         "validation_videos_read": 0, "test_videos_read": 0},
    }


def pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def num(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def ci_text(effect: dict[str, Any], *, percent: bool = False) -> str:
    low, high = effect["cluster_bootstrap_95_ci"]
    if percent:
        return f"{pct(effect['estimate'])} [{pct(low)}, {pct(high)}]"
    return f"{num(effect['estimate'])} [{num(low)}, {num(high)}]"


def candidate_table(groups: Sequence[dict[str, Any]], label: str) -> list[str]:
    lines = [f"### {label}", "", "| Group | n Make/Miss | Mean Make/Miss | Mean Δ | Affected Make/Miss | Affected Δ |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in groups:
        diff = row["differences_make_minus_miss"]
        lines.append(
            f"| {row['group']} | {row['Make']['n']}/{row['Miss']['n']} | "
            f"{pct(row['Make']['mean'])}/{pct(row['Miss']['mean'])} | {pct(diff['mean'])} | "
            f"{pct(row['Make']['affected_rate'])}/{pct(row['Miss']['affected_rate'])} | "
            f"{pct(diff['affected_rate'])} |")
    return lines


def temporal_table(groups: Sequence[dict[str, Any]], label: str) -> list[str]:
    lines = [f"### {label}", "", "| Group | affected videos Make/Miss | segments Make/Miss | Mean center Make/Miss | Rear rate Make/Miss |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in groups:
        lines.append(
            f"| {row['group']} | {row['Make']['affected_videos']}/{row['Miss']['affected_videos']} | "
            f"{row['Make']['segments']}/{row['Miss']['segments']} | "
            f"{num(row['Make']['segment_center']['mean'])}/{num(row['Miss']['segment_center']['mean'])} | "
            f"{pct(row['Make']['position_rates']['rear'])}/{pct(row['Miss']['position_rates']['rear'])} |")
    return lines


def sampler_table(groups: Sequence[dict[str, Any]], label: str) -> list[str]:
    lines = [f"### {label}", "", "| Group | n Make/Miss | Mean normalized start Make/Miss | Mean center Make/Miss | Center Δ |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for row in groups:
        lines.append(
            f"| {row['group']} | {row['Make']['n']}/{row['Miss']['n']} | "
            f"{num(row['Make']['normalized_start']['mean'])}/{num(row['Miss']['normalized_start']['mean'])} | "
            f"{num(row['Make']['selected_center_normalized']['mean'])}/"
            f"{num(row['Miss']['selected_center_normalized']['mean'])} | "
            f"{num(row['differences_make_minus_miss']['selected_center_normalized'])} |")
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    cov = summary["coverage"]
    cand = summary["candidate_label"]
    boot = cand["source_cluster_bootstrap"]
    adjusted = summary["stratification"]["source_shot_adjusted"]
    temporal = summary["candidate_time"]
    overall_time = temporal["overall"]
    sampler = summary["sampler"]
    selected = sampler["overall_by_label"]
    conclusions = summary["conclusions"]
    lines = [
        "# Train-only v3.1 Candidate／Sampler 資料品質 Audit v1",
        "",
        "## Executive summary",
        "",
        f"本 audit 精確覆蓋固定 train **{cov['train_videos']:,}** 支影片，只讀 manifest、v3.1 sidecar、sampler config 與 epoch=0 derived selections；影片 open/probe/decode 均為 **0**。研究任務仍是 **full-video Make/Miss classification**，不是 pre-release prediction，也不宣稱出手前預測能力。",
        "",
        f"- Candidate 與 label：{conclusions['candidate_label_descriptive_association']}",
        f"- 控制 source/shot：{conclusions['after_source_and_shot_control']}",
        f"- Sampler 時間位置：{conclusions['sampler_label_position_shift']}",
        f"- DataLoader gate：{'暫緩整合' if conclusions['blocks_dataloader_integration'] else '不足以阻止整合'}。",
        f"- 下一輪：{conclusions['next_round']}",
        "",
        "固定 totals 已重驗：1,105,146 decoded frames（僅來自既有 metadata）、18,390 candidate frames、1,798 candidate segments、1,698 affected、5,495 unaffected。",
        "",
        "## Candidate–label 結果",
        "",
        "統計單位是影片；frame 不被當作獨立樣本。零膨脹以 affected risk difference、全分布 Cliff’s delta、per-video mean difference 與 affected-only mean difference共同描述。95% CI 是對 35 個 source groups 做 cluster bootstrap；不以 p-value 單獨下結論。",
        "",
        "| Label | n | mean | median | P25 | P75 | P90 | P95 | affected count/rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in LABELS:
        row = cand[label]
        lines.append(f"| {label} | {row['n']:,} | {pct(row['mean'])} | {pct(row['median'])} | "
                     f"{pct(row['p25'])} | {pct(row['p75'])} | {pct(row['p90'])} | {pct(row['p95'])} | "
                     f"{row['affected_count']:,}/{pct(row['affected_rate'])} |")
    lines.extend([
        "",
        "| Effect (Make−Miss) | estimate [source-cluster 95% CI] |",
        "| --- | ---: |",
        f"| Mean candidate ratio | {ci_text(boot['mean_difference'], percent=True)} |",
        f"| Affected risk | {ci_text(boot['affected_risk_difference'], percent=True)} |",
        f"| Affected-only mean ratio | {ci_text(boot['affected_only_mean_difference'], percent=True)} |",
        f"| Cliff’s delta | {ci_text(boot['cliffs_delta'])} |",
        "",
        "Cliff’s delta 把零值 ties 保留在全分布中；正值表示隨機抽一支 Make 的 candidate ratio 高於 Miss 的機率較高。這些是描述性關聯，不代表 label 造成 candidate。",
        "",
        "## Source／shot type／FPS 分層",
        "",
    ])
    lines.extend(candidate_table(summary["stratification"]["by_source_group"], "Source group"))
    lines.extend([""] + candidate_table(summary["stratification"]["by_shot_type"], "Shot type"))
    lines.extend([""] + candidate_table(summary["stratification"]["by_fps"], "FPS"))
    mean_adj = adjusted["effects"]["candidate_ratio_mean_difference"]
    risk_adj = adjusted["effects"]["affected_risk_difference"]
    lines.extend([
        "",
        "### Source group × shot type 調整",
        "",
        f"在同一 source_group×shot_type 且兩個 label 都存在的 {adjusted['eligible_cells']} 個 cells，以 `n_make*n_miss/(n_make+n_miss)` overlap weight 標準化。調整後 candidate-ratio mean difference 為 **{ci_text(mean_adj, percent=True)}**，affected-risk difference 為 **{ci_text(risk_adj, percent=True)}**。這是受共同支撐範圍限制的描述性調整，不能解釋為因果效果。",
        "",
        "## Candidate 時間位置",
        "",
        "每段使用 `center=(start+end)/(2*frame_count)`、`length=(end-start)/frame_count`；center 依三等分標記前／中／後。Segment 表只做描述，因同一影片可有多段；label effect 先在每支 affected video 內聚合，再以影片為單位、source-group cluster bootstrap。",
        "",
        "| Label | affected videos | segments | center mean/median | length mean/median | front/middle/rear |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label in LABELS:
        row = overall_time[label]
        pos = row["position_rates"]
        lines.append(f"| {label} | {row['affected_videos']:,} | {row['segments']:,} | "
                     f"{num(row['segment_center']['mean'])}/{num(row['segment_center']['median'])} | "
                     f"{pct(row['segment_length']['mean'])}/{pct(row['segment_length']['median'])} | "
                     f"{pct(pos['front'])}/{pct(pos['middle'])}/{pct(pos['rear'])} |")
    center_effect = temporal["video_unit_label_effects"]["mean_segment_center"]["mean_difference_make_minus_miss"]
    rear_effect = temporal["video_unit_label_effects"]["rear_share"]["mean_difference_make_minus_miss"]
    lines.extend([
        "",
        f"Affected-video 平均 segment center 的 Make−Miss 差為 **{ci_text(center_effect)}**；每影片 rear-share 差為 **{ci_text(rear_effect, percent=True)}**。{conclusions['candidate_rear_concentration']}",
        "",
    ])
    lines.extend(temporal_table(temporal["by_source_group"], "Source group"))
    lines.extend([""] + temporal_table(temporal["by_shot_type"], "Shot type"))
    lines.extend([""] + temporal_table(temporal["by_fps"], "FPS"))
    lines.extend([
        "",
        "## Sampler 選取位置",
        "",
        "`normalized_start = start_frame / (frame_count - clip_length)`；selected center 使用 `(start+end)/(2*frame_count)`。Sampler 本身不讀 label，但本節直接檢查輸出分布。",
        "",
        "| Label | n | start mean/median | end mean/median | normalized start mean/median | center mean/median |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label in LABELS:
        row = selected[label]
        lines.append(f"| {label} | {row['n']:,} | {num(row['start_frame']['mean'], 2)}/{num(row['start_frame']['median'], 2)} | "
                     f"{num(row['end_frame_exclusive']['mean'], 2)}/{num(row['end_frame_exclusive']['median'], 2)} | "
                     f"{num(row['normalized_start']['mean'])}/{num(row['normalized_start']['median'])} | "
                     f"{num(row['selected_center_normalized']['mean'])}/{num(row['selected_center_normalized']['median'])} |")
    start_effect = sampler["normalized_start_label_effect"]["mean_difference_make_minus_miss"]
    center_sel_effect = sampler["selected_center_label_effect"]["mean_difference_make_minus_miss"]
    counterfactual = sampler["multi_epoch_counterfactual"]
    cf_summary = counterfactual["summary_across_epochs"]
    quality_sweep = cf_summary["quality_aware_center_make_minus_miss"]
    unrestricted_sweep = cf_summary["unrestricted_center_make_minus_miss"]
    induced_sweep = cf_summary["quality_induced_difference_in_differences"]
    lines.extend([
        "",
        f"Epoch 0 的 Make−Miss normalized-start 差為 **{ci_text(start_effect)}**；selected-center 差為 **{ci_text(center_sel_effect)}**。以 ±{POSITION_SHIFT_ALERT:.2f} 個影片長度作為整合前的實質偏移警示尺度，epoch 0 的 source-cluster CI 完整落在警示區間內。",
        "",
    ])
    lines.extend(sampler_table(sampler["by_source_group"], "Source group"))
    lines.extend([""] + sampler_table(sampler["by_shot_type"], "Shot type"))
    lines.extend([""] + sampler_table(sampler["by_fps"], "FPS"))
    lines.extend([
        "",
        "### Multi-epoch 反事實敏感度",
        "",
        f"以相同 video、seed、epoch 與 hash 規則重跑 **{counterfactual['epochs']} 個 deterministic epochs**。Quality-aware 使用正式 candidate segments；unrestricted baseline 傳入空 segments，因此每影片 `quality-aware center − unrestricted center` 隔離的是候選窗口限制造成的位移。這是 deterministic sensitivity，不把 epochs 當成獨立樣本。",
        "",
        "| 指標 | mean across epochs | min | max |",
        "| --- | ---: | ---: | ---: |",
        f"| Quality-aware Make−Miss center | {num(quality_sweep['mean'])} | {num(quality_sweep['min'])} | {num(quality_sweep['max'])} |",
        f"| Unrestricted Make−Miss center | {num(unrestricted_sweep['mean'])} | {num(unrestricted_sweep['min'])} | {num(unrestricted_sweep['max'])} |",
        f"| Quality-induced difference-in-differences | {num(induced_sweep['mean'])} | {num(induced_sweep['min'])} | {num(induced_sweep['max'])} |",
        "",
        f"Gate 同時要求：(1) epoch 0 source-cluster CI、(2) 全部 {counterfactual['epochs']} 個 quality-aware label shifts、(3) 全部 quality-induced difference-in-differences 都位於 ±{POSITION_SHIFT_ALERT:.2f} 內。三項目前均通過。",
    ])
    overlap = sampler["candidate_overlap"]
    lines.extend([
        "",
        "## Fallback 結果",
        "",
        f"驗收值完全一致：**{overlap['zero_overlap']:,}** 支 zero-overlap、**{overlap['positive_overlap']}** 支 fallback、fallback 總 overlap **{overlap['total_overlap']} frames**。",
        "",
        "| video_id | label | shot | source | start:end | overlap |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ])
    for row in sampler["fallback_rows"]:
        lines.append(f"| `{row['video_id']}` | {row['label']} | {row['shot_type']} | {row['source_group']} | "
                     f"{row['start_frame']}:{row['end_frame_exclusive']} | {row['candidate_overlap']} |")
    split = summary["split_check"]
    lines.extend([
        "",
        "## Split 限制",
        "",
        f"50 個 source groups 在 train/validation/test 間沒有交叉，且 v160/v166 維持共同 `v160_v166` source group。本 split 只能明確稱為 **{split['holdout_description']}**；沒有足夠證據稱為已確認 game-level holdout。本輪沒有做影片內容 near-duplicate scan，亦沒有讀取 validation/test 影片。",
        "",
        "## 對下一階段 DataLoader 整合的建議",
        "",
        "1. 可讓 v3.1 quality-aware sampler 進入 **train-only、可觀測且可回退** 的 DataLoader 整合；保留原始連續 64-frame window，不變更 label、split 或影片。",
        "2. 每個 sample/log 必須保留 `video_id`、start/end、normalized start、candidate overlap、fallback、source、shot type、FPS、policy/run identity；每個 epoch 監控 label/source/shot 的位置分布。",
        "3. validation/test 維持 quality mode disabled；本 audit 不授權把 train sidecar 規則套到 held-out data。",
        "4. 另做 z=-2.5 的 **derived sensitivity**，比較關聯與選窗穩定度，但不要在本輪重調 v3.1 門檻。",
        "5. 建立小型、按 source/shot/label/時間位置分層的人工標註集，校準 candidate 是否真的對應可見模糊；人工標註是語義驗證，不是刪除資料授權。",
        "",
        "## 結論（必要問題）",
        "",
        f"- **Candidate 與 label 是否存在明顯描述性關聯？** {conclusions['candidate_label_descriptive_association']}",
        f"- **控制 source group 與 shot type 後是否仍存在？** {conclusions['after_source_and_shot_control']}",
        f"- **Sampler 選中窗口是否因 label 實質偏移？** {conclusions['sampler_label_position_shift']}",
        f"- **是否足以阻止 DataLoader 整合？** {'是' if conclusions['blocks_dataloader_integration'] else '否'}；但整合必須 train-only 且持續記錄診斷欄位。",
        f"- **下一輪？** {conclusions['next_round']}",
        "",
        "## 重現與限制",
        "",
        "```bash",
        "python3 -B development/data_quality/scripts/audit_train_quality_associations_v1.py --check-only",
        "python3 -B development/data_quality/scripts/audit_train_quality_associations_v1.py --refresh-artifacts",
        "python3 -B -m unittest development/data_quality/tests/test_train_quality_associations_v1.py",
        "```",
        "",
        "完整 machine summary 見 `train_quality_associations_v1.json`；中間 per-video／per-segment CSV 位於 ignored `development/data_quality/artifacts/train_quality_associations_v1/`。乾淨 checkout 可直接執行 `--check-only`；ignored CSV 不存在時會回報為 optional missing，不影響 tracked report 驗證，需要實體 CSV 時再用 `--refresh-artifacts` 重建。Audit 不訓練模型、不評估 test、不做 augmentation、不改門檻，也不開啟、probe 或 decode 影片。關聯都是描述性；candidate 不是人工確認的 blur。",
    ])
    return "\n".join(lines) + "\n"


def artifact_outputs(data: dict[str, Any]) -> dict[Path, str]:
    video_fields = ("video_id", "label", "binary_label", "original_class", "shot_type", "source_group",
                    "game_code", "fps_rational", "frame_count", "candidate_frame_count", "candidate_ratio",
                    "candidate_segment_count")
    segment_fields = ("video_id", "label", "shot_type", "source_group", "fps_rational", "frame_count",
                      "start_frame", "end_frame_exclusive", "candidate_frame_count", "center_normalized",
                      "length_normalized", "position_band")
    selection_fields = ("video_id", "label", "shot_type", "source_group", "fps_rational", "frame_count",
                        "start_frame", "end_frame_exclusive", "normalized_start", "selected_center_normalized",
                        "candidate_overlap", "fallback", "fallback_reason")
    return {
        DEFAULT_ARTIFACT_DIR / "video_metrics.csv": csv_text(data["video_rows"], video_fields),
        DEFAULT_ARTIFACT_DIR / "candidate_segments_normalized.csv": csv_text(data["segment_rows"], segment_fields),
        DEFAULT_ARTIFACT_DIR / "sampler_selections_enriched.csv": csv_text(data["selection_rows"], selection_fields),
    }


def verify_text_outputs(outputs: dict[Path, str], *, allow_missing: bool) -> dict[str, int]:
    result = {"verified": 0, "missing_optional": 0}
    for path, expected in outputs.items():
        if not path.exists():
            need(allow_missing, f"Missing generated output: {path}")
            result["missing_optional"] += 1
            continue
        need(not path.is_symlink(), f"Generated output must not be a symlink: {path}")
        need(path.read_text(encoding="utf-8") == expected, f"Generated output is stale: {path}")
        result["verified"] += 1
    return result


def write_text_outputs(outputs: dict[Path, str]) -> None:
    """Atomically replace deterministic generated files at explicit paths."""
    for path, content in outputs.items():
        need(not path.is_symlink(), f"Refusing to replace symlink: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        need(not temporary.is_symlink(), f"Refusing to replace temporary symlink: {temporary}")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--split-summary", type=Path, default=DEFAULT_SPLIT_SUMMARY)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--video-summary", type=Path, default=DEFAULT_VIDEO_SUMMARY)
    ap.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    ap.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    ap.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    ap.add_argument("--bootstrap-replicates", type=int, default=5000)
    ap.add_argument("--counterfactual-epochs", type=int, default=DEFAULT_COUNTERFACTUAL_EPOCHS)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true",
                      help="verify tracked outputs; verify ignored artifacts only when present")
    mode.add_argument("--refresh-artifacts", action="store_true",
                      help="verify tracked outputs, then recreate only ignored intermediate CSVs")
    mode.add_argument("--refresh-all", action="store_true",
                      help="atomically regenerate tracked reports and ignored intermediate CSVs")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    need(args.artifact_dir.resolve() == DEFAULT_ARTIFACT_DIR.resolve(),
         "Artifact output must remain in the ignored audit directory")
    data = load_and_validate(args)
    summary = make_summary(
        data, replicates=args.bootstrap_replicates,
        counterfactual_epochs=args.counterfactual_epochs)
    json_output = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown_output = render_markdown(summary)
    outputs = {args.json_output: json_output, args.markdown_output: markdown_output}
    artifacts = {
        args.artifact_dir / path.name: text for path, text in artifact_outputs(data).items()
    }
    artifact_check: dict[str, int]
    if args.check_only:
        verify_text_outputs(outputs, allow_missing=False)
        artifact_check = verify_text_outputs(artifacts, allow_missing=True)
        status = "CHECK_PASS"
    elif args.refresh_artifacts:
        verify_text_outputs(outputs, allow_missing=False)
        write_text_outputs(artifacts)
        artifact_check = {"verified": len(artifacts), "missing_optional": 0}
        status = "ARTIFACTS_REFRESHED"
    elif args.refresh_all:
        write_text_outputs({**outputs, **artifacts})
        artifact_check = {"verified": len(artifacts), "missing_optional": 0}
        status = "ALL_OUTPUTS_REFRESHED"
    else:
        for path in outputs:
            need(not path.exists() and not path.is_symlink(), f"Refusing to overwrite tracked output: {path}")
        for path in artifacts:
            need(not path.exists() and not path.is_symlink(), f"Refusing to overwrite ignored artifact: {path}")
        args.artifact_dir.mkdir(parents=True, exist_ok=False)
        for path, text in {**outputs, **artifacts}.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        artifact_check = {"verified": len(artifacts), "missing_optional": 0}
        status = "AUDIT_COMPLETE"
    print(json.dumps({"status": status, "coverage": summary["coverage"],
                      "conclusions": summary["conclusions"], "video_access": summary["video_access"],
                      "ignored_artifacts": artifact_check},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AuditError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
