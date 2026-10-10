#!/usr/bin/env python3
"""Run the train-only quality-aware sampler using metadata sidecars."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "development/data_quality"))
from samplers.quality_aware_clip_sampler import (  # noqa: E402
    POLICY_VERSION, CandidateSegment, SamplerError, candidate_overlap, select_clips,
)

DEFAULT_CONFIG = ROOT / "development/data_quality/configs/quality_aware_sampler_v1.yaml"
DEFAULT_MANIFEST = ROOT / "data/processed/dataset_manifest.csv"
DEFAULT_SEGMENTS = ROOT / "development/data_quality/sidecars/blur_candidate_segments_v3_1.csv"
DEFAULT_SUMMARY = ROOT / "development/data_quality/sidecars/blur_video_summary_v3_1.csv"
DEFAULT_OUTPUT = ROOT / "development/data_quality/artifacts/quality_aware_sampler_v1"
METADATA_NAME = "blur_quality_sidecar_v3_1.json"
SELECTION_FIELDS = [
    "video_id", "label", "source_group", "fps_rational", "frame_count", "clip_index",
    "start_frame", "end_frame_exclusive", "candidate_overlap", "fallback",
    "fallback_reason", "policy_version", "selection_mode", "seed", "epoch",
]


class AuditError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    need(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        need(":" in line, f"Invalid config line: {raw}")
        key, value = (part.strip() for part in line.split(":", 1))
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                parsed = value
        cfg[key] = parsed
    expected = {
        "policy_version": POLICY_VERSION, "quality_source": "blur_screening_v3_1",
        "candidate_margin_frames": 0, "primary_rule": "zero_candidate_overlap",
        "fallback_rule": "minimum_candidate_overlap", "train_selection": "seeded_random",
        "validation_quality_mode": "disabled",
    }
    for key, value in expected.items():
        need(cfg.get(key) == value, f"Invalid frozen config value: {key}")
    need(int(cfg.get("clip_length", 0)) > 0, "clip_length must be positive")
    need(int(cfg.get("num_clips_per_video", 0)) > 0, "num_clips_per_video must be positive")
    return cfg


def load_sidecars(
    manifest_path: Path, segments_path: Path, summary_path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, list[CandidateSegment]], dict[str, Any]]:
    metadata_path = segments_path.parent / METADATA_NAME
    metadata = read_json(metadata_path)
    need(metadata.get("schema_version") == "basketball.blur-quality-sidecar.v3.1", "Sidecar schema mismatch")
    csv_meta = metadata["csv_files"]
    need(sha256(segments_path) == csv_meta[segments_path.name]["sha256"], "Segment CSV hash mismatch")
    need(sha256(summary_path) == csv_meta[summary_path.name]["sha256"], "Video-summary CSV hash mismatch")
    source_manifest = ROOT / metadata["manifest"]["path"]
    need(source_manifest.resolve() == DEFAULT_MANIFEST.resolve(), "Sidecar source manifest path mismatch")
    need(sha256(source_manifest) == metadata["manifest"]["sha256"], "Sidecar source manifest hash mismatch")

    source_rows = read_csv(source_manifest)
    source_train = {row["video_id"]: row for row in source_rows
                    if row["split"] == "train" and row["excluded"] == "0"}
    held_out = {row["video_id"] for row in source_rows if row["split"] in {"validation", "test"}}
    requested = read_csv(manifest_path)
    requested_ids: list[str] = []
    for row in requested:
        video_id = row.get("video_id", "")
        split = row.get("split", "train")
        if manifest_path.resolve() == source_manifest.resolve() and (split != "train" or row.get("excluded") != "0"):
            continue
        need(video_id in source_train and video_id not in held_out, f"Non-train/unknown ID rejected: {video_id}")
        need(split == "train", f"Non-train manifest row rejected: {video_id}")
        requested_ids.append(video_id)
    need(len(requested_ids) == len(set(requested_ids)), "Duplicate manifest video_id")

    summaries = {row["video_id"]: row for row in read_csv(summary_path)}
    need(len(summaries) == int(metadata["coverage"]["train_videos"]), "Video-summary coverage mismatch")
    requested_summaries: dict[str, dict[str, str]] = {}
    for video_id in requested_ids:
        need(video_id in summaries, f"Missing video summary: {video_id}")
        row = summaries[video_id]
        need(row["config_hash"] == metadata["config"]["sha256"], f"Config identity mismatch: {video_id}")
        need(row["run_identity"] == metadata["run_identity"], f"Run identity mismatch: {video_id}")
        requested_summaries[video_id] = row

    segments: dict[str, list[CandidateSegment]] = defaultdict(list)
    segment_rows = read_csv(segments_path)
    need(len(segment_rows) == int(metadata["coverage"]["candidate_segments"]), "Segment coverage mismatch")
    for row in segment_rows:
        video_id = row["video_id"]
        need(video_id in source_train and video_id not in held_out, f"Non-train segment ID rejected: {video_id}")
        need(row["config_hash"] == metadata["config"]["sha256"], f"Segment config mismatch: {video_id}")
        need(row["run_identity"] == metadata["run_identity"], f"Segment run identity mismatch: {video_id}")
        segment = CandidateSegment(int(row["start_frame"]), int(row["end_frame_exclusive"]))
        need(segment.end_frame_exclusive - segment.start_frame == int(row["candidate_frame_count"]),
             f"Segment length mismatch: {video_id}")
        segments[video_id].append(segment)
    for video_id, values in segments.items():
        previous_end = -1
        for value in values:
            need(value.start_frame >= previous_end, f"Overlapping/unsorted segments: {video_id}")
            need(value.end_frame_exclusive <= int(summaries[video_id]["frame_count"]),
                 f"Segment exceeds frame count: {video_id}")
            previous_end = value.end_frame_exclusive
    for video_id, row in summaries.items():
        values = segments.get(video_id, [])
        need(len(values) == int(row["candidate_segment_count"]),
             f"Per-video segment count mismatch: {video_id}")
        need(sum(value.end_frame_exclusive - value.start_frame for value in values)
             == int(row["candidate_frame_count"]), f"Per-video candidate total mismatch: {video_id}")
    return requested_summaries, dict(segments), metadata


def stable_rank(seed: int, video_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{video_id}\0quality_sampler_smoke_v1".encode()).digest()


def choose_smoke(rows: Iterable[dict[str, str]], seed: int, size: int = 20) -> list[dict[str, str]]:
    """Choose the fixed balanced metadata-only smoke cohort.

    All 64-frame fallback videos are included, then affected-clean and unaffected
    videos are alternated while preferring unused source groups and label balance.
    No model output participates in selection.
    """
    candidates = list(rows)
    need(size == 20, "The v1 fixed smoke cohort size is 20")
    chosen = sorted((row for row in candidates if row["has_clean_64"] == "false"),
                    key=lambda row: (stable_rank(seed, row["video_id"]), row["video_id"]))
    need(len(chosen) == 5, "Expected five 64-frame fallback videos")
    used_ids = {row["video_id"] for row in chosen}
    used_sources = Counter(row["source_group"] for row in chosen)
    labels = Counter(row["label"] for row in chosen)
    for slot in range(len(chosen), size):
        wanted_label = min(("Make", "Miss"), key=lambda label: (labels[label], label))
        category = "affected" if (slot - 5) % 2 == 0 else "unaffected"
        eligible = [row for row in candidates if row["video_id"] not in used_ids
                    and row["label"] == wanted_label and row["has_clean_64"] == "true"
                    and ((int(row["candidate_frame_count"]) > 0) == (category == "affected"))]
        need(bool(eligible), f"No eligible {wanted_label}/{category} smoke candidate")
        row = min(eligible, key=lambda value: (used_sources[value["source_group"]],
                                               stable_rank(seed, value["video_id"]), value["video_id"]))
        chosen.append(row)
        used_ids.add(row["video_id"])
        used_sources[row["source_group"]] += 1
        labels[row["label"]] += 1
    need(labels == Counter({"Make": 10, "Miss": 10}), "Smoke labels are not balanced")
    return sorted(chosen, key=lambda row: row["video_id"])


def timeline(frame_count: int, segments: list[CandidateSegment], start: int, end: int, width: int = 72) -> str:
    def project(frame: int) -> int:
        return min(width - 1, (frame * width) // max(frame_count, 1))
    candidates = [" "] * width
    selected = [" "] * width
    for segment in segments:
        for index in range(project(segment.start_frame), project(max(segment.start_frame, segment.end_frame_exclusive - 1)) + 1):
            candidates[index] = "█"
    for index in range(project(start), project(max(start, end - 1)) + 1):
        selected[index] = "─"
    selected[project(start)] = "["
    selected[project(max(start, end - 1))] = "]"
    return "\n".join((f"Frames:     0{'─' * (width - 10)}{frame_count}",
                       f"Candidates: {''.join(candidates)}", f"Selected:   {''.join(selected)}"))


def grouped(records: list[dict[str, Any]], dimension: str) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    videos: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = str(record[dimension])
        videos[key].add(str(record["video_id"]))
        groups[key].update(clips=1, fallback=int(record["fallback"]),
                           candidate_overlap=int(record["candidate_overlap"]))
    return {key: {"videos": len(videos[key]), **dict(groups[key])} for key in sorted(groups)}


def audit(
    summaries: dict[str, dict[str, str]], segments: dict[str, list[CandidateSegment]], *,
    seed: int, epoch: int, clip_length: int, num_clips: int, max_videos: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, list[dict[str, str]]]:
    rows = list(summaries.values())
    if max_videos is not None:
        need(max_videos > 0, "max_videos must be positive")
        rows = choose_smoke(rows, seed) if max_videos == 20 else sorted(
            rows, key=lambda row: (stable_rank(seed, row["video_id"]), row["video_id"]))[:max_videos]
    else:
        rows.sort(key=lambda row: row["video_id"])
    records: list[dict[str, Any]] = []
    timelines: list[str] = []
    for row in rows:
        video_id = row["video_id"]
        frame_count = int(row["frame_count"])
        values = segments.get(video_id, [])
        selections = select_clips(
            video_id=video_id, frame_count=frame_count, candidate_segments=values,
            clip_length=clip_length, num_clips=num_clips, seed=seed, epoch=epoch,
            selection_mode="seeded_random", candidate_margin_frames=0)
        for selection in selections:
            overlap = candidate_overlap(selection.start_frame, selection.end_frame_exclusive, values)
            need(overlap == selection.candidate_overlap, f"Independent overlap mismatch: {video_id}")
            record = {
                "video_id": video_id, "label": row["label"], "source_group": row["source_group"],
                "fps_rational": row["fps_rational"], "frame_count": frame_count,
                **selection.to_dict(), "seed": seed, "epoch": epoch,
            }
            records.append(record)
            if len(rows) <= 20:
                timelines.extend((f"## {video_id} clip {selection.clip_index}",
                                  timeline(frame_count, values, selection.start_frame,
                                           selection.end_frame_exclusive),
                                  f"Overlap: {overlap}", f"Fallback: {str(selection.fallback).lower()}", ""))
    availability = {str(size): sum(row[f"has_clean_{size}"] == "true" for row in rows)
                    for size in (16, 32, 64)}
    summary = {
        "schema_version": "basketball.quality-aware-sampler-audit.v1",
        "policy_version": POLICY_VERSION, "seed": seed, "epoch": epoch,
        "clip_length": clip_length, "num_clips_per_video": num_clips,
        "videos": len(rows), "clips": len(records),
        "candidate_free_clips": sum(int(record["candidate_overlap"] == 0) for record in records),
        "fallback_clips": sum(int(record["fallback"]) for record in records),
        "fallback_videos": len({record["video_id"] for record in records if record["fallback"]}),
        "total_candidate_overlap": sum(int(record["candidate_overlap"]) for record in records),
        "clean_window_availability": availability,
        "groups": {dimension: grouped(records, dimension)
                   for dimension in ("label", "source_group", "fps_rational")},
        "video_access": {"opens": 0, "probes": 0, "decodes": 0, "validation": 0, "test": 0},
    }
    return records, summary, "\n".join(timelines), rows


def csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    ap.add_argument("--video-summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--clip-length", type=int)
    ap.add_argument("--num-clips", type=int)
    ap.add_argument("--max-videos", type=int)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    cfg = load_config(args.config)
    seed = int(cfg["global_seed"] if args.seed is None else args.seed)
    clip_length = int(cfg["clip_length"] if args.clip_length is None else args.clip_length)
    num_clips = int(cfg["num_clips_per_video"] if args.num_clips is None else args.num_clips)
    summaries, segments, metadata = load_sidecars(args.manifest, args.segments, args.video_summary)
    records, summary, timelines, selected_rows = audit(
        summaries, segments, seed=seed, epoch=args.epoch, clip_length=clip_length,
        num_clips=num_clips, max_videos=args.max_videos)
    summary["sidecar_run_identity"] = metadata["run_identity"]
    summary["sidecar_config_hash"] = metadata["config"]["sha256"]
    outputs = {
        args.output_dir / "selections.csv": csv_text(records, SELECTION_FIELDS),
        args.output_dir / "summary.json": json.dumps(summary, indent=2, sort_keys=True) + "\n",
        args.output_dir / "timelines.txt": timelines,
        args.output_dir / "selected_manifest.csv": csv_text(selected_rows, list(selected_rows[0])) if selected_rows else "",
    }
    if not args.dry_run:
        for path in outputs:
            need(not path.exists() and not path.is_symlink(), f"Refusing to overwrite existing output: {path}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for path, text in outputs.items():
            path.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "DRY_RUN_PASS" if args.dry_run else "AUDIT_COMPLETE", **summary},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, TypeError, json.JSONDecodeError, AuditError, SamplerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
