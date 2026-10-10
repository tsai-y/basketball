#!/usr/bin/env python3
"""Export compact, train-only quality sidecars from verified v3.1 metadata.

This program reads the fixed manifest and scanner JSON summaries only.  It does
not resolve, stat, open, probe, or decode any video path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "development/data_quality/artifacts/blur_train_v3_full_train"
PUBLICATION_DIR = ROOT / "development/data_quality/artifacts/blur_train_v3_full_train_publication"
RUNTIME_DIR = ROOT / "development/data_quality/artifacts/blur_train_v3_full_train_runtime"
MANIFEST = ROOT / "data/processed/dataset_manifest.csv"
CONFIG = ROOT / "development/data_quality/configs/blur_train_v3.yaml"
OUTPUT_DIR = ROOT / "development/data_quality/sidecars"
REPORT_MD = PUBLICATION_DIR / "blur_screening_full_train_v3_1.md"
REPORT_JSON = PUBLICATION_DIR / "blur_screening_full_train_v3_1.json"
VERIFICATION_JSON = RUNTIME_DIR / "verification-v3.1.json"
SEGMENTS_NAME = "blur_candidate_segments_v3_1.csv"
SUMMARY_NAME = "blur_video_summary_v3_1.csv"
METADATA_NAME = "blur_quality_sidecar_v3_1.json"
SCHEMA_VERSION = "basketball.blur-quality-sidecar.v3.1"
POLICY_VERSION = "blur_screening_v3_1"

SEGMENT_FIELDS = [
    "video_id", "start_frame", "end_frame_exclusive", "start_timestamp",
    "end_timestamp_exclusive", "candidate_frame_count", "duration_seconds",
    "policy_version", "scanner_version", "config_hash", "run_identity",
]
SUMMARY_FIELDS = [
    "video_id", "label", "source_group", "fps_numerator", "fps_denominator",
    "fps_rational", "frame_count", "duration_seconds", "candidate_frame_count",
    "candidate_ratio", "candidate_segment_count", "longest_candidate_segment_frames",
    "has_clean_16", "has_clean_32", "has_clean_64", "config_hash", "run_identity",
]


class ExportError(ValueError):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    need(isinstance(value, dict), f"Expected JSON object: {relative(path)}")
    return value


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def file_record(path: Path) -> dict[str, str]:
    return {"path": relative(path), "sha256": sha256(path)}


def manifest_rows() -> tuple[dict[str, dict[str, str]], set[str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    train = {row["video_id"]: row for row in rows if row["split"] == "train" and row["excluded"] == "0"}
    held_out = {row["video_id"] for row in rows if row["split"] in {"validation", "test"}}
    need(len(train) == 7193, "Fixed train manifest must contain exactly 7,193 eligible rows")
    return train, held_out


def effective_summaries(run: dict[str, Any]) -> list[dict[str, Any]]:
    selection = run.get("context", {}).get("selection", [])
    expected = {row["video_id"] for row in selection}
    need(len(expected) == 7193, "Run selection must contain exactly 7,193 unique IDs")
    summaries: list[dict[str, Any]] = []
    for video_id in sorted(expected):
        video_hash = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
        attempts = sorted((RUN_DIR / "videos" / video_hash).glob("attempt-*/completion.json"))
        valid: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
        for completion_path in attempts:
            completion = read_json(completion_path)
            if completion.get("status") != "ok":
                continue
            summary_path = completion_path.with_name("summary.json")
            need(summary_path.is_file(), f"Missing summary for {video_id}")
            summary = read_json(summary_path)
            need(sha256(summary_path) == completion["output_hashes"]["summary.json"],
                 f"Summary hash mismatch for {video_id}")
            valid.append((completion_path, completion, summary))
        need(len(valid) == 1, f"Expected exactly one successful completion for {video_id}")
        _, completion, summary = valid[0]
        need(summary.get("video_id") == video_id == completion.get("video_id"),
             f"Video identity mismatch for {video_id}")
        summaries.append(summary)
    return summaries


def build() -> tuple[bytes, bytes, dict[str, Any]]:
    for path in (RUN_DIR / "run.json", REPORT_MD, REPORT_JSON, VERIFICATION_JSON, MANIFEST, CONFIG):
        need(path.is_file(), f"Required metadata is missing: {relative(path)}")
    report = read_json(REPORT_JSON)
    verification = read_json(VERIFICATION_JSON)
    run = read_json(RUN_DIR / "run.json")
    need(report.get("run_status") == "COMPLETE", "Source report is not COMPLETE")
    need(verification.get("run_status") == "COMPLETE", "Verification is not COMPLETE")
    for key in (set(report) - {"verification"}):
        need(report[key] == verification.get(key), f"Publication/verification mismatch: {key}")
    provenance = report["provenance"]
    coverage = report["coverage"]
    ctx = run["context"]
    need(sha256(CONFIG) == provenance["config_hash"] == ctx["config_hash"], "Config hash mismatch")
    need(sha256(MANIFEST) == provenance["manifest_hash"] == ctx["inputs"]["manifest"],
         "Manifest hash mismatch")
    need(provenance["run_identity"] == ctx["scanner"]["sha256"], "Run identity mismatch")
    train, held_out = manifest_rows()
    summaries = effective_summaries(run)

    segment_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    total_candidates = affected = 0
    availability = {16: 0, 32: 0, 64: 0}
    seen: set[str] = set()
    for summary in summaries:
        video_id = summary["video_id"]
        need(video_id in train and video_id not in held_out, f"Non-train ID in scanner output: {video_id}")
        need(video_id not in seen, f"Duplicate video summary: {video_id}")
        seen.add(video_id)
        row = train[video_id]
        need(summary["split"] == "train", f"Non-train summary: {video_id}")
        need(summary["source_group"] == row["source_group"], f"Source-group mismatch: {video_id}")
        expected_label = "Make" if row["binary_label"] == "1" else "Miss"
        need(summary["label"] == expected_label, f"Label mismatch: {video_id}")
        need(summary["config_hash"] == provenance["config_hash"], f"Config mismatch: {video_id}")
        need(summary["scanner_version"] == provenance["scanner_version"], f"Scanner mismatch: {video_id}")
        frame_count = int(summary["processed_frame_count"])
        need(frame_count == int(summary["expected_frame_count"]), f"Frame-count mismatch: {video_id}")
        numerator, denominator = (int(value) for value in summary["fps_rational"].split("/"))
        segments = summary["candidate_segments"]
        previous_end = 0
        segment_candidate_count = 0
        longest = 0
        for index, segment in enumerate(segments):
            start = int(segment["start_frame"])
            end = int(segment["end_frame_exclusive"])
            count = int(segment["frames"])
            need(0 <= start < end <= frame_count, f"Invalid segment bounds: {video_id}")
            need(end - start == count, f"Segment frame-count mismatch: {video_id}")
            need(index == 0 or start >= previous_end, f"Overlapping/unsorted segments: {video_id}")
            previous_end = end
            segment_candidate_count += count
            longest = max(longest, count)
            segment_rows.append({
                "video_id": video_id, "start_frame": start, "end_frame_exclusive": end,
                "start_timestamp": f"{start * denominator / numerator:.12g}",
                "end_timestamp_exclusive": f"{end * denominator / numerator:.12g}",
                "candidate_frame_count": count,
                "duration_seconds": f"{count * denominator / numerator:.12g}",
                "policy_version": POLICY_VERSION, "scanner_version": provenance["scanner_version"],
                "config_hash": provenance["config_hash"], "run_identity": provenance["run_identity"],
            })
        candidates = int(summary["candidate_frames"])
        need(segment_candidate_count == candidates, f"Candidate total mismatch: {video_id}")
        total_candidates += candidates
        affected += int(candidates > 0)
        clean: dict[int, bool] = {}
        for size in availability:
            value = summary["noncandidate_windows"][str(size)]
            clean[size] = bool(value["exists"])
            availability[size] += int(clean[size])
        video_rows.append({
            "video_id": video_id, "label": expected_label, "source_group": row["source_group"],
            "fps_numerator": numerator, "fps_denominator": denominator,
            "fps_rational": summary["fps_rational"], "frame_count": frame_count,
            "duration_seconds": f"{frame_count * denominator / numerator:.12g}",
            "candidate_frame_count": candidates, "candidate_ratio": f"{candidates / frame_count:.12g}",
            "candidate_segment_count": len(segments), "longest_candidate_segment_frames": longest,
            "has_clean_16": str(clean[16]).lower(), "has_clean_32": str(clean[32]).lower(),
            "has_clean_64": str(clean[64]).lower(), "config_hash": provenance["config_hash"],
            "run_identity": provenance["run_identity"],
        })
    need(seen == set(train), "Sidecar coverage does not exactly equal fixed train IDs")
    segment_rows.sort(key=lambda row: (row["video_id"], int(row["start_frame"])))
    video_rows.sort(key=lambda row: row["video_id"])
    need(len(segment_rows) == int(coverage["candidate_segments"]) == 1798, "Segment total mismatch")
    need(total_candidates == int(coverage["total_candidate_frames"]) == 18390, "Candidate-frame total mismatch")
    need(affected == int(coverage["affected_videos"]) == 1698, "Affected-video total mismatch")
    need(availability == {16: 7193, 32: 7193, 64: 7188}, "Clean-window availability mismatch")

    segment_data = csv_bytes(segment_rows, SEGMENT_FIELDS)
    summary_data = csv_bytes(video_rows, SUMMARY_FIELDS)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "quality_source": "blur_screening_v3_1",
        "policy_version": POLICY_VERSION,
        "source_report": file_record(REPORT_MD),
        "source_full_json": file_record(REPORT_JSON),
        "verification": file_record(VERIFICATION_JSON),
        "scanner_commit": provenance["scanner_commit"],
        "publisher_commit": provenance["publisher_commit"],
        "config": file_record(CONFIG),
        "manifest": file_record(MANIFEST),
        "run_identity": provenance["run_identity"],
        "seed": provenance["seed"],
        "export_timestamp": provenance["finished_utc"],
        "export_timestamp_basis": "source_publication_finished_utc_for_reproducible_export",
        "csv_files": {
            SEGMENTS_NAME: {"sha256": hashlib.sha256(segment_data).hexdigest(), "rows": len(segment_rows)},
            SUMMARY_NAME: {"sha256": hashlib.sha256(summary_data).hexdigest(), "rows": len(video_rows)},
        },
        "coverage": {
            "train_videos": len(video_rows), "decoded_frames_from_source_metadata": 1105146,
            "candidate_frames": total_candidates, "candidate_segments": len(segment_rows),
            "affected_videos": affected, "unaffected_videos": len(video_rows) - affected,
            "clean_window_availability": {str(key): value for key, value in availability.items()},
            "validation_ids": 0, "test_ids": 0,
        },
        "limitations": [
            "Candidates are low-local-Laplacian flags, not confirmed blur or an exclusion list.",
            "Local-z normalization may miss an entire video with persistently low quality.",
            "The centered three-second scanner uses future frames and is offline-only.",
            "No validation/test quality sidecar exists; quality-aware selection is train-only.",
            "The export reads scanner metadata only and performs no video access.",
        ],
    }
    return segment_data, summary_data, metadata


def write_exclusive(path: Path, data: bytes) -> None:
    need(not path.exists() and not path.is_symlink(), f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--check-only", action="store_true", help="Rebuild in memory and compare existing sidecars")
    ap.add_argument("--dry-run", action="store_true", help="Validate and report without writing")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    segment_data, summary_data, metadata = build()
    metadata_data = (json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    expected = {
        args.output_dir / SEGMENTS_NAME: segment_data,
        args.output_dir / SUMMARY_NAME: summary_data,
        args.output_dir / METADATA_NAME: metadata_data,
    }
    if args.check_only:
        for path, data in expected.items():
            need(path.is_file() and path.read_bytes() == data, f"Sidecar is missing or not reproducible: {path}")
        status = "CHECK_PASS"
    elif args.dry_run:
        status = "DRY_RUN_PASS"
    else:
        for path in expected:
            need(not path.exists() and not path.is_symlink(), f"Refusing existing output: {path}")
        for path, data in expected.items():
            write_exclusive(path, data)
        status = "EXPORTED"
    print(json.dumps({
        "status": status, "train_videos": metadata["coverage"]["train_videos"],
        "candidate_segments": metadata["coverage"]["candidate_segments"],
        "candidate_frames": metadata["coverage"]["candidate_frames"],
        "video_opens": 0, "video_probes": 0, "video_decodes": 0,
        "validation_video_access": 0, "test_video_access": 0,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ExportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
