#!/usr/bin/env python3
"""Verify and publish v3 smoke/full-train derived outputs; never opens videos."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import audit_blur_train_v3 as s


PUBLISHER_VERSION = "blur_train_publisher_v3.1"
SCHEMA_VERSION = "basketball.blur-screening-report.v3.1"
DEFAULT_RUN = s.ARTIFACTS / "blur_train_v3_smoke"
DEFAULT_MANIFEST = s.p.gate.MANIFEST
DEFAULT_CONFIG = s.DEFAULT_CONFIG
LEGACY_JSON = s.ROOT / "docs/data_quality/blur_screening_smoke_v3.json"
LEGACY_MD = LEGACY_JSON.with_suffix(".md")
LIMITATIONS = [
    "The centered 3-second window uses future frames and is an offline screening method.",
    "Local-z normalization is not suitable for detecting an entire video with persistently low quality.",
    "Synthetic calibration is not ground truth for naturally occurring blur.",
    "Source group and FPS may be confounded.",
    "There is no evidence yet that this screening improves Make/Miss classification.",
    "This report is not a data exclusion list; candidate does not mean confirmed blur.",
]
QUANTILE_NAMES = ("p00", "p25", "p50", "p75", "p90", "p95", "p100")


class PublicationError(ValueError):
    """A publication input or policy violation."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=s.ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def quantiles(values: list[float]) -> dict[str, float | None]:
    return {name: percentile(values, q) for name, q in
            (("p00", 0), ("p25", .25), ("p50", .5), ("p75", .75),
             ("p90", .9), ("p95", .95), ("p100", 1))}


def id_sets(manifest: Path) -> tuple[dict[str, dict[str, str]], set[str], set[str]]:
    need(manifest.absolute() == DEFAULT_MANIFEST.absolute() and not manifest.is_symlink(),
         "Only the fixed dataset manifest is accepted")
    rows = s.p.read_csv(manifest)
    by_id: dict[str, dict[str, str]] = {}
    duplicates: set[str] = set()
    for row in rows:
        video_id = row["video_id"]
        if video_id in by_id:
            duplicates.add(video_id)
        by_id[video_id] = row
    need(not duplicates, f"Duplicate IDs in fixed manifest: {sorted(duplicates)[:5]}")
    train = {video_id for video_id, row in by_id.items()
             if row["split"] == "train" and row.get("excluded", "0") == "0"}
    held_out = {video_id for video_id, row in by_id.items()
                if row["split"] in ("validation", "test")}
    need(len(train) == 7193, f"Fixed train set is not 7,193 IDs: {len(train)}")
    return by_id, train, held_out


def safe_relative(path: Path, root: Path) -> str:
    need(not path.is_symlink(), f"Symlink is not allowed in run artifacts: {path}")
    try:
        return str(path.relative_to(root))
    except ValueError as exc:
        raise PublicationError(f"Artifact escapes run directory: {path}") from exc


def jsonl(path: Path, corrupt: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    need(path.is_file() and not path.is_symlink(), f"Illegal ledger path: {path}")
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
                need(isinstance(value, dict), "ledger row is not an object")
                rows.append(value)
            except (json.JSONDecodeError, PublicationError) as exc:
                corrupt.append(f"{path.name}:{number}: {exc}")
    return rows


def add_group(groups: dict[str, dict[str, Any]], name: str, summary: dict[str, Any]) -> None:
    group = groups.setdefault(str(name), dict(videos=0, frames=0, candidate_frames=0,
                                                   candidate_segments=0, affected_videos=0))
    group["videos"] += 1
    group["frames"] += int(summary["processed_frame_count"])
    group["candidate_frames"] += int(summary["candidate_frames"])
    group["candidate_segments"] += len(summary["candidate_segments"])
    group["affected_videos"] += int(summary["candidate_frames"] > 0)


def finish_groups(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for group in groups.values():
        group["candidate_ratio"] = (group["candidate_frames"] / group["frames"]
                                    if group["frames"] else None)
    return dict(sorted(groups.items()))


def attempt_inventory(run_dir: Path, corrupt: list[str]) -> tuple[list[dict[str, Any]], set[str]]:
    """Read every completion marker, including unexpected directories and old attempts."""
    records: list[dict[str, Any]] = []
    incomplete_ids: set[str] = set()
    videos = run_dir / "videos"
    if not videos.is_dir() or videos.is_symlink():
        corrupt.append("videos directory missing or illegal")
        return records, incomplete_ids
    for directory in sorted(videos.iterdir()):
        if not directory.is_dir() or directory.is_symlink() or not re.fullmatch(r"[0-9a-f]{64}", directory.name):
            corrupt.append(f"illegal video artifact path: {safe_relative(directory, run_dir)}")
            continue
        attempts = sorted(directory.iterdir())
        if not attempts:
            corrupt.append(f"empty video artifact directory: {directory.name}")
        for attempt in attempts:
            rel = safe_relative(attempt, run_dir)
            if (not attempt.is_dir() or attempt.is_symlink()
                    or not re.fullmatch(r"attempt-[0-9]{6}", attempt.name)):
                corrupt.append(f"illegal attempt path: {rel}")
                continue
            completion = attempt / "completion.json"
            if not completion.is_file() or completion.is_symlink():
                incomplete_ids.add(directory.name)
                continue
            try:
                record = read_json(completion)
                need(isinstance(record, dict), "completion is not an object")
                video_id = record["video_id"]
                need(isinstance(video_id, str) and video_id, "invalid completion video_id")
                need(directory.name == s.key(video_id), "completion directory hash mismatch")
                records.append(dict(record=record, attempt=attempt, relative=rel))
            except (OSError, json.JSONDecodeError, KeyError, PublicationError) as exc:
                corrupt.append(f"{rel}/completion.json: {exc}")
    return records, incomplete_ids


def latest_invocations(run_dir: Path, corrupt: list[str]) -> list[dict[str, Any]]:
    directory = run_dir / "invocations"
    if not directory.exists():
        return []
    need(directory.is_dir() and not directory.is_symlink(), "Illegal invocations directory")
    result = []
    for path in sorted(directory.glob("*.json")):
        try:
            value = read_json(path)
            need(isinstance(value, dict), "invocation is not an object")
            result.append(value)
        except (OSError, json.JSONDecodeError, PublicationError) as exc:
            corrupt.append(f"{safe_relative(path, run_dir)}: {exc}")
    return result


def verify_run(run_dir: Path, manifest: Path, config: Path, scope: str,
               verify_artifacts: bool = True) -> dict[str, Any]:
    """Return a machine-readable status. It never stats, opens, probes, or decodes videos."""
    run_dir = run_dir.absolute()
    need(run_dir.is_dir() and not run_dir.is_symlink(), f"Run directory missing/illegal: {run_dir}")
    need(config.is_file() and not config.is_symlink(), f"Config missing/illegal: {config}")
    manifest_rows, train_ids, held_out_ids = id_sets(manifest)
    config_hash, manifest_hash = sha256(config), sha256(manifest)
    run = read_json(run_dir / "run.json")
    ctx = run["context"]
    need(ctx["scope"] == scope, f"Requested scope {scope!r} differs from run scope {ctx['scope']!r}")
    corrupt: list[str] = []
    config_hash_consistent = ctx["config_hash"] == config_hash
    manifest_hash_consistent = ctx["inputs"]["manifest"] == manifest_hash
    if not config_hash_consistent:
        corrupt.append("run/config hash mismatch")
    if not manifest_hash_consistent:
        corrupt.append("run/manifest hash mismatch")

    if scope == "smoke":
        cfg = s.load_config(config)
        expected_rows = s.choose_smoke(s.Guard(), cfg["seed"])
        expected_ids = {row["video_id"] for row in expected_rows}
        need(len(expected_ids) == 20, "Smoke scope is not the fixed 20 IDs")
    else:
        expected_rows = [dict(row, label="Make" if row["binary_label"] == "1" else "Miss")
                         for row in manifest_rows.values()
                         if row["video_id"] in train_ids]
        expected_ids = train_ids

    selection = ctx.get("selection", [])
    declared = [row.get("video_id") for row in selection if isinstance(row, dict)]
    selection_duplicates = sorted(video_id for video_id, count in Counter(declared).items() if count > 1)
    declared_ids = {video_id for video_id in declared if isinstance(video_id, str)}
    records, incomplete_dirs = attempt_inventory(run_dir, corrupt)
    records_by_id: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        records_by_id.setdefault(item["record"]["video_id"], []).append(item)

    present_ids = set(records_by_id)
    all_seen_ids = declared_ids | present_ids
    unexpected = sorted(all_seen_ids - expected_ids)
    held_out = sorted(all_seen_ids & held_out_ids)
    unknown = sorted(video_id for video_id in unexpected if video_id not in manifest_rows)
    selection_unexpected = sorted(declared_ids - expected_ids)

    successful: list[str] = []
    terminal_failure: list[str] = []
    corrupt_ids: set[str] = set()
    duplicate_ids: set[str] = set(selection_duplicates)
    summaries: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    effective_attempts: dict[str, str] = {}
    scanner_identity = ctx.get("scanner")
    for row in expected_rows:
        video_id = row["video_id"]
        items = records_by_id.get(video_id, [])
        valid: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for item in items:
            record, attempt = item["record"], item["attempt"]
            try:
                need(record.get("config_hash") == ctx["config_hash"], "completion config hash mismatch")
                need(record.get("scanner") == scanner_identity, "completion scanner/run identity mismatch")
                need(record.get("status") in ("ok", "failed"), "illegal completion status")
                names = {path.name for path in attempt.iterdir()}
                need(names == {"frames.csv", "summary.json", "completion.json"},
                     "partial/unexpected attempt inventory")
                hashes = record.get("output_hashes")
                need(set(hashes or {}) == {"frames.csv", "summary.json"}, "invalid output hash inventory")
                if verify_artifacts:
                    for name, digest in hashes.items():
                        actual = sha256(attempt / name)
                        need(actual == digest, f"artifact hash mismatch: {name}")
                        artifact_hashes[str((attempt / name).relative_to(run_dir))] = actual
                summary = read_json(attempt / "summary.json")
                need(summary.get("video_id") == video_id, "summary ID mismatch")
                need(summary.get("split") == "train", "summary is not train")
                need(summary.get("label") == ("Make" if manifest_rows[video_id]["binary_label"] == "1" else "Miss"),
                     "summary label mismatch")
                need(summary.get("source_group") == manifest_rows[video_id]["source_group"],
                     "summary source group mismatch")
                need(summary.get("config_hash") == ctx["config_hash"], "summary config hash mismatch")
                need(summary.get("scanner_version") == scanner_identity.get("version"),
                     "summary scanner version mismatch")
                need(summary.get("decode_status") == record["status"], "summary/completion status mismatch")
                need(record.get("input_identity", {}).get("video_id") == video_id, "input identity mismatch")
                need(record.get("input_identity", {}).get("manifest_sha256") == manifest_hash,
                     "input manifest identity mismatch")
                expected_path = str(s.p.gate.DATA / manifest_rows[video_id]["video_path"])
                need(record.get("input_identity", {}).get("resolved_path") == expected_path,
                     "input path does not match fixed manifest")
                count = 0
                candidate_flags: list[bool] = []
                with (attempt / "frames.csv").open(newline="") as stream:
                    for frame in csv.DictReader(stream):
                        need(frame.get("video_id") == video_id, "frame ID mismatch")
                        need(int(frame["frame_index"]) == count, "frame index/order mismatch")
                        need(frame.get("split") == "train", "frame is not train")
                        need(frame.get("label") == summary["label"], "frame label mismatch")
                        need(frame.get("source_group") == summary["source_group"], "frame source mismatch")
                        need(frame.get("config_hash") == ctx["config_hash"], "frame config hash mismatch")
                        need(frame.get("scanner_version") == scanner_identity.get("version"),
                             "frame scanner version mismatch")
                        need(frame.get("decode_status") == record["status"], "frame status mismatch")
                        if record["status"] == "ok":
                            need(frame.get("primary_candidate") in ("0", "1"),
                                 "invalid frame candidate flag")
                            candidate_flags.append(frame["primary_candidate"] == "1")
                        count += 1
                need(count == record.get("processed_frame_count") == summary.get("processed_frame_count"),
                     "processed frame count mismatch")
                need(summary.get("expected_frame_count") == record.get("expected_frame_count"),
                     "expected frame count mismatch")
                if record["status"] == "ok":
                    need(count > 0 and count == record["expected_frame_count"], "incomplete successful decode")
                    need(isinstance(summary.get("candidate_segments"), list), "candidate segments unavailable")
                    need(isinstance(summary.get("candidate_frames"), int), "candidate frames unavailable")
                    frame_candidates = sum(candidate_flags)
                    need(frame_candidates == summary["candidate_frames"],
                         "frame/summary candidate count mismatch")
                    need(math.isclose(frame_candidates / count, float(summary["candidate_ratio"]),
                                      rel_tol=0, abs_tol=1e-15),
                         "frame/summary candidate ratio mismatch")
                    starts = [index for index, flag in enumerate(candidate_flags)
                              if flag and (index == 0 or not candidate_flags[index - 1])]
                    ends = [index + 1 for index, flag in enumerate(candidate_flags)
                            if flag and (index + 1 == count or not candidate_flags[index + 1])]
                    need(len(starts) == len(ends) == len(summary["candidate_segments"]),
                         "frame/summary candidate segment count mismatch")
                    for start, end, segment in zip(starts, ends, summary["candidate_segments"]):
                        need(segment["start_frame"] == start and segment["end_frame_exclusive"] == end
                             and segment["frames"] == end - start,
                             "frame/summary candidate segment mismatch")
                valid.append((item, summary))
            except (OSError, json.JSONDecodeError, KeyError, TypeError, PublicationError) as exc:
                corrupt.append(f"{item['relative']}: {exc}")
                corrupt_ids.add(video_id)
        valid_ok = [(item, summary) for item, summary in valid if item["record"]["status"] == "ok"]
        if len(valid_ok) > 1:
            duplicate_ids.add(video_id)
        if valid:
            item, summary = valid[-1]
            effective_attempts[video_id] = item["relative"]
            if item["record"]["status"] == "ok":
                successful.append(video_id)
                summaries.append(summary)
            else:
                terminal_failure.append(video_id)

    ledger_ids: dict[str, set[str]] = {}
    for name in ("access.jsonl", "errors.jsonl", "decode_failures.jsonl"):
        rows = jsonl(run_dir / name, corrupt)
        ledger_ids[name] = {row["video_id"] for row in rows
                            if isinstance(row.get("video_id"), str)}
        bad = ledger_ids[name] - expected_ids
        unexpected.extend(sorted(bad))
        held_out.extend(sorted(bad & held_out_ids))

    invocations = latest_invocations(run_dir, corrupt)
    attempts_ids = set(records_by_id)
    aggregate_ids: set[str] = set()
    aggregate_equal = False
    saved_aggregates = sorted((run_dir / "aggregates").glob("summary-*.json")) \
        if (run_dir / "aggregates").is_dir() else []
    reconstructed = s.aggregate(run_dir, ctx)
    if saved_aggregates:
        try:
            saved = read_json(saved_aggregates[-1])
            aggregate_ids = {row["video_id"] for row in saved.get("videos", [])
                             if isinstance(row, dict) and isinstance(row.get("video_id"), str)}
            aggregate_equal = saved == reconstructed
            if not aggregate_equal:
                corrupt.append("latest aggregate differs from bounded-memory per-video reconstruction")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            corrupt.append(f"latest aggregate is corrupt: {exc}")

    unexpected = sorted(set(unexpected) | (aggregate_ids - expected_ids))
    held_out = sorted(set(held_out) | (aggregate_ids & held_out_ids))
    unknown = sorted(video_id for video_id in unexpected if video_id not in manifest_rows)
    completed_effective = set(successful) | set(terminal_failure)
    missing = sorted(expected_ids - completed_effective)
    incomplete = sorted((expected_ids - present_ids) | (declared_ids - completed_effective))
    hash_to_expected = {s.key(video_id): video_id for video_id in expected_ids}
    incomplete_from_dirs = {hash_to_expected[value] for value in incomplete_dirs if value in hash_to_expected}
    unknown_incomplete_dirs = incomplete_dirs - set(hash_to_expected)
    corrupt.extend(f"attempt without completion under unknown video hash {value}"
                   for value in sorted(unknown_incomplete_dirs))
    incomplete = sorted(set(incomplete) | incomplete_from_dirs)

    invalid = bool(held_out or unknown or selection_unexpected or selection_duplicates
                   or duplicate_ids or corrupt or corrupt_ids)
    all_terminal = not missing and declared_ids == expected_ids
    if invalid:
        status = "INVALID"
    elif all_terminal and terminal_failure:
        status = "COMPLETE_WITH_FAILURES"
    elif all_terminal and len(successful) == len(expected_ids) and aggregate_equal:
        status = "COMPLETE"
    else:
        status = "PARTIAL"

    label_groups: dict[str, dict[str, Any]] = {}
    fps_groups: dict[str, dict[str, Any]] = {}
    source_groups: dict[str, dict[str, Any]] = {}
    ratios: list[float] = []
    durations: list[float] = []
    top_videos: list[dict[str, Any]] = []
    windows = {str(value): dict(videos_available=0, total_windows=0) for value in ctx["config"]["windows"]}
    totals = Counter()
    for summary in summaries:
        frames, candidates = int(summary["processed_frame_count"]), int(summary["candidate_frames"])
        segments = summary["candidate_segments"]
        totals.update(videos=1, frames=frames, candidate_frames=candidates,
                      candidate_segments=len(segments), affected_videos=int(candidates > 0))
        add_group(label_groups, summary["label"], summary)
        add_group(fps_groups, summary["fps_rational"], summary)
        add_group(source_groups, summary["source_group"], summary)
        ratios.append(float(summary["candidate_ratio"]))
        durations.extend(float(segment["duration_seconds"]) for segment in segments)
        for size, value in summary.get("noncandidate_windows", {}).items():
            if size in windows:
                windows[size]["videos_available"] += int(value["exists"])
                windows[size]["total_windows"] += int(value["count"])
        top_videos.append(dict(video_id=summary["video_id"], candidate_ratio=float(summary["candidate_ratio"]),
                               candidate_frames=candidates, frames=frames,
                               candidate_segments=len(segments), source_group=summary["source_group"]))
    top_videos.sort(key=lambda item: (-item["candidate_ratio"], item["video_id"]))
    all_groups = dict(label=finish_groups(label_groups), fps=finish_groups(fps_groups),
                      source_group=finish_groups(source_groups))
    highest_source = None
    if all_groups["source_group"]:
        name, value = max(all_groups["source_group"].items(),
                          key=lambda item: (item[1]["candidate_ratio"], item[0]))
        highest_source = dict(source_group=name, **value)
    longest = None
    for summary in summaries:
        for segment in summary["candidate_segments"]:
            candidate = dict(video_id=summary["video_id"], **segment)
            if longest is None or candidate["duration_seconds"] > longest["duration_seconds"]:
                longest = candidate

    errors = jsonl(run_dir / "errors.jsonl", [])
    decode_failures = jsonl(run_dir / "decode_failures.jsonl", [])
    retry_count = sum("--retry-failures" in invocation.get("command", []) for invocation in invocations)
    resume_count = sum("--resume" in invocation.get("command", []) for invocation in invocations)
    finished = [invocation.get("finished_utc") for invocation in invocations if invocation.get("finished_utc")]
    candidate_ratio = totals["candidate_frames"] / totals["frames"] if totals["frames"] else None
    branch = git_value("branch", "--show-current")
    publisher_commit = git_value("rev-parse", "HEAD")
    publisher_file = Path(__file__).resolve()

    report = dict(
        schema_version=SCHEMA_VERSION,
        run_status=status,
        provenance=dict(
            branch=branch, execution_commit=publisher_commit, scanner_commit=run.get("commit"),
            publisher_commit=publisher_commit, publisher_worktree_dirty=bool(git_value("status", "--porcelain")),
            config_path=str(config), config_hash=config_hash, manifest_path=str(manifest),
            manifest_hash=manifest_hash, scanner_version=scanner_identity.get("version"),
            scanner_identity=scanner_identity, publisher_version=PUBLISHER_VERSION,
            publisher_sha256=sha256(publisher_file), run_identity=scanner_identity.get("sha256"),
            seed=ctx.get("seed"), scope=scope, started_utc=run.get("created_utc"),
            finished_utc=max(finished) if finished else None, output_directory=str(run_dir),
            resume_count=resume_count, retry_count=retry_count),
        coverage=dict(
            expected_videos=len(expected_ids), declared_selected_videos=len(declared_ids),
            successful_videos=len(successful), unresolved_failures=len(terminal_failure),
            incomplete_videos=len(set(incomplete)), corrupt_videos=len(corrupt_ids),
            total_decoded_frames=totals["frames"], total_candidate_frames=totals["candidate_frames"],
            candidate_frame_ratio=candidate_ratio, candidate_segments=totals["candidate_segments"],
            affected_videos=totals["affected_videos"],
            unaffected_videos=totals["videos"] - totals["affected_videos"]),
        integrity=dict(
            expected_but_missing=missing, present_but_unexpected=unexpected,
            duplicate=sorted(duplicate_ids), successful=sorted(successful),
            terminal_failure=sorted(terminal_failure), incomplete=sorted(set(incomplete)),
            corrupt=sorted(set(corrupt)), held_out=held_out, unknown=unknown,
            selected_ids_match=declared_ids == expected_ids,
            config_hash_consistent=config_hash_consistent,
            manifest_hash_consistent=manifest_hash_consistent,
            run_identity_consistent=not any("identity mismatch" in value for value in corrupt),
            artifact_hashes_verified=verify_artifacts, aggregate_rebuilt_equal=aggregate_equal,
            set_counts=dict(expected=len(expected_ids), selected=len(declared_ids), completions=len(present_ids),
                            attempts=len(attempts_ids), aggregate=len(aggregate_ids)),
            set_members=dict(expected=sorted(expected_ids), selected=sorted(declared_ids),
                             completions=sorted(present_ids), attempts=sorted(attempts_ids),
                             aggregate=sorted(aggregate_ids)),
            ledger_ids={name: sorted(values) for name, values in ledger_ids.items()},
            effective_attempts=effective_attempts),
        candidate_summary=dict(
            terminology="candidate / low-local-Laplacian candidate; not confirmed blur",
            total_decoded_frames=totals["frames"], total_candidate_frames=totals["candidate_frames"],
            candidate_frame_ratio=candidate_ratio, candidate_segments=totals["candidate_segments"],
            affected_videos=totals["affected_videos"], windows=windows,
            longest_candidate_segment=longest, highest_candidate_ratio_videos=top_videos[:20],
            highest_candidate_ratio_source=highest_source),
        group_summaries=all_groups,
        distributions=dict(per_video_candidate_ratio=quantiles(ratios),
                           candidate_segment_duration_seconds=quantiles(durations)),
        failures=dict(
            decode_failure_events=len(decode_failures), processing_error_events=len(errors),
            retry_history=[dict(command=invocation.get("command"), status=invocation.get("status"),
                                started_utc=invocation.get("started_utc"), finished_utc=invocation.get("finished_utc"),
                                failed=invocation.get("failed"), suspect=invocation.get("suspect"))
                           for invocation in invocations], unresolved_ids=sorted(terminal_failure)),
        limitations=LIMITATIONS,
        verification=dict(
            verified_utc=utc(), verifier=PUBLISHER_VERSION, video_opens=0, video_probes=0,
            video_decodes=0, validation_video_access=0, test_video_access=0,
            bounded_memory="streams one frames.csv at a time; retains only per-video summaries and segment durations",
            artifact_hash_count=len(artifact_hashes), artifact_hashes=artifact_hashes,
            markdown_json_consistent=None))
    return report


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + [
        "| " + " | ".join(fmt(value) for value in row) + " |" for row in rows]


def render_markdown(report: dict[str, Any]) -> str:
    status, p, c = report["run_status"], report["provenance"], report["coverage"]
    title = f"# Basketball-51 Blur Screening v3.1 — {p['scope']} — {status}"
    banner = ("**PARTIAL REPORT — do not interpret as complete train coverage.**"
              if status != "COMPLETE" else
              "**COMPLETE derived-output verification; candidates are not confirmed blur or an exclusion list.**")
    lines = [title, "", banner, "", "## Run provenance", ""]
    lines += table(["Field", "Value"], [[key, value] for key, value in (
        ("branch", p["branch"]), ("scanner commit", p["scanner_commit"]),
        ("publisher execution commit", p["publisher_commit"]), ("publisher version", p["publisher_version"]),
        ("config", f"{p['config_path']} ({p['config_hash']})"),
        ("manifest", f"{p['manifest_path']} ({p['manifest_hash']})"),
        ("scanner version", p["scanner_version"]), ("run identity", p["run_identity"]),
        ("seed", p["seed"]), ("scope", p["scope"]), ("started UTC", p["started_utc"]),
        ("finished UTC", p["finished_utc"]), ("run directory", p["output_directory"]),
        ("resume / retry count", f"{p['resume_count']} / {p['retry_count']}"))])
    lines += ["", "## Coverage", ""]
    coverage_rows = [
        ("expected_videos", c["expected_videos"]), ("successful_videos", c["successful_videos"]),
        ("unresolved_failures", c["unresolved_failures"]), ("incomplete_videos", c["incomplete_videos"]),
        ("corrupt_videos", c["corrupt_videos"]), ("total_decoded_frames", c["total_decoded_frames"]),
        ("total_candidate_frames", c["total_candidate_frames"]),
        ("candidate_frame_ratio", c["candidate_frame_ratio"]),
        ("candidate_segments", c["candidate_segments"]), ("affected_videos", c["affected_videos"]),
        ("unaffected_videos", c["unaffected_videos"])]
    lines += table(["Metric", "Value"], [[key, value] for key, value in coverage_rows])
    lines += ["", "All candidate values mean low-local-Laplacian candidates, not confirmed blur.",
              "", "## Candidate distributions", ""]
    for dimension in ("label", "fps", "source_group"):
        groups = report["group_summaries"][dimension]
        lines += [f"### {dimension}", ""]
        lines += table(["Group", "Videos", "Frames", "Candidate frames", "Candidate ratio", "Segments", "Affected"],
                       [[name, value["videos"], value["frames"], value["candidate_frames"],
                         value["candidate_ratio"], value["candidate_segments"], value["affected_videos"]]
                        for name, value in groups.items()])
        lines.append("")
    lines += ["### Quantiles and windows", ""]
    lines += table(["Distribution", *QUANTILE_NAMES], [
        ["per-video candidate ratio",
         *(report["distributions"]["per_video_candidate_ratio"][key] for key in QUANTILE_NAMES)],
        ["candidate segment duration seconds",
         *(report["distributions"]["candidate_segment_duration_seconds"][key] for key in QUANTILE_NAMES)]])
    lines += [""] + table(["Non-candidate window", "Videos available", "Total windows"],
                           [[size, value["videos_available"], value["total_windows"]]
                            for size, value in report["candidate_summary"]["windows"].items()])
    lines += ["", "## Integrity and failures", ""]
    integrity = report["integrity"]
    lines += table(["Set difference", "Count", "IDs / evidence"], [
        [key, len(integrity[key]), ", ".join(integrity[key][:20]) + (" …" if len(integrity[key]) > 20 else "")]
        for key in ("expected_but_missing", "present_but_unexpected", "duplicate", "terminal_failure",
                    "incomplete", "corrupt", "held_out")])
    lines += ["", f"Decode failure events: {report['failures']['decode_failure_events']}; "
              f"processing/error events: {report['failures']['processing_error_events']}.",
              "", "## Limitations", ""]
    lines += [f"- {value}" for value in report["limitations"]]
    lines += ["", "## Verification", "",
              f"Artifact hashes verified: `{integrity['artifact_hashes_verified']}`; "
              f"aggregate rebuilt equal: `{integrity['aggregate_rebuilt_equal']}`.", "",
              "Publisher/verification video access: opens=0, probes=0, decodes=0; validation=0; test=0.", ""]
    return "\n".join(lines)


def markdown_core(markdown: str) -> dict[str, str]:
    match = re.search(r"^# .* — (COMPLETE|COMPLETE_WITH_FAILURES|PARTIAL|INVALID)$", markdown, re.M)
    need(match is not None, "Markdown status is missing")
    result = {"run_status": match.group(1)}
    wanted = {"expected_videos", "successful_videos", "unresolved_failures", "incomplete_videos",
              "corrupt_videos", "total_decoded_frames", "total_candidate_frames", "candidate_frame_ratio",
              "candidate_segments", "affected_videos", "unaffected_videos"}
    for line in markdown.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 2 and cells[0] in wanted:
            result[cells[0]] = cells[1]
    need(set(result) == wanted | {"run_status"}, "Markdown core coverage table is incomplete")
    return result


def verify_markdown_json(markdown: str, report: dict[str, Any]) -> None:
    core = markdown_core(markdown)
    need(core["run_status"] == report["run_status"], "Markdown/JSON status mismatch")
    for key, value in report["coverage"].items():
        if key in core:
            need(core[key] == fmt(value), f"Markdown/JSON mismatch: {key}")
    need(markdown == render_markdown(report), "Markdown is not reproducible from JSON summary")


def write_exclusive(path: Path, text: str) -> None:
    need(not path.exists() and not path.is_symlink(), f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)


def legacy_smoke_check() -> None:
    report = read_json(LEGACY_JSON)
    aggregate = report["aggregate"]
    need(aggregate["selected"] == aggregate["ok"] == 20, "Legacy smoke video count changed")
    need(aggregate["processed_frames"] == 3055, "Legacy smoke frame count changed")
    need(aggregate["candidate_frames"] == 54, "Legacy smoke candidate frames changed")
    need(math.isclose(aggregate["candidate_ratio"], .0176759410801964), "Legacy smoke ratio changed")
    need(aggregate["candidate_segments"] == 5 and aggregate["failed"] == 0,
         "Legacy smoke segments/failures changed")
    need(LEGACY_MD.is_file(), "Legacy smoke Markdown missing")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output-dir", dest="legacy_run_dir", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--evidence-dir", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--scope", choices=("smoke", "full-train"), default="smoke")
    ap.add_argument("--output-md", type=Path)
    ap.add_argument("--output-json", type=Path)
    policy = ap.add_mutually_exclusive_group()
    policy.add_argument("--require-complete", action="store_true")
    policy.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--verify-artifacts", action="store_true",
                    help="Explicitly request artifact hashes (always enforced for publication)")
    ap.add_argument("--dry-run", action="store_true", help="Verify and print the plan; write no report")
    ap.add_argument("--check-only", action="store_true",
                    help="Backward-compatible validation of existing tracked smoke publication")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.check_only:
        legacy_smoke_check()
        print("PASS: legacy smoke publication core statistics")
        return 0
    run_dir = args.legacy_run_dir or args.run_dir
    if args.scope == "full-train" and not args.allow_partial:
        args.require_complete = True
    if not args.dry_run:
        if args.scope == "smoke" and args.output_md is None and args.output_json is None:
            args.output_md, args.output_json = LEGACY_MD, LEGACY_JSON
        need(args.output_md is not None and args.output_json is not None,
             "Publication requires both --output-md and --output-json")
        need(args.output_md.absolute() != args.output_json.absolute(), "Markdown and JSON outputs must differ")
        need(not args.output_md.exists() and not args.output_json.exists(), "Refusing existing output")
    report = verify_run(run_dir, args.manifest, args.config, args.scope,
                        verify_artifacts=True if not args.dry_run else args.verify_artifacts)
    status = report["run_status"]
    need(status != "INVALID", "INVALID run cannot be published")
    if args.require_complete:
        need(status == "COMPLETE", f"Complete publication required; run status is {status}")
    elif status != "COMPLETE":
        need(args.allow_partial, f"Run status is {status}; use --allow-partial for an explicit partial report")
    markdown = render_markdown(report)
    verify_markdown_json(markdown, report)
    report["verification"]["markdown_json_consistent"] = True
    markdown = render_markdown(report)
    verify_markdown_json(markdown, report)
    if args.dry_run:
        print(json.dumps(dict(status="DRY_RUN_PASS", run_status=status, scope=args.scope,
                              expected=report["coverage"]["expected_videos"],
                              successful=report["coverage"]["successful_videos"],
                              output_md=str(args.output_md) if args.output_md else None,
                              output_json=str(args.output_json) if args.output_json else None,
                              writes=0, video_opens=0, video_probes=0, video_decodes=0),
                         indent=2, sort_keys=True))
        return 0
    write_exclusive(args.output_json, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    write_exclusive(args.output_md, markdown)
    print(json.dumps(dict(status="PUBLISHED", run_status=status,
                          output_md=str(args.output_md), output_json=str(args.output_json),
                          video_opens=0, video_probes=0, video_decodes=0), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, TypeError, json.JSONDecodeError, PublicationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
