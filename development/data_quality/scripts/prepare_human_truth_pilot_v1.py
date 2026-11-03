"""Prepare a blinded, train-only human quality annotation pilot.

The source videos and frozen sidecars are read-only. Extracted clips use FFV1,
which adds no new lossy compression to the decoded source frames.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT / "archive/Basketball_51 dataset"
MANIFEST = ROOT / "data/processed/dataset_manifest.csv"
SUMMARY = ROOT / "development/data_quality/sidecars/blur_video_summary_v3_1.csv"
SEGMENTS = ROOT / "development/data_quality/sidecars/blur_candidate_segments_v3_1.csv"
SEED = 20261019
CONTEXT_SECONDS = 0.75
MAIN_QUOTAS = {"Make": {"front": 6, "middle": 3, "rear": 11},
               "Miss": {"front": 6, "middle": 3, "rear": 11}}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def rank(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, (SEED, *parts))).encode()).hexdigest()


def band(center: float) -> str:
    if center < 1 / 3:
        return "front"
    if center < 2 / 3:
        return "middle"
    return "rear"


def fps(row: dict[str, str]) -> float:
    return int(row["fps_numerator"]) / int(row["fps_denominator"])


def shot(row: dict[str, str]) -> str:
    return row["video_id"].split("/", 1)[0].rstrip("01")


def prepare_input():
    train = {row["video_id"]: row for row in read_csv(MANIFEST) if row["split"] == "train"}
    summary = {row["video_id"]: row for row in read_csv(SUMMARY)}
    if len(train) != 7193 or set(summary) != set(train):
        raise ValueError("Train manifest and quality summary do not match the frozen 7,193 IDs")
    segments = defaultdict(list)
    candidates = []
    for row in read_csv(SEGMENTS):
        video_id = row["video_id"]
        if video_id not in train:
            raise ValueError("Non-train ID in candidate segments")
        start, end = int(row["start_frame"]), int(row["end_frame_exclusive"])
        count = int(summary[video_id]["frame_count"])
        if not 0 <= start < end <= count:
            raise ValueError(f"Invalid candidate interval: {video_id}")
        segments[video_id].append((start, end))
        candidates.append({"video_id": video_id, "start": start, "end": end,
                           "label": summary[video_id]["label"],
                           "position": band((start + end) / (2 * count))})
    if len(candidates) != 1798:
        raise ValueError("Candidate-segment count differs from the v3.1 sidecar")
    for video_id, intervals in segments.items():
        intervals.sort()
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError(f"Overlapping candidate intervals: {video_id}")
    return train, summary, segments, candidates


def pick_candidates(candidates, summary):
    used_videos = set()
    source_counts, shot_counts, fps_counts = Counter(), Counter(), Counter()
    main = []
    for label in ("Make", "Miss"):
        for position, quota in MAIN_QUOTAS[label].items():
            for _ in range(quota):
                pool = [row for row in candidates if row["label"] == label
                        and row["position"] == position and row["video_id"] not in used_videos]
                if not pool:
                    raise ValueError(f"Insufficient candidate pool: {label}/{position}")
                chosen = min(pool, key=lambda row: (
                    source_counts[summary[row["video_id"]]["source_group"]],
                    shot_counts[shot(summary[row["video_id"]])],
                    fps_counts[summary[row["video_id"]]["fps_rational"]],
                    rank("candidate", row["video_id"], row["start"], row["end"])))
                main.append(chosen)
                used_videos.add(chosen["video_id"])
                info = summary[chosen["video_id"]]
                source_counts[info["source_group"]] += 1
                shot_counts[shot(info)] += 1
                fps_counts[info["fps_rational"]] += 1
    remaining = [row for row in candidates if row["video_id"] not in used_videos]
    practice = []
    for label, quota in (("Make", 3), ("Miss", 2)):
        for _ in range(quota):
            pool = [row for row in remaining if row["label"] == label
                    and row["video_id"] not in used_videos]
            chosen = min(pool, key=lambda row: (
                source_counts[summary[row["video_id"]]["source_group"]],
                rank("practice", row["video_id"], row["start"], row["end"])))
            practice.append(chosen)
            used_videos.add(chosen["video_id"])
            source_counts[summary[chosen["video_id"]]["source_group"]] += 1
    return main, practice


def interval_is_clean(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return all(end <= left or start >= right for left, right in intervals)


def control_start(info, candidate, intervals, duration_frames):
    count = int(info["frame_count"])
    context = round(CONTEXT_SECONDS * fps(info))
    original_count = int(candidate["frame_count"])
    target_center = (candidate["start"] + candidate["end"]) / (2 * original_count)
    options = []
    for start in range(count - duration_frames + 1):
        end = start + duration_frames
        display_start, display_end = max(0, start - context), min(count, end + context)
        if not interval_is_clean(display_start, display_end, intervals):
            continue
        center = (start + end) / (2 * count)
        options.append((band(center) != candidate["position"], abs(center - target_center),
                        rank("control-start", info["video_id"], start), start))
    if not options:
        return None
    return min(options)[-1]


def pair_controls(positives, summary, segments, used_unaffected, *, same_video_count):
    ordered = sorted(positives, key=lambda row: rank("pair", row["video_id"], row["start"]))
    results = []
    unaffected = [row for row in summary.values() if int(row["candidate_segment_count"]) == 0]
    for index, positive in enumerate(ordered):
        positive_info = summary[positive["video_id"]]
        reference = {**positive, "frame_count": positive_info["frame_count"]}
        if index < same_video_count:
            start = control_start(positive_info, reference,
                                  segments[positive["video_id"]], positive["end"] - positive["start"])
            if start is None:
                raise ValueError(f"No clean same-video control: {positive['video_id']}")
            selected_info = positive_info
            length = positive["end"] - positive["start"]
            origin = "same_video_clean_context"
        else:
            pool = [row for row in unaffected if row["label"] == positive["label"]
                    and row["video_id"] not in used_unaffected]
            if not pool:
                raise ValueError("Insufficient unaffected-video control pool")
            selected_info = min(pool, key=lambda row: (
                row["source_group"] != positive_info["source_group"],
                shot(row) != shot(positive_info),
                row["fps_rational"] != positive_info["fps_rational"],
                rank("control-video", positive["video_id"], row["video_id"])))
            used_unaffected.add(selected_info["video_id"])
            length = max(1, round((positive["end"] - positive["start"])
                                  / fps(positive_info) * fps(selected_info)))
            start = control_start(selected_info, reference, [], length)
            if start is None:
                raise ValueError(f"No unaffected control interval: {selected_info['video_id']}")
            origin = "unaffected_video"
        results.append({"video_id": selected_info["video_id"], "start": start,
                        "end": start + length, "label": selected_info["label"],
                        "position": band((2 * start + length) / (2 * int(selected_info["frame_count"]))),
                        "control_origin": origin, "paired_candidate_video_id": positive["video_id"]})
    return results


def anonymize(positives, negatives, prefix):
    pool = [{**row, "is_candidate": True} for row in positives]
    pool += [{**row, "is_candidate": False} for row in negatives]
    pool.sort(key=lambda row: rank("blind", prefix, row["video_id"], row["start"], row["is_candidate"]))
    for index, row in enumerate(pool, 1):
        row["sample_id"] = f"{prefix}{index:03d}"
        row["cohort"] = "practice" if prefix == "P" else "main"
    return pool


def run_command(command):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stderr[-3000:]}")
    return result.stdout


def extract(source: Path, start: int, end: int, destination: Path):
    filter_value = f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
    run_command(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                 "-i", str(source), "-map", "0:v:0", "-vf", filter_value,
                 "-an", "-sn", "-fps_mode", "passthrough", "-c:v", "ffv1", "-level", "3",
                 "-pix_fmt", "yuv420p", str(destination)])
    probe = json.loads(run_command(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                    "-show_entries", "stream=width,height,pix_fmt,nb_read_frames",
                                    "-of", "json", str(destination)]))["streams"][0]
    if (int(probe["nb_read_frames"]) != end - start or int(probe["width"]) != 320
            or int(probe["height"]) != 240 or probe["pix_fmt"] != "yuv420p"):
        raise ValueError(f"Extracted clip does not match source-frame contract: {destination}")
    return digest(destination)


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_plan():
    train, summary, segments, candidates = prepare_input()
    main_pos, practice_pos = pick_candidates(candidates, summary)
    used_unaffected = set()
    main_neg = pair_controls(main_pos, summary, segments, used_unaffected, same_video_count=20)
    practice_neg = pair_controls(practice_pos, summary, segments, used_unaffected, same_video_count=2)
    plan = anonymize(practice_pos, practice_neg, "P") + anonymize(main_pos, main_neg, "M")
    if len(plan) != 90 or len({row["sample_id"] for row in plan}) != 90:
        raise AssertionError("Unexpected pilot size or duplicate sample IDs")
    for row in plan:
        info = summary[row["video_id"]]
        if row["video_id"] not in train or train[row["video_id"]]["video_path"] != row["video_id"]:
            raise ValueError("Sample is not an exact authorized train video")
        count = int(info["frame_count"])
        context = round(CONTEXT_SECONDS * fps(info))
        row["context_start"] = max(0, row["start"] - context)
        row["context_end"] = min(count, row["end"] + context)
        row["frame_count"] = count
        row["source_group"] = info["source_group"]
        row["fps_rational"] = info["fps_rational"]
        row["shot_type"] = shot(info)
        if not row["is_candidate"] and not interval_is_clean(row["context_start"],
                                                                 row["context_end"], segments[row["video_id"]]):
            raise ValueError("Control context overlaps a candidate interval")
    return plan, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "development/data_quality/artifacts/human_truth_pilot_v1_20261019")
    parser.add_argument("--dry-run", action="store_true", help="Validate and plan without opening any video")
    args = parser.parse_args()
    plan, train = build_plan()
    stats = {"seed": SEED, "total": len(plan),
             "cohorts": dict(Counter(row["cohort"] for row in plan)),
             "candidate": sum(row["is_candidate"] for row in plan),
             "control": sum(not row["is_candidate"] for row in plan),
             "main_candidate_position": dict(Counter(row["position"] for row in plan
                                                    if row["cohort"] == "main" and row["is_candidate"])),
             "main_sources": len({row["source_group"] for row in plan if row["cohort"] == "main"})}
    if args.dry_run:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    output = args.output.resolve()
    allowed = (ROOT / "development/data_quality/artifacts").resolve()
    if allowed not in output.parents or output.exists():
        raise ValueError("Output must be a new directory directly beneath ignored data_quality/artifacts")
    output.mkdir(parents=True)
    (output / "private").mkdir()
    (output / "practice").mkdir()
    (output / "main").mkdir()
    blind_rows, key_rows, clip_hashes = [], [], {}
    for row in plan:
        video_path = PurePosixPath(train[row["video_id"]]["video_path"])
        if video_path.is_absolute() or ".." in video_path.parts:
            raise ValueError("Unsafe source path")
        source = DATA_ROOT.joinpath(*video_path.parts)
        if source.is_symlink() or not source.is_file() or DATA_ROOT.resolve() not in source.resolve().parents:
            raise ValueError(f"Unsafe or missing source: {source}")
        name = row["sample_id"]
        target = output / row["cohort"] / f"{name}_target.mkv"
        context = output / row["cohort"] / f"{name}_context.mkv"
        clip_hashes[str(target.relative_to(output))] = extract(source, row["start"], row["end"], target)
        clip_hashes[str(context.relative_to(output))] = extract(source, row["context_start"],
                                                               row["context_end"], context)
        blind_rows.append({"sample_id": name, "target_clip": str(target.relative_to(output)),
                           "context_clip": str(context.relative_to(output)),
                           "target_start_in_context_frame": row["start"] - row["context_start"],
                           "target_end_in_context_frame_exclusive": row["end"] - row["context_start"],
                           "visible_blur": "", "issue_type": "", "severity_0_to_3": "",
                           "affected_region": "", "task_usability": "", "confidence": "", "notes": ""})
        key_rows.append({"sample_id": name, "cohort": row["cohort"],
                         "is_candidate": int(row["is_candidate"]), "video_id": row["video_id"],
                         "label": row["label"], "source_group": row["source_group"],
                         "shot_type": row["shot_type"], "fps_rational": row["fps_rational"],
                         "position": row["position"], "start_frame": row["start"],
                         "end_frame_exclusive": row["end"], "context_start_frame": row["context_start"],
                         "context_end_frame_exclusive": row["context_end"],
                         "control_origin": row.get("control_origin", ""),
                         "paired_candidate_video_id": row.get("paired_candidate_video_id", "")})
    blind_fields = list(blind_rows[0])
    key_fields = list(key_rows[0])
    write_csv(output / "annotation_sheet.csv", blind_rows, blind_fields)
    write_csv(output / "private/sample_key.csv", key_rows, key_fields)
    metadata = {**stats, "scope": "train only", "context_seconds_each_side": CONTEXT_SECONDS,
                "format": "FFV1 lossless yuv420p 320x240; no resize, sharpen, or enhancement",
                "source_sha256": {str(path.relative_to(ROOT)): digest(path)
                                  for path in (MANIFEST, SUMMARY, SEGMENTS)},
                "clip_sha256": clip_hashes}
    (output / "extraction_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                                     encoding="utf-8")
    (output / "README.txt").write_text(
        "Human quality annotation pilot v1 (train only)\n"
        "1. Start with P001-P010 practice clips; revise annotation rules before M001-M080.\n"
        "2. Open *_target.mkv for the scored interval and *_context.mkv for surrounding frames.\n"
        "3. Fill annotation_sheet.csv without opening private/sample_key.csv.\n"
        "4. Mark visible_blur=yes/no/uncertain; keep issue_type and task_usability separate.\n"
        "5. The private key links anonymous clips to original IDs and candidate/control status.\n"
        "6. These balanced pilot samples do not directly estimate dataset prevalence or recall.\n",
        encoding="utf-8")
    print(json.dumps({**stats, "output": str(output), "files_verified": len(clip_hashes)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
