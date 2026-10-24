"""CPU-only, train-only audit of fixed-window candidate metadata.

No video decoding, model construction, optimizer, or cache rewriting occurs.
The shared split manifest is read to establish the train allowlist; no held-out
video or feature-cache record is opened. Candidate flags are not blur truth.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import tempfile

import torch

from development.data_pipeline.multiwindow_v1 import (
    candidate_overlap_counts,
    select_covering_windows,
)
from development.data_pipeline.pipeline import TrainQualityIndex, load_gate_config


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "development/mixnet_i3d/artifacts/feature_cache_v1/67b6400e0982bb6d"
OUTPUT_PARENT = ROOT / "development/data_quality/artifacts/three_window_quality_audit_v1"
EXPECTED_RUN = "67b6400e0982bb6d4be8f659d5067add9a0ba790bb456e00442601b1e563c4f6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def independent_counts(frame_count, windows, segments):
    """Brute-force set oracle, independent of interval intersection arithmetic."""
    require(type(frame_count) is int and frame_count > 0, "Invalid frame count")
    candidates = set()
    previous_end = 0
    for start, end in segments:
        require(type(start) is int and type(end) is int, "Non-integer segment")
        require(0 <= start < end <= frame_count, "Invalid candidate interval")
        require(start >= previous_end, "Unsorted/overlapping candidate intervals")
        candidates.update(range(start, end))
        previous_end = end
    covered, counts = set(), []
    for start, end in windows:
        require(0 <= start < end <= frame_count and end - start == 64,
                "Invalid 64-frame window")
        frames = set(range(start, end))
        covered.update(frames)
        counts.append(len(frames & candidates))
    require(len(windows) == 3 and covered == set(range(frame_count)),
            "Three windows do not cover the full video")
    return tuple(counts), len(candidates)


def validate_record(record, row, run_id, frame_count, windows, expected_counts):
    """Check quality metadata only; this is not a feature-value integrity audit."""
    video_id = row["video_id"]
    require(row["split"] == "train" and row["excluded"] == "0", "Non-train row")
    require(record["run_id"] == run_id, f"Cache run mismatch: {video_id}")
    require(record["video_id"] == video_id, f"Cache ID mismatch: {video_id}")
    require(record["split"] == "train", f"Non-train cache record: {video_id}")
    require(type(record["binary_label"]) is int
            and record["binary_label"] == int(row["binary_label"]),
            f"Label mismatch: {video_id}")
    require(type(record["frame_count"]) is int and record["frame_count"] == frame_count,
            f"Frame count mismatch: {video_id}")
    require(tuple(tuple(pair) for pair in record["windows"]) == windows,
            f"Window mismatch: {video_id}")
    require(record["quality_aware_selection_used"] is False,
            f"Window selection unexpectedly quality-aware: {video_id}")
    require(record["quality_metadata_available"] is True,
            f"Missing quality metadata: {video_id}")
    counts = record["candidate_overlap_frames"]
    require(isinstance(counts, torch.Tensor) and counts.dtype == torch.int64
            and tuple(counts.shape) == (3,), f"Invalid overlap tensor: {video_id}")
    require(tuple(counts.tolist()) == expected_counts, f"Cached overlap mismatch: {video_id}")


def describe(values):
    ordered = sorted(values)
    require(bool(ordered), "Cannot describe an empty population")

    def quantile(q):
        position = (len(ordered) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])

    return {"n": len(ordered), "mean": statistics.fmean(ordered),
            "min": ordered[0], "median": quantile(0.5),
            "p90": quantile(0.9), "p95": quantile(0.95), "max": ordered[-1]}


def histogram(values):
    bands = {"zero": 0, "(0,1%]": 0, "(1%,5%]": 0,
             "(5%,10%]": 0, "(10%,25%]": 0, "(25%,100%]": 0}
    for value in values:
        require(0 <= value <= 1, "Ratio outside [0,1]")
        if value == 0:
            key = "zero"
        elif value <= 0.01:
            key = "(0,1%]"
        elif value <= 0.05:
            key = "(1%,5%]"
        elif value <= 0.10:
            key = "(5%,10%]"
        elif value <= 0.25:
            key = "(10%,25%]"
        else:
            key = "(25%,100%]"
        bands[key] += 1
    return bands


def summarize(rows):
    affected = [row for row in rows if row["unique_candidate_frames"] > 0]
    exposures = [row["window_exposure_ratio"] for row in rows]
    return {
        "videos": len(rows), "affected_videos": len(affected),
        "affected_rate": len(affected) / len(rows),
        "unique_candidate_frames": sum(row["unique_candidate_frames"] for row in rows),
        "decoded_frames": sum(row["frame_count"] for row in rows),
        "candidate_window_exposures": sum(row["candidate_window_exposures"] for row in rows),
        "unique_video_ratio": describe([row["unique_video_ratio"] for row in rows]),
        "window_exposure_ratio": describe(exposures),
        "affected_only_window_exposure_ratio": (
            describe([row["window_exposure_ratio"] for row in affected]) if affected else None
        ),
        "exposure_ratio_histogram": histogram(exposures),
    }


def grouped(rows, field):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row[field]].append(row)
    return {key: summarize(values) for key, values in sorted(buckets.items())}


def run_audit(cache_root):
    config = load_gate_config()
    quality = TrainQualityIndex(config)
    with config.paths["manifest"].open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["split"] == "train" and row["excluded"] == "0"]
    video_ids = [row["video_id"] for row in rows]
    require(len(video_ids) == len(set(video_ids)) == 7193, "Train coverage mismatch")
    require(set(video_ids) == set(quality.summaries), "Quality/train ID mismatch")
    require(Counter(int(row["binary_label"]) for row in rows) == {0: 3421, 1: 3772},
            "Train labels changed")

    def read(path):
        require(path.is_file() and not path.is_symlink(), f"Invalid cache file: {path}")
        return torch.load(path, map_location="cpu", weights_only=True)

    protocol = read(cache_root / "run.pt")
    run_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    require(run_id == EXPECTED_RUN, "Unexpected cache run identity")
    require(protocol["input_identities"] == config.raw["identities"], "Cache inputs changed")
    require(protocol["window_policy"] == "three_window_coverage_v1"
            and protocol["clips_per_video"] == 3, "Cache window protocol mismatch")
    require(protocol["quality_weighting_used"] is False
            and protocol["quality_aware_selection_used"] is False, "Cache is not baseline")
    index = read(cache_root / "train/index.pt")
    require(index["split"] == "train" and index["run_id"] == run_id, "Invalid train index")
    require(index["video_ids"] == video_ids and index["count"] == len(rows),
            "Train index coverage/order mismatch")

    results = []
    for number, row in enumerate(rows, 1):
        video_id = row["video_id"]
        summary = quality.summaries[video_id]
        n = int(summary["frame_count"])
        windows = select_covering_windows(n)
        segments = quality.segments.get(video_id, [])
        counts, unique = independent_counts(n, windows, segments)
        require(candidate_overlap_counts(windows, segments) == counts,
                f"Interval/set count mismatch: {video_id}")
        require(unique == int(summary["candidate_frame_count"]),
                f"Sidecar candidate count mismatch: {video_id}")
        label = int(row["binary_label"])
        require(summary["label"] == ("Make" if label else "Miss")
                and summary["source_group"] == row["source_group"],
                f"Sidecar label/source mismatch: {video_id}")
        filename = hashlib.sha256(video_id.encode()).hexdigest() + ".pt"
        record = read(cache_root / "train" / filename)
        validate_record(record, row, run_id, n, windows, counts)
        original_class = row["original_class"]
        require(original_class in {"2p0", "2p1", "3p0", "3p1", "ft0", "ft1", "mp0", "mp1"}
                and int(original_class[-1]) == label, f"Invalid original class: {video_id}")
        exposure_count = sum(counts)
        results.append({
            "video_id": video_id, "label": "Make" if label else "Miss",
            "original_class": original_class, "shot_type": original_class[:-1],
            "source_group": row["source_group"], "fps_rational": summary["fps_rational"],
            "frame_count": n, "windows": windows, "candidate_overlap_frames": counts,
            "unique_candidate_frames": unique, "unique_video_ratio": unique / n,
            "candidate_window_exposures": exposure_count,
            "window_exposure_ratio": exposure_count / 192,
            "repeated_candidate_exposures": exposure_count - unique,
        })
        if number % 1000 == 0 or number == len(rows):
            print(f"Audited {number}/{len(rows)} train records", flush=True)

    overall = summarize(results)
    require((overall["unique_candidate_frames"], overall["decoded_frames"],
             overall["affected_videos"]) == (18390, 1105146, 1698), "Frozen totals changed")
    window_stats = []
    for i in range(3):
        counts = [row["candidate_overlap_frames"][i] for row in results]
        window_stats.append({"window_index": i, "positive_overlap_videos": sum(x > 0 for x in counts),
                             "candidate_exposures": sum(counts), "overlap_frames": describe(counts)})
    report = {
        "schema_version": "three_window_train_quality_audit_v1",
        "cache_root": str(cache_root.resolve()), "cache_run_id": run_id,
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gate_config_sha256": config.sha256, "input_identities": config.raw["identities"],
        "checks": {"all_train_cache_metadata_matches": True, "set_oracle_matches_interval_counts": True,
                   "all_videos_fully_covered": True, "candidate_intervals_sorted_disjoint": True,
                   "video_decodes": 0, "validation_cache_records_read": 0,
                   "test_cache_records_read": 0, "training_started": False,
                   "features_recomputed": False, "weights_assigned": False},
        "definitions": {
            "unique_video_ratio": "unique candidate frames / original video frame count",
            "window_exposure_ratio": "sum of three window overlap counts / (3*64); repeated input appearances counted",
            "independent_unit": "video; three overlapping windows are not independent samples",
        },
        "overall": overall, "by_label": grouped(results, "label"),
        "by_shot_type": grouped(results, "shot_type"),
        "by_original_class": grouped(results, "original_class"),
        "by_source_group": grouped(results, "source_group"),
        "by_fps": grouped(results, "fps_rational"), "by_window": window_stats,
        "repeated_candidate_exposures": sum(row["repeated_candidate_exposures"] for row in results),
        "largest_exposure_ratio_examples": sorted(results, key=lambda row: (-row["window_exposure_ratio"], row["video_id"]))[:10],
        "limitations": ["Candidate flags are not verified blur or unusability labels.",
                        "Descriptive train-only audit; no evidence here of classification improvement.",
                        "No video pixel inspection or feature-value integrity re-audit.",
                        "The shared manifest is read for split allowlisting; held-out video/features are not opened."],
    }
    return report, results


def render_text(report):
    overall = report["overall"]
    lines = ["Fixed three-window TRAIN quality audit: PASS", "",
             "group       videos affected affected% mean-exposure% affected-mean-exposure%"]
    for key, group in [("ALL", overall), *report["by_label"].items(),
                       *report["by_original_class"].items()]:
        affected = group["affected_only_window_exposure_ratio"]
        lines.append(f"{key:<11} {group['videos']:>6} {group['affected_videos']:>8} "
                     f"{100*group['affected_rate']:>9.2f} "
                     f"{100*group['window_exposure_ratio']['mean']:>14.3f} "
                     f"{100*affected['mean'] if affected else 0:>23.3f}")
    lines.extend(["", f"Unique candidate frames: {overall['unique_candidate_frames']}",
                  f"Candidate window exposures: {overall['candidate_window_exposures']}",
                  f"Repeated candidate exposures: {report['repeated_candidate_exposures']}",
                  "Exposure histogram: " + json.dumps(overall["exposure_ratio_histogram"]),
                  "", "No training, weighting, video decoding, or held-out feature access.",
                  "Candidate flag != confirmed blur. Exposure ratio != unique-video ratio."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report, rows = run_audit(args.cache_root)
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="audit_", dir=OUTPUT_PARENT))
    for filename, value in (("report.json", report), ("per_video.json", rows)):
        with (output / filename).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    text = render_text(report)
    with (output / "report.txt").open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(text, end="")
    print("Saved:", output)


if __name__ == "__main__":
    main()
