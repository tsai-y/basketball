#!/usr/bin/env python3
"""Create a fixed, cleaned Basketball-51 manifest before any model training."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import re

from inspect_dataset import LABELS, exact_duplicate_groups


ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"(?P<cls>2p[01]|3p[01]|ft[01]|mp[01])_v(?P<game>\d+)_(?P<time>\d{6})_x264\.mp4")
SEED = 20260927
SPLITS = ("train", "validation", "test")
RATIOS = (.70, .15, .15)
TRIALS = 20000


def split_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if not row["excluded"]:
            groups[row["source_group"]].append(row)
    group_ids = sorted(groups)
    if len(group_ids) != 50:
        raise ValueError(f"Expected 50 source groups after v160/v166 merge, got {len(group_ids)}")
    # Choose a fixed group assignment using labels only. No video or model result enters selection.
    quotas = (35, 8, 7)
    totals = [sum(len(group) for group in groups.values()),
              sum(sum(int(r["binary_label"]) for r in group) for group in groups.values())]
    rng = random.Random(SEED)
    best = None
    best_score = float("inf")
    for _ in range(TRIALS):
        shuffled = group_ids[:]
        rng.shuffle(shuffled)
        assignment = dict(zip(shuffled, [name for name, quota in zip(SPLITS, quotas) for _ in range(quota)]))
        counts = {name: [0, 0] for name in SPLITS}
        for group_id, group in groups.items():
            bucket = counts[assignment[group_id]]
            bucket[0] += len(group)
            bucket[1] += sum(int(row["binary_label"]) for row in group)
        score = sum(abs(counts[name][0] - totals[0] * ratio) / totals[0] +
                    abs(counts[name][1] - totals[1] * ratio) / totals[1]
                    for name, ratio in zip(SPLITS, RATIOS))
        if score < best_score:
            best_score, best = score, assignment
    return best, best_score


def build(dataset, summary):
    videos = sorted(dataset.rglob("*.mp4"))
    relative = [str(path.relative_to(dataset)) for path in videos]
    actual = Counter(path.parent.name for path in videos)
    if len(videos) != summary["total_videos"] or dict(actual) != summary["classes"]:
        raise ValueError("Dataset file counts differ from inspected summary; stop and reinspect")
    duplicates = exact_duplicate_groups(videos, dataset)
    if {frozenset(group) for group in duplicates} != {
        frozenset(group) for group in summary["exact_duplicate_groups"]
    }:
        raise ValueError("Exact duplicates differ from inspected summary; stop and reinspect")

    duplicate_index = {}
    for index, group in enumerate(sorted((sorted(group) for group in duplicates)), 1):
        labels = {LABELS[Path(name).parent.name] for name in group}
        for name in group:
            duplicate_index[name] = (f"dup_{index:04d}", len(labels) > 1, name != group[0])

    rows = []
    for name in relative:
        path = Path(name)
        match = PATTERN.fullmatch(path.name)
        if match is None or match["cls"] != path.parent.name:
            raise ValueError(f"Unexpected filename: {name}")
        game = match["game"]
        duplicate_group, conflict, extra_copy = duplicate_index.get(name, (f"single:{name}", False, False))
        reason = "conflicting_outcome" if conflict else "duplicate_copy" if extra_copy else ""
        rows.append({
            "video_id": name, "video_path": name, "original_class": match["cls"],
            "binary_label": LABELS[match["cls"]], "game_code": game,
            "source_timestamp": match["time"],
            "source_group": "v160_v166" if game in {"160", "166"} else f"v{game}",
            "duplicate_group": duplicate_group, "excluded": int(bool(reason)),
            "exclusion_reason": reason, "split": "excluded" if reason else "",
        })
    assignment, score = split_groups(rows)
    for row in rows:
        if not row["excluded"]:
            row["split"] = assignment[row["source_group"]]
    return rows, assignment, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "archive" / "Basketball_51 dataset")
    parser.add_argument("--summary", type=Path, default=ROOT / "docs" / "dataset_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "dataset_manifest.csv")
    parser.add_argument("--split-summary", type=Path, default=ROOT / "data" / "processed" / "split_summary.json")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    summary = json.loads(args.summary.read_text())
    rows, assignment, score = build(dataset, summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {name: dict(Counter(int(row["binary_label"]) for row in rows if row["split"] == name))
              for name in (*SPLITS, "excluded")}
    report = {"seed": SEED, "ratios": dict(zip(SPLITS, RATIOS)), "candidate_assignments": TRIALS,
              "selection_score": score, "source_group_count": len(assignment),
              "group_assignment": dict(sorted(assignment.items())), "label_counts": counts,
              "exclusions": dict(Counter(row["exclusion_reason"] for row in rows if row["excluded"])),
              "total_rows": len(rows)}
    args.split_summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
