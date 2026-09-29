#!/usr/bin/env python3
"""Inspect the local Basketball-51 RGB clips without extracting model inputs."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess


DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "archive" / "Basketball_51 dataset"
LABELS = {"2p0": 0, "2p1": 1, "3p0": 0, "3p1": 1,
          "ft0": 0, "ft1": 1, "mp0": 0, "mp1": 1}
NAME = re.compile(r"^(2p[01]|3p[01]|ft[01]|mp[01])_v(\d+)_(\d{6})_x264\.mp4$")


def probe(path, decode):
    result = {"path": str(path), "bytes": path.stat().st_size}
    command = ["ffprobe", "-v", "error", "-show_entries",
               "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,nb_frames,duration",
               "-of", "json", str(path)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if proc.returncode:
            raise ValueError(proc.stderr.strip() or f"ffprobe exit {proc.returncode}")
        data = json.loads(proc.stdout)
        streams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        if len(streams) != 1:
            raise ValueError(f"expected 1 video stream; found {len(streams)}")
        video = streams[0]
        result.update(codec=video.get("codec_name"), width=video.get("width"),
                      height=video.get("height"), fps=video.get("avg_frame_rate"),
                      frames=int(video["nb_frames"]) if video.get("nb_frames", "N/A").isdigit() else None,
                      duration=float(data["format"]["duration"]))
        if result["fps"] not in (None, "0/0"):
            result["fps_value"] = float(Fraction(result["fps"]))
        if decode:
            check = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path),
                                    "-map", "0:v:0", "-f", "null", "-"],
                                   capture_output=True, text=True, timeout=60)
            if check.returncode:
                result["decode_error"] = check.stderr.strip()[-1000:] or f"ffmpeg exit {check.returncode}"
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        result["probe_error"] = str(exc)
    return result


def quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    return {key: round(ordered[round((len(ordered) - 1) * q)], 4)
            for key, q in (("min", 0), ("p05", .05), ("median", .5), ("p95", .95), ("max", 1))}


def exact_duplicate_groups(videos, root):
    by_size = defaultdict(list)
    for path in videos:
        by_size[path.stat().st_size].append(path)
    by_digest = defaultdict(list)
    for paths in by_size.values():
        if len(paths) > 1:
            for path in paths:
                by_digest[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
    return [[str(path.relative_to(root)) for path in paths]
            for paths in by_digest.values() if len(paths) > 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--decode", action="store_true", help="Decode every video stream with ffmpeg")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    root = args.dataset.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Dataset directory is missing: {root}")

    files = sorted(root.rglob("*"))
    videos = [p for p in files if p.is_file() and p.suffix.lower() == ".mp4"]
    extras = [str(p.relative_to(root)) for p in files if p.is_file() and p.suffix.lower() != ".mp4"]
    counts = Counter(p.parent.name for p in videos)
    unknown = sorted(set(counts) - LABELS.keys())
    names = []
    game_groups = defaultdict(list)
    shot_keys = defaultdict(list)
    for path in videos:
        match = NAME.fullmatch(path.name)
        if not match or match.group(1) != path.parent.name:
            names.append(str(path.relative_to(root)))
        else:
            game, timestamp = match.group(2), match.group(3)
            game_groups[game].append(path)
            shot_keys[(game, timestamp)].append(path)

    records = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(probe, p, args.decode) for p in videos]
        for future in as_completed(futures):
            records.append(future.result())
    errors = [r for r in records if "probe_error" in r or "decode_error" in r]
    duplicates = exact_duplicate_groups(videos, root)
    durations = [r["duration"] for r in records if "duration" in r]
    fps_values = [r["fps_value"] for r in records if "fps_value" in r]
    frames = [r["frames"] for r in records if r.get("frames") is not None]
    summary = {
        "dataset": str(root), "total_videos": len(videos), "other_files": extras,
        "classes": dict(sorted(counts.items())), "unknown_classes": unknown,
        "label_mapping": LABELS,
        "make": sum(counts[k] for k, v in LABELS.items() if v == 1),
        "miss": sum(counts[k] for k, v in LABELS.items() if v == 0),
        "formats": dict(Counter(p.suffix.lower() for p in videos)),
        "codecs": dict(Counter(r.get("codec") for r in records)),
        "fps": dict(sorted(Counter(r.get("fps") for r in records).items(), key=lambda x: str(x[0]))),
        "resolution": dict(Counter(f'{r.get("width")}x{r.get("height")}' for r in records)),
        "duration_seconds": quantiles(durations), "fps_value": quantiles(fps_values),
        "frame_count": quantiles(frames), "game_ids_in_filenames": len(game_groups),
        "game_id_range": [min(game_groups), max(game_groups)] if game_groups else None,
        "invalid_filenames": names[:100], "invalid_filename_count": len(names),
        "same_game_timestamp_groups": [
            {"game": game, "timestamp": timestamp, "files": [str(p.relative_to(root)) for p in paths]}
            for (game, timestamp), paths in sorted(shot_keys.items()) if len(paths) > 1
        ],
        "exact_duplicate_groups": duplicates,
        "exact_duplicate_group_count": len(duplicates),
        "exact_duplicate_file_count": sum(map(len, duplicates)),
        "conflicting_outcome_duplicate_groups": [
            group for group in duplicates
            if len({LABELS[Path(name).parent.name] for name in group}) > 1
        ],
        "checked_full_decode": args.decode, "error_count": len(errors),
        "errors": errors[:100],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
