#!/usr/bin/env python3
"""Publish the fixed 20-video metadata-only sampler smoke report."""
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
sys.path.insert(0, str(ROOT / "development/data_quality/scripts"))
import audit_quality_aware_sampler_v1 as audit  # noqa: E402

SMOKE_MANIFEST = ROOT / "development/data_quality/smoke_manifest_quality_sampler_v1.csv"
OUTPUT_JSON = ROOT / "docs/data_quality/quality_aware_clip_sampler_smoke_v1.json"
OUTPUT_MD = ROOT / "docs/data_quality/quality_aware_clip_sampler_smoke_v1.md"
SMOKE_FIELDS = ["video_id", "label", "source_group", "fps_rational", "frame_count",
                "quality_category", "selection_seed", "split"]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def render(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Basketball-51 Quality-aware Clip Sampler v1 — metadata-only smoke",
        "",
        "此固定 smoke 僅使用 train manifest 與品質 sidecar；candidate 是低局部 Laplacian 候選，不是模糊真值或排除名單。",
        "",
        "## 結果",
        "",
        "| Metric | Value |", "| --- | --- |",
        f"| videos / clips | {summary['videos']} / {summary['clips']} |",
        f"| Make / Miss | 10 / 10 |",
        f"| sources | {len(summary['groups']['source_group'])} |",
        f"| candidate-free clips | {summary['candidate_free_clips']} |",
        f"| fallback clips / videos | {summary['fallback_clips']} / {summary['fallback_videos']} |",
        f"| total candidate overlap | {summary['total_candidate_overlap']} frames |",
        f"| clean availability 16 / 32 / 64 | {summary['clean_window_availability']['16']} / {summary['clean_window_availability']['32']} / {summary['clean_window_availability']['64']} |",
        "| video opens / probes / decodes | 0 / 0 / 0 |",
        "| validation / test access | 0 / 0 |",
        "",
        "五支沒有 64-frame 零重疊窗口的影片均使用 minimum-overlap fallback；其餘 15 支選到零 overlap 連續窗口。",
        "",
        "## 固定選樣",
        "",
        "| Video | Label | Source | Category | Selected [start,end) | Overlap | Fallback |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    categories = {row["video_id"]: row["quality_category"] for row in report["smoke_manifest"]}
    for row in report["selections"]:
        lines.append(f"| `{row['video_id']}` | {row['label']} | {row['source_group']} | "
                     f"{categories[row['video_id']]} | [{row['start_frame']},{row['end_frame_exclusive']}) | "
                     f"{row['candidate_overlap']} | {fmt(row['fallback'])} |")
    lines += [
        "", "## Timeline", "",
        "以下只依 frame index 繪製，沒有擷取影片畫面。`█` 是 candidate，`[──]` 是所選連續 clip。", "",
        "```text", report["timelines"].rstrip(), "```", "",
        "## Provenance 與邊界", "",
        f"- policy: `{report['policy_version']}`; seed={report['seed']}; epoch={report['epoch']}; clip length={report['clip_length']}。",
        f"- sidecar run identity: `{report['sidecar_run_identity']}`。",
        f"- sidecar config hash: `{report['sidecar_config_hash']}`。",
        f"- smoke manifest SHA-256: `{report['smoke_manifest_sha256']}`。",
        "- validation/test quality mode disabled；沒有 augmentation、I3D/YOLO、訓練、評估或資料刪除。", "",
    ]
    return "\n".join(lines)


def build() -> tuple[bytes, bytes, bytes]:
    cfg = audit.load_config(audit.DEFAULT_CONFIG)
    summaries, segments, metadata = audit.load_sidecars(
        audit.DEFAULT_MANIFEST, audit.DEFAULT_SEGMENTS, audit.DEFAULT_SUMMARY)
    records, summary, timelines, selected = audit.audit(
        summaries, segments, seed=int(cfg["global_seed"]), epoch=0,
        clip_length=int(cfg["clip_length"]), num_clips=int(cfg["num_clips_per_video"]),
        max_videos=20)
    manifest_rows = []
    for row in selected:
        candidates = int(row["candidate_frame_count"])
        category = "fallback" if row["has_clean_64"] == "false" else (
            "affected_clean" if candidates else "unaffected")
        manifest_rows.append({
            "video_id": row["video_id"], "label": row["label"], "source_group": row["source_group"],
            "fps_rational": row["fps_rational"], "frame_count": row["frame_count"],
            "quality_category": category, "selection_seed": cfg["global_seed"], "split": "train",
        })
    manifest_data = csv_bytes(manifest_rows, SMOKE_FIELDS)
    report = {
        "schema_version": "basketball.quality-aware-sampler-smoke.v1",
        "policy_version": cfg["policy_version"], "seed": int(cfg["global_seed"]),
        "epoch": 0, "clip_length": int(cfg["clip_length"]),
        "num_clips_per_video": int(cfg["num_clips_per_video"]),
        "sidecar_run_identity": metadata["run_identity"],
        "sidecar_config_hash": metadata["config"]["sha256"],
        "smoke_manifest_sha256": sha_bytes(manifest_data),
        "smoke_manifest": manifest_rows, "selections": records, "summary": summary,
        "timelines": timelines,
        "limitations": metadata["limitations"],
    }
    json_data = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    md_data = render(report).encode("utf-8")
    return manifest_data, json_data, md_data


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    manifest_data, json_data, md_data = build()
    data = {SMOKE_MANIFEST: manifest_data, OUTPUT_JSON: json_data, OUTPUT_MD: md_data}
    if args.check_only:
        for path, expected in data.items():
            audit.need(path.is_file() and path.read_bytes() == expected, f"Smoke output is not reproducible: {path}")
        status = "CHECK_PASS"
    elif args.dry_run:
        status = "DRY_RUN_PASS"
    else:
        for path in data:
            audit.need(not path.exists() and not path.is_symlink(), f"Refusing existing output: {path}")
        for path, value in data.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(value)
        status = "PUBLISHED"
    print(json.dumps({"status": status, "videos": 20, "video_opens": 0, "video_probes": 0,
                      "video_decodes": 0, "validation_video_access": 0, "test_video_access": 0},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, TypeError, json.JSONDecodeError, audit.AuditError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
