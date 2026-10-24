"""Train-only simulation of one requested video-level weighting strength.

This is a sensitivity calculation, not training or hyperparameter search.
Candidates are unverified quality flags, not confirmed blur labels.
"""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

import torch

from development.data_quality.scripts import audit_three_window_quality_v1 as audit


ALPHA = 0.25
ALLOWED_ALPHAS = (0.25, 0.5)
OUTPUT_PARENT = audit.ROOT / "development/data_quality/artifacts/quality_weight_simulation_v1"


def validate_alpha(alpha):
    audit.require(type(alpha) in (int, float) and alpha in ALLOWED_ALPHAS,
                  "Only alpha=0.25 or alpha=0.5 is supported")


def video_weight(overlap_counts, *, alpha=ALPHA):
    """Each input is three distinct WINDOW counts; repeated appearances count."""
    validate_alpha(alpha)
    audit.require(len(overlap_counts) == 3, "Expected three overlap counts")
    audit.require(all(type(count) is int and 0 <= count <= 64 for count in overlap_counts),
                  "Overlap counts must be integers in [0,64]")
    return 1.0 - alpha * sum(overlap_counts) / 192.0


def make_weights(rows, *, alpha=ALPHA):
    validate_alpha(alpha)
    audit.require(bool(rows), "Empty train population")
    ids = [row["video_id"] for row in rows]
    audit.require(len(ids) == len(set(ids)), "Duplicate video IDs")
    weights = []
    for row in rows:
        weight = video_weight(row["candidate_overlap_frames"], alpha=alpha)
        ratio = row["window_exposure_ratio"]
        audit.require(math.isfinite(ratio) and 0 <= ratio <= 1,
                      "Invalid exposure ratio")
        audit.require(math.isclose(ratio, sum(row["candidate_overlap_frames"]) / 192,
                                  rel_tol=0, abs_tol=1e-12), "Exposure ratio/count mismatch")
        weights.append({**row, "raw_weight": weight})
    mean_weight = statistics.fmean(row["raw_weight"] for row in weights)
    for row in weights:
        # Optional scale control: fixed TRAIN-wide constant, never batchwise,
        # classwise, shotwise, or derived from validation/test.
        row["global_mean_normalized_weight"] = row["raw_weight"] / mean_weight
    return weights


def summarize(rows, population_count, population_weight):
    raw_sum = math.fsum(row["raw_weight"] for row in rows)
    count = len(rows)
    before = count / population_count
    after = raw_sum / population_weight
    return {
        "videos": count, "raw_weight_sum": raw_sum,
        "raw_weight_removed": count - raw_sum,
        "raw_weight_reduction_percent": 100 * (1 - raw_sum / count),
        "raw_weight_mean": raw_sum / count,
        "raw_weight_min": min(row["raw_weight"] for row in rows),
        "raw_weight_max": max(row["raw_weight"] for row in rows),
        "global_mean_normalized_weight_sum": math.fsum(
            row["global_mean_normalized_weight"] for row in rows),
        "population_share_before": before, "population_share_after": after,
        "population_share_change_percentage_points": 100 * (after - before),
    }


def build_report(rows, *, alpha=ALPHA):
    weights = make_weights(rows, alpha=alpha)
    total = math.fsum(row["raw_weight"] for row in weights)
    count = len(weights)

    def grouped(field):
        buckets = defaultdict(list)
        for row in weights:
            buckets[row[field]].append(row)
        groups = {key: summarize(values, count, total) for key, values in sorted(buckets.items())}
        audit.require(math.isclose(math.fsum(group["raw_weight_sum"] for group in groups.values()),
                                   total, rel_tol=0, abs_tol=1e-8), "Group weights do not sum to total")
        audit.require(math.isclose(math.fsum(group["population_share_after"] for group in groups.values()),
                                   1, rel_tol=0, abs_tol=1e-12), "Group shares do not sum to one")
        return groups

    changed = [row for row in weights if row["raw_weight"] < 1]
    report = {
        "schema_version": "quality_weight_simulation_v1", "alpha": alpha,
        "formula": f"r_i=sum(candidate_overlap_frames_i)/192; w_i=1-{alpha:g}*r_i",
        "optional_scale_normalization": "w_normalized_i=w_i/mean(w_train), ONE global constant",
        "scope": "train only; video-level loss weights; windows/features/sampling unchanged",
        "definitions": {
            "raw_weight_sum": "coefficient mass, NOT expected loss, gradient influence, or actual video count",
            "share_change": "percentage points, not relative percent",
            "normalized_weights": "Optional scale control preserves all group shares and relative weights; not yet adopted for training.",
        },
        "overall": summarize(weights, count, total),
        "videos_downweighted": len(changed), "videos_unchanged": count - len(changed),
        "affected_only_mean_weight": statistics.fmean(row["raw_weight"] for row in changed) if changed else None,
        "by_label": grouped("label"), "by_shot_type": grouped("shot_type"),
        "by_original_class": grouped("original_class"), "by_source_group": grouped("source_group"),
        "by_fps": grouped("fps_rational"),
        "largest_reductions": sorted(weights, key=lambda row: (row["raw_weight"], row["video_id"]))[:10],
        "limitations": ["Candidate overlap is a quality proxy, not confirmed blur or task uselessness.",
                        "Small label-share changes do not prove causal neutrality or improved accuracy.",
                        "A weak intervention may show little measurable effect; no validation-driven coefficient sweep performed."],
        "training_started": False, "validation_used": False, "test_used": False,
    }
    return report, weights


def render_text(report):
    lines = ["Mild video-level quality weighting: SIMULATION ONLY", report["formula"], "",
             "group    videos raw-weight-sum reduction% share-before% share-after% shift(pp)"]
    for name, group in [("ALL", report["overall"]), *report["by_label"].items(),
                        *report["by_shot_type"].items(), *report["by_original_class"].items()]:
        lines.append(f"{name:<8} {group['videos']:>6} {group['raw_weight_sum']:>14.5f} "
                     f"{group['raw_weight_reduction_percent']:>10.4f} "
                     f"{100*group['population_share_before']:>13.4f} "
                     f"{100*group['population_share_after']:>12.4f} "
                     f"{group['population_share_change_percentage_points']:>+9.5f}")
    overall = report["overall"]
    lines += ["", f"Observed raw weight range: {overall['raw_weight_min']:.8f} to {overall['raw_weight_max']:.8f}",
              f"Downweighted/unchanged videos: {report['videos_downweighted']}/{report['videos_unchanged']}",
              f"Affected-only mean weight: {report['affected_only_mean_weight']}",
              "Optional normalization divides each weight by the same TRAIN mean; group shares unchanged.",
              "No training, validation/test access, cache mutation, or weight-strength sweep.",
              "Weight mass is not loss/gradient mass. Candidate flag is not blur truth."]
    return "\n".join(lines) + "\n"


def main():
    torch.set_num_threads(1)
    # Revalidate original sidecars and train-cache metadata instead of trusting
    # a stale, manually edited derived report.
    source_report, source_rows = audit.run_audit(audit.DEFAULT_CACHE)
    report, weights = build_report(source_rows)
    report.update({"cache_run_id": source_report["cache_run_id"],
                   "gate_config_sha256": source_report["gate_config_sha256"],
                   "input_identities": source_report["input_identities"],
                   "audit_checks": source_report["checks"],
                   "source_sha256": {
                       Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       Path(audit.__file__).name: hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest(),
                   }})
    # The audit's 'weights_assigned=False' describes the source audit, not this
    # derived calculation. No weights are applied to a model or original cache.
    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="simulation_", dir=OUTPUT_PARENT))
    for filename, value in (("report.json", report), ("per_video_weights.json", weights)):
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
