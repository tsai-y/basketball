"""Compare the three predeclared seeds against the unchanged temporal baseline."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from tempfile import mkdtemp

from development.temporal_baseline_v1.cached_features import ROOT
from development.temporal_baseline_v1.train_temporal import require
from development.temporal_balanced_v1.sampling import CLASSES


FOLDER = Path(__file__).resolve().parent
BASELINE = ROOT / "development/temporal_baseline_v1/artifacts"
BASELINE_RUNS = {
    20261024: BASELINE / "temporal_seed20261024_y7wv4h0p",
    20261025: BASELINE / "temporal_seed20261025_cwac4gwz",
    20261026: BASELINE / "temporal_seed20261026_59bm89qx",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def inspect_run(path, shot_by_id):
    summary = read_json(path / "summary.json")
    predictions = read_csv(path / "validation_predictions.csv")
    history = read_csv(path / "history.csv")
    require(len(predictions) == len(shot_by_id), "Prediction/validation count mismatch")
    require(len({r["video_id"] for r in predictions}) == len(predictions), "Duplicate prediction IDs")
    require({r["video_id"] for r in predictions} == set(shot_by_id), "Validation ID mismatch")
    minimum = min(history, key=lambda r: float(r["val_loss"]))
    require(int(minimum["epoch"]) == summary["best_epoch"], "Not the minimum-loss checkpoint")
    require(abs(float(minimum["val_loss"]) - summary["validation"]["loss"]) < 1e-6,
            "Best validation loss mismatch")
    cm = [[0, 0], [0, 0]]
    for row in predictions:
        group = shot_by_id[row["video_id"]]
        label, predicted = int(row["true_label"]), int(row["predicted_label"])
        require(label == int(group[-1]), "Validation label mismatch")
        require(predicted == int(float(row["p_make"]) >= .5), "Prediction threshold mismatch")
        cm[label][predicted] += 1
    require(cm == summary["validation"]["confusion_matrix"], "Confusion matrix mismatch")
    ft = [r for r in predictions if shot_by_id[r["video_id"]].startswith("ft")]
    def recall(label):
        rows = [r for r in ft if int(r["true_label"]) == label]
        require(len(rows) > 0, "Missing free-throw label")
        return sum(int(r["predicted_label"]) == label for r in rows) / len(rows)
    return summary, predictions, {
        "accuracy": summary["validation"]["accuracy"],
        "macro_f1": summary["validation"]["macro_f1"],
        "loss": summary["validation"]["loss"],
        "ft_accuracy": sum(r["true_label"] == r["predicted_label"] for r in ft) / len(ft),
        "ft_miss_recall": recall(0), "ft_make_recall": recall(1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-runs", type=Path, nargs=3)
    args = parser.parse_args(argv)
    runs = args.balanced_runs
    if runs is None:
        runs = []
        for seed in BASELINE_RUNS:
            matches = [p for p in (FOLDER / "artifacts").glob(f"balanced_seed{seed}_*")
                       if (p / "summary.json").exists()]
            require(len(matches) == 1, "Specify --balanced-runs when a seed has zero/multiple completed runs")
            runs.extend(matches)
    by_seed = {read_json(p / "summary.json")["seed"]: p for p in runs}
    require(set(by_seed) == set(BASELINE_RUNS) and len(runs) == 3, "Expected the three fixed seeds")
    manifest = ROOT / "data/processed/dataset_manifest.csv"
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    shot_by_id = {r["video_id"]: r["original_class"] for r in read_csv(manifest)
                  if r["split"] == "validation" and r["excluded"] == "0"}
    require(len(shot_by_id) == 1547, "Unexpected validation size")
    data = {"ordinary": [], "balanced": []}
    details = []
    for seed, baseline in BASELINE_RUNS.items():
        balanced = by_seed[seed]
        old_config, new_config = read_json(baseline / "config.json"), read_json(balanced / "config.json")
        require(new_config["experiment"] == "frozen_mixnet_i3d_window_transformer_balanced_v1",
                "Unexpected balanced experiment")
        require(new_config["sampling"]["manifest_sha256"] == manifest_sha,
                "Comparison manifest identity mismatch")
        for key, value in old_config.items():
            if key not in ("experiment", "source_sha256"):
                require(new_config[key] == value, f"Controlled setting changed: {key}")
        for path, digest in old_config["source_sha256"].items():
            require(new_config["source_sha256"][path] == digest, f"Original source changed: {path}")
        old_summary, old_predictions, old_metrics = inspect_run(baseline, shot_by_id)
        new_summary, new_predictions, new_metrics = inspect_run(balanced, shot_by_id)
        require([r["video_id"] for r in old_predictions] == [r["video_id"] for r in new_predictions],
                "Validation order changed")
        draws = read_csv(balanced / "sampling_counts.csv")
        require(len(draws) == new_summary["epochs_completed"], "Missing sampling audit epochs")
        for row in draws:
            require(sum(int(row[g]) for g in CLASSES) == int(row["total_draws"]) == 7193,
                    "Unexpected actual sampling counts")
            require(0 < int(row["unique_videos"]) <= 7193, "Invalid unique-video count")
        data["ordinary"].append(old_metrics)
        data["balanced"].append(new_metrics)
        details.append({"seed": seed, "ordinary_run": str(baseline), "balanced_run": str(balanced),
                        "ordinary_best_epoch": old_summary["best_epoch"],
                        "balanced_best_epoch": new_summary["best_epoch"],
                        "ordinary": old_metrics, "balanced": new_metrics})
    aggregate = {}
    for name, rows in data.items():
        aggregate[name] = {key: {"mean": statistics.mean(r[key] for r in rows),
                                 "sample_std": statistics.stdev(r[key] for r in rows)}
                           for key in rows[0]}
    report = {"evaluation_split": "validation", "seeds": list(BASELINE_RUNS),
              "std_definition": "sample standard deviation across three seeds; not a confidence interval",
              "test_used": False, "controlled_config_checks": "PASS",
              "aggregate": aggregate, "runs": details}
    output = Path(mkdtemp(prefix="comparison_", dir=FOLDER / "artifacts")) / "comparison.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
