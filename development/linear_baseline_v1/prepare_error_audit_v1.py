import csv
import hashlib
import json
import random
from pathlib import Path

from development.data_pipeline.pipeline import AccessLedger, SplitGuard


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "development/linear_baseline_v1/artifacts"
RUNS = (
    "linear_pf39g3iq",
    "linear_nrbgq6p1",
    "linear_vmiem11v",
)
SEED = 20261024


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    output = ARTIFACTS / "error_audit_v1"
    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    manifest = {
        row["video_id"]: row
        for row in read_csv(ROOT / "data/processed/dataset_manifest.csv")
        if row["split"] == "validation" and row["excluded"] == "0"
    }

    predictions = []
    configs = []
    input_hashes = {}

    for run in RUNS:
        folder = ARTIFACTS / run
        path = folder / "validation_predictions.csv"
        rows = read_csv(path)
        records = {row["video_id"]: row for row in rows}

        assert len(records) == len(rows), "Duplicate video ID"
        assert set(records) == set(manifest), "Validation IDs differ"
        for video_id, row in records.items():
            assert int(row["true_label"]) == int(
                manifest[video_id]["binary_label"]
            )
            assert int(row["predicted_label"]) in (0, 1)

        predictions.append(records)
        configs.append(json.loads((folder / "config.json").read_text()))
        input_hashes[run] = hashlib.sha256(path.read_bytes()).hexdigest()

    assert [c["seed"] for c in configs] == [20261024, 20261025, 20261026]
    comparable = [
        {k: v for k, v in c.items() if k not in ("seed", "source_sha256")}
        for c in configs
    ]
    assert comparable[0] == comparable[1] == comparable[2]

    pools = {"Miss_to_Make": [], "Make_to_Miss": []}
    for video_id in sorted(manifest):
        truth = int(manifest[video_id]["binary_label"])
        if all(
            int(records[video_id]["predicted_label"]) != truth
            for records in predictions
        ):
            kind = "Miss_to_Make" if truth == 0 else "Make_to_Miss"
            pools[kind].append(video_id)

    rng = random.Random(SEED)
    selected = []
    for kind, pool in pools.items():
        assert len(pool) >= 10, f"Not enough cases: {kind}"
        selected.extend((kind, vid) for vid in rng.sample(pool, 10))
        print(kind, "available:", len(pool), "selected: 10")

    guard = SplitGuard("validation", AccessLedger())
    sheet = []
    for number, (kind, video_id) in enumerate(selected, start=1):
        metadata = manifest[video_id]
        sheet.append({
            "sample_id": f"E{number:03d}",
            "error_type": kind,
            "video_id": video_id,
            "video_path": str(guard.authorize(video_id)),
            "true_label": int(metadata["binary_label"]),
            "source_group": metadata["source_group"],
            "shot_type": metadata["original_class"][:-1],
            "shot_visible": "",
            "result_judgable": "",
            "other_content": "",
            "label_matches_result": "",
            "notes": "",
        })

    output.mkdir(parents=True, exist_ok=False)

    with (output / "audit_sheet.csv").open(
        "x", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)

    with (output / "review.m3u8").open("x", encoding="utf-8") as stream:
        stream.write("#EXTM3U\n")
        for row in sheet:
            stream.write(
                f"#EXTINF:-1,{row['sample_id']} {row['error_type']}\n"
                f"{row['video_path']}\n"
            )

    protocol = {
        "purpose": "descriptive_validation_error_analysis",
        "seed": SEED,
        "prediction_sha256": input_hashes,
        "available_counts": {k: len(v) for k, v in pools.items()},
        "selected": [
            {"sample_id": r["sample_id"], "video_id": r["video_id"],
             "error_type": r["error_type"]}
            for r in sheet
        ],
    }
    with (output / "sampling_manifest.json").open(
        "x", encoding="utf-8"
    ) as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)

    print("Saved:", output)


if __name__ == "__main__":
    main()
