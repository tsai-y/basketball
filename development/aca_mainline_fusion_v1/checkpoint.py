"""Export verified saved validation logits to stdout; never run a model/GPU."""
import json
from pathlib import Path

import torch

from development.aca_net_v1.data import VideoScope, file_sha256, require
from development.aca_validation_v1.analyze import read_json, verify_protected

MAIN = Path("/home/tsai/projects/basketball")
RUN = MAIN / "development/partial_finetune_v1/artifacts/rgb_seed20261024_bkegdbsu"
ACA = Path(__file__).resolve().parents[2] / "development/aca_validation_v1/artifacts/validation_compare_vykz5xz0"


def json_equivalent(checkpoint_config, saved_json):
    # JSON object keys are strings; torch.save preserves integer label keys.
    # Normalize exactly the original JSON serialization, never drop a field.
    return json.loads(json.dumps(checkpoint_config, allow_nan=False)) == saved_json


def csv_probabilities(logits):
    # Original train_rgb CSV writer uses scalar logit.sigmoid(), not the
    # vectorized CPU kernel (which can differ by FP32 ULPs).
    return [logit.sigmoid().item() for logit in logits]


def inspect():
    require(not torch.cuda.is_initialized(), "CPU checkpoint inspection only")
    torch.set_num_threads(2)
    config, summary = read_json(RUN / "config.json"), read_json(RUN / "summary.json")
    require(config["schema_version"] == "partial_rgb_training_v1" and config["seed"] == 20261024 and
            config["test_used"] is False and summary["test_used"] is False and
            summary["stop_reason"] == "early_stopping", "Wrong/uncompleted mainline run")
    require(config["selection"] == "minimum_validation_loss_including_epoch_0" and
            config["threshold"] == 0.5 and summary["best_epoch"] == 0 and
            summary["selected_initial_weights"] is True, "Unexpected selection protocol")
    scope = VideoScope(source_root=MAIN, split="validation")
    require(scope.hashes == {"manifest": config["input_identities"]["manifest_sha256"],
                             "split_summary": config["input_identities"]["split_summary_sha256"]},
            "Mainline/ACA frozen split mismatch")
    files = {str(RUN / name): file_sha256(RUN / name) for name in
             ("best.pt", "last.pt", "config.json", "history.csv", "summary.json", "validation_predictions.csv")}
    for relative, expected in config["source_sha256"].items():
        path = MAIN / relative
        require(file_sha256(path) == expected, f"Mainline source changed since training: {relative}")
        files[str(path)] = expected
    previous = read_json(ACA / "report.json")
    require(previous["status"] == "completed" and previous["test_videos_opened"] == 0 and
            previous["protected_files_unchanged"], "ACA comparison was not successfully completed")
    provenance = read_json(ACA / "provenance.json")
    verify_protected(provenance["protected_files_sha256"])
    files.update(provenance["protected_files_sha256"])
    for path in ACA.iterdir():
        require(path.is_file() and not path.is_symlink(), "Unexpected ACA output entry")
        files[str(path)] = file_sha256(path)
    checkpoint = torch.load(RUN / "best.pt", map_location="cpu", weights_only=True)
    require(checkpoint["engineering_only"] is False and checkpoint["schema_version"] == config["schema_version"] and
            json_equivalent(checkpoint["config"], config) and checkpoint["epoch"] == summary["best_epoch"] and
            checkpoint["validation"] == summary["validation"], "Best checkpoint identity/summary mismatch")
    ids, labels, logits = (checkpoint[k] for k in ("validation_video_ids", "validation_labels", "validation_logits"))
    expected = {row["video_id"]: row for row in scope.rows}
    require(len(ids) == len(set(ids)) == len(expected) == 1547 and set(ids) == set(expected),
            "Checkpoint has duplicate/missing/foreign validation IDs")
    require(tuple(labels.shape) == tuple(logits.shape) == (1547,) and logits.dtype == torch.float32 and
            torch.isfinite(logits).all().item(), "Invalid checkpoint validation tensors")
    y = torch.tensor([int(expected[i]["binary_label"]) for i in ids], dtype=torch.float32)
    require(torch.equal(labels, y), "Checkpoint labels differ from fixed manifest")
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    require(abs(ce.mean().item() - summary["validation"]["loss"]) < 1e-6, "Saved-logit loss mismatch")
    probs = csv_probabilities(logits)
    log_probs = torch.nn.functional.logsigmoid(torch.stack((-logits, logits), dim=-1).double())
    rows = []
    for i, video_id in enumerate(ids):
        row = expected[video_id]
        rows.append({"video_id": video_id, "split": "validation", "label": int(row["binary_label"]),
                     "original_class": row["original_class"], "shot_type": row["original_class"][:-1],
                     "source_group": row["source_group"], "logit": logits[i].item(),
                     "p_make": probs[i], "prediction": int(logits[i] >= 0),
                     "ce_loss": ce[i].item(), "log_probabilities": log_probs[i].tolist()})
    for field in ("extractor", "head"):
        init = config["initialization"]
        path = Path(init[f"{field}_path"])
        require(file_sha256(path) == init[f"{field}_sha256"], "Mainline initialization source changed")
        files[str(path)] = init[f"{field}_sha256"]
    require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
    return {"mainline_run": str(RUN), "aca_comparison": str(ACA), "config": config, "summary": summary,
            "rows": rows, "expected_validation_rows": list(scope.rows), "protected_files_sha256": files,
            "checkpoint_validation_verified": True, "inference_executed": False, "test_videos_opened": 0}


if __name__ == "__main__":
    print(json.dumps(inspect(), ensure_ascii=False, allow_nan=False))
