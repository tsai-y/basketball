"""No training/test entry point. Existing experiment files are read-only."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import torch

from development.aca_net_v1.data import Basketball50Dataset, POLICY, ROOT, VideoScope, file_sha256, require
from development.aca_net_v1.model import VideoResNet50
from development.aca_net_v1.modules import ReconstructionConfig
from development.aca_net_v1.train import Metrics, canonical_hash, configure_runtime, make_loader
from development.aca_dw3d_v1.model import DW3DConfig, DW3DVideoResNet50

FOLDER = Path(__file__).resolve().parent
RUNS = {
    "dense": ROOT / "development/aca_extension_v1/artifacts/aca_extend15_seed20261025_wuaps2g9",
    "dw3d": ROOT / "development/aca_dw3d_v1/artifacts/aca_dw3d_train_seed20261025_zmjmo2o1",
}
CONTRACTS = {
    "dense": ("aca_dense_candidate_resume_v1", "formal_aca_candidate", 8, 93163870),
    "dw3d": ("aca_dw3d_candidate_resume_v1", "formal_aca_dw3d_candidate", 4, 25464286),
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def metadata(name, run):
    plan, summary, history = [read_json(run / f"{n}.json") for n in ("config", "summary", "history")]
    schema, kind, expected_epoch, parameters = CONTRACTS[name]
    require(plan["schema"] == schema and plan["expected_parameters"] == parameters,
            "Wrong model identity")
    require(summary["status"] == "completed" and summary["test_videos_opened"] == 0,
            "Source run must be completed without test evaluation")
    best = min(history, key=lambda row: row["validation"]["loss"])
    require(summary["best_epoch"] == best["epoch"] == expected_epoch and
            summary["best_validation"] == best["validation"] and best["selected_as_best"],
            "Selected epoch must be the historical strict min-CE checkpoint")
    require(summary["epochs"] == len(history) and summary["optimizer_steps"] == 450 * len(history),
            "Source summary/history inconsistency")
    require(canonical_hash(plan["policy"]) == canonical_hash(POLICY), "Changed preprocessing")
    require(plan["validation_count"] == 1547 and plan["config"]["micro_batch"] == 1,
            "Changed validation scope/batching")
    return {"plan": plan, "summary": summary, "history": history,
            "checkpoint": str(run / "best.pt"), "schema": schema, "kind": kind}


def protected_files(specs):
    paths = set()
    for name, spec in specs.items():
        paths.update(RUNS[name] / f for f in ("config.json", "summary.json", "history.json", "best.pt", "last.pt"))
        for relative, expected in spec["plan"]["source_code_sha256"].items():
            path = ROOT / relative
            require(file_sha256(path) == expected, f"Source changed since training: {relative}")
            paths.add(path)
    paths.update(FOLDER.glob("*.py"))
    return {str(path): file_sha256(path) for path in sorted(paths)}


def verify_protected(expected):
    require(all(file_sha256(Path(path)) == digest for path, digest in expected.items()),
            "A protected source/checkpoint/run file changed during analysis")


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(cpu.dtype), tuple(cpu.shape))).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def validate_payload(value, spec):
    plan, summary = spec["plan"], spec["summary"]
    require(value.get("schema") == spec["schema"] and value.get("kind") == spec["kind"] and
            value.get("engineering_only") is False, "Wrong/engineering checkpoint refused")
    require(value.get("plan_sha256") == canonical_hash(plan), "Checkpoint/plan hash mismatch")
    progress = value["progress"]
    epoch = summary["best_epoch"]
    require(progress["epoch"] == epoch and progress["cursor"] == 0 and
            progress["optimizer_steps"] == 450 * epoch and
            progress["history"] == spec["history"][:epoch] and
            progress["best_loss"] == summary["best_validation"]["loss"],
            "Not the selected best checkpoint or invalid checkpoint progress")
    require(all(isinstance(t, torch.Tensor) and (not t.is_floating_point() or
                (t.dtype == torch.float32 and torch.isfinite(t).all().item()))
                for t in value["model"].values()), "Nonfinite or non-FP32 model state")


def load_model(name, spec):
    value = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    validate_payload(value, spec)
    architecture = spec["plan"]["architecture"]
    if name == "dense":
        model = VideoResNet50(variant="aca", num_classes=2, config=ReconstructionConfig(**architecture))
    else:
        model = DW3DVideoResNet50(num_classes=2, config=DW3DConfig(**architecture))
    require(sum(p.numel() for p in model.parameters()) == spec["plan"]["expected_parameters"],
            "Constructed architecture parameter mismatch")
    model.load_state_dict(value["model"], strict=True)
    before = tensor_digest(model.state_dict())
    del value
    gc.collect()
    return model.requires_grad_(False).eval(), before


def summarize(rows):
    require(bool(rows), "Empty prediction set")
    matrix, shot_matrices = [[0, 0], [0, 0]], {}
    for row in rows:
        y, pred = row["label"], row["prediction"]
        require(type(y) is int and y in (0, 1) and type(pred) is int and pred in (0, 1), "Invalid label")
        require(math.isfinite(row["ce_loss"]) and row["ce_loss"] >= 0 and
                math.isfinite(row["p_make"]) and 0 <= row["p_make"] <= 1, "Invalid probability/loss")
        matrix[y][pred] += 1
        shot_matrices.setdefault(row["shot_type"], [[0, 0], [0, 0]])[y][pred] += 1
    return {**Metrics.from_confusion(matrix), "loss": math.fsum(r["ce_loss"] for r in rows) / len(rows),
            "by_shot_type": {shot: Metrics.from_confusion(m) for shot, m in sorted(shot_matrices.items())}}


def match_metrics(actual, expected):
    for key in ("count", "confusion_rows_true_columns_predicted", "by_shot_type"):
        require(actual[key] == expected[key], f"Validation reproduction failed: {key}")
    for key in ("accuracy", "macro_f1", "miss_recall", "make_recall", "loss"):
        require(abs(actual[key] - expected[key]) <= (1e-6 if key == "loss" else 1e-12),
                f"Validation reproduction failed: {key}: {actual[key]} != {expected[key]}")


def align(left, right, expected):
    def index(rows):
        require(len(rows) == len(expected), "Incomplete predictions")
        ids = [row["video_id"] for row in rows]
        require(len(ids) == len(set(ids)) and set(ids) == {r["video_id"] for r in expected},
                "Duplicate, missing, extra or non-validation IDs")
        return {row["video_id"]: row for row in rows}
    a, b = index(left), index(right)
    result = []
    for source in expected:
        pair = a[source["video_id"]], b[source["video_id"]]
        for row in pair:
            require(row["split"] == "validation" and row["label"] == int(source["binary_label"]) and
                    row["original_class"] == source["original_class"] and
                    row["shot_type"] == source["original_class"][:-1] and
                    row["source_group"] == source["source_group"], "Prediction metadata/split mismatch")
        result.append(pair)
    return result


def compare(left, right, expected):
    # Validate both inputs before any probability arithmetic.
    dense_metrics, dw3d_metrics = summarize(left), summarize(right)
    pairs = align(left, right, expected)
    counts = {"both_correct": 0, "dense_only_correct": 0, "dw3d_only_correct": 0, "both_wrong": 0}
    changed = {name: {"rescued": 0, "harmed": 0} for name in ("dense", "dw3d")}
    fused = []
    for a, b in pairs:
        p = (a["p_make"] + b["p_make"]) / 2
        # Argmax convention: exact probability ties choose Miss (class 0).
        prediction = int(p > 0.5)
        if "log_probabilities" in a and "log_probabilities" in b:
            la, lb = a["log_probabilities"][a["label"]], b["log_probabilities"][b["label"]]
            high = max(la, lb)
            loss = -(high + math.log(math.exp(la-high) + math.exp(lb-high)) - math.log(2))
        else:
            target_p = p if a["label"] else 1 - p
            loss = -math.log(max(target_p, 1e-300))
        row = {k: a[k] for k in ("video_id", "split", "label", "original_class", "shot_type", "source_group")}
        row.update(p_make=p, prediction=prediction, ce_loss=loss,
                   dense_p_make=a["p_make"], dw3d_p_make=b["p_make"], correct=prediction == a["label"])
        fused.append(row)
        ca, cb = a["prediction"] == a["label"], b["prediction"] == b["label"]
        key = "both_correct" if ca and cb else "dense_only_correct" if ca else "dw3d_only_correct" if cb else "both_wrong"
        counts[key] += 1
        for name, correct in (("dense", ca), ("dw3d", cb)):
            changed[name]["rescued"] += int(not correct and row["correct"])
            changed[name]["harmed"] += int(correct and not row["correct"])
    report = {"scope": "validation_only_not_test", "weights": {"dense": 0.5, "dw3d": 0.5},
              "weight_or_threshold_search": False, "threshold": "p_make > 0.5; exact ties Miss (class 0)",
              "dense": dense_metrics, "dw3d": dw3d_metrics, "equal_weight_fusion": summarize(fused),
              "complementarity": counts, "changes_vs_single": changed,
              "label_oracle_ceiling": {"accuracy": 1 - counts["both_wrong"] / len(pairs),
                                       "deployable": False,
                                       "meaning": "Diagnostic only: pick a correct model using ground-truth; not an achieved score"},
              "test_videos_opened": 0, "optimizer_updates": 0,
              "caveat": "Same validation reused for development; no test90 guarantee or statistical superiority claim"}
    return report, fused


def write_rows(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def infer(name, spec, dataset, output, *, device="cuda"):
    started = time.perf_counter()
    model, before = load_model(name, spec)
    model.to(device)
    cfg = SimpleNamespace(**spec["plan"]["config"])
    loader = make_loader(dataset, list(range(len(dataset))), cfg, spec["summary"]["best_epoch"] - 1)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["tensor"].to(device, non_blocking=True))
            require(tuple(logits.shape) == (1, 2) and torch.isfinite(logits).all().item(), "Invalid output")
            loss = torch.nn.functional.cross_entropy(logits, batch["binary_label"].to(device))
            p = logits.double().softmax(dim=-1)[0, 1].item()
            source = dataset.scope.rows[len(rows)]
            require(batch["video_id"][0] == source["video_id"] and batch["split"][0] == "validation",
                    "Loader order/split mismatch")
            row = {"video_id": source["video_id"], "split": "validation", "label": int(source["binary_label"]),
                   "original_class": source["original_class"], "shot_type": source["original_class"][:-1],
                   "source_group": source["source_group"], "logits": logits[0].cpu().tolist(),
                   "log_probabilities": logits.double().log_softmax(-1)[0].cpu().tolist(),
                   "p_make": p, "prediction": logits.argmax(-1).item(), "ce_loss": loss.item()}
            row["correct"] = row["prediction"] == row["label"]
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(dataset):
                print(f"validation {name} videos={len(rows)}/{len(dataset)} seconds={time.perf_counter()-started:.1f}", flush=True)
    require(len(rows) == len(dataset) and all(not m.training for m in model.modules()) and
            all(p.grad is None and not p.requires_grad for p in model.parameters()), "Inference-only contract failed")
    require(tensor_digest(model.state_dict()) == before, "Model/BN buffers changed during inference")
    metrics = summarize(rows)
    match_metrics(metrics, spec["summary"]["best_validation"])
    write_rows(output / f"{name}_predictions.jsonl", rows)
    write_new(output / f"{name}_verification.json", {"status": "passed", "reproduced_best_epoch": spec["summary"]["best_epoch"],
               "metrics": metrics, "seconds": time.perf_counter()-started,
               "model_and_buffers_unchanged": True, "model_state_sha256": before,
               "optimizer_updates": 0, "test_videos_opened": 0})
    del model, loader
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def run(source_root, execute):
    scope = VideoScope(source_root=source_root, split="validation")
    specs = {name: metadata(name, run) for name, run in RUNS.items()}
    for spec in specs.values():
        require(spec["plan"]["input_hashes"] == scope.hashes and
                spec["plan"]["validation_ordered_ids_sha256"] == canonical_hash([r["video_id"] for r in scope.rows]),
                "Validation manifest/order mismatch")
        require(str(Path(source_root).resolve()) == spec["plan"]["source_root"], "Wrong source root")
    if not execute:
        print(json.dumps({"mode": "metadata_only_no_video_no_GPU", "validation_count": len(scope.rows),
                          "runs": {n: s["checkpoint"] for n, s in specs.items()},
                          "fusion": "fixed 50/50; no weight/threshold search"}, indent=2))
        return
    protected = protected_files(specs)
    configure_runtime(20261025)
    artifacts = FOLDER / "artifacts"
    artifacts.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="validation_compare_", dir=artifacts))
    print(f"OUTPUT={output}", flush=True)
    write_new(output / "provenance.json", {"protected_files_sha256": protected,
               "validation_ids_sha256": canonical_hash([r["video_id"] for r in scope.rows]),
               "input_hashes": scope.hashes, "policy": POLICY, "fusion_dense_weight": 0.5,
               "fusion_choice_predeclared_before_inference": True, "device": torch.cuda.get_device_name(),
               "precision": "FP32_no_AMP_no_TF32", "test_videos_opened": 0})
    try:
        dataset = Basketball50Dataset(source_root=source_root, split="validation")
        predictions = {name: infer(name, spec, dataset, output) for name, spec in specs.items()}
        report, rows = compare(predictions["dense"], predictions["dw3d"], scope.rows)
        verify_protected(protected)
        write_rows(output / "fusion_predictions.jsonl", rows)
        write_new(output / "report.json", {"status": "completed", "protected_files_unchanged": True, **report})
        print(json.dumps({"status": "completed", "output": str(output), **report}, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        write_new(output / "failure.json", {"status": "failed", "error": repr(exc),
                                          "no_automatic_retry": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--execute-validation", action="store_true", help="Explicitly authorize GPU validation-only inference")
    args = parser.parse_args()
    run(args.source_root, args.execute_validation)


if __name__ == "__main__":
    main()
