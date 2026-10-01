#!/usr/bin/env python3
"""One-time A2/B/C full-video test evaluator; --check never decodes test video.

Formal mode reads only each run's frozen best.pt, performs inference without
gradients, and refuses existing test outputs. No training or resume path exists.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.data import BasketballVideoDataset, UniformBinSampler
from baseline.model import C3DStyle
from experiment_c.model import ContinuousModifiedC3D
from reference_c3d.aggregate import aggregate_fc6
from reference_c3d.data import C3DVideoDataset, ContinuousClipSampler
from reference_c3d.model import ReferenceC3D
from scripts.run_c3d_reference_train import LEGACY_SOURCE_SHA256
from experiment_c.aggregate import mean_l2_clip_features


EXPERIMENTS = {
    "a2": ("full_video_c3d_style_extended_v1", "c3d_style_extended.yaml", 21),
    "b": ("full_video_c3d_reference_train_v1", "c3d_reference_train.yaml", 25),
    "c": ("full_video_c3d_style_continuous_v1", "c3d_style_continuous.yaml", 4),
}
TEST_COUNT = 1539
TEST_LABEL_COUNTS = {0: 731, 1: 808}
OUTPUT_FILES = ("test_metrics.json", "test_predictions.csv", ".test_evaluation_claim")


def root_path(value):
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def sha256(file_path):
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def require_equal(actual, expected, description):
    if actual != expected:
        raise RuntimeError(f"{description} mismatch: {actual!r} != {expected!r}")


def load_protocol(name):
    run_name, config_name, expected_epoch = EXPERIMENTS[name]
    run_dir = ROOT / "runs" / run_name
    config_path = ROOT / "configs" / config_name
    if not run_dir.is_dir():
        raise RuntimeError(f"Missing completed run: {run_dir}")
    frozen_config = run_dir / "config.yaml"
    require_equal(sha256(frozen_config), sha256(config_path), f"{name} frozen config")
    config = yaml.safe_load(frozen_config.read_text())
    require_equal(root_path(config["run_dir"]).resolve(), run_dir.resolve(), f"{name} run path")
    manifest = ROOT / "data/processed/dataset_manifest.csv"
    split = ROOT / "data/processed/split_summary.json"
    for label, current, snapshot, pinned in (
        ("manifest", manifest, run_dir / "dataset_manifest.csv",
         config["expected_manifest_sha256"]),
        ("split", split, run_dir / "split_summary.json",
         config["expected_split_summary_sha256"]),
    ):
        require_equal(sha256(current), pinned, f"{name} current {label}")
        require_equal(sha256(snapshot), pinned, f"{name} frozen {label}")
    status = json.loads((run_dir / "training_status.json").read_text())
    if status.get("status") not in {"COMPLETE", "EARLY_STOPPED"}:
        raise RuntimeError(f"{name} training has not completed")
    require_equal(status.get("test_evaluation_performed"), False, f"{name} test status")
    require_equal(status["best_epoch"], expected_epoch, f"{name} status best epoch")
    history = json.loads((run_dir / "training_log.json").read_text())
    epochs = [record["epoch"] for record in history]
    if name == "a2":
        require_equal(epochs, list(range(6, status["last_epoch"] + 1)), "A2 history")
    else:
        require_equal(epochs, list(range(1, status["epochs_completed"] + 1)),
                      f"{name} history")
    best_record = next((record for record in history if record["epoch"] == expected_epoch), None)
    if best_record is None:
        raise RuntimeError(f"{name} best epoch missing from history")
    require_equal(best_record["validation"]["loss"], status["best_validation_loss"],
                  f"{name} logged best validation loss")
    if name == "a2":
        for source, pinned in (("baseline/model.py", config["expected_model_sha256"]),
                               ("baseline/data.py", config["expected_data_sha256"])):
            require_equal(sha256(ROOT / source), pinned, f"A2 source {source}")
    elif name == "b":
        for source, pinned in LEGACY_SOURCE_SHA256.items():
            require_equal(sha256(ROOT / source), pinned, f"B source {source}")
        environment = json.loads((run_dir / "environment.json").read_text())
        require_equal(environment["manifest_sha256"], config["expected_manifest_sha256"],
                      "B environment manifest")
        require_equal(environment["split_summary_sha256"],
                      config["expected_split_summary_sha256"], "B environment split")
        require_equal(status["manifest_sha256"], config["expected_manifest_sha256"],
                      "B status manifest")
    else:
        environment = json.loads((run_dir / "environment.json").read_text())
        for source, key in (("baseline/model.py", "model_source_sha256"),
                            ("reference_c3d/data.py", "data_source_sha256"),
                            ("experiment_c/model.py", "experiment_c_model_sha256"),
                            ("experiment_c/aggregate.py", "experiment_c_aggregate_sha256")):
            require_equal(sha256(ROOT / source), environment[key], f"C source {source}")
        reference = yaml.safe_load(root_path(config["reference_clip_config_path"]).read_text())
        for key in ("clip_length", "clip_stride", "tail_policy", "resize_height",
                    "resize_width", "crop_size", "eval_crop", "pixel_preprocessing"):
            require_equal(config[key], reference[key], f"C vs B preprocessing {key}")
    return {"name": name, "run_dir": run_dir, "config_path": config_path,
            "config": config, "status": status, "best_record": best_record,
            "expected_epoch": expected_epoch}


def test_dataset(protocol):
    """Construct metadata rows only. --check must never call __getitem__."""
    name, config, run_dir = protocol["name"], protocol["config"], protocol["run_dir"]
    manifest = run_dir / "dataset_manifest.csv"
    dataset_path = root_path(config["dataset_path"])
    if name == "a2":
        require_equal((config["num_frames"], config["image_size"], config["num_classes"]),
                      (16, 96, 2), "A2 test shape")
        sampler = UniformBinSampler(config["num_frames"], jitter=False)
        result = BasketballVideoDataset(manifest, dataset_path, "test", sampler,
                                        config["image_size"], flip_probability=0.0)
        assert result.sampler.jitter is False and result.flip_probability == 0.0
        protocol["preprocessing"] = "16 bin centers; 96x96 RGB letterbox; [-1,1]; no flip"
        return result
    if name == "b":
        settings = yaml.safe_load(root_path(config["reference_config_path"]).read_text())
    else:
        settings = config
    require_equal((settings["clip_length"], settings["clip_stride"],
                   settings["tail_policy"], settings["resize_height"],
                   settings["resize_width"], settings["crop_size"],
                   settings["eval_crop"], settings["pixel_preprocessing"]),
                  (16, 8, "discard_incomplete", 128, 171, 112, "center",
                   "rgb_float_minus_one_to_one"), f"{name} test preprocessing")
    sampler = ContinuousClipSampler(settings["clip_length"], settings["clip_stride"],
                                    settings["tail_policy"])
    preprocessing = {"resize_height": settings["resize_height"],
                     "resize_width": settings["resize_width"],
                     "crop_size": settings["crop_size"], "crop_mode": settings["eval_crop"],
                     "flip_probability": 0.0,
                     "pixel_preprocessing": settings["pixel_preprocessing"]}
    result = C3DVideoDataset(manifest, dataset_path, "test", sampler, preprocessing)
    assert result.preprocessing["crop_mode"] == "center"
    assert result.preprocessing["flip_probability"] == 0.0
    protocol["preprocessing"] = ("continuous 16 frames; stride 8; discard tail; "
                                 "128x171 resize; 112 center crop; RGB [-1,1]; no flip")
    return result


def audit_test_identities(protocols):
    all_id_sets = {}
    for name, protocol in protocols.items():
        data = test_dataset(protocol)
        protocol["dataset"] = data
        rows = data.rows  # manifest metadata only; never decode in --check.
        ids = [row["video_id"] for row in rows]
        labels = [int(row["binary_label"]) for row in rows]
        require_equal(len(ids), TEST_COUNT, f"{name} test count")
        require_equal(len(set(ids)), TEST_COUNT, f"{name} unique test IDs")
        require_equal({label: labels.count(label) for label in (0, 1)},
                      TEST_LABEL_COUNTS, f"{name} test label counts")
        if any(row["split"] != "test" or row["excluded"] != "0" for row in rows):
            raise RuntimeError(f"{name} test dataset contains non-test/excluded row")
        if any(row["original_class"] not in {"2p0", "2p1", "3p0", "3p1",
                                                  "ft0", "ft1", "mp0", "mp1"}
               or int(row["original_class"][-1]) != int(row["binary_label"])
               for row in rows):
            raise RuntimeError(f"{name} Make/Miss label mapping differs")
        with (protocol["run_dir"] / "dataset_manifest.csv").open(newline="") as handle:
            other_ids = {row["video_id"] for row in csv.DictReader(handle)
                         if row["split"] in {"train", "validation"} and row["excluded"] == "0"}
        if set(ids) & other_ids:
            raise RuntimeError(f"{name} test IDs overlap train/validation")
        summary = json.loads((protocol["run_dir"] / "split_summary.json").read_text())
        require_equal(summary["label_counts"]["test"], {"0": 731, "1": 808},
                      f"{name} split-summary test counts")
        all_id_sets[name] = set(ids)
        protocol["test_id_sha256"] = hashlib.sha256(
            "\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    if not (all_id_sets["a2"] == all_id_sets["b"] == all_id_sets["c"]):
        raise RuntimeError("A2/B/C resolve to different test video IDs")
    return next(iter(protocols.values()))["test_id_sha256"]


def load_best_model(protocol):
    """Strictly load best.pt on CPU; never inspect last.pt."""
    name, config, run_dir = protocol["name"], protocol["config"], protocol["run_dir"]
    best_path = run_dir / "best.pt"
    if not best_path.is_file() or best_path.is_symlink():
        raise RuntimeError(f"{name} frozen best.pt missing or symlinked")
    checkpoint_hash = sha256(best_path)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
    require_equal(checkpoint["epoch"], protocol["expected_epoch"], f"{name} best.pt epoch")
    require_equal(checkpoint["epoch"], protocol["status"]["best_epoch"],
                  f"{name} best.pt vs status epoch")
    best_loss = protocol["status"]["best_validation_loss"]
    if name == "a2":
        require_equal(checkpoint["validation_loss"], best_loss, "A2 best.pt loss")
        require_equal(checkpoint["manifest_sha256"], config["expected_manifest_sha256"],
                      "A2 best.pt manifest")
        require_equal(checkpoint["protocol_sha256"], protocol["status"]["protocol_sha256"],
                      "A2 best.pt protocol")
        require_equal(checkpoint["a1_checkpoint_sha256"],
                      config["expected_a1_checkpoint_sha256"], "A2 initialization identity")
        model = C3DStyle(config["num_classes"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        head = None
    elif name == "b":
        require_equal(checkpoint["best_epoch"], protocol["expected_epoch"], "B best.pt best epoch")
        require_equal(checkpoint["best_loss"], best_loss, "B best.pt loss")
        require_equal(checkpoint["protocol_sha256"], protocol["status"]["protocol_sha256"],
                      "B best.pt protocol")
        settings = yaml.safe_load(root_path(config["reference_config_path"]).read_text())
        model = ReferenceC3D(settings["num_classes"], settings["pool_ceil_mode"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        head = nn.Linear(4096, 2)
        head.load_state_dict(checkpoint["head_state"], strict=True)
    else:
        require_equal(checkpoint["best_epoch"], protocol["expected_epoch"], "C best.pt best epoch")
        require_equal(checkpoint["best_validation_loss"], best_loss, "C best.pt loss")
        require_equal(checkpoint["manifest_sha256"], config["expected_manifest_sha256"],
                      "C best.pt manifest")
        require_equal(checkpoint["config_sha256"], sha256(run_dir / "config.yaml"),
                      "C best.pt config")
        model = ContinuousModifiedC3D()
        model.load_state_dict(checkpoint["model"], strict=True)
        head = None
    protocol["checkpoint_sha256"] = checkpoint_hash
    del checkpoint
    return model, head


def check_output_guard(protocol):
    run_dir = protocol["run_dir"]
    collisions = [name for name in OUTPUT_FILES
                  if (run_dir / name).exists() or (run_dir / name).is_symlink()]
    if collisions:
        raise FileExistsError(f"{protocol['name']} test output/claim already exists: {collisions}")


def single_video_collate(batch):
    if len(batch) != 1:
        raise ValueError("Continuous-clip test loader requires one video per item")
    return batch[0]


def device_for(protocol):
    requested = protocol["config"]["device"]
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen protocol requests CUDA but CUDA is unavailable")
    return device


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def metrics_from_predictions(loss_sum, predictions, protocol):
    confusion = [[0, 0], [0, 0]]
    for row in predictions:
        confusion[row["true_label"]][row["predicted_label"]] += 1
    tn, fp = confusion[0]
    fn, tp = confusion[1]
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    result = {
        "experiment": protocol["name"], "checkpoint": "best.pt",
        "best_epoch": protocol["expected_epoch"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "manifest_sha256": protocol["config"]["expected_manifest_sha256"],
        "split_summary_sha256": protocol["config"]["expected_split_summary_sha256"],
        "test_id_sha256": protocol["test_id_sha256"],
        "positive_class": "Make=1", "prediction_rule": "argmax(logits)",
        "test_loss": loss_sum / len(predictions),
        "accuracy": safe_ratio(tn + tp, len(predictions)),
        "precision_make": precision, "recall_make": recall,
        "f1_make": safe_ratio(2 * precision * recall, precision + recall),
        "confusion_matrix_rows_true_miss_make": confusion,
        "total_test_count": len(predictions),
        "true_miss_count": tn + fp, "true_make_count": fn + tp,
        "predicted_miss_count": tn + fn, "predicted_make_count": fp + tp,
    }
    require_equal(result["total_test_count"], TEST_COUNT, "evaluated test count")
    require_equal((result["true_miss_count"], result["true_make_count"]),
                  (TEST_LABEL_COUNTS[0], TEST_LABEL_COUNTS[1]), "evaluated label counts")
    return result


def formal_evaluate(protocol, model, head):
    """This is called only without --check, after the human-approved command."""
    check_output_guard(protocol)
    run_dir = protocol["run_dir"]
    device = device_for(protocol)
    model = model.to(device).eval()
    if head is not None:
        head = head.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if head is not None:
        for parameter in head.parameters():
            parameter.requires_grad_(False)
    data = protocol["dataset"]
    name = protocol["name"]
    if name == "a2":
        loader = DataLoader(data, batch_size=protocol["config"]["batch_size"],
                            shuffle=False, num_workers=protocol["config"]["num_workers"],
                            pin_memory=device.type == "cuda")
    else:
        loader = DataLoader(data, batch_size=1, shuffle=False,
                            num_workers=protocol["config"]["num_workers"],
                            collate_fn=single_video_collate,
                            pin_memory=device.type == "cuda")
    claim_path = run_dir / ".test_evaluation_claim"
    with os.fdopen(os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w") as claim:
        claim.write(json.dumps({"experiment": protocol["name"],
                                "checkpoint_sha256": protocol["checkpoint_sha256"]}) + "\n")
    predictions = []
    loss_sum = 0.0
    with torch.inference_mode():
        for sample in loader:
            if name == "a2":
                videos, labels, video_ids = sample
                logits = model(videos.to(device, non_blocking=True))
                labels = labels.to(device, non_blocking=True)
                loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())
                probabilities = F.softmax(logits, dim=1)[:, 1]
                predicted = logits.argmax(dim=1)
                for index, video_id in enumerate(video_ids):
                    predictions.append({"video_id": video_id,
                                        "true_label": int(labels[index].item()),
                                        "predicted_label": int(predicted[index].item()),
                                        "probability_make": float(probabilities[index].item())})
            else:
                clips = sample["clips"]
                pieces = []
                for chunk in clips.split(protocol["config"]["clip_batch_size"]):
                    chunk = chunk.to(device, non_blocking=True)
                    pieces.append(model.extract_fc6(chunk) if name == "b"
                                  else model.extract_clip_features(chunk))
                features = torch.cat(pieces, dim=0)
                if name == "b":
                    logits = head(aggregate_fc6(features))
                else:
                    logits = model.classifier(mean_l2_clip_features(features))
                label = torch.tensor([sample["label"]], dtype=torch.long, device=device)
                loss_sum += float(F.cross_entropy(logits.unsqueeze(0), label).item())
                predictions.append({"video_id": sample["video_id"],
                                    "true_label": sample["label"],
                                    "predicted_label": int(logits.argmax().item()),
                                    "probability_make": float(F.softmax(logits, dim=0)[1].item())})
    metrics = metrics_from_predictions(loss_sum, predictions, protocol)
    # Exclusive creation prevents replacing an output even if another writer
    # raced with preflight. A leftover claim/partial output requires review.
    with (run_dir / "test_predictions.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=("video_id", "true_label", "predicted_label",
                                            "probability_make"))
        writer.writeheader()
        writer.writerows(predictions)
    with (run_dir / "test_metrics.json").open("x") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS,
                        help="Required for formal evaluation; optional with --check")
    parser.add_argument("--check", action="store_true",
                        help="Read-only metadata/model audit; never decode test video")
    args = parser.parse_args()
    if not args.check and not args.experiment:
        parser.error("Formal evaluation requires --experiment a2, b, or c")
    protocols = {name: load_protocol(name) for name in EXPERIMENTS}
    test_id_hash = audit_test_identities(protocols)
    report = {}
    selected_model = selected_head = None
    for name, protocol in protocols.items():
        model, head = load_best_model(protocol)
        if name == args.experiment and not args.check:
            selected_model, selected_head = model, head
        report[name] = {"best_epoch": protocol["expected_epoch"],
                        "best_checkpoint_sha256": protocol["checkpoint_sha256"],
                        "model_state_compatible": True,
                        "test_count": len(protocol["dataset"].rows),
                        "preprocessing": protocol["preprocessing"],
                        "test_output_exists": any((protocol["run_dir"] / output).exists()
                                                  or (protocol["run_dir"] / output).is_symlink()
                                                  for output in OUTPUT_FILES)}
    if args.check:
        if args.experiment:
            check_output_guard(protocols[args.experiment])
        else:
            for protocol in protocols.values():
                check_output_guard(protocol)
        print(json.dumps({"status": "PASS", "same_test_identities": True,
                          "test_count": TEST_COUNT, "test_id_sha256": test_id_hash,
                          "test_video_decoded": False, "test_metrics_computed": False,
                          "output_files_written": False, "experiments": report}, indent=2))
        return
    selected = protocols[args.experiment]
    check_output_guard(selected)
    result = formal_evaluate(selected, selected_model, selected_head)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
