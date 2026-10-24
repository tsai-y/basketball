import math
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import mkdtemp

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from development.linear_baseline_v1.cached_features import ROOT, load_vectors

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001):
        if type(patience) is not int or patience < 1:
            raise ValueError("patience must be a positive integer")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")

        self.patience = patience
        self.min_delta = min_delta
        self.reference_loss = float("inf")
        self.bad_epochs = 0

    def update(self, loss):
        if not math.isfinite(loss):
            raise ValueError("Non-finite validation loss")

        if loss < self.reference_loss - self.min_delta:
            self.reference_loss = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience
@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = model(x).squeeze(-1)
    assert torch.isfinite(logits).all().item()

    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    predictions = (logits >= 0).long()  # 等同 P(Make) >= 0.5。

    # 列是真實類別，欄是預測類別；順序為 Miss、Make。
    cm = torch.bincount(
        2 * y.long() + predictions, minlength=4
    ).reshape(2, 2)

    per_class = {}
    for label, name in enumerate(("Miss", "Make")):
        tp = int(cm[label, label])
        fp = int(cm[:, label].sum()) - tp
        fn = int(cm[label, :].sum()) - tp
        per_class[name] = {
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "support": int(cm[label, :].sum()),
        }

    return {
        "loss": float(loss),
        "accuracy": float(cm.diag().sum()) / len(y),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / 2,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }, logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")

    seed = args.seed
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    cache = ROOT / (
        "development/mixnet_i3d/artifacts/"
        "feature_cache_v1/67b6400e0982bb6d"
    )

    print("Loading full train/validation caches...", flush=True)
    train_x, train_y, train_ids, run_id = load_vectors(cache, split="train")
    val_x, val_y, val_ids, val_run_id = load_vectors(cache, split="validation")

    assert run_id == val_run_id
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_y.tolist()) == {0.0, 1.0}
    assert set(val_y.tolist()) == {0.0, 1.0}
    print("Train:", tuple(train_x.shape), "Validation:", tuple(val_x.shape))

    config = {
        "model": "Linear(1536, 1)",
        "device": "cpu",
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": 128,
        "optimizer": "AdamW",
        "lr": 0.001,
        "weight_decay": 0.0001,
        "loss": "BCEWithLogitsLoss",
        "selection": "minimum_validation_loss",
        "threshold": 0.5,
        "pooling": "equal_mean_over_windows_and_time",
        "cache_root": str(cache),
        "cache_run_id": run_id,
        "train_count": len(train_y),
        "validation_count": len(val_y),
        "torch_version": str(torch.__version__),
        "early_stopping": {
            "monitor": "validation_loss",
            "patience": 10,
            "min_delta": 0.0001,
        },
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("train_linear.py", "cached_features.py")
        },
    }

    parent = Path(__file__).resolve().parent / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="linear_", dir=parent))
    with (output / "config.json").open("x", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)
    print("Output:", output, flush=True)

    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )

    model = nn.Linear(1536, 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    best_loss = float("inf")
    early_stopper = EarlyStopping(
        patience=config["early_stopping"]["patience"],
        min_delta=config["early_stopping"]["min_delta"],
    )
    stop_reason = "max_epochs"
    with (output / "history.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1"
        ])

        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0

            for x, y in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)
                assert torch.isfinite(loss).item(), "Non-finite training loss"
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(y)

            train_loss = loss_sum / len(train_y)
            metrics, _ = evaluate(model, val_x, val_y)

            writer.writerow([
                epoch, train_loss, metrics["loss"],
                metrics["accuracy"], metrics["macro_f1"],
            ])
            stream.flush()

            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": best_loss,
                    "config": config,
                }, output / "best.pt")

            print(
                f"epoch {epoch:02d}: train_loss={train_loss:.4f}, "
                f"val_loss={metrics['loss']:.4f}, "
                f"val_macro_f1={metrics['macro_f1']:.4f}",
                flush=True,
            )

            if early_stopper.update(metrics["loss"]):
                stop_reason = "early_stopping"
                print(
                    f"Early stopping at epoch {epoch}: "
                    f"{early_stopper.bad_epochs} epochs without "
                    "significant validation-loss improvement.",
                    flush=True,
                )
                break

    # 用最低 validation loss 的模型產生最終報告。
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"], strict=True)
    metrics, logits = evaluate(model, val_x, val_y)

    summary = {
        "epochs_completed": epoch,
        "stop_reason": stop_reason,
        "best_epoch": best["epoch"],
        "cache_run_id": run_id,
        "validation": metrics,
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    with (output / "validation_predictions.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "true_label", "p_make", "predicted_label"])
        for video_id, label, logit in zip(val_ids, val_y, logits):
            writer.writerow([
                video_id, int(label),
                float(logit.sigmoid()), int(logit >= 0),
            ])

    print(json.dumps(summary, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
