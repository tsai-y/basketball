"""Micro-batch training and full-distribution evaluation for RGB fine-tuning."""

import math

import torch
from torch import nn

from development.linear_baseline_v1.train_linear import evaluate as score_logits
from development.temporal_baseline_v1.train_temporal import require


def move_batch(batch, device):
    x = batch["tensor"]
    y = torch.as_tensor(batch["binary_label"], dtype=torch.float32)
    ids = list(batch["video_id"])
    require(len(x) > 0 and tuple(y.shape) == (len(x),) and len(ids) == len(x), "Invalid batch lengths")
    require(x.dtype == torch.float32 and torch.isfinite(x).all().item(), "Invalid input tensor")
    require(torch.isfinite(y).all().item() and set(y.tolist()) <= {0., 1.}, "Invalid labels")
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True), ids


def metrics_from_logits(logits, labels):
    logits, labels = logits.detach().cpu().float(), labels.detach().cpu().float()
    require(logits.ndim == 1 and logits.shape == labels.shape and len(logits) > 0,
            "Invalid metric inputs")
    require(torch.isfinite(logits).all().item() and torch.isfinite(labels).all().item(),
            "Non-finite metric inputs")
    require(set(labels.tolist()) <= {0., 1.}, "Invalid metric labels")
    metrics, _ = score_logits(nn.Identity(), logits[:, None], labels)
    return metrics


def train_one_epoch(model, loader, optimizer, *, device, effective_batch_size=64,
                    max_grad_norm=1., log_every=50, progress=print):
    """Each update averages its ACTUAL video count, including the final short group.

    With N=7193, micro-batch=2, effective batch=64: 112 groups of 64 and
    one final group of 25 videos. That last loss is divided by 25, NOT 64.
    """
    total = len(loader.dataset)
    micro = loader.batch_size
    require(total > 0 and type(micro) is int and micro > 0 and not loader.drop_last,
            "Expected nonempty fixed-size micro-batches without drop_last")
    require(type(effective_batch_size) is int and effective_batch_size > 0
            and effective_batch_size % micro == 0, "Effective batch must be a multiple of micro-batch")
    require(math.isfinite(max_grad_norm) and max_grad_norm > 0 and log_every > 0,
            "Invalid clipping/progress settings")
    parameters = [p for p in model.parameters() if p.requires_grad]
    require(len(parameters) > 0, "No trainable parameters")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    seen, pending, updates, loss_sum = 0, 0, 0, 0.
    group_size = 0
    observed_ids = set()
    for batch_index, batch in enumerate(loader, 1):
        x, y, ids = move_batch(batch, device)
        require(len(set(ids)) == len(ids) and observed_ids.isdisjoint(ids), "Duplicate training IDs")
        observed_ids.update(ids)
        if pending == 0:
            group_size = min(effective_batch_size, total - seen)
        require(0 < len(y) <= micro and pending + len(y) <= group_size, "Unexpected micro-batch boundary")
        logits = model(x)
        require(logits.shape == y.shape and torch.isfinite(logits).all().item(), "Invalid training logits")
        summed_loss = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum")
        require(torch.isfinite(summed_loss).item(), "Non-finite training loss")
        (summed_loss / group_size).backward()
        loss_sum += float(summed_loss.detach())
        seen += len(y)
        pending += len(y)
        if pending == group_size:
            for parameter in parameters:
                require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                        "Missing/non-finite trainable gradient")
            nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            pending = 0
        if progress is not None and (batch_index == 1 or batch_index % log_every == 0 or seen == total):
            progress(f"train: videos={seen}/{total}, updates={updates}, loss={loss_sum / seen:.5f}")
    require(seen == total and len(observed_ids) == total and pending == 0, "Incomplete training epoch")
    require(updates == math.ceil(total / effective_batch_size), "Wrong optimizer update count")
    return {"loss": loss_sum / seen, "videos": seen, "optimizer_steps": updates}


@torch.no_grad()
def evaluate_loader(model, loader, *, device, expected_ids=None, log_every=50, progress=print):
    """No weight/BN updates; score all videos together, not batch-average metrics."""
    require(len(loader.dataset) > 0 and not loader.drop_last and log_every > 0,
            "Invalid evaluation loader")
    model.eval()
    outputs, labels, ids = [], [], []
    for batch_index, batch in enumerate(loader, 1):
        x, y, batch_ids = move_batch(batch, device)
        logits = model(x)
        require(logits.shape == y.shape and torch.isfinite(logits).all().item(), "Invalid validation logits")
        outputs.append(logits.detach().cpu().float())
        labels.append(y.detach().cpu().float())
        ids.extend(batch_ids)
        if progress is not None and (batch_index == 1 or batch_index % log_every == 0 or len(ids) == len(loader.dataset)):
            progress(f"validation: videos={len(ids)}/{len(loader.dataset)}")
    require(len(ids) == len(loader.dataset) and len(set(ids)) == len(ids), "Incomplete/duplicate validation IDs")
    if expected_ids is not None:
        require(ids == list(expected_ids), "Validation order/IDs changed")
    logits, y = torch.cat(outputs), torch.cat(labels)
    return metrics_from_logits(logits, y), logits, y, ids
