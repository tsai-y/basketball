"""Bounded four-train/four-validation real-video pipeline verification.

--dry-run reads manifest metadata only. --run opens exactly eight videos once
each, writes diagnostic sheets and integrity reports under a NEW module run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import torch

from .audit_architecture import run as architecture_audit
from .data import (Basketball50Dataset, POLICY, VideoScope, file_sha256, guarded_source_sha256,
                   preview_rgb, require, select_smoke_ids)
from .model import VideoResNet50

ARTIFACTS = Path(__file__).parent / "artifacts"


def write_json(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def contact_sheet(sample, output):
    rgb = preview_rgb(sample["tensor"])
    positions = [i * 49 // 7 for i in range(8)]
    canvas = np.full((548, 1280, 3), 24, dtype=np.uint8)
    for cell, position in enumerate(positions):
        row, col = divmod(cell, 4)
        x, y = col * 320, row * 274
        bgr = cv2.cvtColor(rgb[position], cv2.COLOR_RGB2BGR)
        canvas[y + 34:y + 274, x:x + 320] = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_NEAREST)
        source_frame = sample["metadata"]["selected_indices"][position]
        label = f'input t={position:02d} / source f={source_frame}'
        cv2.putText(canvas, label, (x + 5, y + 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
    require(cv2.imwrite(str(output), canvas), "Cannot write contact sheet")


def execute(source_root, *, dry_run):
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    scopes = {s: VideoScope(source_root=source_root, split=s) for s in ("train", "validation")}
    ids = {s: select_smoke_ids(scope) for s, scope in scopes.items()}
    plan = {"policy": POLICY, "input_hashes": scopes["train"].hashes,
            "fixed_split_counts": scopes["train"].counts, "selected_video_ids": ids,
            "selection": "first_sorted_2p0_2p1_ft0_ft1_each_from_distinct_source_group"}
    if dry_run:
        return {**plan, "status": "dry_run_passed", "videos_opened": 0, "files_written": False}
    ARTIFACTS.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="pipeline_smoke_", dir=ARTIFACTS))
    print(f"OUTPUT: {output}", flush=True)
    write_json(output / "config.json", {**plan, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_root": str(Path(source_root).absolute()), "torch": torch.__version__,
               "opencv": cv2.__version__, "numpy": np.__version__,
               "source_sha256": {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")}})
    write_json(output / "architecture_audit.json", architecture_audit())
    records, cards = [], []
    torch.manual_seed(20261025)
    baseline = VideoResNet50(num_classes=2).eval()
    for split in ("train", "validation"):
        dataset = Basketball50Dataset(source_root=source_root, split=split, video_ids=ids[split])
        for i in range(len(dataset)):
            row, path = dataset.scope.authorize(dataset.scope.rows[i]["video_id"])
            before_hash = guarded_source_sha256(dataset.scope, row["video_id"])
            start = time.perf_counter()
            sample = dataset[i]
            seconds = time.perf_counter() - start
            require(guarded_source_sha256(dataset.scope, row["video_id"]) == before_hash,
                    "Source bytes changed during smoke")
            tensor = sample["tensor"]
            shape_forward = None
            if i == 0:  # One per split: numerical integration only, random weights.
                with torch.inference_mode():
                    logits = baseline(tensor.unsqueeze(0))
                require(tuple(logits.shape) == (1, 2) and torch.isfinite(logits).all(),
                        "Baseline integration failed")
                shape_forward = {"output_shape": list(logits.shape), "finite": True,
                                 "weights": "random_not_ImageNet", "accuracy_claim_allowed": False}
            filename = f'{split}_{i + 1:02d}.png'
            contact_sheet(sample, output / filename)
            record = {k: v for k, v in sample.items() if k != "tensor"}
            record.update(tensor_shape=list(tensor.shape), tensor_dtype=str(tensor.dtype),
                          tensor_min=float(tensor.min()), tensor_max=float(tensor.max()),
                          tensor_sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                          source_sha256=before_hash, source_bytes_unchanged=True,
                          pipeline_seconds=seconds, panel=filename, baseline_integration=shape_forward)
            records.append(record)
            caption = html.escape(f'{split} / {row["source_group"]} / {row["original_class"]} / {sample["video_id"]}')
            cards.append(f'<section><h2>{caption}</h2><img src="{filename}"></section>')
    for scope in scopes.values():
        require(scope.hashes == VideoScope(source_root=source_root, split=scope.split).hashes,
                "Manifest/split changed during smoke")
    report = {"status": "completed", "records": records, "videos_opened": len(records),
              "video_opens_by_split": {s: len(ids[s]) for s in ids},
              "original_video_decode_passes": sum(r["metadata"]["video_open_count"] for r in records),
              "source_hash_checks_are_additional_readonly_byte_reads": True,
              "source_byte_hash_opens": 2 * len(records),
              "test_videos_opened": 0, "pretrained_weights_loaded": False,
              "optimizer_steps": 0, "classification_metrics_computed": False,
              "manual_event_visibility_review": "pending", "output": str(output)}
    write_json(output / "report.json", report)
    with (output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write('<!doctype html><meta charset="utf-8"><title>ACA 50幀資料管線驗收</title>'
                     '<style>body{font:16px system-ui;max-width:1340px;margin:24px auto;padding:16px}'
                     'img{max-width:100%}section{margin:30px 0}</style>'
                     '<h1>整支影片均勻取50幀：實際模型輸入</h1>'
                     '<p>每列8張是50幀輸入的稀疏預覽，保留原畫面後縮至160×120；'
                     '展示時最近鄰放大2倍。這不是完整連續播放，也不是標籤重判。</p>'
                     '<p>兩個樣本做隨機權重ResNet50的數值整合驗收；沒有訓練或準確率結果。</p>'
                     + ''.join(cards))
    return {k: v for k, v in report.items() if k != "records"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(execute(args.source_root, dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
