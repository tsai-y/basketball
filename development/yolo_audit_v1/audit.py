"""Train-only, fixed 20-video YOLO11m localization/tracking audit.

Inference/visualization only. No training, ROI-based model input, video removal,
quality weighting, or held-out video access. Human judgments remain blank.
"""

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import html
import json
import math
import os
from pathlib import Path
from tempfile import mkdtemp
from types import SimpleNamespace
import subprocess

import cv2
import numpy as np
import torch

from development.data_pipeline.pipeline import (
    AccessLedger, SplitGuard, TrainQualityIndex, load_gate_config,
)


ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
SEED = 20261024
WEIGHT = ROOT / "yolo11m.pt"
WEIGHT_SHA = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"
CLASSES = ("2p0", "2p1", "3p0", "3p1", "ft0", "ft1", "mp0", "mp1")
EXTRAS = (("2p0", True), ("3p1", False), ("ft0", False), ("mp1", True))
VIDEO_HUMAN = ["shot_event_visible", "pre_release_shooter_missed", "shooter_dropout",
               "non_player_false_boxes", "id_stability", "cut_or_replay",
               "outcome_visible", "roi_would_lose_task_info", "overall_suitability",
               "first_issue_frame", "notes"]
FRAME_HUMAN = ["matched_player_boxes", "non_player_or_duplicate_boxes", "uncertain_boxes",
               "missed_visible_players", "shooter_visible", "shooter_detected",
               "ball_visible", "ball_in_roi", "rim_visible", "rim_in_roi", "notes"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_samples(rows, summaries, *, seed=SEED):
    """Five per shot, 10/10 labels, 10/10 candidate presence; diversify sources."""
    require(rows and all(row["split"] == "train" and row["excluded"] == "0" for row in rows),
            "Non-train/excluded sample rejected before video access")
    require(len({row["video_id"] for row in rows}) == len(rows), "Duplicate train IDs")
    for row in rows:
        require(row["original_class"] in CLASSES and row["binary_label"] == row["original_class"][-1],
                "Class/label mismatch")
        require(row["video_id"] in summaries, "Missing quality summary")
    slots = [(group, affected) for group in CLASSES for affected in (False, True)] + list(EXTRAS)
    selected, used = [], set()
    source_counts, fps_counts = Counter(), Counter()
    for group, affected in slots:
        pool = [row for row in rows if row["original_class"] == group
                and row["video_id"] not in used
                and (int(summaries[row["video_id"]]["candidate_frame_count"]) > 0) == affected]
        require(bool(pool), f"Insufficient train stratum: {group}, candidate={affected}")

        def rank(row):
            material = f"yolo_audit_v1\0{seed}\0{row['video_id']}".encode()
            return (source_counts[row["source_group"]],
                    fps_counts[summaries[row["video_id"]]["fps_rational"]],
                    hashlib.sha256(material).hexdigest())

        row = min(pool, key=rank)
        summary = summaries[row["video_id"]]
        used.add(row["video_id"])
        source_counts[row["source_group"]] += 1
        fps_counts[summary["fps_rational"]] += 1
        selected.append({**row, "candidate_present": affected,
                         "candidate_frame_count": int(summary["candidate_frame_count"]),
                         "frame_count": int(summary["frame_count"]),
                         "fps_rational": summary["fps_rational"]})
    require(len(selected) == 20 and Counter(row["binary_label"] for row in selected) == {"0": 10, "1": 10},
            "Wrong sample label quotas")
    require(Counter(row["original_class"][:-1] for row in selected) == {s: 5 for s in ("2p", "3p", "ft", "mp")},
            "Wrong sample shot quotas")
    require(sum(row["candidate_present"] for row in selected) == 10, "Wrong candidate quota")
    require(len(source_counts) >= 10, "Insufficient source diversity")
    return [{**row, "sample_id": f"Y{i:03d}"} for i, row in enumerate(selected, 1)]


def sample_frame_indices(frame_count):
    require(type(frame_count) is int and frame_count >= 10, "Video too short for five audit frames")
    return tuple(frame_count * q // 10 for q in (1, 3, 5, 7, 9))


def person_union_roi(boxes, width, height):
    """Visual diagnostic ONLY: all person boxes, 10% union-width/height padding."""
    if len(boxes) == 0:
        return [0, 0, width, height], True
    array = np.asarray(boxes, dtype=float)
    require(array.ndim == 2 and array.shape[1] == 4 and np.isfinite(array).all(), "Invalid boxes")
    require(((array[:, 2] > array[:, 0]) & (array[:, 3] > array[:, 1])).all(), "Degenerate box")
    x1, y1 = array[:, :2].min(0)
    x2, y2 = array[:, 2:].max(0)
    dx, dy = .1 * (x2 - x1), .1 * (y2 - y1)
    roi = [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
           min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))]
    require(roi[2] > roi[0] and roi[3] > roi[1], "ROI outside image")
    return roi, False


def attach_track_ids(boxes, confidences, tracks):
    """ByteTrack output includes original detection index as the last column."""
    require(len(boxes) == len(confidences), "Box/confidence mismatch")
    linked = {}
    for track in tracks:
        require(len(track) == 8, "Unexpected ByteTrack output schema")
        index = int(track[-1])
        require(track[-1] == index and 0 <= index < len(boxes) and index not in linked,
                "Invalid tracking/detection mapping")
        require(int(track[6]) == 0 and int(track[4]) >= 0, "Invalid person track")
        linked[index] = int(track[4])
    return [{"detection_id": i + 1, "xyxy": list(map(float, box)),
             "confidence": float(score), "class_id": 0, "track_id": linked.get(i)}
            for i, (box, score) in enumerate(zip(boxes, confidences))]


def draw_pair(bgr, detections, roi, *, sample_id, frame_index, fps):
    height, width = bgr.shape[:2]
    right = bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [round(value) for value in det["xyxy"]]
        color = (0, 220, 255) if det["track_id"] is None else (60, 235, 60)
        cv2.rectangle(right, (x1, y1), (x2, y2), color, 1)
        track = "--" if det["track_id"] is None else str(det["track_id"])
        text = f"D{det['detection_id']} T{track} {det['confidence']:.2f}"
        cv2.putText(right, text, (max(0, min(x1, width - 115)), max(9, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, .29, color, 1, cv2.LINE_AA)
    x1, y1, x2, y2 = roi
    cv2.rectangle(right, (x1, y1), (x2 - 1, y2 - 1), (255, 120, 255), 1)
    pair = np.zeros((height + 24, width * 2, 3), dtype=np.uint8)
    pair[24:, :width] = bgr
    pair[24:, width:] = right
    cv2.putText(pair, f"{sample_id} f={frame_index} t={frame_index / fps:.3f}s ORIGINAL", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(pair, f"PERSON D/T IDs; pink=ROI; boxes={len(detections)}", (width + 4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, .34, (255, 255, 255), 1, cv2.LINE_AA)
    return pair


def open_encoder(path, fps_rational):
    require(not path.exists(), "Refusing to overwrite an audit video")
    return subprocess.Popen([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "640x264",
        "-r", fps_rational, "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(path),
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def process_video(sample, guard, model, tracker_config, output, device):
    from ultralytics.trackers.byte_tracker import BYTETracker

    # Train authorization precedes all video path access, including metadata lookup.
    require(sample["split"] == "train", "Non-train video rejected")
    sample_id, video_id = sample["sample_id"], sample["video_id"]
    tracker = BYTETracker(args=SimpleNamespace(**tracker_config))
    before = guard.identity(video_id)
    fd = guard.open(video_id, sample["video_path"], "yolo_audit_sequential_full_video")
    cap, encoder = None, None
    records, frame_rows, snapshots = [], [], []
    n, fps = sample["frame_count"], float(Fraction(sample["fps_rational"]))
    audit_indices = sample_frame_indices(n)
    try:
        cap = cv2.VideoCapture(f"/proc/self/fd/{fd}", cv2.CAP_FFMPEG)
        require(cap.isOpened(), f"Cannot decode authorized video: {sample_id}")
        require(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n, "Frame count changed")
        require(abs(cap.get(cv2.CAP_PROP_FPS) - fps) < 1e-4, "FPS changed")
        encoder = open_encoder(output / "videos" / f"{sample_id}.mp4", sample["fps_rational"])
        for index in range(n):
            ok, bgr = cap.read()
            require(ok and bgr.dtype == np.uint8 and bgr.shape == (240, 320, 3), "Truncated/wrong video frame")
            result = model.predict(source=bgr, imgsz=640, conf=.25, iou=.7, classes=[0], max_det=100,
                                   device=device, half=False, verbose=False, save=False, stream=False)[0]
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            require(np.isfinite(boxes).all() and np.isfinite(scores).all(), "Non-finite detections")
            require((result.boxes.cls.cpu().numpy() == 0).all(), "Unexpected detection class")
            tracks = tracker.update(result.boxes.cpu().numpy(), bgr)
            detections = attach_track_ids(boxes, scores, tracks)
            roi, fallback = person_union_roi(boxes, 320, 240)
            pair = draw_pair(bgr, detections, roi, sample_id=sample_id, frame_index=index, fps=fps)
            encoder.stdin.write(pair.tobytes())
            records.append({"frame_index": index, "time_seconds": index / fps,
                            "detections": detections, "roi_xyxy": roi, "roi_full_frame_fallback": fallback,
                            "roi_area_fraction": (roi[2] - roi[0]) * (roi[3] - roi[1]) / (320 * 240)})
            if index in audit_indices:
                image_name = f"frames/{sample_id}_f{index:06d}.png"
                require(cv2.imwrite(str(output / image_name), pair), "Cannot save audit PNG")
                snapshots.append(pair)
                frame_rows.append({"frame_sample_id": f"{sample_id}_f{index:06d}",
                                   "sample_id": sample_id, "frame_index": index,
                                   "time_seconds": round(index / fps, 6), "image_path": image_name,
                                   "predicted_person_boxes": len(detections),
                                   **{field: None for field in FRAME_HUMAN}})
        require(not cap.read()[0], "Unexpected extra decoded frames")
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors="replace")
        require(encoder.wait(timeout=60) == 0, "Audit encoder failed: " + error[-600:])
    finally:
        if cap is not None:
            cap.release()
        os.close(fd)
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    require(guard.identity(video_id) == before, "Source video changed during read")
    with (output / "detections" / f"{sample_id}.json").open("x", encoding="utf-8") as stream:
        json.dump({"sample_id": sample_id, "video_id": video_id, "frames": records}, stream, allow_nan=False)
    require(len(snapshots) == 5, "Expected five audit images")
    require(cv2.imwrite(str(output / "contacts" / f"{sample_id}.png"), np.vstack(snapshots)), "Contact export failed")
    counts = [len(row["detections"]) for row in records]
    ids = {det["track_id"] for row in records for det in row["detections"] if det["track_id"] is not None}
    automatic = {"sample_id": sample_id, "video_id": video_id, "frames": n,
                 "person_boxes": sum(counts), "frames_without_person": sum(value == 0 for value in counts),
                 "frames_without_track": sum(not any(d["track_id"] is not None for d in r["detections"]) for r in records),
                 "boxes_without_track_id": sum(d["track_id"] is None for r in records for d in r["detections"]),
                 "distinct_track_ids": len(ids), "source_unchanged": True,
                 "mean_roi_area_fraction": sum(r["roi_area_fraction"] for r in records) / n}
    video_row = {"sample_id": sample_id, "video_id": video_id, "original_class": sample["original_class"],
                 "source_group": sample["source_group"], "video_path": f"videos/{sample_id}.mp4",
                 **{field: None for field in VIDEO_HUMAN}}
    return automatic, video_row, frame_rows


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_gallery(output, samples, results):
    cards = []
    for row, result in zip(samples, results):
        sid = row["sample_id"]
        cards.append(f'''<section><h2>{sid} <small>{html.escape(row['original_class'])} · {html.escape(row['source_group'])}</small></h2>
<p>{html.escape(row['video_id'])}</p><video controls preload="metadata" src="videos/{sid}.mp4"></video>
<p>{result['frames']} frames · {result['person_boxes']} person boxes · {result['frames_without_person']} frames without person detections.
These are not player precision/recall.</p><details><summary>Five fixed review frames</summary><img loading="lazy" src="contacts/{sid}.png"></details></section>''')
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><title>YOLO 人物定位 Audit</title>
<style>body{font:16px sans-serif;max-width:1000px;margin:32px auto;padding:0 20px;color:#17212e;background:#f5f7fa}section{background:white;padding:20px;margin:20px 0;border-radius:8px}video,img{width:640px;max-width:100%}small{font-weight:normal}p{line-height:1.6}a{color:#175ca4}</style>
<h1>YOLO 人物定位 Audit：20 支 train 影片</h1>
<p>左：原畫面；右：人物框與追蹤。D 是該幀偵測框編號，T 是本影片追蹤 ID；T-- 表示未確認追蹤。
粉紅框只是「所有人物框聯集＋10%」的 ROI 示意，尚未用於分類模型。沒有偵測時顯示完整畫面。</p>
<p>目前只偵測 person，沒有偵測球或籃框，也沒有辨識投籃事件。人工欄位尚未填寫，不能把「有框」當成「有框到投籃者」。</p>
<p>先看 Y001–Y003 完整影片練習，再填其餘影片。球、籃框是否被 ROI 切掉，需要人工確認。</p>
<p><a href="outputs/yolo_audit_v1/annotation_book.xlsx">人工檢查表 XLSX</a> ·
<a href="outputs/yolo_audit_v1/video_annotations.csv">影片層 CSV</a> ·
<a href="outputs/yolo_audit_v1/frame_annotations.csv">逐幀 CSV</a></p>'''
    with (output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(page + "\n".join(cards) + "</html>")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Metadata only; no model/video open or output")
    mode.add_argument("--run", action="store_true", help="Generate inference audit; never train")
    parser.add_argument("--device", choices=("cpu", "0"), default="cpu")
    args = parser.parse_args(argv)
    config = load_gate_config()
    quality = TrainQualityIndex(config)
    ledger = AccessLedger()
    guard = SplitGuard("train", ledger)
    require(len(guard.allowed) == 7193, "Train universe changed")
    samples = select_samples(list(guard.allowed.values()), quality.summaries)
    selection = {"seed": SEED, "split": "train", "videos": 20,
                 "labels": dict(Counter(row["binary_label"] for row in samples)),
                 "original_classes": dict(Counter(row["original_class"] for row in samples)),
                 "source_groups": len({row["source_group"] for row in samples}),
                 "candidate_present": sum(row["candidate_present"] for row in samples),
                 "frame_review": "five uniform locations at 10/30/50/70/90%; not event annotations",
                 "sampling_limitation": "Balanced purposeful diagnostic sample, not prevalence-representative",
                 "samples": samples}
    if args.dry_run:
        print(json.dumps({"status": "PASS", "dry_run": True, "model_loaded": False,
                          "video_opens": 0, "output_files_written": False, "selection": selection}, indent=2))
        return
    require(WEIGHT.is_file() and not WEIGHT.is_symlink() and file_sha(WEIGHT) == WEIGHT_SHA,
            "Pinned local YOLO11m weight missing/changed")
    require(args.device == "cpu" or torch.cuda.is_available(), "Requested CUDA unavailable")
    import ultralytics
    import yaml
    from ultralytics import YOLO
    tracker_path = Path(ultralytics.__file__).parent / "cfg/trackers/bytetrack.yaml"
    tracker_config = yaml.safe_load(tracker_path.read_text())
    require(tracker_config["tracker_type"] == "bytetrack", "Unexpected tracker")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = FOLDER / "artifacts"
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix=f"audit_seed{SEED}_", dir=parent))
    for subdir in ("videos", "frames", "contacts", "detections"):
        (output / subdir).mkdir()
    write_json(output / "manifest.json", selection)  # Freeze selection BEFORE inference.
    protocol = {"schema_version": "yolo_person_audit_v1", "model": "yolo11m.pt",
                "weight_sha256": WEIGHT_SHA, "ultralytics_version": ultralytics.__version__,
                "torch_version": str(torch.__version__), "opencv_version": cv2.__version__,
                "device": args.device, "imgsz": 640, "conf": .25, "iou": .7, "classes": [0],
                "max_det": 100, "half": False, "every_frame": True, "tracker": tracker_config,
                "track_buffer_units": "frames (installed ByteTrack implementation, not FPS-normalized)",
                "tracker_config_sha256": file_sha(tracker_path), "tracker_reset": "new ByteTracker per video",
                "tracking_limitation": "Detector floor .25 means ByteTrack does not receive .1-.25 candidates",
                "roi": "all raw person boxes union, 10% padding; VISUALIZATION ONLY; no-box fallback full frame",
                "input_identities": config.raw["identities"], "gate_config_sha256": config.sha256,
                "source_sha256": file_sha(Path(__file__)),
                "training_started": False, "classifier_or_feature_cache_modified": False,
                "person_only_not_ball_rim_or_shot_detection": True,
                "video_encoding": "diagnostic H264 CRF18; PNG frames lossless; not a blur-truth audit"}
    write_json(output / "config.json", protocol)
    print("Output:", output, flush=True)
    model = YOLO(str(WEIGHT))
    require(model.names[0] == "person", "Unexpected class mapping")
    results, videos, frames = [], [], []
    for number, sample in enumerate(samples, 1):
        result, video_row, frame_rows = process_video(sample, guard, model, tracker_config, output, args.device)
        results.append(result)
        videos.append(video_row)
        frames.extend(frame_rows)
        print(f"{number}/20 {sample['sample_id']}: frames={result['frames']}, boxes={result['person_boxes']}, "
              f"no-person={result['frames_without_person']}", flush=True)
    require(len(videos) == 20 and len(frames) == 100, "Audit output coverage mismatch")
    access = ledger.statistics()
    require(access["train"]["opens"] == 20 and access["validation"]["opens"] == access["test"]["opens"] == 0,
            "Video access scope mismatch")
    access_summary = {split: {"opens": counts["opens"],
                              "decoded_videos": len(results) if split == "train" else 0,
                              "decoded_frames": sum(row["frames"] for row in results) if split == "train" else 0}
                      for split, counts in access.items()}
    summary = {"videos": 20, "frames": sum(row["frames"] for row in results),
               "person_boxes": sum(row["person_boxes"] for row in results),
               "frames_without_person": sum(row["frames_without_person"] for row in results),
               "review_frames": 100, "human_annotation_status": "pending",
               "precision_recall_available": False, "training_started": False,
               "source_videos_unchanged": True, "video_access": access_summary, "per_video": results}
    write_json(output / "summary.json", summary)
    write_json(output / "annotation_payload.json", {"videos": videos, "frames": frames,
                                                  "video_human_fields": VIDEO_HUMAN, "frame_human_fields": FRAME_HUMAN})
    write_gallery(output, samples, results)
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()
