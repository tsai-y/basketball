"""Compare cached person ROI with a conservative context ROI; never train.

Only reads four already-generated train diagnostic movies and detector JSON.
The left movie panel is decoded diagnostic imagery, NOT original dataset pixels.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
from tempfile import mkdtemp

import cv2
import numpy as np


AUDIT_ROOT = Path(__file__).resolve().parent / "artifacts"
SAMPLES = ("Y004", "Y018", "Y012", "Y019")
REVIEW_FRAMES = {
    "Y004": (0, 65, 81, 98, 114, 163),
    "Y018": (0, 40, 66, 79, 92, 105),
    "Y012": (0, 13, 26, 36, 40, 53),
    "Y019": (66, 79, 89, 105, 119, 120),
}
WIDTH, HEIGHT, HEADER, FOOTER = 320, 240, 48, 24
OUTPUT_SHAPE = (HEADER + HEIGHT + FOOTER, WIDTH * 3, 3)


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ContextPolicy:
    """Unvalidated diagnostic settings, not selected on validation accuracy."""

    min_person_boxes: int = 3
    min_union_area_fraction: float = .35
    min_union_width_fraction: float = .50
    bottom_cluster_top_fraction: float = .55
    min_context_height_fraction: float = .90


def validated_boxes(boxes, width, height):
    require(type(width) is int and type(height) is int and width > 0 and height > 0,
            "Invalid frame dimensions")
    result = []
    for box in boxes:
        require(len(box) == 4, "Expected xyxy box")
        values = [float(value) for value in box]
        require(all(math.isfinite(value) for value in values), "Non-finite box")
        x1, y1, x2, y2 = values
        require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
                "Box outside frame or degenerate")
        result.append(values)
    return result


def legacy_roi(boxes, width=WIDTH, height=HEIGHT):
    """Reproduce the audit's all-person union + 10% padding exactly."""
    boxes = validated_boxes(boxes, width, height)
    if not boxes:
        return [0, 0, width, height]
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    dx, dy = .1 * (x2 - x1), .1 * (y2 - y1)
    return [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))]


def roi_area(roi):
    return (roi[2] - roi[0]) * (roi[3] - roi[1])


def context_roi(boxes, width=WIDTH, height=HEIGHT, policy=None):
    policy = policy or ContextPolicy()
    require(type(policy.min_person_boxes) is int and policy.min_person_boxes >= 1,
            "Invalid minimum person count")
    require(all(0 < value <= 1 for value in (
        policy.min_union_area_fraction, policy.min_union_width_fraction,
        policy.bottom_cluster_top_fraction, policy.min_context_height_fraction)),
        "Policy fractions must be in (0, 1]")
    old = legacy_roi(boxes, width, height)
    full = [0, 0, width, height]
    reasons = []
    if not boxes:
        reasons.append("no_person_boxes")
    else:
        if len(boxes) < policy.min_person_boxes:
            reasons.append("few_person_boxes")
        if roi_area(old) / (width * height) < policy.min_union_area_fraction:
            reasons.append("small_union")
        if (old[2] - old[0]) / width < policy.min_union_width_fraction:
            reasons.append("narrow_union")
        if old[1] >= policy.bottom_cluster_top_fraction * height:
            reasons.append("bottom_cluster")
    if reasons:
        roi = full
    else:
        roi = [0, 0, width, min(height, max(old[3], math.ceil(
            policy.min_context_height_fraction * height)))]
    return {"roi_xyxy": roi, "fallback_used": bool(reasons),
            "fallback_reasons": reasons, "full_frame_used": roi == full,
            "kept_area_fraction": roi_area(roi) / (width * height)}


def kept_region(bgr, roi, color):
    """Black outside the ROI, without resizing/magnifying the kept image."""
    x1, y1, x2, y2 = roi
    shown = np.zeros_like(bgr)
    shown[y1:y2, x1:x2] = bgr[y1:y2, x1:x2]
    cv2.rectangle(shown, (x1, y1), (x2 - 1, y2 - 1), color, 1)
    return shown


def draw_comparison(bgr, old, candidate, *, sid, index, fps, count):
    require(bgr.dtype == np.uint8 and bgr.shape == (HEIGHT, WIDTH, 3),
            "Wrong diagnostic image shape/dtype")
    canvas = np.zeros(OUTPUT_SHAPE, dtype=np.uint8)
    old_fraction = roi_area(old) / (WIDTH * HEIGHT)
    cv2.putText(canvas, f"{sid} f={index} t={index / fps:.3f}s  person boxes={count}  "
                f"old={old_fraction:.1%}  context={candidate['kept_area_fraction']:.1%}",
                (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (245, 245, 245), 1, cv2.LINE_AA)
    panels = (bgr, kept_region(bgr, old, (255, 120, 255)),
              kept_region(bgr, candidate["roi_xyxy"], (255, 230, 70)))
    labels = ("ORIGINAL DIAGNOSTIC IMAGE", "OLD PERSON UNION: KEPT REGION",
              "CONTEXT CANDIDATE: KEPT REGION")
    for number, (panel, label) in enumerate(zip(panels, labels)):
        start = number * WIDTH
        canvas[HEADER:HEADER + HEIGHT, start:start + WIDTH] = panel
        cv2.putText(canvas, label, (start + 4, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    .36, (245, 245, 245), 1, cv2.LINE_AA)
    reason = ",".join(candidate["fallback_reasons"]) or "context expansion only"
    status = "FULL-FRAME FALLBACK" if candidate["fallback_used"] else (
        "FULL FRAME AFTER EXPANSION" if candidate["full_frame_used"] else "CONTEXT ROI")
    cv2.putText(canvas, f"{status}: {reason}  |  visualization only; no event/player filter",
                (5, HEADER + HEIGHT + 16), cv2.FONT_HERSHEY_SIMPLEX,
                .38, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def select_cached_samples(manifest):
    require(manifest.get("split") == "train", "Only cached train audit is allowed")
    samples = manifest.get("samples", [])
    require(samples and all(s.get("split") == "train" and s.get("excluded") == "0"
                            for s in samples), "Non-train/excluded cached sample")
    ids = [sample["sample_id"] for sample in samples]
    require(len(ids) == len(set(ids)), "Duplicate cached sample IDs")
    require(set(SAMPLES).issubset(ids), "Missing one of the four fixed review samples")
    selected = [{s["sample_id"]: s for s in samples}[sid] for sid in SAMPLES]
    for sample in selected:
        n = sample["frame_count"]
        require(type(n) is int and n > max(REVIEW_FRAMES[sample["sample_id"]]),
                "Cached frame count does not cover review bookmarks")
        require(Fraction(sample["fps_rational"]) > 0, "Invalid cached FPS")
    return selected


def load_inputs(audit_run):
    root = AUDIT_ROOT.resolve()
    source = Path(audit_run).resolve()
    require(source.is_relative_to(root) and source.is_dir(),
            "Source must be an existing local YOLO audit artifact directory")
    samples = select_cached_samples(read_json(source / "manifest.json"))
    inputs, protected = {}, [source / "manifest.json", source / "config.json"]
    for sample in samples:
        sid = sample["sample_id"]
        video, detections = source / "videos" / f"{sid}.mp4", source / "detections" / f"{sid}.json"
        require(video.is_file() and not video.is_symlink() and detections.is_file()
                and not detections.is_symlink(), "Cached movie/detections missing or symlinked")
        data = read_json(detections)
        require(data["sample_id"] == sid and data["video_id"] == sample["video_id"],
                "Cached detection identity mismatch")
        frames = data["frames"]
        require(len(frames) == sample["frame_count"], "Cached detection count mismatch")
        fps = float(Fraction(sample["fps_rational"]))
        for index, frame in enumerate(frames):
            require(frame["frame_index"] == index and
                    abs(frame["time_seconds"] - index / fps) < 1e-6,
                    "Cached frame ordering/time mismatch")
            require(all(d["class_id"] == 0 for d in frame["detections"]), "Non-person detection")
            boxes = [d["xyxy"] for d in frame["detections"]]
            require(legacy_roi(boxes) == frame["roi_xyxy"], "Cached old ROI mismatch")
        inputs[sid] = frames
        protected.extend((video, detections))
    # Existing human/AI annotations are protected inputs, never edited here.
    protected.extend(p for p in (source / "outputs").rglob("*") if p.is_file()
                     and p.name in {"annotation_book.xlsx", "video_annotations.csv",
                                    "frame_annotations.csv", "ai_video_review_draft.xlsx",
                                    "ai_video_review_draft.csv", "review_provenance.json"})
    return source, samples, inputs, {str(p): sha256(p) for p in protected}


def open_encoder(path, fps_rational):
    return subprocess.Popen([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "960x312",
        "-r", fps_rational, "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-n", str(path),
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def process_cached_video(source, sample, frames, output, policy):
    sid, n = sample["sample_id"], sample["frame_count"]
    fps = float(Fraction(sample["fps_rational"]))
    cap = cv2.VideoCapture(str(source / "videos" / f"{sid}.mp4"))
    encoder = None
    records, snapshots = [], {}
    try:
        require(cap.isOpened(), "Cannot open cached diagnostic movie")
        require(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n and
                abs(cap.get(cv2.CAP_PROP_FPS) - fps) < 1e-4, "Diagnostic movie metadata mismatch")
        encoder = open_encoder(output / "videos" / f"{sid}.mp4", sample["fps_rational"])
        for index, frame in enumerate(frames):
            ok, pair = cap.read()
            require(ok and pair.dtype == np.uint8 and pair.shape == (264, 640, 3),
                    "Truncated or wrong-sized diagnostic movie")
            bgr = pair[24:, :WIDTH].copy()
            old = frame["roi_xyxy"]
            boxes = [d["xyxy"] for d in frame["detections"]]
            candidate = context_roi(boxes, policy=policy)
            comparison = draw_comparison(bgr, old, candidate, sid=sid, index=index,
                                         fps=fps, count=len(boxes))
            encoder.stdin.write(comparison.tobytes())
            record = {"frame_index": index, "time_seconds": index / fps,
                      "person_box_count": len(boxes), "old_roi_xyxy": old,
                      "old_kept_area_fraction": roi_area(old) / (WIDTH * HEIGHT),
                      **candidate}
            records.append(record)
            if index in REVIEW_FRAMES[sid]:
                name = f"frames/{sid}_f{index:06d}.png"
                require(cv2.imwrite(str(output / name), comparison), "Cannot save comparison PNG")
                snapshots[index] = comparison
        require(not cap.read()[0], "Extra diagnostic movie frames")
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors="replace")
        require(encoder.wait(timeout=30) == 0, "Video encoder failed: " + error[-600:])
    finally:
        cap.release()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    for page in range(2):
        indices = REVIEW_FRAMES[sid][page * 3:page * 3 + 3]
        require(cv2.imwrite(str(output / "contacts" / f"{sid}_{page}.png"),
                            np.vstack([snapshots[i] for i in indices])), "Cannot save contact sheet")
    write_json(output / "decisions" / f"{sid}.json", {"sample_id": sid,
               "source_video_id": sample["video_id"], "frames": records})
    return {"sample_id": sid, "frames": n,
            "fallback_frames": sum(r["fallback_used"] for r in records),
            "full_frame_frames": sum(r["full_frame_used"] for r in records),
            "context_crop_frames": sum(not r["full_frame_used"] for r in records),
            "fallback_reason_counts": dict(Counter(reason for r in records for reason in r["fallback_reasons"])),
            "mean_old_kept_area_fraction": sum(r["old_kept_area_fraction"] for r in records) / n,
            "mean_new_kept_area_fraction": sum(r["kept_area_fraction"] for r in records) / n,
            "review_frames": list(REVIEW_FRAMES[sid]),
            "fps": fps, "fps_rational": sample["fps_rational"]}


def write_report(output, results):
    cards = []
    for result in results:
        sid = result["sample_id"]
        bookmarks = " ".join(f'<button type="button" data-sample="{sid}" data-frame="{index}">f={index}</button>'
                             for index in result["review_frames"])
        cards.append(f'''<section id="{sid}"><h2>{sid}</h2>
<p>{result['frames']} 幀；規則觸發全畫面備援 {result['fallback_frames']} 幀；
最終使用全畫面 {result['full_frame_frames']} 幀（包含擴張後剛好全畫面的情況）。
平均保留面積：舊 {result['mean_old_kept_area_fraction']:.1%}，新 {result['mean_new_kept_area_fraction']:.1%}。</p>
<video id="video-{sid}" controls preload="metadata" data-fps="{result['fps']}" data-frame-count="{result['frames']}" src="videos/{sid}.mp4"></video>
<p>{bookmarks}</p><p><button type="button" data-sample="{sid}" data-step="-1">前一幀</button>
<button type="button" data-sample="{sid}" data-step="1">後一幀</button>
<label>播放速度 <select data-rate-sample="{sid}"><option value="1">1×</option><option value="0.25">0.25×</option></select></label>
<span id="position-{sid}" aria-live="polite"></span></p>
<details><summary>六個固定複查幀</summary><img loading="lazy" src="contacts/{sid}_0.png" alt="{sid}前三個複查幀">
<img loading="lazy" src="contacts/{sid}_1.png" alt="{sid}後三個複查幀"></details></section>''')
    page = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROI 保留資訊對照</title><style>
body{font:16px system-ui;max-width:1100px;margin:26px auto;padding:0 18px;color:#233247;background:#f5f7fa}
section{background:white;padding:18px;margin:22px 0;border:1px solid #d6dfe8;border-radius:8px}
video,img{width:960px;max-width:100%;height:auto}p{line-height:1.6}button,select{font:inherit;margin:3px;padding:6px 10px}
summary{cursor:pointer;padding:10px;background:#edf2f7}h1{font-size:25px}.warning{border-left:4px solid #bc651d;padding-left:12px}
table{border-collapse:collapse}td,th{text-align:left;padding:7px 12px;border-bottom:1px solid #d6dfe8}</style>
<h1>原畫面／舊人物 ROI／保留場景候選 ROI</h1>
<p>左：完整診斷畫面。中：舊ROI保留區（粉紅邊）。右：新候選保留區（青色邊）。黑色表示裁掉的位置，三欄不縮放、不放大。</p>
<p class="warning">只作可視化，不改模型輸入、不刪片、不訓練。沿用已產生的train診斷影片，並非重新讀取原片。畫面經H264轉碼，不作模糊真值。</p>
<table><tr><th>候選規則（尚未驗證）</th><th>處理</th></tr>
<tr><td>人物框少於3個、舊ROI面積小於35%、寬度小於50%，或上緣已在畫面高度55%以下</td><td>退回全畫面；無框也退回全畫面</td></tr>
<tr><td>其餘情況</td><td>保留完整寬度與頂部，高度至少90%，並包含舊ROI底部</td></tr></table>
<p>這是幾何規則，不會區分球員／觀眾，不會偵測球、籃框、轉場或投籃。
新ROI至少保留90%面積，可能多數幀等同全畫面；不能據此宣稱人物偵測改善或分類準確率提升。
四片是已知失敗案例的開發樣本，不是獨立驗證集。複查幀只供導覽，不參與ROI決策。</p>'''
    script = '''<script>
for(const video of document.querySelectorAll('video')){
  const sid=video.id.slice(6),fps=Number(video.dataset.fps),count=Number(video.dataset.frameCount);
  function seek(frame){const f=Math.max(0,Math.min(count-1,frame));video.pause();
    const go=()=>{video.currentTime=(f+0.05)/fps;};
    if(video.readyState>=1)go();else{video.addEventListener('loadedmetadata',go,{once:true});video.load();}}
  for(const button of document.querySelectorAll(`button[data-sample="${sid}"]`))button.addEventListener('click',()=>{
    const current=Math.round(video.currentTime*fps);
    seek(button.dataset.frame!==undefined?Number(button.dataset.frame):current+Number(button.dataset.step));});
  document.querySelector(`select[data-rate-sample="${sid}"]`).addEventListener('change',event=>{video.playbackRate=Number(event.target.value);});
  const show=()=>{document.getElementById(`position-${sid}`).textContent=`frame ${Math.min(count-1,Math.max(0,Math.round(video.currentTime*fps)))}/${count-1}（0-based）`;};
  video.addEventListener('timeupdate',show);video.addEventListener('seeked',show);show();
}
</script></html>'''
    (output / "index.html").write_text(page + "\n".join(cards) + script, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    source, samples, inputs, protected = load_inputs(args.audit_run)
    policy = ContextPolicy()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "samples": list(SAMPLES),
                          "frames": sum(s["frame_count"] for s in samples),
                          "policy": asdict(policy), "video_opens": 0,
                          "output_files_written": False, "training_started": False}, indent=2))
        return
    cv2.setNumThreads(1)
    output = Path(mkdtemp(prefix="roi_context_probe_v1_", dir=source / "outputs"))
    for subdir in ("videos", "frames", "contacts", "decisions"):
        (output / subdir).mkdir()
    protocol = {"schema": "roi_context_probe_v1", "source_audit": str(source),
                "samples": list(SAMPLES), "policy": asdict(policy),
                "policy_semantics": "geometry-only; not a player, ball, rim, shot or transition detector",
                "uses_labels_quality_or_human_animation_intervals_for_roi": False,
                "source_pixels": "left panel of already-encoded diagnostic movie, not original dataset video",
                "cropping_applied_to_model": False, "training_started": False,
                "original_dataset_video_opens": 0, "validation_test_video_opens": 0,
                "yolo_rerun": False, "classifier_and_feature_cache_modified": False,
                "opencv_version": cv2.__version__, "script_sha256": sha256(Path(__file__)),
                "immutable_inputs_sha256": protected,
                "review_frame_indices": REVIEW_FRAMES,
                "limitation": "Four known failure cases used for development; not held-out validation or accuracy evidence"}
    write_json(output / "protocol.json", protocol)
    print("Output:", output, flush=True)
    try:
        results = []
        for sample in samples:
            result = process_cached_video(source, sample, inputs[sample["sample_id"]], output, policy)
            results.append(result)
            print(f"{result['sample_id']}: {result['frames']} frames; "
                  f"fallback={result['fallback_frames']}; full-frame={result['full_frame_frames']}", flush=True)
        require(all(sha256(Path(p)) == digest for p, digest in protected.items()), "Protected audit input changed")
        summary = {"videos": len(results), "frames": sum(r["frames"] for r in results),
                   "training_started": False, "original_inputs_unchanged": True,
                   "human_annotation_files_unchanged": True,
                   "diagnostic_video_opens": len(results), "original_dataset_video_opens": 0,
                   "validation_test_video_opens": 0, "accuracy_precision_recall_available": False,
                   "per_video": results}
        write_json(output / "summary.json", summary)
        write_report(output, results)
    except Exception as error:
        (output / "FAILED.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        raise
    print("Saved:", output / "index.html", flush=True)


if __name__ == "__main__":
    main()
