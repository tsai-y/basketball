"""Build a 24-case human review checklist from existing validation predictions."""
import argparse
from collections import Counter
import hashlib
from html import escape
import json
import math
from pathlib import Path
import stat
import tempfile

from scripts import audit_i3d_temporal_feasibility as gate

ROOT = Path(__file__).resolve().parents[2]
FOLDER = Path(__file__).resolve().parent
SOURCE = Path("/home/tsai/projects/basketball")
FUSION = ROOT / "development/aca_mainline_fusion_v1/artifacts/mainline_fusion_4Lbbzt"
SEED = "aca_review_v1_seed20261026"
STRATA = (
    ("ft_false_make", "罰球沒進卻判進", 8),
    ("ft_false_miss", "罰球進卻判沒進", 4),
    ("ft_correct_miss", "罰球沒進且判對（對照）", 4),
    ("ft_correct_make", "罰球進且判對（對照）", 4),
    ("other_joint_error", "其他投籃：三模型共同判錯", 4),
)
QUESTIONS = (
    "人工判斷結果：沒進／進球／無法辨識；判斷依據與時間點",
    "資料集標籤：同意／不同意／無法確認（不直接改原標籤）",
    "籃球可見度：清楚／部分遮擋或模糊／無法追蹤",
    "籃框可見度：清楚／部分遮擋或模糊／無法辨識",
    "關鍵結果時間（秒）：接近籃框、進框、彈出或落球的時段",
    "影片是否在可判定結果前結束？是／否／不確定",
    "後續抽幀對照：ACA 50 張與主線三視窗是否保留上述時段？本次尚未檢查",
    "其他備註：人物／鏡頭移動、慢動作或其他可見現象；原因未確認前不下結論",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def stratum(row):
    if row["shot_type"] == "ft":
        return {(0, 1): "ft_false_make", (1, 0): "ft_false_miss",
                (0, 0): "ft_correct_miss", (1, 1): "ft_correct_make"}[(row["label"], row["prediction"])]
    if all(value["prediction"] != row["label"] for value in row["models"].values()):
        return "other_joint_error"
    return None


def rank_id(video_id):
    return hashlib.sha256((SEED + ":" + video_id).encode()).hexdigest()


def select(rows):
    require(len({r["video_id"] for r in rows}) == len(rows), "Duplicate input ID")
    require(all(r["split"] == "validation" for r in rows), "Only validation cases accepted")
    pools = {key: [r for r in rows if stratum(r) == key] for key, _, _ in STRATA}
    for key, _, quota in STRATA:
        require(len(pools[key]) >= quota, f"Insufficient pool for {key}")
    picked = {key: [] for key, _, _ in STRATA}
    used, total_groups = set(), Counter()
    per_stratum_groups = {key: Counter() for key, _, _ in STRATA}
    # Interleaved rounds. Group usage takes priority, hash order breaks ties.
    # Confidence is deliberately NOT used for selection.
    other_shots = ("2p", "3p", "mp", "2p")
    for round_index in range(max(q for _, _, q in STRATA)):
        for key, _, quota in STRATA:
            if round_index >= quota:
                continue
            candidates = [r for r in pools[key] if r["video_id"] not in used]
            if key == "other_joint_error":
                candidates = [r for r in candidates if r["shot_type"] == other_shots[len(picked[key])]]
                require(candidates, "Missing predeclared non-FT joint-error shot stratum")
            chosen = min(candidates, key=lambda r: (per_stratum_groups[key][r["source_group"]],
                         total_groups[r["source_group"]], rank_id(r["video_id"])))
            picked[key].append(chosen)
            used.add(chosen["video_id"])
            total_groups[chosen["source_group"]] += 1
            per_stratum_groups[key][chosen["source_group"]] += 1
    selected = []
    for key, title, _ in STRATA:
        for row in picked[key]:
            selected.append({**row, "stratum": key, "selection_reason": title,
                             "review_id": f"R{len(selected)+1:02d}"})
    require(len(selected) == len(used) == 24, "Expected 24 unique cases")
    return selected, {key: len(value) for key, value in pools.items()}


def source_path(row, allowed, source_root=SOURCE):
    # Exact allowlist check must precede ANY source filesystem lookup.
    original = allowed.get(row["video_id"])
    require(original is not None and original["split"] == "validation" and row["split"] == "validation" and
            original["binary_label"] == str(row["label"]) and
            original["original_class"] == row["original_class"] and
            original["source_group"] == row["source_group"], "Forbidden or relabelled source row")
    relative = gate.safe_relative(original["video_path"])
    path = source_root / "archive/Basketball_51 dataset" / relative
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        require(not current.is_symlink(), "Symlink source rejected")
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "Nonregular or hardlinked source rejected")
    return path


def load_inputs():
    manifest, _, metadata = gate.load_scope(["validation"])
    allowed = {r["video_id"]: r for r in manifest}
    candidate, report = read_json(FUSION / "candidate.json"), read_json(FUSION / "report.json")
    provenance = read_json(FUSION / "provenance.json")
    require(report["status"] == "completed" and candidate["selected_method"] == "three_equal" and
            candidate["test_evaluated"] is False and report["test_videos_opened"] == 0,
            "Expected completed, validation-only three-model candidate")
    require(gate.digest(FUSION / "three_equal_predictions.jsonl") == candidate["selected_predictions_sha256"] and
            gate.digest(FUSION / "report.json") == candidate["report_sha256"], "Candidate source hash mismatch")
    previous = Path(provenance["aca_comparison"])
    prediction_files = {"dense": previous / "dense_predictions.jsonl", "dw3d": previous / "dw3d_predictions.jsonl",
                        "mainline": FUSION / "mainline_verified_predictions.jsonl"}
    mappings = {}
    for name, path in prediction_files.items():
        values = read_rows(path)
        require(len(values) == 1547 and len({r["video_id"] for r in values}) == 1547 and
                {r["video_id"] for r in values} == set(allowed), "Single-model prediction ID mismatch")
        mappings[name] = {r["video_id"]: r for r in values}
    fused = read_rows(FUSION / "three_equal_predictions.jsonl")
    require(len(fused) == 1547 and len({r["video_id"] for r in fused}) == 1547 and
            {r["video_id"] for r in fused} == set(allowed), "Fusion ID mismatch")
    for row in fused:
        original = allowed[row["video_id"]]
        require(row["split"] == "validation" and row["label"] == int(original["binary_label"]) and
                row["original_class"] == original["original_class"] and row["source_group"] == original["source_group"] and
                row["shot_type"] == original["original_class"][:-1], "Fusion metadata mismatch")
        row["models"] = {}
        for name, mapping in mappings.items():
            value = mapping[row["video_id"]]
            require(all(value[k] == row[k] for k in ("split", "label", "original_class", "shot_type", "source_group")) and
                    type(value["prediction"]) is int and value["prediction"] in (0, 1) and
                    math.isfinite(value["p_make"]) and 0 <= value["p_make"] <= 1, "Single-model metadata mismatch")
            row["models"][name] = {"p_make": value["p_make"], "prediction": value["prediction"]}
        expected_p = sum(value["p_make"] / 3 for value in row["models"].values())
        require(abs(expected_p-row["p_make"]) < 1e-15 and row["prediction"] == int(row["p_make"] > .5),
                "Fusion probability/threshold mismatch")
    protected = dict(provenance["protected_files_sha256"])
    paths = [gate.MANIFEST, gate.SPLIT, *prediction_files.values(), FUSION / "candidate.json", FUSION / "report.json",
             FUSION / "provenance.json", FUSION / "three_equal_predictions.jsonl", *FOLDER.glob("*.py")]
    protected.update({str(path): gate.digest(path) for path in paths})
    verify(protected)
    return fused, allowed, protected, metadata


def verify(protected):
    require(all(gate.digest(Path(path)) == expected for path, expected in protected.items()),
            "A protected input/source/model file changed")


def review_fields():
    return {"reviewed": False, "reviewer": "", "human_outcome": None, "label_agreement": None,
            "ball_visibility": None, "rim_visibility": None, "key_result_time_seconds": None,
            "clip_ends_before_outcome": None, "aca_sampled_frames_checked": False,
            "aca_key_event_present": None, "mainline_windows_checked": False,
            "mainline_key_event_present": None, "notes": ""}


def render(cases, summary):
    cards = []
    names = {0: "沒進", 1: "進球"}
    for case in cases:
        predictions = "；".join(f"{name}：{value['p_make']:.2%}（{names[value['prediction']]}）"
                               for name, value in case["models"].items())
        questions = "".join(f"<li>□ {escape(q)}<span class='blank'></span></li>" for q in QUESTIONS)
        cards.append(f"""<article id='{case['review_id']}'>
<h2>{case['review_id']}　{escape(case['selection_reason'])}</h2>
<p>來源 {escape(case['source_group'])}　類型 {escape(case['shot_type'])}　<a href='{escape(case['source_url'], quote=True)}'>開啟原始影片</a></p>
<p class='filename'>{escape(case['video_id'])}</p>
<details><summary>展開資料集標籤與模型預測（建議先觀看影片）</summary>
<p>資料集標籤（待人工確認）：{names[case['label']]}。融合判斷：{names[case['prediction']]}，進球機率 {case['p_make']:.2%}。</p>
<p>{escape(predictions)}</p></details>
<ol>{questions}</ol></article>""")
    counts = "".join(f"<li>{escape(title)}：{quota} 部</li>" for _, title, quota in STRATA)
    return f"""<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<meta http-equiv='Content-Security-Policy' content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'">
<title>ACA／主線融合：24 部 validation 檢視清單</title>
<style>body{{font-family:system-ui,sans-serif;line-height:1.65;max-width:980px;margin:32px auto;padding:0 20px;color:#20242a;background:#f5f6f8}}
header,article{{background:white;padding:20px 26px;margin-bottom:20px;border:1px solid #d7dce3;border-radius:8px}}
h1{{font-size:25px}}h2{{font-size:20px}}a{{color:#1857a4}}.notice{{border-left:4px solid #a87518;padding-left:12px}}
.filename{{overflow-wrap:anywhere;color:#59636f;font-size:14px}}details{{padding:10px;background:#f0f3f7}}
li{{margin:8px 0}}.blank{{display:block;border-bottom:1px solid #d3d8df;min-height:22px}}
@media print{{body{{background:white;margin:0;max-width:none}}article{{break-inside:avoid;page-break-inside:avoid}}a{{color:black}}details{{display:block}}}}
</style></head><body><header><h1>24 部 validation 影片檢視清單</h1>
<p>模型：Dense ACA＋ACA-DW3D＋MixNet-I3D，各占 1/3；分類門檻保持 0.5。</p>
<p class='notice'>清單尚未人工檢視。只建立 metadata 與本機影片連結，未解碼、未複製影片、未建立抽幀對照，未讀 test。檔名含原標籤，因此不宣稱盲審。</p>
<ul>{counts}</ul><p>共涵蓋 {summary['selected_source_group_count']} 個來源群組；固定分層配額、來源平衡與 SHA256 排序，不按信心分數挑例。</p>
<p>這是刻意增加罰球與錯誤案例的診斷清單，不代表資料分布，不可用這 24 部估計模型準確率。</p>
<p>逐片依問題檢查，可列印填寫，或在 <a href='review_template.json'>review_template.json</a> 記錄人工答案。此頁不儲存填答；尚未觀看的欄位均保持未確認。</p>
<p><a href='sampling_summary.json'>抽樣規則與來源紀錄</a>。先記錄可見事實，再判斷是否需要改取樣、ROI 或訓練；不直接修改原標籤。</p>
</header>{''.join(cards)}</body></html>"""


def build():
    rows, allowed, protected, metadata = load_inputs()
    cases, pools = select(rows)
    for case in cases:
        path = source_path(case, allowed)
        case["source_path"], case["source_url"] = str(path), path.as_uri()
        case["human_review"] = review_fields()
    summary = {"schema": "aca_validation_review_v1", "status": "checklist_created_not_reviewed", "seed": SEED,
               "selection": "interleaved fixed strata; fewest same-stratum source uses, then global source uses, then SHA256(seed:ID)",
               "confidence_used_for_sampling": False, "quotas": {k: q for k, _, q in STRATA}, "pool_counts": pools,
               "other_joint_error_shots": ["2p", "3p", "mp", "2p"],
               "selected_count": len(cases), "selected_source_groups": dict(sorted(Counter(c["source_group"] for c in cases).items())),
               "selected_source_group_count": len({c["source_group"] for c in cases}),
               "selected_ids_sha256": gate.objhash([c["video_id"] for c in cases]),
               "input_hashes": metadata["input_hashes"], "protected_files_sha256": protected,
               "fusion_candidate": str(FUSION / "candidate.json"), "human_review_complete": False,
               "test_videos_opened": 0, "videos_opened_or_decoded": 0, "source_video_bytes_read": 0,
               "sampling_is_diagnostic_not_accuracy_estimate": True}
    verify(protected)
    artifacts = FOLDER / "artifacts"
    artifacts.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="review24_", dir=artifacts))
    write_new(output / "review_template.json", {"schema": "aca_validation_human_review_template_v1", "cases": cases})
    with (output / "index.html").open("x", encoding="utf-8") as stream:
        stream.write(render(cases, summary))
    verify(protected)
    summary["protected_files_unchanged"] = True
    write_new(output / "sampling_summary.json", summary)
    print(json.dumps({"output": str(output), "count": len(cases), "quotas": summary["quotas"],
                      "pool_counts": pools, "source_groups": summary["selected_source_groups"],
                      "human_review_complete": False, "videos_decoded": 0}, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-list", action="store_true")
    args = parser.parse_args()
    if args.build_list:
        build()
    else:
        print(json.dumps({"mode": "plan_only", "seed": SEED, "strata": STRATA, "no_video_decoding": True}, ensure_ascii=False))
