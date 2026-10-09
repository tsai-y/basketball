"""Publisher v3.1 fixtures use only metadata/CSV/JSON and never touch videos."""
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_blur_train_v3 as scanner
import publish_blur_train_v3 as publisher


class PublisherV31(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = scanner.load_config(scanner.DEFAULT_CONFIG)
        cls.guard = scanner.Guard()
        cls.rows = scanner.choose_smoke(cls.guard, cls.cfg["seed"])

    def make_fixture(self, root: Path, rows=None) -> Path:
        rows = copy.deepcopy(rows if rows is not None else self.rows)
        out = root / "run"
        (out / "videos").mkdir(parents=True)
        (out / "aggregates").mkdir()
        (out / "invocations").mkdir()
        ctx = dict(config_hash=scanner.digest(scanner.DEFAULT_CONFIG), config=self.cfg,
                   scanner=scanner.code_identity(), inputs=self.guard.scope["input_hashes"],
                   selection=rows, scope="smoke", seed=self.cfg["seed"], environment={})
        (out / "run.json").write_text(scanner.p.dumps(dict(context=ctx,
            created_utc="2026-10-09T00:00:00+00:00", commit="scanner-commit", command=["scanner"])))
        for index, row in enumerate(rows):
            attempt = out / "videos" / scanner.key(row["video_id"]) / "attempt-000001"
            attempt.mkdir(parents=True)
            candidate = int(index == 0)
            frames = [dict(video_id=row["video_id"], frame_index=0, split="train",
                           label=row["label"], source_group=row["source_group"],
                           config_hash=ctx["config_hash"], scanner_version=ctx["scanner"]["version"],
                           decode_status="ok", primary_candidate=candidate)]
            (attempt / "frames.csv").write_text(scanner.csv_text(frames))
            segment = [dict(start_frame=0, end_frame_exclusive=1, start_seconds=0.0,
                            end_seconds_exclusive=.04, duration_seconds=.04, frames=1)] if candidate else []
            summary = dict(video_id=row["video_id"], label=row["label"],
                           source_group=row["source_group"], split="train", fps=25.0,
                           fps_rational="25/1", expected_frame_count=1, processed_frame_count=1,
                           decode_status="ok", config_hash=ctx["config_hash"],
                           scanner_version=ctx["scanner"]["version"], candidate_frames=candidate,
                           candidate_ratio=float(candidate), candidate_segments=segment,
                           noncandidate_segments=[] if candidate else segment,
                           noncandidate_windows={str(size): dict(exists=False, count=0)
                                                 for size in self.cfg["windows"]},
                           processing_seconds=.01, longest_candidate_seconds=.04 if candidate else 0)
            (attempt / "summary.json").write_text(scanner.p.dumps(summary))
            identity = dict(video_id=row["video_id"],
                            resolved_path=str(scanner.p.gate.DATA / row["video_path"]),
                            manifest_sha256=ctx["inputs"]["manifest"])
            completion = dict(video_id=row["video_id"], input_identity=identity,
                              config_hash=ctx["config_hash"], scanner=ctx["scanner"],
                              expected_frame_count=1, processed_frame_count=1, status="ok",
                              completion_timestamp="2026-10-09T00:00:01+00:00",
                              output_hashes={name: scanner.digest(attempt / name)
                                             for name in ("frames.csv", "summary.json")})
            (attempt / "completion.json").write_text(scanner.p.dumps(completion))
        self.rebuild(out)
        return out

    def rebuild(self, out: Path) -> None:
        ctx = publisher.read_json(out / "run.json")["context"]
        for path in (out / "aggregates").glob("*.json"):
            path.unlink()
        (out / "aggregates" / "summary-1.json").write_text(scanner.p.dumps(scanner.aggregate(out, ctx)))

    def report(self, out: Path):
        return publisher.verify_run(out, scanner.p.gate.MANIFEST, scanner.DEFAULT_CONFIG, "smoke", True)

    def test_complete_fixture_and_bounded_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.report(self.make_fixture(Path(tmp)))
            self.assertEqual(report["run_status"], "COMPLETE")
            self.assertTrue(report["integrity"]["aggregate_rebuilt_equal"])
            self.assertIn("one frames.csv at a time", report["verification"]["bounded_memory"])

    def test_missing_completion_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.make_fixture(Path(tmp))
            video_id = self.rows[0]["video_id"]
            (out / "videos" / scanner.key(video_id) / "attempt-000001" / "completion.json").unlink()
            self.rebuild(out)
            report = self.report(out)
            self.assertEqual(report["run_status"], "PARTIAL")
            self.assertIn(video_id, report["integrity"]["incomplete"])

    def test_duplicate_successful_completion_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.make_fixture(Path(tmp)); video_id = self.rows[0]["video_id"]
            directory = out / "videos" / scanner.key(video_id)
            shutil.copytree(directory / "attempt-000001", directory / "attempt-000002")
            self.rebuild(out)
            report = self.report(out)
            self.assertEqual(report["run_status"], "INVALID")
            self.assertIn(video_id, report["integrity"]["duplicate"])

    def test_unexpected_unknown_id_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.make_fixture(Path(tmp))
            (out / "errors.jsonl").write_text(json.dumps({"video_id": "unknown/video.mp4"}) + "\n")
            report = self.report(out)
            self.assertEqual(report["run_status"], "INVALID")
            self.assertIn("unknown/video.mp4", report["integrity"]["present_but_unexpected"])

    def test_held_out_id_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.make_fixture(Path(tmp))
            heldout = next(row["video_id"] for row in scanner.p.read_csv(scanner.p.gate.MANIFEST)
                           if row["split"] in ("validation", "test"))
            (out / "errors.jsonl").write_text(json.dumps({"video_id": heldout}) + "\n")
            report = self.report(out)
            self.assertEqual(report["run_status"], "INVALID")
            self.assertIn(heldout, report["integrity"]["held_out"])

    def test_config_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.make_fixture(Path(tmp))
            run = publisher.read_json(out / "run.json")
            run["context"]["config_hash"] = "0" * 64
            (out / "run.json").write_text(scanner.p.dumps(run))
            report = self.report(out)
            self.assertEqual(report["run_status"], "INVALID")
            self.assertFalse(report["integrity"]["config_hash_consistent"])

    def test_corrupt_artifact_hash_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.make_fixture(Path(tmp)); video_id = self.rows[0]["video_id"]
            frames = out / "videos" / scanner.key(video_id) / "attempt-000001" / "frames.csv"
            with frames.open("a") as stream:
                stream.write("corrupt\n")
            report = self.report(out)
            self.assertEqual(report["run_status"], "INVALID")
            self.assertTrue(any("artifact hash mismatch" in value for value in report["integrity"]["corrupt"]))

    def test_partial_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.make_fixture(Path(tmp), self.rows[:-1])
            report = self.report(out)
            self.assertEqual(report["run_status"], "PARTIAL")
            self.assertIn(self.rows[-1]["video_id"], report["integrity"]["expected_but_missing"])

    def test_unresolved_failure_is_complete_with_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.make_fixture(Path(tmp)); video_id = self.rows[0]["video_id"]
            attempt = out / "videos" / scanner.key(video_id) / "attempt-000001"
            summary = publisher.read_json(attempt / "summary.json")
            summary.update(decode_status="failed", candidate_frames=None, candidate_ratio=None,
                           candidate_segments=None, quality_unavailable=True)
            (attempt / "summary.json").write_text(scanner.p.dumps(summary))
            rows = scanner.p.read_csv(attempt / "frames.csv")
            rows[0]["decode_status"] = "failed"
            (attempt / "frames.csv").write_text(scanner.csv_text(rows))
            completion = publisher.read_json(attempt / "completion.json")
            completion["status"] = "failed"
            completion["output_hashes"]["summary.json"] = scanner.digest(attempt / "summary.json")
            completion["output_hashes"]["frames.csv"] = scanner.digest(attempt / "frames.csv")
            (attempt / "completion.json").write_text(scanner.p.dumps(completion))
            self.rebuild(out)
            report = self.report(out)
            self.assertEqual(report["run_status"], "COMPLETE_WITH_FAILURES")
            self.assertEqual(report["failures"]["unresolved_ids"], [video_id])

    def test_aggregate_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = self.make_fixture(Path(tmp))
            aggregate = publisher.read_json(out / "aggregates" / "summary-1.json")
            aggregate["candidate_frames"] += 1
            (out / "aggregates" / "summary-1.json").write_text(scanner.p.dumps(aggregate))
            report = self.report(out)
            self.assertEqual(report["run_status"], "INVALID")
            self.assertFalse(report["integrity"]["aggregate_rebuilt_equal"])

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(tmp); out = self.make_fixture(root)
            md, js = root / "report.md", root / "report.json"
            md.write_text("existing")
            with self.assertRaisesRegex(publisher.PublicationError, "existing output"):
                publisher.main(["--run-dir", str(out), "--scope", "smoke", "--require-complete",
                                "--output-md", str(md), "--output-json", str(js)])

    def test_markdown_json_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.report(self.make_fixture(Path(tmp)))
            markdown = publisher.render_markdown(report)
            publisher.verify_markdown_json(markdown, report)
            round_tripped = json.loads(json.dumps(report, sort_keys=True))
            publisher.verify_markdown_json(markdown, round_tripped)
            changed = markdown.replace("| successful_videos | 20 |", "| successful_videos | 19 |")
            with self.assertRaisesRegex(publisher.PublicationError, "successful_videos"):
                publisher.verify_markdown_json(changed, report)

    def test_existing_smoke_core_statistics_unchanged(self):
        publisher.legacy_smoke_check()
        run = scanner.ARTIFACTS / "blur_train_v3_smoke"
        if run.is_dir():
            report = self.report(run)
            self.assertEqual(report["run_status"], "COMPLETE")
            self.assertEqual(report["coverage"]["total_decoded_frames"], 3055)
            self.assertEqual(report["coverage"]["total_candidate_frames"], 54)
            self.assertEqual(report["coverage"]["candidate_segments"], 5)
            self.assertAlmostEqual(report["coverage"]["candidate_frame_ratio"], .0176759410801964)


if __name__ == "__main__":
    unittest.main(verbosity=2)
