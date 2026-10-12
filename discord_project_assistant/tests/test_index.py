from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import unittest

from project_assistant_bot.conflicts import detect_conflicts
from project_assistant_bot.index import ProjectIndex
from project_assistant_bot.sources import SafeSourceLoader, SourcePolicy


class ProjectIndexTests(unittest.TestCase):
    def test_incremental_refresh_ranking_metadata_and_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            docs = root / "docs"
            docs.mkdir()
            current = docs / "current.md"
            current.write_text("# 目前研究方向\n更新日期：2026-10-10\n## 下一步\n優先研究影像增強。", encoding="utf-8")
            other = docs / "other.md"
            other.write_text("# 其他\n籃球分類。", encoding="utf-8")
            loader = SafeSourceLoader(root, SourcePolicy((), ("docs/*.md",)))
            index = ProjectIndex(Path(directory) / "data" / "index.sqlite3")
            first = index.refresh(loader)
            self.assertEqual(first.added, 2)
            self.assertEqual(first.updated, 0)
            second = index.refresh(loader)
            self.assertEqual(second.unchanged, 2)
            hits = index.search("下一步影像增強", 3)
            self.assertEqual(hits[0].chunk.path, "docs/current.md")
            self.assertIn("下一步", hits[0].chunk.heading)
            self.assertRegex(hits[0].chunk.content_hash, r"^[0-9a-f]{64}$")
            current.write_text("# 目前研究方向\n更新日期：2026-10-11\n## 下一步\n優先研究特徵融合。", encoding="utf-8")
            third = index.refresh(loader)
            self.assertEqual(third.updated, 1)
            other.unlink()
            fourth = index.refresh(loader)
            self.assertEqual(fourth.removed, 1)
            index.close()

    def test_refresh_is_safe_from_worker_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "README.md").write_text("# Status\ncomplete", encoding="utf-8")
            loader = SafeSourceLoader(root, SourcePolicy(("README.md",), ()))
            index = ProjectIndex(Path(directory) / "index.sqlite3")
            errors: list[BaseException] = []

            def run() -> None:
                try:
                    index.refresh(loader)
                except BaseException as exc:  # pragma: no cover - assertion captures worker failures
                    errors.append(exc)

            worker = threading.Thread(target=run)
            worker.start()
            worker.join()
            self.assertEqual(errors, [])
            self.assertEqual(index.stats()["documents"], 1)
            index.close()

    def test_conflict_detection_prefers_newer_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            (root / "old.md").write_text("# Log\n2026-10-01\nExperiment D planned。", encoding="utf-8")
            (root / "new.md").write_text("# Current\n2026-10-07\nExperiment D paused。", encoding="utf-8")
            loader = SafeSourceLoader(root, SourcePolicy(("old.md", "new.md"), ()))
            index = ProjectIndex(Path(directory) / "index.sqlite3")
            index.refresh(loader)
            conflicts = detect_conflicts(index.search("Experiment D planned paused", 10))
            self.assertTrue(conflicts)
            self.assertEqual(conflicts[0].newer_status, "paused")
            self.assertEqual(conflicts[0].newer_date, "2026-10-07")
            index.close()


if __name__ == "__main__":
    unittest.main()
