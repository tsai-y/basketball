from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from project_assistant_bot.sources import SafeSourceLoader, SourcePolicy, UnsafeSourceError


class SafeSourceLoaderTests(unittest.TestCase):
    def test_rejects_symlink_secret_and_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "ok.md").write_text("# 狀態\n完成", encoding="utf-8")
            (root / "docs" / "config.md").write_text(
                "OPENAI_API_KEY=" + "sk-" + "a" * 22, encoding="utf-8"
            )
            (root / "docs" / "large.md").write_text("x" * 101, encoding="utf-8")
            (root / "docs" / "link.md").symlink_to(root / "docs" / "ok.md")
            policy = SourcePolicy(exact=(), globs=("docs/*.md",), max_file_bytes=100, max_total_bytes=1000)
            documents, skipped = SafeSourceLoader(root, policy).load_all()
            self.assertEqual([item.relative_path for item in documents], ["docs/ok.md"])
            joined = "\n".join(skipped)
            self.assertIn("large.md", joined)
            self.assertIn("link.md", joined)
            self.assertIn("config.md", joined)

    def test_rejects_traversal_even_if_called_directly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = SourcePolicy(exact=("../outside.md",), globs=())
            loader = SafeSourceLoader(root, policy)
            with self.assertRaises(UnsafeSourceError):
                loader._validate_path(root / ".." / "outside.md", "../outside.md")

    def test_explicit_deny_overrides_glob(self) -> None:
        policy = SourcePolicy(exact=(), globs=("docs/*.md",), deny=("docs/private*.md",))
        self.assertTrue(policy.permits("docs/public.md"))
        self.assertFalse(policy.permits("docs/private_notes.md"))


if __name__ == "__main__":
    unittest.main()
