from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from project_assistant_bot.config import Settings
from project_assistant_bot.index import ProjectIndex
from project_assistant_bot.provider import AIProvider, DisabledProvider, ProviderHealth
from project_assistant_bot.service import ProjectAssistant, SlidingWindowLimiter, split_discord_message
from project_assistant_bot.sources import SafeSourceLoader, SourcePolicy


def settings(root: Path, data: Path) -> Settings:
    return Settings(
        project_root=root, data_dir=data, sources_config=root / "sources.yaml", discord_token=None,
        allowed_guild_ids=frozenset(), allowed_channel_ids=frozenset(), allowed_role_ids=frozenset(),
        admin_role_ids=frozenset(), ai_provider="local", codex_auth_mode="chatgpt_login", codex_binary="codex",
        codex_home=data / "codex-home", codex_model=None, codex_reasoning_effort="low",
        codex_timeout_seconds=1, ai_max_answer_chars=100, max_context_chunks=5, max_context_chars=5000,
        daily_ai_request_limit=2, per_user_daily_ai_request_limit=1, per_user_minute_command_limit=2,
        log_level="INFO", discord_message_chars=120, sync_commands=False,
    )


class FailingProvider(AIProvider):
    def health(self) -> ProviderHealth:
        return ProviderHealth("fake", configured=True, available=True, detail="test")

    async def answer(self, question, hits, conflicts):
        raise TimeoutError("simulated")


class ServiceTests(unittest.TestCase):
    def test_offline_answer_has_citation_and_sources_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            data = Path(directory) / "data"
            root.mkdir()
            (root / "README.md").write_text("# 專案\n## 狀態\nA1 已完成並凍結。", encoding="utf-8")
            loader = SafeSourceLoader(root, SourcePolicy(("README.md",), ()))
            index = ProjectIndex(data / "index.sqlite3")
            app = ProjectAssistant(settings(root, data), index, loader, DisabledProvider())
            app.refresh()
            answer = asyncio.run(app.answer("u1", "A1 狀態", prefer_ai=True))
            self.assertFalse(answer.used_ai)
            self.assertIn("README.md", answer.text)
            self.assertIn("文件明確記載", answer.text)
            self.assertIn("README.md", app.sources("u1"))
            index.close()

    def test_split_and_burst_limit(self) -> None:
        text = "第一段。" * 20 + "\n\n" + "第二段。" * 20
        parts = split_discord_message(text, 120)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part) <= 120 for part in parts))
        limiter = SlidingWindowLimiter(2)
        self.assertTrue(limiter.allow("u"))
        self.assertTrue(limiter.allow("u"))
        self.assertFalse(limiter.allow("u"))

    def test_daily_ai_budget_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = ProjectIndex(Path(directory) / "index.sqlite3")
            self.assertEqual(index.claim_ai_request("u", 2, 1), (True, "ok"))
            self.assertFalse(index.claim_ai_request("u", 2, 1)[0])
            self.assertTrue(index.claim_ai_request("v", 2, 1)[0])
            self.assertFalse(index.claim_ai_request("w", 2, 1)[0])
            index.close()

    def test_provider_failure_falls_back_with_disclosure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            data = Path(directory) / "data"
            root.mkdir()
            (root / "README.md").write_text("# 專案\nA1 已完成。", encoding="utf-8")
            loader = SafeSourceLoader(root, SourcePolicy(("README.md",), ()))
            index = ProjectIndex(data / "index.sqlite3")
            app = ProjectAssistant(settings(root, data), index, loader, FailingProvider())
            app.refresh()
            answer = asyncio.run(app.answer("u1", "A1 狀態"))
            self.assertFalse(answer.used_ai)
            self.assertIn("已切換本機檢索", answer.text)
            self.assertIn("README.md", answer.text)
            index.close()


if __name__ == "__main__":
    unittest.main()
