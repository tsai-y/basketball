from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import tempfile

from .config import Settings
from .index import ProjectIndex
from .provider import CodexAppServerProvider, DisabledProvider
from .service import ProjectAssistant
from .sources import SafeSourceLoader, SourcePolicy


def build_assistant(settings: Settings, *, index_path: Path | None = None) -> ProjectAssistant:
    selected_index = (index_path or settings.index_path).resolve()
    if selected_index.is_relative_to(settings.project_root.resolve()):
        raise ValueError("BOT_DATA_DIR/index path 不可位於唯讀 project root 內")
    if settings.codex_home.is_relative_to(settings.project_root.resolve()):
        raise ValueError("CODEX_HOME 不可位於唯讀 project root 內")
    policy = SourcePolicy.load(settings.sources_config)
    loader = SafeSourceLoader(settings.project_root, policy)
    index = ProjectIndex(selected_index)
    if settings.ai_provider == "codex" and settings.codex_auth_mode == "chatgpt_login":
        provider = CodexAppServerProvider(
            binary=settings.codex_binary,
            codex_home=settings.codex_home,
            model=settings.codex_model,
            reasoning_effort=settings.codex_reasoning_effort,
            timeout_seconds=settings.codex_timeout_seconds,
            max_answer_chars=settings.ai_max_answer_chars,
        )
    elif settings.ai_provider == "local":
        provider = DisabledProvider("AI_PROVIDER=local；僅使用本機檢索")
    elif settings.ai_provider == "codex":
        provider = DisabledProvider(
            f"不支援 CODEX_AUTH_MODE={settings.codex_auth_mode!r}；"
            "請使用 chatgpt_login 或設 AI_PROVIDER=local"
        )
    else:
        provider = DisabledProvider(f"不支援 AI_PROVIDER={settings.ai_provider!r}；僅使用本機檢索")
    return ProjectAssistant(settings, index, loader, provider)


async def _dry_run(settings: Settings, question: str) -> int:
    with tempfile.TemporaryDirectory(prefix="basketball-bot-smoke-") as directory:
        assistant = build_assistant(settings, index_path=Path(directory) / "index.sqlite3")
        report = assistant.refresh()
        answer = await assistant.answer("dry-run", question, prefer_ai=False)
        print(
            f"SMOKE PASS: documents={report.scanned} added={report.added} "
            f"chunks={assistant.index.stats()['chunks']} skipped={len(report.skipped)}"
        )
        print(answer.text)
        assistant.index.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Basketball-51 Discord project assistant")
    parser.add_argument("--dry-run", action="store_true", help="建立暫存索引並做離線查詢，不連 Discord/Codex")
    parser.add_argument("--question", default="目前完成了什麼，下一步是什麼？")
    parser.add_argument("--refresh-only", action="store_true", help="只更新正式索引後退出")
    args = parser.parse_args()
    settings = Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.dry_run:
        return asyncio.run(_dry_run(settings, args.question))
    assistant = build_assistant(settings)
    if args.refresh_only:
        report = assistant.refresh()
        print(report)
        return 0
    from .discord_bot import run_discord

    run_discord(settings, assistant)
    return 0
