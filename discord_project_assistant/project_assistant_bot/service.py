from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import time
from typing import Iterable, Sequence

from .config import Settings
from .conflicts import EvidenceConflict, detect_conflicts
from .index import ProjectIndex, RefreshReport, SearchHit
from .provider import AIProvider
from .sources import SafeSourceLoader


QUESTION_LIMIT = 1000


@dataclass(frozen=True)
class Answer:
    text: str
    hits: tuple[SearchHit, ...]
    used_ai: bool
    degraded_reason: str | None = None


def split_discord_message(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= limit:
            parts.append(remaining)
            break
        boundary = max(
            remaining.rfind("\n\n", 0, limit),
            remaining.rfind("\n", 0, limit),
            remaining.rfind("。", 0, limit),
            remaining.rfind(" ", 0, limit),
        )
        if boundary < limit // 2:
            boundary = limit
        else:
            boundary += 1
        parts.append(remaining[:boundary].rstrip())
        remaining = remaining[boundary:].lstrip()
    return parts


def _source_lines(hits: Sequence[SearchHit]) -> str:
    unique: list[str] = []
    for hit in hits:
        citation = hit.chunk.citation
        if citation not in unique:
            unique.append(citation)
    if not unique:
        return "來源：資料不足（索引中沒有匹配片段）"
    return "來源：\n" + "\n".join(f"- [S{index}] {citation}" for index, citation in enumerate(unique, 1))


def _conflict_text(conflicts: Sequence[EvidenceConflict]) -> str:
    if not conflicts:
        return ""
    lines = []
    for item in conflicts[:4]:
        resolution = (
            "較新者優先。"
            if item.older_date and item.newer_date and item.older_date != item.newer_date
            else "日期相同或缺漏；依較高優先來源呈現，必要時人工核對。"
        )
        lines.append(
            f"{item.subject}: {item.older_date or '日期未知'} {item.older_status} ({item.older_citation})；"
            f"{item.newer_date or '日期未知'} {item.newer_status} ({item.newer_citation})。{resolution}"
        )
    return "\n".join(lines)


class SlidingWindowLimiter:
    def __init__(self, limit: int, seconds: int = 60):
        self.limit = limit
        self.seconds = seconds
        self.events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        queue = self.events[key]
        while queue and queue[0] <= now - self.seconds:
            queue.popleft()
        if len(queue) >= self.limit:
            return False
        queue.append(now)
        return True


class ProjectAssistant:
    def __init__(
        self,
        settings: Settings,
        index: ProjectIndex,
        loader: SafeSourceLoader,
        provider: AIProvider,
    ):
        self.settings = settings
        self.index = index
        self.loader = loader
        self.provider = provider
        self.last_sources: dict[str, tuple[SearchHit, ...]] = {}
        self.command_limiter = SlidingWindowLimiter(settings.per_user_minute_command_limit)

    def refresh(self) -> RefreshReport:
        return self.index.refresh(self.loader)

    def _retrieve(self, query: str) -> tuple[SearchHit, ...]:
        hits = self.index.search(query, self.settings.max_context_chunks)
        selected: list[SearchHit] = []
        used = 0
        for hit in hits:
            if used + len(hit.chunk.content) > self.settings.max_context_chars:
                continue
            selected.append(hit)
            used += len(hit.chunk.content)
        return tuple(selected)

    def _retrieve_many(self, queries: Sequence[str], per_query: int = 3) -> tuple[SearchHit, ...]:
        selected: list[SearchHit] = []
        seen: set[tuple[str, int]] = set()
        used = 0
        for query in queries:
            for hit in self.index.search(query, per_query):
                key = (hit.chunk.path, hit.chunk.line_start)
                if key in seen or used + len(hit.chunk.content) > self.settings.max_context_chars:
                    continue
                selected.append(hit)
                seen.add(key)
                used += len(hit.chunk.content)
                if len(selected) >= self.settings.max_context_chunks:
                    return tuple(selected)
        return tuple(selected)

    def _local_answer(self, label: str, hits: Sequence[SearchHit], conflict_text: str = "") -> str:
        if not hits:
            return f"資料不足：索引中找不到足以回答「{label}」的可信內容。\n\n{_source_lines(hits)}"
        excerpts = []
        selected: list[SearchHit] = []
        seen_paths: set[str] = set()
        for hit in hits:
            if hit.chunk.path in seen_paths:
                continue
            selected.append(hit)
            seen_paths.add(hit.chunk.path)
            if len(selected) == 4:
                break
        for hit in selected:
            content = re.sub(r"\s+", " ", hit.chunk.content).strip()
            excerpts.append(f"- {content[:420]}{'…' if len(content) > 420 else ''} [{hit.chunk.citation}]")
        result = "文件明確記載：\n" + "\n".join(excerpts)
        if conflict_text:
            result += "\n\n文件衝突：\n" + conflict_text
        result += "\n\n" + _source_lines(selected)
        return result

    async def answer(self, user_id: str, question: str, *, prefer_ai: bool = True) -> Answer:
        question = question.strip()
        if not question:
            return Answer("資料不足：問題不可為空。", (), False)
        if len(question) > QUESTION_LIMIT:
            return Answer(f"問題過長；請限制在 {QUESTION_LIMIT} 字內。", (), False)
        hits = self._retrieve(question)
        self.last_sources[user_id] = hits
        conflicts = detect_conflicts(hits)
        conflict_text = _conflict_text(conflicts)
        provider_health = self.provider.health()
        if prefer_ai and provider_health.configured and provider_health.available and hits:
            allowed, reason = self.index.claim_ai_request(
                user_id, self.settings.daily_ai_request_limit, self.settings.per_user_daily_ai_request_limit
            )
            if allowed:
                try:
                    text = await self.provider.answer(question, hits, conflict_text)
                    return Answer(text + "\n\n" + _source_lines(hits), hits, True)
                except Exception:
                    local = self._local_answer(question, hits, conflict_text)
                    degraded = "AI 暫時不可用，已切換本機檢索"
                    return Answer(f"（{degraded}。）\n\n{local}", hits, False, degraded)
            return Answer(f"（{reason}，已使用本機檢索。）\n\n{self._local_answer(question, hits, conflict_text)}", hits, False, reason)
        reason = None if hits else "沒有匹配證據"
        return Answer(self._local_answer(question, hits, conflict_text), hits, False, reason)

    async def status(self, user_id: str) -> Answer:
        hits = self._retrieve_many(
            (
                "Visual Preprocessing Augmentation Pilot v1 完成 正式政策 尚未定案 停止點",
                "2026-10-10 Quality-aware Clip Sampler v1 7193 7188 fallback 驗證 邊界",
                "目前專案狀態 已完成 正在進行 A1 A2 B C frozen Gate paused",
                "A1 A2 B C 一次性正式 test 完成 accuracy frozen full-video",
            ),
            per_query=1,
        )
        self.last_sources[user_id] = hits
        conflicts = _conflict_text(detect_conflicts(hits))
        return Answer(self._local_answer("目前專案狀態", hits, conflicts), hits, False)

    async def next_steps(self, user_id: str) -> Answer:
        hits = self._retrieve_many(
            (
                "研究範圍 順序 優先 自動品質篩檢 影像增強 特徵擷取 多模型互補 融合",
                "Visual Preprocessing Pilot 視覺檢查 決策 暫留候選 尚待決定 正式預設",
                "研究限制 停止點 正式影像增強政策 未完成 下一輪",
            ),
            per_query=3,
        )
        self.last_sources[user_id] = hits
        conflicts = _conflict_text(detect_conflicts(hits))
        return Answer(self._local_answer("下一步、阻塞事項與原因", hits, conflicts), hits, False)

    async def experiments(self, user_id: str) -> Answer:
        return await self.answer(
            user_id,
            "A1 A2 B C 實驗 正式 test accuracy precision recall F1 結果 限制 不可推論 frozen",
            prefer_ai=False,
        )

    def sources(self, user_id: str) -> str:
        return _source_lines(self.last_sources.get(user_id, ()))

    def health(self) -> str:
        stats = self.index.stats()
        provider = self.provider.health()
        return (
            "Bot：正常（唯讀檢索模式）\n"
            f"索引：{stats['documents']} 份文件／{stats['chunks']} 個片段；最後更新 {stats['last_refresh'] or '尚未建立'}\n"
            f"AI provider：{provider.name}；{'可用' if provider.available else '本機 fallback'}；{provider.detail}\n"
            f"今日 AI 請求：{self.index.ai_usage_today()}/{self.settings.daily_ai_request_limit}"
        )
