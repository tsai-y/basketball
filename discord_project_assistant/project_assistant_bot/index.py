from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
from typing import Iterable

from .sources import SafeDocument, SafeSourceLoader


ASCII_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_./+-]*|\d+(?:\.\d+)*")
CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = ASCII_TOKEN_RE.findall(lowered)
    for run in CJK_RUN_RE.findall(lowered):
        tokens.extend(run)
        if len(run) > 1:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass(frozen=True)
class Chunk:
    path: str
    title: str
    heading: str
    line_start: int
    line_end: int
    content: str
    modified_ns: int
    content_hash: str
    document_date: str | None

    @property
    def citation(self) -> str:
        location = self.heading or self.title or f"L{self.line_start}"
        return f"{self.path} — {location} (L{self.line_start}–{self.line_end})"


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class RefreshReport:
    scanned: int
    added: int
    updated: int
    unchanged: int
    removed: int
    chunks: int
    skipped: tuple[str, ...]


def _document_date(content: str, modified_ns: int) -> str:
    match = DATE_RE.search(content[:5000])
    if match:
        return match.group(1)
    return datetime.fromtimestamp(modified_ns / 1_000_000_000, timezone.utc).date().isoformat()


def chunk_document(document: SafeDocument, max_chars: int = 1800, overlap_lines: int = 2) -> list[Chunk]:
    lines = document.content.splitlines()
    title = Path(document.relative_path).stem
    current_heading = ""
    heading_stack: list[str] = []
    chunks: list[Chunk] = []
    buffer: list[tuple[int, str]] = []
    document_date = _document_date(document.content, document.modified_ns)
    active_date = document_date

    def flush(keep_overlap: bool = True) -> None:
        nonlocal buffer
        if not buffer:
            return
        content = "\n".join(line for _, line in buffer).strip()
        nonblank = [line for _, line in buffer if line.strip()]
        headings_only = bool(nonblank) and all(HEADING_RE.match(line) for line in nonblank)
        if content and not headings_only:
            local_date_match = DATE_RE.search(current_heading)
            date = local_date_match.group(1) if local_date_match else active_date
            chunks.append(
                Chunk(
                    path=document.relative_path,
                    title=title,
                    heading=current_heading,
                    line_start=buffer[0][0],
                    line_end=buffer[-1][0],
                    content=content,
                    modified_ns=document.modified_ns,
                    content_hash=document.sha256,
                    document_date=date,
                )
            )
        buffer = buffer[-overlap_lines:] if keep_overlap and overlap_lines else []

    for line_no, line in enumerate(lines, 1):
        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush(keep_overlap=False)
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading_text)
            current_heading = " > ".join(heading_stack[1:] or heading_stack)
            heading_date = DATE_RE.search(heading_text)
            if heading_date:
                active_date = heading_date.group(1)
            if level == 1:
                title = heading_text
        if buffer and sum(len(value) + 1 for _, value in buffer) + len(line) > max_chars:
            flush()
        buffer.append((line_no, line))
        if not line.strip() and sum(len(value) + 1 for _, value in buffer) > max_chars // 2:
            flush()
    flush()
    return chunks


class ProjectIndex:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS documents (
                path TEXT PRIMARY KEY,
                modified_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                indexed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL REFERENCES documents(path) ON DELETE CASCADE,
                title TEXT NOT NULL,
                heading TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                content TEXT NOT NULL,
                modified_ns INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                document_date TEXT,
                term_counts TEXT NOT NULL,
                token_count INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);
            CREATE TABLE IF NOT EXISTS usage (
                day TEXT NOT NULL,
                user_id TEXT NOT NULL,
                ai_requests INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(day, user_id)
            );
            """
        )
        self.connection.commit()

    def refresh(self, loader: SafeSourceLoader) -> RefreshReport:
        with self._lock:
            return self._refresh_locked(loader)

    def _refresh_locked(self, loader: SafeSourceLoader) -> RefreshReport:
        documents, skipped = loader.load_all()
        existing = {
            row["path"]: row
            for row in self.connection.execute("SELECT path, modified_ns, size, sha256 FROM documents")
        }
        seen: set[str] = set()
        added = updated = unchanged = chunks_added = 0
        now = datetime.now(timezone.utc).isoformat()
        with self.connection:
            for document in documents:
                seen.add(document.relative_path)
                previous = existing.get(document.relative_path)
                if previous and previous["sha256"] == document.sha256:
                    unchanged += 1
                    if previous["modified_ns"] != document.modified_ns or previous["size"] != document.size:
                        self.connection.execute(
                            "UPDATE documents SET modified_ns=?, size=?, indexed_at=? WHERE path=?",
                            (document.modified_ns, document.size, now, document.relative_path),
                        )
                        self.connection.execute(
                            "UPDATE chunks SET modified_ns=? WHERE path=?",
                            (document.modified_ns, document.relative_path),
                        )
                    continue
                if previous:
                    updated += 1
                    self.connection.execute("DELETE FROM documents WHERE path=?", (document.relative_path,))
                else:
                    added += 1
                self.connection.execute(
                    "INSERT INTO documents(path, modified_ns, size, sha256, indexed_at) VALUES(?,?,?,?,?)",
                    (document.relative_path, document.modified_ns, document.size, document.sha256, now),
                )
                for chunk in chunk_document(document):
                    counts = Counter(tokenize(f"{chunk.title} {chunk.heading} {chunk.content}"))
                    self.connection.execute(
                        """INSERT INTO chunks(
                           path,title,heading,line_start,line_end,content,modified_ns,content_hash,
                           document_date,term_counts,token_count
                           ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            chunk.path, chunk.title, chunk.heading, chunk.line_start, chunk.line_end,
                            chunk.content, chunk.modified_ns, chunk.content_hash, chunk.document_date,
                            json.dumps(counts, ensure_ascii=False, separators=(",", ":")),
                            sum(counts.values()),
                        ),
                    )
                    chunks_added += 1
            removed_paths = set(existing) - seen
            for path in removed_paths:
                self.connection.execute("DELETE FROM documents WHERE path=?", (path,))
        return RefreshReport(
            scanned=len(documents), added=added, updated=updated, unchanged=unchanged,
            removed=len(set(existing) - seen), chunks=chunks_added, skipped=tuple(skipped),
        )

    def stats(self) -> dict[str, int | str | None]:
        with self._lock:
            document_count = self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            latest = self.connection.execute("SELECT MAX(indexed_at) FROM documents").fetchone()[0]
            return {"documents": document_count, "chunks": chunk_count, "last_refresh": latest}

    def search(self, query: str, limit: int = 8) -> list[SearchHit]:
        with self._lock:
            return self._search_locked(query, limit)

    def _search_locked(self, query: str, limit: int = 8) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        rows = list(self.connection.execute("SELECT * FROM chunks"))
        if not rows:
            return []
        counts_by_id = {row["id"]: json.loads(row["term_counts"]) for row in rows}
        total_documents = len(rows)
        doc_frequency = Counter()
        for token in set(query_tokens):
            doc_frequency[token] = sum(token in counts for counts in counts_by_id.values())
        average_length = sum(row["token_count"] for row in rows) / total_documents
        query_counts = Counter(query_tokens)
        scored: list[SearchHit] = []
        k1, b = 1.5, 0.75
        for row in rows:
            terms = counts_by_id[row["id"]]
            length = max(row["token_count"], 1)
            score = 0.0
            for token, qtf in query_counts.items():
                tf = terms.get(token, 0)
                if not tf:
                    continue
                df = doc_frequency[token]
                idf = math.log(1 + (total_documents - df + 0.5) / (df + 0.5))
                score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / average_length))) * min(qtf, 2)
            if score <= 0:
                continue
            chunk = Chunk(
                path=row["path"], title=row["title"], heading=row["heading"],
                line_start=row["line_start"], line_end=row["line_end"], content=row["content"],
                modified_ns=row["modified_ns"], content_hash=row["content_hash"],
                document_date=row["document_date"],
            )
            source_boost = 1.0
            if chunk.path == "README.md":
                source_boost = 1.7
            elif chunk.path == "docs/current_research_direction.md":
                source_boost = 2.0
            elif chunk.path == "docs/progress_log.md":
                source_boost = 1.35
            elif chunk.path == "docs/research_baseline_v1.3.md":
                source_boost = 1.3
            elif "protocol" in chunk.path or "results" in chunk.path:
                source_boost = 1.08
            scored.append(SearchHit(chunk=chunk, score=score * source_boost))
        return sorted(scored, key=lambda item: (-item.score, item.chunk.path, item.chunk.line_start))[:limit]

    def claim_ai_request(self, user_id: str, global_limit: int, user_limit: int) -> tuple[bool, str]:
        with self._lock:
            day = datetime.now(timezone.utc).date().isoformat()
            with self.connection:
                global_used = self.connection.execute(
                    "SELECT COALESCE(SUM(ai_requests),0) FROM usage WHERE day=?", (day,)
                ).fetchone()[0]
                user_used = self.connection.execute(
                    "SELECT COALESCE(ai_requests,0) FROM usage WHERE day=? AND user_id=?", (day, user_id)
                ).fetchone()
                user_count = user_used[0] if user_used else 0
                if global_used >= global_limit:
                    return False, "已達每日 AI 全域上限"
                if user_count >= user_limit:
                    return False, "已達個人每日 AI 上限"
                self.connection.execute(
                    """INSERT INTO usage(day,user_id,ai_requests) VALUES(?,?,1)
                       ON CONFLICT(day,user_id) DO UPDATE SET ai_requests=ai_requests+1""",
                    (day, user_id),
                )
        return True, "ok"

    def ai_usage_today(self) -> int:
        with self._lock:
            day = datetime.now(timezone.utc).date().isoformat()
            return int(self.connection.execute(
                "SELECT COALESCE(SUM(ai_requests),0) FROM usage WHERE day=?", (day,)
            ).fetchone()[0])
