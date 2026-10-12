from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .index import SearchHit


SUBJECT_RE = re.compile(r"\b(A1|A2|B|C|Gate\s*1[ABC]|Experiment\s*D|Pre-release)\b", re.IGNORECASE)
STATUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("unresolved", re.compile(r"\b(unresolved|未決|未凍結|尚未決定|仍待確認)\b", re.IGNORECASE)),
    ("frozen", re.compile(r"\b(frozen|(?<!未)凍結|封存)\b", re.IGNORECASE)),
    ("completed", re.compile(r"\b(completed|complete|完成|已執行)\b", re.IGNORECASE)),
    ("paused", re.compile(r"\b(paused|暫停)\b", re.IGNORECASE)),
    ("not_started", re.compile(r"\b(not started|尚未開始|未執行|沒有啟動|尚未評估|未評估|仍未知)\b", re.IGNORECASE)),
    ("out_of_scope", re.compile(r"\b(out of scope|不在目前範圍|不納入目前範圍)\b", re.IGNORECASE)),
    ("planned", re.compile(r"\b(next|planned|規劃|下一步|待確認)\b", re.IGNORECASE)),
)


@dataclass(frozen=True)
class EvidenceConflict:
    subject: str
    older_status: str
    older_date: str | None
    older_citation: str
    newer_status: str
    newer_date: str | None
    newer_citation: str


def _nearest_status(text: str, position: int) -> str | None:
    matches: list[tuple[int, str]] = []
    for name, pattern in STATUS_PATTERNS:
        matches.extend((match.start(), name) for match in pattern.finditer(text))
    if not matches:
        return None
    return min(matches, key=lambda item: abs(item[0] - position))[1]


def detect_conflicts(hits: Iterable[SearchHit]) -> list[EvidenceConflict]:
    observations: dict[str, list[tuple[str, str | None, str, int]]] = {}
    for hit in hits:
        authority = 1
        if hit.chunk.path == "docs/current_research_direction.md":
            authority = 5
        elif hit.chunk.path in {"README.md", "docs/research_baseline_v1.3.md"}:
            authority = 4
        elif "protocol" in hit.chunk.path or "results" in hit.chunk.path:
            authority = 3
        elif hit.chunk.path == "docs/progress_log.md":
            authority = 2
        for sentence in re.split(r"(?<=[。.!?；;，])\s*|\n+", hit.chunk.content):
            for subject_match in SUBJECT_RE.finditer(sentence):
                status = _nearest_status(sentence, subject_match.start())
                if not status:
                    continue
                subject = subject_match.group(0)
                key = re.sub(r"\s+", "", subject).upper()
                observations.setdefault(key, []).append(
                    (status, hit.chunk.document_date, hit.chunk.citation, authority)
                )

    conflicts: list[EvidenceConflict] = []
    compatible = {frozenset({"completed", "frozen"}), frozenset({"completed", "planned"})}
    for subject, items in observations.items():
        unique: dict[tuple[str, str], tuple[str, str | None, str, int]] = {}
        for item in items:
            unique[(item[0], item[2])] = item
        ordered = sorted(unique.values(), key=lambda item: (item[1] or "", item[3], item[2]))
        newer = ordered[-1]
        older = next(
            (
                item for item in ordered
                if item[0] != newer[0] and frozenset({item[0], newer[0]}) not in compatible
            ),
            None,
        )
        if older:
            conflicts.append(
                EvidenceConflict(
                    subject=subject, older_status=older[0], older_date=older[1], older_citation=older[2],
                    newer_status=newer[0], newer_date=newer[1], newer_citation=newer[2],
                )
            )
    return conflicts
