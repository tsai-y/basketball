from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Iterable

import yaml


ALLOWED_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".py"}
FORBIDDEN_PARTS = {
    ".env", ".git", ".ssh", "archive", "artifacts", "checkpoints", "credentials",
    "dataset", "datasets", "node_modules", "papers", "private", "secrets", "tokens",
}
SECRET_NAME_RE = re.compile(
    r"(?:^|[._-])(secrets?|credentials?|password|passwd|tokens?|private[_-]?key|id_rsa)(?:[._-]|$)",
    re.IGNORECASE,
)
SECRET_CONTENT_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?:api[_-]?key|client[_-]?secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourcePolicy:
    exact: tuple[str, ...]
    globs: tuple[str, ...]
    deny: tuple[str, ...] = ()
    max_file_bytes: int = 1_000_000
    max_total_bytes: int = 12_000_000
    max_files: int = 250

    @classmethod
    def load(cls, path: Path) -> "SourcePolicy":
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        allow = raw.get("allow", {})
        limits = raw.get("limits", {})
        return cls(
            exact=tuple(allow.get("exact", [])),
            globs=tuple(allow.get("globs", [])),
            deny=tuple(raw.get("deny", [])),
            max_file_bytes=int(limits.get("max_file_bytes", 1_000_000)),
            max_total_bytes=int(limits.get("max_total_bytes", 12_000_000)),
            max_files=int(limits.get("max_files", 250)),
        )

    def permits(self, relative: str) -> bool:
        normalized = relative.replace("\\", "/").lstrip("./")
        if any(fnmatch.fnmatchcase(normalized, pattern) for pattern in self.deny):
            return False
        return normalized in self.exact or any(fnmatch.fnmatchcase(normalized, pattern) for pattern in self.globs)


@dataclass(frozen=True)
class SafeDocument:
    relative_path: str
    content: str
    modified_ns: int
    size: int
    sha256: str


class UnsafeSourceError(ValueError):
    pass


class SafeSourceLoader:
    def __init__(self, root: Path, policy: SourcePolicy):
        self.root = root.resolve(strict=True)
        self.policy = policy

    def discover(self) -> tuple[list[Path], list[str]]:
        candidates: dict[str, Path] = {}
        skipped: list[str] = []
        for relative in self.policy.exact:
            path = self.root / relative
            if path.exists() or path.is_symlink():
                candidates[relative] = path
        for pattern in self.policy.globs:
            for path in self.root.glob(pattern):
                if path.is_file() or path.is_symlink():
                    relative = path.relative_to(self.root).as_posix()
                    candidates[relative] = path

        accepted: list[Path] = []
        total = 0
        for relative, path in sorted(candidates.items()):
            try:
                size = self._validate_path(path, relative)
            except UnsafeSourceError as exc:
                skipped.append(f"{relative}: {exc}")
                continue
            if len(accepted) >= self.policy.max_files:
                skipped.append(f"{relative}: 超過 max_files")
                continue
            if total + size > self.policy.max_total_bytes:
                skipped.append(f"{relative}: 超過 max_total_bytes")
                continue
            accepted.append(path)
            total += size
        return accepted, skipped

    def _validate_path(self, path: Path, relative: str) -> int:
        normalized = Path(relative)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise UnsafeSourceError("path traversal")
        if not self.policy.permits(relative):
            raise UnsafeSourceError("不在 allowlist")
        lowered_parts = {part.lower() for part in normalized.parts}
        if lowered_parts & FORBIDDEN_PARTS:
            raise UnsafeSourceError("命中禁止路徑")
        if SECRET_NAME_RE.search(normalized.name):
            raise UnsafeSourceError("疑似 secret 檔名")
        if normalized.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise UnsafeSourceError("不允許的副檔名")
        if path.is_symlink():
            raise UnsafeSourceError("不讀取 symlink")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(self.root):
            raise UnsafeSourceError("symlink escape")
        if not resolved.is_file():
            raise UnsafeSourceError("不是一般檔案")
        size = resolved.stat().st_size
        if size > self.policy.max_file_bytes:
            raise UnsafeSourceError("檔案過大")
        return size

    def read(self, path: Path) -> SafeDocument:
        relative = path.relative_to(self.root).as_posix()
        size = self._validate_path(path, relative)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened_stat = os.fstat(descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise UnsafeSourceError("不是一般檔案")
            chunks: list[bytes] = []
            remaining = self.policy.max_file_bytes + 1
            while remaining > 0:
                block = os.read(descriptor, min(65_536, remaining))
                if not block:
                    break
                chunks.append(block)
                remaining -= len(block)
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
        if len(raw) != size:
            raise UnsafeSourceError("讀取期間檔案大小變更")
        if b"\x00" in raw:
            raise UnsafeSourceError("疑似 binary")
        content = raw.decode("utf-8")
        if SECRET_CONTENT_RE.search(content):
            raise UnsafeSourceError("內容疑似包含 secret")
        if path.suffix.lower() == ".json":
            json.loads(content)
        return SafeDocument(
            relative_path=relative,
            content=content,
            modified_ns=opened_stat.st_mtime_ns,
            size=size,
            sha256=hashlib.sha256(raw).hexdigest(),
        )

    def load_all(self) -> tuple[list[SafeDocument], list[str]]:
        paths, skipped = self.discover()
        documents: list[SafeDocument] = []
        for path in paths:
            try:
                documents.append(self.read(path))
            except (OSError, UnicodeError, ValueError, UnsafeSourceError) as exc:
                skipped.append(f"{path.relative_to(self.root).as_posix()}: {exc}")
        return documents, skipped


def iter_safe_relative_paths(root: Path, policy: SourcePolicy) -> Iterable[str]:
    loader = SafeSourceLoader(root, policy)
    paths, _ = loader.discover()
    return (path.relative_to(loader.root).as_posix() for path in paths)
