from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _ints(value: str) -> frozenset[int]:
    values: set[int] = set()
    for item in _csv(value):
        values.add(int(item))
    return frozenset(values)


def _bool(value: str, default: bool = False) -> bool:
    if not value:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    sources_config: Path
    discord_token: str | None
    allowed_guild_ids: frozenset[int]
    allowed_channel_ids: frozenset[int]
    allowed_role_ids: frozenset[int]
    admin_role_ids: frozenset[int]
    ai_provider: str
    codex_auth_mode: str
    codex_binary: str
    codex_home: Path
    codex_model: str | None
    codex_reasoning_effort: str
    codex_timeout_seconds: float
    ai_max_answer_chars: int
    max_context_chunks: int
    max_context_chars: int
    daily_ai_request_limit: int
    per_user_daily_ai_request_limit: int
    per_user_minute_command_limit: int
    log_level: str
    discord_message_chars: int
    sync_commands: bool

    @property
    def index_path(self) -> Path:
        return self.data_dir / "project_index.sqlite3"

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(os.environ.get("PROJECT_ROOT", "/repo")).resolve()
        data_dir = Path(os.environ.get("BOT_DATA_DIR", "/data")).resolve()
        default_sources = project_root / "config" / "project_sources.yaml"
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            sources_config=Path(os.environ.get("SOURCES_CONFIG", str(default_sources))).resolve(),
            discord_token=os.environ.get("DISCORD_TOKEN") or None,
            allowed_guild_ids=_ints(os.environ.get("DISCORD_GUILD_IDS", "")),
            allowed_channel_ids=_ints(os.environ.get("DISCORD_CHANNEL_IDS", "")),
            allowed_role_ids=_ints(os.environ.get("DISCORD_ALLOWED_ROLE_IDS", "")),
            admin_role_ids=_ints(os.environ.get("DISCORD_ADMIN_ROLE_IDS", "")),
            ai_provider=os.environ.get("AI_PROVIDER", "codex").strip().lower(),
            codex_auth_mode=os.environ.get("CODEX_AUTH_MODE", "chatgpt_login").strip().lower(),
            codex_binary=os.environ.get("CODEX_BINARY", "codex").strip() or "codex",
            codex_home=Path(os.environ.get("CODEX_HOME", str(data_dir / "codex-home"))).resolve(),
            codex_model=os.environ.get("CODEX_MODEL") or None,
            codex_reasoning_effort=os.environ.get("CODEX_REASONING_EFFORT", "low").strip().lower(),
            codex_timeout_seconds=float(os.environ.get("CODEX_TIMEOUT_SECONDS", "45")),
            ai_max_answer_chars=int(os.environ.get("AI_MAX_ANSWER_CHARS", "3500")),
            max_context_chunks=int(os.environ.get("MAX_CONTEXT_CHUNKS", "8")),
            max_context_chars=int(os.environ.get("MAX_CONTEXT_CHARS", "16000")),
            daily_ai_request_limit=int(os.environ.get("DAILY_AI_REQUEST_LIMIT", "40")),
            per_user_daily_ai_request_limit=int(os.environ.get("PER_USER_DAILY_AI_REQUEST_LIMIT", "5")),
            per_user_minute_command_limit=int(os.environ.get("PER_USER_MINUTE_COMMAND_LIMIT", "12")),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            discord_message_chars=min(int(os.environ.get("DISCORD_MESSAGE_CHARS", "1900")), 1950),
            sync_commands=_bool(os.environ.get("DISCORD_SYNC_COMMANDS", "true"), True),
        )

    def validate_for_discord(self) -> list[str]:
        errors: list[str] = []
        if not self.discord_token:
            errors.append("DISCORD_TOKEN 未設定")
        if not self.allowed_guild_ids:
            errors.append("DISCORD_GUILD_IDS 必須至少包含一個 server ID")
        if not self.allowed_channel_ids:
            errors.append("DISCORD_CHANNEL_IDS 必須至少包含一個 channel ID")
        if not self.allowed_role_ids:
            errors.append("DISCORD_ALLOWED_ROLE_IDS 必須至少包含一個 role ID")
        if not self.admin_role_ids:
            errors.append("DISCORD_ADMIN_ROLE_IDS 必須至少包含一個 role ID")
        return errors
