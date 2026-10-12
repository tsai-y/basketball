from __future__ import annotations

import asyncio
import logging

import discord
from discord import app_commands

from .config import Settings
from .service import ProjectAssistant, split_discord_message


LOGGER = logging.getLogger(__name__)


def has_role_access(
    *,
    guild_id: int,
    member_role_ids: set[int],
    required_role_ids: frozenset[int],
    admin: bool,
    is_discord_admin: bool,
) -> bool:
    """Evaluate role access, treating a guild ID as Discord's @everyone role.

    Interaction member payloads may omit the implicit @everyone role even
    though its role ID is the guild ID.  That convention is supported only for
    ordinary commands; admin access always requires an explicit admin role or
    Discord's Administrator permission.
    """
    if admin and is_discord_admin:
        return True
    if not admin and guild_id in required_role_ids:
        return True
    return bool(member_role_ids & required_role_ids)


class ProjectDiscordClient(discord.Client):
    def __init__(self, settings: Settings, assistant: ProjectAssistant):
        intents = discord.Intents.none()
        intents.guilds = True
        super().__init__(intents=intents)
        self.settings = settings
        self.assistant = assistant
        self.tree = app_commands.CommandTree(self)
        self._register_commands()

    async def setup_hook(self) -> None:
        await asyncio.to_thread(self.assistant.refresh)
        if self.settings.sync_commands:
            for guild_id in self.settings.allowed_guild_ids:
                guild = discord.Object(id=guild_id)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)

    async def on_ready(self) -> None:
        LOGGER.info("Bot ready: user_id=%s", self.user.id if self.user else "unknown")

    def _authorized(self, interaction: discord.Interaction, *, admin: bool = False) -> tuple[bool, str]:
        if interaction.guild_id not in self.settings.allowed_guild_ids:
            return False, "此 Discord server 不在 allowlist。"
        if interaction.channel_id not in self.settings.allowed_channel_ids:
            return False, "此 channel 不在 allowlist。"
        member = interaction.user
        if not isinstance(member, discord.Member):
            return False, "無法驗證成員角色。"
        role_ids = {role.id for role in member.roles}
        required = self.settings.admin_role_ids if admin else self.settings.allowed_role_ids
        is_discord_admin = member.guild_permissions.administrator
        if not has_role_access(
            guild_id=interaction.guild_id,
            member_role_ids=role_ids,
            required_role_ids=required,
            admin=admin,
            is_discord_admin=is_discord_admin,
        ):
            return False, "角色未獲授權。"
        if not self.assistant.command_limiter.allow(str(member.id)):
            return False, "指令過於頻繁，請稍後再試。"
        return True, "ok"

    async def _respond(self, interaction: discord.Interaction, text: str) -> None:
        parts = split_discord_message(text, self.settings.discord_message_chars)
        await interaction.followup.send(parts[0], ephemeral=True, allowed_mentions=discord.AllowedMentions.none())
        for part in parts[1:]:
            await interaction.followup.send(part, ephemeral=True, allowed_mentions=discord.AllowedMentions.none())

    async def _guard(self, interaction: discord.Interaction, *, admin: bool = False) -> bool:
        ok, reason = self._authorized(interaction, admin=admin)
        if not ok:
            await interaction.response.send_message(reason, ephemeral=True)
            return False
        await interaction.response.defer(ephemeral=True, thinking=True)
        return True

    def _register_commands(self) -> None:
        @self.tree.command(name="status", description="顯示目前專案狀態與來源")
        async def status(interaction: discord.Interaction) -> None:
            if not await self._guard(interaction):
                return
            answer = await self.assistant.status(str(interaction.user.id))
            await self._respond(interaction, answer.text)

        @self.tree.command(name="next", description="顯示下一步、阻塞事項與原因")
        async def next_steps(interaction: discord.Interaction) -> None:
            if not await self._guard(interaction):
                return
            answer = await self.assistant.next_steps(str(interaction.user.id))
            await self._respond(interaction, answer.text)

        @self.tree.command(name="experiments", description="列出實驗狀態、結果與限制")
        async def experiments(interaction: discord.Interaction) -> None:
            if not await self._guard(interaction):
                return
            answer = await self.assistant.experiments(str(interaction.user.id))
            await self._respond(interaction, answer.text)

        @self.tree.command(name="why", description="查詢專案決策原因")
        @app_commands.describe(question="想了解原因的決策或方法")
        async def why(interaction: discord.Interaction, question: app_commands.Range[str, 1, 1000]) -> None:
            if not await self._guard(interaction):
                return
            answer = await self.assistant.answer(str(interaction.user.id), f"為什麼：{question}")
            await self._respond(interaction, answer.text)

        @self.tree.command(name="ask", description="以自然語言查詢專案文件")
        @app_commands.describe(question="專案問題")
        async def ask(interaction: discord.Interaction, question: app_commands.Range[str, 1, 1000]) -> None:
            if not await self._guard(interaction):
                return
            answer = await self.assistant.answer(str(interaction.user.id), question)
            await self._respond(interaction, answer.text)

        @self.tree.command(name="sources", description="顯示你最近一次答案的來源")
        async def sources(interaction: discord.Interaction) -> None:
            if not await self._guard(interaction):
                return
            await self._respond(interaction, self.assistant.sources(str(interaction.user.id)))

        @self.tree.command(name="refresh", description="重新索引已變更的 allowlisted 文件（管理員）")
        async def refresh(interaction: discord.Interaction) -> None:
            if not await self._guard(interaction, admin=True):
                return
            report = await asyncio.to_thread(self.assistant.refresh)
            await self._respond(
                interaction,
                f"索引更新完成：掃描 {report.scanned}、新增 {report.added}、更新 {report.updated}、"
                f"未變 {report.unchanged}、移除 {report.removed}、新建片段 {report.chunks}、略過 {len(report.skipped)}。",
            )

        @self.tree.command(name="health", description="檢查 Bot、索引與 AI provider 狀態")
        async def health(interaction: discord.Interaction) -> None:
            if not await self._guard(interaction):
                return
            await self._respond(interaction, self.assistant.health())


def run_discord(settings: Settings, assistant: ProjectAssistant) -> None:
    errors = settings.validate_for_discord()
    if errors:
        raise SystemExit("；".join(errors))
    client = ProjectDiscordClient(settings, assistant)
    client.run(settings.discord_token, log_handler=None)
