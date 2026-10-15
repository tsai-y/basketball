from __future__ import annotations

import unittest

from project_assistant_bot.discord_bot import has_role_access


class DiscordAuthorizationTests(unittest.TestCase):
    def test_guild_id_allows_everyone_for_ordinary_commands(self) -> None:
        self.assertTrue(
            has_role_access(
                guild_id=100,
                member_role_ids=set(),
                required_role_ids=frozenset({100}),
                admin=False,
                is_discord_admin=False,
            )
        )

    def test_guild_id_does_not_grant_admin_access(self) -> None:
        self.assertFalse(
            has_role_access(
                guild_id=100,
                member_role_ids=set(),
                required_role_ids=frozenset({100}),
                admin=True,
                is_discord_admin=False,
            )
        )

    def test_explicit_roles_and_discord_administrator(self) -> None:
        self.assertTrue(
            has_role_access(
                guild_id=100,
                member_role_ids={200},
                required_role_ids=frozenset({200}),
                admin=True,
                is_discord_admin=False,
            )
        )
        self.assertTrue(
            has_role_access(
                guild_id=100,
                member_role_ids=set(),
                required_role_ids=frozenset({300}),
                admin=True,
                is_discord_admin=True,
            )
        )

    def test_unlisted_role_is_rejected(self) -> None:
        self.assertFalse(
            has_role_access(
                guild_id=100,
                member_role_ids={201},
                required_role_ids=frozenset({200}),
                admin=False,
                is_discord_admin=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
