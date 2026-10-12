from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class DeploymentFileTests(unittest.TestCase):
    def test_compose_enforces_read_only_repository_and_restart(self) -> None:
        compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
        service = compose["services"]["project-assistant-bot"]
        self.assertEqual(service["restart"], "unless-stopped")
        self.assertTrue(service["read_only"])
        self.assertIn("../:/repo:ro", service["volumes"])
        self.assertEqual(service["cap_drop"], ["ALL"])
        self.assertIn("no-new-privileges:true", service["security_opt"])
        self.assertEqual(service["environment"]["CODEX_AUTH_MODE"], "chatgpt_login")
        self.assertNotIn("CODEX_ACCESS_TOKEN", service["environment"])
        self.assertNotIn("OPENAI_API_KEY", service["environment"])

    def test_systemd_and_dockerfile_hardening(self) -> None:
        unit = (ROOT / "deploy" / "basketball-project-bot.service").read_text(encoding="utf-8")
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("ReadOnlyPaths=/home/tsai/projects/basketball", unit)
        self.assertIn("NoNewPrivileges=true", unit)
        self.assertIn(
            "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6 AF_NETLINK", unit
        )
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("USER bot", dockerfile)
        self.assertIn("bubblewrap", dockerfile)
        self.assertIn("@openai/codex@${CODEX_VERSION}", dockerfile)
        self.assertIn('["python", "-m", "project_assistant_bot"]', dockerfile)

    def test_example_environment_has_no_real_credentials(self) -> None:
        example = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("DISCORD_TOKEN=replace_with_discord_bot_token", example)
        self.assertIn("CODEX_AUTH_MODE=chatgpt_login", example)
        self.assertNotIn("CODEX_ACCESS_TOKEN=", example)
        self.assertNotIn("OPENAI_API_KEY=", example)
        self.assertNotRegex(example, r"sk-[A-Za-z0-9_-]{20,}")
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("openai==", requirements)


if __name__ == "__main__":
    unittest.main()
