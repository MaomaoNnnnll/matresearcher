"""Environment loading: project .env (non-secret) + user secrets.env (API keys).

Security design:
- Project `.env` contains ONLY non-secret config (base URLs, model names, paths).
- API keys live in `~/.matresearcher/secrets.env` (user home directory, OUTSIDE
  the project), so uploading/pushing/packaging the project can never leak keys.
- Keys from secrets.env override earlier values (override=True), so editing the
  file takes effect immediately without opening a new shell.
"""
from __future__ import annotations

from pathlib import Path

_SECRETS_FILE = Path.home() / ".matresearcher" / "secrets.env"


def load_env_files() -> None:
    """Load project .env first, then user secrets.env (keys win)."""
    from dotenv import load_dotenv

    # 1) Project .env: non-secret config (base_url, model names, paths, flags)
    load_dotenv(override=False)

    # 2) User secrets: API keys stored outside the project directory
    if _SECRETS_FILE.exists():
        load_dotenv(_SECRETS_FILE, override=True)
