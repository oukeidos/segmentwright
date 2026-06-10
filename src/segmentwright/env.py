from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values


def load_gemini_api_key(env_file: Path) -> str:
    env_file = env_file.expanduser()
    if not env_file.exists():
        raise FileNotFoundError(f"Gemini env file not found: {env_file}")
    values = dotenv_values(env_file)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = values.get(name)
        if value and value.strip():
            return value.strip()
    raise ValueError(
        f"Gemini API key not found in {env_file}; expected GEMINI_API_KEY or GOOGLE_API_KEY"
    )
