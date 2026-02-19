from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(chunk.strip().lower() for chunk in raw.split(",") if chunk.strip())


@dataclass(slots=True)
class Settings:
    discord_token: str = field(default_factory=lambda: os.getenv("DISCORD_TOKEN", ""))
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    github_org: str = field(default_factory=lambda: os.getenv("GITHUB_ORG", ""))
    repo_prefix: str = field(default_factory=lambda: os.getenv("NOTA_REPO_PREFIX", "nota-kb"))

    base_data_dir: Path = field(default_factory=lambda: Path(os.getenv("NOTA_DATA_DIR", "./data")))
    sqlite_path: Path = field(init=False)
    chroma_path: Path = field(init=False)

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    lm_studio_base_url: str = field(default_factory=lambda: os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"))
    lm_studio_model: str = field(default_factory=lambda: os.getenv("LM_STUDIO_MODEL", "local-model"))
    lm_studio_max_prompt_chars: int = field(default_factory=lambda: int(os.getenv("LM_STUDIO_MAX_PROMPT_CHARS", "3500")))
    lm_studio_context_window: int = field(default_factory=lambda: int(os.getenv("LM_STUDIO_CONTEXT_WINDOW", "2048")))
    lm_studio_reserved_tokens: int = field(default_factory=lambda: int(os.getenv("LM_STUDIO_RESERVED_TOKENS", "512")))
    lm_studio_chars_per_token: float = field(default_factory=lambda: float(os.getenv("LM_STUDIO_CHARS_PER_TOKEN", "3.0")))

    default_branch: str = field(default_factory=lambda: os.getenv("DEFAULT_BRANCH", "main"))
    thread_parent_channel_id: int = field(default_factory=lambda: int(os.getenv("NOTA_THREAD_PARENT_CHANNEL_ID", "0")))
    admin_user_ids: tuple[int, ...] = field(init=False)

    # Security controls
    allowed_local_file_extensions: tuple[str, ...] = field(
        default_factory=lambda: _parse_csv(os.getenv("NOTA_ALLOWED_LOCAL_FILE_EXTENSIONS", "txt,json,pdf"))
    )
    local_file_max_bytes: int = field(default_factory=lambda: int(os.getenv("NOTA_LOCAL_FILE_MAX_BYTES", str(3 * 1024 * 1024))))
    web_max_bytes: int = field(default_factory=lambda: int(os.getenv("NOTA_WEB_MAX_BYTES", str(2 * 1024 * 1024))))
    web_request_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("NOTA_WEB_REQUEST_TIMEOUT_SECONDS", "15")))
    allowed_web_domains: tuple[str, ...] = field(
        default_factory=lambda: _parse_csv(os.getenv("NOTA_ALLOWED_WEB_DOMAINS", ""))
    )
    command_cooldown_seconds: int = field(default_factory=lambda: int(os.getenv("NOTA_COMMAND_COOLDOWN_SECONDS", "5")))
    command_window_seconds: int = field(default_factory=lambda: int(os.getenv("NOTA_COMMAND_WINDOW_SECONDS", "60")))
    command_max_per_window: int = field(default_factory=lambda: int(os.getenv("NOTA_COMMAND_MAX_PER_WINDOW", "12")))

    def __post_init__(self) -> None:
        self.sqlite_path = self.base_data_dir / "nota.sqlite3"
        self.chroma_path = self.base_data_dir / "chroma"
        self.base_data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.admin_user_ids = _parse_int_tuple(os.getenv("NOTA_ADMIN_USER_IDS", "").strip())

    def validate(self) -> None:
        missing = []
        if not self.discord_token:
            missing.append("DISCORD_TOKEN")
        if not self.github_token:
            missing.append("GITHUB_TOKEN")
        if not self.thread_parent_channel_id:
            missing.append("NOTA_THREAD_PARENT_CHANNEL_ID")
        if self.local_file_max_bytes <= 0:
            missing.append("NOTA_LOCAL_FILE_MAX_BYTES(>0)")
        if self.web_max_bytes <= 0:
            missing.append("NOTA_WEB_MAX_BYTES(>0)")
        if self.command_max_per_window <= 0:
            missing.append("NOTA_COMMAND_MAX_PER_WINDOW(>0)")
        if missing:
            raise ValueError(f"Missing or invalid environment variables: {', '.join(missing)}")


settings = Settings()
