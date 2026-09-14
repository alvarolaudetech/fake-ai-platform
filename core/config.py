"""Application-wide configuration loaded from environment variables.

All settings are centralized here via pydantic-settings so that every
module (routes, services, workers) shares a single validated source of
truth instead of reading os.environ directly.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- General ---
    app_name: str = "fake-ai-platform"
    environment: Literal["local", "staging", "production"] = "local"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # --- Security / Auth ---
    jwt_secret_key: str = Field(..., description="HMAC secret used to sign access tokens")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 14
    api_key_header_name: str = "X-API-Key"

    # --- Database ---
    database_url: PostgresDsn = Field(
        default="postgresql+psycopg2://platform:platform@localhost:5432/fake_ai_platform"
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_recycle_seconds: int = 1800

    # --- Cache / rate limiting ---
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    default_rate_limit_per_minute: int = 60
    default_daily_token_quota: int = 1_000_000

    # --- Upstream LLM providers ---
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    local_inference_base_url: str = "http://localhost:8001/v1"
    request_timeout_seconds: float = 30.0
    max_retries: int = 3

    # --- Observability ---
    log_level: str = "INFO"
    enable_prometheus: bool = True

    @field_validator("jwt_secret_key")
    @classmethod
    def _reject_placeholder_secret(cls, value: str) -> str:
        if value in {"", "changeme", "secret"}:
            raise ValueError("jwt_secret_key must be overridden via environment in non-local envs")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so env parsing only happens once."""
    return Settings()
