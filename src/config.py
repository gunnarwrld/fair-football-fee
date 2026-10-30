from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://fff_user:fff_pass@localhost:5433/fair_football_fee"
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl_seconds: int = 3600
    redis_lock_timeout_seconds: int = 5
    # Redis is optional: short timeouts, and after an error the API skips it for this long
    redis_socket_timeout_seconds: float = 0.25
    redis_retry_after_seconds: float = 30.0
    metadata_refresh_seconds: int = 60

    # Both active model_version and data_version are resolved from the database
    # (model_versions.is_active and player_current_features.data_version) and
    # refreshed periodically (~60s) rather than hardcoded in config.

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    api_v1_prefix: str = "/v1"
    # Per client IP, applied to POST /v1/valuations only (the one path a client can make uncacheable)
    rate_limit_per_minute: int = 60
    # Per worker: past this many requests in flight, answer 503 at once instead of growing a queue
    max_concurrent_requests: int = 16
    # How long a request waits for a free database connection before a 503 (SQLAlchemy's default: 30 s)
    db_pool_timeout_seconds: float = 2.0
    cors_origins: list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def raw_manifest_path(self) -> Path:
        return self.raw_data_dir / "MANIFEST.json"

    @property
    def clean_data_dir(self) -> Path:
        return self.data_dir / "clean"

    @property
    def features_data_dir(self) -> Path:
        return self.data_dir / "features"


settings = Settings()
