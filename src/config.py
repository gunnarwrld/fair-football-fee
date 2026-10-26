from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://fff_user:fff_pass@localhost:5432/fair_football_fee"
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl_seconds: int = 3600
    redis_lock_timeout_seconds: int = 5

    # Active model version fallback; data_version is stamped by ingestion into
    # data/raw/MANIFEST.json and read from player_current_features at API startup.
    model_version: str = "v1.0.0"

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")

    api_v1_prefix: str = "/v1"
    rate_limit_per_minute: int = 60
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
