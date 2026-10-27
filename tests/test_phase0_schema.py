import subprocess
import sys
from pathlib import Path

from src.config import Settings
from src.db.models import Base
from src.db.session import engine

EXPECTED_SERVING_TABLES = {
    "players",
    "player_seasons",
    "valuations",
    "player_current_features",
    "model_versions",
    "predictions_log",
}


def test_six_serving_tables_registered() -> None:
    actual_tables = set(Base.metadata.tables.keys())
    assert actual_tables == EXPECTED_SERVING_TABLES


def test_all_columns_strictly_snake_case() -> None:
    for table_name, table in Base.metadata.tables.items():
        for col in table.columns:
            assert col.name == col.name.lower(), (
                f"Column {table_name}.{col.name} is not snake_case"
            )
            assert " " not in col.name


def test_players_has_no_fbref_id_and_uses_trgm_index() -> None:
    players = Base.metadata.tables["players"]
    col_names = {col.name for col in players.columns}
    assert "fbref_player_id" not in col_names
    assert "tm_player_id" in col_names
    index_names = {idx.name for idx in players.indexes}
    assert "ix_players_normalized_name_trgm" in index_names
    assert "ix_players_normalized_name" not in index_names


def test_player_seasons_unique_key_includes_tm_club_id() -> None:
    player_seasons = Base.metadata.tables["player_seasons"]
    col_names = {col.name for col in player_seasons.columns}
    assert "tm_club_id" in col_names
    uq_constraints = {
        c.name: tuple(col.name for col in c.columns)
        for c in player_seasons.constraints
        if c.name == "uq_player_seasons_player_season_club"
    }
    assert uq_constraints == {
        "uq_player_seasons_player_season_club": ("player_id", "season", "tm_club_id")
    }


def test_model_versions_single_active_partial_unique_index() -> None:
    model_versions = Base.metadata.tables["model_versions"]
    indexes = {idx.name: idx for idx in model_versions.indexes}
    assert "ix_model_versions_single_active" in indexes
    idx = indexes["ix_model_versions_single_active"]
    assert idx.unique is True


def test_player_current_features_schema() -> None:
    table = Base.metadata.tables["player_current_features"]
    col_names = {col.name for col in table.columns}
    assert col_names == {
        "player_id",
        "as_of_date",
        "data_version",
        "league_code",
        "trailing_league_median_eur",
        "features",
        "updated_at",
    }
    pk_cols = [col.name for col in table.primary_key.columns]
    assert pk_cols == ["player_id"]


def test_no_raw_sql_schema_file_exists() -> None:
    """Ensure Alembic is the sole source of truth and sql/001_schema.sql does not exist."""
    assert not Path("sql/001_schema.sql").exists()


def test_settings_defaults_isolated_from_env_file() -> None:
    isolated = Settings(_env_file=None)
    assert not hasattr(isolated, "model_version")
    assert not hasattr(isolated, "data_version")
    assert isolated.metadata_refresh_seconds == 60
    assert isolated.raw_data_dir == Path("data/raw")
    assert isolated.raw_manifest_path == Path("data/raw/MANIFEST.json")
    assert isolated.clean_data_dir == Path("data/clean")
    assert isolated.features_data_dir == Path("data/features")


def test_connection_pool_sizing() -> None:
    assert engine.pool.size() == 5
    assert engine.pool._max_overflow == 10


def test_alembic_offline_sql_generation() -> None:
    """Verify Alembic generates complete DDL for 0001_initial_serving_schema."""
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        capture_output=True,
        text=True,
        check=True,
    )
    sql_output = result.stdout
    for table_name in EXPECTED_SERVING_TABLES:
        assert f"CREATE TABLE {table_name}" in sql_output
    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm;" in sql_output
    assert "WHERE is_active IS TRUE" in sql_output
