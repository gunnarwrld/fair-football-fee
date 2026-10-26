"""Initial serving schema with 6 operational tables.

Revision ID: 0001_initial_serving_schema
Revises: None
Create Date: 2026-10-26 20:28:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_serving_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    op.create_table(
        "players",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("tm_player_id", sa.BigInteger(), nullable=False),
        sa.Column("full_name", sa.String(length=200), nullable=False),
        sa.Column("normalized_name", sa.String(length=200), nullable=False),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("nationality", sa.String(length=100), nullable=True),
        sa.Column("position_group", sa.String(length=8), nullable=False),
        sa.Column("primary_position", sa.String(length=64), nullable=True),
        sa.Column("preferred_foot", sa.String(length=16), nullable=True),
        sa.Column("height_cm", sa.Integer(), nullable=True),
        sa.Column("current_club", sa.String(length=150), nullable=True),
        sa.Column("current_league_code", sa.String(length=16), nullable=False),
        sa.Column("contract_expires_at", sa.Date(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "position_group IN ('GK', 'DEF', 'MID', 'FWD')",
            name="ck_players_position_group",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tm_player_id", name="uq_players_tm_player_id"),
    )
    op.create_index(
        "ix_players_normalized_name_trgm",
        "players",
        ["normalized_name"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"normalized_name": "gin_trgm_ops"},
    )
    op.create_index(
        "ix_players_current_league_code", "players", ["current_league_code"], unique=False
    )

    op.create_table(
        "player_seasons",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("season", sa.String(length=16), nullable=False),
        sa.Column("season_end_date", sa.Date(), nullable=False),
        sa.Column("tm_club_id", sa.BigInteger(), nullable=False),
        sa.Column("club_name", sa.String(length=150), nullable=True),
        sa.Column("league_code", sa.String(length=16), nullable=False),
        sa.Column("appearances", sa.Integer(), server_default="0", nullable=False),
        sa.Column("minutes_played", sa.Integer(), server_default="0", nullable=False),
        sa.Column("goals", sa.Integer(), server_default="0", nullable=False),
        sa.Column("assists", sa.Integer(), server_default="0", nullable=False),
        sa.Column("yellow_cards", sa.Integer(), server_default="0", nullable=False),
        sa.Column("red_cards", sa.Integer(), server_default="0", nullable=False),
        sa.Column("clean_sheets", sa.Integer(), server_default="0", nullable=False),
        sa.Column("goals_conceded", sa.Integer(), server_default="0", nullable=False),
        sa.Column("goals_per90", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("assists_per90", sa.Float(), server_default="0.0", nullable=False),
        sa.Column("xg_per90", sa.Float(), nullable=True),
        sa.Column("xa_per90", sa.Float(), nullable=True),
        sa.Column("prog_actions_per90", sa.Float(), nullable=True),
        sa.Column(
            "has_advanced_stats", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("minutes_played >= 0", name="ck_player_seasons_minutes_nonneg"),
        sa.ForeignKeyConstraint(["player_id"], ["players.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "player_id", "season", "tm_club_id", name="uq_player_seasons_player_season_club"
        ),
    )
    op.create_index(
        "ix_player_seasons_player_id_season_end",
        "player_seasons",
        ["player_id", "season_end_date"],
        unique=False,
    )

    op.create_table(
        "valuations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("valuation_date", sa.Date(), nullable=False),
        sa.Column("market_value_eur", sa.BigInteger(), nullable=False),
        sa.Column("club_at_valuation", sa.String(length=150), nullable=True),
        sa.Column("league_at_valuation", sa.String(length=16), nullable=False),
        sa.Column("age_at_valuation", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("market_value_eur > 0", name="ck_valuations_positive_eur"),
        sa.ForeignKeyConstraint(["player_id"], ["players.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("player_id", "valuation_date", name="uq_valuations_player_date"),
    )
    op.create_index(
        "ix_valuations_league_date",
        "valuations",
        ["league_at_valuation", "valuation_date"],
        unique=False,
    )

    op.create_table(
        "player_current_features",
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("data_version", sa.String(length=32), nullable=False),
        sa.Column("league_code", sa.String(length=16), nullable=False),
        sa.Column("trailing_league_median_eur", sa.Float(), nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "trailing_league_median_eur > 0",
            name="ck_player_current_features_positive_median",
        ),
        sa.ForeignKeyConstraint(["player_id"], ["players.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("player_id"),
    )
    op.create_index(
        "ix_player_current_features_data_version",
        "player_current_features",
        ["data_version"],
        unique=False,
    )

    op.create_table(
        "model_versions",
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("model_type", sa.String(length=64), nullable=False),
        sa.Column("data_version", sa.String(length=32), nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("artifact_path", sa.String(length=255), nullable=False),
        sa.Column("feature_list", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("conformal_quantiles", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("model_version"),
    )
    op.create_index(
        "ix_model_versions_single_active",
        "model_versions",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active IS TRUE"),
    )

    op.create_table(
        "predictions_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("data_version", sa.String(length=32), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=True),
        sa.Column("is_what_if", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("input_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("predicted_value_eur", sa.Float(), nullable=False),
        sa.Column("interval_low_eur", sa.Float(), nullable=False),
        sa.Column("interval_high_eur", sa.Float(), nullable=False),
        sa.Column("top_shap_drivers", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("cache_status", sa.String(length=16), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "cache_status IN ('HIT', 'MISS', 'BYPASS')",
            name="ck_predictions_log_cache_status",
        ),
        sa.ForeignKeyConstraint(["model_version"], ["model_versions.model_version"]),
        sa.ForeignKeyConstraint(["player_id"], ["players.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_log_request_id", "predictions_log", ["request_id"], unique=False)
    op.create_index(
        "ix_predictions_log_model_version", "predictions_log", ["model_version"], unique=False
    )
    op.create_index("ix_predictions_log_player_id", "predictions_log", ["player_id"], unique=False)
    op.create_index("ix_predictions_log_created_at", "predictions_log", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_table("predictions_log")
    op.drop_table("model_versions")
    op.drop_table("player_current_features")
    op.drop_table("valuations")
    op.drop_table("player_seasons")
    op.drop_table("players")
