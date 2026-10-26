from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for operational serving tables.

    Note: `onupdate=func.now()` only fires on SQLAlchemy ORM unit-of-work updates.
    All bulk upserts (`INSERT ... ON CONFLICT DO UPDATE`) in Phases 1 & 2 MUST
    explicitly set `updated_at=func.now()` in `set_={...}`.
    """


class Player(Base):
    __tablename__ = "players"
    __table_args__ = (
        UniqueConstraint("tm_player_id", name="uq_players_tm_player_id"),
        CheckConstraint(
            "position_group IN ('GK', 'DEF', 'MID', 'FWD')",
            name="ck_players_position_group",
        ),
        Index(
            "ix_players_normalized_name_trgm",
            "normalized_name",
            postgresql_using="gin",
            postgresql_ops={"normalized_name": "gin_trgm_ops"},
        ),
        Index("ix_players_current_league_code", "current_league_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tm_player_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    nationality: Mapped[str | None] = mapped_column(String(100), nullable=True)
    position_group: Mapped[str] = mapped_column(String(8), nullable=False)
    primary_position: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preferred_foot: Mapped[str | None] = mapped_column(String(16), nullable=True)
    height_cm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_club: Mapped[str | None] = mapped_column(String(150), nullable=True)
    current_league_code: Mapped[str] = mapped_column(String(16), nullable=False)
    contract_expires_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    seasons: Mapped[list["PlayerSeason"]] = relationship(
        back_populates="player", cascade="all, delete-orphan"
    )
    valuations: Mapped[list["Valuation"]] = relationship(
        back_populates="player", cascade="all, delete-orphan"
    )
    current_features: Mapped["PlayerCurrentFeatures | None"] = relationship(
        back_populates="player", cascade="all, delete-orphan", uselist=False
    )


class PlayerSeason(Base):
    __tablename__ = "player_seasons"
    __table_args__ = (
        UniqueConstraint(
            "player_id", "season", "tm_club_id", name="uq_player_seasons_player_season_club"
        ),
        CheckConstraint("minutes_played >= 0", name="ck_player_seasons_minutes_nonneg"),
        Index("ix_player_seasons_player_id_season_end", "player_id", "season_end_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False
    )
    season: Mapped[str] = mapped_column(String(16), nullable=False)
    season_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    tm_club_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    club_name: Mapped[str | None] = mapped_column(String(150), nullable=True)
    league_code: Mapped[str] = mapped_column(String(16), nullable=False)
    appearances: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    minutes_played: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    goals: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    assists: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    yellow_cards: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    red_cards: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    clean_sheets: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    goals_conceded: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    goals_per90: Mapped[float] = mapped_column(Float, server_default="0.0", nullable=False)
    assists_per90: Mapped[float] = mapped_column(Float, server_default="0.0", nullable=False)
    xg_per90: Mapped[float | None] = mapped_column(Float, nullable=True)
    xa_per90: Mapped[float | None] = mapped_column(Float, nullable=True)
    prog_actions_per90: Mapped[float | None] = mapped_column(Float, nullable=True)
    has_advanced_stats: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    player: Mapped["Player"] = relationship(back_populates="seasons")


class Valuation(Base):
    __tablename__ = "valuations"
    __table_args__ = (
        UniqueConstraint("player_id", "valuation_date", name="uq_valuations_player_date"),
        CheckConstraint("market_value_eur > 0", name="ck_valuations_positive_eur"),
        Index("ix_valuations_league_date", "league_at_valuation", "valuation_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id", ondelete="CASCADE"), nullable=False
    )
    valuation_date: Mapped[date] = mapped_column(Date, nullable=False)
    market_value_eur: Mapped[int] = mapped_column(BigInteger, nullable=False)
    club_at_valuation: Mapped[str | None] = mapped_column(String(150), nullable=True)
    league_at_valuation: Mapped[str] = mapped_column(String(16), nullable=False)
    age_at_valuation: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    player: Mapped["Player"] = relationship(back_populates="valuations")


class PlayerCurrentFeatures(Base):
    """Precomputed latest feature vector per player written by the Phase 2 pipeline.

    Allows GET /v1/players/{id}/valuation and POST /v1/valuations to read a single
    row and apply what-if overrides without recomputing rolling stats at request time.
    """

    __tablename__ = "player_current_features"
    __table_args__ = (
        CheckConstraint(
            "trailing_league_median_eur > 0",
            name="ck_player_current_features_positive_median",
        ),
        Index("ix_player_current_features_data_version", "data_version"),
    )

    player_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("players.id", ondelete="CASCADE"), primary_key=True
    )
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    data_version: Mapped[str] = mapped_column(String(32), nullable=False)
    league_code: Mapped[str] = mapped_column(String(16), nullable=False)
    trailing_league_median_eur: Mapped[float] = mapped_column(Float, nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    player: Mapped["Player"] = relationship(back_populates="current_features")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        Index(
            "ix_model_versions_single_active",
            "is_active",
            unique=True,
            postgresql_where=text("is_active IS TRUE"),
        ),
    )

    model_version: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    data_version: Mapped[str] = mapped_column(String(32), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_list: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    conformal_quantiles: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PredictionLog(Base):
    __tablename__ = "predictions_log"
    __table_args__ = (
        CheckConstraint(
            "cache_status IN ('HIT', 'MISS', 'BYPASS')",
            name="ck_predictions_log_cache_status",
        ),
        Index("ix_predictions_log_request_id", "request_id"),
        Index("ix_predictions_log_model_version", "model_version"),
        Index("ix_predictions_log_player_id", "player_id"),
        Index("ix_predictions_log_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_versions.model_version"), nullable=False
    )
    data_version: Mapped[str] = mapped_column(String(32), nullable=False)
    player_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("players.id", ondelete="SET NULL"), nullable=True
    )
    is_what_if: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), nullable=False
    )
    input_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    predicted_value_eur: Mapped[float] = mapped_column(Float, nullable=False)
    interval_low_eur: Mapped[float] = mapped_column(Float, nullable=False)
    interval_high_eur: Mapped[float] = mapped_column(Float, nullable=False)
    top_shap_drivers: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    cache_status: Mapped[str] = mapped_column(String(16), nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
