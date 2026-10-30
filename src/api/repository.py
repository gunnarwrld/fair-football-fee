"""SQL reads and writes the API needs. Routes and services never build queries themselves."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.db.models import ModelVersion, Player, PlayerCurrentFeatures, PredictionLog, Valuation


@dataclass(frozen=True)
class PlayerRow:
    id: int
    tm_player_id: int
    full_name: str
    position_group: str
    current_club: str | None
    nationality: str | None
    contract_expires_at: date | None = None


@dataclass(frozen=True)
class CurrentFeatures:
    player_id: int
    as_of_date: date
    data_version: str
    trailing_league_median_eur: float
    features: dict[str, Any]


@dataclass(frozen=True)
class PlayerHit:
    id: int
    full_name: str
    position_group: str
    current_club: str | None
    has_valuation: bool


# Word similarity matches a query inside a longer name ("kane" in "harry kane"); both operators
# use the pg_trgm GIN index on normalized_name.
_SEARCH_SQL = text("""
    SELECT p.id, p.full_name, p.position_group, p.current_club, (f.player_id IS NOT NULL) AS has_valuation
    FROM players p
    LEFT JOIN player_current_features f ON f.player_id = p.id
    WHERE :q <% p.normalized_name OR p.normalized_name LIKE '%' || :q || '%'
    ORDER BY has_valuation DESC, word_similarity(:q, p.normalized_name) DESC, p.full_name
    LIMIT :limit
""")


class Repository:
    def __init__(self, db: Session):
        self.db = db

    def ping(self) -> bool:
        try:
            self.db.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def search_players(self, normalized_query: str, limit: int) -> list[PlayerHit]:
        rows = self.db.execute(_SEARCH_SQL, {"q": normalized_query, "limit": limit}).all()
        return [PlayerHit(r.id, r.full_name, r.position_group, r.current_club, bool(r.has_valuation)) for r in rows]

    def player(self, player_id: int) -> PlayerRow | None:
        p = self.db.get(Player, player_id)
        if p is None:
            return None
        return PlayerRow(
            p.id, p.tm_player_id, p.full_name, p.position_group, p.current_club, p.nationality, p.contract_expires_at
        )

    def current_features(self, player_id: int) -> CurrentFeatures | None:
        f = self.db.get(PlayerCurrentFeatures, player_id)
        if f is None:
            return None
        return CurrentFeatures(f.player_id, f.as_of_date, f.data_version, float(f.trailing_league_median_eur), dict(f.features))

    def market_value_on_or_before(self, player_id: int, as_of: date) -> tuple[float, date] | None:
        """The latest Transfermarkt value published on or before `as_of`, for comparison only."""
        row = self.db.execute(
            select(Valuation.market_value_eur, Valuation.valuation_date)
            .where(Valuation.player_id == player_id, Valuation.valuation_date <= as_of)
            .order_by(Valuation.valuation_date.desc())
            .limit(1)
        ).first()
        return None if row is None else (float(row.market_value_eur), row.valuation_date)


def active_model(db: Session) -> ModelVersion | None:
    return db.scalars(select(ModelVersion).where(ModelVersion.is_active.is_(True))).first()


def serving_data_versions(db: Session) -> list[str]:
    return list(db.scalars(select(func.distinct(PlayerCurrentFeatures.data_version))))


def insert_prediction_log(db: Session, entry: dict[str, Any]) -> None:
    db.add(PredictionLog(**entry))
