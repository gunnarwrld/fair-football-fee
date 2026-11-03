"""Request and response models for `/v1` (they also generate the OpenAPI docs at `/docs`)."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.features.whatif import WHATIF_BOUNDS

DISCLAIMER = "Model estimate of the Transfermarkt crowd valuation, not a transfer fee."


class PlayerHitOut(BaseModel):
    id: int
    full_name: str
    position_group: str
    current_club: str | None
    has_valuation: bool = Field(description="False when the player is outside the model's population")


class PlayerSearchOut(BaseModel):
    query: str
    results: list[PlayerHitOut]


class PlayerOut(BaseModel):
    id: int
    full_name: str
    position_group: str
    current_club: str | None
    nationality: str | None
    age: float | None


class IntervalOut(BaseModel):
    level: float
    low_eur: float
    high_eur: float


class DriverOut(BaseModel):
    feature: str
    label: str
    value: float | str | None
    phi: float = Field(description="SHAP value in log units")
    effect_pct: float = Field(description="Multiplicative effect on the valuation: exp(phi) - 1")


class AdjustmentOut(BaseModel):
    label: str = "Latest update-round adjustment"
    phi: float
    effect_pct: float
    source_batch: dict[str, Any] | None


class MarketValueOut(BaseModel):
    value_eur: float
    date: date


class ValuationOut(BaseModel):
    player: PlayerOut
    as_of_date: date
    model_version: str
    data_version: str
    estimate_eur: float
    interval: IntervalOut = Field(description="The displayed interval")
    intervals: list[IntervalOut]
    market_value: MarketValueOut | None = Field(description="Transfermarkt value published on or before as_of_date")
    league_median_eur: float
    baseline_eur: float = Field(description="A typical player at today's league level; drivers multiply this")
    drivers: list[DriverOut] = Field(description="Largest drivers first")
    other_drivers_effect_pct: float
    adjustment: AdjustmentOut
    disclaimer: str = DISCLAIMER


def _bounded(name: str, kind: type) -> Any:
    lo, hi = WHATIF_BOUNDS[name]
    if kind is int:
        return Field(None, ge=int(lo), le=int(hi))
    return Field(None, ge=lo, le=hi)


class ScenarioIn(BaseModel):
    """Inputs you set are taken as given; 12-month counts you don't set keep their per-minute rate."""

    model_config = ConfigDict(extra="forbid")

    age_years: float | None = _bounded("age_years", float)
    minutes_last_12m: int | None = _bounded("minutes_last_12m", int)
    appearances_last_12m: int | None = _bounded("appearances_last_12m", int)
    goals_last_12m: int | None = _bounded("goals_last_12m", int)
    assists_last_12m: int | None = _bounded("assists_last_12m", int)

    @model_validator(mode="after")
    def _not_empty(self) -> ScenarioIn:
        if not self.as_dict():
            raise ValueError("set at least one scenario input")
        return self

    def as_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class WhatIfIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: int = Field(ge=1)
    scenario: ScenarioIn


class FeatureChangeOut(BaseModel):
    feature: str
    before: float | None
    after: float | None


class WhatIfOut(ValuationOut):
    scenario: dict[str, float]
    current_estimate_eur: float
    change_pct: float
    changes: list[FeatureChangeOut]


class HealthOut(BaseModel):
    status: str = Field(description="ok, degraded (Redis down, still serving) or unavailable")
    database: bool
    redis: bool
    model_version: str | None
    data_version: str | None
    problems: list[str]


class ModelOut(BaseModel):
    model_version: str
    model_type: str
    trained_at: str
    training_data_version: str
    serving_data_version: str | None
    display_level: float
    interval_levels: list[float]
    interval_method: str
    recentering: dict[str, Any]
    test_metrics: dict[str, Any]
    features: list[str]
    disclaimer: str = DISCLAIMER
