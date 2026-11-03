"""Valuation use cases: a player's current valuation and a what-if scenario on top of it.

Routes stay thin: they resolve dependencies, go through the cache, and call these functions.
Errors are raised as `ApiError` subclasses and turned into JSON by the app's exception handler.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from src.api.repository import CurrentFeatures, PlayerRow, Repository
from src.api.runtime import ServingState
from src.api.schemas import (
    AdjustmentOut,
    ContextOut,
    DriverOut,
    FeatureChangeOut,
    InputsOut,
    IntervalOut,
    MarketValueOut,
    PlayerOut,
    ValuationOut,
    WhatIfOut,
)
from src.features.whatif import ScenarioError, apply_scenario, scenario_inputs
from src.ml.predictor import Valuation

TOP_DRIVERS = 5
# Part of every cache key: bump it whenever the response shape changes, so a deploy never parses
# JSON cached by the previous version (keys also carry the model and data versions)
RESPONSE_SCHEMA = "s2"


class ApiError(Exception):
    status_code = 500

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.message = message
        self.code = code


class NotFound(ApiError):
    status_code = 404


class Unavailable(ApiError):
    status_code = 503


class InvalidScenario(ApiError):
    status_code = 422


def valuation_key(state: ServingState, player_id: int) -> str:
    return f"valuation:{RESPONSE_SCHEMA}:{state.model_version}:{state.data_version}:{player_id}"


def whatif_key(state: ServingState, player_id: int, scenario: dict[str, float]) -> str:
    payload = json.dumps({"player_id": player_id, "scenario": scenario}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"valuation:{RESPONSE_SCHEMA}:{state.model_version}:{state.data_version}:whatif:{digest}"


def require_servable(state: ServingState) -> None:
    if state.problems or state.predictor is None:
        raise Unavailable("; ".join(state.problems) or "model not loaded", code="not_ready")


def _load(repo: Repository, state: ServingState, player_id: int) -> tuple[PlayerRow, CurrentFeatures]:
    player = repo.player(player_id)
    if player is None:
        raise NotFound(f"player {player_id} not found", code="player_not_found")
    cf = repo.current_features(player_id)
    if cf is None:
        raise NotFound(
            "no valuation for this player: the model covers Premier League players with at least 450 "
            "tracked minutes in the last 12 months",
            code="no_valuation",
        )
    if cf.data_version != state.data_version:
        raise Unavailable("serving data is being refreshed; retry shortly", code="data_refreshing")
    return player, cf


def _build(
    player: PlayerRow, cf: CurrentFeatures, v: Valuation, state: ServingState, market: MarketValueOut | None,
    features: dict[str, Any],
) -> dict[str, Any]:
    """Response fields shared by a current valuation and a what-if; `features` is the row valued."""
    predictor = state.predictor
    assert predictor is not None
    intervals = [IntervalOut(level=i.level, low_eur=i.low_eur, high_eur=i.high_eur) for i in v.intervals]
    rec = state.model_info["conformal"]["recentering"]
    return {
        "player": PlayerOut(
            id=player.id, full_name=player.full_name, position_group=player.position_group,
            current_club=player.current_club, nationality=player.nationality, age=cf.features.get("age_at_valuation"),
        ),
        "as_of_date": cf.as_of_date,
        "model_version": state.model_version,
        "data_version": state.data_version,
        "estimate_eur": v.estimate_eur,
        "interval": next(i for i in intervals if i.level == predictor.display_level),
        "intervals": intervals,
        "market_value": market,
        "league_median_eur": v.league_median_eur,
        "baseline_eur": v.baseline_eur,
        "drivers": [DriverOut(**d) for d in v.drivers[:TOP_DRIVERS]],
        "other_drivers_effect_pct": float(np.expm1(sum(d["phi"] for d in v.drivers[TOP_DRIVERS:]))),
        "adjustment": AdjustmentOut(
            phi=v.offset, effect_pct=float(np.expm1(v.offset)),
            source_batch=rec["serving"]["source_batch"] if rec["enabled"] else None,
        ),
        "inputs": InputsOut(**scenario_inputs(features)),
        "context": ContextOut(
            injury_days_12m=cf.features.get("injury_days_12m"),
            injury_days_24m=cf.features.get("injury_days_24m"),
            injury_count_24m=cf.features.get("injury_count_24m"),
            contract_expires_at=player.contract_expires_at,
        ),
    }


def current_valuation(repo: Repository, state: ServingState, player_id: int) -> ValuationOut:
    player, cf = _load(repo, state, player_id)
    found = repo.market_value_on_or_before(player_id, cf.as_of_date)
    # Every read is done: give the connection back before the CPU-bound model and SHAP run. Holding it
    # through them made the pool, not the CPU, the limit once requests stopped hitting the cache.
    repo.release()
    assert state.predictor is not None
    v = state.predictor.value(cf.features, cf.trailing_league_median_eur)
    market = None if found is None else MarketValueOut(value_eur=found[0], date=found[1])
    return ValuationOut(**_build(player, cf, v, state, market, cf.features))


def whatif_valuation(
    repo: Repository, state: ServingState, player_id: int, scenario: dict[str, float], current: ValuationOut
) -> WhatIfOut:
    player, cf = _load(repo, state, player_id)
    repo.release()  # as in current_valuation: no connection held while the model runs
    assert state.predictor is not None
    try:
        features, changes = apply_scenario(cf.features, scenario)
    except ScenarioError as exc:
        raise InvalidScenario(str(exc), code="invalid_scenario") from exc
    v = state.predictor.value(features, cf.trailing_league_median_eur)
    return WhatIfOut(
        **_build(player, cf, v, state, current.market_value, features),
        scenario=scenario,
        current_estimate_eur=current.estimate_eur,
        change_pct=v.estimate_eur / current.estimate_eur - 1,
        changes=[FeatureChangeOut(feature=k, before=b, after=a) for k, (b, a) in sorted(changes.items())],
    )
