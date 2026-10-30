"""`/v1` endpoints. Plain `def` routes: their work (SQLAlchemy, redis-py, XGBoost) is blocking, so
FastAPI runs them in its threadpool and the event loop never waits on them."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, Path, Query, Request, Response

from src.api.cache import CacheStatus
from src.api.deps import CacheDep, LogWriterDep, RepoDep, ServableStateDep, StateDep, rate_limit
from src.api.runtime import ServingState
from src.api.schemas import (
    HealthOut,
    ModelOut,
    PlayerHitOut,
    PlayerSearchOut,
    ValuationOut,
    WhatIfIn,
    WhatIfOut,
)
from src.api.service import (
    Unavailable,
    current_valuation,
    valuation_key,
    whatif_key,
    whatif_valuation,
)
from src.data.clean import normalize_player_name

router = APIRouter()


def _log_entry(
    request: Request, state: ServingState, player_id: int, out: ValuationOut, status: CacheStatus,
    started: float, payload: dict[str, Any], is_what_if: bool,
) -> dict[str, Any]:
    return {
        "request_id": request.state.request_id,
        "model_version": state.model_version,
        "data_version": state.data_version,
        "player_id": player_id,
        "is_what_if": is_what_if,
        "input_payload": payload,
        "predicted_value_eur": out.estimate_eur,
        "interval_low_eur": out.interval.low_eur,
        "interval_high_eur": out.interval.high_eur,
        "top_shap_drivers": [d.model_dump() for d in out.drivers],
        "cache_status": str(status),
        "latency_ms": round(1000 * (time.perf_counter() - started), 3),
    }


@router.get("/health", response_model=HealthOut)
def health(response: Response, repo: RepoDep, state: StateDep, cache: CacheDep) -> HealthOut:
    database, redis_ok = repo.ping(), cache.ping()
    problems = list(state.problems)
    if not database and "database unreachable" not in problems:
        problems.append("database unreachable")
    response.status_code = 503 if problems else 200
    return HealthOut(
        status="unavailable" if problems else ("ok" if redis_ok else "degraded"),
        database=database, redis=redis_ok,
        model_version=state.model_version, data_version=state.data_version, problems=problems,
    )


@router.get("/model", response_model=ModelOut)
def model_info(state: StateDep) -> ModelOut:
    if not state.model_info:
        raise Unavailable("; ".join(state.problems) or "no active model", code="not_ready")
    m = state.model_info
    conformal, metrics = m["conformal"], m["metrics"]
    rec = conformal["recentering"]
    offset = float(rec["serving"]["offset"]) if rec["enabled"] else 0.0
    return ModelOut(
        model_version=m["model_version"],
        model_type=m["model_type"],
        trained_at=m["trained_at"],
        training_data_version=m["training_data_version"],
        serving_data_version=state.data_version,
        display_level=conformal["display_level"],
        interval_levels=conformal["levels"],
        interval_method=conformal["config"],
        recentering={
            "enabled": rec["enabled"], "offset": offset, "effect_pct": float(np.expm1(offset)),
            "source_batch": rec["serving"]["source_batch"], "batch_gap_days": rec["batch_gap_days"],
            "min_batch_rows": rec["min_batch_rows"],
        },
        test_metrics={"point": metrics.get("test"), "coverage": metrics.get("test_coverage"),
                      "best_baseline": metrics.get("best_baseline")},
        features=m["feature_list"],
    )


@router.get("/players", response_model=PlayerSearchOut)
def search_players(
    repo: RepoDep,
    q: Annotated[str, Query(min_length=2, max_length=100, description="Part of a name; accents and case are ignored")],
    limit: Annotated[int, Query(ge=1, le=25)] = 10,
) -> PlayerSearchOut:
    normalized = normalize_player_name(q)
    hits = repo.search_players(normalized, limit) if len(normalized) >= 2 else []
    return PlayerSearchOut(query=q, results=[PlayerHitOut(**asdict(h)) for h in hits])


@router.get("/players/{player_id}/valuation", response_model=ValuationOut)
def player_valuation(
    player_id: Annotated[int, Path(ge=1)],
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    repo: RepoDep,
    state: ServableStateDep,
    cache: CacheDep,
    log_writer: LogWriterDep,
) -> ValuationOut:
    started = time.perf_counter()
    raw, status = cache.get_or_compute(
        valuation_key(state, player_id), lambda: current_valuation(repo, state, player_id).model_dump_json()
    )
    out = ValuationOut.model_validate_json(raw)
    response.headers["X-Cache"] = str(status)
    background_tasks.add_task(
        log_writer, _log_entry(request, state, player_id, out, status, started, {"player_id": player_id}, False)
    )
    return out


@router.post("/valuations", response_model=WhatIfOut, dependencies=[Depends(rate_limit)])
def what_if(
    body: WhatIfIn,
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    repo: RepoDep,
    state: ServableStateDep,
    cache: CacheDep,
    log_writer: LogWriterDep,
) -> WhatIfOut:
    started = time.perf_counter()
    pid, scenario = body.player_id, body.scenario.as_dict()
    raw_current, _ = cache.get_or_compute(
        valuation_key(state, pid), lambda: current_valuation(repo, state, pid).model_dump_json()
    )
    current = ValuationOut.model_validate_json(raw_current)
    raw, status = cache.get_or_compute(
        whatif_key(state, pid, scenario), lambda: whatif_valuation(repo, state, pid, scenario, current).model_dump_json()
    )
    out = WhatIfOut.model_validate_json(raw)
    response.headers["X-Cache"] = str(status)
    background_tasks.add_task(
        log_writer, _log_entry(request, state, pid, out, status, started, {"player_id": pid, "scenario": scenario}, True)
    )
    return out
