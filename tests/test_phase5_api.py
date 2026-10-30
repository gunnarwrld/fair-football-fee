"""Phase 5: what-if math, Redis cache-aside and rate limiting, and the `/v1` API contract.

Unit tests use fakeredis, an in-memory repository and a tiny XGBoost model trained on obviously
fake players. Tests marked `data` check the what-if math against the real serving rows; tests
marked `db` run the real app against the local Postgres inside rolled-back transactions.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fakeredis
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from scripts.export_openapi import OPENAPI_PATH, openapi_text
from src.api.cache import CacheStatus, RateLimiter, RedisCache
from src.api.deps import get_repo
from src.api.main import create_app
from src.api.repository import CurrentFeatures, PlayerHit, PlayerRow
from src.api.runtime import ServingState
from src.features.engineer import MAIN_MODEL_FEATURES, SHRINKAGE_M0_MINUTES
from src.features.whatif import INPUT_FEATURE, WHATIF_BOUNDS, ScenarioError, apply_scenario
from src.ml.data import feature_frame
from src.ml.predictor import Predictor

ARTIFACT_DIR = Path("artifacts/v1.0.0")
LEVELS = {"position_group": ["DEF", "FWD", "GK", "MID"], "preferred_foot": ["both", "left", "right", "unknown"]}

# An obviously fake player whose 12-month counts are mutually consistent
FAKE_FEATURES: dict[str, Any] = {
    "position_group": "MID", "preferred_foot": "right", "age_at_valuation": 24.5, "age_squared": 600.25,
    "height_cm": 180.0, "club_tenure_days_capped": 400.0, "prior_season_appearances": 30.0,
    "prior_season_minutes": 2200.0, "prior_season_goals": 4.0, "prior_season_assists": 3.0,
    "prior_2seasons_minutes": 4000.0, "prior_season_goals_per90_shrunk": 0.12, "prior_season_assists_per90_shrunk": 0.1,
    "mins_last_12m": 2000.0, "mins_last_24m": 3600.0, "appearances_last_12m": 28.0, "appearances_last_24m": 50.0,
    "goals_last_12m": 4.0, "assists_last_12m": 3.0, "goals_per90_shrunk_12m": 0.155172, "assists_per90_shrunk_12m": 0.124138,
    "cards_per90_12m": 0.124138, "clean_sheets_per90_12m": 0.217241, "goals_conceded_per90_12m": 1.055172,
    "minutes_trend_12m_vs_prev12m": 400.0, "injury_days_12m": None, "injury_days_24m": None, "injury_count_12m": None,
    "injury_count_24m": None, "is_currently_injured_at_t": None, "days_since_last_injury_end_capped": None,
}
FAKE_CONFORMAL: dict[str, Any] = {
    "config": "mondrian + recentering", "data_version": "fake-data", "created_at": "2026-01-01T00:00:00+00:00",
    "display_level": 0.8, "levels": [0.8, 0.9], "bins": {"variable": "predicted_log_ratio", "edges": [-0.5, 0.0, 0.5]},
    "quantiles": {"0.8": [0.7, 0.5, 0.5, 0.45], "0.9": [0.9, 0.65, 0.65, 0.6]}, "n_calib": [10, 10, 10, 10],
    "recentering": {"enabled": True, "batch_gap_days": 14, "min_batch_rows": 50, "serving": {
        "as_of_date": "2026-06-03", "offset": -0.1, "source_batch": {"start": "2026-03-09", "end": "2026-03-09", "n_rows": 200}}},
}


# ----------------------------------------------------------------------------- what-if math

def test_empty_scenario_changes_nothing() -> None:
    out, changes = apply_scenario(FAKE_FEATURES, {})
    assert changes == {} and out == FAKE_FEATURES


def test_age_sets_age_and_age_squared_like_the_sql() -> None:
    out, changes = apply_scenario(FAKE_FEATURES, {"age_years": 30.456})
    assert out["age_at_valuation"] == 30.46 and out["age_squared"] == round(30.46**2, 2)
    assert set(changes) == {"age_at_valuation", "age_squared"}


def test_minutes_alone_keep_every_per_minute_rate_and_shift_the_24_month_window() -> None:
    f = FAKE_FEATURES
    out, _ = apply_scenario(f, {"minutes_last_12m": 3000})
    scale = 3000 / f["mins_last_12m"]
    assert out["mins_last_24m"] == f["mins_last_24m"] + 1000
    assert out["minutes_trend_12m_vs_prev12m"] == 2 * 3000 - out["mins_last_24m"]
    assert out["appearances_last_12m"] == round(f["appearances_last_12m"] * scale, 2)
    assert out["appearances_last_24m"] == round(f["appearances_last_24m"] + out["appearances_last_12m"] - f["appearances_last_12m"], 2)
    assert out["goals_last_12m"] / 3000 == pytest.approx(f["goals_last_12m"] / f["mins_last_12m"], rel=1e-3)
    m0 = SHRINKAGE_M0_MINUTES
    prior_term = f["goals_per90_shrunk_12m"] * (f["mins_last_12m"] + m0) - 90 * f["goals_last_12m"]
    assert out["goals_per90_shrunk_12m"] == pytest.approx((90 * out["goals_last_12m"] + prior_term) / (3000 + m0), abs=1e-6)
    cards = f["cards_per90_12m"] * (f["mins_last_12m"] + m0) / 90 * scale
    assert out["cards_per90_12m"] == pytest.approx(90 * cards / (3000 + m0), abs=1e-6)
    for held in ("prior_season_minutes", "height_cm", "position_group", "injury_days_12m"):
        assert out[held] == f[held]


def test_a_count_you_set_is_taken_as_given() -> None:
    out, changes = apply_scenario(FAKE_FEATURES, {"goals_last_12m": 10})
    assert out["goals_last_12m"] == 10 and out["mins_last_12m"] == FAKE_FEATURES["mins_last_12m"]
    assert set(changes) == {"goals_last_12m", "goals_per90_shrunk_12m"}


@pytest.mark.parametrize(("scenario", "message"), [
    ({"minutes_last_12m": 5000, "appearances_last_12m": 10}, "at least 42 appearances"),
    ({"minutes_last_12m": 400}, "outside the trained range"),
    ({"injury_days_12m": 0}, "unknown inputs"),
])
def test_impossible_or_untrained_scenarios_are_refused(scenario: dict[str, float], message: str) -> None:
    with pytest.raises(ScenarioError, match=message):
        apply_scenario(FAKE_FEATURES, scenario)


def test_implied_counts_outside_the_trained_range_are_refused() -> None:
    prolific = dict(FAKE_FEATURES, goals_last_12m=40.0, mins_last_12m=3000.0, mins_last_24m=5000.0, appearances_last_12m=40.0)
    with pytest.raises(ScenarioError, match="implied by keeping the per-minute rate"):
        apply_scenario(prolific, {"minutes_last_12m": 4500})


@pytest.mark.data
def test_whatif_math_matches_the_feature_pipeline_on_every_serving_row() -> None:
    serving = Path("data/features/current_serving_features.parquet")
    if not serving.exists():
        pytest.skip("Serving features not present on clean clone")
    priors = json.loads(Path("data/features/priors.json").read_text(encoding="utf-8"))["match_goals_per90_by_pos"]
    rows = [json.loads(s) for s in pd.read_parquet(serving)["features_json"]]
    m0 = SHRINKAGE_M0_MINUTES
    for f in rows:
        _, changes = apply_scenario(f, {"minutes_last_12m": f["mins_last_12m"]})
        assert changes == {}, f"no-op scenario changed {changes}"
        if f["goals_last_12m"] + 1 <= WHATIF_BOUNDS["goals_last_12m"][1]:
            out, _ = apply_scenario(f, {"goals_last_12m": f["goals_last_12m"] + 1})
            expected = round((90 * (f["goals_last_12m"] + 1) + m0 * priors[f["position_group"]]) / (f["mins_last_12m"] + m0), 6)
            assert out["goals_per90_shrunk_12m"] == pytest.approx(expected, abs=2e-6)


@pytest.mark.data
def test_whatif_bounds_are_the_training_range() -> None:
    if not Path("data/features/valuation_snapshots.parquet").exists():
        pytest.skip("Feature snapshots not present on clean clone")
    from src.ml.data import load_population, split

    trained = split(load_population(), "train", "val_tune")
    slack = {"age_years": 1.0, "minutes_last_12m": 100.0}
    assert set(INPUT_FEATURE) == set(WHATIF_BOUNDS)
    for name, col in INPUT_FEATURE.items():
        lo, hi = WHATIF_BOUNDS[name]
        assert lo <= trained[col].min() and hi >= trained[col].max(), name
        assert trained[col].min() - lo <= slack.get(name, 0.0) and hi - trained[col].max() <= slack.get(name, 0.0), name


# ----------------------------------------------------------------------------- cache and rate limit

class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _DeadRedis:
    """Every call fails like a Redis that is down; counts how often it was tried."""

    def __init__(self) -> None:
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            raise RedisConnectionError("connection refused")
        return fail


def _cache(client: Any = None, clock: Any = time.monotonic) -> RedisCache:
    return RedisCache(client or fakeredis.FakeRedis(), ttl_seconds=60, lock_seconds=2, retry_after_seconds=30,
                      poll_seconds=0.01, clock=clock)


def test_cache_aside_computes_once_then_hits() -> None:
    cache, calls = _cache(), []
    compute = lambda: calls.append(1) or "value"
    assert cache.get_or_compute("k", compute) == ("value", CacheStatus.MISS)
    assert cache.get_or_compute("k", compute) == ("value", CacheStatus.HIT)
    assert len(calls) == 1 and 0 < cache.client.ttl("k") <= 60
    assert cache.client.get("lock:k") is None


def test_concurrent_misses_compute_once_singleflight() -> None:
    cache, calls, results = _cache(), [], []

    def compute() -> str:
        calls.append(1)
        time.sleep(0.2)
        return "slow value"

    threads = [threading.Thread(target=lambda: results.append(cache.get_or_compute("hot", compute))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert {v for v, _ in results} == {"slow value"}
    assert sorted(s for _, s in results).count(CacheStatus.MISS) == 1


def test_a_failed_compute_releases_the_lock_and_caches_nothing() -> None:
    cache = _cache()

    def boom() -> str:
        raise LookupError("not found")

    with pytest.raises(LookupError):
        cache.get_or_compute("k", boom)
    assert cache.client.get("lock:k") is None and cache.client.get("k") is None


def test_lock_is_only_released_by_its_owner() -> None:
    cache = _cache()
    cache.client.set("lock:k", "someone-else")
    cache._release("lock:k", "my-token")
    assert cache.client.get("lock:k") == b"someone-else"


def test_dead_redis_bypasses_and_the_breaker_stops_trying_until_retry() -> None:
    dead, clock = _DeadRedis(), _Clock()
    cache = _cache(dead, clock)
    assert cache.get_or_compute("k", lambda: "v") == ("v", CacheStatus.BYPASS)
    tried = dead.calls
    assert cache.get_or_compute("k", lambda: "v") == ("v", CacheStatus.BYPASS)
    assert dead.calls == tried and not cache.ping()  # open: Redis is not touched at all
    clock.now += 31
    cache.get_or_compute("k", lambda: "v")
    assert dead.calls > tried  # half-open: one new attempt after the retry window


def test_rate_limiter_counts_per_minute_and_fails_open() -> None:
    clock = _Clock()
    limiter = RateLimiter(_cache(), limit_per_minute=3, clock=clock)
    assert [limiter.hit("1.2.3.4")[0] for _ in range(4)] == [True, True, True, False]
    assert limiter.hit("5.6.7.8")[0]
    clock.now += 60
    assert limiter.hit("1.2.3.4")[0]
    assert RateLimiter(_cache(_DeadRedis()), limit_per_minute=1).hit("x")[0]
    assert RateLimiter(_cache(_DeadRedis()), limit_per_minute=1).hit("x")[0]


# ----------------------------------------------------------------------------- API contract

class FakeRuntime:
    def __init__(self, state: ServingState):
        self._state = state

    def state(self) -> ServingState:
        return self._state


class FakeRepo:
    def __init__(self) -> None:
        self.players = {
            1: PlayerRow(1, 9001, "Fake Player One", "MID", "Test FC", "Testland", date(2028, 6, 30)),
            2: PlayerRow(2, 9002, "Fake Player Two", "DEF", "Test FC", "Testland"),  # no valuation
        }
        self.features = {1: CurrentFeatures(1, date(2026, 6, 3), "fake-data", 20e6, dict(FAKE_FEATURES))}

    def release(self) -> None:
        pass

    def ping(self) -> bool:
        return True

    def search_players(self, normalized_query: str, limit: int) -> list[PlayerHit]:
        return [PlayerHit(p.id, p.full_name, p.position_group, p.current_club, p.id in self.features)
                for p in self.players.values() if normalized_query in p.full_name.lower()][:limit]

    def player(self, player_id: int) -> PlayerRow | None:
        return self.players.get(player_id)

    def current_features(self, player_id: int) -> CurrentFeatures | None:
        return self.features.get(player_id)

    def market_value_on_or_before(self, player_id: int, as_of: date) -> tuple[float, date] | None:
        return (18e6, as_of) if player_id in self.features else None


@pytest.fixture(scope="module")
def fake_predictor() -> Predictor:
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(300):
        row = {k: (v * rng.uniform(0.5, 1.5) if isinstance(v, float) else v) for k, v in FAKE_FEATURES.items()}
        row["position_group"] = rng.choice(LEVELS["position_group"])
        rows.append(row)
    X = feature_frame(pd.DataFrame(rows), MAIN_MODEL_FEATURES, LEVELS)
    y = 0.0005 * (X["mins_last_12m"] - 2000) - 0.02 * (X["age_at_valuation"] - 26) ** 2 + rng.normal(0, 0.05, len(X))
    model = xgb.XGBRegressor(n_estimators=40, max_depth=3, enable_categorical=True, tree_method="hist").fit(X, y)
    return Predictor(model, MAIN_MODEL_FEATURES, LEVELS, FAKE_CONFORMAL)


def _state(predictor: Predictor, problems: tuple[str, ...] = (), model_version: str = "v-test") -> ServingState:
    info = {
        "model_version": model_version, "model_type": "xgboost_regressor", "training_data_version": "fake-data",
        "trained_at": "2026-01-01T00:00:00+00:00", "artifact_path": "nowhere/model.ubj", "feature_list": MAIN_MODEL_FEATURES,
        "metrics": {"test": {"mdape": 0.3}, "test_coverage": {"0.8": {"coverage": 0.8}}}, "conformal": FAKE_CONFORMAL,
    }
    return ServingState(model_version, "fake-data", predictor, info, problems)


@pytest.fixture()
def api(fake_predictor: Predictor) -> Any:
    cache = _cache()
    ns = SimpleNamespace(
        logs=[], runtime=FakeRuntime(_state(fake_predictor)), cache=cache,
        limiter=RateLimiter(cache, limit_per_minute=5, clock=lambda: 1_000_000.0),  # one fixed window
    )
    app = create_app(runtime=ns.runtime, cache=ns.cache, limiter=ns.limiter, log_writer=ns.logs.append)
    app.dependency_overrides[get_repo] = FakeRepo
    with TestClient(app) as client:
        ns.client = client
        yield ns


def test_feature_frame_reads_json_nulls_as_missing_numbers() -> None:
    row = pd.DataFrame([{"position_group": "MID", "injury_days_12m": None, "mins_last_12m": 900.0}])
    X = feature_frame(row, ["position_group", "injury_days_12m", "mins_last_12m"], {"position_group": ["MID"]})
    assert X["injury_days_12m"].dtype == float and X["injury_days_12m"].isna().all()
    assert X["position_group"].dtype == "category"


def test_valuation_is_cached_traced_and_logged(api: Any, fake_predictor: Predictor) -> None:
    first = api.client.get("/v1/players/1/valuation", headers={"X-Request-ID": "trace-123"})
    second = api.client.get("/v1/players/1/valuation")
    assert first.status_code == second.status_code == 200
    assert (first.headers["X-Cache"], second.headers["X-Cache"]) == ("MISS", "HIT")
    assert first.headers["X-Request-ID"] == "trace-123" and len(second.headers["X-Request-ID"]) == 32
    assert first.json() == second.json()
    body = first.json()
    expected = fake_predictor.value(FAKE_FEATURES, 20e6)
    assert body["estimate_eur"] == pytest.approx(expected.estimate_eur)
    assert body["interval"]["level"] == 0.8 and body["interval"]["low_eur"] < body["estimate_eur"] < body["interval"]["high_eur"]
    assert body["market_value"] == {"value_eur": 18e6, "date": "2026-06-03"}
    assert [log["cache_status"] for log in api.logs] == ["MISS", "HIT"] and api.logs[0]["request_id"] == "trace-123"


def test_valuation_carries_slider_inputs_and_read_only_context(api: Any) -> None:
    body = api.client.get("/v1/players/1/valuation").json()
    assert body["inputs"] == {"age_years": 24.5, "minutes_last_12m": 2000.0, "appearances_last_12m": 28.0,
                              "goals_last_12m": 4.0, "assists_last_12m": 3.0}
    assert body["context"] == {"injury_days_12m": None, "injury_days_24m": None, "injury_count_24m": None,
                               "contract_expires_at": "2028-06-30"}
    bounds = api.client.get("/v1/model").json()["whatif_bounds"]
    assert bounds == {k: list(v) for k, v in WHATIF_BOUNDS.items()}


def test_drivers_multiply_back_to_the_estimate(api: Any) -> None:
    body = api.client.get("/v1/players/1/valuation").json()
    effects = [d["effect_pct"] for d in body["drivers"]] + [body["other_drivers_effect_pct"], body["adjustment"]["effect_pct"]]
    assert 1 + body["estimate_eur"] == pytest.approx((1 + body["baseline_eur"]) * np.prod([1 + e for e in effects]), rel=1e-5)
    assert len(body["drivers"]) == 5
    phis = [abs(d["phi"]) for d in body["drivers"]]
    assert phis == sorted(phis, reverse=True)
    assert body["adjustment"]["effect_pct"] == pytest.approx(np.expm1(-0.1))


def test_unknown_player_and_player_outside_the_population_are_404(api: Any) -> None:
    missing = api.client.get("/v1/players/99/valuation")
    uncovered = api.client.get("/v1/players/2/valuation")
    assert missing.status_code == uncovered.status_code == 404
    assert missing.json()["detail"]["code"] == "player_not_found"
    assert uncovered.json()["detail"]["code"] == "no_valuation"
    assert api.client.get("/v1/players/0/valuation").status_code == 422


def test_what_if_returns_changes_and_the_difference(api: Any) -> None:
    r = api.client.post("/v1/valuations", json={"player_id": 1, "scenario": {"minutes_last_12m": 2600, "age_years": 23}})
    assert r.status_code == 200, r.json()
    assert api.logs[-1]["is_what_if"] and api.logs[-1]["input_payload"]["scenario"] == {"age_years": 23.0, "minutes_last_12m": 2600}
    body = r.json()
    current = api.client.get("/v1/players/1/valuation").json()["estimate_eur"]
    assert body["current_estimate_eur"] == pytest.approx(current)
    assert body["change_pct"] == pytest.approx(body["estimate_eur"] / current - 1)
    changed = {c["feature"] for c in body["changes"]}
    assert {"mins_last_12m", "mins_last_24m", "age_at_valuation", "age_squared"} <= changed
    assert body["inputs"]["minutes_last_12m"] == 2600 and body["inputs"]["age_years"] == 23
    assert body["inputs"]["goals_last_12m"] == pytest.approx(4 * 2600 / 2000)  # kept its per-minute rate
    assert not any(name.startswith("injury") for name in changed)


@pytest.mark.parametrize("payload", [
    {"player_id": 1, "scenario": {}},
    {"player_id": 1, "scenario": {"injury_days_12m": 0}},
    {"player_id": 1, "scenario": {"minutes_last_12m": 100}},
    {"player_id": 1, "scenario": {"age_years": 25}, "extra": 1},
])
def test_invalid_what_if_payloads_are_422(api: Any, payload: dict[str, Any]) -> None:
    assert api.client.post("/v1/valuations", json=payload).status_code == 422


def test_inconsistent_scenario_explains_itself(api: Any) -> None:
    r = api.client.post("/v1/valuations", json={"player_id": 1, "scenario": {"minutes_last_12m": 5000, "appearances_last_12m": 10}})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "invalid_scenario"
    assert "appearances" in r.json()["detail"]["message"]


def test_what_if_is_rate_limited_per_client(api: Any) -> None:
    payload = {"player_id": 1, "scenario": {"goals_last_12m": 5}}
    codes = [api.client.post("/v1/valuations", json=payload).status_code for _ in range(6)]
    assert codes == [200] * 5 + [429]
    r = api.client.post("/v1/valuations", json=payload)
    assert r.json()["detail"]["code"] == "rate_limited" and 0 < int(r.headers["Retry-After"]) <= 60
    assert api.client.get("/v1/players/1/valuation").status_code == 200  # GETs are not limited


def _served_by_uvicorn(app: Any, peer: str, forwarded_allow_ips: str) -> TestClient:
    """The app behind uvicorn's proxy-headers handling (on by default), reached from address `peer`."""
    return TestClient(ProxyHeadersMiddleware(app, trusted_hosts=forwarded_allow_ips), client=(peer, 50000))


def test_forwarded_for_is_believed_only_from_a_trusted_proxy(api: Any) -> None:
    payload = {"player_id": 1, "scenario": {"goals_last_12m": 5}}

    def codes(client: TestClient, forwarded_for: list[str]) -> list[int]:
        return [client.post("/v1/valuations", json=payload, headers={"X-Forwarded-For": xff}).status_code
                for xff in forwarded_for]

    # A client that forges X-Forwarded-For is still counted by its own address
    direct = _served_by_uvicorn(api.client.app, "198.51.100.7", "127.0.0.1")
    assert codes(direct, [f"203.0.113.{i}" for i in range(6)]) == [200] * 5 + [429]
    # Behind a proxy listed in FORWARDED_ALLOW_IPS, each forwarded client has its own budget...
    proxied = _served_by_uvicorn(api.client.app, "10.0.0.2", "10.0.0.2")
    assert codes(proxied, [f"203.0.113.{i}" for i in range(6)]) == [200] * 6
    # ...and the proxy appends the real address, so entries a client prepends are ignored
    assert codes(proxied, [f"192.0.2.{i}, 203.0.113.50" for i in range(6)]) == [200] * 5 + [429]


def test_a_new_model_version_never_reads_old_cache_entries(api: Any, fake_predictor: Predictor) -> None:
    assert api.client.get("/v1/players/1/valuation").headers["X-Cache"] == "MISS"
    api.runtime._state = _state(fake_predictor, model_version="v-next")
    r = api.client.get("/v1/players/1/valuation")
    assert r.headers["X-Cache"] == "MISS" and r.json()["model_version"] == "v-next"


def test_serving_problems_make_valuations_and_health_503(api: Any, fake_predictor: Predictor) -> None:
    api.runtime._state = _state(fake_predictor, problems=("intervals were calibrated on other data",))
    r = api.client.get("/v1/players/1/valuation")
    assert r.status_code == 503 and r.json()["detail"]["code"] == "not_ready"
    health = api.client.get("/v1/health")
    assert health.status_code == 503 and health.json()["status"] == "unavailable"


def test_health_is_degraded_but_serving_when_redis_is_down(fake_predictor: Predictor) -> None:
    cache = _cache(_DeadRedis())
    app = create_app(runtime=FakeRuntime(_state(fake_predictor)), cache=cache,
                     limiter=RateLimiter(cache, 5), log_writer=lambda entry: None)
    app.dependency_overrides[get_repo] = FakeRepo
    with TestClient(app) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200 and health.json()["status"] == "degraded" and not health.json()["redis"]
        r = client.get("/v1/players/1/valuation")
        assert r.status_code == 200 and r.headers["X-Cache"] == "BYPASS"
        assert client.post("/v1/valuations", json={"player_id": 1, "scenario": {"goals_last_12m": 6}}).status_code == 200


def test_search_and_model_endpoints(api: Any) -> None:
    r = api.client.get("/v1/players", params={"q": "Fake Pláyer"})
    assert r.status_code == 200
    assert [(h["id"], h["has_valuation"]) for h in r.json()["results"]] == [(1, True), (2, False)]
    assert api.client.get("/v1/players", params={"q": "x"}).status_code == 422
    model = api.client.get("/v1/model").json()
    assert model["model_version"] == "v-test" and model["display_level"] == 0.8
    assert model["recentering"]["source_batch"]["start"] == "2026-03-09"


# ----------------------------------------------------------------------------- deployment (Phase 7)

def test_committed_openapi_contract_matches_the_app() -> None:
    """frontend/openapi.json is the contract the UI's TypeScript types are generated from."""
    assert OPENAPI_PATH.read_text(encoding="utf-8") == openapi_text(), "run `make api-types` and commit both files"


def test_past_the_concurrency_limit_requests_get_an_immediate_503(fake_predictor: Predictor) -> None:
    release, started = threading.Event(), threading.Semaphore(0)

    class SlowRepo(FakeRepo):
        def search_players(self, normalized_query: str, limit: int) -> list[PlayerHit]:
            started.release()
            release.wait(5)
            return super().search_players(normalized_query, limit)

    cache = _cache()
    app = create_app(runtime=FakeRuntime(_state(fake_predictor)), cache=cache, limiter=RateLimiter(cache, 5),
                     log_writer=lambda entry: None, max_concurrent=2)
    app.dependency_overrides[get_repo] = SlowRepo
    with TestClient(app) as client, ThreadPoolExecutor(2) as pool:
        in_flight = [pool.submit(client.get, "/v1/players?q=fake") for _ in range(2)]
        assert started.acquire(timeout=5) and started.acquire(timeout=5)
        busy = client.get("/v1/players?q=fake", headers={"Origin": "http://localhost:3000"})
        assert busy.status_code == 503 and busy.json()["detail"]["code"] == "busy"
        assert busy.headers["Retry-After"] == "1"
        assert busy.headers["access-control-allow-origin"] == "http://localhost:3000"  # a browser can read it
        assert client.get("/v1/health").status_code == 200  # probes are never shed: the worker is alive
        release.set()
        assert [f.result().status_code for f in in_flight] == [200, 200]
        assert client.get("/v1/players?q=fake").status_code == 200  # finished requests free their slots


def test_the_model_runs_only_after_the_database_connection_is_released(fake_predictor: Predictor) -> None:
    events: list[str] = []

    class RecordingRepo(FakeRepo):
        def player(self, player_id: int) -> PlayerRow | None:
            events.append("read player")
            return super().player(player_id)

        def current_features(self, player_id: int) -> CurrentFeatures | None:
            events.append("read features")
            return super().current_features(player_id)

        def market_value_on_or_before(self, player_id: int, as_of: date) -> tuple[float, date] | None:
            events.append("read market value")
            return super().market_value_on_or_before(player_id, as_of)

        def release(self) -> None:
            events.append("release connection")

    class RecordingPredictor:
        def __getattr__(self, name: str) -> Any:
            return getattr(fake_predictor, name)

        def value(self, features: dict[str, Any], league_median_eur: float) -> Any:
            events.append("run model")
            return fake_predictor.value(features, league_median_eur)

    cache = _cache()
    app = create_app(runtime=FakeRuntime(_state(RecordingPredictor())), cache=cache,
                     limiter=RateLimiter(cache, 5), log_writer=lambda entry: None)
    app.dependency_overrides[get_repo] = RecordingRepo
    with TestClient(app) as client:
        assert client.get("/v1/players/1/valuation").status_code == 200
        assert events == ["read player", "read features", "read market value", "release connection", "run model"]
        events.clear()
        assert client.post("/v1/valuations", json={"player_id": 1, "scenario": {"age_years": 25}}).status_code == 200
        assert events == ["read player", "read features", "release connection", "run model"]  # current one is cached


def test_no_free_database_connection_is_a_503_busy_not_a_500(api: Any) -> None:
    class ExhaustedPool(FakeRepo):
        def search_players(self, normalized_query: str, limit: int) -> list[PlayerHit]:
            raise PoolTimeoutError("QueuePool limit of size 5 overflow 10 reached, connection timed out")

    api.client.app.dependency_overrides[get_repo] = ExhaustedPool
    r = api.client.get("/v1/players?q=fake")
    assert r.status_code == 503 and r.json()["detail"]["code"] == "busy" and r.headers["Retry-After"] == "1"


_SERVE_ONE_PLAYER = """
import json, sys
from pathlib import Path
import src.api.main  # the app module, as uvicorn imports it
from src.ml.predictor import Predictor
args = json.load(sys.stdin)
Predictor.from_artifacts(Path(args["dir"]), args["conformal"]).value(args["features"], 20e6)
print(json.dumps(sorted({name.split(".")[0] for name in sys.modules})))
"""


def test_serving_imports_nothing_outside_the_core_dependencies(fake_predictor: Predictor, tmp_path: Path) -> None:
    """The API image installs only [project.dependencies]: loading a model and valuing a player must not
    need the `ml` or `dev` extras (a fresh interpreter, so other tests' imports don't count)."""
    fake_predictor.model.save_model(tmp_path / "model.ubj")
    spec = {"features": MAIN_MODEL_FEATURES, "categorical_levels": LEVELS}
    (tmp_path / "features.json").write_text(json.dumps(spec), encoding="utf-8")
    args = {"dir": str(tmp_path), "conformal": FAKE_CONFORMAL, "features": FAKE_FEATURES}
    run = subprocess.run([sys.executable, "-c", _SERVE_ONE_PLAYER], input=json.dumps(args), capture_output=True,
                         text=True, check=False, cwd=Path(__file__).resolve().parents[1])
    assert run.returncode == 0, run.stderr
    assert not set(json.loads(run.stdout)) & {"lightgbm", "optuna", "shap", "matplotlib", "httpx", "pytest", "fakeredis"}


# ----------------------------------------------------------------------------- real database

def _db_engine() -> Any:
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.exc import OperationalError

    from src.config import settings

    engine = create_engine(settings.database_url)
    try:
        ready = inspect(engine).has_table("model_versions")
    except OperationalError:
        pytest.skip("Postgres not reachable (run `make db`)")
    if not ready:
        pytest.skip("Schema not migrated (run `make db`)")
    return engine


@pytest.mark.db
def test_real_api_matches_the_offline_phase4_estimate() -> None:
    if not (ARTIFACT_DIR / "phase4_metrics.json").exists() or not (ARTIFACT_DIR / "model.ubj").exists():
        pytest.skip("Model artifacts not present on clean clone")
    _db_engine().dispose()
    example = json.loads((ARTIFACT_DIR / "phase4_metrics.json").read_text(encoding="utf-8"))["shap_example"]
    expected = float(np.expm1(example["pred_ratio"] + example["offset"] + np.log1p(example["league_median_eur"])))
    logs: list[dict[str, Any]] = []
    app = create_app(cache=_cache(), log_writer=logs.append)
    with TestClient(app) as client:
        health = client.get("/v1/health").json()
        if health["problems"]:
            pytest.skip(f"serving state not ready: {health['problems']}")
        surname = example["full_name"].split()[-1]
        hits = client.get("/v1/players", params={"q": surname}).json()["results"]
        hit = next(h for h in hits if h["full_name"] == example["full_name"])
        assert hit["has_valuation"]
        body = client.get(f"/v1/players/{hit['id']}/valuation").json()
    assert body["estimate_eur"] == pytest.approx(expected, rel=1e-9)
    assert body["drivers"][0]["feature"] == example["top_drivers"][0]["feature"]
    assert len(logs) == 1


@pytest.mark.db
def test_real_api_values_a_player_without_injury_history() -> None:
    from sqlalchemy import text

    engine = _db_engine()
    with engine.connect() as conn:
        player_id = conn.execute(text(
            "SELECT player_id FROM player_current_features WHERE features->>'injury_days_12m' IS NULL LIMIT 1"
        )).scalar()
    engine.dispose()
    if player_id is None:
        pytest.skip("every serving player has an injury history")
    app = create_app(cache=_cache(), log_writer=lambda entry: None)
    with TestClient(app) as client:
        r = client.get(f"/v1/players/{player_id}/valuation")
    assert r.status_code == 200, r.text
    assert not any(d["feature"].startswith("injury") and d["value"] is not None for d in r.json()["drivers"])


@pytest.mark.db
def test_prediction_log_writer_inserts_a_row() -> None:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from src.api.predlog import PredictionLogWriter
    from src.db.models import ModelVersion, Player, PredictionLog

    engine = _db_engine()
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            session = Session(bind=conn, join_transaction_mode="create_savepoint")
            model = session.scalars(select(ModelVersion.model_version)).first()
            player = session.scalars(select(Player.id)).first()
            if model is None or player is None:
                pytest.skip("No registered model or players (run `make db` and `make register`)")
            writer = PredictionLogWriter(lambda: Session(bind=conn, join_transaction_mode="create_savepoint"))
            writer({
                "request_id": "test-fake-request", "model_version": model, "data_version": "fake", "player_id": player,
                "is_what_if": True, "input_payload": {"player_id": player, "scenario": {"goals_last_12m": 3}},
                "predicted_value_eur": 1.0, "interval_low_eur": 0.5, "interval_high_eur": 2.0,
                "top_shap_drivers": [{"feature": "age", "phi": 0.1}], "cache_status": "MISS", "latency_ms": 1.5,
            })
            rows = session.scalars(select(PredictionLog).where(PredictionLog.request_id == "test-fake-request")).all()
            assert len(rows) == 1 and rows[0].input_payload["scenario"] == {"goals_last_12m": 3}
            session.close()
        finally:
            trans.rollback()
    engine.dispose()


@pytest.mark.db
def test_runtime_refuses_to_serve_a_stale_calibration() -> None:
    from sqlalchemy import update
    from sqlalchemy.orm import Session

    from src.api.runtime import Runtime
    from src.db.models import ModelVersion

    engine = _db_engine()
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            session = Session(bind=conn, join_transaction_mode="create_savepoint")
            stale = {"data_version": "an-older-data-version", "recentering": {"enabled": False}, "created_at": "x"}
            session.execute(update(ModelVersion).where(ModelVersion.is_active.is_(True)).values(conformal_quantiles=stale))
            session.flush()
            loaded: list[Path] = []
            runtime = Runtime(lambda: Session(bind=conn, join_transaction_mode="create_savepoint"), refresh_seconds=60,
                              loader=lambda path, conformal: loaded.append(path) or object())
            state = runtime.state()
            if state.model_version is None:
                pytest.skip("No active model (run `make register`)")
            assert any("calibrated on data an-older-data-version" in p for p in state.problems)
            session.close()
        finally:
            trans.rollback()
    engine.dispose()
