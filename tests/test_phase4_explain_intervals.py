"""Phase 4: conformal intervals, point-in-time recentering, TreeSHAP and model registration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import MAIN_MODEL_FEATURES
from src.ml.conformal import (
    ConformalIntervals,
    batch_of,
    batch_offsets,
    batch_table,
    conformal_quantile,
)
from src.ml.explain import (
    DRIVER_LABELS,
    FEATURE_GROUPS,
    group_contributions,
    shap_contributions,
    top_drivers,
)

ARTIFACT_DIR = Path("artifacts/v1.0.0")


def test_conformal_quantile_uses_finite_sample_rank() -> None:
    scores = np.arange(1, 10, dtype=float)  # n = 9
    assert conformal_quantile(scores, 0.8) == 8.0  # ceil(10 * 0.8) = 8th smallest
    assert conformal_quantile(scores, 0.5) == 5.0
    assert conformal_quantile(scores, 0.95) == np.inf  # ceil(9.5) = 10 > n


def test_split_conformal_reaches_nominal_coverage_on_exchangeable_data() -> None:
    rng = np.random.default_rng(0)
    covered = []
    for _ in range(300):
        cal, test = rng.standard_t(4, 150), rng.standard_t(4, 150)
        ci = ConformalIntervals.fit(np.zeros(150), cal, levels=[0.8])
        covered.append(np.mean(np.abs(test) <= ci.halfwidth(np.zeros(150), 0.8)))
    assert 0.79 <= np.mean(covered) <= 0.83  # guarantee: >= 0.8 on average (here 121/151), minus Monte Carlo noise


def test_mondrian_bins_fix_coverage_when_noise_depends_on_the_prediction() -> None:
    rng = np.random.default_rng(1)

    def draw(n: int) -> tuple[np.ndarray, np.ndarray]:
        pred = rng.uniform(-1, 1, n)
        return pred, rng.normal(0, np.where(pred < -0.5, 1.0, 0.3))  # the lowest quarter is noisier

    pred_c, res_c = draw(20000)
    pred_t, res_t = draw(20000)
    low = pred_t < -0.5
    for n_bins, ok in ((1, False), (4, True)):
        ci = ConformalIntervals.fit(pred_c, res_c, levels=[0.8], n_bins=n_bins)
        cov_low = np.mean(np.abs(res_t[low]) <= ci.halfwidth(pred_t[low], 0.8))
        assert bool(abs(cov_low - 0.8) < 0.03) == ok  # global: about 40% in the noisy bin


def test_conformal_intervals_round_trip_through_json() -> None:
    ci = ConformalIntervals.fit(np.linspace(-1, 1, 101), np.linspace(-2, 2, 101), levels=[0.8, 0.9], n_bins=4, recentered=True)
    back = ConformalIntervals.from_dict(json.loads(json.dumps(ci.to_dict())))
    assert back == ci
    x = np.array([-0.9, 0.0, 0.9])
    assert np.array_equal(back.halfwidth(x, 0.9), ci.halfwidth(x, 0.9))
    assert sum(ci.n_calib) == 101 and len(ci.edges) == 3


def test_batch_table_splits_on_gaps_longer_than_two_weeks() -> None:
    dates = ["2024-10-01", "2024-10-01", "2024-10-09", "2024-12-16", "2025-03-18", "2025-05-30", "2025-06-12"]
    table = batch_table(dates)
    assert table["n"].tolist() == [3, 1, 1, 2]
    assert table["end"].dt.strftime("%Y-%m-%d").tolist() == ["2024-10-09", "2024-12-16", "2025-03-18", "2025-06-12"]
    assert batch_of(["2024-10-05", "2024-11-01", "2025-06-12"], table).tolist() == [0, -1, 3]


def _batches(n_per: list[int], starts: list[str], resid: list[float]) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    dates = [s for s, n in zip(starts, n_per, strict=True) for _ in range(n)]
    res = np.concatenate([np.full(n, r) for n, r in zip(n_per, resid, strict=True)])
    return batch_table(dates), dates, res


def test_batch_offsets_read_only_earlier_qualifying_batches() -> None:
    starts = ["2024-10-01", "2024-12-16", "2025-03-18", "2025-05-30"]
    table, dates, res = _batches([60, 10, 60, 60], starts, [-0.1, 0.5, -0.3, 0.2])
    offsets, sources = batch_offsets(starts + ["2025-07-15"], table, dates, res, min_rows=50)
    # Oct: nothing earlier. Dec: Oct. Mar: Dec has only 10 rows, so Oct. May: Mar. July (no batch): May.
    assert sources.tolist() == [-1, 0, 0, 2, 3]
    assert offsets == pytest.approx([0.0, -0.1, -0.1, -0.3, 0.2])


def test_batch_offsets_are_unchanged_when_later_rows_are_removed() -> None:
    rng = np.random.default_rng(3)
    base = pd.Timestamp("2020-01-01")
    dates = [base + pd.Timedelta(days=int(d)) for d in np.sort(rng.choice(np.arange(0, 1500, 70), 20, replace=False)) for _ in range(60)]
    res = rng.normal(0, 0.2, len(dates))
    table = batch_table(dates)
    queries = pd.Series(sorted(set(dates)))
    full, _ = batch_offsets(queries, table, dates, res)
    for i, t in enumerate(queries):
        start = table.loc[batch_of([t], table)[0], "start"]
        keep = np.array([d < start for d in dates])
        truncated, _ = batch_offsets([t], table, np.array(dates)[keep], res[keep])
        assert truncated[0] == pytest.approx(full[i])


def _fake_model_and_data() -> tuple[Any, pd.DataFrame]:
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(7)
    n = 300  # obviously fake players: random numbers, no identities
    X = pd.DataFrame({
        "position_group": pd.Categorical(rng.choice(["DEF", "MID", "FWD"], n), categories=["DEF", "FWD", "MID"]),
        "age_at_valuation": rng.uniform(18, 35, n),
        "mins_last_12m": rng.uniform(450, 3400, n),
    })
    X["age_squared"] = X["age_at_valuation"] ** 2
    y = 0.001 * X["mins_last_12m"] - 0.05 * (X["age_at_valuation"] - 26) ** 2 / 10 + rng.normal(0, 0.1, n)
    model = xgb.XGBRegressor(n_estimators=60, max_depth=3, enable_categorical=True, tree_method="hist").fit(X, y)
    return model, X


def test_shap_values_add_up_and_multiply_back_to_the_valuation() -> None:
    model, X = _fake_model_and_data()
    phi, base = shap_contributions(model, X)
    margin = model.predict(X, output_margin=True)
    assert np.abs(phi.sum(axis=1).to_numpy() + base - margin).max() < 1e-4
    league_median = 20e6
    value = np.expm1(margin + np.log1p(league_median))
    product = (1 + league_median) * np.exp(base) * np.exp(phi.to_numpy()).prod(axis=1) - 1
    assert product == pytest.approx(value, rel=1e-4)


def test_grouping_keeps_totals_and_top_drivers_are_json_ready() -> None:
    model, X = _fake_model_and_data()
    phi, _ = shap_contributions(model, X)
    grouped = group_contributions(phi)
    assert "age" in grouped.columns and "age_squared" not in grouped.columns
    assert grouped.sum(axis=1).to_numpy() == pytest.approx(phi.sum(axis=1).to_numpy(), abs=1e-6)
    drivers = top_drivers(grouped.iloc[0], X.iloc[0], k=2)
    assert len(drivers) == 2 and abs(drivers[0]["phi"]) >= abs(drivers[1]["phi"])
    assert drivers[0]["effect_pct"] == pytest.approx(np.expm1(drivers[0]["phi"]))
    json.dumps(drivers)
    age = next(d for d in top_drivers(grouped.iloc[0], X.iloc[0], k=3) if d["feature"] == "age")
    assert age["value"] == pytest.approx(X.iloc[0]["age_at_valuation"])


def test_every_model_feature_has_a_driver_label() -> None:
    grouped = {m for members in FEATURE_GROUPS.values() for m in members}
    drivers = [f for f in MAIN_MODEL_FEATURES if f not in grouped] + list(FEATURE_GROUPS)
    assert sorted(drivers) == sorted(DRIVER_LABELS)


@pytest.mark.data
def test_conformal_artifact_reproduces_from_val_calib_only() -> None:
    if not (ARTIFACT_DIR / "model.ubj").exists() or not (ARTIFACT_DIR / "conformal.json").exists():
        pytest.skip("Model artifacts not present on clean clone")
    import xgboost as xgb

    from src.ml.data import TARGET, feature_frame, load_population, split

    spec = json.loads((ARTIFACT_DIR / "features.json").read_text(encoding="utf-8"))
    conf = json.loads((ARTIFACT_DIR / "conformal.json").read_text(encoding="utf-8"))
    model = xgb.XGBRegressor()
    model.load_model(ARTIFACT_DIR / "model.ubj")
    pop = load_population()
    oos = split(pop, "val_calib", "test")
    pred = model.predict(feature_frame(oos, spec["features"], spec["categorical_levels"]))
    resid = oos[TARGET].to_numpy() - pred
    offsets, _ = batch_offsets(oos["as_of_date"], batch_table(pop["as_of_date"]), oos["as_of_date"], resid)
    calib = (oos["split"] == "val_calib").to_numpy()
    shift = offsets if conf["recentering"]["enabled"] else 0.0
    ci = ConformalIntervals.fit(pred[calib], (resid - shift)[calib], levels=conf["levels"], n_bins=len(conf["bins"]["edges"]) + 1)
    assert sum(conf["n_calib"]) == int(calib.sum())
    for lv in conf["levels"]:
        assert ci.quantiles[str(lv)] == pytest.approx(conf["quantiles"][str(lv)], abs=1e-9)

    serving = conf["recentering"]["serving"]
    if serving["source_batch"]:
        assert serving["source_batch"]["end"] < serving["as_of_date"]


@pytest.mark.data
def test_registry_record_is_built_from_artifacts() -> None:
    if not (ARTIFACT_DIR / "model.ubj").exists() or not (ARTIFACT_DIR / "conformal.json").exists():
        pytest.skip("Model artifacts not present on clean clone")
    from src.ml.registry import registry_record

    rec = registry_record("v1.0.0")
    assert rec["feature_list"] == MAIN_MODEL_FEATURES
    assert rec["conformal_quantiles"]["calibration_split"] == "val_calib"
    assert set(rec["metrics"]["test_coverage"]) == {str(lv) for lv in rec["conformal_quantiles"]["levels"]}


@pytest.mark.db
def test_register_model_leaves_exactly_one_active_version() -> None:
    if not (ARTIFACT_DIR / "model.ubj").exists() or not (ARTIFACT_DIR / "conformal.json").exists():
        pytest.skip("Model artifacts not present on clean clone")
    from datetime import UTC, datetime

    from sqlalchemy import create_engine, inspect, select
    from sqlalchemy.exc import OperationalError
    from sqlalchemy.orm import Session

    from src.config import settings
    from src.db.models import ModelVersion
    from src.ml.registry import register_model

    engine = create_engine(settings.database_url)
    try:
        migrated = inspect(engine).has_table("model_versions")
    except OperationalError:
        pytest.skip("Postgres not reachable (run `make db`)")
    if not migrated:
        pytest.skip("Schema not migrated (run `make db`)")
    # Everything happens inside one outer transaction that is rolled back, so the database is untouched
    with engine.connect() as conn:
        trans = conn.begin()
        session = Session(bind=conn, join_transaction_mode="create_savepoint")
        try:
            session.execute(
                ModelVersion.__table__.update().where(ModelVersion.is_active.is_(True)).values(is_active=False)
            )
            session.add(ModelVersion(
                model_version="v0.0.0-fake", model_type="fake", data_version="fake", trained_at=datetime.now(UTC),
                artifact_path="nowhere", feature_list=[], metrics={}, conformal_quantiles={}, is_active=True,
            ))
            session.flush()
            result = register_model("v1.0.0", activate=True, session=session)
            assert result["is_active"] and result["deactivated"] == ["v0.0.0-fake"]
            active = session.scalars(select(ModelVersion.model_version).where(ModelVersion.is_active.is_(True))).all()
            assert active == ["v1.0.0"]
        finally:
            session.close()
            trans.rollback()
    engine.dispose()
