"""Phase 3: ML layer tests (metrics, folds, baseline, encoding, artifact reproducibility)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import MAIN_MODEL_FEATURES
from src.ml.data import feature_frame, ratio_to_log_value, to_eur
from src.ml.metrics import block_bootstrap_diff, regression_metrics
from src.ml.models import PositionAgeMedianBaseline
from src.ml.tuning import expanding_window_folds

ARTIFACT_DIR = Path("artifacts/v1.0.0")


def test_regression_metrics_known_values() -> None:
    y = np.array([10e6, 20e6, 40e6, 5e6])
    p = np.array([10e6, 25e6, 30e6, 5e6])  # APE: 0, 0.25, 0.25, 0
    m = regression_metrics(y, p)
    assert m["mdape"] == pytest.approx(0.125)
    assert m["within_25pct"] == pytest.approx(1.0)
    assert m["mae_eur"] == pytest.approx(3.75e6)
    resid = np.log1p(y) - np.log1p(p)
    assert m["log_rmse"] == pytest.approx(float(np.sqrt(np.mean(resid**2))))
    assert m["mean_log_bias"] == pytest.approx(float(resid.mean()))


def test_block_bootstrap_is_deterministic_and_zero_for_identical_errors() -> None:
    ids = np.repeat(np.arange(30), 3)
    a = np.random.default_rng(0).random(90)
    same = block_bootstrap_diff(ids, a, a)
    assert same["diff"] == 0.0 and same["ci_low"] == 0.0 and same["ci_high"] == 0.0
    assert same["n_players"] == 30
    b = a + 0.1
    r1, r2 = block_bootstrap_diff(ids, a, b, seed=7), block_bootstrap_diff(ids, a, b, seed=7)
    assert r1 == r2
    assert r1["ci_low"] <= r1["diff"] <= r1["ci_high"] < 0


def test_expanding_window_folds_never_train_on_the_future() -> None:
    cycles = pd.Series(["2019/2020"] * 3 + ["2020/2021"] * 2 + ["2021/2022"] * 2 + ["2022/2023"] * 2)
    folds = expanding_window_folds(cycles, ("2020/2021", "2021/2022", "2022/2023"))
    assert len(folds) == 3
    for (tr, va), val_cycle in zip(folds, ("2020/2021", "2021/2022", "2022/2023"), strict=True):
        assert set(cycles.iloc[va]) == {val_cycle}
        assert (cycles.iloc[tr] < val_cycle).all()
        assert not set(tr) & set(va)


def test_position_age_baseline_falls_back_to_position_then_global() -> None:
    X = pd.DataFrame({"position_group": ["FWD", "FWD", "DEF"], "age_at_valuation": [22.0, 23.0, 30.0]})
    model = PositionAgeMedianBaseline().fit(X, np.array([1.0, 3.0, -1.0]))
    pred = model.predict(pd.DataFrame({"position_group": ["FWD", "FWD", "GK"], "age_at_valuation": [21.5, 33.0, 25.0]}))
    assert pred[0] == pytest.approx(2.0)  # (FWD, 21-24) median
    assert pred[1] == pytest.approx(2.0)  # FWD 32+ unseen -> FWD median
    assert pred[2] == pytest.approx(1.0)  # GK unseen -> global median


def test_feature_frame_maps_unknown_categories_to_missing() -> None:
    df = pd.DataFrame({"position_group": ["MID", "XX"], "preferred_foot": ["left", "right"], "age_at_valuation": [25.0, 26.0]})
    X = feature_frame(df, ["position_group", "preferred_foot", "age_at_valuation"],
                      {"position_group": ["DEF", "MID"], "preferred_foot": ["left", "right"]})
    assert list(X["position_group"].cat.categories) == ["DEF", "MID"]
    assert X["position_group"].isna().tolist() == [False, True]


def test_ratio_back_transform_round_trips() -> None:
    df = pd.DataFrame({"trailing_league_median_eur": [20e6, 5e6], "market_value_eur": [30e6, 2e6]})
    ratio = np.log1p(df["market_value_eur"]) - np.log1p(df["trailing_league_median_eur"])
    assert to_eur(ratio_to_log_value(ratio.to_numpy(), df)) == pytest.approx(df["market_value_eur"].to_numpy())


@pytest.mark.data
def test_saved_model_reproduces_reported_test_metrics() -> None:
    if not (ARTIFACT_DIR / "model.ubj").exists():
        pytest.skip("Model artifacts not present on clean clone")
    import xgboost as xgb

    from src.ml.data import load_population, split

    spec = json.loads((ARTIFACT_DIR / "features.json").read_text(encoding="utf-8"))
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert spec["features"] == MAIN_MODEL_FEATURES

    model = xgb.XGBRegressor()
    model.load_model(ARTIFACT_DIR / "model.ubj")
    test = split(load_population(), "test")
    pred = to_eur(ratio_to_log_value(model.predict(feature_frame(test, spec["features"], spec["categorical_levels"])), test))
    reproduced = regression_metrics(test["market_value_eur"].to_numpy(), pred)
    assert reproduced["mdape"] == pytest.approx(metrics["stage_b"]["xgb_tuned"]["test"]["mdape"], abs=1e-9)


@pytest.mark.data
def test_final_model_never_trained_on_calibration_or_test_rows() -> None:
    meta_path = ARTIFACT_DIR / "metadata.json"
    if not meta_path.exists():
        pytest.skip("Model artifacts not present on clean clone")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["training_rows"]["train+val_tune"]["to"] < "2024-07-01"
    assert meta["evaluation_rows"]["val_calib"]["from"] >= "2024-07-01"
    preds = pd.read_parquet(ARTIFACT_DIR / "predictions.parquet")
    assert set(preds["split"]) == {"val_tune", "val_calib", "test"}
