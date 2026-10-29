"""Candidate features evaluated with train-only expanding-window CV (reproducible record).

Phase 3 saw an end-of-season over-prediction on test. Two point-in-time candidates were
tested to explain it, and the decision rule is train-internal CV, never the test split:
  - `days_into_season`: days since 1 July of the valuation's season.
  - `index_growth_365d`: ln(index_t) - ln(index_{t-365}), the speed of market inflation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.engineer import MAIN_MODEL_FEATURES
from src.ml.data import TARGET, feature_frame
from src.ml.models import make_xgb
from src.ml.tuning import expanding_window_folds

CANDIDATE_CV_CYCLES: tuple[str, ...] = ("2017/2018", "2018/2019", "2019/2020", "2020/2021", "2021/2022", "2022/2023")


def add_candidate_features(pop: pd.DataFrame) -> pd.DataFrame:
    out = pop.copy()
    dt = pd.to_datetime(out["as_of_date"]).astype("datetime64[ns]")
    season_start = pd.to_datetime((dt.dt.year - (dt.dt.month < 7).astype(int)).astype(str) + "-07-01")
    out["days_into_season"] = (dt - season_start).dt.days.astype(float)

    index = (
        pd.DataFrame({"dt": dt, "index": out["trailing_league_median_eur"].to_numpy()})
        .drop_duplicates("dt").sort_values("dt")
    )
    lookup = pd.DataFrame({"dt_prev": (dt - pd.Timedelta(days=365)).astype("datetime64[ns]"), "row": np.arange(len(out))})
    prev = pd.merge_asof(
        lookup.sort_values("dt_prev"), index.rename(columns={"dt": "dt_idx", "index": "index_prev"}),
        left_on="dt_prev", right_on="dt_idx", direction="backward",
    ).sort_values("row")
    out["index_growth_365d"] = np.log1p(out["trailing_league_median_eur"].to_numpy()) - np.log1p(prev["index_prev"].to_numpy())
    return out


def candidate_feature_cv(train: pd.DataFrame, params: dict[str, Any], levels: dict[str, list[str]]) -> dict[str, Any]:
    """Mean out-of-time RMSE and bias on the target scale for main vs main + each candidate."""
    train = add_candidate_features(train).reset_index(drop=True)
    folds = expanding_window_folds(train["cycle"], CANDIDATE_CV_CYCLES)
    y = train[TARGET].to_numpy()
    results: dict[str, Any] = {}
    for label, extra in (("main", []), ("main + days_into_season", ["days_into_season"]),
                         ("main + index_growth_365d", ["index_growth_365d"])):
        X = feature_frame(train, MAIN_MODEL_FEATURES + extra, levels)
        rmse, bias = [], []
        for tr, va in folds:
            pred = make_xgb(params).fit(X.iloc[tr], y[tr]).predict(X.iloc[va])
            rmse.append(float(np.sqrt(np.mean((y[va] - pred) ** 2))))
            bias.append(float(np.mean(y[va] - pred)))
        results[label] = {"mean_rmse": float(np.mean(rmse)), "mean_bias": float(np.mean(bias)), "fold_rmse": rmse, "fold_bias": bias}
    results["folds"] = list(CANDIDATE_CV_CYCLES)
    return results
