"""Model factories: two baselines, LightGBM, and XGBoost (main model)."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ml.metrics import AGE_BANDS, band

SEED = 42


def load_native(module: str) -> ModuleType:
    """Import XGBoost/LightGBM lazily, so baselines and tests don't need their native libraries.

    Both ship macOS wheels linked against the OpenMP runtime, which macOS doesn't include.
    """
    try:
        return importlib.import_module(module)
    except OSError as exc:  # dlopen failure, e.g. "Library not loaded: @rpath/libomp.dylib"
        raise RuntimeError(
            f"{module} is installed but its native library failed to load ({exc}). "
            "On macOS install the OpenMP runtime: `brew install libomp`."
        ) from exc

# Untuned reference configuration (fixed tree count: val_tune is never used for early stopping)
XGB_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


class PositionAgeMedianBaseline:
    """Median target by (position group, age band), falling back to position, then global.

    This is the "what a scout's rule of thumb would say" floor every model must beat.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> PositionAgeMedianBaseline:
        df = pd.DataFrame({
            "pos": X["position_group"].astype(str).to_numpy(),
            "age": band(X["age_at_valuation"], AGE_BANDS).to_numpy(),
            "y": np.asarray(y, dtype=float),
        })
        self.by_pos_age_ = df.groupby(["pos", "age"])["y"].median().to_dict()
        self.by_pos_ = df.groupby("pos")["y"].median().to_dict()
        self.global_ = float(df["y"].median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pos = X["position_group"].astype(str).to_numpy()
        age = band(X["age_at_valuation"], AGE_BANDS).to_numpy()
        return np.array([
            self.by_pos_age_.get((p, a), self.by_pos_.get(p, self.global_)) for p, a in zip(pos, age, strict=True)
        ])


def make_ridge(numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("pre", pre), ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25)))])


def make_lightgbm() -> Any:
    lgb = load_native("lightgbm")
    return lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=SEED, n_jobs=4, verbose=-1,
    )


def make_xgb(params: dict[str, Any], **overrides: Any) -> Any:
    xgb = load_native("xgboost")
    cfg = {
        "tree_method": "hist",
        "enable_categorical": True,
        "max_cat_to_onehot": 8,
        "objective": "reg:squarederror",
        "random_state": SEED,
        "n_jobs": 4,
        **params,
        **overrides,
    }
    return xgb.XGBRegressor(**cfg)
