"""Load the Phase 2.1 modeling population and convert between target and EUR scales.

Target: `target_log_ratio_vs_league_median = ln((y + 1) / (M + 1))`, where `M` is the
trailing-365-day league median over the same population (point-in-time).
A prediction `r_hat` maps back to EUR as `expm1(r_hat + ln(M + 1))`. That is a median-type
estimate (exp of a log-scale prediction), which is why MdAPE is the headline metric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.features.engineer import MAIN_MODEL_CATEGORICAL_FEATURES

TARGET = "target_log_ratio_vs_league_median"
SPLITS: tuple[str, ...] = ("train", "val_tune", "val_calib", "test")
ID_COLUMNS: list[str] = ["tm_player_id", "as_of_date", "cycle", "split", "market_value_eur", "trailing_league_median_eur"]


def load_population(path: Path | None = None) -> pd.DataFrame:
    """All `in_model_population` snapshots, sorted by date, with categorical dtypes set."""
    path = path or settings.features_data_dir / "valuation_snapshots.parquet"
    df = pd.read_parquet(path)
    pop = df[df["in_model_population"]].copy()
    pop["as_of_date"] = pd.to_datetime(pop["as_of_date"]).dt.date
    pop = pop.sort_values(["as_of_date", "tm_player_id"]).reset_index(drop=True)
    return pop


def category_levels(df: pd.DataFrame) -> dict[str, list[str]]:
    """Fixed category levels (saved with the model so serving encodes identically)."""
    return {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in MAIN_MODEL_CATEGORICAL_FEATURES}


def feature_frame(df: pd.DataFrame, features: list[str], levels: dict[str, list[str]]) -> pd.DataFrame:
    """Select model features and apply fixed categorical levels (unknown levels become NaN).

    Numeric columns that arrive as Python objects (a serving row built from JSON, where a missing
    injury history is `null`) become numbers with NaN, which XGBoost treats as missing.
    """
    X = df[features].copy()
    for col, cats in levels.items():
        if col in X.columns:
            values = X[col].astype("string")
            X[col] = pd.Categorical(values.where(values.isin(cats)), categories=cats)
    for col in X.columns:
        if col not in levels and pd.api.types.is_object_dtype(X[col]):
            X[col] = pd.to_numeric(X[col])
    return X


def split(df: pd.DataFrame, *names: str) -> pd.DataFrame:
    return df[df["split"].isin(names)]


def log_value(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df["market_value_eur"].to_numpy(dtype=float))


def ratio_to_log_value(pred_ratio: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(pred_ratio, dtype=float) + np.log1p(df["trailing_league_median_eur"].to_numpy(dtype=float))


def to_eur(log_values: np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(log_values, dtype=float))
