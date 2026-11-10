"""Evaluation metrics, slices and player-level block bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

VALUE_BUCKETS: list[tuple[str, float, float]] = [
    ("<5M", 0, 5e6),
    ("5-15M", 5e6, 15e6),
    ("15-40M", 15e6, 40e6),
    ("40-80M", 40e6, 80e6),
    ("80M+", 80e6, np.inf),
]
AGE_BANDS: list[tuple[str, float, float]] = [
    ("<21", 0, 21),
    ("21-24", 21, 25),
    ("25-28", 25, 29),
    ("29-31", 29, 32),
    ("32+", 32, np.inf),
]


def band(values: pd.Series | np.ndarray, bands: list[tuple[str, float, float]]) -> pd.Series:
    values = pd.Series(np.asarray(values, dtype=float))
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    for label, lo, hi in bands:
        out[(values >= lo) & (values < hi)] = label
    return out


def regression_metrics(y_eur: np.ndarray, pred_eur: np.ndarray) -> dict[str, float]:
    """Metrics on log scale (what the model optimizes) and EUR scale (what users read)."""
    y = np.asarray(y_eur, dtype=float)
    p = np.asarray(pred_eur, dtype=float)
    ly, lp = np.log1p(y), np.log1p(p)
    resid = ly - lp
    ape = np.abs(p - y) / y
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {
        "n": len(y),
        "log_rmse": float(np.sqrt(np.mean(resid**2))),
        "log_mae": float(np.mean(np.abs(resid))),
        "r2_log": float(1 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "mae_eur": float(np.mean(np.abs(p - y))),
        "mdape": float(np.median(ape)),
        "within_25pct": float(np.mean(ape <= 0.25)),
        "mean_log_bias": float(np.mean(resid)),
    }


def sliced_metrics(y_eur: np.ndarray, pred_eur: np.ndarray, groups: pd.Series, min_n: int = 1) -> pd.DataFrame:
    rows = []
    groups = pd.Series(np.asarray(groups, dtype=object))
    for g in pd.unique(groups.dropna()):
        mask = (groups == g).to_numpy()
        if mask.sum() < min_n:
            continue
        m = regression_metrics(np.asarray(y_eur)[mask], np.asarray(pred_eur)[mask])
        rows.append({"slice": g, **m})
    return pd.DataFrame(rows)


def block_bootstrap_diff(
    player_ids: np.ndarray,
    values_a: np.ndarray,
    values_b: np.ndarray,
    stat: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap of stat(a) - stat(b), resampling whole players.

    Rows of the same player are correlated (several valuations per season), so resampling
    rows would understate uncertainty.
    """
    ids = np.asarray(player_ids)
    a, b = np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)
    uniq, inverse = np.unique(ids, return_inverse=True)
    rows_by_player = [np.flatnonzero(inverse == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows_by_player[k] for k in pick])
        diffs[i] = stat(a[idx]) - stat(b[idx])
    return {
        "diff": float(stat(a) - stat(b)),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_players": len(uniq),
    }
