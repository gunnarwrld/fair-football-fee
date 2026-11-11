"""Exact TreeSHAP attributions for the main model, in log space and as multiplicative effects.

XGBoost computes exact path-dependent TreeSHAP natively (`pred_contribs=True`). It is the same
algorithm as `shap.TreeExplainer`, it handles the model's categorical splits, and it keeps the
serving container free of a second explainer library.

The model predicts r = ln((1 + value) / (1 + M)), where M is the league median index. SHAP splits
r additively, r = phi_0 + sum_i phi_i, which on the value scale is a product:
    (1 + value) = (1 + M) * exp(phi_0) * prod_i exp(phi_i)
so feature i multiplies the valuation by exp(phi_i), shown as `exp(phi_i) - 1` (e.g. +18%).
`phi_0` is the model's average log ratio, so (1 + M) * exp(phi_0) - 1 is the value of a typical
player in the training population at today's league level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Features that encode one input and should be read (and summed) as one driver
FEATURE_GROUPS: dict[str, list[str]] = {"age": ["age_at_valuation", "age_squared"]}
# The feature whose value is shown next to a grouped driver
GROUP_DISPLAY_FEATURE: dict[str, str] = {"age": "age_at_valuation"}
# Plain-language driver names for reports, the API and the UI (grouped names included)
DRIVER_LABELS: dict[str, str] = {
    "age": "Age",
    "position_group": "Position",
    "preferred_foot": "Preferred foot",
    "height_cm": "Height",
    "club_tenure_days_capped": "Time at current club",
    "prior_season_appearances": "Appearances, last completed season",
    "prior_season_minutes": "Minutes, last completed season",
    "prior_season_goals": "Goals, last completed season",
    "prior_season_assists": "Assists, last completed season",
    "prior_2seasons_minutes": "Minutes, last two completed seasons",
    "prior_season_goals_per90_shrunk": "Goals per 90, last completed season",
    "prior_season_assists_per90_shrunk": "Assists per 90, last completed season",
    "mins_last_12m": "Minutes, last 12 months",
    "mins_last_24m": "Minutes, last 24 months",
    "appearances_last_12m": "Appearances, last 12 months",
    "appearances_last_24m": "Appearances, last 24 months",
    "goals_last_12m": "Goals, last 12 months",
    "assists_last_12m": "Assists, last 12 months",
    "goals_per90_shrunk_12m": "Goals per 90, last 12 months",
    "assists_per90_shrunk_12m": "Assists per 90, last 12 months",
    "cards_per90_12m": "Cards per 90, last 12 months",
    "clean_sheets_per90_12m": "Clean sheets per 90, last 12 months",
    "goals_conceded_per90_12m": "Goals conceded per 90, last 12 months",
    "minutes_trend_12m_vs_prev12m": "Minutes trend vs the previous 12 months",
    "injury_days_12m": "Days injured, last 12 months",
    "injury_days_24m": "Days injured, last 24 months",
    "injury_count_12m": "Injuries, last 12 months",
    "injury_count_24m": "Injuries, last 24 months",
    "is_currently_injured_at_t": "Currently injured",
    "days_since_last_injury_end_capped": "Days since last injury ended",
}


def shap_contributions(model: Any, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Per-row SHAP values (log-ratio units, one column per feature) and the base value phi_0."""
    import xgboost as xgb  # the model object exists, so the native library already loaded

    contribs = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    phi = pd.DataFrame(contribs[:, :-1], columns=list(X.columns), index=X.index)
    return phi, float(contribs[0, -1])


def group_contributions(phi: pd.DataFrame) -> pd.DataFrame:
    """Sum grouped features into one column each. Row totals are unchanged (SHAP is additive)."""
    out = phi.copy()
    for group, members in FEATURE_GROUPS.items():
        present = [m for m in members if m in out.columns]
        if present:
            out[group] = out[present].sum(axis=1)
            out = out.drop(columns=present)
    return out


def global_importance(phi_grouped: pd.DataFrame) -> pd.DataFrame:
    """Mean |phi| per driver and the typical multiplicative effect it implies, largest first."""
    mean_abs = phi_grouped.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({
        "driver": mean_abs.index,
        "mean_abs_phi": mean_abs.to_numpy(),
        "typical_effect_pct": np.expm1(mean_abs.to_numpy()),
    })


def top_drivers(phi_row: pd.Series, feature_row: pd.Series | dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    """The k largest |phi| drivers of one prediction, as JSON-ready dicts (grouped phi expected)."""
    def _value(name: str) -> Any:
        v = feature_row.get(GROUP_DISPLAY_FEATURE.get(name, name))
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return None
        return v.item() if isinstance(v, np.generic) else (v if isinstance(v, (int, float)) else str(v))

    order = phi_row.abs().sort_values(ascending=False).index[:k]
    return [
        {
            "feature": name,
            "value": _value(name),
            "phi": float(phi_row[name]),
            "multiplier": float(np.exp(phi_row[name])),
            "effect_pct": float(np.expm1(phi_row[name])),
        }
        for name in order
    ]
