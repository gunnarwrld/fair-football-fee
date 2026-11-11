"""What-if scenarios: turn a few user-level inputs into a consistent main-model feature row.

The API applies a scenario on top of a player's precomputed serving row
(`player_current_features.features`). Every feature that depends on a changed input is recomputed
with the same formulas as `engineer.py`, so the model never sees a row the feature code couldn't
have produced:

- Age sets `age_at_valuation` and `age_squared` (both rounded like the SQL).
- Minutes in the last 12 months set `mins_last_12m`. `mins_last_24m` moves by the same amount (the
  last 12 months sit inside the 24-month window) and `minutes_trend_12m_vs_prev12m = 2·m12 − m24`.
- A count you set (appearances, goals, assists) is taken as given. A count you don't set keeps its
  per-minute rate, so changing only the minutes means "the same player with more or less playing
  time". Cards, clean sheets and goals conceded are only stored as per-90 rates, so their counts
  are read back from the rate and scaled the same way.
- Shrunk per-90 rates are recomputed from the new counts and minutes. The position prior is read
  back from the stored row (rate·(m + M0) − 90·count), so serving needs no prior file.

Held fixed: completed-season stats, club tenure, height, position, foot and every injury feature.
More injury days raise the estimate because they explain low minutes (Phase 4), so injury inputs
are deliberately not offered.
"""

from __future__ import annotations

import math
from typing import Any

from src.features.engineer import SHRINKAGE_M0_MINUTES as M0

# Range of each input in the rows the active model was trained on (train + val_tune, min and max
# rounded outward). Anything outside is extrapolation and is refused; a data test checks the
# numbers against the training population.
WHATIF_BOUNDS: dict[str, tuple[float, float]] = {
    "age_years": (17.0, 44.0),
    "minutes_last_12m": (450.0, 5800.0),
    "appearances_last_12m": (5.0, 68.0),
    "goals_last_12m": (0.0, 51.0),
    "assists_last_12m": (0.0, 31.0),
}
MAX_MINUTES_PER_APPEARANCE = 120  # 90 minutes plus extra time

# 12-month counts a scenario can set directly, and the shrunk per-90 rate each one feeds
_SHRUNK_RATE_OF_COUNT = {"goals_last_12m": "goals_per90_shrunk_12m", "assists_last_12m": "assists_per90_shrunk_12m"}
# Rates stored without their count: 90·count / (minutes + M0)
_RATE_ONLY = ("cards_per90_12m", "clean_sheets_per90_12m", "goals_conceded_per90_12m")


class ScenarioError(ValueError):
    """A scenario that is inconsistent or outside the range the model was trained on."""


def _check_range(name: str, value: float, implied: bool = False) -> None:
    lo, hi = WHATIF_BOUNDS[name]
    if not lo <= value <= hi:
        how = " (implied by keeping the per-minute rate; set it explicitly)" if implied else ""
        raise ScenarioError(f"{name}={value:g}{how} is outside the trained range [{lo:g}, {hi:g}]")


def apply_scenario(
    features: dict[str, Any], scenario: dict[str, float]
) -> tuple[dict[str, Any], dict[str, tuple[Any, Any]]]:
    """Return the scenario's feature row and {feature: (before, after)} for every changed feature."""
    unknown = sorted(set(scenario) - set(WHATIF_BOUNDS))
    if unknown:
        raise ScenarioError(f"unknown inputs: {', '.join(unknown)}")
    for name, value in scenario.items():
        _check_range(name, float(value))

    out = dict(features)
    if "age_years" in scenario:
        age = round(float(scenario["age_years"]), 2)
        out["age_at_valuation"] = age
        out["age_squared"] = round(age**2, 2)

    m = float(features["mins_last_12m"])
    m_new = float(scenario.get("minutes_last_12m", m))
    scale = m_new / m

    counts = {}
    for name in ("appearances_last_12m", *_SHRUNK_RATE_OF_COUNT):
        if name in scenario:
            counts[name] = float(scenario[name])
        else:
            counts[name] = round(float(features[name]) * scale, 2)
            _check_range(name, counts[name], implied=True)

    apps = counts["appearances_last_12m"]
    if m_new > MAX_MINUTES_PER_APPEARANCE * apps:
        need = math.ceil(m_new / MAX_MINUTES_PER_APPEARANCE)
        raise ScenarioError(
            f"{m_new:g} minutes in {apps:g} appearances is impossible: an appearance lasts at most "
            f"{MAX_MINUTES_PER_APPEARANCE} minutes, so at least {need} appearances are needed"
        )

    out["mins_last_12m"] = m_new
    out["mins_last_24m"] = round(float(features["mins_last_24m"]) + (m_new - m), 2)
    out["minutes_trend_12m_vs_prev12m"] = round(2 * m_new - out["mins_last_24m"], 2)
    out["appearances_last_12m"] = apps
    out["appearances_last_24m"] = round(
        float(features["appearances_last_24m"]) + (apps - float(features["appearances_last_12m"])), 2
    )

    for count_name, rate_name in _SHRUNK_RATE_OF_COUNT.items():
        prior_term = float(features[rate_name]) * (m + M0) - 90.0 * float(features[count_name])
        out[count_name] = counts[count_name]
        out[rate_name] = round((90.0 * counts[count_name] + prior_term) / (m_new + M0), 6)
    for rate_name in _RATE_ONLY:
        if features.get(rate_name) is not None:
            count = float(features[rate_name]) * (m + M0) / 90.0 * scale
            out[rate_name] = round(90.0 * count / (m_new + M0), 6)

    changed = {k: (features.get(k), v) for k, v in out.items() if _differs(features.get(k), v)}
    return out, changed


def _differs(before: Any, after: Any) -> bool:
    if isinstance(before, str) or isinstance(after, str):
        return before != after
    if before is None or after is None:
        return (before is None) != (after is None)
    return not math.isclose(float(before), float(after), abs_tol=1e-9)
