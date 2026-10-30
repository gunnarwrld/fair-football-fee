"""Serving-time valuation of one feature row: estimate, prediction intervals and SHAP drivers.

Applies exactly the Phase 4 serving config (`conformal.json`, stored as
`model_versions.conformal_quantiles`):
    estimate = expm1(pred_ratio + offset + ln(1 + M))
    interval = expm1(pred_ratio + offset ± q[level][bin(pred_ratio)] + ln(1 + M))
where M is the trailing league median of the row and the bin comes from the raw model output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.conformal import ConformalIntervals
from src.ml.data import feature_frame
from src.ml.explain import DRIVER_LABELS, group_contributions, shap_contributions, top_drivers
from src.ml.models import load_native


@dataclass(frozen=True)
class Interval:
    level: float
    low_eur: float
    high_eur: float


@dataclass(frozen=True)
class Valuation:
    estimate_eur: float
    pred_ratio: float  # raw model output: ln((1 + value) / (1 + M)) before recentering
    offset: float
    league_median_eur: float
    baseline_eur: float  # a typical training-population player at today's league level
    intervals: list[Interval]
    drivers: list[dict[str, Any]]  # every grouped driver, largest |phi| first, with labels


class Predictor:
    def __init__(self, model: Any, features: list[str], levels: dict[str, list[str]], conformal: dict[str, Any]):
        self.model = model
        self.features = features
        self.levels = levels
        self.conformal = conformal
        self.display_level = float(conformal["display_level"])
        self.intervals = ConformalIntervals(
            levels=[float(lv) for lv in conformal["levels"]],
            edges=list(conformal["bins"]["edges"]),
            quantiles=conformal["quantiles"],
            n_calib=list(conformal["n_calib"]),
            recentered=bool(conformal["recentering"]["enabled"]),
        )
        rec = conformal["recentering"]
        self.offset = float(rec["serving"]["offset"]) if rec["enabled"] else 0.0

    @classmethod
    def from_artifacts(cls, artifact_dir: Path, conformal: dict[str, Any] | None = None) -> Predictor:
        """Load `model.ubj` + `features.json`; `conformal` defaults to the artifact's `conformal.json`."""
        model = load_native("xgboost").XGBRegressor()
        model.load_model(artifact_dir / "model.ubj")
        # One thread per prediction: the API already runs requests in parallel threads
        model.set_params(n_jobs=1)
        model.get_booster().set_param({"nthread": 1})
        spec = json.loads((artifact_dir / "features.json").read_text(encoding="utf-8"))
        if conformal is None:
            conformal = json.loads((artifact_dir / "conformal.json").read_text(encoding="utf-8"))
        return cls(model, spec["features"], spec["categorical_levels"], conformal)

    def value(self, features: dict[str, Any], league_median_eur: float) -> Valuation:
        X = feature_frame(pd.DataFrame([features]), self.features, self.levels)
        pred_ratio = float(self.model.predict(X)[0])
        phi, base = shap_contributions(self.model, X)
        grouped = group_contributions(phi).iloc[0]

        log_m = float(np.log1p(league_median_eur))
        center = pred_ratio + self.offset
        intervals = []
        for level in self.intervals.levels:
            q = float(self.intervals.halfwidth(np.array([pred_ratio]), level)[0])
            intervals.append(Interval(level, float(np.expm1(center - q + log_m)), float(np.expm1(center + q + log_m))))
        drivers = [
            {**d, "label": DRIVER_LABELS.get(d["feature"], d["feature"])}
            for d in top_drivers(grouped, features, k=len(grouped))
        ]
        return Valuation(
            estimate_eur=float(np.expm1(center + log_m)),
            pred_ratio=pred_ratio,
            offset=self.offset,
            league_median_eur=float(league_median_eur),
            baseline_eur=float(np.expm1(base + log_m)),
            intervals=intervals,
            drivers=drivers,
        )
