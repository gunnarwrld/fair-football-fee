"""Optuna tuning for XGBoost with expanding-window, time-ordered folds inside `train`.

Each fold trains on all train cycles before a validation cycle and validates on that cycle,
so tuning never looks at val_tune, val_calib or test, and never trains on the future.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd

from src.ml.models import make_xgb

CV_VALIDATION_CYCLES: tuple[str, ...] = ("2020/2021", "2021/2022", "2022/2023")
MAX_TREES = 3000
EARLY_STOPPING_ROUNDS = 100


def expanding_window_folds(cycles: pd.Series, validation_cycles: tuple[str, ...] = CV_VALIDATION_CYCLES) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, val_idx) pairs where every training cycle precedes the validation cycle."""
    cycles = pd.Series(np.asarray(cycles, dtype=object))
    folds = []
    for vc in validation_cycles:
        tr = np.flatnonzero((cycles < vc).to_numpy())
        va = np.flatnonzero((cycles == vc).to_numpy())
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
    }


def cv_score(params: dict[str, Any], X: pd.DataFrame, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, int]:
    """Mean fold RMSE on the target scale and mean best iteration."""
    rmses, iters = [], []
    for tr, va in folds:
        model = make_xgb(params, n_estimators=MAX_TREES, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        pred = model.predict(X.iloc[va])
        rmses.append(float(np.sqrt(np.mean((y[va] - pred) ** 2))))
        iters.append(int(model.best_iteration) + 1)
    return float(np.mean(rmses)), int(np.mean(iters))


def tune_xgb(
    X: pd.DataFrame,
    y: np.ndarray,
    cycles: pd.Series,
    n_trials: int,
    seed: int = 42,
    storage: str | None = None,
    timeout: float | None = None,
) -> dict[str, Any] | None:
    """Run (or resume) the study until it has `n_trials` completed trials.

    With `storage` (e.g. `sqlite:///artifacts/v1.0.0/optuna.db`) trials persist, so tuning can
    be split across several runs. Returns None while fewer than `n_trials` are complete. The
    saved `tuning.json` is the reproducible record; a resumed TPE sequence is not bit-identical.
    """
    folds = expanding_window_folds(cycles)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        study_name="xgb_main" if storage else None,
        load_if_exists=bool(storage),
    )

    def objective(trial: optuna.Trial) -> float:
        score, n_iter = cv_score(suggest_params(trial), X, y, folds)
        trial.set_user_attr("n_estimators", n_iter)
        return score

    def n_complete() -> int:
        return sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

    remaining = n_trials - n_complete()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, timeout=timeout)
    if n_complete() < n_trials:
        return None
    best = study.best_trial
    # The final refit uses more data than any fold, so allow ~10% more trees than the CV mean
    n_estimators = round(best.user_attrs["n_estimators"] * 1.1)
    return {
        "params": {**best.params, "n_estimators": n_estimators},
        "cv_rmse": float(best.value),
        "n_trials": n_trials,
        "cv_validation_cycles": list(CV_VALIDATION_CYCLES),
    }
