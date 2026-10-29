"""Phase 3 entrypoint: baselines, XGBoost (default + Optuna-tuned), evaluation, artifacts.

    python -m src.ml.train [--trials 40] [--model-version v1.0.0] [--reuse-tuning]

Protocol
  Stage A (selection): every model is fit on `train` and compared on `val_tune`.
  Stage B (final):     every model is refit on `train + val_tune` and scored once on `test`.
                       `val_calib` is also scored, but it is reserved for Phase 4 conformal
                       calibration, so nothing is tuned on it.
  Tuning uses expanding-window folds inside `train` only (see `src/ml/tuning.py`).
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost

from src.config import settings
from src.features.engineer import (
    CONTRACT_ABLATION_FEATURES,
    MAIN_MODEL_CATEGORICAL_FEATURES,
    MAIN_MODEL_FEATURES,
    MAIN_MODEL_NUMERIC_FEATURES,
    VALUE_UPDATE_MODEL_FEATURES,
)
from src.ml.data import (
    TARGET,
    category_levels,
    feature_frame,
    load_population,
    ratio_to_log_value,
    split,
    to_eur,
)
from src.ml.experiments import candidate_feature_cv
from src.ml.metrics import (
    AGE_BANDS,
    VALUE_BUCKETS,
    band,
    block_bootstrap_diff,
    regression_metrics,
    sliced_metrics,
)
from src.ml.models import (
    SEED,
    XGB_DEFAULT_PARAMS,
    PositionAgeMedianBaseline,
    make_lightgbm,
    make_ridge,
    make_xgb,
)
from src.ml.tuning import tune_xgb

MODEL_NAMES = ("pos_age_median", "ridge", "lightgbm", "xgb_default", "xgb_tuned")
BASELINE_NAMES = ("pos_age_median", "ridge", "lightgbm")
SUSPICIOUS_IMPORTANCE_SHARE = 0.40
MIN_CONTRACT_TRAIN_ROWS = 200


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_model(name: str, tuned_params: dict[str, Any]) -> Any:
    if name == "pos_age_median":
        return PositionAgeMedianBaseline()
    if name == "ridge":
        return make_ridge(MAIN_MODEL_NUMERIC_FEATURES, MAIN_MODEL_CATEGORICAL_FEATURES)
    if name == "lightgbm":
        return make_lightgbm()
    if name == "xgb_default":
        return make_xgb(XGB_DEFAULT_PARAMS)
    if name == "xgb_tuned":
        return make_xgb(tuned_params)
    raise ValueError(name)


def fit_predict(
    model: Any, train_df: pd.DataFrame, eval_dfs: dict[str, pd.DataFrame], features: list[str], levels: dict[str, list[str]],
    target: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fit on `train_df` and return EUR predictions for each evaluation frame."""
    y = train_df[TARGET].to_numpy() if target is None else target
    model.fit(feature_frame(train_df, features, levels), y)
    return {
        name: to_eur(ratio_to_log_value(model.predict(feature_frame(df, features, levels)), df))
        for name, df in eval_dfs.items()
    }


def run(
    n_trials: int, model_version: str, reuse_tuning: bool, tune_timeout: float | None = None, optuna_db: Path | None = None
) -> dict[str, Any] | None:
    t0 = time.time()
    pop = load_population()
    levels = category_levels(split(pop, "train"))
    train, val_tune = split(pop, "train"), split(pop, "val_tune")
    train_full, val_calib, test = split(pop, "train", "val_tune"), split(pop, "val_calib"), split(pop, "test")

    out_dir = settings.artifacts_dir / model_version
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Tuning (train-only expanding-window CV)
    tuning_path = out_dir / "tuning.json"
    if reuse_tuning and tuning_path.exists():
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    else:
        tuning = tune_xgb(
            feature_frame(train, MAIN_MODEL_FEATURES, levels), train[TARGET].to_numpy(), train["cycle"], n_trials,
            seed=SEED, storage=f"sqlite:///{(optuna_db or out_dir / 'optuna.db').as_posix()}", timeout=tune_timeout,
        )
        if tuning is None:
            return None
        tuning_path.write_text(json.dumps(tuning, indent=2), encoding="utf-8")
    tuned_params = tuning["params"]

    # --- Stage A: fit on train, compare on val_tune
    stage_a: dict[str, dict[str, float]] = {}
    stage_a_preds: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        preds = fit_predict(build_model(name, tuned_params), train, {"val_tune": val_tune}, MAIN_MODEL_FEATURES, levels)
        stage_a_preds[name] = preds["val_tune"]
        stage_a[name] = regression_metrics(val_tune["market_value_eur"].to_numpy(), preds["val_tune"])

    # Is the tuned XGBoost distinguishable from the best baseline at selection time?
    best_baseline = min(BASELINE_NAMES, key=lambda n: stage_a[n]["mdape"])
    y_vt = val_tune["market_value_eur"].to_numpy()
    selection_vs_baseline = block_bootstrap_diff(
        val_tune["tm_player_id"].to_numpy(),
        np.abs(stage_a_preds["xgb_tuned"] - y_vt) / y_vt,
        np.abs(stage_a_preds[best_baseline] - y_vt) / y_vt,
    )

    # Sanity check: the tuned model on a shuffled target must not beat the rule-of-thumb baseline
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(train[TARGET].to_numpy())
    shuffled_pred = fit_predict(
        build_model("xgb_tuned", tuned_params), train, {"val_tune": val_tune}, MAIN_MODEL_FEATURES, levels, target=shuffled
    )["val_tune"]
    shuffled_metrics = regression_metrics(val_tune["market_value_eur"].to_numpy(), shuffled_pred)

    # --- Stage B: refit on train + val_tune, score val_calib and test once
    stage_b: dict[str, dict[str, dict[str, float]]] = {}
    stage_b_preds: dict[str, dict[str, np.ndarray]] = {}
    final_model = None
    for name in MODEL_NAMES:
        model = build_model(name, tuned_params)
        preds = fit_predict(model, train_full, {"val_calib": val_calib, "test": test}, MAIN_MODEL_FEATURES, levels)
        stage_b_preds[name] = preds
        stage_b[name] = {
            s: regression_metrics(df["market_value_eur"].to_numpy(), preds[s]) for s, df in (("val_calib", val_calib), ("test", test))
        }
        if name == "xgb_tuned":
            final_model = model
    assert final_model is not None

    y_test = test["market_value_eur"].to_numpy()
    final_test_pred = stage_b_preds["xgb_tuned"]["test"]
    ape_final = np.abs(final_test_pred - y_test) / y_test
    ape_base = np.abs(stage_b_preds[best_baseline]["test"] - y_test) / y_test
    vs_baseline = block_bootstrap_diff(test["tm_player_id"].to_numpy(), ape_final, ape_base)

    # --- Slices on test (final model)
    slices = {
        "value_bucket": sliced_metrics(y_test, final_test_pred, band(y_test, VALUE_BUCKETS)),
        "position_group": sliced_metrics(y_test, final_test_pred, test["position_group"].astype(str).to_numpy()),
        "age_band": sliced_metrics(y_test, final_test_pred, band(test["age_at_valuation"], AGE_BANDS)),
    }
    bias_by_split = {
        "val_tune (stage A)": stage_a["xgb_tuned"]["mean_log_bias"],
        "val_calib (stage B)": stage_b["xgb_tuned"]["val_calib"]["mean_log_bias"],
        "test (stage B)": stage_b["xgb_tuned"]["test"]["mean_log_bias"],
    }

    importance = (
        pd.Series(final_model.get_booster().get_score(importance_type="gain"))
        .reindex(MAIN_MODEL_FEATURES).fillna(0.0)
    )
    importance = (importance / importance.sum()).sort_values(ascending=False)
    suspicious = importance[importance > SUSPICIOUS_IMPORTANCE_SHARE].index.tolist()

    # --- Secondary experiment 1: value-update model vs carry-forward
    vu_test = test[test["prev_market_value_eur"].notna()]
    vu_pred = fit_predict(make_xgb(tuned_params), train_full, {"test": vu_test}, VALUE_UPDATE_MODEL_FEATURES, levels)["test"]
    y_vu = vu_test["market_value_eur"].to_numpy()
    carry = vu_test["prev_market_value_eur"].to_numpy(dtype=float)
    main_on_vu = final_test_pred[test["prev_market_value_eur"].notna().to_numpy()]
    value_update = {
        "n_test": len(vu_test),
        "carry_forward": regression_metrics(y_vu, carry),
        "value_update_xgb": regression_metrics(y_vu, vu_pred),
        "main_model_same_rows": regression_metrics(y_vu, main_on_vu),
        "value_update_minus_carry_mdape": block_bootstrap_diff(
            vu_test["tm_player_id"].to_numpy(), np.abs(vu_pred - y_vu) / y_vu, np.abs(carry - y_vu) / y_vu
        ),
    }

    # --- Secondary experiment 2: contract ablation (rows with a point-in-time-valid contract only)
    c_train = train_full[train_full["valid_contract_years_remaining"].notna()]
    c_test_mask = test["valid_contract_years_remaining"].notna().to_numpy()
    c_test = test[c_test_mask]
    contract: dict[str, Any] = {"n_train": len(c_train), "n_test": len(c_test)}
    if len(c_train) >= MIN_CONTRACT_TRAIN_ROWS and len(c_test) > 0:
        y_c = c_test["market_value_eur"].to_numpy()
        with_c = fit_predict(make_xgb(tuned_params), c_train, {"test": c_test}, CONTRACT_ABLATION_FEATURES, levels)["test"]
        without_c = fit_predict(make_xgb(tuned_params), c_train, {"test": c_test}, MAIN_MODEL_FEATURES, levels)["test"]
        contract.update({
            "status": "evaluated",
            "subset_with_contract": regression_metrics(y_c, with_c),
            "subset_without_contract": regression_metrics(y_c, without_c),
            "main_model_same_rows": regression_metrics(y_c, final_test_pred[c_test_mask]),
            "with_minus_without_mdape": block_bootstrap_diff(
                c_test["tm_player_id"].to_numpy(), np.abs(with_c - y_c) / y_c, np.abs(without_c - y_c) / y_c
            ),
        })
    else:
        contract["status"] = "insufficient_rows"

    # --- Candidate features for the out-of-time bias (decided by train-only CV)
    candidates = candidate_feature_cv(train, tuned_params, levels)

    # --- Artifacts
    final_model.save_model(out_dir / "model.ubj")
    manifest = json.loads((settings.raw_data_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    features_spec = {
        "features": MAIN_MODEL_FEATURES,
        "categorical_levels": levels,
        "target": TARGET,
        "back_transform": "value_eur = expm1(prediction + ln(1 + trailing_league_median_eur))",
    }
    (out_dir / "features.json").write_text(json.dumps(features_spec, indent=2), encoding="utf-8")

    pred_frames = []
    for s, df in (("val_calib", val_calib), ("test", test)):
        frame = df[["tm_player_id", "as_of_date", "split", "position_group", "age_at_valuation", "market_value_eur",
                    "trailing_league_median_eur"]].copy()
        frame["pred_eur"] = stage_b_preds["xgb_tuned"][s]
        pred_frames.append(frame)
    frame = val_tune[["tm_player_id", "as_of_date", "split", "position_group", "age_at_valuation", "market_value_eur",
                      "trailing_league_median_eur"]].copy()
    frame["pred_eur"] = stage_a_preds["xgb_tuned"]
    pred_frames.insert(0, frame)
    pd.concat(pred_frames).to_parquet(out_dir / "predictions.parquet", index=False)

    def _range(df: pd.DataFrame) -> dict[str, Any]:
        return {"n": len(df), "from": str(df["as_of_date"].min()), "to": str(df["as_of_date"].max())}

    metadata = {
        "model_version": model_version,
        "model_type": "xgboost_regressor",
        "trained_at": datetime.now(UTC).isoformat(),
        "data_version": manifest["data_version"],
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "seed": SEED,
        "params": tuned_params,
        "training_rows": {"train+val_tune": _range(train_full)},
        "evaluation_rows": {"val_calib": _range(val_calib), "test": _range(test)},
        "libraries": {
            "python": platform.python_version(), "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
            "scikit-learn": sklearn.__version__, "optuna": optuna.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results = {
        "model_version": model_version,
        "tuning": tuning,
        "population_rows": {s: int((pop["split"] == s).sum()) for s in ("train", "val_tune", "val_calib", "test")},
        "stage_a_val_tune": stage_a,
        "shuffled_target_val_tune": shuffled_metrics,
        "stage_b": stage_b,
        "best_baseline": best_baseline,
        "selection_xgb_vs_best_baseline_val_tune_mdape": selection_vs_baseline,
        "candidate_feature_cv": candidates,
        "final_vs_best_baseline_test_mdape": vs_baseline,
        "bias_by_split": bias_by_split,
        "slices_test": {k: v.to_dict(orient="records") for k, v in slices.items()},
        "feature_importance_gain_share": importance.round(6).to_dict(),
        "suspicious_features": suspicious,
        "value_update_experiment": value_update,
        "contract_ablation": contract,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--model-version", default="v1.0.0")
    parser.add_argument("--reuse-tuning", action="store_true", help="reuse artifacts/<version>/tuning.json if present")
    parser.add_argument("--tune-timeout", type=float, default=None, help="stop tuning after N seconds; rerun to resume")
    parser.add_argument("--optuna-db", type=Path, default=None, help="Optuna SQLite path (default artifacts/<version>/optuna.db)")
    args = parser.parse_args()
    results = run(args.trials, args.model_version, args.reuse_tuning, args.tune_timeout, args.optuna_db)
    if results is None:
        print(f"Tuning incomplete (< {args.trials} trials). Rerun the same command to resume.")
        return

    from src.ml.report import write_report

    report_path = write_report(results, Path(settings.artifacts_dir) / args.model_version)
    print(json.dumps({
        "model_version": results["model_version"],
        "val_tune_mdape": {k: round(v["mdape"], 4) for k, v in results["stage_a_val_tune"].items()},
        "test_mdape": {k: round(v["test"]["mdape"], 4) for k, v in results["stage_b"].items()},
        "report": str(report_path),
        "runtime_seconds": results["runtime_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
