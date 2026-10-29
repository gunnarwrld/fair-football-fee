"""Phase 4 entrypoint: TreeSHAP checks, conformal intervals, update-batch recentering, serving config.

    python -m src.ml.calibrate [--model-version v1.0.0]

Decision protocol (fixed before any result was seen; test never decides anything)
  Backtest: for each evaluation season E in BACKTEST_EVAL_CYCLES, fit the tuned XGBoost on every
  cycle before E-1, calibrate on E-1 and evaluate on E. That is the final layout (fit on
  train + val_tune, calibrate on val_calib, evaluate on test) moved back one season at a time,
  and it only reads train and val_tune rows.
  1. Recentering is adopted if it lowers pooled backtest MdAPE with a 95% CI entirely below
     zero. The bootstrap resamples whole batches, because the offset is shared within a batch.
  2. Mondrian bins (quartiles of the predicted ratio) replace the single global quantile if, at
     the display level, the global interval's worst bin is more than MONDRIAN_TRIGGER away from
     nominal coverage and the Mondrian worst-bin gap is smaller.
  Final: the chosen configuration is calibrated on val_calib with the saved model (trained on
  train + val_tune) and scored once on test. All four configurations are scored on test too,
  for the record only.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.ml.conformal import (
    BATCH_GAP_DAYS,
    DISPLAY_LEVEL,
    LEVELS,
    MIN_BATCH_ROWS,
    N_BINS,
    ConformalIntervals,
    batch_of,
    batch_offsets,
    batch_table,
    coverage_by,
)
from src.ml.data import TARGET, feature_frame, load_population, ratio_to_log_value, split, to_eur
from src.ml.explain import global_importance, group_contributions, shap_contributions, top_drivers
from src.ml.metrics import (
    VALUE_BUCKETS,
    band,
    block_bootstrap_ci,
    block_bootstrap_diff,
    regression_metrics,
)
from src.ml.models import load_native, make_xgb

BACKTEST_EVAL_CYCLES: tuple[str, ...] = ("2019/2020", "2020/2021", "2021/2022", "2022/2023", "2023/2024")
MONDRIAN_TRIGGER = 0.05
CONFIGS: dict[str, tuple[bool, int]] = {  # name -> (recentered, n_bins)
    "global": (False, 1),
    "global + recentering": (True, 1),
    "mondrian": (False, N_BINS),
    "mondrian + recentering": (True, N_BINS),
}
ADDITIVITY_TOLERANCE = 1e-4


def _frame(df: pd.DataFrame, pred_ratio: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """Rows with the prediction, residual and batch label needed for calibration and scoring."""
    out = df[["tm_player_id", "as_of_date", "cycle", "split", "position_group", "market_value_eur",
              "trailing_league_median_eur", TARGET]].copy()
    out["position_group"] = out["position_group"].astype(str)
    out["pred_ratio"] = np.asarray(pred_ratio, dtype=float)
    out["resid"] = out[TARGET].to_numpy() - out["pred_ratio"].to_numpy()
    ids = batch_of(out["as_of_date"], table)
    starts = table.set_index("batch")["start"].dt.strftime("%Y-%m-%d")
    out["batch"] = starts.reindex(ids).to_numpy()
    return out


def _with_offsets(frame: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Add point-in-time offsets, using only this frame's (out-of-sample) rows as sources."""
    out = frame.copy()
    out["offset"], _ = batch_offsets(out["as_of_date"], table, out["as_of_date"], out["resid"].to_numpy())
    return out


def apply_config(cal: pd.DataFrame, ev: pd.DataFrame, recentered: bool, n_bins: int) -> tuple[ConformalIntervals, pd.DataFrame]:
    """Calibrate on `cal`, then return per-row coverage and half-widths on `ev`."""
    off_cal = cal["offset"].to_numpy() if recentered else 0.0
    ci = ConformalIntervals.fit(cal["pred_ratio"].to_numpy(), cal["resid"].to_numpy() - off_cal,
                                n_bins=n_bins, recentered=recentered)
    off_ev = ev["offset"].to_numpy() if recentered else np.zeros(len(ev))
    centered = ev["resid"].to_numpy() - off_ev
    out = pd.DataFrame(index=ev.index)
    out["centered_resid"] = centered
    out["pred_ratio_served"] = ev["pred_ratio"].to_numpy() + off_ev
    for lv in LEVELS:
        hw = ci.halfwidth(ev["pred_ratio"].to_numpy(), lv)
        out[f"halfwidth_{lv}"] = hw
        out[f"covered_{lv}"] = np.abs(centered) <= hw
    return ci, out


def summarize(ev: pd.DataFrame, res: pd.DataFrame, eval_bins: np.ndarray) -> dict[str, Any]:
    """Coverage overall, per batch, per predicted-value bin and per position, plus widths."""
    y = ev["market_value_eur"].to_numpy()
    out: dict[str, Any] = {"n": len(ev), "levels": {}}
    for lv in LEVELS:
        cov = res[f"covered_{lv}"].to_numpy()
        by_bin = coverage_by(cov, eval_bins)
        out["levels"][str(lv)] = {
            "coverage": float(cov.mean()),
            "coverage_ci": block_bootstrap_ci(ev["tm_player_id"].to_numpy(), cov),
            "median_factor": float(np.median(np.exp(res[f"halfwidth_{lv}"]))),
            "by_batch": coverage_by(cov, ev["batch"]),
            "by_bin": by_bin,
            "worst_bin_gap": float(max(abs(b["coverage"] - lv) for b in by_bin)),
            "by_position": coverage_by(cov, ev["position_group"]),
            "by_actual_value": coverage_by(cov, band(y, VALUE_BUCKETS).fillna("n/a")),
        }
    pred_eur = to_eur(ratio_to_log_value(res["pred_ratio_served"].to_numpy(), ev))
    out["point_metrics"] = regression_metrics(y, pred_eur)
    return out


def _decide(backtest: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered rules to backtest results only."""
    rec = backtest["recentering_mdape_diff"]
    recentered = rec["ci_high"] < 0
    g = backtest["configs"]["global + recentering" if recentered else "global"]["levels"][str(DISPLAY_LEVEL)]
    m = backtest["configs"]["mondrian + recentering" if recentered else "mondrian"]["levels"][str(DISPLAY_LEVEL)]
    mondrian = g["worst_bin_gap"] > MONDRIAN_TRIGGER and m["worst_bin_gap"] < g["worst_bin_gap"]
    name = ("mondrian" if mondrian else "global") + (" + recentering" if recentered else "")
    return {
        "config": name,
        "recentered": recentered,
        "mondrian": mondrian,
        "reason_recentering": f"pooled backtest MdAPE diff CI high {rec['ci_high']:+.4f} {'<' if recentered else '>='} 0",
        "reason_mondrian": (
            f"global worst-bin gap {g['worst_bin_gap']:.3f} vs trigger {MONDRIAN_TRIGGER}, "
            f"Mondrian worst-bin gap {m['worst_bin_gap']:.3f}"
        ),
    }


def _shap_library_check(model: Any, X: pd.DataFrame, n: int = 200) -> float | None:
    """Max |difference| between XGBoost's native TreeSHAP and `shap.TreeExplainer` on n rows."""
    try:
        import shap
    except ImportError:
        return None
    native, _ = shap_contributions(model, X.iloc[:n])
    return float(np.abs(shap.TreeExplainer(model).shap_values(X.iloc[:n]) - native.to_numpy()).max())


def run_backtest(pop: pd.DataFrame, params: dict[str, Any], levels: dict[str, list[str]], features: list[str],
                 table: pd.DataFrame) -> dict[str, Any]:
    cycles = sorted(pop["cycle"].unique())
    ev_parts, fold_rows = [], []
    per_config: dict[str, list[pd.DataFrame]] = {name: [] for name in CONFIGS}
    bins_all: list[np.ndarray] = []
    for e in BACKTEST_EVAL_CYCLES:
        calib_cycle = cycles[cycles.index(e) - 1]
        fit_df = pop[pop["cycle"] < calib_cycle]
        oos = pop[pop["cycle"].isin([calib_cycle, e])]
        model = make_xgb(params).fit(feature_frame(fit_df, features, levels), fit_df[TARGET].to_numpy())
        frame = _with_offsets(_frame(oos, model.predict(feature_frame(oos, features, levels)), table), table)
        frame["fold"] = e
        frame["batch"] = e + " | " + frame["batch"]
        cal, ev = frame[frame["cycle"] == calib_cycle], frame[frame["cycle"] == e]
        edges = np.quantile(cal["pred_ratio"], np.linspace(0, 1, N_BINS + 1)[1:-1])
        bins_all.append(np.searchsorted(edges, ev["pred_ratio"].to_numpy(), side="right"))
        for name, (rc, nb) in CONFIGS.items():
            per_config[name].append(apply_config(cal, ev, rc, nb)[1])
        ev_parts.append(ev)
        fold_rows.append({"eval_cycle": e, "calib_cycle": calib_cycle, "n_fit": len(fit_df), "n_calib": len(cal), "n_eval": len(ev)})

    ev_all = pd.concat(ev_parts)
    eval_bins = np.concatenate(bins_all)
    configs = {name: summarize(ev_all, pd.concat(parts), eval_bins) for name, parts in per_config.items()}

    y = ev_all["market_value_eur"].to_numpy()
    plain = to_eur(ratio_to_log_value(ev_all["pred_ratio"].to_numpy(), ev_all))
    shifted = to_eur(ratio_to_log_value(ev_all["pred_ratio"].to_numpy() + ev_all["offset"].to_numpy(), ev_all))
    rec_diff = block_bootstrap_diff(ev_all["batch"].to_numpy(), np.abs(shifted - y) / y, np.abs(plain - y) / y)
    rec_diff["n_batches"] = rec_diff.pop("n_players")

    batches = (
        ev_all.assign(after=ev_all["resid"] - ev_all["offset"])
        .groupby(["fold", "batch"], sort=True)
        .agg(n=("resid", "size"), bias_before=("resid", "mean"), bias_after=("after", "mean"), offset=("offset", "first"))
        .reset_index()
    )
    big = batches[batches["n"] >= 20]
    return {
        "folds": fold_rows,
        "configs": configs,
        "recentering_mdape_diff": rec_diff,
        "batch_bias": batches.to_dict(orient="records"),
        "mean_abs_batch_bias": {"before": float(big["bias_before"].abs().mean()), "after": float(big["bias_after"].abs().mean())},
    }


def run(model_version: str) -> dict[str, Any]:
    out_dir = settings.artifacts_dir / model_version
    spec = json.loads((out_dir / "features.json").read_text(encoding="utf-8"))
    tuning = json.loads((out_dir / "tuning.json").read_text(encoding="utf-8"))
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads(settings.raw_manifest_path.read_text(encoding="utf-8"))
    features, levels = spec["features"], spec["categorical_levels"]

    pop = load_population()
    table = batch_table(pop["as_of_date"])

    # --- Backtest (train + val_tune only) and decisions
    backtest = run_backtest(split(pop, "train", "val_tune"), tuning["params"], levels, features, table)
    decision = _decide(backtest)
    recentered, n_bins = CONFIGS[decision["config"]]

    # --- Final: saved model, calibrate on val_calib, score test once
    model = load_native("xgboost").XGBRegressor()
    model.load_model(out_dir / "model.ubj")
    val_calib, test = split(pop, "val_calib"), split(pop, "test")
    oos_df = pd.concat([val_calib, test])
    frame = _with_offsets(_frame(oos_df, model.predict(feature_frame(oos_df, features, levels)), table), table)

    saved = pd.read_parquet(out_dir / "predictions.parquet")
    saved = saved.loc[saved["split"].isin(["val_calib", "test"]), ["tm_player_id", "as_of_date", "pred_eur"]]
    saved["as_of_date"] = pd.to_datetime(saved["as_of_date"]).dt.date
    check = frame.merge(saved, on=["tm_player_id", "as_of_date"], how="inner")
    reproduced = to_eur(ratio_to_log_value(check["pred_ratio"].to_numpy(), check))
    max_rel = float(np.max(np.abs(reproduced - check["pred_eur"].to_numpy()) / check["pred_eur"].to_numpy()))
    if len(check) != len(frame) or max_rel > 1e-5:
        raise RuntimeError(f"saved model does not reproduce Phase 3 predictions (max rel diff {max_rel:.2e})")

    cal, ev = frame[frame["split"] == "val_calib"], frame[frame["split"] == "test"]
    final_ci, final_res = apply_config(cal, ev, recentered, n_bins)
    eval_edges = np.quantile(cal["pred_ratio"], np.linspace(0, 1, N_BINS + 1)[1:-1])
    eval_bins = np.searchsorted(eval_edges, ev["pred_ratio"].to_numpy(), side="right")
    test_configs = {name: summarize(ev, apply_config(cal, ev, rc, nb)[1], eval_bins) for name, (rc, nb) in CONFIGS.items()}
    test_batches = (
        ev.assign(after=ev["resid"] - ev["offset"]).groupby("batch")
        .agg(n=("resid", "size"), bias_before=("resid", "mean"), bias_after=("after", "mean"), offset=("offset", "first"))
        .reset_index().to_dict(orient="records")
    )

    # --- Serving rows: same model, offset from the latest out-of-sample batch before their batch
    serving_raw = pd.read_parquet(settings.features_data_dir / "current_serving_features.parquet")
    serving = pd.concat(
        [serving_raw[["tm_player_id", "as_of_date", "trailing_league_median_eur"]].reset_index(drop=True),
         pd.DataFrame([json.loads(s) for s in serving_raw["features_json"]])],
        axis=1,
    )
    serving["as_of_date"] = pd.to_datetime(serving["as_of_date"]).dt.date
    X_serv = feature_frame(serving, features, levels)
    serving["pred_ratio"] = model.predict(X_serv)
    offsets, sources = batch_offsets(serving["as_of_date"], table, frame["as_of_date"], frame["resid"].to_numpy())
    serving["offset_raw"] = offsets
    serving["offset"] = offsets if recentered else 0.0
    if len(set(offsets.round(12))) != 1 or len(set(sources)) != 1:
        raise RuntimeError("serving rows span more than one source batch; the serving config assumes one offset")
    src = table.set_index("batch").loc[sources[0]] if sources[0] >= 0 else None

    parity = serving.merge(frame[["tm_player_id", "as_of_date", "pred_ratio", "offset"]], on=["tm_player_id", "as_of_date"],
                           suffixes=("", "_test"))
    parity_info = {
        "n_matched": len(parity),
        "max_abs_pred_diff": float(np.abs(parity["pred_ratio"] - parity["pred_ratio_test"]).max()) if len(parity) else None,
        "offsets_equal": bool(np.allclose(parity["offset_raw"], parity["offset_test"])) if len(parity) else None,
    }

    # --- TreeSHAP: additivity on test and serving, global importance, one worked example
    shap_info: dict[str, Any] = {}
    grouped_phi: dict[str, pd.DataFrame] = {}
    for name, X in (("test", feature_frame(test, features, levels)), ("serving", X_serv)):
        phi, base = shap_contributions(model, X)
        err = float(np.abs(phi.sum(axis=1).to_numpy() + base - model.predict(X, output_margin=True)).max())
        if err > ADDITIVITY_TOLERANCE:
            raise RuntimeError(f"SHAP additivity error {err:.2e} on {name}")
        grouped_phi[name] = group_contributions(phi)
        shap_info[name] = {"base_value": base, "max_additivity_error": err,
                           "global": global_importance(grouped_phi[name]).to_dict(orient="records")}
    serving_phi = grouped_phi["serving"]
    shap_info["shap_library_max_abs_diff"] = _shap_library_check(model, X_serv)

    players = pd.read_parquet(settings.clean_data_dir / "players.parquet", columns=["tm_player_id", "full_name"])
    pick = int((serving["pred_ratio"] - serving["pred_ratio"].median()).abs().idxmin())
    ex_row = serving.loc[pick]
    example = {
        "tm_player_id": int(ex_row["tm_player_id"]),
        "full_name": players.set_index("tm_player_id")["full_name"].get(int(ex_row["tm_player_id"])),
        "as_of_date": str(ex_row["as_of_date"]),
        "league_median_eur": float(ex_row["trailing_league_median_eur"]),
        "base_value": shap_info["serving"]["base_value"],
        "offset": float(ex_row["offset"]),
        "pred_ratio": float(ex_row["pred_ratio"]),
        "phi": serving_phi.loc[pick].to_dict(),
        "top_drivers": top_drivers(serving_phi.loc[pick], ex_row, k=8),
    }

    # --- Artifacts
    ci_dict = final_ci.to_dict()
    conformal = {
        "model_version": model_version,
        "data_version": manifest["data_version"],
        "created_at": datetime.now(UTC).isoformat(),
        "config": decision["config"],
        "score": "abs_log_residual",
        "display_level": DISPLAY_LEVEL,
        "levels": ci_dict["levels"],
        "bins": {"variable": "predicted_log_ratio", "edges": ci_dict["edges"]},
        "quantiles": ci_dict["quantiles"],
        "n_calib": ci_dict["n_calib"],
        "calibration_split": "val_calib",
        "recentering": {
            "enabled": recentered,
            "batch_gap_days": BATCH_GAP_DAYS,
            "min_batch_rows": MIN_BATCH_ROWS,
            "serving": {
                "as_of_date": str(serving["as_of_date"].iloc[0]),
                "offset": float(offsets[0]) if recentered else 0.0,
                "source_batch": None if src is None else {
                    "start": src["start"].strftime("%Y-%m-%d"), "end": src["end"].strftime("%Y-%m-%d"), "n_rows": int(src["n"]),
                },
            },
        },
        "interval_formula": "value_eur in expm1(pred_ratio + offset ± q[level][bin] + ln(1 + trailing_league_median_eur))",
    }
    (out_dir / "conformal.json").write_text(json.dumps(conformal, indent=2), encoding="utf-8")

    intervals = ev[["tm_player_id", "as_of_date", "batch", "market_value_eur", "trailing_league_median_eur", "pred_ratio", "offset"]].copy()
    intervals = pd.concat([intervals, final_res], axis=1)
    intervals.to_parquet(out_dir / "intervals_test.parquet", index=False)

    results = {
        "model_version": model_version,
        "data_version": manifest["data_version"],
        "model_trained_at": metadata["trained_at"],
        "levels": list(LEVELS),
        "display_level": DISPLAY_LEVEL,
        "backtest": backtest,
        "decision": decision,
        "final": {
            "n_calib": len(cal),
            "n_test": len(ev),
            "reproduced_phase3_predictions_max_rel_diff": max_rel,
            "chosen": test_configs[decision["config"]],
            "all_configs_test_record_only": test_configs,
            "test_batch_bias": test_batches,
            "conformal": conformal,
        },
        "serving": {
            "n": len(serving),
            "parity_with_test_rows": parity_info,
            "median_interval_factor": float(np.exp(np.median(final_ci.halfwidth(serving["pred_ratio"].to_numpy(), DISPLAY_LEVEL)))),
        },
        "shap": shap_info,
        "shap_example": example,
    }
    (out_dir / "phase4_metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-version", default="v1.0.0")
    args = parser.parse_args()
    results = run(args.model_version)

    from src.ml.report_phase4 import write_model_card, write_phase4_report

    report = write_phase4_report(results, Path(settings.artifacts_dir) / args.model_version)
    card = write_model_card(Path(settings.artifacts_dir) / args.model_version)
    chosen = results["final"]["chosen"]["levels"][str(DISPLAY_LEVEL)]
    print(json.dumps({
        "decision": results["decision"],
        "test_coverage_display_level": round(chosen["coverage"], 4),
        "test_median_factor": round(chosen["median_factor"], 3),
        "report": str(report),
        "model_card": str(card),
    }, indent=2))


if __name__ == "__main__":
    main()
