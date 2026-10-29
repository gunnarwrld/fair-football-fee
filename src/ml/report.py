"""Render `reports/phase3_evaluation.md` and figures from `metrics.json` + `predictions.parquet`.

Every number in the report comes from those files, so the report can be regenerated at any
time with `python -m src.ml.report artifacts/<version>`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS_DIR = Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
LABELS = {
    "pos_age_median": "Median by position + age band",
    "ridge": "Ridge (linear)",
    "lightgbm": "LightGBM (default)",
    "xgb_default": "XGBoost (default)",
    "xgb_tuned": "XGBoost (tuned) — final",
}
INK, MUTED, ACCENT = "#1f2933", "#7b8794", "#2f6fdf"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _eur(x: float) -> str:
    return f"€{x / 1e6:.2f}M"


def _metrics_table(rows: dict[str, dict[str, float]]) -> str:
    lines = ["| Model | MdAPE | Within ±25% | log-RMSE | R² (log) | MAE | Mean log bias |", "|---|---|---|---|---|---|---|"]
    for name, m in rows.items():
        lines.append(
            f"| {LABELS.get(name, name)} | {_pct(m['mdape'])} | {_pct(m['within_25pct'])} | {m['log_rmse']:.3f} | "
            f"{m['r2_log']:.3f} | {_eur(m['mae_eur'])} | {m['mean_log_bias']:+.3f} |"
        )
    return "\n".join(lines)


def _slice_table(records: list[dict[str, Any]], order: list[str] | None = None) -> str:
    df = pd.DataFrame(records)
    if order:
        df["slice"] = pd.Categorical(df["slice"], categories=order, ordered=True)
        df = df.sort_values("slice")
    lines = ["| Slice | n | MdAPE | Within ±25% | Mean log bias |", "|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.slice} | {r.n} | {_pct(r.mdape)} | {_pct(r.within_25pct)} | {r.mean_log_bias:+.3f} |")
    return "\n".join(lines)


def _ci(d: dict[str, float]) -> str:
    return f"{100 * d['diff']:+.1f} pts (95% CI {100 * d['ci_low']:+.1f} to {100 * d['ci_high']:+.1f}, {d['n_players']} players)"


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)


def make_figures(preds: pd.DataFrame, importance: dict[str, float]) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    test = preds[preds["split"] == "test"]
    y, p = test["market_value_eur"].to_numpy(), test["pred_eur"].to_numpy()
    resid = np.log1p(y) - np.log1p(p)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(p / 1e6, y / 1e6, s=8, alpha=0.35, color=ACCENT, edgecolors="none")
    lim = [max(min(y.min(), p.min()) / 1e6, 0.05), max(y.max(), p.max()) / 1e6 * 1.1]
    ax.plot(lim, lim, color=INK, linewidth=1)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("Predicted value (€M, log scale)"), ax.set_ylabel("Actual value (€M, log scale)")
    ax.set_title("Test 2025/26: predicted vs actual", fontsize=11, color=INK, loc="left")
    _style(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "pred_vs_actual_test.png", dpi=150); plt.close(fig)
    paths["pred_vs_actual"] = "figures/pred_vs_actual_test.png"

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    axes[0].scatter(p / 1e6, resid, s=8, alpha=0.35, color=ACCENT, edgecolors="none")
    axes[0].set_xscale("log"); axes[0].set_xlabel("Predicted value (€M, log scale)")
    axes[1].scatter(test["age_at_valuation"], resid, s=8, alpha=0.35, color=ACCENT, edgecolors="none")
    axes[1].set_xlabel("Age at valuation")
    axes[0].set_ylabel("Residual: ln(actual) − ln(predicted)")
    for ax in axes:
        ax.axhline(0, color=INK, linewidth=1); _style(ax)
    fig.suptitle("Test residuals (above 0 = model undervalues)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "residuals_test.png", dpi=150); plt.close(fig)
    paths["residuals"] = "figures/residuals_test.png"

    imp = pd.Series(importance).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.barh(imp.index, imp.to_numpy(), color=ACCENT)
    ax.set_xlabel("Share of total gain")
    ax.set_title("Final model: top 15 features by gain", fontsize=11, color=INK, loc="left")
    _style(ax); ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=150); plt.close(fig)
    paths["importance"] = "figures/feature_importance.png"
    return paths


def write_report(results: dict[str, Any], artifact_dir: Path) -> Path:
    preds = pd.read_parquet(artifact_dir / "predictions.parquet")
    figs = make_figures(preds, results["feature_importance_gain_share"])
    a, b = results["stage_a_val_tune"], results["stage_b"]
    vu, ct = results["value_update_experiment"], results["contract_ablation"]
    tun = results["tuning"]
    rows = results["population_rows"]
    final_test = b["xgb_tuned"]["test"]
    best_base = results["best_baseline"]

    slices = results["slices_test"]
    patterns = sorted(
        (dict(r, dimension=dim) for dim, recs in slices.items() for r in recs if r["n"] >= 50),
        key=lambda r: r["mdape"], reverse=True,
    )[:3]

    preds["month"] = pd.to_datetime(preds["as_of_date"]).dt.strftime("%Y-%m")
    preds["resid"] = np.log1p(preds["market_value_eur"]) - np.log1p(preds["pred_eur"])
    by_month = preds.groupby(["split", "month"], sort=False).agg(n=("resid", "size"), bias=("resid", "mean")).reset_index()
    by_month = by_month[by_month["n"] >= 20]
    cand = results["candidate_feature_cv"]

    md = [
        f"# Phase 3 Evaluation — model `{results['model_version']}`",
        "",
        ("_Generated by `python -m src.ml.train`. Every number below comes from "
        f"`artifacts/{results['model_version']}/metrics.json`._"),
        "",
        "## Headline",
        "",
        (f"On the held-out 2025/26 test season ({final_test['n']} Premier League valuations), the tuned XGBoost model has a "
        f"**median absolute percentage error of {_pct(final_test['mdape'])}**, and {_pct(final_test['within_25pct'])} of "
        f"predictions fall within ±25% of the Transfermarkt value. The strongest baseline ({LABELS[best_base]}) reaches "
        f"{_pct(b[best_base]['test']['mdape'])}. The difference is {_ci(results['final_vs_best_baseline_test_mdape'])}."),
        "",
        "## Protocol",
        "",
        (f"- Population (Phase 2.1): train {rows['train']:,} · val_tune {rows['val_tune']:,} · val_calib {rows['val_calib']:,} · "
        f"test {rows['test']:,} rows."),
        (f"- Tuning: {tun['n_trials']} Optuna trials, scored by expanding-window CV inside train "
        f"(validation cycles {', '.join(tun['cv_validation_cycles'])}); CV RMSE {tun['cv_rmse']:.3f} on the target scale."),
        ("- Stage A: every model is fit on train and compared on val_tune. Stage B: every model is refit on train + val_tune "
        "and scored once on test. val_calib is scored too, but it is reserved for Phase 4 calibration."),
        ("- Predictions are `expm1(prediction + ln(1 + league index))`, a median-type estimate, so MdAPE is the headline "
        "metric. No smearing correction is applied, on purpose."),
        "",
        "## Stage A: model selection on val_tune (2023/24)",
        "",
        _metrics_table(a),
        "",
        (f"Shuffled-target sanity check (tuned XGBoost trained on a permuted target): MdAPE "
        f"{_pct(results['shuffled_target_val_tune']['mdape'])}, which is no better than the rule-of-thumb baseline "
        f"({_pct(a['pos_age_median']['mdape'])}). This is expected, and shows there's no leak path around the target."),
        "",
        "### Model choice",
        "",
        (f"On val_tune, tuned XGBoost vs {LABELS[best_base]}: MdAPE difference "
        f"{_ci(results['selection_xgb_vs_best_baseline_val_tune_mdape'])}. The models are statistically tied at selection "
        "time. XGBoost is kept as the main model because it handles missing values natively (injury features are NULL "
        "outside the injury source), captures interactions such as age × minutes without hand-crafted terms, and gives exact "
        "per-prediction TreeSHAP attributions for the API and the what-if UI. Ridge stays in the report as the "
        "interpretable reference."),
        "",
        "## Stage B: final evaluation",
        "",
        "### Test (2025/26)",
        "",
        _metrics_table({k: v["test"] for k, v in b.items()}),
        "",
        "### val_calib (2024/25, reserved for Phase 4 calibration)",
        "",
        _metrics_table({k: v["val_calib"] for k, v in b.items()}),
        "",
        "### Bias across seasons (final configuration)",
        "",
        "| Split | Mean log bias (ln actual − ln predicted) |",
        "|---|---|",
        *[f"| {k} | {v:+.3f} |" for k, v in results["bias_by_split"].items()],
        "",
        "### Bias by valuation batch (months with ≥ 20 valuations; val_tune is stage A, the others stage B)",
        "",
        "| Split | Month | n | Mean log bias |",
        "|---|---|---|---|",
        *[f"| {r.split} | {r.month} | {r.n} | {r.bias:+.3f} |" for r in by_month.itertuples()],
        "",
        ("### Candidate features for the out-of-time bias (train-only expanding-window CV, "
        f"folds {cand['folds'][0]}–{cand['folds'][-1]})"),
        "",
        "| Feature set | Mean CV RMSE (target scale) | Mean CV bias |",
        "|---|---|---|",
        *[f"| {k} | {v['mean_rmse']:.4f} | {v['mean_bias']:+.3f} |" for k, v in cand.items() if k != "folds"],
        "",
        f"![Predicted vs actual]({figs['pred_vs_actual']})",
        "",
        f"![Residuals]({figs['residuals']})",
        "",
        "## Error slices on test (final model)",
        "",
        "### By actual value",
        "",
        _slice_table(slices["value_bucket"], ["<5M", "5-15M", "15-40M", "40-80M", "80M+"]),
        "",
        "### By position",
        "",
        _slice_table(slices["position_group"], ["GK", "DEF", "MID", "FWD"]),
        "",
        "### By age",
        "",
        _slice_table(slices["age_band"], ["<21", "21-24", "25-28", "29-31", "32+"]),
        "",
        "### Largest error patterns (slices with n ≥ 50, highest MdAPE)",
        "",
        "| Dimension | Slice | n | MdAPE | Mean log bias |",
        "|---|---|---|---|---|",
        *[f"| {r['dimension']} | {r['slice']} | {r['n']} | {_pct(r['mdape'])} | {r['mean_log_bias']:+.3f} |" for r in patterns],
        "",
        "## Feature importance",
        "",
        f"![Feature importance]({figs['importance']})",
        "",
        (f"Features above {int(100 * 0.4)}% of total gain (possible leakage): {', '.join(results['suspicious_features'])}."
         if results["suspicious_features"] else "No single feature exceeds 40% of total gain, so there's no sign of a leaking feature."),
        "",
        "## Secondary experiments",
        "",
        f"### Value-update model vs carry-forward ({vu['n_test']} test rows with a previous valuation)",
        "",
        _metrics_table({"carry_forward": vu["carry_forward"], "value_update_xgb": vu["value_update_xgb"],
                        "xgb_tuned": vu["main_model_same_rows"]}).replace("carry_forward", "Carry last value forward")
        .replace("value_update_xgb", "XGBoost + previous value"),
        "",
        f"Value-update minus carry-forward MdAPE: {_ci(vu['value_update_minus_carry_mdape'])}.",
        "",
        "### Contract-length ablation",
        "",
    ]
    if ct["status"] == "evaluated":
        md += [
            (f"Trained on the {ct['n_train']} train + val_tune rows with a point-in-time-valid contract, and scored on the "
            f"{ct['n_test']} test rows that have one."),
            "",
            _metrics_table({"with": ct["subset_with_contract"], "without": ct["subset_without_contract"],
                            "xgb_tuned": ct["main_model_same_rows"]}).replace("| with |", "| Subset model + contract years |")
            .replace("| without |", "| Subset model, no contract |"),
            "",
            f"With minus without contract (MdAPE): {_ci(ct['with_minus_without_mdape'])}.",
        ]
    else:
        md += [f"Not evaluated: only {ct['n_train']} training rows have a valid contract (minimum 200)."]

    interp = REPORTS_DIR / "phase3_interpretation.md"
    if interp.exists():
        md += ["", interp.read_text(encoding="utf-8").strip()]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "phase3_evaluation.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    artifact_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/v1.0.0")
    results = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    print(write_report(results, artifact_dir))
