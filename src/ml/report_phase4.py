"""Render `reports/phase4_explain_intervals.md`, the model card and Phase 4 figures.

Every number comes from `artifacts/<version>/{metadata,metrics,phase4_metrics,conformal}.json`
or `intervals_test.parquet`, so both documents can be regenerated at any time with
`python -m src.ml.report_phase4 artifacts/<version>`.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ml.conformal import ConformalIntervals
from src.ml.explain import DRIVER_LABELS
from src.ml.metrics import VALUE_BUCKETS
from src.ml.report import (
    ACCENT,
    FIG_DIR,
    INK,
    MUTED,
    REPORTS_DIR,
    _eur,
    _metrics_table,
    _pct,
    _style,
)

WARN = "#c2410c"


def _label(driver: str) -> str:
    return DRIVER_LABELS.get(driver, driver)


def _cov(level: dict[str, Any]) -> str:
    ci = level["coverage_ci"]
    return f"{_pct(level['coverage'])} (95% CI {_pct(ci['ci_low'])}–{_pct(ci['ci_high'])})"


def _factor(q: float) -> str:
    return f"×/÷ {np.exp(q):.2f}"


def _bin_ranges(edges: list[float]) -> list[str]:
    """Predicted value relative to the league median for each bin, e.g. '0.59–1.13 × median'."""
    bounds = [None, *np.exp(edges).tolist(), None]
    out = []
    for lo, hi in itertools.pairwise(bounds):
        if lo is None:
            out.append(f"below {hi:.2f} × median")
        elif hi is None:
            out.append(f"above {lo:.2f} × median")
        else:
            out.append(f"{lo:.2f}–{hi:.2f} × median")
    return out


def make_figures(p4: dict[str, Any], conformal: dict[str, Any]) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    lv = str(p4["display_level"])

    # Global TreeSHAP importance on the serving population
    glob = pd.DataFrame(p4["shap"]["serving"]["global"]).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([_label(d) for d in glob["driver"]], 100 * glob["typical_effect_pct"], color=ACCENT)
    ax.set_xlabel("Typical effect on the valuation, ±% (exp of mean |SHAP|)")
    ax.set_title("What moves the estimate (current Premier League players)", fontsize=11, color=INK, loc="left")
    _style(ax); ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_global_serving.png", dpi=150); plt.close(fig)
    paths["shap_global"] = "figures/shap_global_serving.png"

    # Worked example: top drivers as multiplicative effects, then the market adjustment
    ex = p4["shap_example"]
    shown = {d["feature"] for d in ex["top_drivers"]}
    other = sum(v for k, v in ex["phi"].items() if k not in shown)
    rows = [(_label(d["feature"]), d["phi"]) for d in ex["top_drivers"]] + [("All other features", other)]
    if ex["offset"]:
        rows.append(("Latest update-round adjustment", ex["offset"]))
    labels, phis = zip(*rows[::-1], strict=True)
    pct = 100 * np.expm1(np.asarray(phis))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(labels, pct, color=[ACCENT if v >= 0 else WARN for v in pct])
    ax.axvline(0, color=INK, linewidth=1)
    baseline = np.expm1(ex["base_value"] + np.log1p(ex["league_median_eur"]))
    served = np.expm1(ex["pred_ratio"] + ex["offset"] + np.log1p(ex["league_median_eur"]))
    ax.set_xlabel("Effect on the valuation (%)")
    ax.set_title(f"{ex['full_name']}, {ex['as_of_date']}\ntypical player {_eur(baseline)} → estimate {_eur(served)}",
                 fontsize=10, color=INK, loc="left")
    _style(ax); ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_example.png", dpi=150); plt.close(fig)
    paths["shap_example"] = "figures/shap_example.png"

    # Coverage per batch: backtest and test, chosen configuration vs the plain global interval
    chosen = p4["decision"]["config"]
    series = []
    for stage, configs in (("backtest", p4["backtest"]["configs"]), ("test", p4["final"]["all_configs_test_record_only"])):
        for name in ("global", chosen):
            for b in configs[name]["levels"][lv]["by_batch"]:
                if b["n"] >= 50:
                    series.append({"stage": stage, "config": name, "batch": b["group"].split(" | ")[-1], "cov": b["coverage"]})
    cov = pd.DataFrame(series)
    order = sorted(cov["batch"].unique())
    x = {b: i for i, b in enumerate(order)}
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for name, color, marker in (("global", MUTED, "o"), (chosen, ACCENT, "s")):
        sub = cov[cov["config"] == name]
        ax.scatter([x[b] for b in sub["batch"]], 100 * sub["cov"], color=color, marker=marker, s=22, label=name, zorder=3)
    ax.axhline(100 * float(lv), color=INK, linewidth=1, linestyle="--")
    test_start = min(x[b] for b in cov.loc[cov["stage"] == "test", "batch"])
    ax.axvline(test_start - 0.5, color=MUTED, linewidth=1)
    ax.text(test_start - 0.3, 0.97, "test →", color=INK, fontsize=8, va="top", transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(order)), [b[:7] for b in order], rotation=90, fontsize=7)
    ax.set_ylabel(f"Coverage of the {int(100 * float(lv))}% interval (%)")
    ax.set_title("Coverage per Transfermarkt update batch (batches with ≥ 50 rows)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "coverage_by_batch.png", dpi=150); plt.close(fig)
    paths["coverage"] = "figures/coverage_by_batch.png"
    return paths


def _coverage_config_table(configs: dict[str, Any], levels: list[float], chosen: str) -> list[str]:
    head = "| Configuration | " + " | ".join(f"Coverage {int(100 * lv)}%" for lv in levels)
    lines = [head + " | Worst bin gap (80%) | Median 80% interval | MdAPE |", "|---" * (len(levels) + 4) + "|"]
    for name, c in configs.items():
        lvd = c["levels"]
        mark = " ← chosen" if name == chosen else ""
        lines.append(
            f"| {name}{mark} | " + " | ".join(_pct(lvd[str(lv)]["coverage"]) for lv in levels)
            + f" | {100 * lvd['0.8']['worst_bin_gap']:.1f} pts | ×/÷ {lvd['0.8']['median_factor']:.2f} | "
            f"{_pct(c['point_metrics']['mdape'])} |"
        )
    return lines


def _group_table(
    title: str, records_by_level: dict[str, list[dict[str, Any]]], rename: dict[str, str] | None = None,
    order: list[str] | None = None,
) -> list[str]:
    levels = list(records_by_level)
    lines = [f"| {title} | n | " + " | ".join(f"Coverage {int(100 * float(lv))}%" for lv in levels) + " |",
             "|---|---|" + "---|" * len(levels)]
    first = records_by_level[levels[0]]
    rank = {g: k for k, g in enumerate(order or [])}
    for i in sorted(range(len(first)), key=lambda i: rank.get(first[i]["group"], len(rank))):
        r = first[i]
        name = (rename or {}).get(r["group"], r["group"])
        lines.append(f"| {name} | {r['n']} | " + " | ".join(_pct(records_by_level[lv][i]["coverage"]) for lv in levels) + " |")
    return lines


def write_phase4_report(p4: dict[str, Any], artifact_dir: Path) -> Path:
    conformal = json.loads((artifact_dir / "conformal.json").read_text(encoding="utf-8"))
    figs = make_figures(p4, conformal)
    bt, fin, dec = p4["backtest"], p4["final"], p4["decision"]
    levels = p4["levels"]
    lv = str(p4["display_level"])
    chosen = fin["chosen"]
    plain = fin["all_configs_test_record_only"]["global"]
    rec = bt["recentering_mdape_diff"]
    bt_plain, bt_rec = bt["configs"]["global"]["point_metrics"], bt["configs"]["global + recentering"]["point_metrics"]
    edges = conformal["bins"]["edges"]
    ranges = _bin_ranges(edges)
    bin_names = {str(i): f"Bin {i + 1}: {r}" for i, r in enumerate(ranges)}
    serving = conformal["recentering"]["serving"]
    ex = p4["shap_example"]
    ci = ConformalIntervals.from_dict({k: conformal[k] for k in ("levels", "quantiles", "n_calib")}
                                      | {"edges": edges, "recentered": conformal["recentering"]["enabled"]})
    ex_served = ex["pred_ratio"] + ex["offset"]
    ex_q = float(ci.halfwidth(np.array([ex["pred_ratio"]]), p4["display_level"])[0])
    ex_lm = np.log1p(ex["league_median_eur"])
    base_eur = np.expm1(ex["base_value"] + ex_lm)

    md = [
        f"# Phase 4 — Explanations and prediction intervals, model `{p4['model_version']}`",
        "",
        ("_Generated by `python -m src.ml.calibrate`. Every number below comes from "
         f"`artifacts/{p4['model_version']}/phase4_metrics.json` and `conformal.json`._"),
        "",
        "## Headline",
        "",
        (f"On the 2025/26 test season, scored once, the served {int(100 * float(lv))}% intervals contain the Transfermarkt "
         f"value for **{_cov(chosen['levels'][lv])}** of {fin['n_test']} valuations, and the 90% intervals for "
         f"{_cov(chosen['levels']['0.9'])}. A typical {int(100 * float(lv))}% interval is the estimate "
         f"**×/÷ {chosen['levels'][lv]['median_factor']:.2f}**. Recentering on the latest Transfermarkt update round lowers "
         f"test MdAPE from {_pct(plain['point_metrics']['mdape'])} to **{_pct(chosen['point_metrics']['mdape'])}**, and the "
         f"mean log bias from {plain['point_metrics']['mean_log_bias']:+.3f} to {chosen['point_metrics']['mean_log_bias']:+.3f}. "
         "Both choices (recentering and per-bin quantiles) were made on a backtest that never reads val_calib or test."),
        "",
        "## How an interval is built",
        "",
        ("1. The model predicts r̂, the log ratio of the player's value to the league median index M."),
        (f"2. **Recentering.** Transfermarkt revalues the league in update rounds. A batch is a run of valuation dates with "
         f"no gap longer than {conformal['recentering']['batch_gap_days']} days. The estimate is shifted by the median "
         "residual of the most recent batch that ended before the prediction's own batch began and has at least "
         f"{conformal['recentering']['min_batch_rows']} rows the model never trained on. Every row it reads is dated "
         "before the prediction."),
        (f"3. **Split-conformal interval.** On val_calib, the score is |actual log ratio − recentered prediction|. Rows are "
         f"split into {len(edges) + 1} bins by the model's predicted ratio (quartiles), and each bin keeps the "
         "⌈(n+1)·level⌉-th smallest score as its quantile q. The interval is r̂ + offset ± q, which in euros is the "
         "estimate ×/÷ e^q."),
        "",
        "## Decision protocol: backtest inside train and val_tune",
        "",
        ("For each evaluation season E, the tuned XGBoost is fit on every cycle before E−1, calibrated on E−1 and evaluated "
         "on E. That repeats the final layout (fit on train + val_tune, calibrate on val_calib, evaluate on test) one "
         "season earlier each time. The two rules below were fixed in `src/ml/calibrate.py` before any result was seen."),
        "",
        "| Evaluation season | Calibration season | Fit rows | Calibration rows | Evaluation rows |",
        "|---|---|---|---|---|",
        *[f"| {f['eval_cycle']} | {f['calib_cycle']} | {f['n_fit']:,} | {f['n_calib']:,} | {f['n_eval']:,} |" for f in bt["folds"]],
        "",
        "### Rule 1: recentering",
        "",
        ("Adopt if pooled backtest MdAPE drops with a 95% CI entirely below zero. The bootstrap resamples whole batches, "
         "because every row in a batch shares one offset."),
        "",
        f"- Pooled backtest MdAPE: {_pct(bt_plain['mdape'])} without, {_pct(bt_rec['mdape'])} with recentering.",
        (f"- Difference: {100 * rec['diff']:+.1f} pts (95% CI {100 * rec['ci_low']:+.1f} to {100 * rec['ci_high']:+.1f}, "
         f"{rec['n_batches']} batches)."),
        (f"- Mean |batch bias| over batches with ≥ 20 rows: {bt['mean_abs_batch_bias']['before']:.3f} before, "
         f"{bt['mean_abs_batch_bias']['after']:.3f} after."),
        f"- **Decision: {'adopt' if dec['recentered'] else 'reject'}** ({dec['reason_recentering']}).",
        "",
        "### Rule 2: one global quantile or one per predicted-value bin (Mondrian)",
        "",
        (f"Use per-bin quantiles if the global interval's worst bin is more than 5 points from nominal coverage at "
         f"{int(100 * float(lv))}% and the per-bin version's worst gap is smaller."),
        "",
        *_coverage_config_table(bt["configs"], levels, dec["config"]),
        "",
        f"Coverage of the {int(100 * float(lv))}% interval per predicted-value bin (backtest):",
        "",
        "| Configuration | " + " | ".join(f"Bin {i + 1}" for i in range(len(edges) + 1)) + " |",
        "|---|" + "---|" * (len(edges) + 1),
        *[f"| {name} | " + " | ".join(_pct(b["coverage"]) for b in c["levels"][lv]["by_bin"]) + " |"
          for name, c in bt["configs"].items()],
        "",
        f"- **Decision: {'per-bin (Mondrian)' if dec['mondrian'] else 'global'}** ({dec['reason_mondrian']}).",
        f"- Configuration served: **{dec['config']}**.",
        "",
        "## Final calibration on val_calib",
        "",
        (f"The saved model (trained on train + val_tune) reproduces the Phase 3 predictions (max relative difference "
         f"{fin['reproduced_phase3_predictions_max_rel_diff']:.1e}). Calibrated on {fin['n_calib']:,} val_calib rows:"),
        "",
        "| Bin (model's estimate vs league median) | Calibration rows | " + " | ".join(f"{int(100 * x)}% interval" for x in levels) + " |",
        "|---|---|" + "---|" * len(levels),
        *[f"| {ranges[i]} | {conformal['n_calib'][i]} | " + " | ".join(_factor(conformal["quantiles"][str(x)][i]) for x in levels) + " |"
          for i in range(len(ranges))],
        "",
        "## Test evaluation (2025/26, scored once)",
        "",
        "### Coverage",
        "",
        *[f"- {int(100 * float(k))}% interval: {_cov(v)}, median width ×/÷ {v['median_factor']:.2f}."
          for k, v in chosen["levels"].items()],
        "",
        *_group_table("Update batch", {k: v["by_batch"] for k, v in chosen["levels"].items()}),
        "",
        *_group_table("Predicted-value bin", {k: v["by_bin"] for k, v in chosen["levels"].items()}, bin_names),
        "",
        *_group_table("Position", {k: v["by_position"] for k, v in chosen["levels"].items()}),
        "",
        ("By actual value (for information only: no method can guarantee coverage conditional on the true value, because "
         "it isn't known when the interval is made):"),
        "",
        *_group_table("Actual value", {k: v["by_actual_value"] for k, v in chosen["levels"].items()},
                      order=[b[0] for b in VALUE_BUCKETS]),
        "",
        f"![Coverage by batch]({figs['coverage']})",
        "",
        "### Point accuracy with and without recentering",
        "",
        _metrics_table({"XGBoost v1.0.0 (Phase 3)": plain["point_metrics"], "XGBoost v1.0.0 + recentering (served)":
                        chosen["point_metrics"]}),
        "",
        "| Batch | n | Offset | Mean log bias before | Mean log bias after |",
        "|---|---|---|---|---|",
        *[f"| {b['batch']} | {b['n']} | {b['offset']:+.3f} | {b['bias_before']:+.3f} | {b['bias_after']:+.3f} |"
          for b in fin["test_batch_bias"]],
        "",
        ("Test has only four batches, so the evidence for recentering is the backtest CI above, not this table. Every "
         "configuration was scored on test after the decision, for the record:"),
        "",
        *_coverage_config_table(fin["all_configs_test_record_only"], levels, dec["config"]),
        "",
        "## Serving configuration (`conformal.json`, stored in `model_versions.conformal_quantiles`)",
        "",
        (f"- Serving rows are dated {serving['as_of_date']}. Their offset is {serving['offset']:+.3f} "
         f"(×{np.exp(serving['offset']):.3f}), the median residual of the {serving['source_batch']['start']} batch "
         f"({serving['source_batch']['n_rows']} rows)." if serving["source_batch"] else "- No recentering offset."),
        (f"- Parity: all {p4['serving']['parity_with_test_rows']['n_matched']} of {p4['serving']['n']} serving players match "
         "their test row on the same date (max prediction difference "
         f"{p4['serving']['parity_with_test_rows']['max_abs_pred_diff']:.1e}, same offset: "
         f"{p4['serving']['parity_with_test_rows']['offsets_equal']})."),
        f"- Median {int(100 * float(lv))}% interval for serving players: ×/÷ {p4['serving']['median_interval_factor']:.2f}.",
        "- The offset depends on the data: rerun `make calibrate` after every data refresh.",
        "",
        "## TreeSHAP explanations",
        "",
        ("XGBoost's built-in exact TreeSHAP (`pred_contribs=True`) explains each prediction. The values add up to the "
         f"model output: max additivity error {p4['shap']['test']['max_additivity_error']:.1e} on test and "
         f"{p4['shap']['serving']['max_additivity_error']:.1e} on serving rows."
         + (f" `shap.TreeExplainer` gives the same values (max difference {p4['shap']['shap_library_max_abs_diff']:.1e} "
            "on 200 rows)." if p4["shap"].get("shap_library_max_abs_diff") is not None else "")),
        "",
        ("**Reading them.** SHAP values are in log-ratio units and add up: r̂ = φ₀ + Σφᵢ. In euros that becomes a product, "
         "(1 + value) = (1 + M) · e^φ₀ · Π e^φᵢ · e^offset, so each driver multiplies the valuation by e^φᵢ and is shown as "
         "e^φᵢ − 1 (for example +18%). The starting point (1 + M) · e^φ₀ − 1 is a typical player in the training "
         "population at today's league level. Age and age² are summed into one driver."),
        "",
        "### What moves the estimate (mean |SHAP| over current players)",
        "",
        "| Driver | Typical effect |",
        "|---|---|",
        *[f"| {_label(r['driver'])} | ±{_pct(r['typical_effect_pct'])} |" for r in p4["shap"]["serving"]["global"][:12]],
        "",
        f"![Global SHAP]({figs['shap_global']})",
        "",
        "### Worked example",
        "",
        (f"{ex['full_name']} on {ex['as_of_date']} (the current player whose estimate is closest to the median). League "
         f"median index {_eur(ex['league_median_eur'])}; a typical player is worth {_eur(base_eur)}."),
        "",
        "| Driver | Value | Effect |",
        "|---|---|---|",
        *[f"| {_label(d['feature'])} | {d['value'] if not isinstance(d['value'], float) else round(d['value'], 3)} | "
          f"{100 * d['effect_pct']:+.1f}% |" for d in ex["top_drivers"]],
        f"| Latest update-round adjustment | — | {100 * np.expm1(ex['offset']):+.1f}% |",
        "",
        (f"Estimate: **{_eur(np.expm1(ex_served + ex_lm))}**, {int(100 * float(lv))}% interval "
         f"{_eur(np.expm1(ex_served - ex_q + ex_lm))}–{_eur(np.expm1(ex_served + ex_q + ex_lm))}."),
        "",
        f"![Worked example]({figs['shap_example']})",
    ]

    interp = REPORTS_DIR / "phase4_interpretation.md"
    if interp.exists():
        md += ["", interp.read_text(encoding="utf-8").strip()]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "phase4_explain_intervals.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def write_model_card(artifact_dir: Path) -> Path:
    meta = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    m3 = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    p4 = json.loads((artifact_dir / "phase4_metrics.json").read_text(encoding="utf-8"))
    conformal = json.loads((artifact_dir / "conformal.json").read_text(encoding="utf-8"))
    version = meta["model_version"]
    chosen = p4["final"]["chosen"]
    plain = p4["final"]["all_configs_test_record_only"]["global"]
    lv = str(p4["display_level"])
    best = m3["best_baseline"]
    best_test = m3["stage_b"][best]["test"]
    slices = m3["slices_test"]
    worst_age = max((r for r in slices["age_band"] if r["n"] >= 50), key=lambda r: r["mdape"])
    worst_value = max((r for r in slices["value_bucket"] if r["n"] >= 50), key=lambda r: r["mdape"])
    rows = m3["population_rows"]
    top = [_label(r["driver"]) for r in p4["shap"]["serving"]["global"][:5]]

    md = [
        f"# Model card — fair-football-fee `{version}`",
        "",
        ("_Generated by `python -m src.ml.calibrate` from the JSON files in "
        f"`artifacts/{version}/`. Hand-written text states facts that don't change between runs._"),
        "",
        "## Model details",
        "",
        (f"- **Type:** XGBoost gradient-boosted trees (`{meta['model_type']}`), {meta['params']['n_estimators']} trees, "
        f"max depth {meta['params']['max_depth']}, seed {meta['seed']}."),
        f"- **Trained:** {meta['trained_at'][:10]} on data version `{meta['data_version']}` "
        f"(git `{meta['git_commit'][:7]}`). Libraries: " + ", ".join(f"{k} {v}" for k, v in meta["libraries"].items()) + ".",
        ("- **Output:** an estimate of the player's Transfermarkt market value at date t, with an "
        f"{int(100 * float(lv))}% and a 90% prediction interval and per-prediction TreeSHAP drivers."),
        "- **Owner:** Gunnar (`gunnarwrld/fair-football-fee`), portfolio project.",
        "",
        "## Intended use",
        "",
        ("- Explore what a Premier League player's recent playing record, output, age and fitness imply about their "
        "crowd-sourced market value, and how the estimate would change in a what-if scenario."),
        ("- A teaching and portfolio artifact for point-in-time feature engineering, honest evaluation and calibrated "
        "uncertainty."),
        "",
        "## Out of scope",
        "",
        ("- **Transfer fees.** The target is the Transfermarkt crowd valuation, not a negotiated fee. Fees depend on release "
        "clauses, add-ons, contract length and negotiating leverage, none of which the model sees."),
        "- Players outside the Premier League, or with fewer than 450 tracked minutes in the last 12 months.",
        ("- Contract decisions. The main model has no contract feature (contract dates can't be reconstructed point-in-time "
        "from open data), so it can't price an expiring contract."),
        "- Any decision about a real person's career or pay.",
        "",
        "## Data",
        "",
        ("- dcaribou/transfermarkt-datasets (CC0): valuations, appearances, games, transfers. salimt/football-datasets "
        "(no license): injuries and profiles. Both pinned by commit and SHA-256 in `data/raw/MANIFEST.json`."),
        "- salimt-derived data is never redistributed, and raw injury records are never exposed by the API.",
        ("- Every feature uses source rows dated strictly before t; ongoing injuries are clipped at t. Verified by truncation "
        "invariance and train/serve parity tests."),
        (f"- Population: Premier League snapshots with ≥ 450 tracked minutes in the prior 365 days and a club confirmed by "
        f"match data. Rows: train {rows['train']:,} · val_tune {rows['val_tune']:,} · val_calib {rows['val_calib']:,} · "
        f"test {rows['test']:,}."),
        "",
        "## Target",
        "",
        ("`ln((value + 1) / (M + 1))`, where M is the trailing-365-day median value of the same population (point-in-time). "
        "Predictions are converted back with the same M, so the model learns value relative to the league and doesn't have "
        "to extrapolate inflation."),
        "",
        "## Evaluation protocol",
        "",
        "- Splits by whole July–June cycles: train < 2023-07-01 ≤ val_tune < 2024-07-01 ≤ val_calib < 2025-07-01 ≤ test.",
        ("- Hyperparameters: Optuna with expanding-window CV inside train. Model choice on val_tune. Final model fit on "
        "train + val_tune. Intervals calibrated on val_calib. Test scored once per model version."),
        "- Recentering and the interval method were chosen on a rolling backtest inside train and val_tune.",
        "",
        f"## Performance on test (2025/26, {chosen['n']} valuations)",
        "",
        _metrics_table({f"{version} + recentering (served)": chosen["point_metrics"],
                        f"{version} without recentering": plain["point_metrics"],
                        f"Best baseline ({best})": best_test}),
        "",
        (f"- {int(100 * float(lv))}% interval coverage: {_cov(chosen['levels'][lv])}; median width ×/÷ "
        f"{chosen['levels'][lv]['median_factor']:.2f}."),
        f"- 90% interval coverage: {_cov(chosen['levels']['0.9'])}; median width ×/÷ {chosen['levels']['0.9']['median_factor']:.2f}.",
        ("- XGBoost and Ridge are statistically tied on point accuracy (Phase 3). XGBoost ships for native missing-value "
        "handling, interactions and exact TreeSHAP."),
        "",
        "## Explanations",
        "",
        (f"Exact TreeSHAP per prediction, shown as multiplicative effects vs a typical player. Largest drivers across current "
        f"players: {'; '.join(top)}. See `reports/phase4_explain_intervals.md`."),
        "",
        "## Known limitations",
        "",
        (f"- **Older players** are over-valued: the {worst_age['slice']} age band has MdAPE {_pct(worst_age['mdape'])} and mean "
        f"log bias {worst_age['mean_log_bias']:+.3f} on test (Phase 3, before recentering). Transfermarkt depreciates age "
        "faster than output declines."),
        (f"- **Low-value players**: the {worst_value['slice']} bucket (by actual value) has MdAPE {_pct(worst_value['mdape'])} "
        "(Phase 3). The lowest predicted-value bin gets a wider interval for this reason."),
        ("- **Update-round shocks.** Each Transfermarkt revaluation round can move the whole league. Recentering uses the "
        "latest completed round, so a sudden new shift is only absorbed one round later."),
        (f"- **Coverage under drift.** Test coverage of the {int(100 * float(lv))}% interval is "
        f"{_cov(chosen['levels'][lv])}. Conformal guarantees assume calibration and test rows are exchangeable, and later "
        "seasons drift, so coverage is checked per batch and per bin in the Phase 4 report. Players who end up valued "
        f"under €5M are covered least ({_pct(next(r['coverage'] for r in chosen['levels'][lv]['by_actual_value'] if r['group'] == '<5M'))})."),
        ("- The recentering offset is computed from the latest Transfermarkt update round in the data, so it must be "
        "recomputed (`make calibrate`) whenever the data is refreshed; registration refuses a stale offset."),
        ("- Club data in older seasons is noisy; rows whose club can't be confirmed by match data are excluded (about 8% of "
        "train)."),
        "",
        "## Ethical considerations",
        "",
        ("- The players are real people. Estimates describe a crowd valuation, not a player's worth or ability, and the API "
        "and UI must label them that way."),
        "- Injury history is used only as aggregated features; individual injury records are never shown or returned.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "make pipeline   # ingest → clean → features",
        "make train      # Phase 3 model and report (reuses tuning.json)",
        "make calibrate  # Phase 4 intervals, SHAP, this card",
        "make register   # write model_versions and mark the model active",
        "```",
        "",
        f"Serving config: `artifacts/{version}/conformal.json` ({conformal['config']}, calibrated on {conformal['calibration_split']}).",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"model_card_{version}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    artifact_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/v1.0.0")
    p4_results = json.loads((artifact_dir / "phase4_metrics.json").read_text(encoding="utf-8"))
    print(write_phase4_report(p4_results, artifact_dir))
    print(write_model_card(artifact_dir))
