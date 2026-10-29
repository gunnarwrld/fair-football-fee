# Project Log: Decisions and Why

A running record of design decisions, review findings and rejected alternatives, newest at the bottom. Each entry gives the reason in plain language so it can be defended. `CLAUDE.md` holds the current rules; this file explains how they came about.

## Origin (2026-10-26)
Brief: estimate fair market value from historical stats, injuries and contract length (Premier League and/or Liga Pro Ecuador), with XGBoost served through a REST API with Redis caching.

## Phase 0: Architecture
- **Target = Transfermarkt market value, not transfer fees.** It's the only open, consistent target. Fees depend on release clauses, add-ons and leverage. Disclosed in the README and model card.
- **FastAPI over Express + ONNX.** ONNX graphs export predictions only, not TreeSHAP attributions. Serving in Python also shares the exact feature code with training, so there is no train/serve skew. Rejected: Node + ONNX (permutation SHAP would be slow, and the features would have to be re-implemented in TypeScript).
- **Parquet lakehouse + 6 Postgres serving tables.** Raw, clean and features live as Parquet (fast, reproducible). Postgres holds only what the API serves. Rejected: a 4-schema, 15-table Postgres ETL (too much boilerplate for a portfolio project).
- **`player_current_features` table.** The API reads one precomputed feature row and applies what-if overrides on top, so nothing is recomputed per request.
- **Alembic is the only schema source.** A hand-written SQL file alongside it would drift. `alembic check` is part of verification.
- **`player_seasons` unique key** `(player_id, season, tm_club_id)`, because mid-season transfers within the same league would collide on a league-based key.
- **Ports bound to 127.0.0.1**, a partial unique index guarantees a single active model, and `data_version` comes from the ingestion manifest rather than config.
- **Main model is contract-free.** The only contract-expiry fields are current snapshots, so using them for historical valuations leaks later renewals, and renewals correlate with quality.
- **FBref advanced stats unavailable.** Opta terminated FBref's feed in early 2026 and the xG/progressive metrics were removed, so the core features use Transfermarkt-derived data only.

## Phase 1: Audit, ingestion, cleaning
- **Sources:** dcaribou/transfermarkt-datasets (CC0; valuations, appearances, games, transfers) and salimt/football-datasets (no license; injuries and profiles). Both are pinned by SHA-256 and commit SHA, and their hashes are recorded in `MANIFEST.json`.
- **Point-in-time league attribution** comes from games per club-season, because the clubs table stores a club's *current* league. Promotion and relegation would otherwise leak. This fix caught 8,332 mislabeled "GB1" rows.
- **Contract validity check:** the per-row test uses salimt's own snapshot consistently. Mixing two snapshots (dcaribou expiry from June 2026 with salimt extensions from Sept 2025) hid renewals. Only about 1,600 rows are valid, almost none in test, so the contract ablation is expected to be inconclusive. Show contract length in the UI as read-only; don't promise a slider.
- **EC1 identified as a survivorship sample:** Ecuador-stage valuations of players who later reached tracked clubs.

## Phase 2: Point-in-time features
- One feature code path (`engineer_features_for_snapshots`) serves both training snapshots and serving rows.
- Empirical-Bayes per-90 shrinkage (m0 = 900 minutes) with priors fitted on train only. Injuries are clipped at `t`. A lagged value is used only by the value-update model.
- Tests: synthetic edge cases at `== t` / `> t`; truncation invariance on real snapshots (delete everything dated ≥ t, recompute, require bit-for-bit equality); train/serve parity.
- Target normalized by the trailing league median, because trees can't extrapolate a year or level trend into the test period.

## Phase 2.1: Era-stable population (2026-10-29)
Review of the Phase 2 data found that the dataset's population changes right at the train/val boundary. Full evidence is in `docs/phase2_1_population_findings.md`. In short:
- Older seasons contain fringe players who are in the source only because they appear later; recent seasons haven't had time to collect them. The median value jumped from €7M to €15M in one season while the same players' values were flat, so the old league index tracked population changes, not inflation (val_tune +0.13 vs test −0.28 mean log ratio).
- **Decision:** GB1 rows with ≥ 450 tracked minutes in the prior 365 days, plus club attribution confirmed by match data. The league index is computed on that population. Era-dependent features are replaced (appearance-based tenure capped at 730 days, capped injury recency, injury features NULL outside the injury source). Constant columns are no longer features.
- **Liga Pro Ecuador dropped** (owner's decision): 27% of rows have season stats, under 1% have match data, about 180 rows would survive the population rule, single digits in test.
- Result: a smooth index from €4.8M (2013) to €23.5M (test); mean log ratio between −0.07 and −0.13 across val/test; null-rate gap train vs test ≤ 1.5 points for every feature. Population: train 8,401 / val_tune 1,118 / val_calib 1,101 / test 1,288. Serving: 439 players.
- Follow-up: derive club-at-valuation from appearances in Phase 1 instead of excluding unverified rows.

## Phase 3: Baselines, XGBoost, evaluation (2026-10-29)
- **Protocol:** Stage A fits every model on train and compares on val_tune. Stage B refits on train + val_tune and scores test once. val_calib is reserved for Phase 4. Optuna (80 trials) uses expanding-window folds inside train only. Results: `reports/phase3_evaluation.md`, generated entirely from `artifacts/v1.0.0/metrics.json`.
- **Model choice:** tuned XGBoost and Ridge are statistically tied (player-level block bootstrap CIs include 0 on val_tune and test). XGBoost ships for native missing values, interactions and exact TreeSHAP; Ridge stays as the interpretable reference. Saying "tied" honestly is stronger in a defense than claiming a win that isn't significant.
- **Bug found and fixed:** the value-update model used `prev_log_market_value` (absolute) against a ratio target. Trees can't extrapolate, so in an inflated season ordinary players' previous values looked like past stars', giving a worse model than carry-forward. Replaced with `prev_log_ratio_vs_league_median` (same scale as the target); it now beats carry-forward with a CI clear of zero.
- **Out-of-time bias:** predictions run systematically high on later seasons, and the shift differs per Transfermarkt update batch (a trailing median lags market-wide corrections). Two point-in-time candidates (`days_into_season`, `index_growth_365d`) were **rejected by train-only CV**. The test split looked better with the season-phase feature; it was still rejected, because choosing on test would be snooping. Both comparisons are recomputed by `train.py` for the record. Carried to Phase 4: recentering by the latest batch residual, and per-batch conformal coverage.
- **Contract ablation:** with/without difference is inside its CI on the ~200 test rows with a valid contract, so contract length is read-only in the UI.
- **Ops:** XGBoost CPU wheel (`xgboost-cpu`) in the Linux VM to avoid a 291 MB GPU dependency. Optuna storage is SQLite with resume support (`--optuna-db`, `--tune-timeout`). `make train` reuses the committed `tuning.json` so results are reproducible.

## Ops
- Host Postgres port changed to 5433 (2026-10-29). The owner's Mac already runs a native Postgres on 5432 without the `fff_user` role, so `--load-db` connected to the wrong server.
- From 2026-10-29 Claude develops the project directly (owner's instruction), keeping `CLAUDE.md`, this log and the claude.ai Project doc current so no context is lost.
