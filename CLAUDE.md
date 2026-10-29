# CLAUDE.md — fair-football-fee

Claude is the lead developer on this project (owner: Gunnar, `gunnarwrld/fair-football-fee`). This file is the working contract for every session. The full decision history and its reasons are in `docs/PROJECT_LOG.md`; read it before changing anything in the data or ML layers.

## What this is
A portfolio regression system that estimates a Premier League player's **Transfermarkt crowd-sourced market value** (not an actual transfer fee) at a date `t`, using only data dated strictly before `t`. XGBoost model → FastAPI service with Redis cache-aside → Next.js UI. The owner is a backend-focused full-stack student and needs to be able to defend every design choice.

## Hard invariants (do not break without updating PROJECT_LOG.md)
1. **Point-in-time:** every feature uses source rows dated `< as_of_date`. Ongoing injuries are clipped at `t`. `tests/test_leakage.py` (truncation invariance and train/serve parity) must stay green.
2. **Modeling population (Phase 2.1):** train, evaluate and serve only rows with `in_model_population == True`: `GB1`, ≥ 450 tracked minutes in the prior 365 days, and a club confirmed by match data. Liga Pro Ecuador is out of scope. See `docs/phase2_1_population_findings.md`.
3. **Target:** `target_log_ratio_vs_league_median = ln((y+1)/(M+1))`, where `M` is the trailing-365-day league median **over the same population**. Predictions are un-normalized with the same `M`. SHAP values are log-space and are shown as multiplicative `exp(phi)-1` "vs league median".
4. **Main model has no contract feature and no lagged value.** `VALUE_UPDATE_MODEL_FEATURES` (lagged value; baseline = carry the last value forward) and `CONTRACT_ABLATION_FEATURES` (tiny recent window) are secondary experiments only.
5. **Splits** (`src/features/engineer.py`, whole July–June cycles): train < 2023-07-01 ≤ val_tune < 2024-07-01 ≤ val_calib < 2025-07-01 ≤ test. Tune on val_tune; conformal calibration on val_calib; touch test once per model version.
6. **Feature and model decisions come from train-only CV or val_tune, never from test.** If a test number tempts you, write it in PROJECT_LOG and still decide by CV (Phase 3 precedent: season-phase feature).
7. **Alembic is the only schema source.** No raw SQL schema files. `alembic check` must report no drift.
8. **`data_version`** comes from `data/raw/MANIFEST.json` (written by ingestion) and is never hardcoded. The active model comes from `model_versions.is_active`, not config.
9. **salimt data has no license:** never commit its raw or derived parquet, and never expose raw injury records through the API.
10. No synthetic data attached to real players. Test fixtures use obviously fake players.

## Layout
- `src/data/` — `ingest.py` (pinned downloads → `data/raw/*.parquet` + `MANIFEST.json`), `pit_sql.py` (shared point-in-time SQL), `clean.py` (→ `data/clean/`, `--load-db`)
- `src/features/` — `engineer.py` (single feature code path for training and serving), `pipeline.py` (snapshots, serving features, `--load-db`)
- `src/ml/` — `data.py` (population, back-transform), `metrics.py`, `models.py` (baselines, XGBoost), `tuning.py` (Optuna, expanding-window CV inside train), `experiments.py` (candidate features decided by CV), `train.py` (entrypoint), `report.py` (→ `reports/phase3_evaluation.md`)
- `artifacts/v1.0.0/` — `model.ubj` (gitignored), `features.json`, `metadata.json`, `metrics.json`, `tuning.json` (tracked)
- `src/db/` — SQLAlchemy models and session; `migrations/` — Alembic
- `scripts/audit_datasets.py` — DuckDB dataset/license audit → `docs/data_audit_report.md`
- `data/` and `artifacts/` binaries are gitignored; JSON manifests and reports are tracked.

## Commands (see `Makefile`)
- `make pipeline` ingest → clean → features · `make train` Phase 3 (reuses `tuning.json`) · `make train-retune` · `make report` · `make test` / `make test-unit` · `make lint`
- Remote VM: SQLite can't run on the mounted repo (no delete permission for journals), so pass `--optuna-db ~/optuna_v1.db --tune-timeout 150` and rerun until tuning completes.
- `make db` = docker postgres+redis → `alembic upgrade head` → load clean tables → load serving features. Host Postgres port is **5433** (5432 is taken on the owner's Mac by a native Postgres).
- Tests marked `@pytest.mark.data` need the local lakehouse and skip on a clean clone.

## Environment notes
- Owner's machine: macOS, Python 3.12 `.venv`, Docker Desktop (must be running for `make db`).
- Remote Claude sessions run in a Linux VM that mounts the repo. The macOS `.venv` does not run there, so use a separate env outside the repo (`uv venv -p 3.12 ~/fffenv`, then `uv pip install` the deps). Ruff is pinned to the 0.17.x behaviour the owner uses (`ruff check src tests migrations`).
- In that VM, deleting files and git lock cleanup need the user's delete permission. Pushing needs the owner's credentials: commit locally, then the owner pushes.

## Workflow
- One branch per phase (`feat/phase-N-...`). Each phase ends with: tests green, lint clean, docs updated, and a short entry in `docs/PROJECT_LOG.md`.
- Measure before claiming. Numbers in docs come from code output, never literals.
- Explain decisions in 2–3 plain sentences the owner could say in a defense.

## Status (update at the end of every phase)
- Done: Phase 0 (scaffold), Phase 1 (audit, ingest, clean), Phase 2 (point-in-time features + serving writer), Phase 2.1 (era-stable population), Phase 3 (baselines, tuned XGBoost `v1.0.0`, evaluation report).
- Phase 3 facts to carry forward (details in `reports/phase3_evaluation.md`): XGBoost and Ridge are statistically tied; XGBoost ships for TreeSHAP and missing-value handling. Out-of-time predictions are systematically too high and the shift varies by Transfermarkt update batch. The contract feature shows no detectable signal, so contract length is read-only in the UI. The value-update model beats carry-forward.
- Next: Phase 4 — TreeSHAP (multiplicative "vs league median"), split-conformal intervals calibrated on val_calib (check coverage per batch and per predicted-value bucket), evaluate recentering by the latest completed batch's median residual, model card, register in `model_versions`. Then 5 (FastAPI + Redis), 6 (Next.js), 7 (CI, load test, docs).
- Open follow-ups: re-derive club-at-valuation from appearances in Phase 1 (club data in old seasons is noisy; currently those rows are excluded, about 8% of train); `scripts/audit_datasets.py` has an unused `EC_COMP_IDS` import.
