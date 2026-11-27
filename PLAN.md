# Plan: fair-football-fee
Goal: Evolve the Premier League Transfermarkt market-value predictor (XGBoost, FastAPI/Redis, Next.js) from v1.0.0 to v1.1.0 and hosted readiness.     Milestone: v1.1.0 & Hosted Readiness, 2026-12-18

## Tasks
| id | task | type | size | state |
|----|------|------|------|-------|
| T01 | Write PLAN.md + STATUS.md from README and docs/PROJECT_LOG.md | docs | S | done |
| T02 | Evaluate v1.1 age×minutes interaction and post-peak monotone constraint on train-only CV to address 29+ over-valuation | research | M | todo |
| T03 | Re-derive club-at-valuation from appearances in Phase 1 cleaning to recover ~8% excluded training rows | refactor | M | todo |
| T04 | Add regression and truncation-invariance tests for appearance-derived club attribution | tests | S | todo |
| T05 | Calibrate, evaluate once on test, and update model card and README metrics if v1.1.0 beats v1.0.0 in CV | feature | M | todo |
| T06 | Prepare hosted deployment config and runbook (managed Postgres/Redis, read-only artifacts volume, FORWARDED_ALLOW_IPS) | ci | M | todo |
| T07 | Harden hosted config (configurable CORS, proxy rate-limit verification) and check Starlette/httpx TestClient deprecation | security | S | todo |

## Done when
- `PLAN.md` and `STATUS.md` reflect the completed Phases 0–7 baseline and prioritized post-v1.0.0 follow-ups from `README.md`, `docs/PROJECT_LOG.md`, and `CLAUDE.md`.
- Any v1.1 feature or data change is decided strictly by train-only CV / `val_tune` and keeps `tests/test_leakage.py` (truncation invariance and train/serve parity) green.
- If v1.1.0 is promoted, `make calibrate`, `make register`, reports, model card, and README headline numbers are regenerated from code output.
- Hosted deployment configuration is documented and all offline/CI checks (`ruff`, `pytest`, `alembic check`, `make api-types`, `make web-check`, Docker builds) pass.
