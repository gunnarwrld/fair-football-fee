# Fair Football Fee — Player Market Value Predictor

A full-stack machine learning system that estimates a football player's crowd-sourced market valuation (`market_value_eur`) at a given date $t$ using strictly point-in-time performance statistics, injury burden, and club/league context available prior to $t$.

> **Important Target Limitation**: The target variable (`market_value_eur`) is a **Transfermarkt crowd-sourced market valuation**, **not** an actual club-to-club negotiated transfer fee. Real transfer fees are heavily influenced by release clauses, buyer/seller financial leverage, amortization accounting, add-on clauses, and transfer-window urgency.

## Scope

The model covers **Premier League players with at least 450 tracked minutes in the prior 12 months**. That restriction removes a 2023 break in which players the open source contains, and keeps training, evaluation and serving on the same population. Liga Pro Ecuador was evaluated and dropped: the open data has almost no Ecuadorian match statistics, and its rows are a survivorship sample. See [`docs/phase2_1_population_findings.md`](docs/phase2_1_population_findings.md) for the evidence.

## Stack
- **Data & ML**: Python 3.12, DuckDB, PyArrow (Parquet Lakehouse), Pandas, Scikit-Learn, XGBoost, LightGBM, Optuna, TreeSHAP, Split Conformal Prediction
- **Backend & Caching**: FastAPI, SQLAlchemy 2.0 + Alembic, PostgreSQL 16 (6 operational serving tables), Redis 7 (cache-aside with stampede lock and graceful fallback)
- **Frontend**: Next.js (App Router)

## Reports
- [Phase 3 evaluation](reports/phase3_evaluation.md): baselines, tuned XGBoost, error slices, secondary experiments.
- [Phase 4 explanations and intervals](reports/phase4_explain_intervals.md): TreeSHAP drivers, update-round recentering, conformal prediction intervals and their coverage.
- [Model card for v1.0.0](reports/model_card_v1.0.0.md): intended use, data, performance, limitations.
- Decision history and the reasons behind each choice: [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md).
