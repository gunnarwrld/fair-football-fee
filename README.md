# Fair Football Fee — Player Market Value Predictor

A full-stack machine learning system that estimates a football player's crowd-sourced market valuation (`market_value_eur`) at a given date $t$ using strictly point-in-time performance statistics, injury burden, and club/league context available prior to $t$.

> **Important Target Limitation**: The target variable (`market_value_eur`) is a **Transfermarkt crowd-sourced market valuation**, **not** an actual club-to-club negotiated transfer fee. Real transfer fees are heavily influenced by release clauses, buyer/seller financial leverage, amortization accounting, add-on clauses, and transfer-window urgency.

## Stack
- **Data & ML**: Python 3.12, DuckDB, PyArrow (Parquet Lakehouse), Pandas, Scikit-Learn, XGBoost, LightGBM, Optuna, TreeSHAP, Split Conformal Prediction
- **Backend & Caching**: FastAPI, SQLAlchemy 2.0 + Alembic, PostgreSQL 16 (6 operational serving tables), Redis 7 (cache-aside with stampede lock and graceful fallback)
- **Frontend**: Next.js (App Router)
