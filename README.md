# Fair Football Fee — Player Market Value Predictor

A machine learning project that estimates a football player's crowd-sourced market valuation (`market_value_eur`) at a given date $t$ using strictly point-in-time performance statistics, injury burden, and club/league context available prior to $t$.

> **Important Target Limitation**: The target variable (`market_value_eur`) is a **Transfermarkt crowd-sourced market valuation**, **not** an actual club-to-club negotiated transfer fee. Real transfer fees are heavily influenced by release clauses, buyer/seller financial leverage, amortization accounting, add-on clauses, and transfer-window urgency.

> **Status: work in progress.** Merged on `main`: the data audit, raw ingestion to Parquet, cleaning, and point-in-time feature engineering (phases 0–2), with tests for leakage and each phase. Still in open pull requests, not on `main`: the modelling population (#2), tuned models and evaluation (#3), explanations and conformal intervals (#5), the FastAPI service (#6), the Next.js frontend (#7), and Docker/CI/load testing (#8). No model results are published yet.

## Stack

Merged on `main`:
- **Data**: Python 3.11+, DuckDB, PyArrow (Parquet), Pandas
- **Storage**: PostgreSQL 16 via SQLAlchemy 2.0 + Alembic (serving schema), Redis 7 (in `docker-compose.yml`)

Planned in open pull requests:
- **ML**: Scikit-Learn, XGBoost, LightGBM, Optuna, TreeSHAP, split conformal prediction
- **API and UI**: FastAPI with Redis cache-aside, Next.js (App Router)

## Data and licensing

Raw data is downloaded at run time and is **not** committed. Sources, pinned versions, SHA-256 hashes and license notes are in [`docs/data_audit_report.md`](docs/data_audit_report.md):

- [`dcaribou/transfermarkt-datasets`](https://github.com/dcaribou/transfermarkt-datasets) — CC0-1.0.
- [`salimt/football-datasets`](https://github.com/salimt/football-datasets) — no license file, so all rights are reserved by default. Raw and derived files stay gitignored, and raw injury records are never exposed.

The plan for phases and data is in [`docs/phase0_architecture_and_data_plan.md`](docs/phase0_architecture_and_data_plan.md).

## Run the merged pipeline

You need Python 3.11 or newer and Docker.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

docker compose up -d                # PostgreSQL 16 and Redis 7 on localhost
alembic upgrade head                # serving tables

python scripts/audit_datasets.py    # downloads missing pinned sources, checks SHA-256, writes the audit report
python -m src.data.ingest           # raw CSV -> typed Parquet in data/raw/ + MANIFEST.json
python -m src.data.clean            # add --load-db to upsert clean tables into PostgreSQL
python -m src.features.pipeline     # point-in-time features; add --load-db to fill player_current_features
```

Settings come from environment variables or a `.env` file (see `src/config.py`). The defaults match `docker-compose.yml`.

## Tests

```bash
pytest
```

The suite covers the phase 0 schema, phase 1 ingestion and cleaning, phase 2 features, and checks that no feature uses data from after the valuation date. Without downloaded data, 17 tests pass and 7 that need the data files are skipped.

## License

Code: MIT — see [LICENSE](LICENSE). Data keeps its own licenses (see above).
