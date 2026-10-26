# Phase 0: System Architecture & Data Engineering Plan

## 1. Scope & Target Variable Disclosure

`fair-football-fee` predicts a football player's crowd-sourced market valuation (`market_value_eur`) at a valuation date $t$ using performance statistics, injury burden, and club/league context strictly prior to $t$ ($t_{\text{data}} < t$).

> **Target Limitation**: The target variable is Transfermarkt's crowd-sourced market value estimate, **not** an actual club-to-club transfer fee. Real transfer fees depend on release clauses, accounting amortization, buyer/seller urgency, and add-on structures.

---

## 2. Serving Architecture: FastAPI vs. Express + ONNX Runtime

| Criterion | FastAPI + Native XGBoost/TreeSHAP (Selected) | Node.js/Express + ONNX (`onnxruntime-node`) |
| :--- | :--- | :--- |
| **Point Inference** | Native `xgboost.Booster` evaluation in ~1–3 ms. | Fast (~1 ms) via `onnxruntime-node` after `onnxmltools` graph export. |
| **Per-Request SHAP (`top-5 drivers`)** | Exact C++ `shap.TreeExplainer` runs in ~2–5 ms per request (including dynamic `POST /v1/valuations` what-if payloads). | Standard ONNX model exports do not emit TreeSHAP attributions; computing SHAP in Node.js requires slow permutation sampling or custom native bindings. |
| **Training-Serving Parity** | 100% shared Python feature transformation, Bayesian shrinkage, and target un-normalization code. | Requires re-implementing feature logic in TypeScript, creating training-serving skew risk. |
| **Concurrency Model** | Endpoints using synchronous SQLAlchemy `Session` (`get_db`) are declared as plain `def` routes so FastAPI executes them in its worker threadpool without blocking the asyncio event loop. | Native async I/O. |

---

## 3. Storage Layer: Parquet Lakehouse + 6-Table PostgreSQL Serving Schema (Alembic Only)

1. **Analytical Lakehouse (`data/raw/`, `data/clean/`, `data/features/`)**: Stored as versioned **Parquet** files (`pyarrow` + `duckdb`).
   - Phase 1 ingestion generates `data/raw/MANIFEST.json` containing the ingestion timestamp and SHA-256 digests of raw files to produce a deterministic `data_version` hash.
2. **Operational Serving Layer (PostgreSQL 16 via Alembic)**: Managed exclusively via SQLAlchemy (`src/db/models.py`) and Alembic (`migrations/versions/0001_initial_serving_schema.py`) as the single source of truth:
   - `players`: Master player profile (`tm_player_id` unique, `full_name`, `normalized_name` with `pg_trgm` GIN index `ix_players_normalized_name_trgm`, `position_group`, `current_club`, `current_league_code`).
   - `player_seasons`: Per-season/per-club aggregated stats keyed by `UNIQUE (player_id, season, tm_club_id)` so mid-season intra-league transfers never collide.
   - `valuations`: Historical valuation series keyed by `UNIQUE (player_id, valuation_date)`.
   - `player_current_features`: Precomputed serving feature vector (`player_id` PK, `as_of_date`, `data_version`, `league_code`, `trailing_league_median_eur`, `features` JSONB). Written by the Phase 2 feature pipeline using the same code path as training. At startup, the FastAPI service reads `data_version` from the database rather than relying on hardcoded environment defaults.
   - `model_versions`: Registered model metadata (`model_version` PK, `model_type`, `data_version`, `feature_list`, `metrics`, `conformal_quantiles`, `is_active`). Enforces at most one active model via partial unique index `ix_model_versions_single_active` (`WHERE is_active IS TRUE`). Each model's `metadata.json` also records exact runtime library versions (`xgboost`, `shap`, `numpy`, `scikit-learn`, `pandas`, `pyarrow`).
   - `predictions_log`: Audit log of API predictions (`request_id`, `model_version`, `data_version`, `player_id`, `is_what_if`, `predicted_value_eur`, `interval_low_eur`, `interval_high_eur`, `top_shap_drivers`, `cache_status`, `latency_ms`).
   - **Upsert Rule**: Because SQLAlchemy's `onupdate=func.now()` only fires on ORM unit-of-work updates, all bulk `INSERT ... ON CONFLICT DO UPDATE` statements in Phases 1 & 2 explicitly set `updated_at = sa.func.now()`.

---

## 4. Data Sources, DuckDB Audit & Canonical Table Assignment

1. **Post-Opta Feature Set**: Because Opta terminated Sports Reference's advanced data feed earlier this year (removing xG, xA, and progressive actions from FBref), the core model is built strictly on Transfermarkt-derived metrics: valuations, appearances, minutes, goals, assists, cards, clean sheets/goals conceded, injury history (days/games missed), age, height, preferred foot, position, club, and league.
2. **Phase 1 Task 1 (`scripts/audit_datasets.py` via DuckDB)**:
   - Inspects the actual `LICENSE` files in `dcaribou/transfermarkt-datasets` and `salimt/football-datasets`.
   - Executes fast DuckDB SQL queries over the source CSV/Parquet files to report row counts per league and season (`GB1` vs. `EC1`), non-null coverage for contracts/injuries/valuations, and unmatched ID rates.
   - Assigns **one canonical dataset per table** (e.g., one authoritative source for valuations, one for appearances, `salimt` for injuries) so duplicate or conflicting valuation records are never merged.
   - If `EC1` (Liga Pro Ecuador) is thin in the open dumps, `EC1` is evaluated as an out-of-distribution (OOD) generalization test slice rather than a co-headline training league.
3. **Entity Resolution**:
   - Both `dcaribou` and `salimt` use Transfermarkt numeric `player_id`s (Tier 1 exact ID join verified by `date_of_birth`).
   - Secondary fuzzy matching (`rapidfuzz.fuzz.token_set_ratio`) is deferred unless the Phase 1 audit shows an unmatched rate $>0\%$; if invoked, it enforces an **exact `date_of_birth` match** + minimum name length gate so subset tokens like `"Silva"` never cross-match.
4. **No Synthetic Training Data**:
   - Models are trained and evaluated exclusively on real open-dataset records. Synthetic fixtures (`Test Player Alpha`, `player_id = 999001`) are restricted to `pytest` unit tests.

---

## 5. Leakage Prevention, Inflation Normalization & Split Design

1. **Contract Expiry Leakage Elimination**:
   - Static `contract_expiration_date` in player profile tables reflects current renewals rather than historical contract state at past valuation dates, and setting historical rows to `NaN` creates an era proxy that tree models exploit.
   - **Solution**: Train the **Main Model (Full History)** with **no contract feature**. Alongside it, train a **Contract-Aware Variant** restricted to the **recent valuation window** (where static expiry is approximately contemporaneous). Phase 3 compares both on held-out data to decide which ships and whether the UI contract slider has genuine signal.
2. **Target Normalization for Valuation Inflation (with Sparse-League Fallback)**:
   - Decision trees cannot extrapolate `valuation_year` or `trailing_league_median` beyond the training range. Instead, we normalize the log target by the trailing 365-day league median valuation $M_{\text{league}, t^-}$ computed strictly from prior valuations $\in [t - 365\text{d}, t)$:
     $$z_{\text{norm}} = \log(1 + y) - \log(1 + M_{\text{league}, t^-})$$
   - **Minimum-Count Fallback for Sparse Leagues (`EC1`)**: If a league has fewer than $N_{\min} = 30$ valuations in $[t - 365\text{d}, t)$, $M_{\text{league}, t^-}$ falls back to a shrinkage-blended prior median across all historical valuations $< t$ for that league (or global non-top-5 prior median when cumulative league count $< 30$) to prevent noisy target normalization.
   - Predictions are transformed back at inference via $\hat{y} = \exp(\hat{z}_{\text{norm}} + \log(1 + M_{\text{league}, t^-})) - 1$.
3. **4-Way Full-Season Chronological Split + Unseen-Player Check**:
   - Primary split is strictly chronological with cutoffs aligned to **full 12-month valuation cycles** (so every split captures both winter and summer Transfermarkt batch updates):
     `train` $\rightarrow$ `val_tune` (Optuna tuning) $\rightarrow$ `val_calib` (split conformal calibration) $\rightarrow$ `test` (final evaluation).
   - Secondary check reports `test` metrics on the **unseen-player slice** (players in `test` with zero prior rows in `train`/`val_tune`/`val_calib`).
   - The secondary **Value-Update Model** (with `prev_market_value_eur`) is compared against a **Naive Last-Value Carry-Forward** baseline ($\hat{y}_t = y_{t-1}$).
4. **Explainability & Conformal Uncertainty**:
   - SHAP drivers on log-target predictions are reported as **multiplicative percentage effects** $(\exp(\phi_j) - 1) \times 100\%$.
   - 80% prediction intervals use **Predicted-Value Bucketed Split Conformal Prediction** calibrated on `val_calib` terciles of $\hat{z}$, reporting empirical coverage per season on `test`.
