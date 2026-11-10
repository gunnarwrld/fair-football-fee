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
| **Concurrency & Pool Sizing** | Endpoints using synchronous SQLAlchemy `Session` (`get_db`) are declared as plain `def` routes so FastAPI executes them in its worker threadpool without blocking the asyncio event loop. Background tasks (`predictions_log` writes) open their own `SessionLocal()` session. Pool sizing is `pool_size=5, max_overflow=10` (max 15 connections/worker $\times 4$ workers = 60 < Postgres default 100). | Native async I/O. |

---

## 3. Storage Layer: Parquet Lakehouse + 6-Table PostgreSQL Serving Schema (Alembic Only)

1. **Analytical Lakehouse (`data/raw/`, `data/clean/`, `data/features/`)**: Stored as versioned **Parquet** files (`pyarrow` + `duckdb`).
   - Phase 1 ingestion generates `data/raw/MANIFEST.json` containing the ingestion timestamp and SHA-256 digests of raw files to produce a deterministic `data_version` hash.
2. **Operational Serving Layer (PostgreSQL 16 via Alembic)**: Managed exclusively via SQLAlchemy (`src/db/models.py`) and Alembic (`migrations/versions/0001_initial_serving_schema.py`, verified with `compare_server_default=True`) as the single source of truth:
   - `players`: Master player profile (`tm_player_id` unique, `full_name`, `normalized_name` with `pg_trgm` GIN index `ix_players_normalized_name_trgm`, `position_group`, `current_club`, `current_league_code`).
   - `player_seasons`: Per-season/per-club aggregated stats keyed by `UNIQUE (player_id, season, tm_club_id)` so mid-season intra-league transfers never collide.
   - `valuations`: Historical valuation series keyed by `UNIQUE (player_id, valuation_date)`.
   - `player_current_features`: Precomputed serving feature vector (`player_id` PK, `as_of_date`, `data_version`, `league_code`, `trailing_league_median_eur`, `features` JSONB). Replaced atomically in a single database transaction during pipeline reloads. The FastAPI service asserts `COUNT(DISTINCT data_version) == 1` and refreshes the active `data_version` and active `model_version` (`model_versions WHERE is_active IS TRUE`) from the database every ~60 seconds.
   - `model_versions`: Registered model metadata (`model_version` PK, `model_type`, `data_version`, `feature_list`, `metrics`, `conformal_quantiles`, `is_active`). Enforces at most one active model via partial unique index `ix_model_versions_single_active` (`WHERE is_active IS TRUE`). Each model's `metadata.json` also records exact runtime library versions (`xgboost`, `shap`, `numpy`, `scikit-learn`, `pandas`, `pyarrow`).
   - `predictions_log`: Audit log of API predictions (`request_id`, `model_version`, `data_version`, `player_id`, `is_what_if`, `predicted_value_eur`, `interval_low_eur`, `interval_high_eur`, `top_shap_drivers`, `cache_status`, `latency_ms`).
   - **Upsert Rule**: Because SQLAlchemy's `onupdate=func.now()` only fires on ORM unit-of-work updates, all bulk `INSERT ... ON CONFLICT DO UPDATE` statements in Phases 1 & 2 explicitly set `updated_at = sa.func.now()`.

---

## 4. Data Sources, DuckDB Audit & Canonical Table Assignment

1. **Post-Opta Feature Set**: Because Opta terminated Sports Reference's advanced data feed earlier this year (removing xG, xA, and progressive actions from FBref), the core model is built strictly on Transfermarkt-derived metrics: valuations, appearances, minutes, goals, assists, cards, clean sheets/goals conceded, injury history (days/games missed), age, height, preferred foot, position, club, and league.
2. **Phase 1 Task 1 (`scripts/audit_datasets.py` via DuckDB)**:
   - Inspects the actual `LICENSE` files in `dcaribou/transfermarkt-datasets` and `salimt/football-datasets`.
   - Assigns **one canonical dataset per table** so duplicate or conflicting valuation records are never merged.
   - Audits and reports:
     1. Row counts per league and season (`GB1` vs. `EC1`) and non-null coverage for contracts, injuries, and valuations.
     2. Distribution of `position` / `sub_position` values (including `"Missing"` or `NULL`) against `ck_players_position_group`.
     3. Count of players with no current club or league (free agents, retirees) against `current_league_code NOT NULL`.
     4. Valuations that are $\le 0$, duplicated per `(player_id, valuation_date)`, or belong to players with `date_of_birth IS NULL` (`age_at_valuation NOT NULL`), logging exact drop counts at each cleaning step.
     5. Appearance coverage by year (`minutes_played`, `goals`) to set the training start date cutoff.
     6. Valuation counts per July–June cycle (excluding the incomplete cycle from July 2026 onward) to confirm every split gets sufficient rows.
     7. Point-in-time `league_at_valuation` derivation from match `appearances`/`games` in that season rather than static `clubs.domestic_competition_id` (which leaks promotions/relegations).
     8. Viability of the contract-aware model variant (checking if valid contemporaneous contract data spans $\ge 2$ full July–June cycles; if $< 2$ cycles, drop the variant and contract what-if slider, displaying contract length as read-only in the UI).
3. **Entity Resolution**:
   - Both `dcaribou` and `salimt` use Transfermarkt numeric `player_id`s (Tier 1 exact ID join verified by `date_of_birth`).
   - Secondary fuzzy matching (`rapidfuzz.fuzz.token_set_ratio`) is deferred unless the Phase 1 audit shows an unmatched rate $>0\%$; if invoked, it enforces an **exact `date_of_birth` match** + minimum name length gate so subset tokens like `"Silva"` never cross-match.
4. **No Synthetic Training Data**:
   - Models are trained and evaluated exclusively on real open-dataset records. Synthetic fixtures (`Test Player Alpha`, `player_id = 999001`) are restricted to `pytest` unit tests.

---

## 5. Leakage Prevention, Inflation Normalization, SHAP & Conformal Uncertainty

> **Phase 2.1 amendment (supersedes the EC1 and serving-scope details below):** training, evaluation and serving are restricted to `GB1` snapshots with ≥ 450 tracked minutes in the prior 365 days and a match-verified club. The trailing league median is computed on that population, and `EC1` is out of scope. See [`phase2_1_population_findings.md`](phase2_1_population_findings.md).

1. **Point-in-Time Feature Engineering & Leakage Guardrails (Phase 2 Spec)**:
   - **`player_seasons` is Display-Only**: Because `player_seasons` aggregates whole seasons (including matches played *after* a mid-season valuation date $t$), feature engineering in `src/features/` never reads `player_seasons`. All rolling 12m/24m match-level features (`mins_last_12m`, `goals_per90_shrunk_12m`, `assists_per90_shrunk_12m`, `cards_per90_12m`, `clean_sheets_per90_12m`, `goals_conceded_per90_12m`, `starts_ratio_12m`, `availability_ratio_12m`) are computed strictly from match-level `data/clean/appearances.parquet` filtered by `match_date < valuation_date`.
   - **Common-Granularity Prior-Season Features + Match-Level Indicator**: Because `EC1` domestic appearances (`data/raw/ec1_performances.parquet`) exist only as seasonal aggregates (`season_end_date < valuation_date`), Phase 2 defines two distinct feature groups to prevent train/serve granularity skew:
     1. *Completed Prior-Season Features (Common Granularity for both `GB1` and `EC1`)*: `prior_season_minutes`, `prior_season_appearances`, `prior_season_goals_per90_shrunk`, `prior_season_assists_per90_shrunk`, `prior_2seasons_minutes`, computed strictly from seasons with `season_end_date < valuation_date`.
     2. *Match-Level Rolling 12m/24m Features + `has_match_level_appearances` Indicator*: Computed from `appearances.parquet` (`match_date < valuation_date`). When a player has only seasonal `EC1` records and zero match-level logs in `[t - 365d, t)`, `has_match_level_appearances = False` and match-level rolling rates are left as `NaN` (handled natively by XGBoost default split directions rather than imputed as 0 minutes).
   - **Point-in-Time Injury Clipping at Valuation Date $t$**: An injury in `data/clean/injuries.parquet` starting at `from_date < t` that is still ongoing at $t$ (`end_date > t` or `end_date IS NULL`) would leak its future recovery duration if raw `days_missed` were summed. Phase 2 clips every injury interval to $[t - W, t)$:
     $$\text{clipped\_days\_missed}(t, W) = \max\bigl(0,\; \min(\text{COALESCE}(\text{end\_date}, t),\, t) - \max(\text{from\_date},\, t - W)\bigr)$$
     and sets `is_currently_injured_at_t = 1` when `from_date <= t < end_date`.
   - **Per-Row Contemporaneous Contract Filter & Phase 3 Ablation**:
     - The main model (`xgb_main`) is strictly contract-free across all `19,961` rows.
     - For the recent-window contract-aware variant, `valid_contract_years_remaining` is populated on a per-row basis only when `dc_contract_expiration_date > valuation_date`, the player is still at the same club (`tm_club_id_at_valuation = dc_current_club_id`), `sal_joined IS NOT NULL`, and $\max(\text{joined}, \text{date\_of\_last\_contract\_extension}) \le t$ (`2,596` valid rows across `2020/21`–`2025/26`). Phase 3's ablation determines whether the contract-aware variant improves accuracy on the valid recent subset and whether the UI contract slider is active or read-only.
   - **Serving Scope (`player_current_features`)**: Out of `2,552` historical cohort players, `player_current_features` is populated strictly for active players where `players.current_league_code IN ('GB1', 'EC1')`. For players now outside supported leagues (e.g., transferred to La Liga or retired), the API returns a clear `422 Unprocessable Entity` (`"outside supported leagues"`).
   - **Licensing & Privacy Guardrails (`salimt`)**: Because `salimt/football-datasets` has no `LICENSE` file (all rights reserved by default), its raw and derived Parquet files remain gitignored, and raw injury records are never stored in PostgreSQL or exposed through the API.
   - **Mandatory Leakage Test Suite (`tests/test_leakage.py`)**: Explicitly tests (1) `player_seasons` exclusion / future-match invariance, (2) common-granularity prior-season features + `has_match_level_appearances` indicator, and (3) ongoing-injury clipping at $t$.

2. **Target Normalization for Valuation Inflation (with Sparse-League Fallback)**:
   - Normalize the log target by the trailing 365-day league median valuation $M_{\text{league}, t^-}$ computed strictly from prior valuations $\in [t - 365\text{d}, t)$:
     $$z_{\text{norm}} = \log(1 + y) - \log(1 + M_{\text{league}, t^-})$$
   - **Minimum-Count Fallback ($N_{\min} = 30$)**: If a league has fewer than 30 valuations in $[t - 365\text{d}, t)$, $M_{\text{league}, t^-}$ falls back to a shrinkage-blended prior median across all historical valuations $< t$ for that league (or global non-top-5 prior median when cumulative league count $< 30$).
   - Predictions are transformed back at inference via $\hat{y} = \exp(\hat{z}_{\text{norm}} + \log(1 + M_{\text{league}, t^-})) - 1$.
3. **4-Way Full-Season Chronological Split + `EC1` Survivorship Bootstrap**:
   - Cutoffs align to **full July 1 – June 30 valuation cycles** (excluding incomplete cycles $\ge$ July 2026):
     `train` (`2013/14`–`2022/23`, `15,313` rows) $\rightarrow$ `val_tune` (`2023/24`, `1,577` rows) $\rightarrow$ `val_calib` (`2024/25`, `1,485` rows) $\rightarrow$ `test` (`2025/26`, `1,586` rows: `1,530` `GB1` + `56` `EC1`).
   - Secondary check reports `test` metrics on the **unseen-player slice** (players in `test` with zero prior rows in `train`/`val_tune`/`val_calib`).
   - Because the `EC1` cohort (`688` valuations across `138` players at `27` clubs; `56` valuations across `35` players in `test`) represents **Ecuador-stage valuations of players who reached Transfermarkt-tracked clubs (`EC1` survivorship OOD slice)**, Phase 3 reports **player-level block bootstrap 95% confidence intervals** on the `EC1` test metrics.
   - The secondary **Value-Update Model** (with `prev_market_value_eur`) is compared against a **Naive Last-Value Carry-Forward** baseline ($\hat{y}_t = y_{t-1}$).
4. **Multiplicative SHAP Drivers Relative to League Median**:
   - Because the model predicts $z_{\text{norm}} = \log(1 + y) - \log(1 + M_{\text{league}, t^-})$, the TreeSHAP base value $\phi_0$ represents the log-ratio vs. the trailing league median ("vs. league median baseline").
   - Each top-5 driver is reported as a multiplicative percentage effect $(\exp(\phi_j) - 1) \times 100\%$ (e.g., `"+18.4% vs. league median"`).
5. **Predicted-Value Bucketed Split Conformal Intervals**:
   - Partition `val_calib` into 3 tercile buckets of predicted log value $\hat{z}$ (Low, Mid, High).
   - Compute finite-sample corrected 80% quantiles of absolute log residuals $r_i = |\log(1+y_i) - \log(1+\hat{y}_i)|$:
     $$q_{80}^{(b)} = \text{Quantile}_{\lceil (n_b + 1)(1 - \alpha) \rceil / n_b}\left(\{r_i : i \in \text{Calib}_b\}\right), \quad \alpha = 0.20$$
   - Report empirical 80% interval coverage per season on `test`.

---

## 6. Redis Cache-Aside & API Specification (Phase 5)

- **Endpoints (`/v1`)**: `GET /v1/players?q=`, `GET /v1/players/{id}/valuation`, `POST /v1/valuations`, `GET /v1/model`, `GET /v1/health`.
- **Cache Keys**:
  - Player valuation: `valuation:{model_version}:{data_version}:{player_id}`
  - What-if valuation: `valuation:{model_version}:{data_version}:whatif:{payload_sha256}`
- **Resilience**: Singleflight lock (`SET NX EX 5`) for stampede protection, circuit-breaker graceful fallback to direct DB/model inference if Redis is down, and `X-Cache: HIT|MISS|BYPASS` response headers.

---

## 7. Eight-Phase Roadmap

- **Phase 0**: Architecture, Data Plan & 6-Table Alembic Schema (`feat/phase-0-arch`)
- **Phase 1**: DuckDB Dataset & License Audit (`scripts/audit_datasets.py`) + Ingestion + Cleaning (`feat/phase-1-ingestion`)
- **Phase 2**: Point-in-Time Feature Engineering + `player_current_features` Population (`feat/phase-2-features`)
- **Phase 3**: Baselines, Main XGBoost, Value-Update Model & Slice/Inflation Evaluation (`feat/phase-3-modeling`)
- **Phase 4**: Multiplicative SHAP + Predicted-Bucket Conformal Intervals + Model Card (`feat/phase-4-explainability`)
- **Phase 5**: FastAPI Service + Redis Cache-Aside (`feat/phase-5-api`)
- **Phase 6**: Next.js (Current Stable) Frontend (`feat/phase-6-frontend`)
- **Phase 7**: Leakage/Pipeline/API Pytest Suite, `k6` Load Test, GitHub Actions CI & Docs (`feat/phase-7-ci-docs`)
