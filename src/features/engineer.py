"""Phase 2: Point-in-Time Feature Engineering Engine (`src/features/engineer.py`).

Provides a single deterministic, vectorized DuckDB/pandas feature computation path
shared between:
  1. Historical training/evaluation snapshot generation (`valuation_snapshots.parquet`)
  2. Live serving precomputation (`player_current_features` PostgreSQL table)

Strict Point-in-Time Invariant (`t_data < as_of_date`):
  - Every source join (`appearances`, `player_seasons`, `injuries`, `transfers`,
    `valuations`, `all_valuations`) filters strictly on `< as_of_date`.
  - Any ongoing injury (`from_date < as_of_date` and `end_date IS NULL OR end_date >= as_of_date`)
    is clipped at `as_of_date` (`obs_end_date = LEAST(COALESCE(end_date, as_of_date), as_of_date)`),
    never reading static `days_missed`/`games_missed` or post-`as_of_date` recovery dates.
  - Empirical Bayes shrinkage priors ($M_0 = 900$ mins) and fallback league medians
    are estimated strictly on the `train` split (`as_of_date < 2023-07-01`).

Phase 2.1 Era-Stable Modeling Population:
  - The source only contains players who appear in a tracked match at some point, so
    historical rows include fringe players kept in the data only because of later
    appearances. Rows are therefore restricted to Premier League (`GB1`) snapshots whose
    player had >= 450 tracked minutes in the 365 days before `as_of_date`
    (`in_model_population`, `exclusion_reason`).
  - The trailing league median (target normalization index) is computed on that same
    population, point-in-time per prior valuation.
  - Features whose missingness depends on source coverage era (transfer-based club tenure,
    unbounded injury recency) are replaced by capped, appearance-based equivalents.
  - The source's club-at-valuation field is less reliable in older seasons (it matches the
    club of the player's latest tracked appearance in ~70% of 2013/14 GB1 rows vs ~99.8% in
    2025/26). Rows are kept only when the valuation club is corroborated by the latest
    tracked appearance before `as_of_date`, or by a transfer to that club since then.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import duckdb
import numpy as np
import pandas as pd

SHRINKAGE_M0_MINUTES: float = 900.0
LEAGUE_MEDIAN_N_MIN: int = 30
TRAIN_CUTOFF_DATE: str = "2023-07-01"
VAL_TUNE_CUTOFF_DATE: str = "2024-07-01"
VAL_CALIB_CUTOFF_DATE: str = "2025-07-01"

# Phase 2.1: era-stable modeling population and capped coverage-dependent features
POPULATION_MIN_TRACKED_MINUTES_12M: int = 450
SUPPORTED_LEAGUES: tuple[str, ...] = ("GB1",)
CLUB_TENURE_CAP_DAYS: int = 730
INJURY_RECENCY_CAP_DAYS: int = 730


def _sql_in_list(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def population_valuations_sql(vals_view: str, app_view: str) -> str:
    """Valuations whose player had >= `POPULATION_MIN_TRACKED_MINUTES_12M` tracked minutes in
    the 365 days strictly before that valuation date (point-in-time per valuation).

    Used for both the trailing league median index and its train-split fallback, so the
    normalization index tracks the same population the model is trained and served on.
    """
    return f"""
        SELECT
            v.league_at_valuation,
            CAST(v.valuation_date AS DATE) AS valuation_date,
            v.market_value_eur
        FROM {vals_view} v
        JOIN {app_view} a
          ON CAST(v.tm_player_id AS BIGINT) = CAST(a.tm_player_id AS BIGINT)
         AND CAST(a.match_date AS DATE) < CAST(v.valuation_date AS DATE)
         AND CAST(a.match_date AS DATE)
             >= CAST(CAST(v.valuation_date AS DATE) - INTERVAL 365 DAY AS DATE)
        GROUP BY v.tm_player_id, v.valuation_date, v.league_at_valuation, v.market_value_eur
        HAVING SUM(a.minutes_played) >= {POPULATION_MIN_TRACKED_MINUTES_12M}
    """

# `league_code` is constant within the GB1-only modeling population, so it is not a feature.
MAIN_MODEL_CATEGORICAL_FEATURES: list[str] = [
    "position_group",
    "preferred_foot",
]

MAIN_MODEL_NUMERIC_FEATURES: list[str] = [
    # Demographic & club tenure context
    "age_at_valuation",
    "age_squared",
    "height_cm",
    "club_tenure_days_capped",
    # Completed prior-season stats (populated for both GB1 and EC1)
    "prior_season_appearances",
    "prior_season_minutes",
    "prior_season_goals",
    "prior_season_assists",
    "prior_2seasons_minutes",
    "prior_season_goals_per90_shrunk",
    "prior_season_assists_per90_shrunk",
    # Match-level rolling 12m/24m stats (always covered inside the modeling population)
    "mins_last_12m",
    "mins_last_24m",
    "appearances_last_12m",
    "appearances_last_24m",
    "goals_last_12m",
    "assists_last_12m",
    "goals_per90_shrunk_12m",
    "assists_per90_shrunk_12m",
    "cards_per90_12m",
    "clean_sheets_per90_12m",
    "goals_conceded_per90_12m",
    "minutes_trend_12m_vs_prev12m",
    # Clipped point-in-time injury burden (NULL when the player is not in the injury source)
    "injury_days_12m",
    "injury_days_24m",
    "injury_count_12m",
    "injury_count_24m",
    "is_currently_injured_at_t",
    "days_since_last_injury_end_capped",
]

MAIN_MODEL_FEATURES: list[str] = MAIN_MODEL_CATEGORICAL_FEATURES + MAIN_MODEL_NUMERIC_FEATURES

VALUE_UPDATE_MODEL_FEATURES: list[str] = MAIN_MODEL_FEATURES + [
    "prev_log_market_value",
    "days_since_prev_valuation",
]

CONTRACT_ABLATION_FEATURES: list[str] = MAIN_MODEL_FEATURES + [
    "valid_contract_years_remaining",
]


@dataclass(frozen=True)
class EmpiricalBayesPriors:
    """Position-group per-90 priors and league medians fitted strictly on `train` (< 2023-07-01)."""

    match_goals_per90_by_pos: dict[str, float]
    match_assists_per90_by_pos: dict[str, float]
    season_goals_per90_by_pos: dict[str, float]
    season_assists_per90_by_pos: dict[str, float]
    train_league_median_eur: dict[str, float]
    shrinkage_m0_minutes: float = SHRINKAGE_M0_MINUTES
    league_median_n_min: int = LEAGUE_MEDIAN_N_MIN

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmpiricalBayesPriors:
        return cls(
            match_goals_per90_by_pos={k: float(v) for k, v in data["match_goals_per90_by_pos"].items()},
            match_assists_per90_by_pos={k: float(v) for k, v in data["match_assists_per90_by_pos"].items()},
            season_goals_per90_by_pos={k: float(v) for k, v in data["season_goals_per90_by_pos"].items()},
            season_assists_per90_by_pos={k: float(v) for k, v in data["season_assists_per90_by_pos"].items()},
            train_league_median_eur={k: float(v) for k, v in data["train_league_median_eur"].items()},
            shrinkage_m0_minutes=float(data.get("shrinkage_m0_minutes", SHRINKAGE_M0_MINUTES)),
            league_median_n_min=int(data.get("league_median_n_min", LEAGUE_MEDIAN_N_MIN)),
        )


def compute_empirical_bayes_priors(
    *,
    players_df: pd.DataFrame,
    appearances_df: pd.DataFrame,
    player_seasons_df: pd.DataFrame,
    valuations_df: pd.DataFrame,
) -> EmpiricalBayesPriors:
    """Compute position-group per-90 priors and fallback league medians strictly on `< 2023-07-01`."""
    con = duckdb.connect()
    con.register("src_players", players_df)
    con.register("src_app", appearances_df)
    con.register("src_seasons", player_seasons_df)
    con.register("src_vals", valuations_df)

    match_rows = con.sql(f"""
        SELECT
            p.position_group,
            COALESCE(
                90.0 * SUM(a.goals) / NULLIF(SUM(a.minutes_played), 0),
                0.0
            ) AS goals_per90,
            COALESCE(
                90.0 * SUM(a.assists) / NULLIF(SUM(a.minutes_played), 0),
                0.0
            ) AS assists_per90
        FROM src_app a
        JOIN src_players p ON a.tm_player_id = p.tm_player_id
        WHERE CAST(a.match_date AS DATE) < DATE '{TRAIN_CUTOFF_DATE}'
        GROUP BY p.position_group
    """).fetchall()

    season_rows = con.sql(f"""
        SELECT
            p.position_group,
            COALESCE(
                90.0 * SUM(ps.goals) / NULLIF(SUM(ps.minutes_played), 0),
                0.0
            ) AS goals_per90,
            COALESCE(
                90.0 * SUM(ps.assists) / NULLIF(SUM(ps.minutes_played), 0),
                0.0
            ) AS assists_per90
        FROM src_seasons ps
        JOIN src_players p ON ps.tm_player_id = p.tm_player_id
        WHERE CAST(ps.season_end_date AS DATE) < DATE '{TRAIN_CUTOFF_DATE}'
        GROUP BY p.position_group
    """).fetchall()

    league_rows = con.sql(f"""
        SELECT
            league_at_valuation,
            CAST(MEDIAN(market_value_eur) AS DOUBLE) AS median_val
        FROM ({population_valuations_sql("src_vals", "src_app")}) pv
        WHERE valuation_date < DATE '{TRAIN_CUTOFF_DATE}'
        GROUP BY league_at_valuation
    """).fetchall()

    default_pos = {"GK": 0.0, "DEF": 0.04, "MID": 0.12, "FWD": 0.30}
    match_g = {**default_pos, **{r[0]: round(float(r[1]), 6) for r in match_rows}}
    match_a = {**default_pos, **{r[0]: round(float(r[2]), 6) for r in match_rows}}
    season_g = {**default_pos, **{r[0]: round(float(r[1]), 6) for r in season_rows}}
    season_a = {**default_pos, **{r[0]: round(float(r[2]), 6) for r in season_rows}}

    default_league = {"GB1": 8_000_000.0, "EC1": 400_000.0}
    train_medians = {**default_league, **{r[0]: float(r[1]) for r in league_rows}}

    return EmpiricalBayesPriors(
        match_goals_per90_by_pos=match_g,
        match_assists_per90_by_pos=match_a,
        season_goals_per90_by_pos=season_g,
        season_assists_per90_by_pos=season_a,
        train_league_median_eur=train_medians,
    )


def assign_temporal_split(as_of_date_val: date | str) -> str:
    d_str = as_of_date_val.isoformat() if isinstance(as_of_date_val, date) else str(as_of_date_val)[:10]
    if d_str < TRAIN_CUTOFF_DATE:
        return "train"
    if d_str < VAL_TUNE_CUTOFF_DATE:
        return "val_tune"
    if d_str < VAL_CALIB_CUTOFF_DATE:
        return "val_calib"
    return "test"


def engineer_features_for_snapshots(
    snapshots_df: pd.DataFrame,
    *,
    players_df: pd.DataFrame,
    appearances_df: pd.DataFrame,
    player_seasons_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
    transfers_df: pd.DataFrame,
    valuations_df: pd.DataFrame,
    all_valuations_df: pd.DataFrame,
    priors: EmpiricalBayesPriors | None = None,
    injury_coverage_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute point-in-time features strictly from records dated `< as_of_date`.

    `injury_coverage_df` lists the `tm_player_id`s present in the injury source. Players
    outside it get NULL injury features instead of a misleading zero. When omitted, every
    player in `players_df` is treated as covered.

    Required columns in `snapshots_df`:
      - `tm_player_id` (int)
      - `as_of_date` (date)
      - `tm_club_id` (int)
      - `club_name` (str | None)
      - `league_code` (str: 'GB1' or 'EC1')
    Optional columns carried through when present (training snapshots):
      - `market_value_eur` (int)
      - `season_year` (int)
      - `cycle` (str)
      - `valid_contract_years_remaining` (float | None)
    """
    if snapshots_df.empty:
        return pd.DataFrame()

    if priors is None:
        priors = compute_empirical_bayes_priors(
            players_df=players_df,
            appearances_df=appearances_df,
            player_seasons_df=player_seasons_df,
            valuations_df=valuations_df,
        )

    work_snap = snapshots_df.copy().reset_index(drop=True)
    work_snap["snapshot_id"] = np.arange(len(work_snap), dtype=np.int64)
    if "market_value_eur" not in work_snap.columns:
        work_snap["market_value_eur"] = np.nan
    if "season_year" not in work_snap.columns:
        work_snap["season_year"] = np.nan
    if "cycle" not in work_snap.columns:
        work_snap["cycle"] = None
    if "valid_contract_years_remaining" not in work_snap.columns:
        work_snap["valid_contract_years_remaining"] = np.nan

    priors_rows = [
        {
            "position_group": pos,
            "match_goals_prior": priors.match_goals_per90_by_pos[pos],
            "match_assists_prior": priors.match_assists_per90_by_pos[pos],
            "season_goals_prior": priors.season_goals_per90_by_pos[pos],
            "season_assists_prior": priors.season_assists_per90_by_pos[pos],
        }
        for pos in ("GK", "DEF", "MID", "FWD")
    ]
    priors_df = pd.DataFrame(priors_rows)

    league_priors_df = pd.DataFrame(
        [
            {"league_code": lg, "train_league_median_eur": med}
            for lg, med in priors.train_league_median_eur.items()
        ]
    )

    m0 = float(priors.shrinkage_m0_minutes)
    n_min = int(priors.league_median_n_min)
    min_pop_minutes = int(POPULATION_MIN_TRACKED_MINUTES_12M)
    supported_leagues_sql = _sql_in_list(SUPPORTED_LEAGUES)

    if injury_coverage_df is None:
        injury_coverage_df = players_df[["tm_player_id"]]
    coverage_df = (
        injury_coverage_df[["tm_player_id"]]
        .dropna()
        .astype({"tm_player_id": "int64"})
        .drop_duplicates()
    )

    con = duckdb.connect()
    con.register("raw_snap", work_snap)
    con.register("src_players", players_df)
    con.register("src_app", appearances_df)
    con.register("src_seasons", player_seasons_df)
    con.register("src_inj", injuries_df)
    con.register("src_transfers", transfers_df)
    con.register("src_vals", valuations_df)
    con.register("src_all_vals", all_valuations_df)
    con.register("pos_priors", priors_df)
    con.register("league_priors", league_priors_df)
    con.register("src_inj_cov", coverage_df)

    result_df = con.sql(f"""
        WITH snap AS (
            SELECT
                CAST(snapshot_id AS BIGINT) AS snapshot_id,
                CAST(tm_player_id AS BIGINT) AS tm_player_id,
                CAST(as_of_date AS DATE) AS as_of_date,
                CAST(tm_club_id AS BIGINT) AS tm_club_id,
                CAST(club_name AS VARCHAR) AS club_name,
                CAST(league_code AS VARCHAR) AS league_code,
                TRY_CAST(market_value_eur AS DOUBLE) AS market_value_eur,
                TRY_CAST(season_year AS INTEGER) AS season_year,
                CAST(cycle AS VARCHAR) AS cycle,
                TRY_CAST(valid_contract_years_remaining AS DOUBLE) AS valid_contract_years_remaining
            FROM raw_snap
        ),
        -- 1. Point-in-time club tenure from transfers strictly < as_of_date
        club_tenure AS (
            SELECT
                s.snapshot_id,
                MAX(CAST(t.transfer_date AS DATE)) AS joined_club_date
            FROM snap s
            JOIN src_transfers t
              ON s.tm_player_id = CAST(t.tm_player_id AS BIGINT)
             AND s.tm_club_id = CAST(t.to_club_id AS BIGINT)
             AND CAST(t.transfer_date AS DATE) < s.as_of_date
            GROUP BY s.snapshot_id
        ),
        -- 2. Completed prior-season stats strictly < as_of_date
        season_by_player_season AS (
            SELECT
                s.snapshot_id,
                ps.season,
                MAX(CAST(ps.season_end_date AS DATE)) AS season_end_date,
                SUM(ps.appearances) AS appearances,
                SUM(ps.minutes_played) AS minutes_played,
                SUM(ps.goals) AS goals,
                SUM(ps.assists) AS assists
            FROM snap s
            JOIN src_seasons ps
              ON s.tm_player_id = CAST(ps.tm_player_id AS BIGINT)
             AND CAST(ps.season_end_date AS DATE) < s.as_of_date
             AND CAST(ps.season_end_date AS DATE) >= CAST(s.as_of_date - INTERVAL 730 DAY AS DATE)
            GROUP BY s.snapshot_id, ps.season
        ),
        ranked_prior_seasons AS (
            SELECT
                snapshot_id,
                season,
                season_end_date,
                appearances,
                minutes_played,
                goals,
                assists,
                ROW_NUMBER() OVER (
                    PARTITION BY snapshot_id
                    ORDER BY season_end_date DESC, season DESC
                ) AS rn
            FROM season_by_player_season
        ),
        prior_season_agg AS (
            SELECT
                r.snapshot_id,
                COALESCE(MAX(r.appearances) FILTER (
                    WHERE r.rn = 1 AND r.season_end_date >= CAST(s.as_of_date - INTERVAL 550 DAY AS DATE)
                ), 0) AS prior_season_appearances,
                COALESCE(MAX(r.minutes_played) FILTER (
                    WHERE r.rn = 1 AND r.season_end_date >= CAST(s.as_of_date - INTERVAL 550 DAY AS DATE)
                ), 0) AS prior_season_minutes,
                COALESCE(MAX(r.goals) FILTER (
                    WHERE r.rn = 1 AND r.season_end_date >= CAST(s.as_of_date - INTERVAL 550 DAY AS DATE)
                ), 0) AS prior_season_goals,
                COALESCE(MAX(r.assists) FILTER (
                    WHERE r.rn = 1 AND r.season_end_date >= CAST(s.as_of_date - INTERVAL 550 DAY AS DATE)
                ), 0) AS prior_season_assists,
                COALESCE(SUM(r.minutes_played), 0) AS prior_2seasons_minutes
            FROM ranked_prior_seasons r
            JOIN snap s ON r.snapshot_id = s.snapshot_id
            GROUP BY r.snapshot_id
        ),
        -- 3. Match-level rolling 12m/24m stats strictly < as_of_date
        match_rolling AS (
            SELECT
                s.snapshot_id,
                COUNT(*) AS raw_apps_24m,
                COUNT(*) FILTER (
                    WHERE CAST(a.match_date AS DATE) >= CAST(s.as_of_date - INTERVAL 365 DAY AS DATE)
                ) AS raw_apps_12m,
                COALESCE(SUM(a.minutes_played) FILTER (
                    WHERE CAST(a.match_date AS DATE) >= CAST(s.as_of_date - INTERVAL 365 DAY AS DATE)
                ), 0) AS raw_mins_12m,
                COALESCE(SUM(a.minutes_played), 0) AS raw_mins_24m,
                COALESCE(SUM(a.goals) FILTER (
                    WHERE CAST(a.match_date AS DATE) >= CAST(s.as_of_date - INTERVAL 365 DAY AS DATE)
                ), 0) AS raw_goals_12m,
                COALESCE(SUM(a.assists) FILTER (
                    WHERE CAST(a.match_date AS DATE) >= CAST(s.as_of_date - INTERVAL 365 DAY AS DATE)
                ), 0) AS raw_assists_12m,
                COALESCE(SUM(a.yellow_cards + a.red_cards) FILTER (
                    WHERE CAST(a.match_date AS DATE) >= CAST(s.as_of_date - INTERVAL 365 DAY AS DATE)
                ), 0) AS raw_cards_12m,
                COALESCE(SUM(a.clean_sheet) FILTER (
                    WHERE CAST(a.match_date AS DATE) >= CAST(s.as_of_date - INTERVAL 365 DAY AS DATE)
                ), 0) AS raw_cs_12m,
                COALESCE(SUM(a.goals_conceded) FILTER (
                    WHERE CAST(a.match_date AS DATE) >= CAST(s.as_of_date - INTERVAL 365 DAY AS DATE)
                ), 0) AS raw_gc_12m
            FROM snap s
            JOIN src_app a
              ON s.tm_player_id = CAST(a.tm_player_id AS BIGINT)
             AND CAST(a.match_date AS DATE) < s.as_of_date
             AND CAST(a.match_date AS DATE) >= CAST(s.as_of_date - INTERVAL 730 DAY AS DATE)
            GROUP BY s.snapshot_id
        ),
        -- 4. Point-in-time clipped injury intervals strictly < as_of_date
        injury_intervals AS (
            SELECT
                s.snapshot_id,
                s.as_of_date,
                CAST(i.from_date AS DATE) AS from_date,
                LEAST(COALESCE(CAST(i.end_date AS DATE), s.as_of_date), s.as_of_date) AS obs_end_date,
                CASE
                    WHEN i.end_date IS NULL OR CAST(i.end_date AS DATE) >= s.as_of_date THEN 1
                    ELSE 0
                END AS is_ongoing_at_t,
                CASE
                    WHEN i.end_date IS NOT NULL AND CAST(i.end_date AS DATE) < s.as_of_date
                    THEN CAST(i.end_date AS DATE)
                    ELSE NULL
                END AS completed_end_before_t,
                GREATEST(
                    0,
                    date_diff(
                        'day',
                        GREATEST(CAST(i.from_date AS DATE), CAST(s.as_of_date - INTERVAL 365 DAY AS DATE)),
                        LEAST(COALESCE(CAST(i.end_date AS DATE), s.as_of_date), s.as_of_date)
                    )
                ) AS overlap_12m,
                GREATEST(
                    0,
                    date_diff(
                        'day',
                        GREATEST(CAST(i.from_date AS DATE), CAST(s.as_of_date - INTERVAL 730 DAY AS DATE)),
                        LEAST(COALESCE(CAST(i.end_date AS DATE), s.as_of_date), s.as_of_date)
                    )
                ) AS overlap_24m
            FROM snap s
            JOIN src_inj i
              ON s.tm_player_id = CAST(i.tm_player_id AS BIGINT)
             AND CAST(i.from_date AS DATE) < s.as_of_date
        ),
        injury_rolling AS (
            SELECT
                snapshot_id,
                1 AS has_injury_history_flag,
                CAST(COALESCE(SUM(overlap_12m), 0) AS DOUBLE) AS injury_days_12m,
                CAST(COALESCE(SUM(overlap_24m), 0) AS DOUBLE) AS injury_days_24m,
                CAST(COUNT(*) FILTER (
                    WHERE overlap_12m > 0
                       OR from_date >= CAST(as_of_date - INTERVAL 365 DAY AS DATE)
                ) AS INTEGER) AS injury_count_12m,
                CAST(COUNT(*) FILTER (
                    WHERE overlap_24m > 0
                       OR from_date >= CAST(as_of_date - INTERVAL 730 DAY AS DATE)
                ) AS INTEGER) AS injury_count_24m,
                CAST(COALESCE(MAX(is_ongoing_at_t), 0) AS INTEGER) AS is_currently_injured_at_t,
                CASE
                    WHEN COALESCE(MAX(is_ongoing_at_t), 0) = 1 THEN 0.0
                    WHEN MAX(completed_end_before_t) IS NOT NULL
                    THEN CAST(date_diff('day', MAX(completed_end_before_t), MAX(as_of_date)) AS DOUBLE)
                    ELSE NULL
                END AS days_since_last_injury_end
            FROM injury_intervals
            GROUP BY snapshot_id
        ),
        -- 4b. Tracked-appearance coverage and appearance-based club tenure strictly < as_of_date
        ever_app AS (
            SELECT
                s.snapshot_id,
                MIN(CAST(a.match_date AS DATE)) AS first_tracked_app_date
            FROM snap s
            JOIN src_app a
              ON s.tm_player_id = CAST(a.tm_player_id AS BIGINT)
             AND CAST(a.match_date AS DATE) < s.as_of_date
            GROUP BY s.snapshot_id
        ),
        club_apps AS (
            SELECT
                s.snapshot_id,
                MIN(CAST(a.match_date AS DATE)) AS first_club_app_date
            FROM snap s
            JOIN src_app a
              ON s.tm_player_id = CAST(a.tm_player_id AS BIGINT)
             AND s.tm_club_id = CAST(a.tm_club_id AS BIGINT)
             AND CAST(a.match_date AS DATE) < s.as_of_date
            GROUP BY s.snapshot_id
        ),
        last_app AS (
            SELECT
                s.snapshot_id,
                ARG_MAX(CAST(a.tm_club_id AS BIGINT), CAST(a.match_date AS DATE)) AS last_app_club_id,
                MAX(CAST(a.match_date AS DATE)) AS last_app_date
            FROM snap s
            JOIN src_app a
              ON s.tm_player_id = CAST(a.tm_player_id AS BIGINT)
             AND CAST(a.match_date AS DATE) < s.as_of_date
             AND CAST(a.match_date AS DATE) >= CAST(s.as_of_date - INTERVAL 365 DAY AS DATE)
            GROUP BY s.snapshot_id
        ),
        signed_since_last_app AS (
            SELECT s.snapshot_id, 1 AS signed_flag
            FROM snap s
            JOIN last_app la ON s.snapshot_id = la.snapshot_id
            JOIN src_transfers t
              ON s.tm_player_id = CAST(t.tm_player_id AS BIGINT)
             AND CAST(t.to_club_id AS BIGINT) = s.tm_club_id
             AND CAST(t.transfer_date AS DATE) >= la.last_app_date
             AND CAST(t.transfer_date AS DATE) < s.as_of_date
            GROUP BY s.snapshot_id
        ),
        injury_cov AS (
            SELECT DISTINCT CAST(tm_player_id AS BIGINT) AS tm_player_id FROM src_inj_cov
        ),
        -- 5. Trailing 365-day league median over the modeling population strictly < as_of_date
        val_pop AS ({population_valuations_sql("src_vals", "src_app")}),
        distinct_league_dates AS (
            SELECT DISTINCT league_code, as_of_date FROM snap
        ),
        trailing_medians AS (
            SELECT
                d.league_code,
                d.as_of_date,
                COUNT(v.market_value_eur) AS n_365,
                CAST(MEDIAN(v.market_value_eur) AS DOUBLE) AS m_365,
                lp.train_league_median_eur,
                CASE
                    WHEN COUNT(v.market_value_eur) >= {n_min}
                    THEN CAST(MEDIAN(v.market_value_eur) AS DOUBLE)
                    WHEN COUNT(v.market_value_eur) > 0
                    THEN (CAST(COUNT(v.market_value_eur) AS DOUBLE) / {n_min}.0) * CAST(MEDIAN(v.market_value_eur) AS DOUBLE)
                       + (1.0 - CAST(COUNT(v.market_value_eur) AS DOUBLE) / {n_min}.0) * lp.train_league_median_eur
                    ELSE lp.train_league_median_eur
                END AS trailing_league_median_eur
            FROM distinct_league_dates d
            JOIN league_priors lp ON d.league_code = lp.league_code
            LEFT JOIN val_pop v
              ON d.league_code = v.league_at_valuation
             AND v.valuation_date >= CAST(d.as_of_date - INTERVAL 365 DAY AS DATE)
             AND v.valuation_date < d.as_of_date
            GROUP BY d.league_code, d.as_of_date, lp.train_league_median_eur
        ),
        -- 6. Lagged valuation strictly < as_of_date (for secondary Value-Update model only)
        lagged_val AS (
            SELECT
                s.snapshot_id,
                CAST(ARG_MAX(av.market_value_eur, CAST(av.valuation_date AS DATE)) AS DOUBLE) AS prev_market_value_eur,
                MAX(CAST(av.valuation_date AS DATE)) AS prev_valuation_date
            FROM snap s
            JOIN src_all_vals av
              ON s.tm_player_id = CAST(av.tm_player_id AS BIGINT)
             AND CAST(av.valuation_date AS DATE) < s.as_of_date
            GROUP BY s.snapshot_id
        )
        SELECT
            s.snapshot_id,
            s.tm_player_id,
            s.as_of_date,
            s.tm_club_id,
            s.club_name,
            s.league_code,
            s.season_year,
            s.cycle,
            s.market_value_eur,
            -- Categorical features
            p.position_group,
            COALESCE(NULLIF(TRIM(p.preferred_foot), ''), 'unknown') AS preferred_foot,
            -- Demographic & club tenure features
            ROUND(date_diff('day', CAST(p.date_of_birth AS DATE), s.as_of_date) / 365.25, 2) AS age_at_valuation,
            ROUND(
                POW(ROUND(date_diff('day', CAST(p.date_of_birth AS DATE), s.as_of_date) / 365.25, 2), 2),
                2
            ) AS age_squared,
            CAST(p.height_cm AS DOUBLE) AS height_cm,
            CASE
                WHEN ct.joined_club_date IS NOT NULL
                THEN CAST(date_diff('day', ct.joined_club_date, s.as_of_date) AS DOUBLE)
                ELSE NULL
            END AS days_since_joined_club,
            CASE
                WHEN ct.joined_club_date IS NOT NULL
                 AND date_diff('day', ct.joined_club_date, s.as_of_date) <= 365
                THEN 1
                ELSE 0
            END AS joined_within_365d_flag,
            CASE
                WHEN ca.first_club_app_date IS NULL THEN 0.0
                ELSE CAST(
                    LEAST(date_diff('day', ca.first_club_app_date, s.as_of_date), {CLUB_TENURE_CAP_DAYS})
                    AS DOUBLE
                )
            END AS club_tenure_days_capped,
            -- Completed prior-season features
            CAST(COALESCE(psa.prior_season_appearances, 0) AS DOUBLE) AS prior_season_appearances,
            CAST(COALESCE(psa.prior_season_minutes, 0) AS DOUBLE) AS prior_season_minutes,
            CAST(COALESCE(psa.prior_season_goals, 0) AS DOUBLE) AS prior_season_goals,
            CAST(COALESCE(psa.prior_season_assists, 0) AS DOUBLE) AS prior_season_assists,
            CAST(COALESCE(psa.prior_2seasons_minutes, 0) AS DOUBLE) AS prior_2seasons_minutes,
            ROUND(
                (90.0 * COALESCE(psa.prior_season_goals, 0) + {m0} * pp.season_goals_prior)
                / (COALESCE(psa.prior_season_minutes, 0) + {m0}),
                6
            ) AS prior_season_goals_per90_shrunk,
            ROUND(
                (90.0 * COALESCE(psa.prior_season_assists, 0) + {m0} * pp.season_assists_prior)
                / (COALESCE(psa.prior_season_minutes, 0) + {m0}),
                6
            ) AS prior_season_assists_per90_shrunk,
            -- Match-level rolling 12m/24m features + coverage indicator
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL THEN 1.0
                ELSE 0.0
            END AS has_match_level_appearances,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN CAST(COALESCE(mr.raw_mins_12m, 0) AS DOUBLE)
                ELSE NULL
            END AS mins_last_12m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN CAST(COALESCE(mr.raw_mins_24m, 0) AS DOUBLE)
                ELSE NULL
            END AS mins_last_24m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN CAST(COALESCE(mr.raw_apps_12m, 0) AS DOUBLE)
                ELSE NULL
            END AS appearances_last_12m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN CAST(COALESCE(mr.raw_apps_24m, 0) AS DOUBLE)
                ELSE NULL
            END AS appearances_last_24m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN CAST(COALESCE(mr.raw_goals_12m, 0) AS DOUBLE)
                ELSE NULL
            END AS goals_last_12m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN CAST(COALESCE(mr.raw_assists_12m, 0) AS DOUBLE)
                ELSE NULL
            END AS assists_last_12m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN ROUND(
                    (90.0 * COALESCE(mr.raw_goals_12m, 0) + {m0} * pp.match_goals_prior)
                    / (COALESCE(mr.raw_mins_12m, 0) + {m0}),
                    6
                )
                ELSE NULL
            END AS goals_per90_shrunk_12m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN ROUND(
                    (90.0 * COALESCE(mr.raw_assists_12m, 0) + {m0} * pp.match_assists_prior)
                    / (COALESCE(mr.raw_mins_12m, 0) + {m0}),
                    6
                )
                ELSE NULL
            END AS assists_per90_shrunk_12m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN ROUND(90.0 * COALESCE(mr.raw_cards_12m, 0) / (COALESCE(mr.raw_mins_12m, 0) + {m0}), 6)
                ELSE NULL
            END AS cards_per90_12m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN ROUND(90.0 * COALESCE(mr.raw_cs_12m, 0) / (COALESCE(mr.raw_mins_12m, 0) + {m0}), 6)
                ELSE NULL
            END AS clean_sheets_per90_12m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN ROUND(90.0 * COALESCE(mr.raw_gc_12m, 0) / (COALESCE(mr.raw_mins_12m, 0) + {m0}), 6)
                ELSE NULL
            END AS goals_conceded_per90_12m,
            CASE
                WHEN ea.first_tracked_app_date IS NOT NULL
                THEN CAST(
                    2 * COALESCE(mr.raw_mins_12m, 0) - COALESCE(mr.raw_mins_24m, 0)
                    AS DOUBLE
                )
                ELSE NULL
            END AS minutes_trend_12m_vs_prev12m,
            0.0 AS has_advanced_stats,
            -- Clipped injury burden features
            CASE WHEN ic.tm_player_id IS NOT NULL THEN 1.0 ELSE 0.0 END AS in_injury_source,
            CAST(COALESCE(ir.has_injury_history_flag, 0) AS DOUBLE) AS has_injury_history_flag,
            CASE WHEN ic.tm_player_id IS NULL THEN NULL
                 ELSE COALESCE(ir.injury_days_12m, 0.0) END AS injury_days_12m,
            CASE WHEN ic.tm_player_id IS NULL THEN NULL
                 ELSE COALESCE(ir.injury_days_24m, 0.0) END AS injury_days_24m,
            CASE WHEN ic.tm_player_id IS NULL THEN NULL
                 ELSE CAST(COALESCE(ir.injury_count_12m, 0) AS DOUBLE) END AS injury_count_12m,
            CASE WHEN ic.tm_player_id IS NULL THEN NULL
                 ELSE CAST(COALESCE(ir.injury_count_24m, 0) AS DOUBLE) END AS injury_count_24m,
            CASE WHEN ic.tm_player_id IS NULL THEN NULL
                 ELSE CAST(COALESCE(ir.is_currently_injured_at_t, 0) AS DOUBLE)
            END AS is_currently_injured_at_t,
            ir.days_since_last_injury_end,
            CASE
                WHEN ic.tm_player_id IS NULL THEN NULL
                ELSE LEAST(
                    COALESCE(ir.days_since_last_injury_end, {INJURY_RECENCY_CAP_DAYS}.0),
                    {INJURY_RECENCY_CAP_DAYS}.0
                )
            END AS days_since_last_injury_end_capped,
            -- Normalization & target columns
            ROUND(tm.trailing_league_median_eur, 2) AS trailing_league_median_eur,
            CASE
                WHEN s.market_value_eur IS NOT NULL AND s.market_value_eur > 0
                THEN LN(s.market_value_eur + 1.0)
                ELSE NULL
            END AS target_log_market_value,
            CASE
                WHEN s.market_value_eur IS NOT NULL AND s.market_value_eur > 0
                THEN LN((s.market_value_eur + 1.0) / (tm.trailing_league_median_eur + 1.0))
                ELSE NULL
            END AS target_log_ratio_vs_league_median,
            -- Secondary Value-Update model lagged features (excluded from MAIN_MODEL_FEATURES)
            lv.prev_market_value_eur,
            CASE
                WHEN lv.prev_market_value_eur IS NOT NULL AND lv.prev_market_value_eur > 0
                THEN LN(lv.prev_market_value_eur + 1.0)
                ELSE NULL
            END AS prev_log_market_value,
            CASE
                WHEN lv.prev_valuation_date IS NOT NULL
                THEN CAST(date_diff('day', lv.prev_valuation_date, s.as_of_date) AS DOUBLE)
                ELSE NULL
            END AS days_since_prev_valuation,
            -- Recent-window contract ablation feature (excluded from MAIN_MODEL_FEATURES)
            s.valid_contract_years_remaining,
            -- Phase 2.1 modeling population
            CASE
                WHEN s.league_code NOT IN {supported_leagues_sql} THEN 'league_not_supported'
                WHEN ea.first_tracked_app_date IS NULL
                  OR COALESCE(mr.raw_mins_12m, 0) < {min_pop_minutes}
                THEN 'insufficient_tracked_minutes'
                WHEN la.last_app_club_id IS DISTINCT FROM s.tm_club_id AND sl.signed_flag IS NULL
                THEN 'club_attribution_unverified'
                ELSE NULL
            END AS exclusion_reason
        FROM snap s
        JOIN src_players p ON s.tm_player_id = CAST(p.tm_player_id AS BIGINT)
        JOIN pos_priors pp ON p.position_group = pp.position_group
        JOIN trailing_medians tm
          ON s.league_code = tm.league_code AND s.as_of_date = tm.as_of_date
        LEFT JOIN club_tenure ct ON s.snapshot_id = ct.snapshot_id
        LEFT JOIN prior_season_agg psa ON s.snapshot_id = psa.snapshot_id
        LEFT JOIN match_rolling mr ON s.snapshot_id = mr.snapshot_id
        LEFT JOIN injury_rolling ir ON s.snapshot_id = ir.snapshot_id
        LEFT JOIN lagged_val lv ON s.snapshot_id = lv.snapshot_id
        LEFT JOIN ever_app ea ON s.snapshot_id = ea.snapshot_id
        LEFT JOIN club_apps ca ON s.snapshot_id = ca.snapshot_id
        LEFT JOIN last_app la ON s.snapshot_id = la.snapshot_id
        LEFT JOIN signed_since_last_app sl ON s.snapshot_id = sl.snapshot_id
        LEFT JOIN injury_cov ic ON s.tm_player_id = ic.tm_player_id
        ORDER BY s.snapshot_id
    """).df()

    result_df["as_of_date"] = pd.to_datetime(result_df["as_of_date"]).dt.date
    result_df["split"] = [assign_temporal_split(d) for d in result_df["as_of_date"]]
    result_df["in_model_population"] = result_df["exclusion_reason"].isna()
    return result_df.drop(columns=["snapshot_id"])


def build_serving_feature_dict(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Serialize a feature row from `engineer_features_for_snapshots` into the JSONB
    payload stored in `player_current_features.features`.

    Contains every column in `MAIN_MODEL_FEATURES` (for direct main-model inference
    and what-if overrides) plus auxiliary context fields (`tm_club_id`, `club_name`,
    and secondary Value-Update / contract-ablation fields).
    """
    payload: dict[str, Any] = {}
    cols_to_export = (
        MAIN_MODEL_FEATURES
        + [
            "tm_club_id",
            "club_name",
            "prev_market_value_eur",
            "prev_log_market_value",
            "days_since_prev_valuation",
            "valid_contract_years_remaining",
        ]
    )
    for col in cols_to_export:
        val = row.get(col) if isinstance(row, dict) else row[col]
        if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
            payload[col] = None
        elif isinstance(val, (np.integer, int)):
            payload[col] = int(val)
        elif isinstance(val, (np.floating, float)):
            payload[col] = float(val)
        else:
            payload[col] = str(val)
    return payload
