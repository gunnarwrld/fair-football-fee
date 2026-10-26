"""Phase 1 (Task 2b): Deterministic Cleaning Pipeline & PostgreSQL Loader.

Reads `data/raw/*.parquet`, uses the shared point-in-time SQL definitions in
`src.data.pit_sql` (avoiding static club-table promotion/relegation leakage and
keeping audit + cleaning SQL identical), logs exact row drop counts at every
cleaning step to `data/clean/cleaning_report.json`, writes `data/clean/*.parquet`,
and bulk-upserts the operational serving tables (`players`, `player_seasons`,
`valuations`) into PostgreSQL while deleting stale keys removed upstream.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.config import settings
from src.data.pit_sql import register_pit_views
from src.db.models import Player, PlayerSeason, Valuation
from src.db.session import SessionLocal


def normalize_player_name(name: str) -> str:
    if not name:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(name))
    ascii_str = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    cleaned = re.sub(r"[^a-z0-9\s-]", " ", ascii_str)
    return re.sub(r"\s+", " ", cleaned).strip()


def run_cleaning_pipeline(
    raw_dir: Path | None = None,
    clean_dir: Path | None = None,
) -> dict[str, Any]:
    raw_dir = raw_dir or settings.raw_data_dir
    clean_dir = clean_dir or settings.clean_data_dir
    clean_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / "MANIFEST.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"data_version": "unknown"}
    )

    con = duckdb.connect()
    con.create_function("norm_name", normalize_player_name, ["VARCHAR"], "VARCHAR")

    con.execute(
        f"CREATE VIEW raw_comp AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/competitions.parquet')"
    )
    con.execute(
        f"CREATE VIEW raw_clubs AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/clubs.parquet')"
    )
    con.execute(
        f"CREATE VIEW raw_players AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/players.parquet')"
    )
    con.execute(
        f"CREATE VIEW raw_sal_prof AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/salimt_profiles.parquet')"
    )
    con.execute(
        f"CREATE VIEW raw_val AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/valuations.parquet')"
    )
    con.execute(
        f"CREATE VIEW raw_app AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/appearances.parquet')"
    )
    con.execute(
        f"CREATE VIEW raw_games AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/games.parquet')"
    )
    con.execute(
        f"CREATE VIEW raw_inj AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/injuries.parquet')"
    )
    con.execute(
        f"CREATE VIEW raw_tcs AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/team_competitions_seasons.parquet')"
    )
    con.execute(
        f"CREATE VIEW raw_ec1_perf AS SELECT * FROM read_parquet('{raw_dir.as_posix()}/ec1_performances.parquet')"
    )

    # 1. Register shared point-in-time club-season & staged_valuations views
    register_pit_views(con)

    # 2. Step-by-step drop accounting
    drop_counts_row = con.sql("""
        SELECT
            COUNT(*) AS raw_total,
            COUNT(*) FILTER (WHERE market_value_eur IS NULL OR market_value_eur <= 0) AS dropped_non_positive,
            COUNT(*) FILTER (WHERE market_value_eur > 0 AND dup_rn > 1) AS dropped_duplicates,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0 AND dup_rn = 1 AND date_of_birth IS NULL
            ) AS dropped_missing_dob,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0 AND dup_rn = 1 AND date_of_birth IS NOT NULL
                  AND resolved_position NOT IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
            ) AS dropped_missing_position,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0 AND dup_rn = 1 AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date < DATE '2013-07-01'
            ) AS dropped_before_2013_07_01,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0 AND dup_rn = 1 AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2026-07-01'
            ) AS dropped_incomplete_cycle_from_2026_07_01,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0 AND dup_rn = 1 AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2013-07-01' AND valuation_date < DATE '2026-07-01'
                  AND league_at_valuation NOT IN ('GB1', 'EC1')
            ) AS excluded_other_leagues,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0 AND dup_rn = 1 AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2013-07-01' AND valuation_date < DATE '2026-07-01'
                  AND league_at_valuation IN ('GB1', 'EC1')
            ) AS kept_gb1_ec1_valuations
        FROM staged_valuations
    """).fetchall()[0]

    # 3. Export clean GB1 + EC1 valuations.parquet (including per-row contract validity flag for Phase 2/3 ablation)
    val_out = clean_dir / "valuations.parquet"
    con.execute(f"""
        COPY (
            SELECT
                tm_player_id,
                valuation_date,
                market_value_eur,
                tm_club_id_at_valuation,
                club_at_valuation,
                league_at_valuation,
                season_year,
                cycle,
                ROUND(date_diff('day', date_of_birth, valuation_date) / 365.25, 2) AS age_at_valuation,
                CASE
                    WHEN dc_contract_expiration_date IS NOT NULL
                     AND dc_contract_expiration_date > valuation_date
                     AND tm_club_id_at_valuation = dc_current_club_id
                     AND sal_joined IS NOT NULL
                     AND sal_joined <= DATE '2025-09-13'
                     AND GREATEST(
                         sal_joined,
                         COALESCE(
                             CASE WHEN sal_last_extension <= DATE '2025-09-12'
                                  THEN sal_last_extension ELSE NULL END,
                             sal_joined
                         )
                     ) <= valuation_date
                    THEN ROUND(date_diff('day', valuation_date, dc_contract_expiration_date) / 365.25, 2)
                    ELSE NULL
                END AS valid_contract_years_remaining
            FROM staged_valuations
            WHERE market_value_eur > 0
              AND dup_rn = 1
              AND date_of_birth IS NOT NULL
              AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
              AND valuation_date >= DATE '2013-07-01'
              AND valuation_date < DATE '2026-07-01'
              AND league_at_valuation IN ('GB1', 'EC1')
            ORDER BY valuation_date, tm_player_id
        ) TO '{val_out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # 4. Identify the target cohort of distinct players (all players in clean GB1 + EC1 valuations)
    con.execute(f"""
        CREATE TEMP VIEW target_cohort AS
        SELECT
            tm_player_id,
            ARG_MAX(league_at_valuation, valuation_date) AS latest_pit_league,
            ARG_MAX(club_at_valuation, valuation_date) AS latest_pit_club
        FROM read_parquet('{val_out.as_posix()}')
        GROUP BY tm_player_id
    """)

    # Also save all_valuations.parquet for the target cohort across all career leagues
    # (used in Phase 2 solely for the secondary Value-Update model's lagged valuation lookup)
    all_val_out = clean_dir / "all_valuations.parquet"
    con.execute(f"""
        COPY (
            SELECT
                sv.tm_player_id,
                sv.valuation_date,
                sv.market_value_eur,
                sv.tm_club_id_at_valuation,
                sv.club_at_valuation,
                sv.league_at_valuation,
                sv.season_year,
                sv.cycle,
                ROUND(date_diff('day', sv.date_of_birth, sv.valuation_date) / 365.25, 2) AS age_at_valuation
            FROM staged_valuations sv
            JOIN target_cohort tc ON sv.tm_player_id = tc.tm_player_id
            WHERE sv.market_value_eur > 0
              AND sv.dup_rn = 1
              AND sv.date_of_birth IS NOT NULL
              AND sv.valuation_date < DATE '2026-07-01'
            ORDER BY sv.tm_player_id, sv.valuation_date
        ) TO '{all_val_out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # 5. Export clean players.parquet
    players_out = clean_dir / "players.parquet"
    con.execute(f"""
        COPY (
            SELECT
                CAST(p.player_id AS BIGINT) AS tm_player_id,
                TRIM(p.name) AS full_name,
                norm_name(p.name) AS normalized_name,
                CAST(p.date_of_birth AS DATE) AS date_of_birth,
                NULLIF(TRIM(p.country_of_citizenship), '') AS nationality,
                CASE COALESCE(NULLIF(p.position, 'Missing'), sp.main_position)
                    WHEN 'Goalkeeper' THEN 'GK'
                    WHEN 'Defender' THEN 'DEF'
                    WHEN 'Midfield' THEN 'MID'
                    WHEN 'Attack' THEN 'FWD'
                END AS position_group,
                NULLIF(TRIM(COALESCE(p.sub_position, sp.position)), '') AS primary_position,
                CASE
                    WHEN LOWER(TRIM(COALESCE(p.foot, sp.foot, ''))) IN ('left', 'right', 'both')
                    THEN LOWER(TRIM(COALESCE(p.foot, sp.foot)))
                    ELSE NULL
                END AS preferred_foot,
                CASE
                    WHEN p.height_in_cm IS NOT NULL AND p.height_in_cm BETWEEN 150 AND 220
                    THEN CAST(p.height_in_cm AS INTEGER)
                    ELSE NULL
                END AS height_cm,
                COALESCE(NULLIF(TRIM(p.current_club_name), ''), tc.latest_pit_club) AS current_club,
                COALESCE(
                    NULLIF(TRIM(p.current_club_domestic_competition_id), ''),
                    tc.latest_pit_league,
                    'UNKNOWN'
                ) AS current_league_code,
                CAST(p.contract_expiration_date AS DATE) AS contract_expires_at
            FROM raw_players p
            JOIN target_cohort tc ON p.player_id = tc.tm_player_id
            LEFT JOIN raw_sal_prof sp ON p.player_id = sp.player_id
            ORDER BY p.player_id
        ) TO '{players_out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # 6. Export clean match-level appearances.parquet for target cohort
    app_out = clean_dir / "appearances.parquet"
    con.execute(f"""
        COPY (
            SELECT
                a.appearance_id,
                CAST(a.game_id AS BIGINT) AS game_id,
                CAST(a.player_id AS BIGINT) AS tm_player_id,
                CAST(a.player_club_id AS BIGINT) AS tm_club_id,
                CAST(a.date AS DATE) AS match_date,
                a.competition_id,
                COALESCE(CAST(g.season AS INTEGER),
                    CASE WHEN EXTRACT(MONTH FROM a.date) >= 7 THEN CAST(EXTRACT(YEAR FROM a.date) AS INTEGER)
                         ELSE CAST(EXTRACT(YEAR FROM a.date) - 1 AS INTEGER) END
                ) AS season_year,
                GREATEST(COALESCE(CAST(a.minutes_played AS INTEGER), 0), 0) AS minutes_played,
                GREATEST(COALESCE(CAST(a.goals AS INTEGER), 0), 0) AS goals,
                GREATEST(COALESCE(CAST(a.assists AS INTEGER), 0), 0) AS assists,
                GREATEST(COALESCE(CAST(a.yellow_cards AS INTEGER), 0), 0) AS yellow_cards,
                GREATEST(COALESCE(CAST(a.red_cards AS INTEGER), 0), 0) AS red_cards,
                CASE
                    WHEN g.home_club_id = a.player_club_id THEN COALESCE(CAST(g.away_club_goals AS INTEGER), 0)
                    WHEN g.away_club_id = a.player_club_id THEN COALESCE(CAST(g.home_club_goals AS INTEGER), 0)
                    ELSE 0
                END AS goals_conceded,
                CASE
                    WHEN COALESCE(a.minutes_played, 0) >= 45
                     AND (
                        (g.home_club_id = a.player_club_id AND COALESCE(g.away_club_goals, 0) = 0)
                        OR (g.away_club_id = a.player_club_id AND COALESCE(g.home_club_goals, 0) = 0)
                     )
                    THEN 1
                    ELSE 0
                END AS clean_sheet
            FROM raw_app a
            JOIN target_cohort tc ON a.player_id = tc.tm_player_id
            LEFT JOIN raw_games g ON a.game_id = g.game_id
            WHERE a.date IS NOT NULL
              AND a.player_club_id IS NOT NULL
              AND a.date < DATE '2026-07-01'
            ORDER BY a.player_id, a.date
        ) TO '{app_out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # 7. Export clean player_seasons.parquet (display-only seasonal summary keyed by tm_player_id, season, tm_club_id)
    seasons_out = clean_dir / "player_seasons.parquet"
    con.execute(f"""
        COPY (
            WITH dc_seasonal AS (
                SELECT
                    ap.tm_player_id,
                    CAST(ap.season_year AS VARCHAR) || '/' || CAST(ap.season_year + 1 AS VARCHAR) AS season,
                    MAKE_DATE(ap.season_year + 1, 6, 30) AS season_end_date,
                    ap.tm_club_id,
                    MAX(c.name) AS club_name,
                    COALESCE(MAX(csp.pit_league_code), MAX(ap.competition_id), 'OTHER') AS league_code,
                    COUNT(*) AS appearances,
                    SUM(ap.minutes_played) AS minutes_played,
                    SUM(ap.goals) AS goals,
                    SUM(ap.assists) AS assists,
                    SUM(ap.yellow_cards) AS yellow_cards,
                    SUM(ap.red_cards) AS red_cards,
                    SUM(ap.clean_sheet) AS clean_sheets,
                    SUM(ap.goals_conceded) AS goals_conceded
                FROM read_parquet('{app_out.as_posix()}') ap
                LEFT JOIN raw_clubs c ON ap.tm_club_id = c.club_id
                LEFT JOIN club_season_pit csp
                  ON ap.tm_club_id = csp.club_id AND ap.season_year = csp.season
                GROUP BY ap.tm_player_id, ap.season_year, ap.tm_club_id
            ),
            ec1_seasonal AS (
                SELECT
                    CAST(ep.player_id AS BIGINT) AS tm_player_id,
                    CASE
                        WHEN LENGTH(TRIM(ep.season_name)) = 4
                        THEN TRIM(ep.season_name) || '/' || CAST(CAST(TRIM(ep.season_name) AS INTEGER) + 1 AS VARCHAR)
                        ELSE '20' || SUBSTR(TRIM(ep.season_name), 1, 2) || '/20' || SUBSTR(TRIM(ep.season_name), 4, 2)
                    END AS season,
                    CASE
                        WHEN LENGTH(TRIM(ep.season_name)) = 4
                        THEN MAKE_DATE(CAST(TRIM(ep.season_name) AS INTEGER), 12, 31)
                        ELSE MAKE_DATE(2000 + CAST(SUBSTR(TRIM(ep.season_name), 4, 2) AS INTEGER), 6, 30)
                    END AS season_end_date,
                    CAST(ep.tm_club_id AS BIGINT) AS tm_club_id,
                    MAX(ep.club_name) AS club_name,
                    'EC1' AS league_code,
                    SUM(ep.appearances) AS appearances,
                    SUM(ep.minutes_played) AS minutes_played,
                    SUM(ep.goals) AS goals,
                    SUM(ep.assists) AS assists,
                    SUM(ep.yellow_cards) AS yellow_cards,
                    SUM(ep.red_cards) AS red_cards,
                    SUM(ep.clean_sheets) AS clean_sheets,
                    SUM(ep.goals_conceded) AS goals_conceded
                FROM raw_ec1_perf ep
                JOIN target_cohort tc ON ep.player_id = tc.tm_player_id
                WHERE ep.tm_club_id IS NOT NULL
                GROUP BY 1, 2, 3, 4
            ),
            combined AS (
                SELECT * FROM dc_seasonal
                UNION ALL
                SELECT e.*
                FROM ec1_seasonal e
                LEFT JOIN dc_seasonal d
                  ON e.tm_player_id = d.tm_player_id
                 AND e.season = d.season
                 AND e.tm_club_id = d.tm_club_id
                WHERE d.tm_player_id IS NULL
            )
            SELECT
                tm_player_id,
                season,
                season_end_date,
                tm_club_id,
                club_name,
                league_code,
                CAST(appearances AS INTEGER) AS appearances,
                CAST(minutes_played AS INTEGER) AS minutes_played,
                CAST(goals AS INTEGER) AS goals,
                CAST(assists AS INTEGER) AS assists,
                CAST(yellow_cards AS INTEGER) AS yellow_cards,
                CAST(red_cards AS INTEGER) AS red_cards,
                CAST(clean_sheets AS INTEGER) AS clean_sheets,
                CAST(goals_conceded AS INTEGER) AS goals_conceded,
                CASE
                    WHEN minutes_played > 0 THEN ROUND(90.0 * goals / minutes_played, 4)
                    ELSE 0.0
                END AS goals_per90,
                CASE
                    WHEN minutes_played > 0 THEN ROUND(90.0 * assists / minutes_played, 4)
                    ELSE 0.0
                END AS assists_per90,
                CAST(NULL AS DOUBLE) AS xg_per90,
                CAST(NULL AS DOUBLE) AS xa_per90,
                CAST(NULL AS DOUBLE) AS prog_actions_per90,
                FALSE AS has_advanced_stats
            FROM combined
            ORDER BY tm_player_id, season_end_date, tm_club_id
        ) TO '{seasons_out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    # 8. Export clean injuries.parquet for target cohort (lakehouse-only; never exposed via API)
    inj_out = clean_dir / "injuries.parquet"
    con.execute(f"""
        COPY (
            SELECT
                CAST(i.player_id AS BIGINT) AS tm_player_id,
                TRIM(i.season_name) AS season_name,
                COALESCE(NULLIF(TRIM(i.injury_reason), ''), 'Unknown injury') AS injury_reason,
                CAST(i.from_date AS DATE) AS from_date,
                CAST(i.end_date AS DATE) AS end_date,
                GREATEST(COALESCE(TRY_CAST(i.days_missed AS DOUBLE), 0.0), 0.0) AS days_missed,
                GREATEST(COALESCE(TRY_CAST(i.games_missed AS INTEGER), 0), 0) AS games_missed
            FROM raw_inj i
            JOIN target_cohort tc ON i.player_id = tc.tm_player_id
            WHERE i.from_date IS NOT NULL
              AND CAST(i.from_date AS DATE) < DATE '2026-07-01'
            ORDER BY i.player_id, i.from_date
        ) TO '{inj_out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    report = {
        "data_version": manifest.get("data_version"),
        "cleaned_at": datetime.now(UTC).isoformat(),
        "valuation_drops": {
            "raw_total": drop_counts_row[0],
            "dropped_non_positive": drop_counts_row[1],
            "dropped_duplicates": drop_counts_row[2],
            "dropped_missing_dob": drop_counts_row[3],
            "dropped_missing_position": drop_counts_row[4],
            "dropped_before_2013_07_01": drop_counts_row[5],
            "dropped_incomplete_cycle_from_2026_07_01": drop_counts_row[6],
            "excluded_other_leagues": drop_counts_row[7],
            "kept_gb1_ec1_valuations": drop_counts_row[8],
        },
        "clean_table_counts": {
            "players": con.sql(f"SELECT COUNT(*) FROM read_parquet('{players_out.as_posix()}')").fetchall()[0][0],
            "valuations": con.sql(f"SELECT COUNT(*) FROM read_parquet('{val_out.as_posix()}')").fetchall()[0][0],
            "all_valuations": con.sql(f"SELECT COUNT(*) FROM read_parquet('{all_val_out.as_posix()}')").fetchall()[0][0],
            "appearances": con.sql(f"SELECT COUNT(*) FROM read_parquet('{app_out.as_posix()}')").fetchall()[0][0],
            "player_seasons": con.sql(f"SELECT COUNT(*) FROM read_parquet('{seasons_out.as_posix()}')").fetchall()[0][0],
            "injuries": con.sql(f"SELECT COUNT(*) FROM read_parquet('{inj_out.as_posix()}')").fetchall()[0][0],
        },
    }

    report_path = clean_dir / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of dicts with all NaN/NaT values replaced by None."""
    records: list[dict[str, Any]] = df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
    for row in records:
        for k, v in row.items():
            if v is not None and pd.isna(v):
                row[k] = None
    return records


def load_clean_to_postgres(
    session: Session | None = None,
    clean_dir: Path | None = None,
    batch_size: int = 2000,
) -> dict[str, int]:
    """Atomically upsert clean players, player_seasons, and valuations into PostgreSQL,
    and delete any stale rows whose keys are no longer present in the clean dataset.

    Explicitly sets `updated_at=func.now()` on `ON CONFLICT DO UPDATE` because
    SQLAlchemy's `onupdate` only triggers on ORM unit-of-work updates.
    """
    clean_dir = clean_dir or settings.clean_data_dir
    owns_session = session is None
    db = session or SessionLocal()

    try:
        players_df = pd.read_parquet(clean_dir / "players.parquet")
        seasons_df = pd.read_parquet(clean_dir / "player_seasons.parquet")
        valuations_df = pd.read_parquet(clean_dir / "valuations.parquet")

        # 1. Upsert players & delete stale players not in current clean set
        player_records = _df_to_records(players_df)
        for i in range(0, len(player_records), batch_size):
            chunk = player_records[i : i + batch_size]
            stmt = pg_insert(Player).values(chunk)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_players_tm_player_id",
                set_={
                    "full_name": stmt.excluded.full_name,
                    "normalized_name": stmt.excluded.normalized_name,
                    "date_of_birth": stmt.excluded.date_of_birth,
                    "nationality": stmt.excluded.nationality,
                    "position_group": stmt.excluded.position_group,
                    "primary_position": stmt.excluded.primary_position,
                    "preferred_foot": stmt.excluded.preferred_foot,
                    "height_cm": stmt.excluded.height_cm,
                    "current_club": stmt.excluded.current_club,
                    "current_league_code": stmt.excluded.current_league_code,
                    "contract_expires_at": stmt.excluded.contract_expires_at,
                    "updated_at": func.now(),
                },
            )
            db.execute(stmt)
        db.flush()

        valid_tm_ids = {int(x) for x in players_df["tm_player_id"].tolist()}
        existing_players = db.execute(select(Player.id, Player.tm_player_id)).all()
        stale_player_ids = [int(pid) for pid, tm_id in existing_players if int(tm_id) not in valid_tm_ids]
        players_deleted_stale = 0
        if stale_player_ids:
            del_res = db.execute(delete(Player).where(Player.id.in_(stale_player_ids)))
            players_deleted_stale = int(del_res.rowcount or 0)
            db.flush()

        # Build tm_player_id -> internal players.id lookup map
        rows = db.execute(select(Player.tm_player_id, Player.id)).all()
        tm_to_id = {int(tm_id): int(pid) for tm_id, pid in rows}

        # 2. Upsert player_seasons & delete stale (player_id, season, tm_club_id) keys
        seasons_df["player_id"] = seasons_df["tm_player_id"].map(tm_to_id)
        seasons_cols = [
            "player_id",
            "season",
            "season_end_date",
            "tm_club_id",
            "club_name",
            "league_code",
            "appearances",
            "minutes_played",
            "goals",
            "assists",
            "yellow_cards",
            "red_cards",
            "clean_sheets",
            "goals_conceded",
            "goals_per90",
            "assists_per90",
            "xg_per90",
            "xa_per90",
            "prog_actions_per90",
            "has_advanced_stats",
        ]
        season_records = _df_to_records(seasons_df[seasons_cols])
        for i in range(0, len(season_records), batch_size):
            chunk = season_records[i : i + batch_size]
            stmt = pg_insert(PlayerSeason).values(chunk)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_player_seasons_player_season_club",
                set_={
                    "season_end_date": stmt.excluded.season_end_date,
                    "club_name": stmt.excluded.club_name,
                    "league_code": stmt.excluded.league_code,
                    "appearances": stmt.excluded.appearances,
                    "minutes_played": stmt.excluded.minutes_played,
                    "goals": stmt.excluded.goals,
                    "assists": stmt.excluded.assists,
                    "yellow_cards": stmt.excluded.yellow_cards,
                    "red_cards": stmt.excluded.red_cards,
                    "clean_sheets": stmt.excluded.clean_sheets,
                    "goals_conceded": stmt.excluded.goals_conceded,
                    "goals_per90": stmt.excluded.goals_per90,
                    "assists_per90": stmt.excluded.assists_per90,
                    "xg_per90": stmt.excluded.xg_per90,
                    "xa_per90": stmt.excluded.xa_per90,
                    "prog_actions_per90": stmt.excluded.prog_actions_per90,
                    "has_advanced_stats": stmt.excluded.has_advanced_stats,
                    "updated_at": func.now(),
                },
            )
            db.execute(stmt)
        db.flush()

        valid_season_keys = {
            (int(r["player_id"]), str(r["season"]), int(r["tm_club_id"]))
            for r in season_records
        }
        existing_seasons = db.execute(
            select(
                PlayerSeason.id,
                PlayerSeason.player_id,
                PlayerSeason.season,
                PlayerSeason.tm_club_id,
            )
        ).all()
        stale_season_ids = [
            int(sid)
            for sid, pid, s_str, cid in existing_seasons
            if (int(pid), str(s_str), int(cid)) not in valid_season_keys
        ]
        seasons_deleted_stale = 0
        if stale_season_ids:
            for i in range(0, len(stale_season_ids), batch_size):
                del_res = db.execute(
                    delete(PlayerSeason).where(
                        PlayerSeason.id.in_(stale_season_ids[i : i + batch_size])
                    )
                )
                seasons_deleted_stale += int(del_res.rowcount or 0)
            db.flush()

        # 3. Upsert valuations & delete stale (player_id, valuation_date) keys
        valuations_df["player_id"] = valuations_df["tm_player_id"].map(tm_to_id)
        val_cols = [
            "player_id",
            "valuation_date",
            "market_value_eur",
            "club_at_valuation",
            "league_at_valuation",
            "age_at_valuation",
        ]
        val_records = _df_to_records(valuations_df[val_cols])
        for i in range(0, len(val_records), batch_size):
            chunk = val_records[i : i + batch_size]
            stmt = pg_insert(Valuation).values(chunk)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_valuations_player_date",
                set_={
                    "market_value_eur": stmt.excluded.market_value_eur,
                    "club_at_valuation": stmt.excluded.club_at_valuation,
                    "league_at_valuation": stmt.excluded.league_at_valuation,
                    "age_at_valuation": stmt.excluded.age_at_valuation,
                },
            )
            db.execute(stmt)
        db.flush()

        valid_val_keys = {
            (
                int(r["player_id"]),
                r["valuation_date"].date()
                if hasattr(r["valuation_date"], "date") and callable(r["valuation_date"].date)
                else r["valuation_date"],
            )
            for r in val_records
        }
        existing_vals = db.execute(
            select(Valuation.id, Valuation.player_id, Valuation.valuation_date)
        ).all()
        stale_val_ids = [
            int(vid)
            for vid, pid, vdate in existing_vals
            if (int(pid), vdate) not in valid_val_keys
        ]
        vals_deleted_stale = 0
        if stale_val_ids:
            for i in range(0, len(stale_val_ids), batch_size):
                del_res = db.execute(
                    delete(Valuation).where(
                        Valuation.id.in_(stale_val_ids[i : i + batch_size])
                    )
                )
                vals_deleted_stale += int(del_res.rowcount or 0)
            db.flush()

        if owns_session:
            db.commit()

        return {
            "players_upserted": len(player_records),
            "players_deleted_stale": players_deleted_stale,
            "player_seasons_upserted": len(season_records),
            "player_seasons_deleted_stale": seasons_deleted_stale,
            "valuations_upserted": len(val_records),
            "valuations_deleted_stale": vals_deleted_stale,
        }
    except Exception:
        if owns_session:
            db.rollback()
        raise
    finally:
        if owns_session:
            db.close()


if __name__ == "__main__":
    import sys

    cleaning_summary = run_cleaning_pipeline()
    if "--load-db" in sys.argv:
        cleaning_summary["postgres_upsert"] = load_clean_to_postgres()
    print(json.dumps(cleaning_summary, indent=2))
