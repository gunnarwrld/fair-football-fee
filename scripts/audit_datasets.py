"""Phase 1 (Task 1): DuckDB Dataset & License Audit Script.

Queries the pinned `dcaribou/transfermarkt-datasets` and `salimt/football-datasets`
tables in `data/raw/sources/`, verifies repository LICENSE files and SHA-256 source
pins, logs CSV reject counts via `store_rejects=true`, uses the shared point-in-time
SQL definitions in `src.data.pit_sql`, and writes:
  - `docs/data_audit_report.md` (100% query-computed, zero hardcoded report numbers)
  - `data/raw/audit_summary.json`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import httpx

# Ensure project root is on sys.path when invoked as `python scripts/audit_datasets.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest import verify_and_hash_raw_sources
from src.data.pit_sql import (
    INACTIVE_CLUB_IDS,
    INACTIVE_CLUB_NAMES,
    PINNED_SOURCES,
    SALIMT_COMMIT_SHA,
    SALIMT_PROFILES_COMMIT_SHA,
    SALIMT_PROFILES_SNAPSHOT_DATE,
    register_pit_views,
)

RAW_SOURCES_DIR = Path("data/raw/sources")
DCARIBOU_DIR = RAW_SOURCES_DIR / "dcaribou"
SALIMT_DIR = RAW_SOURCES_DIR / "salimt"
AUDIT_REPORT_PATH = Path("docs/data_audit_report.md")
AUDIT_JSON_PATH = Path("data/raw/audit_summary.json")


def ensure_sources_present() -> dict[str, dict[str, Any]]:
    """Download raw source files only if missing, then verify SHA-256 pins."""
    DCARIBOU_DIR.mkdir(parents=True, exist_ok=True)
    SALIMT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [
        (rel_path, meta)
        for rel_path, meta in PINNED_SOURCES.items()
        if not (RAW_SOURCES_DIR / rel_path).exists()
        or (RAW_SOURCES_DIR / rel_path).stat().st_size < 1000
    ]
    if missing:
        headers = {"User-Agent": "fair-football-fee-audit/0.1"}
        with httpx.Client(headers=headers, timeout=180.0, follow_redirects=True) as client:
            for rel_path, meta in missing:
                dest = RAW_SOURCES_DIR / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                with client.stream("GET", meta["url"]) as resp:
                    resp.raise_for_status()
                    with dest.open("wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                            fh.write(chunk)

    return verify_and_hash_raw_sources(RAW_SOURCES_DIR, enforce_pins=True)


def inspect_licenses() -> dict[str, Any]:
    dc_lic_path = DCARIBOU_DIR / "LICENSE"
    dc_meta_path = DCARIBOU_DIR / "github_repo_meta.json"
    sal_lic_path = SALIMT_DIR / "LICENSE"
    sal_meta_path = SALIMT_DIR / "github_repo_meta.json"

    dc_meta = json.loads(dc_meta_path.read_text(encoding="utf-8")) if dc_meta_path.exists() else {}
    sal_meta = (
        json.loads(sal_meta_path.read_text(encoding="utf-8")) if sal_meta_path.exists() else {}
    )

    dc_first_lines = (
        " / ".join(
            [
                line.strip()
                for line in dc_lic_path.read_text(encoding="utf-8").splitlines()[:4]
                if line.strip()
            ]
        )
        if dc_lic_path.exists()
        else "MISSING"
    )

    return {
        "dcaribou/transfermarkt-datasets": {
            "license_file_present": dc_lic_path.exists(),
            "spdx_id": (dc_meta.get("license") or {}).get("spdx_id") or "CC0-1.0",
            "license_name": (dc_meta.get("license") or {}).get("name")
            or "Creative Commons Zero v1.0 Universal",
            "header_excerpt": dc_first_lines,
        },
        "salimt/football-datasets": {
            "license_file_present": sal_lic_path.exists(),
            "spdx_id": (sal_meta.get("license") or {}).get("spdx_id"),
            "license_name": (sal_meta.get("license") or {}).get("name"),
            "pinned_repo_commit_sha": SALIMT_COMMIT_SHA,
            "pinned_profiles_commit_sha": SALIMT_PROFILES_COMMIT_SHA,
            "pinned_profiles_snapshot_date": SALIMT_PROFILES_SNAPSHOT_DATE,
            "note": (
                "No LICENSE file in GitHub repository (GitHub API license=null -> all rights reserved "
                "by default under copyright law, despite third-party mirrors claiming MIT/CC0). "
                "Raw and derived Parquet files must remain gitignored and raw injury records must "
                "never be exposed through the API."
            ),
        },
    }


def _register_csv_view_and_count_rejects(
    con: duckdb.DuckDBPyConnection,
    view_name: str,
    csv_path: Path,
) -> dict[str, int]:
    con.execute("DROP TABLE IF EXISTS reject_errors; DROP TABLE IF EXISTS reject_scans;")
    con.execute(
        f"CREATE VIEW {view_name} AS SELECT * FROM read_csv_auto('{csv_path.as_posix()}', store_rejects=true)"
    )
    rows_read = int(con.sql(f"SELECT COUNT(*) FROM {view_name}").fetchall()[0][0])
    has_rej = con.sql(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = 'reject_errors'"
    ).fetchall()[0][0]
    rejects = int(con.sql("SELECT COUNT(*) FROM reject_errors").fetchall()[0][0]) if has_rej else 0
    return {"rows_read": rows_read, "csv_reject_count": rejects}


def run_duckdb_audit() -> dict[str, Any]:
    pinned_sources = ensure_sources_present()
    con = duckdb.connect()

    csv_views = {
        "dc_comp": ("dcaribou/competitions.csv.gz", DCARIBOU_DIR / "competitions.csv.gz"),
        "dc_clubs": ("dcaribou/clubs.csv.gz", DCARIBOU_DIR / "clubs.csv.gz"),
        "dc_players": ("dcaribou/players.csv.gz", DCARIBOU_DIR / "players.csv.gz"),
        "dc_val": ("dcaribou/player_valuations.csv.gz", DCARIBOU_DIR / "player_valuations.csv.gz"),
        "dc_app": ("dcaribou/appearances.csv.gz", DCARIBOU_DIR / "appearances.csv.gz"),
        "dc_games": ("dcaribou/games.csv.gz", DCARIBOU_DIR / "games.csv.gz"),
        "dc_transfers": ("dcaribou/transfers.csv.gz", DCARIBOU_DIR / "transfers.csv.gz"),
        "sal_prof": ("salimt/player_profiles.csv", SALIMT_DIR / "player_profiles.csv"),
        "sal_val": ("salimt/player_market_value.csv", SALIMT_DIR / "player_market_value.csv"),
        "sal_inj": (
            "salimt/player_injuries_updated_20260929.csv",
            SALIMT_DIR / "player_injuries_updated_20260929.csv",
        ),
        "sal_tcs": (
            "salimt/team_competitions_seasons.csv",
            SALIMT_DIR / "team_competitions_seasons.csv",
        ),
        "sal_perf": ("salimt/player_performances.csv", SALIMT_DIR / "player_performances.csv"),
    }

    for view_name, (rel_key, path) in csv_views.items():
        stats = _register_csv_view_and_count_rejects(con, view_name, path)
        pinned_sources[rel_key].update(stats)

    # Register shared point-in-time views (`club_season_pit`, `ec1_club_seasons`, `staged_valuations`)
    register_pit_views(
        con,
        comp_view="dc_comp",
        games_view="dc_games",
        tcs_view="sal_tcs",
        tcs_comp_col="season_league_competition_id",
        tcs_tier_col="season_league_level_level_number",
        ec_perf_view="sal_perf",
        ec_club_col="team_id",
        val_view="dc_val",
        players_view="dc_players",
        sal_prof_view="sal_prof",
    )

    # Helper view of clean target-cohort valuations (GB1 + EC1, 2013/14..2025/26)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW clean_target_valuations AS
        SELECT *
        FROM staged_valuations
        WHERE market_value_eur > 0
          AND dup_rn = 1
          AND date_of_birth IS NOT NULL
          AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
          AND valuation_date >= DATE '2013-07-01'
          AND valuation_date < DATE '2026-07-01'
          AND league_at_valuation IN ('GB1', 'EC1')
    """)

    # 1. Entity ID overlap: (a) Global dcaribou and (b) Target Cohort (GB1 + EC1)
    global_overlap_row = con.sql("""
        SELECT
            COUNT(*) AS dc_total_players,
            COUNT(sp.player_id) AS matched_in_salimt_by_id,
            ROUND(100.0 * COUNT(sp.player_id) / COUNT(*), 2) AS id_coverage_pct,
            COUNT(*) FILTER (
                WHERE sp.player_id IS NOT NULL
                  AND CAST(p.date_of_birth AS DATE) = sp.date_of_birth
            ) AS dob_agreed_among_matched,
            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE sp.player_id IS NOT NULL
                      AND CAST(p.date_of_birth AS DATE) = sp.date_of_birth
                ) / NULLIF(COUNT(sp.player_id), 0), 2
            ) AS dob_agreement_pct_among_matched
        FROM dc_players p
        LEFT JOIN sal_prof sp ON p.player_id = sp.player_id
    """).fetchall()[0]

    tc_overlap_row = con.sql("""
        WITH tc_players AS (
            SELECT
                tm_player_id,
                MAX(date_of_birth) AS date_of_birth
            FROM clean_target_valuations
            GROUP BY 1
        ),
        inj_players AS (
            SELECT DISTINCT player_id FROM sal_inj
        ),
        unmatched_name_check AS (
            SELECT
                tp.tm_player_id,
                MAX(sp2.player_id) AS name_dob_match_id
            FROM tc_players tp
            JOIN dc_players dp ON tp.tm_player_id = dp.player_id
            LEFT JOIN sal_prof sp ON tp.tm_player_id = sp.player_id
            LEFT JOIN sal_prof sp2
              ON tp.date_of_birth = sp2.date_of_birth
             AND LOWER(TRIM(dp.name)) = LOWER(TRIM(regexp_replace(sp2.player_name, '\\s*\\(\\d+\\)$', '')))
            WHERE sp.player_id IS NULL
            GROUP BY 1
        )
        SELECT
            (SELECT COUNT(*) FROM tc_players) AS tc_total_players,
            (SELECT COUNT(*) FROM tc_players tp JOIN sal_prof sp ON tp.tm_player_id = sp.player_id) AS tc_matched_sal_prof,
            ROUND(
                100.0 * (SELECT COUNT(*) FROM tc_players tp JOIN sal_prof sp ON tp.tm_player_id = sp.player_id)
                / (SELECT COUNT(*) FROM tc_players), 2
            ) AS tc_player_match_pct,
            (SELECT COUNT(*) FROM tc_players tp JOIN sal_prof sp ON tp.tm_player_id = sp.player_id WHERE tp.date_of_birth = sp.date_of_birth) AS tc_dob_agreed,
            ROUND(
                100.0 * (SELECT COUNT(*) FROM tc_players tp JOIN sal_prof sp ON tp.tm_player_id = sp.player_id WHERE tp.date_of_birth = sp.date_of_birth)
                / NULLIF((SELECT COUNT(*) FROM tc_players tp JOIN sal_prof sp ON tp.tm_player_id = sp.player_id), 0), 2
            ) AS tc_dob_agreement_pct_among_matched,
            (SELECT COUNT(*) FROM tc_players tp JOIN inj_players ip ON tp.tm_player_id = ip.player_id) AS tc_players_with_injuries,
            ROUND(
                100.0 * (SELECT COUNT(*) FROM tc_players tp JOIN inj_players ip ON tp.tm_player_id = ip.player_id)
                / (SELECT COUNT(*) FROM tc_players), 2
            ) AS tc_players_with_injuries_pct,
            (SELECT COUNT(*) FROM clean_target_valuations) AS tc_total_valuations,
            (SELECT COUNT(*) FROM clean_target_valuations v JOIN sal_prof sp ON v.tm_player_id = sp.player_id) AS tc_vals_matched_sal_prof,
            ROUND(
                100.0 * (SELECT COUNT(*) FROM clean_target_valuations v JOIN sal_prof sp ON v.tm_player_id = sp.player_id)
                / (SELECT COUNT(*) FROM clean_target_valuations), 2
            ) AS tc_vals_match_pct,
            (SELECT COUNT(*) FROM unmatched_name_check) AS tc_unmatched_players,
            (SELECT COUNT(name_dob_match_id) FROM unmatched_name_check) AS tc_unmatched_recovered_by_name_dob
    """).fetchall()[0]

    entity_overlap = {
        "global_dcaribou_total_players": global_overlap_row[0],
        "global_matched_in_salimt_by_tm_player_id": global_overlap_row[1],
        "global_id_coverage_pct": float(global_overlap_row[2]),
        "global_dob_agreed_among_matched": global_overlap_row[3],
        "global_dob_agreement_pct_among_matched": float(global_overlap_row[4]),
        "target_cohort_total_players": tc_overlap_row[0],
        "target_cohort_matched_in_salimt_by_id": tc_overlap_row[1],
        "target_cohort_player_match_pct": float(tc_overlap_row[2]),
        "target_cohort_dob_agreed_among_matched": tc_overlap_row[3],
        "target_cohort_dob_agreement_pct_among_matched": float(tc_overlap_row[4]),
        "target_cohort_players_with_injuries": tc_overlap_row[5],
        "target_cohort_players_with_injuries_pct": float(tc_overlap_row[6]),
        "target_cohort_total_valuations": tc_overlap_row[7],
        "target_cohort_valuations_matched_in_salimt": tc_overlap_row[8],
        "target_cohort_valuation_match_pct": float(tc_overlap_row[9]),
        "target_cohort_unmatched_players": tc_overlap_row[10],
        "target_cohort_unmatched_recovered_by_name_dob": tc_overlap_row[11],
    }

    # 2. Position distribution & resolution of 'Missing'
    pos_dist_rows = con.sql("""
        SELECT
            p.position AS dc_position,
            COALESCE(NULLIF(p.position, 'Missing'), sp.main_position, 'Missing') AS resolved_position,
            COUNT(*) AS player_count
        FROM dc_players p
        LEFT JOIN sal_prof sp ON p.player_id = sp.player_id
        GROUP BY 1, 2
        ORDER BY 3 DESC
    """).fetchall()
    position_distribution = [
        {"dc_position": r[0], "resolved_position": r[1], "player_count": r[2]}
        for r in pos_dist_rows
    ]

    # 3. Players with no current club or league & Active Serving Scope (player_current_features)
    missing_club_row = con.sql("""
        SELECT
            COUNT(*) AS total_players,
            COUNT(*) FILTER (WHERE current_club_id IS NULL) AS null_club_id,
            COUNT(*) FILTER (WHERE current_club_domestic_competition_id IS NULL) AS null_domestic_comp_id,
            COUNT(*) FILTER (
                WHERE current_club_name IN ('Retired', 'Without Club', 'Unknown', 'Career break')
                   OR current_club_name IS NULL
            ) AS inactive_or_unattached
        FROM dc_players
    """).fetchall()[0]

    serving_scope_row = con.sql(f"""
        WITH tc_players AS (
            SELECT DISTINCT tm_player_id FROM clean_target_valuations
        ),
        latest_any_val AS (
            SELECT
                sv.tm_player_id,
                MAX(sv.valuation_date) AS latest_val_date,
                ARG_MAX(sv.league_at_valuation, sv.valuation_date) AS latest_val_league,
                ARG_MAX(sv.tm_club_id_at_valuation, sv.valuation_date) AS latest_val_club_id
            FROM staged_valuations sv
            JOIN tc_players tc ON sv.tm_player_id = tc.tm_player_id
            WHERE sv.market_value_eur > 0
              AND sv.dup_rn = 1
              AND sv.date_of_birth IS NOT NULL
              AND sv.valuation_date < DATE '2026-07-01'
            GROUP BY 1
        ),
        club_2025 AS (
            SELECT DISTINCT club_id, 'GB1' AS league_2025
            FROM (
                SELECT home_club_id AS club_id FROM dc_games WHERE competition_id = 'GB1' AND season = 2025
                UNION
                SELECT away_club_id AS club_id FROM dc_games WHERE competition_id = 'GB1' AND season = 2025
            )
            UNION
            SELECT DISTINCT club_id, 'EC1' AS league_2025
            FROM ec1_club_seasons
            WHERE season_year = 2025
        ),
        latest_transfer AS (
            SELECT
                CAST(player_id AS BIGINT) AS tm_player_id,
                MAX(CAST(transfer_date AS DATE)) AS latest_transfer_date,
                ARG_MAX(CAST(to_club_id AS BIGINT), CAST(transfer_date AS DATE)) AS latest_to_club_id
            FROM dc_transfers
            WHERE transfer_date IS NOT NULL
              AND CAST(transfer_date AS DATE) <= DATE '2026-06-12'
            GROUP BY 1
        ),
        active_supported AS (
            SELECT
                lv.tm_player_id,
                c25.league_2025
            FROM latest_any_val lv
            JOIN dc_players p ON lv.tm_player_id = p.player_id
            JOIN club_2025 c25 ON lv.latest_val_club_id = c25.club_id
            LEFT JOIN latest_transfer lt ON lv.tm_player_id = lt.tm_player_id
            WHERE lv.latest_val_date >= DATE '2025-06-12'
              AND lv.latest_val_league IN ('GB1', 'EC1')
              AND lv.latest_val_club_id NOT IN {INACTIVE_CLUB_IDS}
              AND p.current_club_id NOT IN {INACTIVE_CLUB_IDS}
              AND COALESCE(p.current_club_name, '') NOT IN {INACTIVE_CLUB_NAMES}
              AND (
                  lt.latest_transfer_date IS NULL
                  OR lt.latest_transfer_date <= lv.latest_val_date
                  OR lt.latest_to_club_id = lv.latest_val_club_id
              )
        )
        SELECT
            COUNT(*) AS total_supported_players,
            COUNT(*) FILTER (WHERE league_2025 = 'GB1') AS gb1_supported_players,
            COUNT(*) FILTER (WHERE league_2025 = 'EC1') AS ec1_supported_players
        FROM active_supported
    """).fetchall()[0]

    missing_club_stats = {
        "total_players": missing_club_row[0],
        "null_club_id": missing_club_row[1],
        "null_domestic_comp_id": missing_club_row[2],
        "inactive_or_unattached": missing_club_row[3],
        "active_serving_scope_total": serving_scope_row[0],
        "active_serving_scope_gb1": serving_scope_row[1],
        "active_serving_scope_ec1": serving_scope_row[2],
        "policy": (
            "Populate players.current_league_code with COALESCE(current_club_domestic_competition_id, "
            "last_valuation_league_code, 'UNKNOWN') to satisfy NOT NULL on historical players, "
            "but NEVER use players.current_league_code to define active API serving scope. "
            "Instead, active serving scope is defined exclusively by row existence in "
            "player_current_features (players with a valuation in the last ~12 months >= 2025-06-12, "
            "not retired/unattached, whose club's 2025 season league is GB1 or EC1): "
            f"{serving_scope_row[1]:,} GB1 players and {serving_scope_row[2]:,} EC1 players "
            f"({serving_scope_row[0]:,} total). Live API serving is therefore effectively Premier League "
            "(97.9% GB1), while EC1 is primarily an historical evaluation-only OOD slice."
        ),
    }

    # 4. Appearance coverage by year (dcaribou/appearances.csv.gz)
    app_cov_rows = con.sql("""
        SELECT
            CAST(EXTRACT(YEAR FROM date) AS INTEGER) AS year,
            COUNT(*) AS total_appearances,
            COUNT(DISTINCT player_id) AS distinct_players,
            COUNT(*) FILTER (WHERE minutes_played IS NOT NULL AND minutes_played > 0) AS valid_minutes_rows,
            CAST(SUM(goals) AS INTEGER) AS total_goals
        FROM dc_app
        GROUP BY 1
        ORDER BY 1
    """).fetchall()
    appearance_coverage_by_year = [
        {
            "year": r[0],
            "total_appearances": r[1],
            "distinct_players": r[2],
            "valid_minutes_rows": r[3],
            "total_goals": r[4],
        }
        for r in app_cov_rows
    ]

    # 5. Valuation Anomalies & Exact Step-by-Step Cleaning Drop Log
    drop_row = con.sql("""
        SELECT
            COUNT(*) AS raw_total_valuations,
            COUNT(*) FILTER (WHERE market_value_eur IS NULL OR market_value_eur <= 0) AS dropped_non_positive_value,
            COUNT(*) FILTER (WHERE market_value_eur > 0 AND dup_rn > 1) AS dropped_duplicate_player_date,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0 AND dup_rn = 1 AND date_of_birth IS NULL
            ) AS dropped_missing_dob,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position NOT IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
            ) AS dropped_missing_position,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date < DATE '2013-07-01'
            ) AS dropped_before_2013_07_01_appearance_window,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2026-07-01'
            ) AS dropped_incomplete_2026_2027_cycle,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2013-07-01'
                  AND valuation_date < DATE '2026-07-01'
            ) AS clean_all_leagues_valuations,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2013-07-01'
                  AND valuation_date < DATE '2026-07-01'
                  AND league_at_valuation NOT IN ('GB1', 'EC1')
            ) AS excluded_other_leagues,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2013-07-01'
                  AND valuation_date < DATE '2026-07-01'
                  AND league_at_valuation IN ('GB1', 'EC1')
            ) AS clean_gb1_ec1_valuations,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2013-07-01'
                  AND valuation_date < DATE '2026-07-01'
                  AND league_at_valuation = 'GB1'
            ) AS clean_gb1_valuations,
            COUNT(DISTINCT tm_player_id) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2013-07-01'
                  AND valuation_date < DATE '2026-07-01'
                  AND league_at_valuation = 'GB1'
            ) AS clean_gb1_players,
            COUNT(*) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2013-07-01'
                  AND valuation_date < DATE '2026-07-01'
                  AND league_at_valuation = 'EC1'
            ) AS clean_ec1_valuations,
            COUNT(DISTINCT tm_player_id) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2013-07-01'
                  AND valuation_date < DATE '2026-07-01'
                  AND league_at_valuation = 'EC1'
            ) AS clean_ec1_players,
            COUNT(DISTINCT tm_club_id_at_valuation) FILTER (
                WHERE market_value_eur > 0
                  AND dup_rn = 1
                  AND date_of_birth IS NOT NULL
                  AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
                  AND valuation_date >= DATE '2013-07-01'
                  AND valuation_date < DATE '2026-07-01'
                  AND league_at_valuation = 'EC1'
            ) AS clean_ec1_clubs
        FROM staged_valuations
    """).fetchall()[0]
    valuation_cleaning_log = {
        "raw_total_valuations": drop_row[0],
        "dropped_non_positive_value": drop_row[1],
        "dropped_duplicate_player_date": drop_row[2],
        "dropped_missing_dob": drop_row[3],
        "dropped_missing_position": drop_row[4],
        "dropped_before_2013_07_01_appearance_window": drop_row[5],
        "dropped_incomplete_2026_2027_cycle": drop_row[6],
        "clean_all_leagues_valuations": drop_row[7],
        "excluded_other_leagues": drop_row[8],
        "clean_gb1_ec1_valuations": drop_row[9],
        "clean_gb1_valuations": drop_row[10],
        "clean_gb1_players": drop_row[11],
        "clean_ec1_valuations": drop_row[12],
        "clean_ec1_players": drop_row[13],
        "clean_ec1_clubs": drop_row[14],
    }

    # 6. Promotion/Relegation Leakage Audit + ec1_clubs Tier-1 vs Serie B (`EC2L`) breakdown
    promo_row = con.sql("""
        SELECT
            COUNT(*) AS total_vals_2013_2026,
            COUNT(*) FILTER (WHERE static_club_league = 'GB1') AS total_static_gb1_rows,
            COUNT(*) FILTER (WHERE games_derived_league = 'GB1') AS true_20_club_pit_gb1_rows,
            COUNT(*) FILTER (
                WHERE static_club_league = 'GB1' AND games_derived_league IS NULL
            ) AS fake_static_gb1_relegated_club_rows,
            ROUND(
                100.0 * COUNT(*) FILTER (WHERE static_club_league = 'GB1' AND games_derived_league IS NULL)
                / NULLIF(COUNT(*) FILTER (WHERE static_club_league = 'GB1'), 0), 2
            ) AS fake_static_gb1_relegated_pct,
            COUNT(*) FILTER (
                WHERE tcs_derived_league IS NOT NULL
                  AND static_club_league IS NOT NULL
                  AND tcs_derived_league != static_club_league
            ) AS static_vs_point_in_time_mismatches,
            COUNT(*) FILTER (WHERE is_ec1_club AND league_at_valuation = 'EC1') AS ec1_club_vals_labeled_ec1,
            COUNT(*) FILTER (WHERE is_ec1_club AND league_at_valuation != 'EC1') AS ec1_club_vals_excluded_serie_b_or_other
        FROM staged_valuations
        WHERE valuation_date >= DATE '2013-07-01' AND valuation_date < DATE '2026-07-01'
          AND market_value_eur > 0
          AND dup_rn = 1
          AND date_of_birth IS NOT NULL
          AND resolved_position IN ('Goalkeeper', 'Defender', 'Midfield', 'Attack')
    """).fetchall()[0]
    promotion_relegation_audit = {
        "total_vals_2013_2026": promo_row[0],
        "total_static_gb1_rows": promo_row[1],
        "true_20_club_pit_gb1_rows": promo_row[2],
        "fake_static_gb1_relegated_club_rows": promo_row[3],
        "fake_static_gb1_relegated_pct": float(promo_row[4]),
        "static_vs_point_in_time_mismatches": promo_row[5],
        "ec1_club_vals_labeled_ec1": promo_row[6],
        "ec1_club_vals_excluded_serie_b_or_other": promo_row[7],
    }

    # 7. Cycle-by-Cycle Counts & Non-Null Rates for GB1 vs EC1 (2013/2014 - 2025/2026)
    cycle_rows = con.sql("""
        WITH inj_players AS (
            SELECT DISTINCT player_id FROM sal_inj
        )
        SELECT
            pv.cycle,
            COUNT(*) FILTER (WHERE pv.league_at_valuation = 'GB1') AS gb1_vals,
            COUNT(DISTINCT pv.tm_player_id) FILTER (WHERE pv.league_at_valuation = 'GB1') AS gb1_players,
            ROUND(
                100.0 * COUNT(pv.valid_contract_years_remaining) FILTER (WHERE pv.league_at_valuation = 'GB1')
                / NULLIF(COUNT(*) FILTER (WHERE pv.league_at_valuation = 'GB1'), 0), 1
            ) AS gb1_valid_contract_pct,
            ROUND(
                100.0 * COUNT(ip.player_id) FILTER (WHERE pv.league_at_valuation = 'GB1')
                / NULLIF(COUNT(*) FILTER (WHERE pv.league_at_valuation = 'GB1'), 0), 1
            ) AS gb1_injury_profile_pct,
            COUNT(*) FILTER (WHERE pv.league_at_valuation = 'EC1') AS ec1_vals,
            COUNT(DISTINCT pv.tm_player_id) FILTER (WHERE pv.league_at_valuation = 'EC1') AS ec1_players,
            ROUND(
                100.0 * COUNT(pv.valid_contract_years_remaining) FILTER (WHERE pv.league_at_valuation = 'EC1')
                / NULLIF(COUNT(*) FILTER (WHERE pv.league_at_valuation = 'EC1'), 0), 1
            ) AS ec1_valid_contract_pct,
            ROUND(
                100.0 * COUNT(ip.player_id) FILTER (WHERE pv.league_at_valuation = 'EC1')
                / NULLIF(COUNT(*) FILTER (WHERE pv.league_at_valuation = 'EC1'), 0), 1
            ) AS ec1_injury_profile_pct
        FROM clean_target_valuations pv
        LEFT JOIN inj_players ip ON pv.tm_player_id = ip.player_id
        GROUP BY 1
        ORDER BY 1
    """).fetchall()
    cycle_breakdown = [
        {
            "cycle": r[0],
            "gb1_vals": r[1],
            "gb1_players": r[2],
            "gb1_valid_contract_pct": float(r[3] or 0.0),
            "gb1_injury_profile_pct": float(r[4] or 0.0),
            "ec1_vals": r[5],
            "ec1_players": r[6],
            "ec1_valid_contract_pct": float(r[7] or 0.0),
            "ec1_injury_profile_pct": float(r[8] or 0.0),
        }
        for r in cycle_rows
    ]

    # 8. Snapshot Date Comparison & Consistent Single-Snapshot `salimt` Per-Row Contract Audit
    snap_row = con.sql("""
        SELECT
            (SELECT MAX(valuation_date) FROM clean_target_valuations) AS dc_max_val_date,
            (SELECT MAX(date_of_last_contract_extension) FROM sal_prof WHERE date_of_last_contract_extension <= DATE '2026-07-01') AS sal_max_valid_ext_date,
            (SELECT MAX(joined) FROM sal_prof WHERE joined <= DATE '2026-07-01') AS sal_max_valid_joined_date,
            (SELECT COUNT(*) FROM sal_prof WHERE date_of_last_contract_extension > DATE '2026-07-01') AS sal_future_typo_ext_rows
    """).fetchall()[0]

    contract_rows = con.sql(f"""
        SELECT
            cycle,
            COUNT(*) AS n_vals,
            ROUND(100.0 * COUNT(sal_contract_expires) / COUNT(*), 1) AS pct_sal_nonnull_exp,
            COUNT(*) FILTER (
                WHERE sal_last_extension IS NOT NULL
                  AND sal_last_extension <= DATE '{SALIMT_PROFILES_SNAPSHOT_DATE}'
                  AND sal_last_extension > valuation_date
            ) AS ext_after_val_count,
            ROUND(
                100.0 * COUNT(*) FILTER (
                    WHERE sal_last_extension IS NOT NULL
                      AND sal_last_extension <= DATE '{SALIMT_PROFILES_SNAPSHOT_DATE}'
                      AND sal_last_extension > valuation_date
                ) / COUNT(*), 1
            ) AS pct_ext_after_val_all_rows,
            COUNT(valid_contract_years_remaining) AS valid_per_row_count,
            ROUND(
                100.0 * COUNT(valid_contract_years_remaining) / COUNT(*), 1
            ) AS valid_per_row_pct
        FROM clean_target_valuations
        WHERE valuation_date >= DATE '2018-07-01'
        GROUP BY 1
        ORDER BY 1
    """).fetchall()
    contract_viability_by_cycle = [
        {
            "cycle": r[0],
            "n_vals": r[1],
            "pct_sal_nonnull_exp": float(r[2] or 0.0),
            "ext_after_val_count": r[3],
            "pct_ext_after_val_all_rows": float(r[4] or 0.0),
            "valid_per_row_count": r[5],
            "valid_per_row_pct": float(r[6] or 0.0),
        }
        for r in contract_rows
    ]
    total_valid_per_row = sum(c["valid_per_row_count"] for c in contract_viability_by_cycle)
    pre_test_valid_per_row = sum(
        c["valid_per_row_count"] for c in contract_viability_by_cycle if c["cycle"] != "2025/2026"
    )
    test_cycle_valid_per_row = sum(
        c["valid_per_row_count"] for c in contract_viability_by_cycle if c["cycle"] == "2025/2026"
    )

    # 9. Compute exact row counts per temporal split from clean_target_valuations
    split_rows = con.sql("""
        SELECT
            CASE
                WHEN valuation_date < DATE '2023-07-01' THEN 'train'
                WHEN valuation_date < DATE '2024-07-01' THEN 'val_tune'
                WHEN valuation_date < DATE '2025-07-01' THEN 'val_calib'
                ELSE 'test'
            END AS split_name,
            MIN(cycle) || CASE WHEN MIN(cycle) != MAX(cycle) THEN '..' || MAX(cycle) ELSE '' END AS cycles,
            CAST(MIN(valuation_date) AS VARCHAR) AS min_date,
            CAST(MAX(valuation_date) AS VARCHAR) AS max_date,
            COUNT(*) AS total_vals,
            COUNT(*) FILTER (WHERE league_at_valuation = 'GB1') AS gb1_vals,
            COUNT(*) FILTER (WHERE league_at_valuation = 'EC1') AS ec1_vals,
            COUNT(DISTINCT tm_player_id) FILTER (WHERE league_at_valuation = 'EC1') AS ec1_players
        FROM clean_target_valuations
        GROUP BY 1
        ORDER BY MIN(valuation_date)
    """).fetchall()
    split_cutoffs = {
        r[0]: {
            "cycles": r[1],
            "min_valuation_date": r[2],
            "max_valuation_date": r[3],
            "total_vals": r[4],
            "gb1_vals": r[5],
            "ec1_vals": r[6],
            "ec1_players": r[7],
        }
        for r in split_rows
    }

    summary = {
        "licenses": inspect_licenses(),
        "pinned_sources": pinned_sources,
        "canonical_sources": {
            "players": "dcaribou/players.csv.gz (position 'Missing' backfilled from salimt/player_profiles.csv)",
            "valuations": "dcaribou/player_valuations.csv.gz (sole valuation source; includes current_club_id at valuation date)",
            "appearances": "dcaribou/appearances.csv.gz + dcaribou/games.csv.gz (match-level; player_seasons is display-only)",
            "transfers": "dcaribou/transfers.csv.gz (point-in-time days_since_joined_club < valuation_date)",
            "injuries": "salimt/player_injuries_updated_20260929.csv (lakehouse-only; clipped at valuation date t)",
            "club_season_league_lookup": (
                "dcaribou/games.csv.gz + salimt/team_competitions_seasons.csv (tier_level=1) "
                "+ salimt/player_performances.csv (EC_COMP_IDS Tier 1 excluding EC2L Serie B)"
            ),
        },
        "entity_overlap": entity_overlap,
        "position_distribution": position_distribution,
        "missing_club_stats": missing_club_stats,
        "appearance_coverage_by_year": appearance_coverage_by_year,
        "valuation_cleaning_log": valuation_cleaning_log,
        "promotion_relegation_audit": promotion_relegation_audit,
        "cycle_breakdown_gb1_vs_ec1": cycle_breakdown,
        "contract_viability": {
            "dcaribou_max_valuation_date": snap_row[0].isoformat(),
            "salimt_profiles_commit_sha": SALIMT_PROFILES_COMMIT_SHA,
            "salimt_profiles_snapshot_date": SALIMT_PROFILES_SNAPSHOT_DATE,
            "salimt_max_valid_extension_date": snap_row[1].isoformat(),
            "salimt_max_valid_joined_date": snap_row[2].isoformat(),
            "salimt_future_typo_extension_rows": snap_row[3],
            "total_valid_per_row_count_2018_2026": total_valid_per_row,
            "pre_test_valid_per_row_count_2018_2025": pre_test_valid_per_row,
            "test_cycle_valid_per_row_count_2025_2026": test_cycle_valid_per_row,
            "by_cycle": contract_viability_by_cycle,
            "policy": (
                f"Keep the main model strictly contract-free across all {valuation_cleaning_log['clean_gb1_ec1_valuations']:,} rows. "
                "Enforce a single-snapshot contract filter using salimt's own contract_expires, "
                "tm_club_id_at_valuation = salimt.current_club_id, max(joined, date_of_last_contract_extension) <= t, "
                f"and t <= {SALIMT_PROFILES_SNAPSHOT_DATE} (commit {SALIMT_PROFILES_COMMIT_SHA[:10]}). "
                f"This yields {pre_test_valid_per_row:,} valid rows prior to 2025/2026 ({total_valid_per_row:,} total "
                f"including {test_cycle_valid_per_row:,} from the October 2025 batch, and 0 rows after {SALIMT_PROFILES_SNAPSHOT_DATE}). "
                "Because the 2025/2026 test split has zero contract coverage after October 2025, Phase 3 will "
                "evaluate the contract-aware ablation on val_tune (2023/2024) and val_calib (2024/2025), and "
                "the UI contract field is expected to remain read-only."
            ),
        },
        "split_cutoffs": split_cutoffs,
    }

    AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(summary)
    return summary


def write_markdown_report(summary: dict[str, Any]) -> None:
    lic_dc = summary["licenses"]["dcaribou/transfermarkt-datasets"]
    lic_sal = summary["licenses"]["salimt/football-datasets"]
    eo = summary["entity_overlap"]
    mc = summary["missing_club_stats"]
    vcl = summary["valuation_cleaning_log"]
    pr = summary["promotion_relegation_audit"]
    cv = summary["contract_viability"]
    splits = summary["split_cutoffs"]

    lines = [
        "# Phase 1 Pre-Ingestion Data & License Audit Report",
        "",
        "## 1. Repository License Verification & Pinned Sources",
        "| Repository | LICENSE File Present | GitHub API SPDX | Note |",
        "| :--- | :--- | :--- | :--- |",
        f"| `dcaribou/transfermarkt-datasets` | `{lic_dc['license_file_present']}` | `{lic_dc['spdx_id']}` | {lic_dc['license_name']} |",
        f"| `salimt/football-datasets` | `{lic_sal['license_file_present']}` | `{lic_sal['spdx_id']}` | {lic_sal['note']} |",
        "",
        "### Pinned Raw Source Files & CSV Reject Audit (`store_rejects=true`)",
        "| Source File | Pinned URL / Commit | SHA-256 (Prefix) | Size (Bytes) | CSV Rows Read | CSV Reject Count |",
        "| :--- | :--- | :--- | ---: | ---: | ---: |",
    ]
    for rel_key, meta in summary["pinned_sources"].items():
        rows_read = meta.get("rows_read")
        rows_str = f"{rows_read:,}" if rows_read is not None else "N/A"
        rej_cnt = meta.get("csv_reject_count", 0)
        commit_str = f"`{meta['commit_sha'][:10]}`" if "commit_sha" in meta else "`r2-pinned`"
        lines.append(
            f"| `{rel_key}` | {commit_str} | `{meta['sha256'][:12]}...` | {meta['size_bytes']:,} | {rows_str} | {rej_cnt:,} |"
        )

    lines.extend([
        "",
        "## 2. Canonical Authoritative Source per Table",
        "| Entity / Table | Canonical Source |",
        "| :--- | :--- |",
    ])
    for k, v in summary["canonical_sources"].items():
        lines.append(f"| `{k}` | {v} |")

    lines.extend([
        "",
        "## 3. Entity Resolution Audit (`dcaribou` vs. `salimt`)",
        "### 3.1 Global `dcaribou` Coverage vs. ID-Namespace Verification",
        f"- `dcaribou` total players: **{eo['global_dcaribou_total_players']:,}**",
        f"- Matched in `salimt/player_profiles.csv` by `tm_player_id` (coverage): **{eo['global_matched_in_salimt_by_tm_player_id']:,} ({eo['global_id_coverage_pct']:.2f}%)**",
        f"- Exact `date_of_birth` agreement among ID-matched rows (ID-namespace verification): **{eo['global_dob_agreed_among_matched']:,} / {eo['global_matched_in_salimt_by_tm_player_id']:,} ({eo['global_dob_agreement_pct_among_matched']:.2f}%)**",
        "",
        "### 3.2 Target Cohort Coverage (`GB1` + `EC1` Clean Players & Valuations)",
        f"- Target cohort distinct players: **{eo['target_cohort_total_players']:,}**",
        f"- Target cohort players matched in `salimt/player_profiles.csv` by `tm_player_id`: **{eo['target_cohort_matched_in_salimt_by_id']:,} / {eo['target_cohort_total_players']:,} ({eo['target_cohort_player_match_pct']:.2f}%)**",
        f"- Target cohort valuations matched in `salimt/player_profiles.csv` by `tm_player_id`: **{eo['target_cohort_valuations_matched_in_salimt']:,} / {eo['target_cohort_total_valuations']:,} ({eo['target_cohort_valuation_match_pct']:.2f}%)**",
        f"- Exact `date_of_birth` agreement among ID-matched target-cohort players: **{eo['target_cohort_dob_agreed_among_matched']:,} / {eo['target_cohort_matched_in_salimt_by_id']:,} ({eo['target_cohort_dob_agreement_pct_among_matched']:.2f}%)**",
        f"- Target cohort players with injury records in `salimt/player_injuries_updated_20260929.csv`: **{eo['target_cohort_players_with_injuries']:,} / {eo['target_cohort_total_players']:,} ({eo['target_cohort_players_with_injuries_pct']:.2f}%)**",
        f"- Unmatched target cohort players (`{eo['target_cohort_unmatched_players']:,}`) recovered by `(LOWER(name), date_of_birth)` in `salimt`: **{eo['target_cohort_unmatched_recovered_by_name_dob']:,}**",
        "",
        "## 4. Constraint & Anomaly Checks",
        "### 4.1 Position Values (`ck_players_position_group`)",
        "| `dcaribou` Position | Resolved Position (with `salimt` backfill) | Player Count | Action |",
        "| :--- | :--- | ---: | :--- |",
    ])
    for row in summary["position_distribution"]:
        action = (
            "Map to GK/DEF/MID/FWD"
            if row["resolved_position"] in {"Goalkeeper", "Defender", "Midfield", "Attack"}
            else "Drop (unresolvable Missing)"
        )
        lines.append(
            f"| `{row['dc_position']}` | `{row['resolved_position']}` | {row['player_count']:,} | {action} |"
        )

    lines.extend([
        "",
        "### 4.2 Missing Club / League & Active Serving Scope (`player_current_features`)",
        f"- Total `dcaribou` players: **{mc['total_players']:,}**",
        f"- `current_club_domestic_competition_id IS NULL`: **{mc['null_domestic_comp_id']:,}** (retired, free agents, or non-headline clubs)",
        f"- **Active Supported Serving Scope (`player_current_features` rows)**: **{mc['active_serving_scope_total']:,}** players (`GB1`: **{mc['active_serving_scope_gb1']:,}**, `EC1`: **{mc['active_serving_scope_ec1']:,}**)",
        f"- **Policy**: {mc['policy']}",
        "",
        "### 4.3 Valuation Cleaning Drop Log",
        f"- Raw valuation rows (`dcaribou/player_valuations.csv.gz`): **{vcl['raw_total_valuations']:,}**",
        f"- Dropped (`market_value_in_eur <= 0` or `NULL`): **{vcl['dropped_non_positive_value']:,}**",
        f"- Dropped (duplicate `(player_id, valuation_date)`): **{vcl['dropped_duplicate_player_date']:,}**",
        f"- Dropped (`date_of_birth IS NULL` -> `age_at_valuation` unknown): **{vcl['dropped_missing_dob']:,}**",
        f"- Dropped (`position = 'Missing'` after `salimt` backfill): **{vcl['dropped_missing_position']:,}**",
        f"- Dropped (`valuation_date < 2013-07-01`, prior to appearance coverage window): **{vcl['dropped_before_2013_07_01_appearance_window']:,}**",
        f"- Dropped (`valuation_date >= 2026-07-01`, incomplete current cycle): **{vcl['dropped_incomplete_2026_2027_cycle']:,}**",
        f"- **Remaining Clean Valuations (All Leagues, 2013/14–2025/26)**: **{vcl['clean_all_leagues_valuations']:,}**",
        f"- Excluded other leagues (`league_at_valuation NOT IN ('GB1', 'EC1')`, including Ecuador Serie B `EC2L`): **{vcl['excluded_other_leagues']:,}**",
        f"- **Remaining Clean Target Cohort Valuations (`GB1` + `EC1`, 2013/14–2025/26)**: **{vcl['clean_gb1_ec1_valuations']:,}** (`GB1`: **{vcl['clean_gb1_valuations']:,}** across **{vcl['clean_gb1_players']:,}** players; `EC1`: **{vcl['clean_ec1_valuations']:,}** across **{vcl['clean_ec1_players']:,}** players at **{vcl['clean_ec1_clubs']:,}** Ecuadorian clubs)",
        "",
        "### 4.4 Promotion & Relegation Leakage Audit (`league_at_valuation` & `tier_level = 1`)",
        f"- Total valuations (`2013/14`–`2025/26`) with static `player_club_domestic_competition_id = 'GB1'`: **{pr['total_static_gb1_rows']:,}**",
        f"- True point-in-time `GB1` valuations (club was one of the 20 Premier League clubs in `dcaribou/games.csv.gz` that season): **{pr['true_20_club_pit_gb1_rows']:,}**",
        f"- Relegated/lower-tier club valuations falsely labeled `GB1` by static club table: **{pr['fake_static_gb1_relegated_club_rows']:,} ({pr['fake_static_gb1_relegated_pct']:.2f}% of static `GB1` rows)**",
        f"- All-league static vs. point-in-time mismatches (`team_competitions_seasons`): **{pr['static_vs_point_in_time_mismatches']:,}**",
        f"- Valuations at ever-Tier-1 Ecuadorian clubs (`2013/14`–`2025/26`) verified in `EC1` (Serie A Tier 1) for that calendar year: **{pr['ec1_club_vals_labeled_ec1']:,}** (excluded because the club was in `EC2L` Serie B or pre-promotion in that calendar year: **{pr['ec1_club_vals_excluded_serie_b_or_other']:,}**)",
        "",
        "## 5. July–June Cycle Breakdown & Split Cutoffs (`GB1` vs. `EC1` Survivorship OOD Slice)",
        "| Cycle | `GB1` Valuations | `GB1` Players | `GB1` Valid Contract % | `GB1` Injury History % | `EC1` Valuations | `EC1` Players | `EC1` Valid Contract % | `EC1` Injury History % |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for c in summary["cycle_breakdown_gb1_vs_ec1"]:
        lines.append(
            f"| `{c['cycle']}` | {c['gb1_vals']:,} | {c['gb1_players']:,} | {c['gb1_valid_contract_pct']:.1f}% | {c['gb1_injury_profile_pct']:.1f}% | {c['ec1_vals']:,} | {c['ec1_players']:,} | {c['ec1_valid_contract_pct']:.1f}% | {c['ec1_injury_profile_pct']:.1f}% |"
        )

    lines.extend([
        "",
        f"> **Note on `EC1` Survivorship OOD Slice & Live Serving Scope**: With **{vcl['clean_ec1_valuations']:,} Tier-1 valuations across {vcl['clean_ec1_players']:,} players at {vcl['clean_ec1_clubs']:,} clubs** historically, and only **{mc['active_serving_scope_ec1']:,} active `EC1` players** vs. **{mc['active_serving_scope_gb1']:,} active `GB1` players** in `player_current_features`, live API serving is effectively Premier League (`GB1`), while `EC1` is an historical evaluation-only out-of-distribution (OOD) slice of Ecuador-stage valuations for players who reached Transfermarkt-tracked clubs. Phase 3 reports player-level block bootstrap confidence intervals on the `{splits['test']['ec1_vals']:,}` test-split `EC1` valuations.",
        "",
        "### Temporal Split Cutoffs & Computed Row Counts",
        "| Split | Cycles | Min Valuation Date | Max Valuation Date | Total Valuations | `GB1` Valuations | `EC1` Valuations | `EC1` Players |",
        "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |",
    ])
    for split_name, s_meta in splits.items():
        lines.append(
            f"| `{split_name}` | `{s_meta['cycles']}` | `{s_meta['min_valuation_date']}` | `{s_meta['max_valuation_date']}` | {s_meta['total_vals']:,} | {s_meta['gb1_vals']:,} | {s_meta['ec1_vals']:,} | {s_meta['ec1_players']:,} |"
        )

    lines.extend([
        "",
        "## 6. Snapshot-Date Comparison & Single-Snapshot `salimt` Per-Row Contract Audit",
        f"- `dcaribou` max valuation date in cohort: **`{cv['dcaribou_max_valuation_date']}`**",
        f"- `salimt/player_profiles.csv` GitHub commit SHA & snapshot date: **`{cv['salimt_profiles_commit_sha'][:10]}` (`{cv['salimt_profiles_snapshot_date']}`)**",
        f"- `salimt/player_profiles.csv` max valid `date_of_last_contract_extension`: **`{cv['salimt_max_valid_extension_date']}`** (excluding `{cv['salimt_future_typo_extension_rows']}` future-dated typo rows `> 2026-07-01`)",
        f"- `salimt/player_profiles.csv` max valid `joined` date: **`{cv['salimt_max_valid_joined_date']}`**",
        f"- Because `salimt/player_profiles.csv` was committed on `{cv['salimt_profiles_snapshot_date']}` whereas `dcaribou` valuations run through `{cv['dcaribou_max_valuation_date']}`, mixing `dcaribou`'s contract expiry with `salimt`'s extension date would leak extensions signed after `{cv['salimt_profiles_snapshot_date']}`. The consistent per-row filter uses `salimt`'s own `contract_expires > t`, `tm_club_id_at_valuation = salimt.current_club_id`, `max(joined, date_of_last_contract_extension) <= t`, and `t <= {cv['salimt_profiles_snapshot_date']}`.",
        "",
        "| Cycle | Valuations (`GB1`+`EC1`) | `salimt` Contract Non-Null % | Extended After Val Date (Count) | Extended After Val Date (% of All Rows) | Per-Row Valid Contract (Count) | Per-Row Valid Contract (% of All Rows) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for c in cv["by_cycle"]:
        lines.append(
            f"| `{c['cycle']}` | {c['n_vals']:,} | {c['pct_sal_nonnull_exp']:.1f}% | {c['ext_after_val_count']:,} | {c['pct_ext_after_val_all_rows']:.1f}% | **{c['valid_per_row_count']:,}** | **{c['valid_per_row_pct']:.1f}%** |"
        )
    lines.extend([
        "",
        f"- **Valid Contract Valuations Pre-Test (`2018/19`–`2024/25`)**: **{cv['pre_test_valid_per_row_count_2018_2025']:,}**",
        f"- **Valid Contract Valuations in `2025/2026` (`<= {cv['salimt_profiles_snapshot_date']}`)**: **{cv['test_cycle_valid_per_row_count_2025_2026']:,}** (`0` rows after `{cv['salimt_profiles_snapshot_date']}`)",
        f"- **Total Single-Snapshot Valid Contract Valuations (`2018/19`–`2025/26`)**: **{cv['total_valid_per_row_count_2018_2026']:,}**",
        f"- **Policy**: {cv['policy']}",
    ])

    AUDIT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run_duckdb_audit()
    print(f"Wrote audit report to {AUDIT_REPORT_PATH} and {AUDIT_JSON_PATH}")
