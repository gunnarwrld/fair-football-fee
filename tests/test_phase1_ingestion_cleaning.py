import json
from datetime import date
from pathlib import Path

import duckdb
import pytest

from src.data.clean import normalize_player_name, run_cleaning_pipeline
from src.data.pit_sql import PINNED_SOURCES, SALIMT_PROFILES_SNAPSHOT_DATE

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")


def test_normalize_player_name_strips_diacritics_and_punctuation() -> None:
    assert normalize_player_name("Piero Martín Hincapié Reyna") == "piero martin hincapie reyna"
    assert normalize_player_name("Moisés Caicedo") == "moises caicedo"
    assert normalize_player_name("  João   Pedro  ") == "joao pedro"


def test_audit_report_and_summary_invariants() -> None:
    audit_json = RAW_DIR / "audit_summary.json"
    audit_md = Path("docs/data_audit_report.md")
    assert audit_json.exists()
    assert audit_md.exists()

    summary = json.loads(audit_json.read_text(encoding="utf-8"))
    assert summary["licenses"]["dcaribou/transfermarkt-datasets"]["spdx_id"] == "CC0-1.0"
    assert summary["licenses"]["salimt/football-datasets"]["spdx_id"] is None

    # Pinned sources (including commit SHA for salimt) and CSV reject counts must be logged
    assert set(summary["pinned_sources"].keys()) == set(PINNED_SOURCES.keys())
    for rel_key, src_meta in summary["pinned_sources"].items():
        assert len(src_meta["sha256"]) == 64
        assert src_meta["pin_verified"] is True
        assert "csv_reject_count" in src_meta
        assert src_meta["csv_reject_count"] >= 0
        if rel_key.startswith("salimt/"):
            assert "commit_sha" in src_meta
            assert "/main/" not in src_meta["url"]

    # Target cohort match rate & DOB verification invariants
    eo = summary["entity_overlap"]
    assert eo["target_cohort_player_match_pct"] > 85.0
    assert eo["target_cohort_valuation_match_pct"] > 90.0
    assert eo["target_cohort_dob_agreement_pct_among_matched"] > 99.0

    # Active serving scope invariants
    mc = summary["missing_club_stats"]
    assert mc["active_serving_scope_gb1"] > 500
    assert mc["active_serving_scope_ec1"] < 25
    assert (
        mc["active_serving_scope_gb1"] + mc["active_serving_scope_ec1"]
        == mc["active_serving_scope_total"]
    )

    # Drop accounting reconciliation invariant
    vcl = summary["valuation_cleaning_log"]
    reconciled_total = (
        vcl["dropped_non_positive_value"]
        + vcl["dropped_duplicate_player_date"]
        + vcl["dropped_missing_dob"]
        + vcl["dropped_missing_position"]
        + vcl["dropped_before_2013_07_01_appearance_window"]
        + vcl["dropped_incomplete_2026_2027_cycle"]
        + vcl["excluded_other_leagues"]
        + vcl["clean_gb1_ec1_valuations"]
    )
    assert vcl["raw_total_valuations"] == reconciled_total
    assert vcl["clean_gb1_valuations"] + vcl["clean_ec1_valuations"] == vcl["clean_gb1_ec1_valuations"]

    # Promotion/relegation identity & EC1 Tier-1 vs Serie B (`EC2L`) exclusion verification
    pr = summary["promotion_relegation_audit"]
    assert pr["fake_static_gb1_relegated_club_rows"] > 0
    assert (
        pr["true_20_club_pit_gb1_rows"] + pr["fake_static_gb1_relegated_club_rows"]
        == pr["total_static_gb1_rows"]
    )
    assert pr["ec1_club_vals_labeled_ec1"] > 0
    assert pr["ec1_club_vals_excluded_serie_b_or_other"] > 0

    # Single-snapshot salimt per-row contract validity audit invariants
    cv = summary["contract_viability"]
    assert cv["salimt_profiles_snapshot_date"] == SALIMT_PROFILES_SNAPSHOT_DATE
    assert cv["salimt_max_valid_extension_date"] < cv["dcaribou_max_valuation_date"]
    assert cv["pre_test_valid_per_row_count_2018_2025"] > 1000
    assert cv["total_valid_per_row_count_2018_2026"] >= cv["pre_test_valid_per_row_count_2018_2025"]


@pytest.mark.data
def test_raw_manifest_and_parquet_files() -> None:
    manifest_path = RAW_DIR / "MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip("MANIFEST.json not present")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_version"]
    assert "_" in manifest["data_version"]
    assert set(manifest["sources"].keys()) == set(PINNED_SOURCES.keys())
    assert len(manifest["files"]) == 11

    for fname, meta in manifest["files"].items():
        p = RAW_DIR / fname
        if not p.exists():
            pytest.skip(f"Gitignored raw Parquet file {p} not present on clean clone")
        assert meta["row_count"] > 0
        assert len(meta["sha256"]) == 64
        assert len(meta["source_sha256"]) == 64
        assert meta["source_url"].startswith("https://")
        assert meta["csv_reject_count"] >= 0


@pytest.mark.data
def test_clean_players_constraints() -> None:
    players_path = CLEAN_DIR / "players.parquet"
    if not players_path.exists():
        pytest.skip("Gitignored clean/players.parquet not present on clean clone")

    con = duckdb.connect()
    stats = con.sql(f"""
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT tm_player_id) AS distinct_ids,
            COUNT(*) FILTER (WHERE position_group NOT IN ('GK', 'DEF', 'MID', 'FWD') OR position_group IS NULL) AS bad_pos,
            COUNT(*) FILTER (WHERE current_league_code IS NULL OR current_league_code = '') AS null_league,
            COUNT(*) FILTER (WHERE date_of_birth IS NULL) AS null_dob
        FROM read_parquet('{players_path.as_posix()}')
    """).fetchall()[0]

    assert stats[0] > 0
    assert stats[0] == stats[1], "Duplicate tm_player_id in clean/players.parquet"
    assert stats[2] == 0, "Invalid or Missing position_group found"
    assert stats[3] == 0, "NULL current_league_code found"
    assert stats[4] == 0, "NULL date_of_birth found"


@pytest.mark.data
def test_clean_valuations_invariants_and_no_promotion_leakage() -> None:
    val_path = CLEAN_DIR / "valuations.parquet"
    if not val_path.exists():
        pytest.skip("Gitignored clean/valuations.parquet not present on clean clone")

    con = duckdb.connect()
    stats = con.sql(f"""
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT (tm_player_id, valuation_date)) AS unique_pairs,
            COUNT(*) FILTER (WHERE market_value_eur <= 0 OR market_value_eur IS NULL) AS non_positive,
            COUNT(*) FILTER (WHERE age_at_valuation IS NULL OR age_at_valuation < 14 OR age_at_valuation > 50) AS bad_age,
            MIN(valuation_date) AS min_date,
            MAX(valuation_date) AS max_date,
            COUNT(*) FILTER (WHERE league_at_valuation NOT IN ('GB1', 'EC1')) AS other_leagues,
            COUNT(*) FILTER (WHERE league_at_valuation = 'GB1') AS gb1_count,
            COUNT(*) FILTER (WHERE league_at_valuation = 'EC1') AS ec1_count,
            COUNT(valid_contract_years_remaining) AS valid_contract_rows,
            COUNT(*) FILTER (
                WHERE valid_contract_years_remaining IS NOT NULL
                  AND valuation_date > DATE '{SALIMT_PROFILES_SNAPSHOT_DATE}'
            ) AS post_snapshot_valid_contract_rows
        FROM read_parquet('{val_path.as_posix()}')
    """).fetchall()[0]

    assert stats[0] > 0
    assert stats[0] == stats[1], "Duplicate (tm_player_id, valuation_date) in valuations.parquet"
    assert stats[2] == 0, "Non-positive market_value_eur found"
    assert stats[3] == 0, "Invalid or NULL age_at_valuation found"
    assert stats[4] >= date(2013, 7, 1), "Valuation before 2013-07-01 appearance window found"
    assert stats[5] < date(2026, 7, 1), "Valuation from incomplete >= 2026-07-01 cycle found"
    assert stats[6] == 0
    assert stats[7] > stats[8] > 0
    assert stats[7] + stats[8] == stats[0]
    assert stats[9] > 1000
    assert stats[10] == 0, "Found valid_contract_years_remaining after salimt snapshot date"


@pytest.mark.data
def test_clean_player_seasons_unique_key_with_midseason_transfers() -> None:
    seasons_path = CLEAN_DIR / "player_seasons.parquet"
    if not seasons_path.exists():
        pytest.skip("Gitignored clean/player_seasons.parquet not present on clean clone")

    con = duckdb.connect()
    stats = con.sql(f"""
        SELECT
            COUNT(*) AS total,
            COUNT(DISTINCT (tm_player_id, season, tm_club_id)) AS unique_keys,
            COUNT(*) FILTER (WHERE minutes_played < 0 OR minutes_played IS NULL) AS bad_minutes,
            COUNT(*) FILTER (WHERE tm_club_id IS NULL) AS null_club_id
        FROM read_parquet('{seasons_path.as_posix()}')
    """).fetchall()[0]

    assert stats[0] > 0
    assert stats[0] == stats[1], "Duplicate (tm_player_id, season, tm_club_id) in player_seasons.parquet"
    assert stats[2] == 0
    assert stats[3] == 0


def test_cleaning_pipeline_on_synthetic_fixtures(tmp_path: Path) -> None:
    """Verify cleaning drop rules, promotion/relegation guard, and per-row contract filter on fake fixtures."""
    syn_raw = tmp_path / "raw"
    syn_clean = tmp_path / "clean"
    syn_raw.mkdir()

    (syn_raw / "MANIFEST.json").write_text(
        json.dumps({"data_version": "test-fixture-v1"}), encoding="utf-8"
    )
    con = duckdb.connect()

    con.execute(f"""
        COPY (
            SELECT 'GB1' AS competition_id, 'domestic_league' AS type
        ) TO '{(syn_raw / "competitions.parquet").as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT 9001 AS club_id, 'Test FC Alpha' AS name, 'GB1' AS domestic_competition_id
            UNION ALL
            SELECT 9002 AS club_id, 'Test FC Relegated' AS name, 'GB1' AS domestic_competition_id
        ) TO '{(syn_raw / "clubs.parquet").as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT
                999001 AS player_id, 'Test Player Alpha' AS name, DATE '2000-01-15' AS date_of_birth,
                'England' AS country_of_citizenship, 'Midfield' AS position, 'Central Midfield' AS sub_position,
                'right' AS foot, 180 AS height_in_cm, 9001 AS current_club_id, 'Test FC Alpha' AS current_club_name,
                NULL AS current_club_domestic_competition_id, DATE '2027-06-30' AS contract_expiration_date
            UNION ALL
            SELECT
                999002 AS player_id, 'Test Player MissingPos' AS name, DATE '1999-05-10' AS date_of_birth,
                'Ecuador' AS country_of_citizenship, 'Missing' AS position, NULL AS sub_position,
                'left' AS foot, 175 AS height_in_cm, NULL AS current_club_id, 'Retired' AS current_club_name,
                NULL AS current_club_domestic_competition_id, NULL AS contract_expiration_date
            UNION ALL
            SELECT
                999003 AS player_id, 'Test Player NoDOB' AS name, NULL AS date_of_birth,
                'England' AS country_of_citizenship, 'Attack' AS position, 'Centre-Forward' AS sub_position,
                'right' AS foot, 185 AS height_in_cm, 9001 AS current_club_id, 'Test FC Alpha' AS current_club_name,
                'GB1' AS current_club_domestic_competition_id, NULL AS contract_expiration_date
        ) TO '{(syn_raw / "players.parquet").as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT
                999001 AS player_id, 'Test Player Alpha' AS player_name, DATE '2000-01-15' AS date_of_birth,
                'Midfield' AS position, 'Midfield' AS main_position, 'right' AS foot, 180 AS height,
                9001 AS current_club_id, 'Test FC Alpha' AS current_club_name,
                DATE '2022-07-01' AS joined, DATE '2027-06-30' AS contract_expires, DATE '2023-07-01' AS date_of_last_contract_extension
        ) TO '{(syn_raw / "salimt_profiles.parquet").as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            -- Valid GB1 valuation for 999001 in 2023/2024 (joined 2022-07-01, extended 2023-07-01 <= 2024-01-15)
            SELECT 999001 AS player_id, DATE '2024-01-15' AS date, 25000000 AS market_value_in_eur,
                   'Test FC Alpha' AS current_club_name, 9001 AS current_club_id, 'GB1' AS player_club_domestic_competition_id
            UNION ALL
            -- Zero valuation (should drop)
            SELECT 999001 AS player_id, DATE '2023-09-01' AS date, 0 AS market_value_in_eur,
                   'Test FC Alpha' AS current_club_name, 9001 AS current_club_id, 'GB1' AS player_club_domestic_competition_id
            UNION ALL
            -- Valuation at club 9002 which had static_club_league='GB1' but was actually in GB2 in 2023 (should exclude from GB1)
            SELECT 999001 AS player_id, DATE '2023-10-01' AS date, 15000000 AS market_value_in_eur,
                   'Test FC Relegated' AS current_club_name, 9002 AS current_club_id, 'GB1' AS player_club_domestic_competition_id
            UNION ALL
            -- Missing position player (should drop)
            SELECT 999002 AS player_id, DATE '2024-01-15' AS date, 5000000 AS market_value_in_eur,
                   'Test FC Alpha' AS current_club_name, 9001 AS current_club_id, 'GB1' AS player_club_domestic_competition_id
            UNION ALL
            -- Missing DOB player (should drop)
            SELECT 999003 AS player_id, DATE '2024-01-15' AS date, 8000000 AS market_value_in_eur,
                   'Test FC Alpha' AS current_club_name, 9001 AS current_club_id, 'GB1' AS player_club_domestic_competition_id
        ) TO '{(syn_raw / "valuations.parquet").as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT 1 AS game_id, 'GB1' AS competition_id, 2023 AS season, DATE '2023-11-01' AS date,
                   9001 AS home_club_id, 9099 AS away_club_id, 2 AS home_club_goals, 0 AS away_club_goals
        ) TO '{(syn_raw / "games.parquet").as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT '1_999001' AS appearance_id, 1 AS game_id, 999001 AS player_id, 9001 AS player_club_id,
                   DATE '2023-11-01' AS date, 'GB1' AS competition_id, 0 AS yellow_cards, 0 AS red_cards,
                   1 AS goals, 1 AS assists, 90 AS minutes_played
        ) TO '{(syn_raw / "appearances.parquet").as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT 999001 AS player_id, DATE '2022-07-01' AS transfer_date, '22/23' AS transfer_season,
                   8000 AS from_club_id, 9001 AS to_club_id, 'Old FC' AS from_club_name,
                   'Test FC Alpha' AS to_club_name, 10000000.0 AS transfer_fee, 8000000.0 AS market_value_in_eur,
                   'Test Player Alpha' AS player_name
        ) TO '{(syn_raw / "transfers.parquet").as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT 999001 AS player_id, '23/24' AS season_name, 'Hamstring' AS injury_reason,
                   DATE '2023-12-01' AS from_date, DATE '2023-12-15' AS end_date, 14 AS days_missed, 2 AS games_missed
        ) TO '{(syn_raw / "injuries.parquet").as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT 9002 AS club_id, 'Test FC Relegated' AS team_name, 2023 AS season_id, '23/24' AS season_season,
                   'GB2' AS competition_id, 'Championship' AS competition_name, 2 AS tier_level,
                   46 AS season_total_matches, 60 AS season_points, 8 AS season_rank
        ) TO '{(syn_raw / "team_competitions_seasons.parquet").as_posix()}' (FORMAT PARQUET)
    """)
    con.execute(f"""
        COPY (
            SELECT 999999 AS player_id, '2023' AS season_name, 'EL1A' AS competition_id,
                   'LigaPro' AS competition_name, 9500 AS tm_club_id, 'Test EC Club' AS club_name,
                   1 AS appearances, 90 AS minutes_played, 0 AS goals, 0 AS assists,
                   0 AS yellow_cards, 0 AS red_cards, 0 AS clean_sheets, 0 AS goals_conceded
            WHERE 1 = 0
        ) TO '{(syn_raw / "ec1_performances.parquet").as_posix()}' (FORMAT PARQUET)
    """)

    report = run_cleaning_pipeline(raw_dir=syn_raw, clean_dir=syn_clean)
    assert report["valuation_drops"]["dropped_non_positive"] == 1
    assert report["valuation_drops"]["dropped_missing_dob"] == 1
    assert report["valuation_drops"]["dropped_missing_position"] == 1
    assert report["valuation_drops"]["excluded_other_leagues"] == 1
    assert report["valuation_drops"]["kept_gb1_ec1_valuations"] == 1
    assert report["clean_table_counts"]["players"] == 1
    assert report["clean_table_counts"]["transfers"] == 1

    val_row = con.sql(
        f"SELECT valid_contract_years_remaining FROM read_parquet('{(syn_clean / 'valuations.parquet').as_posix()}')"
    ).fetchall()[0]
    assert val_row[0] is not None and val_row[0] > 3.0
