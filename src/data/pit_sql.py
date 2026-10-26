"""Shared Point-in-Time SQL CTEs and Pinned Source Registry.

Used by both `scripts/audit_datasets.py` and `src/data/clean.py` ( plus `src/data/ingest.py`)
so the audit and the cleaning pipeline share a single source of truth for:
  - Pinned upstream dataset URLs and expected SHA-256 digests (`PINNED_SOURCES`)
  - Ecuadorian Liga Pro competition stage codes (`EC_COMP_IDS`)
  - Point-in-time `(club_id, season_year) -> league_code` resolution (`club_season_pit`)
  - Staged valuation enrichment (`staged_valuations`) without static-club promotion/relegation leakage
"""

from __future__ import annotations

from typing import Any

import duckdb

EC_COMP_IDS: tuple[str, ...] = ("EL1A", "EL1S", "EC1N", "ECPE", "ECSE", "LPPL", "ECPL")

DCARIBOU_BASE_R2 = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
SALIMT_BASE_RAW = (
    "https://raw.githubusercontent.com/salimt/football-datasets/main/datalake/transfermarkt"
)
SALIMT_BASE_LFS = (
    "https://media.githubusercontent.com/media/salimt/football-datasets/main/datalake/transfermarkt"
)

# Frozen upstream snapshot pinned by URL, SHA-256 digest, and byte size
PINNED_SOURCES: dict[str, dict[str, Any]] = {
    "dcaribou/competitions.csv.gz": {
        "url": f"{DCARIBOU_BASE_R2}/competitions.csv.gz",
        "sha256": "8924ddfbc0e9989f4a42a3c32ebb6faa671614625086d7fa84353f955352ea88",
        "size_bytes": 2242,
    },
    "dcaribou/clubs.csv.gz": {
        "url": f"{DCARIBOU_BASE_R2}/clubs.csv.gz",
        "sha256": "aa57ab56e87faffaea1e8697439a32caf10116d8f2f2dfb3a57193c0bb1d641b",
        "size_bytes": 51281,
    },
    "dcaribou/players.csv.gz": {
        "url": f"{DCARIBOU_BASE_R2}/players.csv.gz",
        "sha256": "d22e407981d5b51a79bf8ff59835729f3526f7dc3495a6d8ed2f852ed2e86403",
        "size_bytes": 4389958,
    },
    "dcaribou/player_valuations.csv.gz": {
        "url": f"{DCARIBOU_BASE_R2}/player_valuations.csv.gz",
        "sha256": "608e32e5ad2231074f3505c469ad727be66d7e6efda6af918afd85379439631d",
        "size_bytes": 7197935,
    },
    "dcaribou/appearances.csv.gz": {
        "url": f"{DCARIBOU_BASE_R2}/appearances.csv.gz",
        "sha256": "e096c2b158fc3d52c4550526c4d4850dd62f31007652f4bcae112eadb1646802",
        "size_bytes": 45253638,
    },
    "dcaribou/games.csv.gz": {
        "url": f"{DCARIBOU_BASE_R2}/games.csv.gz",
        "sha256": "142561989017d379bcf9e72bad0b87e0996bae0e7b710ebb3def9255f464c741",
        "size_bytes": 4995595,
    },
    "dcaribou/transfers.csv.gz": {
        "url": f"{DCARIBOU_BASE_R2}/transfers.csv.gz",
        "sha256": "90326983daf7e6ac7aabdfe62b90936d9bf1dd2171e53dec75c3751eb1620a83",
        "size_bytes": 5809514,
    },
    "salimt/player_profiles.csv": {
        "url": f"{SALIMT_BASE_RAW}/player_profiles/player_profiles.csv",
        "sha256": "7ef9afbacf97577d099f21a0ef0ea0d46381bd4806a119f095b9562762a20096",
        "size_bytes": 26380409,
    },
    "salimt/player_market_value.csv": {
        "url": f"{SALIMT_BASE_RAW}/player_market_value/player_market_value.csv",
        "sha256": "e9e1868c883f0f40559127a407f5755c6a7f2e1ded1b4cec842d9185c4edb984",
        "size_bytes": 23873747,
    },
    "salimt/player_injuries_updated_20260929.csv": {
        "url": f"{SALIMT_BASE_RAW}/player_injuries/player_injuries_updated_20260929.csv",
        "sha256": "c622ed5684ec42e456ee23a9f5a78d61f74293d55e38882a7829ad71fb40d815",
        "size_bytes": 9455168,
    },
    "salimt/team_competitions_seasons.csv": {
        "url": f"{SALIMT_BASE_RAW}/team_competitions_seasons/team_competitions_seasons.csv",
        "sha256": "83f06c8108e87d74ebccd7d02df9b7a5fe5769bbc7004771af64230b1eff6319",
        "size_bytes": 11032370,
    },
    "salimt/player_performances.csv": {
        "url": f"{SALIMT_BASE_LFS}/player_performances/player_performances.csv",
        "sha256": "21eda6556654478fe986de8972d3b94aea6551ebcd74d5eb153b65edb53d132b",
        "size_bytes": 157647289,
    },
}


def register_pit_views(
    con: duckdb.DuckDBPyConnection,
    *,
    comp_view: str = "raw_comp",
    games_view: str = "raw_games",
    tcs_view: str = "raw_tcs",
    tcs_comp_col: str = "competition_id",
    ec_perf_view: str = "raw_ec1_perf",
    ec_club_col: str = "tm_club_id",
    ec_filter_clause: str = "1 = 1",
    val_view: str = "raw_val",
    players_view: str = "raw_players",
    sal_prof_view: str = "raw_sal_prof",
) -> None:
    """Create `club_season_pit`, `ec1_clubs`, and `staged_valuations` temp views on `con`.

    Guarantees identical point-in-time league resolution and duplicate/anomaly tagging
    across `scripts/audit_datasets.py` and `src/data/clean.py`.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW club_season_pit AS
        WITH domestic_leagues AS (
            SELECT competition_id FROM {comp_view} WHERE type = 'domestic_league'
        ),
        games_club_season AS (
            SELECT home_club_id AS club_id, season, g.competition_id, COUNT(*) AS n_g
            FROM {games_view} g JOIN domestic_leagues dl ON g.competition_id = dl.competition_id
            GROUP BY 1, 2, 3
            UNION ALL
            SELECT away_club_id AS club_id, season, g.competition_id, COUNT(*) AS n_g
            FROM {games_view} g JOIN domestic_leagues dl ON g.competition_id = dl.competition_id
            GROUP BY 1, 2, 3
        ),
        dc_club_season AS (
            SELECT club_id, season, competition_id,
                   ROW_NUMBER() OVER (PARTITION BY club_id, season ORDER BY SUM(n_g) DESC) AS rn
            FROM games_club_season
            GROUP BY 1, 2, 3
        ),
        tcs_club_season AS (
            SELECT
                club_id,
                season_id AS season,
                CASE
                    WHEN {tcs_comp_col} IN {EC_COMP_IDS} THEN 'EC1'
                    ELSE {tcs_comp_col}
                END AS competition_id,
                ROW_NUMBER() OVER (
                    PARTITION BY club_id, season_id
                    ORDER BY season_total_matches DESC NULLS LAST
                ) AS rn
            FROM {tcs_view}
            WHERE {tcs_comp_col} IS NOT NULL
        ),
        all_club_seasons AS (
            SELECT club_id, season FROM dc_club_season WHERE rn = 1
            UNION
            SELECT club_id, season FROM tcs_club_season WHERE rn = 1
        )
        SELECT
            acs.club_id,
            acs.season,
            dcs.competition_id AS games_derived_league,
            tcs.competition_id AS tcs_derived_league,
            COALESCE(dcs.competition_id, tcs.competition_id) AS pit_league_code
        FROM all_club_seasons acs
        LEFT JOIN dc_club_season dcs
          ON acs.club_id = dcs.club_id AND acs.season = dcs.season AND dcs.rn = 1
        LEFT JOIN tcs_club_season tcs
          ON acs.club_id = tcs.club_id AND acs.season = tcs.season AND tcs.rn = 1
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW ec1_clubs AS
        SELECT DISTINCT CAST({ec_club_col} AS BIGINT) AS club_id
        FROM {ec_perf_view}
        WHERE {ec_club_col} IS NOT NULL AND ({ec_filter_clause})
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW staged_valuations AS
        SELECT
            CAST(v.player_id AS BIGINT) AS tm_player_id,
            CAST(v.date AS DATE) AS valuation_date,
            CAST(v.market_value_in_eur AS BIGINT) AS market_value_eur,
            CAST(v.current_club_id AS BIGINT) AS tm_club_id_at_valuation,
            NULLIF(TRIM(v.current_club_name), '') AS club_at_valuation,
            v.player_club_domestic_competition_id AS static_club_league,
            CASE
                WHEN EXTRACT(MONTH FROM v.date) >= 7 THEN CAST(EXTRACT(YEAR FROM v.date) AS INTEGER)
                ELSE CAST(EXTRACT(YEAR FROM v.date) - 1 AS INTEGER)
            END AS season_year,
            CASE
                WHEN EXTRACT(MONTH FROM v.date) >= 7
                THEN CAST(EXTRACT(YEAR FROM v.date) AS VARCHAR) || '/' || CAST(EXTRACT(YEAR FROM v.date) + 1 AS VARCHAR)
                ELSE CAST(EXTRACT(YEAR FROM v.date) - 1 AS VARCHAR) || '/' || CAST(EXTRACT(YEAR FROM v.date) AS VARCHAR)
            END AS cycle,
            csp.games_derived_league,
            csp.tcs_derived_league,
            (ec.club_id IS NOT NULL) AS is_ec1_club,
            COALESCE(
                csp.pit_league_code,
                CASE WHEN ec.club_id IS NOT NULL THEN 'EC1' ELSE 'OTHER' END
            ) AS league_at_valuation,
            CAST(p.date_of_birth AS DATE) AS date_of_birth,
            CAST(p.contract_expiration_date AS DATE) AS dc_contract_expiration_date,
            CAST(p.current_club_id AS BIGINT) AS dc_current_club_id,
            COALESCE(NULLIF(p.position, 'Missing'), sp.main_position, 'Missing') AS resolved_position,
            CAST(sp.current_club_id AS BIGINT) AS sal_current_club_id,
            CAST(sp.contract_expires AS DATE) AS sal_contract_expires,
            CAST(sp.joined AS DATE) AS sal_joined,
            CAST(sp.date_of_last_contract_extension AS DATE) AS sal_last_extension,
            ROW_NUMBER() OVER (
                PARTITION BY v.player_id, CAST(v.date AS DATE)
                ORDER BY v.market_value_in_eur DESC NULLS LAST
            ) AS dup_rn
        FROM {val_view} v
        LEFT JOIN {players_view} p ON v.player_id = p.player_id
        LEFT JOIN {sal_prof_view} sp ON v.player_id = sp.player_id
        LEFT JOIN club_season_pit csp
          ON v.current_club_id = csp.club_id
         AND (CASE WHEN EXTRACT(MONTH FROM v.date) >= 7 THEN EXTRACT(YEAR FROM v.date) ELSE EXTRACT(YEAR FROM v.date) - 1 END) = csp.season
        LEFT JOIN ec1_clubs ec
          ON v.current_club_id = ec.club_id
    """)
