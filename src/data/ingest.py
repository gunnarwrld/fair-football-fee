"""Phase 1 (Task 2a): Raw Data Ingestion & MANIFEST.json Generator.

Verifies raw source files in `data/raw/sources/` against `PINNED_SOURCES`, converts
canonical tables into typed ZSTD Parquet files in `data/raw/*.parquet` via DuckDB,
logs CSV reject counts via `store_rejects=true` (never silently dropping rows with
unlogged `ignore_errors=true`), and writes `data/raw/MANIFEST.json` with pinned
source URLs, source SHA-256 hashes, output Parquet SHA-256 hashes, and deterministic
`data_version`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from src.config import settings
from src.data.pit_sql import EC_COMP_IDS, PINNED_SOURCES


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_hash_raw_sources(
    sources_dir: Path,
    *,
    enforce_pins: bool = True,
) -> dict[str, dict[str, Any]]:
    """Verify every raw source file against PINNED_SOURCES and return source manifest metadata."""
    source_records: dict[str, dict[str, Any]] = {}
    for rel_path, pin_meta in PINNED_SOURCES.items():
        fpath = sources_dir / rel_path
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required raw source file: {fpath}")
        actual_sha = sha256_file(fpath)
        actual_size = fpath.stat().st_size
        expected_sha = pin_meta["sha256"]
        if enforce_pins and actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for pinned source {rel_path}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        source_records[rel_path] = {
            "url": pin_meta["url"],
            "sha256": actual_sha,
            "expected_sha256": expected_sha,
            "pin_verified": actual_sha == expected_sha,
            "size_bytes": actual_size,
        }
    return source_records


def _count_csv_rejects(con: duckdb.DuckDBPyConnection) -> int:
    has_table = con.sql(
        "SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = 'reject_errors'"
    ).fetchall()[0][0]
    if not has_table:
        return 0
    return int(con.sql("SELECT COUNT(*) FROM reject_errors").fetchall()[0][0])


def ingest_raw_to_parquet(
    sources_dir: Path | None = None,
    raw_dir: Path | None = None,
    *,
    enforce_pins: bool = True,
) -> dict[str, Any]:
    raw_dir = raw_dir or settings.raw_data_dir
    sources_dir = sources_dir or (raw_dir / "sources")
    dc_dir = sources_dir / "dcaribou"
    sal_dir = sources_dir / "salimt"

    raw_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = verify_and_hash_raw_sources(sources_dir, enforce_pins=enforce_pins)

    con = duckdb.connect()

    # Map canonical output parquet files to (source_rel_path, DuckDB extraction query)
    # Using store_rejects=true instead of silent ignore_errors=true so any malformed CSV row is logged
    extractions: dict[str, tuple[str, str]] = {
        "competitions.parquet": (
            "dcaribou/competitions.csv.gz",
            f"SELECT * FROM read_csv_auto('{dc_dir.as_posix()}/competitions.csv.gz', store_rejects=true)",
        ),
        "clubs.parquet": (
            "dcaribou/clubs.csv.gz",
            f"SELECT * FROM read_csv_auto('{dc_dir.as_posix()}/clubs.csv.gz', store_rejects=true)",
        ),
        "players.parquet": (
            "dcaribou/players.csv.gz",
            f"SELECT * FROM read_csv_auto('{dc_dir.as_posix()}/players.csv.gz', store_rejects=true)",
        ),
        "salimt_profiles.parquet": (
            "salimt/player_profiles.csv",
            f"""
            SELECT
                player_id,
                player_name,
                date_of_birth,
                position,
                main_position,
                foot,
                height,
                current_club_id,
                current_club_name,
                joined,
                contract_expires,
                date_of_last_contract_extension
            FROM read_csv_auto('{sal_dir.as_posix()}/player_profiles.csv', store_rejects=true)
            """,
        ),
        "valuations.parquet": (
            "dcaribou/player_valuations.csv.gz",
            f"SELECT * FROM read_csv_auto('{dc_dir.as_posix()}/player_valuations.csv.gz', store_rejects=true)",
        ),
        "appearances.parquet": (
            "dcaribou/appearances.csv.gz",
            f"SELECT * FROM read_csv_auto('{dc_dir.as_posix()}/appearances.csv.gz', store_rejects=true)",
        ),
        "games.parquet": (
            "dcaribou/games.csv.gz",
            f"SELECT * FROM read_csv_auto('{dc_dir.as_posix()}/games.csv.gz', store_rejects=true)",
        ),
        "injuries.parquet": (
            "salimt/player_injuries_updated_20260929.csv",
            f"""
            SELECT
                player_id,
                season_name,
                injury_reason,
                from_date,
                end_date,
                days_missed,
                games_missed
            FROM read_csv_auto('{sal_dir.as_posix()}/player_injuries_updated_20260929.csv', store_rejects=true)
            """,
        ),
        "team_competitions_seasons.parquet": (
            "salimt/team_competitions_seasons.csv",
            f"""
            SELECT
                club_id,
                team_name,
                season_id,
                season_season,
                season_league_competition_id AS competition_id,
                season_league_league_name AS competition_name,
                season_league_level_level_number AS tier_level,
                season_total_matches,
                season_points,
                season_rank
            FROM read_csv_auto('{sal_dir.as_posix()}/team_competitions_seasons.csv', store_rejects=true)
            """,
        ),
        "ec1_performances.parquet": (
            "salimt/player_performances.csv",
            f"""
            SELECT
                player_id,
                season_name,
                competition_id,
                competition_name,
                team_id AS tm_club_id,
                team_name AS club_name,
                COALESCE(TRY_CAST(nb_on_pitch AS INTEGER), 0) AS appearances,
                COALESCE(TRY_CAST(minutes_played AS INTEGER), 0) AS minutes_played,
                COALESCE(TRY_CAST(goals AS INTEGER), 0) AS goals,
                COALESCE(TRY_CAST(assists AS INTEGER), 0) AS assists,
                COALESCE(TRY_CAST(yellow_cards AS INTEGER), 0) AS yellow_cards,
                COALESCE(TRY_CAST(second_yellow_cards AS INTEGER), 0)
                  + COALESCE(TRY_CAST(direct_red_cards AS INTEGER), 0) AS red_cards,
                COALESCE(TRY_CAST(clean_sheets AS INTEGER), 0) AS clean_sheets,
                COALESCE(TRY_CAST(goals_conceded AS INTEGER), 0) AS goals_conceded
            FROM read_csv_auto('{sal_dir.as_posix()}/player_performances.csv', store_rejects=true)
            WHERE competition_id IN {EC_COMP_IDS}
            """,
        ),
    }

    manifest_files: dict[str, Any] = {}
    composite_hasher = hashlib.sha256()

    for filename, (source_rel_path, query) in extractions.items():
        con.execute("DROP TABLE IF EXISTS reject_errors; DROP TABLE IF EXISTS reject_scans;")
        out_path = raw_dir / filename
        con.execute(f"COPY ({query}) TO '{out_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        row_count = int(
            con.sql(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchall()[0][0]
        )
        reject_count = _count_csv_rejects(con)
        file_hash = sha256_file(out_path)
        src_meta = source_manifest[source_rel_path]
        composite_hasher.update(
            f"{filename}:{file_hash}:src={src_meta['sha256']}\n".encode()
        )
        manifest_files[filename] = {
            "canonical_source": source_rel_path,
            "source_url": src_meta["url"],
            "source_sha256": src_meta["sha256"],
            "csv_reject_count": reject_count,
            "row_count": row_count,
            "size_bytes": out_path.stat().st_size,
            "sha256": file_hash,
        }
        source_manifest[source_rel_path]["csv_reject_count"] = reject_count

    # Determine max complete valuation date (< 2026-07-01) for snapshot stamp
    val_parquet = raw_dir / "valuations.parquet"
    max_val_date = con.sql(
        f"SELECT MAX(date) FROM read_parquet('{val_parquet.as_posix()}') WHERE date < DATE '2026-07-01'"
    ).fetchall()[0][0]
    digest10 = composite_hasher.hexdigest()[:10]
    data_version = f"{max_val_date.isoformat()}_{digest10}"

    manifest = {
        "data_version": data_version,
        "max_complete_valuation_date": max_val_date.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(),
        "sources": source_manifest,
        "files": manifest_files,
    }

    manifest_path = raw_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    manifest_data = ingest_raw_to_parquet()
    print(f"Ingested raw Parquet tables. data_version={manifest_data['data_version']}")
