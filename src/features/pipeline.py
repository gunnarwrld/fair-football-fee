"""Phase 2: Feature Pipeline & `player_current_features` Serving Table Writer.

Builds:
  1. `data/features/priors.json` (Empirical Bayes shrinkage priors & train league medians)
  2. `data/features/valuation_snapshots.parquet` (all 19,938 historical GB1 + EC1 snapshots)
  3. `data/features/current_serving_features.parquet` (active supported players for API serving;
     Phase 2.1: Premier League players with >= 450 tracked minutes in the prior 365 days)
  4. `data/features/feature_manifest.json`
  5. PostgreSQL `player_current_features` rows (single-transaction upsert + stale row deletion)
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.config import settings
from src.data.pit_sql import (
    INACTIVE_CLUB_IDS,
    INACTIVE_CLUB_NAMES,
)
from src.db.models import Player, PlayerCurrentFeatures
from src.db.session import SessionLocal
from src.features.engineer import (
    CONTRACT_ABLATION_FEATURES,
    MAIN_MODEL_CATEGORICAL_FEATURES,
    MAIN_MODEL_FEATURES,
    MAIN_MODEL_NUMERIC_FEATURES,
    POPULATION_MIN_TRACKED_MINUTES_12M,
    SUPPORTED_LEAGUES,
    VALUE_UPDATE_MODEL_FEATURES,
    EmpiricalBayesPriors,
    build_serving_feature_dict,
    compute_empirical_bayes_priors,
    engineer_features_for_snapshots,
)


def season_year_for(d: date) -> int:
    """Football season start year (July-June cycle) for a date."""
    return d.year if d.month >= 7 else d.year - 1


def load_clean_tables(
    clean_dir: Path | None = None, raw_dir: Path | None = None
) -> dict[str, pd.DataFrame]:
    clean_dir = clean_dir or settings.clean_data_dir
    raw_dir = raw_dir or settings.raw_data_dir
    injury_coverage = (
        pd.read_parquet(raw_dir / "salimt_profiles.parquet", columns=["player_id"])
        .rename(columns={"player_id": "tm_player_id"})
        .dropna()
        .astype({"tm_player_id": "int64"})
        .drop_duplicates()
    )
    return {
        "players": pd.read_parquet(clean_dir / "players.parquet"),
        "valuations": pd.read_parquet(clean_dir / "valuations.parquet"),
        "all_valuations": pd.read_parquet(clean_dir / "all_valuations.parquet"),
        "appearances": pd.read_parquet(clean_dir / "appearances.parquet"),
        "player_seasons": pd.read_parquet(clean_dir / "player_seasons.parquet"),
        "transfers": pd.read_parquet(clean_dir / "transfers.parquet"),
        "injuries": pd.read_parquet(clean_dir / "injuries.parquet"),
        # Players present in the injury source (salimt profiles); others get NULL injury features
        "injury_coverage": injury_coverage,
    }


def select_active_serving_requests(
    *,
    clean_dir: Path | None = None,
    raw_dir: Path | None = None,
    max_complete_val_date: str,
    as_of_date_override: date | None = None,
) -> pd.DataFrame:
    """Select candidate active players for `player_current_features`.

    A player is a serving candidate if and only if:
      1. Their latest valuation across all career leagues (`all_valuations.parquet`)
         is within the 365 days before `max_complete_val_date` (from `MANIFEST.json`).
      2. Neither their latest valuation club nor their profile club is a retired/unattached
         pseudo-club (`123`, `515`, `75`, `2113` or `'Retired'`, `'Without Club'`, etc.),
         and they have not subsequently transferred to another club in `transfers.parquet`.
      3. Their club plays in a supported league (`GB1`) in the season containing
         `max_complete_val_date`.
    `run_feature_pipeline` then keeps only candidates inside the modeling population
    (>= 450 tracked minutes in the prior 365 days).
    """
    clean_dir = clean_dir or settings.clean_data_dir
    raw_dir = raw_dir or settings.raw_data_dir
    current_season = season_year_for(date.fromisoformat(max_complete_val_date))
    supported_leagues_sql = "(" + ", ".join(f"'{lg}'" for lg in SUPPORTED_LEAGUES) + ")"

    con = duckdb.connect()
    active_df = con.sql(f"""
        WITH latest_any_val AS (
            SELECT
                tm_player_id,
                MAX(CAST(valuation_date AS DATE)) AS latest_val_date,
                ARG_MAX(league_at_valuation, CAST(valuation_date AS DATE)) AS latest_val_league,
                ARG_MAX(CAST(tm_club_id_at_valuation AS BIGINT), CAST(valuation_date AS DATE)) AS latest_val_club_id,
                ARG_MAX(club_at_valuation, CAST(valuation_date AS DATE)) AS latest_val_club_name,
                ARG_MAX(market_value_eur, CAST(valuation_date AS DATE)) AS latest_market_value_eur
            FROM read_parquet('{(clean_dir / "all_valuations.parquet").as_posix()}')
            GROUP BY 1
        ),
        club_current AS (
            SELECT DISTINCT CAST(club_id AS BIGINT) AS club_id, competition_id AS league_current
            FROM (
                SELECT home_club_id AS club_id, competition_id
                FROM read_parquet('{(raw_dir / "games.parquet").as_posix()}')
                WHERE competition_id IN {supported_leagues_sql} AND season = {current_season}
                UNION
                SELECT away_club_id AS club_id, competition_id
                FROM read_parquet('{(raw_dir / "games.parquet").as_posix()}')
                WHERE competition_id IN {supported_leagues_sql} AND season = {current_season}
            )
        ),
        latest_transfer AS (
            SELECT
                CAST(tm_player_id AS BIGINT) AS tm_player_id,
                MAX(CAST(transfer_date AS DATE)) AS latest_transfer_date,
                ARG_MAX(CAST(to_club_id AS BIGINT), CAST(transfer_date AS DATE)) AS latest_to_club_id
            FROM read_parquet('{(clean_dir / "transfers.parquet").as_posix()}')
            WHERE CAST(transfer_date AS DATE) <= DATE '{max_complete_val_date}'
            GROUP BY 1
        ),
        val_contract AS (
            SELECT
                tm_player_id,
                CAST(valuation_date AS DATE) AS valuation_date,
                valid_contract_years_remaining
            FROM read_parquet('{(clean_dir / "valuations.parquet").as_posix()}')
        )
        SELECT
            lv.tm_player_id,
            lv.latest_val_date AS as_of_date,
            lv.latest_val_club_id AS tm_club_id,
            COALESCE(lv.latest_val_club_name, cp.current_club) AS club_name,
            cc.league_current AS league_code,
            lv.latest_market_value_eur AS market_value_eur,
            vc.valid_contract_years_remaining
        FROM latest_any_val lv
        JOIN read_parquet('{(raw_dir / "players.parquet").as_posix()}') rp
          ON lv.tm_player_id = CAST(rp.player_id AS BIGINT)
        JOIN read_parquet('{(clean_dir / "players.parquet").as_posix()}') cp
          ON lv.tm_player_id = CAST(cp.tm_player_id AS BIGINT)
        JOIN club_current cc
          ON lv.latest_val_club_id = cc.club_id
        LEFT JOIN latest_transfer lt
          ON lv.tm_player_id = lt.tm_player_id
        LEFT JOIN val_contract vc
          ON lv.tm_player_id = vc.tm_player_id
         AND lv.latest_val_date = vc.valuation_date
        WHERE lv.latest_val_date >= CAST(DATE '{max_complete_val_date}' - INTERVAL 365 DAY AS DATE)
          AND lv.latest_val_league IN {supported_leagues_sql}
          AND lv.latest_val_club_id NOT IN {INACTIVE_CLUB_IDS}
          AND rp.current_club_id NOT IN {INACTIVE_CLUB_IDS}
          AND COALESCE(rp.current_club_name, '') NOT IN {INACTIVE_CLUB_NAMES}
          AND (
              lt.latest_transfer_date IS NULL
              OR lt.latest_transfer_date <= lv.latest_val_date
              OR lt.latest_to_club_id = lv.latest_val_club_id
          )
        ORDER BY lv.tm_player_id
    """).df()

    active_df["as_of_date"] = pd.to_datetime(active_df["as_of_date"]).dt.date
    if as_of_date_override is not None:
        active_df["as_of_date"] = as_of_date_override
    return active_df


def compute_serving_features_for_requests(
    requests_df: pd.DataFrame,
    *,
    clean_tables: dict[str, pd.DataFrame],
    priors: EmpiricalBayesPriors,
    data_version: str,
) -> pd.DataFrame:
    """Compute serving feature vectors using the exact same code path as training snapshots.

    Accepts any `requests_df` with `(tm_player_id, as_of_date, tm_club_id, club_name, league_code)`
    (including historical `(player_id, valuation_date)` rows for train/serve parity tests).
    """
    engineered_df = engineer_features_for_snapshots(
        requests_df,
        players_df=clean_tables["players"],
        appearances_df=clean_tables["appearances"],
        player_seasons_df=clean_tables["player_seasons"],
        injuries_df=clean_tables["injuries"],
        transfers_df=clean_tables["transfers"],
        valuations_df=clean_tables["valuations"],
        all_valuations_df=clean_tables["all_valuations"],
        priors=priors,
        injury_coverage_df=clean_tables.get("injury_coverage"),
    )

    records: list[dict[str, Any]] = []
    for _, row in engineered_df.iterrows():
        feat_dict = build_serving_feature_dict(row)
        records.append(
            {
                "tm_player_id": int(row["tm_player_id"]),
                "as_of_date": row["as_of_date"],
                "data_version": data_version,
                "league_code": str(row["league_code"]),
                "trailing_league_median_eur": float(row["trailing_league_median_eur"]),
                "features_json": json.dumps(feat_dict, sort_keys=True),
                "exclusion_reason": row["exclusion_reason"],
            }
        )
    return pd.DataFrame(records)


def run_feature_pipeline(
    *,
    raw_dir: Path | None = None,
    clean_dir: Path | None = None,
    features_dir: Path | None = None,
) -> dict[str, Any]:
    raw_dir = raw_dir or settings.raw_data_dir
    clean_dir = clean_dir or settings.clean_data_dir
    features_dir = features_dir or settings.features_data_dir
    features_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} missing; run ingestion first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_version = str(manifest["data_version"])
    max_val_date = str(manifest["max_complete_valuation_date"])

    clean_tables = load_clean_tables(clean_dir, raw_dir)

    # 1. Fit Empirical Bayes priors & fallback league medians strictly on train (< 2023-07-01)
    priors = compute_empirical_bayes_priors(
        players_df=clean_tables["players"],
        appearances_df=clean_tables["appearances"],
        player_seasons_df=clean_tables["player_seasons"],
        valuations_df=clean_tables["valuations"],
    )
    priors_path = features_dir / "priors.json"
    priors_path.write_text(json.dumps(priors.to_dict(), indent=2), encoding="utf-8")

    # 2. Build all historical valuation snapshots (train, val_tune, val_calib, test)
    vals_df = clean_tables["valuations"].rename(
        columns={
            "valuation_date": "as_of_date",
            "tm_club_id_at_valuation": "tm_club_id",
            "club_at_valuation": "club_name",
            "league_at_valuation": "league_code",
        }
    )
    snapshots_df = engineer_features_for_snapshots(
        vals_df,
        players_df=clean_tables["players"],
        appearances_df=clean_tables["appearances"],
        player_seasons_df=clean_tables["player_seasons"],
        injuries_df=clean_tables["injuries"],
        transfers_df=clean_tables["transfers"],
        valuations_df=clean_tables["valuations"],
        all_valuations_df=clean_tables["all_valuations"],
        priors=priors,
        injury_coverage_df=clean_tables["injury_coverage"],
    )
    snapshots_df["data_version"] = data_version
    snapshots_out = features_dir / "valuation_snapshots.parquet"
    snapshots_df.to_parquet(snapshots_out, index=False, compression="zstd")

    # 3. Build active serving features using the exact same feature engineering path
    active_requests_df = select_active_serving_requests(
        clean_dir=clean_dir,
        raw_dir=raw_dir,
        max_complete_val_date=max_val_date,
    )
    serving_candidates_df = compute_serving_features_for_requests(
        active_requests_df,
        clean_tables=clean_tables,
        priors=priors,
        data_version=data_version,
    )
    serving_exclusions = (
        serving_candidates_df["exclusion_reason"].dropna().value_counts().to_dict()
    )
    serving_df = (
        serving_candidates_df[serving_candidates_df["exclusion_reason"].isna()]
        .drop(columns=["exclusion_reason"])
        .reset_index(drop=True)
    )
    serving_out = features_dir / "current_serving_features.parquet"
    serving_df.to_parquet(serving_out, index=False, compression="zstd")

    split_counts = (
        snapshots_df.groupby(["split", "league_code"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    serving_league_counts = serving_df["league_code"].value_counts().to_dict()
    population_df = snapshots_df[snapshots_df["in_model_population"]]
    population_counts = population_df["split"].value_counts().to_dict()
    exclusion_counts = (
        snapshots_df.groupby(["split", "exclusion_reason"]).size().unstack(fill_value=0)
        .to_dict(orient="index")
    )

    feature_manifest = {
        "data_version": data_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "total_valuation_snapshots": len(snapshots_df),
        "snapshot_counts_by_split": {
            split: {
                "total": int(sum(lg_counts.values())),
                "GB1": int(lg_counts.get("GB1", 0)),
                "EC1": int(lg_counts.get("EC1", 0)),
            }
            for split, lg_counts in split_counts.items()
        },
        "model_population_rule": {
            "supported_leagues": list(SUPPORTED_LEAGUES),
            "min_tracked_minutes_prior_365d": POPULATION_MIN_TRACKED_MINUTES_12M,
        },
        "model_population_counts_by_split": {
            split: int(population_counts.get(split, 0))
            for split in ("train", "val_tune", "val_calib", "test")
        },
        "excluded_snapshot_counts_by_split": {
            split: {reason: int(n) for reason, n in reasons.items()}
            for split, reasons in exclusion_counts.items()
        },
        "active_serving_scope": {
            "total": len(serving_df),
            "GB1": int(serving_league_counts.get("GB1", 0)),
            "EC1": int(serving_league_counts.get("EC1", 0)),
            "excluded_candidates": {k: int(v) for k, v in serving_exclusions.items()},
        },
        "main_model_categorical_features": MAIN_MODEL_CATEGORICAL_FEATURES,
        "main_model_numeric_features": MAIN_MODEL_NUMERIC_FEATURES,
        "main_model_features": MAIN_MODEL_FEATURES,
        "value_update_model_features": VALUE_UPDATE_MODEL_FEATURES,
        "contract_ablation_features": CONTRACT_ABLATION_FEATURES,
        "priors": priors.to_dict(),
    }
    manifest_out = features_dir / "feature_manifest.json"
    manifest_out.write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    return feature_manifest


def populate_player_current_features(
    session: Session | None = None,
    features_dir: Path | None = None,
    batch_size: int = 1000,
) -> dict[str, int]:
    """Atomically upsert `player_current_features` in PostgreSQL and delete stale rows."""
    features_dir = features_dir or settings.features_data_dir
    serving_path = features_dir / "current_serving_features.parquet"
    serving_df = pd.read_parquet(serving_path)

    owns_session = session is None
    db = session or SessionLocal()

    try:
        rows = db.execute(select(Player.tm_player_id, Player.id)).all()
        tm_to_id = {int(tm_id): int(pid) for tm_id, pid in rows}

        db_records: list[dict[str, Any]] = []
        for _, r in serving_df.iterrows():
            tm_id = int(r["tm_player_id"])
            if tm_id not in tm_to_id:
                continue
            as_of = r["as_of_date"]
            if hasattr(as_of, "date") and callable(as_of.date):
                as_of = as_of.date()
            db_records.append(
                {
                    "player_id": tm_to_id[tm_id],
                    "as_of_date": as_of,
                    "data_version": str(r["data_version"]),
                    "league_code": str(r["league_code"]),
                    "trailing_league_median_eur": float(r["trailing_league_median_eur"]),
                    "features": json.loads(str(r["features_json"])),
                }
            )

        for i in range(0, len(db_records), batch_size):
            chunk = db_records[i : i + batch_size]
            stmt = pg_insert(PlayerCurrentFeatures).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=["player_id"],
                set_={
                    "as_of_date": stmt.excluded.as_of_date,
                    "data_version": stmt.excluded.data_version,
                    "league_code": stmt.excluded.league_code,
                    "trailing_league_median_eur": stmt.excluded.trailing_league_median_eur,
                    "features": stmt.excluded.features,
                    "updated_at": func.now(),
                },
            )
            db.execute(stmt)
        db.flush()

        valid_pids = {int(r["player_id"]) for r in db_records}
        existing_pids = [
            int(pid) for (pid,) in db.execute(select(PlayerCurrentFeatures.player_id)).all()
        ]
        stale_pids = [pid for pid in existing_pids if pid not in valid_pids]
        deleted_stale = 0
        if stale_pids:
            del_res = db.execute(
                delete(PlayerCurrentFeatures).where(PlayerCurrentFeatures.player_id.in_(stale_pids))
            )
            deleted_stale = int(del_res.rowcount or 0)
            db.flush()

        if owns_session:
            db.commit()

        return {
            "player_current_features_upserted": len(db_records),
            "player_current_features_deleted_stale": deleted_stale,
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

    summary = run_feature_pipeline()
    if "--load-db" in sys.argv:
        summary["postgres_upsert"] = populate_player_current_features()
    print(json.dumps(summary, indent=2))
