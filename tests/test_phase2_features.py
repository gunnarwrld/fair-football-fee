"""Phase 2: Feature Engineering & Serving Scope Unit/Data Tests (`tests/test_phase2_features.py`)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.features.engineer import (
    MAIN_MODEL_FEATURES,
    compute_empirical_bayes_priors,
    engineer_features_for_snapshots,
)

FEATURES_DIR = Path("data/features")


def test_empirical_bayes_priors_ignore_post_train_cutoff_data() -> None:
    """Verify priors are computed strictly from `< 2023-07-01` and ignore `val`/`test` rows."""
    players_df = pd.DataFrame(
        [
            {"tm_player_id": 1, "position_group": "FWD"},
            {"tm_player_id": 2, "position_group": "MID"},
        ]
    )
    app_train = pd.DataFrame(
        [
            {"tm_player_id": 1, "match_date": date(2022, 5, 1), "minutes_played": 900, "goals": 5, "assists": 2},
            {"tm_player_id": 2, "match_date": date(2022, 5, 1), "minutes_played": 900, "goals": 1, "assists": 3},
        ]
    )
    seasons_train = pd.DataFrame(
        [
            {"tm_player_id": 1, "season_end_date": date(2022, 6, 30), "minutes_played": 900, "goals": 5, "assists": 2},
        ]
    )
    vals_train = pd.DataFrame(
        [
            {
                "tm_player_id": 1,
                "league_at_valuation": "GB1",
                "valuation_date": date(2022, 6, 1),
                "market_value_eur": 10_000_000,
            },
            {
                "tm_player_id": 2,
                "league_at_valuation": "EC1",
                "valuation_date": date(2022, 6, 1),
                "market_value_eur": 500_000,
            },
        ]
    )

    priors_base = compute_empirical_bayes_priors(
        players_df=players_df,
        appearances_df=app_train,
        player_seasons_df=seasons_train,
        valuations_df=vals_train,
    )

    # Append extreme post-2023-07-01 rows
    app_with_future = pd.concat(
        [
            app_train,
            pd.DataFrame(
                [{"tm_player_id": 1, "match_date": date(2024, 5, 1), "minutes_played": 900, "goals": 999, "assists": 999}]
            ),
        ],
        ignore_index=True,
    )
    vals_with_future = pd.concat(
        [
            vals_train,
            pd.DataFrame(
                [
                    {
                        "tm_player_id": 1,
                        "league_at_valuation": "GB1",
                        "valuation_date": date(2024, 6, 1),
                        "market_value_eur": 999_000_000,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    priors_after = compute_empirical_bayes_priors(
        players_df=players_df,
        appearances_df=app_with_future,
        player_seasons_df=seasons_train,
        valuations_df=vals_with_future,
    )

    assert priors_base.to_dict() == priors_after.to_dict()
    # Both train valuations have >= 450 tracked minutes in the prior 365 days, so they
    # define the population-based fallback league medians.
    assert priors_base.train_league_median_eur["GB1"] == 10_000_000.0
    assert priors_base.train_league_median_eur["EC1"] == 500_000.0


def test_ec1_snapshot_without_match_appearances_sets_match_rolling_to_nan() -> None:
    """Verify EC1 snapshots without dcaribou match rows get `has_match_level_appearances == 0`
    and `NaN` for match-level rolling columns while retaining completed prior-season stats.
    """
    players_df = pd.DataFrame(
        [
            {
                "tm_player_id": 77,
                "full_name": "EC1 Midfielder",
                "normalized_name": "ec1 midfielder",
                "date_of_birth": date(2002, 3, 15),
                "nationality": "Ecuador",
                "position_group": "MID",
                "primary_position": "Central Midfield",
                "preferred_foot": "right",
                "height_cm": 178,
                "current_club": "Independiente del Valle",
                "current_league_code": "EC1",
                "contract_expires_at": None,
            }
        ]
    )
    empty_app = pd.DataFrame(
        columns=[
            "appearance_id",
            "game_id",
            "tm_player_id",
            "tm_club_id",
            "match_date",
            "competition_id",
            "season_year",
            "minutes_played",
            "goals",
            "assists",
            "yellow_cards",
            "red_cards",
            "goals_conceded",
            "clean_sheet",
        ]
    )
    seasons_df = pd.DataFrame(
        [
            {
                "tm_player_id": 77,
                "season": "2023/2024",
                "season_end_date": date(2023, 12, 31),
                "tm_club_id": 19309,
                "club_name": "Independiente del Valle",
                "league_code": "EC1",
                "appearances": 28,
                "minutes_played": 2340,
                "goals": 4,
                "assists": 6,
                "yellow_cards": 5,
                "red_cards": 0,
                "clean_sheets": 0,
                "goals_conceded": 0,
                "goals_per90": 0.1538,
                "assists_per90": 0.2308,
                "xg_per90": None,
                "xa_per90": None,
                "prog_actions_per90": None,
                "has_advanced_stats": False,
            }
        ]
    )
    empty_inj = pd.DataFrame(
        columns=["tm_player_id", "season_name", "injury_reason", "from_date", "end_date", "days_missed", "games_missed"]
    )
    empty_tr = pd.DataFrame(
        columns=[
            "tm_player_id",
            "transfer_date",
            "transfer_season",
            "from_club_id",
            "to_club_id",
            "from_club_name",
            "to_club_name",
            "transfer_fee_eur",
        ]
    )
    vals_df = pd.DataFrame(
        [
            {
                "tm_player_id": 77,
                "valuation_date": date(2022, 11, 1),
                "market_value_eur": 800_000,
                "tm_club_id_at_valuation": 19309,
                "club_at_valuation": "Independiente del Valle",
                "league_at_valuation": "EC1",
                "season_year": 2022,
                "cycle": "2022/2023",
                "age_at_valuation": 20.6,
                "valid_contract_years_remaining": None,
            }
        ]
    )

    snap_req = pd.DataFrame(
        [
            {
                "tm_player_id": 77,
                "as_of_date": date(2024, 6, 1),
                "tm_club_id": 19309,
                "club_name": "Independiente del Valle",
                "league_code": "EC1",
                "market_value_eur": 2_000_000,
                "season_year": 2023,
                "cycle": "2023/2024",
                "valid_contract_years_remaining": None,
            }
        ]
    )

    out = engineer_features_for_snapshots(
        snap_req,
        players_df=players_df,
        appearances_df=empty_app,
        player_seasons_df=seasons_df,
        injuries_df=empty_inj,
        transfers_df=empty_tr,
        valuations_df=vals_df,
        all_valuations_df=vals_df,
    )
    row = out.iloc[0]
    assert row["has_match_level_appearances"] == 0.0
    assert pd.isna(row["mins_last_12m"])
    assert pd.isna(row["goals_per90_shrunk_12m"])
    assert row["prior_season_minutes"] == 2340.0
    assert row["prior_season_goals"] == 4.0
    assert row["prior_season_goals_per90_shrunk"] > 0.0
    assert row["exclusion_reason"] == "league_not_supported"
    assert not bool(row["in_model_population"])


@pytest.mark.data
def test_feature_artifacts_and_serving_scope_invariants() -> None:
    manifest_path = FEATURES_DIR / "feature_manifest.json"
    snapshots_path = FEATURES_DIR / "valuation_snapshots.parquet"
    serving_path = FEATURES_DIR / "current_serving_features.parquet"
    if not manifest_path.exists() or not snapshots_path.exists() or not serving_path.exists():
        pytest.skip("Feature artifacts not present on clean clone")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots_df = pd.read_parquet(snapshots_path)
    serving_df = pd.read_parquet(serving_path)

    assert len(snapshots_df) == manifest["total_valuation_snapshots"]
    assert set(snapshots_df["split"].unique()) == {"train", "val_tune", "val_calib", "test"}
    assert (snapshots_df["trailing_league_median_eur"] > 0).all()
    assert snapshots_df["target_log_market_value"].notna().all()
    assert snapshots_df["target_log_ratio_vs_league_median"].notna().all()

    for col in MAIN_MODEL_FEATURES:
        assert col in snapshots_df.columns, f"Missing feature column {col} in valuation_snapshots.parquet"

    # Phase 2.1 modeling population: GB1 only, no EC1 rows inside it
    pop = snapshots_df[snapshots_df["in_model_population"]]
    assert (pop["league_code"] == "GB1").all()
    assert (pop["mins_last_12m"] >= 450).all()
    assert snapshots_df.loc[snapshots_df["league_code"] == "EC1", "exclusion_reason"].eq(
        "league_not_supported"
    ).all()
    for split in ("train", "val_tune", "val_calib", "test"):
        assert manifest["model_population_counts_by_split"][split] == int((pop["split"] == split).sum())

    # Active serving scope invariants: Premier League only, inside the modeling population
    assert len(serving_df) == manifest["active_serving_scope"]["total"]
    assert manifest["active_serving_scope"]["GB1"] > 300
    assert manifest["active_serving_scope"]["EC1"] == 0
    assert (serving_df["league_code"] == "GB1").all()
    assert (serving_df["trailing_league_median_eur"] > 0).all()
    assert serving_df["tm_player_id"].is_unique
    serving_minutes = serving_df["features_json"].map(lambda s: json.loads(s)["mins_last_12m"])
    assert (serving_minutes >= 450).all()
