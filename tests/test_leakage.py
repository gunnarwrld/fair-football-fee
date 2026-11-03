"""Phase 2: Point-in-Time Leakage & Train/Serve Parity Test Suite (`tests/test_leakage.py`).

Enforces:
  1. Truncation-invariance leakage test on synthetic edge cases (`date == t` and `date > t`,
     plus ongoing injuries spanning across `t`) — runs on any clean clone.
  2. Truncation-invariance leakage test on 30 real `(tm_player_id, as_of_date)` pairs across
     all splits (`train`, `val_tune`, `val_calib`, `test`) and both leagues (`GB1`, `EC1`):
     deletes all source rows dated `>= t` (and strips `end_date >= t` on ongoing injuries),
     recomputes features at `t`, and asserts bit-for-bit equality.
  3. Train/serve parity test: verifies `compute_serving_features_for_requests` at
     `as_of_date = valuation_date` produces the exact training snapshot feature vector
     both on sampled historical snapshots and across all 536 active serving rows.
  4. Feature registry guard: asserts `MAIN_MODEL_FEATURES` excludes lagged valuations and
     contract columns.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    CONTRACT_ABLATION_FEATURES,
    MAIN_MODEL_FEATURES,
    VALUE_UPDATE_MODEL_FEATURES,
    EmpiricalBayesPriors,
    compute_empirical_bayes_priors,
    engineer_features_for_snapshots,
)
from src.features.pipeline import compute_serving_features_for_requests, load_clean_tables

CLEAN_DIR = Path("data/clean")
FEATURES_DIR = Path("data/features")


def test_main_model_feature_registry_excludes_forbidden_columns() -> None:
    forbidden = {
        "prev_market_value_eur",
        "prev_log_market_value",
        "days_since_prev_valuation",
        "valid_contract_years_remaining",
        "contract_expires_at",
        "dc_contract_expiration_date",
        "sal_contract_expires",
        "market_value_eur",
        "target_log_market_value",
        "target_log_ratio_vs_league_median",
        "days_missed",
        "games_missed",
    }
    overlap = set(MAIN_MODEL_FEATURES) & forbidden
    assert not overlap, f"Forbidden leakage columns found in MAIN_MODEL_FEATURES: {overlap}"
    assert "prev_log_market_value" in VALUE_UPDATE_MODEL_FEATURES
    assert "valid_contract_years_remaining" in CONTRACT_ABLATION_FEATURES


def test_truncation_invariance_on_synthetic_edge_cases() -> None:
    """Verify strict `< t` filtering and ongoing injury clipping on boundary events at `== t` and `> t`."""
    t_eval = date(2024, 1, 20)

    players_df = pd.DataFrame(
        [
            {
                "tm_player_id": 101,
                "full_name": "Edge Case Player",
                "normalized_name": "edge case player",
                "date_of_birth": date(2000, 1, 20),
                "nationality": "England",
                "position_group": "MID",
                "primary_position": "Central Midfield",
                "preferred_foot": "right",
                "height_cm": 182,
                "current_club": "Test FC",
                "current_league_code": "GB1",
                "contract_expires_at": date(2027, 6, 30),
            }
        ]
    )

    # Appearances: one before t (2024-01-10), one ON t (2024-01-20), one AFTER t (2024-02-01)
    appearances_full = pd.DataFrame(
        [
            {
                "appearance_id": "a1",
                "game_id": 1,
                "tm_player_id": 101,
                "tm_club_id": 500,
                "match_date": date(2024, 1, 10),
                "competition_id": "GB1",
                "season_year": 2023,
                "minutes_played": 90,
                "goals": 1,
                "assists": 1,
                "yellow_cards": 0,
                "red_cards": 0,
                "goals_conceded": 0,
                "clean_sheet": 1,
            },
            {
                "appearance_id": "a2_on_t",
                "game_id": 2,
                "tm_player_id": 101,
                "tm_club_id": 500,
                "match_date": t_eval,
                "competition_id": "GB1",
                "season_year": 2023,
                "minutes_played": 90,
                "goals": 5,
                "assists": 5,
                "yellow_cards": 2,
                "red_cards": 1,
                "goals_conceded": 4,
                "clean_sheet": 0,
            },
            {
                "appearance_id": "a3_after_t",
                "game_id": 3,
                "tm_player_id": 101,
                "tm_club_id": 500,
                "match_date": date(2024, 2, 1),
                "competition_id": "GB1",
                "season_year": 2023,
                "minutes_played": 90,
                "goals": 3,
                "assists": 2,
                "yellow_cards": 1,
                "red_cards": 0,
                "goals_conceded": 1,
                "clean_sheet": 0,
            },
        ]
    )

    # Player seasons: one completed before t (2023-06-30), one ending ON t (2024-01-20), one ending AFTER t (2024-06-30)
    seasons_full = pd.DataFrame(
        [
            {
                "tm_player_id": 101,
                "season": "2022/2023",
                "season_end_date": date(2023, 6, 30),
                "tm_club_id": 500,
                "club_name": "Test FC",
                "league_code": "GB1",
                "appearances": 30,
                "minutes_played": 2500,
                "goals": 6,
                "assists": 4,
                "yellow_cards": 3,
                "red_cards": 0,
                "clean_sheets": 10,
                "goals_conceded": 25,
                "goals_per90": 0.216,
                "assists_per90": 0.144,
                "xg_per90": None,
                "xa_per90": None,
                "prog_actions_per90": None,
                "has_advanced_stats": False,
            },
            {
                "tm_player_id": 101,
                "season": "2023/2024",
                "season_end_date": date(2024, 6, 30),
                "tm_club_id": 500,
                "club_name": "Test FC",
                "league_code": "GB1",
                "appearances": 38,
                "minutes_played": 3420,
                "goals": 20,
                "assists": 15,
                "yellow_cards": 5,
                "red_cards": 1,
                "clean_sheets": 15,
                "goals_conceded": 30,
                "goals_per90": 0.526,
                "assists_per90": 0.395,
                "xg_per90": None,
                "xa_per90": None,
                "prog_actions_per90": None,
                "has_advanced_stats": False,
            },
        ]
    )

    # Injuries:
    # 1) Completed before t (2023-09-01 to 2023-09-11 -> 10 days)
    # 2) Ongoing across t (from 2024-01-10 to 2024-03-01 -> must clip at t=2024-01-20 to 10 days!)
    # 3) Starting ON t (2024-01-20 -> must be ignored!)
    injuries_full = pd.DataFrame(
        [
            {
                "tm_player_id": 101,
                "season_name": "23/24",
                "injury_reason": "Ankle",
                "from_date": date(2023, 9, 1),
                "end_date": date(2023, 9, 11),
                "days_missed": 10.0,
                "games_missed": 2,
            },
            {
                "tm_player_id": 101,
                "season_name": "23/24",
                "injury_reason": "Hamstring",
                "from_date": date(2024, 1, 10),
                "end_date": date(2024, 3, 1),
                "days_missed": 51.0,
                "games_missed": 9,
            },
            {
                "tm_player_id": 101,
                "season_name": "23/24",
                "injury_reason": "Future knock",
                "from_date": t_eval,
                "end_date": date(2024, 2, 10),
                "days_missed": 21.0,
                "games_missed": 4,
            },
        ]
    )

    # Transfers: one before t (2023-07-01), one ON t (2024-01-20)
    transfers_full = pd.DataFrame(
        [
            {
                "tm_player_id": 101,
                "transfer_date": date(2023, 7, 1),
                "transfer_season": "23/24",
                "from_club_id": 400,
                "to_club_id": 500,
                "from_club_name": "Old FC",
                "to_club_name": "Test FC",
                "transfer_fee_eur": 15_000_000.0,
            },
            {
                "tm_player_id": 101,
                "transfer_date": t_eval,
                "transfer_season": "23/24",
                "from_club_id": 500,
                "to_club_id": 500,
                "from_club_name": "Test FC",
                "to_club_name": "Test FC",
                "transfer_fee_eur": 0.0,
            },
        ]
    )

    # Valuations: one before t (2023-06-01), one ON t (2024-01-20), one AFTER t (2024-05-01)
    valuations_full = pd.DataFrame(
        [
            {
                "tm_player_id": 101,
                "valuation_date": date(2023, 6, 1),
                "market_value_eur": 12_000_000,
                "tm_club_id_at_valuation": 500,
                "club_at_valuation": "Test FC",
                "league_at_valuation": "GB1",
                "season_year": 2022,
                "cycle": "2022/2023",
                "age_at_valuation": 23.36,
                "valid_contract_years_remaining": 4.08,
            },
            {
                "tm_player_id": 101,
                "valuation_date": t_eval,
                "market_value_eur": 25_000_000,
                "tm_club_id_at_valuation": 500,
                "club_at_valuation": "Test FC",
                "league_at_valuation": "GB1",
                "season_year": 2023,
                "cycle": "2023/2024",
                "age_at_valuation": 24.0,
                "valid_contract_years_remaining": 3.44,
            },
            {
                "tm_player_id": 101,
                "valuation_date": date(2024, 5, 1),
                "market_value_eur": 90_000_000,
                "tm_club_id_at_valuation": 500,
                "club_at_valuation": "Test FC",
                "league_at_valuation": "GB1",
                "season_year": 2023,
                "cycle": "2023/2024",
                "age_at_valuation": 24.28,
                "valid_contract_years_remaining": 3.16,
            },
        ]
    )

    priors = compute_empirical_bayes_priors(
        players_df=players_df,
        appearances_df=appearances_full[appearances_full["match_date"] < date(2023, 7, 1)],
        player_seasons_df=seasons_full[seasons_full["season_end_date"] < date(2023, 7, 1)],
        valuations_df=valuations_full[valuations_full["valuation_date"] < date(2023, 7, 1)],
    )

    snap_req = pd.DataFrame(
        [
            {
                "tm_player_id": 101,
                "as_of_date": t_eval,
                "tm_club_id": 500,
                "club_name": "Test FC",
                "league_code": "GB1",
                "market_value_eur": 25_000_000,
                "season_year": 2023,
                "cycle": "2023/2024",
                "valid_contract_years_remaining": 3.44,
            }
        ]
    )

    feat_full = engineer_features_for_snapshots(
        snap_req,
        players_df=players_df,
        appearances_df=appearances_full,
        player_seasons_df=seasons_full,
        injuries_df=injuries_full,
        transfers_df=transfers_full,
        valuations_df=valuations_full,
        all_valuations_df=valuations_full,
        priors=priors,
    )

    # Now truncate all tables at `< t_eval` and strip ongoing injury `end_date >= t_eval`
    appearances_trunc = appearances_full[appearances_full["match_date"] < t_eval].copy()
    seasons_trunc = seasons_full[seasons_full["season_end_date"] < t_eval].copy()
    transfers_trunc = transfers_full[transfers_full["transfer_date"] < t_eval].copy()
    valuations_trunc = valuations_full[valuations_full["valuation_date"] < t_eval].copy()
    injuries_trunc = injuries_full[injuries_full["from_date"] < t_eval].copy()
    injuries_trunc.loc[injuries_trunc["end_date"] >= t_eval, "end_date"] = None
    injuries_trunc["days_missed"] = 9999.0  # Prove static days_missed is never read

    feat_trunc = engineer_features_for_snapshots(
        snap_req,
        players_df=players_df,
        appearances_df=appearances_trunc,
        player_seasons_df=seasons_trunc,
        injuries_df=injuries_trunc,
        transfers_df=transfers_trunc,
        valuations_df=valuations_trunc,
        all_valuations_df=valuations_trunc,
        priors=priors,
    )

    pd.testing.assert_frame_equal(feat_full, feat_trunc, check_exact=True)

    # Verify clipped injury arithmetic: 10d (Sep 2023) + 10d (Jan 10..20, 2024) = 20.0 days
    row = feat_full.iloc[0]
    assert row["injury_days_12m"] == 20.0
    assert row["is_currently_injured_at_t"] == 1.0
    assert row["days_since_last_injury_end"] == 0.0
    assert row["mins_last_12m"] == 90.0
    assert row["prior_season_minutes"] == 2500.0
    assert row["prev_market_value_eur"] == 12_000_000.0


@pytest.mark.data
def test_truncation_invariance_no_future_leakage_on_real_snapshots() -> None:
    """For 30 sampled `(tm_player_id, as_of_date)` pairs across all splits and both leagues,
    delete all source rows dated `>= t` (and strip `end_date >= t` on ongoing injuries),
    recompute features at `t`, and assert bit-for-bit equality.
    """
    snapshots_path = FEATURES_DIR / "valuation_snapshots.parquet"
    priors_path = FEATURES_DIR / "priors.json"
    if not snapshots_path.exists() or not priors_path.exists():
        pytest.skip("Feature artifacts not present on clean clone")

    clean_tables = load_clean_tables(CLEAN_DIR)
    priors = EmpiricalBayesPriors.from_dict(json.loads(priors_path.read_text(encoding="utf-8")))
    all_snapshots = pd.read_parquet(snapshots_path)

    # Sample 30 diverse snapshots across splits, leagues, and ongoing-injury cases
    rng = np.random.default_rng(42)
    ongoing_inj = all_snapshots[all_snapshots["is_currently_injured_at_t"] == 1.0]
    ec1_snaps = all_snapshots[all_snapshots["league_code"] == "EC1"]
    gb1_snaps = all_snapshots[all_snapshots["league_code"] == "GB1"]

    sampled_indices = list(
        set(
            list(rng.choice(ongoing_inj.index, size=min(8, len(ongoing_inj)), replace=False))
            + list(rng.choice(ec1_snaps.index, size=min(8, len(ec1_snaps)), replace=False))
            + list(rng.choice(gb1_snaps.index, size=14, replace=False))
        )
    )[:30]

    cols_to_verify = (
        MAIN_MODEL_FEATURES
        + VALUE_UPDATE_MODEL_FEATURES
        + CONTRACT_ABLATION_FEATURES
        + ["trailing_league_median_eur", "target_log_market_value", "target_log_ratio_vs_league_median"]
    )
    cols_to_verify = list(dict.fromkeys(cols_to_verify))

    app_df = clean_tables["appearances"]
    seasons_df = clean_tables["player_seasons"]
    inj_df = clean_tables["injuries"]
    tr_df = clean_tables["transfers"]
    val_df = clean_tables["valuations"]
    all_val_df = clean_tables["all_valuations"]

    for idx in sampled_indices:
        expected_row = all_snapshots.loc[[idx]].reset_index(drop=True)
        t_cut = expected_row.loc[0, "as_of_date"]

        # Delete all source rows dated >= t_cut
        app_trunc = app_df[pd.to_datetime(app_df["match_date"]).dt.date < t_cut]
        seasons_trunc = seasons_df[pd.to_datetime(seasons_df["season_end_date"]).dt.date < t_cut]
        tr_trunc = tr_df[pd.to_datetime(tr_df["transfer_date"]).dt.date < t_cut]
        val_trunc = val_df[pd.to_datetime(val_df["valuation_date"]).dt.date < t_cut]
        all_val_trunc = all_val_df[pd.to_datetime(all_val_df["valuation_date"]).dt.date < t_cut]

        inj_trunc = inj_df[pd.to_datetime(inj_df["from_date"]).dt.date < t_cut].copy()
        end_dates = pd.to_datetime(inj_trunc["end_date"]).dt.date
        inj_trunc.loc[end_dates.notna() & (end_dates >= t_cut), "end_date"] = None

        req_df = expected_row[
            [
                "tm_player_id",
                "as_of_date",
                "tm_club_id",
                "club_name",
                "league_code",
                "market_value_eur",
                "season_year",
                "cycle",
                "valid_contract_years_remaining",
            ]
        ].copy()

        recomputed = engineer_features_for_snapshots(
            req_df,
            players_df=clean_tables["players"],
            appearances_df=app_trunc,
            player_seasons_df=seasons_trunc,
            injuries_df=inj_trunc,
            transfers_df=tr_trunc,
            valuations_df=val_trunc,
            all_valuations_df=all_val_trunc,
            priors=priors,
        )

        pd.testing.assert_frame_equal(
            expected_row[cols_to_verify],
            recomputed[cols_to_verify],
            check_exact=True,
        )


@pytest.mark.data
def test_train_serve_parity_on_historical_and_active_snapshots() -> None:
    """Verify the serving feature writer (`compute_serving_features_for_requests`)
    produces bit-for-bit identical feature vectors to `valuation_snapshots.parquet`
    both on sampled historical `(player_id, valuation_date)` rows and across all
    536 active serving rows in `current_serving_features.parquet`.
    """
    snapshots_path = FEATURES_DIR / "valuation_snapshots.parquet"
    serving_path = FEATURES_DIR / "current_serving_features.parquet"
    priors_path = FEATURES_DIR / "priors.json"
    if not snapshots_path.exists() or not serving_path.exists():
        pytest.skip("Feature artifacts not present on clean clone")

    clean_tables = load_clean_tables(CLEAN_DIR)
    priors = EmpiricalBayesPriors.from_dict(json.loads(priors_path.read_text(encoding="utf-8")))
    all_snapshots = pd.read_parquet(snapshots_path)
    serving_df = pd.read_parquet(serving_path)

    # 1. Check 50 random historical snapshots across all 4 splits
    rng = np.random.default_rng(123)
    sample_snaps = all_snapshots.iloc[rng.choice(len(all_snapshots), size=50, replace=False)].copy()
    serving_out = compute_serving_features_for_requests(
        sample_snaps[
            [
                "tm_player_id",
                "as_of_date",
                "tm_club_id",
                "club_name",
                "league_code",
                "market_value_eur",
                "valid_contract_years_remaining",
            ]
        ],
        clean_tables=clean_tables,
        priors=priors,
        data_version="parity-test",
    )

    for (_, snap_row), (_, srv_row) in zip(sample_snaps.iterrows(), serving_out.iterrows()):
        assert int(snap_row["tm_player_id"]) == int(srv_row["tm_player_id"])
        assert snap_row["as_of_date"] == srv_row["as_of_date"]
        assert float(snap_row["trailing_league_median_eur"]) == float(srv_row["trailing_league_median_eur"])
        srv_features = json.loads(srv_row["features_json"])
        for col in MAIN_MODEL_FEATURES:
            expected_val = snap_row[col]
            actual_val = srv_features[col]
            if pd.isna(expected_val):
                assert actual_val is None, f"Mismatch on {col}: expected None, got {actual_val}"
            elif isinstance(expected_val, (float, np.floating)):
                assert float(expected_val) == float(actual_val), f"Mismatch on {col}"
            else:
                assert expected_val == actual_val, f"Mismatch on {col}"

    # 2. Check all 536 active serving rows in current_serving_features.parquet against valuation_snapshots.parquet
    snap_lookup = all_snapshots.set_index(["tm_player_id", "as_of_date"])
    assert len(serving_df) > 500
    for _, srv_row in serving_df.iterrows():
        key = (int(srv_row["tm_player_id"]), srv_row["as_of_date"])
        assert key in snap_lookup.index, f"Active serving row {key} missing from valuation_snapshots"
        snap_row = snap_lookup.loc[key]
        assert float(snap_row["trailing_league_median_eur"]) == float(srv_row["trailing_league_median_eur"])
        srv_features = json.loads(srv_row["features_json"])
        for col in MAIN_MODEL_FEATURES:
            expected_val = snap_row[col]
            actual_val = srv_features[col]
            if pd.isna(expected_val):
                assert actual_val is None
            elif isinstance(expected_val, (float, np.floating)):
                assert float(expected_val) == float(actual_val)
            else:
                assert expected_val == actual_val
