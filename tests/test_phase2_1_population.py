"""Phase 2.1: Era-Stable Modeling Population Tests (`tests/test_phase2_1_population.py`).

Unit tests (run on any clean clone) cover the population rules, the population-based
trailing league median, and injury-source coverage. Data tests (`@pytest.mark.data`)
guard the rebuilt lakehouse against era-dependent feature drift between train and test.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.features.engineer import (
    CLUB_TENURE_CAP_DAYS,
    INJURY_RECENCY_CAP_DAYS,
    MAIN_MODEL_NUMERIC_FEATURES,
    POPULATION_MIN_TRACKED_MINUTES_12M,
    EmpiricalBayesPriors,
    engineer_features_for_snapshots,
)

FEATURES_DIR = Path("data/features")
T_EVAL = date(2024, 3, 1)
CLUB_A = 500
CLUB_B = 600

APP_COLUMNS = [
    "appearance_id", "game_id", "tm_player_id", "tm_club_id", "match_date", "competition_id",
    "season_year", "minutes_played", "goals", "assists", "yellow_cards", "red_cards",
    "goals_conceded", "clean_sheet",
]
SEASON_COLUMNS = ["tm_player_id", "season", "season_end_date", "appearances", "minutes_played", "goals", "assists"]
INJURY_COLUMNS = ["tm_player_id", "season_name", "injury_reason", "from_date", "end_date", "days_missed", "games_missed"]
TRANSFER_COLUMNS = ["tm_player_id", "transfer_date", "to_club_id", "from_club_id"]
VALUATION_COLUMNS = ["tm_player_id", "valuation_date", "market_value_eur", "league_at_valuation"]

PRIORS = EmpiricalBayesPriors(
    match_goals_per90_by_pos={"GK": 0.0, "DEF": 0.04, "MID": 0.12, "FWD": 0.30},
    match_assists_per90_by_pos={"GK": 0.0, "DEF": 0.04, "MID": 0.12, "FWD": 0.30},
    season_goals_per90_by_pos={"GK": 0.0, "DEF": 0.04, "MID": 0.12, "FWD": 0.30},
    season_assists_per90_by_pos={"GK": 0.0, "DEF": 0.04, "MID": 0.12, "FWD": 0.30},
    train_league_median_eur={"GB1": 8_000_000.0, "EC1": 400_000.0},
)


def _player(pid: int) -> dict:
    return {
        "tm_player_id": pid,
        "full_name": f"Test Player {pid}",
        "normalized_name": f"test player {pid}",
        "date_of_birth": date(1998, 1, 1),
        "nationality": "Testland",
        "position_group": "MID",
        "primary_position": "Central Midfield",
        "preferred_foot": "right",
        "height_cm": 180,
        "current_club": "Test FC Alpha",
        "current_league_code": "GB1",
        "contract_expires_at": None,
    }


def _apps(pid: int, club: int, n_matches: int, start: date = date(2023, 10, 1)) -> list[dict]:
    return [
        {
            "appearance_id": f"{pid}-{club}-{i}",
            "game_id": pid * 1000 + i,
            "tm_player_id": pid,
            "tm_club_id": club,
            "match_date": start + timedelta(days=7 * i),
            "competition_id": "GB1",
            "season_year": 2023,
            "minutes_played": 90,
            "goals": 0,
            "assists": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "goals_conceded": 1,
            "clean_sheet": 0,
        }
        for i in range(n_matches)
    ]


def _snap(pid: int, club: int = CLUB_A, league: str = "GB1") -> dict:
    return {
        "tm_player_id": pid,
        "as_of_date": T_EVAL,
        "tm_club_id": club,
        "club_name": "Test FC Alpha",
        "league_code": league,
        "market_value_eur": 10_000_000,
    }


def _frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def _run(
    snaps: list[dict],
    *,
    apps: list[dict],
    transfers: list[dict] | None = None,
    valuations: list[dict] | None = None,
    injury_coverage: list[int] | None = None,
) -> pd.DataFrame:
    player_ids = sorted({s["tm_player_id"] for s in snaps} | {a["tm_player_id"] for a in apps})
    vals_df = _frame(valuations or [], VALUATION_COLUMNS)
    return engineer_features_for_snapshots(
        pd.DataFrame(snaps),
        players_df=pd.DataFrame([_player(pid) for pid in player_ids]),
        appearances_df=_frame(apps, APP_COLUMNS),
        player_seasons_df=_frame([], SEASON_COLUMNS),
        injuries_df=_frame([], INJURY_COLUMNS),
        transfers_df=_frame(transfers or [], TRANSFER_COLUMNS),
        valuations_df=vals_df,
        all_valuations_df=vals_df,
        priors=PRIORS,
        injury_coverage_df=(
            None if injury_coverage is None else pd.DataFrame({"tm_player_id": injury_coverage})
        ),
    ).set_index("tm_player_id")


def test_population_rules_assign_expected_exclusion_reasons() -> None:
    apps = (
        _apps(1, CLUB_A, 10)  # regular at the valuation club -> in population
        + _apps(3, CLUB_B, 10)  # plays for another club, no transfer -> unverified club
        + _apps(4, CLUB_B, 10)  # plays for another club, then signs for CLUB_A -> in population
        + _apps(5, CLUB_A, 3)  # 270 tracked minutes -> insufficient
        + _apps(6, CLUB_A, 10)  # Ecuador snapshot -> unsupported league
    )
    transfers = [{"tm_player_id": 4, "transfer_date": date(2024, 1, 15), "to_club_id": CLUB_A, "from_club_id": CLUB_B}]
    out = _run(
        [_snap(1), _snap(2), _snap(3), _snap(4), _snap(5), _snap(6, league="EC1")],
        apps=apps,
        transfers=transfers,
    )

    assert POPULATION_MIN_TRACKED_MINUTES_12M == 450
    assert pd.isna(out.loc[1, "exclusion_reason"]) and bool(out.loc[1, "in_model_population"])
    assert out.loc[2, "exclusion_reason"] == "insufficient_tracked_minutes"
    assert out.loc[3, "exclusion_reason"] == "club_attribution_unverified"
    assert pd.isna(out.loc[4, "exclusion_reason"]) and bool(out.loc[4, "in_model_population"])
    assert out.loc[5, "exclusion_reason"] == "insufficient_tracked_minutes"
    assert out.loc[6, "exclusion_reason"] == "league_not_supported"

    # A player with no tracked appearance before t has unknown (NULL) match features, not zeros
    assert out.loc[2, "has_match_level_appearances"] == 0.0
    assert pd.isna(out.loc[2, "mins_last_12m"])
    assert out.loc[1, "mins_last_12m"] == 900.0


def test_club_tenure_is_appearance_based_and_capped() -> None:
    long_tenure = _apps(7, CLUB_A, 60, start=date(2021, 6, 1))  # first club app ~1,000 days before t
    out = _run(
        [_snap(1), _snap(4), _snap(7)],
        apps=_apps(1, CLUB_A, 10) + _apps(4, CLUB_B, 10) + long_tenure,
        transfers=[{"tm_player_id": 4, "transfer_date": date(2024, 1, 15), "to_club_id": CLUB_A, "from_club_id": CLUB_B}],
    )
    assert out.loc[1, "club_tenure_days_capped"] == float((T_EVAL - date(2023, 10, 1)).days)
    assert out.loc[4, "club_tenure_days_capped"] == 0.0  # new signing, no club appearance yet
    assert out.loc[7, "club_tenure_days_capped"] == float(CLUB_TENURE_CAP_DAYS)


def test_trailing_league_median_ignores_low_minute_valuations() -> None:
    apps = _apps(1, CLUB_A, 10) + _apps(5, CLUB_A, 3) + _apps(8, CLUB_A, 10)
    regular_val = {"tm_player_id": 1, "valuation_date": date(2024, 2, 1), "market_value_eur": 20_000_000, "league_at_valuation": "GB1"}
    low_minutes_val = {"tm_player_id": 5, "valuation_date": date(2024, 2, 1), "market_value_eur": 100_000, "league_at_valuation": "GB1"}
    second_regular_val = {"tm_player_id": 8, "valuation_date": date(2024, 2, 10), "market_value_eur": 30_000_000, "league_at_valuation": "GB1"}

    base = _run([_snap(1)], apps=apps, valuations=[regular_val])
    with_low = _run([_snap(1)], apps=apps, valuations=[regular_val, low_minutes_val])
    with_regular = _run([_snap(1)], apps=apps, valuations=[regular_val, second_regular_val])

    n_min = PRIORS.league_median_n_min
    expected = (1 / n_min) * 20_000_000 + (1 - 1 / n_min) * 8_000_000
    assert base.loc[1, "trailing_league_median_eur"] == pytest.approx(expected, abs=0.01)
    assert with_low.loc[1, "trailing_league_median_eur"] == base.loc[1, "trailing_league_median_eur"]
    assert with_regular.loc[1, "trailing_league_median_eur"] != base.loc[1, "trailing_league_median_eur"]


def test_injury_features_null_when_player_not_in_injury_source() -> None:
    out = _run([_snap(1), _snap(8)], apps=_apps(1, CLUB_A, 10) + _apps(8, CLUB_A, 10), injury_coverage=[1])
    assert out.loc[1, "in_injury_source"] == 1.0
    assert out.loc[1, "injury_days_12m"] == 0.0
    assert out.loc[1, "days_since_last_injury_end_capped"] == float(INJURY_RECENCY_CAP_DAYS)
    assert out.loc[8, "in_injury_source"] == 0.0
    assert pd.isna(out.loc[8, "injury_days_12m"])
    assert pd.isna(out.loc[8, "days_since_last_injury_end_capped"])


def _population() -> pd.DataFrame:
    path = FEATURES_DIR / "valuation_snapshots.parquet"
    if not path.exists():
        pytest.skip("Feature artifacts not present on clean clone")
    df = pd.read_parquet(path)
    return df[df["in_model_population"]]


@pytest.mark.data
def test_no_era_dependent_missingness_between_train_and_test() -> None:
    """Null and zero rates of every main-model numeric feature must match across eras.

    Coverage artifacts (e.g. transfer-based club tenure missing in 73% of 2013 rows and 0% of
    test rows) let trees learn the era instead of the player.
    """
    pop = _population()
    train, test = pop[pop["split"] == "train"], pop[pop["split"] == "test"]
    for col in MAIN_MODEL_NUMERIC_FEATURES:
        null_gap = abs(train[col].isna().mean() - test[col].isna().mean())
        zero_gap = abs((train[col] == 0).mean() - (test[col] == 0).mean())
        assert null_gap <= 0.05, f"{col}: null-rate gap {null_gap:.3f} between train and test"
        assert zero_gap <= 0.10, f"{col}: zero-rate gap {zero_gap:.3f} between train and test"


@pytest.mark.data
def test_trailing_league_index_tracks_population_level() -> None:
    """The normalization index must follow the population it normalizes: mean log ratio near 0."""
    pop = _population()
    for split in ("val_tune", "val_calib", "test"):
        mean_ratio = pop.loc[pop["split"] == split, "target_log_ratio_vs_league_median"].mean()
        assert abs(mean_ratio) <= 0.3, f"{split}: mean log ratio vs league index {mean_ratio:.3f}"


@pytest.mark.data
def test_population_club_attribution_is_corroborated() -> None:
    """Zero appearance-based tenure should only mean 'new signing', at a stable low rate."""
    pop = _population()
    zero_tenure = pop.groupby("split")["club_tenure_days_capped"].apply(lambda x: (x == 0).mean())
    assert (zero_tenure <= 0.05).all(), zero_tenure.to_dict()
