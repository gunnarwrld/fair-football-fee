# Phase 2.1: Era-Stable Modeling Population

**Decision:** the model is trained, evaluated and served only on **Premier League (`GB1`) valuation snapshots whose player had ≥ 450 tracked minutes in the 365 days before the valuation date, and whose valuation club is confirmed by match data.** Liga Pro Ecuador (`EC1`) is out of scope. The trailing league median used for target normalization is computed on that same population.

Rows outside the population stay in `valuation_snapshots.parquet` with `in_model_population = false` and an `exclusion_reason`, so the exclusions can be audited.

## Why: who is in the source changes in 2023

The source (dcaribou/transfermarkt-datasets) contains players who appear in a tracked match at some point. Old seasons therefore include fringe players who are present only because of things that happened later. Recent seasons haven't had time to accumulate them. The break falls exactly on the train/val_tune boundary.

| Evidence (GB1, before the fix) | 2013/14 – 2022/23 (train) | 2023/24 onward |
|---|---|---|
| Rows with zero tracked minutes in prior 12 months | 21–32% | 4–9% |
| ...of which the first tracked appearance comes **after** the valuation date | 100% | 100% |
| Median value, players still active in 2023+ vs. players who left the tracked leagues | €6–15M vs. €0.3–2M | all players still active |
| Distinct PL players with valuations in the raw source (2022 → 2025) | 1,360 | 948 → 794 → 756 |
| `days_since_joined_club` missing (transfer-based) | 73% (2013/14) → 27% | 3% → 0% |
| Same-player median change in log value between consecutive valuations | 0.000 in every season | 0.000 |

The median valuation jumped from €7M to €15M in a single season while the same players' values did not move. That is a change in population, not inflation. The old trailing-median index followed it, giving a mean log ratio of **+0.13 in val_tune vs. −0.28 in test**: a systematic bias between splits.

## Fix and result

1. **Population:** `GB1` and ≥ 450 tracked minutes in the prior 365 days (`insufficient_tracked_minutes` otherwise).
2. **Club attribution:** the valuation's club must match the club of the player's latest tracked appearance, or the player must have signed for that club since then (`club_attribution_unverified` otherwise). The source's club field matches match data in 70.5% of 2013/14 GB1 rows but 99.8% of 2025/26 rows. This rule removed 788 train rows, 1 val_tune row and none from val_calib or test.
3. **League index:** the trailing 365-day median uses only prior valuations that themselves meet the minutes rule, point-in-time per valuation.
4. **Era-dependent features replaced:** transfer-based tenure becomes `club_tenure_days_capped` (first appearance for the club, capped at 730 days). Uncapped injury recency becomes `days_since_last_injury_end_capped` (730 days). Injury features are NULL when the player is not in the injury source (`in_injury_source`), rather than a misleading zero. Constant-within-population columns (`league_code`, `has_match_level_appearances`, `has_advanced_stats`) are no longer model features.

| Season | Split | Rows | Median value | League index | Mean log ratio |
|---|---|---|---|---|---|
| 2013/14 | train | 420 | €4.8M | €4.0M | −0.15 |
| 2016/17 | train | 903 | €8.0M | €8.0M | −0.06 |
| 2019/20 | train | 888 | €15.0M | €15.0M | −0.06 |
| 2022/23 | train | 1,163 | €18.0M | €16.0M | −0.07 |
| 2023/24 | val_tune | 1,118 | €20.0M | €18.0M | −0.13 |
| 2024/25 | val_calib | 1,101 | €22.0M | €20.0M | −0.07 |
| 2025/26 | test | 1,288 | €23.5M | €22.0M | −0.12 |

The series is now smooth and matches real Premier League inflation. Every main-model feature's null rate is within 1.5 points between train and test (`tests/test_phase2_1_population.py` enforces ≤ 5 points for null rates and ≤ 10 for zero rates).

**Population sizes:** train 8,401 · val_tune 1,118 · val_calib 1,101 · test 1,288. **Serving:** 439 Premier League players. 86 otherwise-active players are excluded for having fewer than 450 tracked minutes; the API should report them as unsupported with that reason.

## Why Ecuador is out of scope

Only 27% of `EC1` train rows have any prior-season statistics and fewer than 1% have match-level data, so the model would mainly see age, position and height. The `EC1` rows are also a survivorship sample: Ecuador-stage valuations of players who later reached Transfermarkt-tracked clubs. Under the minutes rule, about 180 rows would remain in total, with single digits in test. That's too few to report honest metrics, so Ecuador is documented as a limitation of the open data rather than a supported league.

## Known follow-up (Phase 1)

The proper fix for club attribution is to derive club-at-valuation from appearances during cleaning, so that `league_at_valuation`, and therefore which rows count as `GB1`, is also corrected. The current rule excludes unverified rows instead of re-attributing them, which is conservative but loses about 8% of train.
