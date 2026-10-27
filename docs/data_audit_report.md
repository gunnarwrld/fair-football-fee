# Phase 1 Pre-Ingestion Data & License Audit Report

## 1. Repository License Verification & Pinned Sources
| Repository | LICENSE File Present | GitHub API SPDX | Note |
| :--- | :--- | :--- | :--- |
| `dcaribou/transfermarkt-datasets` | `True` | `CC0-1.0` | Creative Commons Zero v1.0 Universal |
| `salimt/football-datasets` | `False` | `None` | No LICENSE file in GitHub repository (GitHub API license=null -> all rights reserved by default under copyright law, despite third-party mirrors claiming MIT/CC0). Raw and derived Parquet files must remain gitignored and raw injury records must never be exposed through the API. |

### Pinned Raw Source Files & CSV Reject Audit (`store_rejects=true`)
| Source File | Pinned URL / Commit | SHA-256 (Prefix) | Size (Bytes) | CSV Rows Read | CSV Reject Count |
| :--- | :--- | :--- | ---: | ---: | ---: |
| `dcaribou/competitions.csv.gz` | `r2-pinned` | `8924ddfbc0e9...` | 2,242 | 65 | 0 |
| `dcaribou/clubs.csv.gz` | `r2-pinned` | `aa57ab56e87f...` | 51,281 | 796 | 0 |
| `dcaribou/players.csv.gz` | `r2-pinned` | `d22e407981d5...` | 4,389,958 | 50,149 | 0 |
| `dcaribou/player_valuations.csv.gz` | `r2-pinned` | `608e32e5ad22...` | 7,197,935 | 656,301 | 0 |
| `dcaribou/appearances.csv.gz` | `r2-pinned` | `e096c2b158fc...` | 45,253,638 | 1,894,350 | 0 |
| `dcaribou/games.csv.gz` | `r2-pinned` | `142561989017...` | 4,995,595 | 88,958 | 0 |
| `dcaribou/transfers.csv.gz` | `r2-pinned` | `90326983daf7...` | 5,809,514 | 175,165 | 0 |
| `salimt/player_profiles.csv` | `62ca05786a` | `7ef9afbacf97...` | 26,380,409 | 92,671 | 0 |
| `salimt/player_market_value.csv` | `da9227e1f4` | `e9e1868c883f...` | 23,873,747 | 901,429 | 0 |
| `salimt/player_injuries_updated_20260929.csv` | `da9227e1f4` | `c622ed5684ec...` | 9,455,168 | 168,290 | 0 |
| `salimt/team_competitions_seasons.csv` | `da9227e1f4` | `83f06c8108e8...` | 11,032,370 | 58,247 | 0 |
| `salimt/player_performances.csv` | `da9227e1f4` | `21eda6556654...` | 157,647,289 | 1,878,716 | 3 |

## 2. Canonical Authoritative Source per Table
| Entity / Table | Canonical Source |
| :--- | :--- |
| `players` | dcaribou/players.csv.gz (position 'Missing' backfilled from salimt/player_profiles.csv) |
| `valuations` | dcaribou/player_valuations.csv.gz (sole valuation source; includes current_club_id at valuation date) |
| `appearances` | dcaribou/appearances.csv.gz + dcaribou/games.csv.gz (match-level; player_seasons is display-only) |
| `transfers` | dcaribou/transfers.csv.gz (point-in-time days_since_joined_club < valuation_date) |
| `injuries` | salimt/player_injuries_updated_20260929.csv (lakehouse-only; clipped at valuation date t) |
| `club_season_league_lookup` | dcaribou/games.csv.gz + salimt/team_competitions_seasons.csv (tier_level=1) + salimt/player_performances.csv (EC_COMP_IDS Tier 1 excluding EC2L Serie B) |

## 3. Entity Resolution Audit (`dcaribou` vs. `salimt`)
### 3.1 Global `dcaribou` Coverage vs. ID-Namespace Verification
- `dcaribou` total players: **50,149**
- Matched in `salimt/player_profiles.csv` by `tm_player_id` (coverage): **30,361 (60.54%)**
- Exact `date_of_birth` agreement among ID-matched rows (ID-namespace verification): **30,230 / 30,361 (99.57%)**

### 3.2 Target Cohort Coverage (`GB1` + `EC1` Clean Players & Valuations)
- Target cohort distinct players: **2,551**
- Target cohort players matched in `salimt/player_profiles.csv` by `tm_player_id`: **2,357 / 2,551 (92.40%)**
- Target cohort valuations matched in `salimt/player_profiles.csv` by `tm_player_id`: **18,941 / 19,938 (95.00%)**
- Exact `date_of_birth` agreement among ID-matched target-cohort players: **2,349 / 2,357 (99.66%)**
- Target cohort players with injury records in `salimt/player_injuries_updated_20260929.csv`: **1,989 / 2,551 (77.97%)**
- Unmatched target cohort players (`194`) recovered by `(LOWER(name), date_of_birth)` in `salimt`: **0**

## 4. Constraint & Anomaly Checks
### 4.1 Position Values (`ck_players_position_group`)
| `dcaribou` Position | Resolved Position (with `salimt` backfill) | Player Count | Action |
| :--- | :--- | ---: | :--- |
| `Defender` | `Defender` | 15,961 | Map to GK/DEF/MID/FWD |
| `Midfield` | `Midfield` | 14,272 | Map to GK/DEF/MID/FWD |
| `Attack` | `Attack` | 13,502 | Map to GK/DEF/MID/FWD |
| `Goalkeeper` | `Goalkeeper` | 5,828 | Map to GK/DEF/MID/FWD |
| `Missing` | `Missing` | 491 | Drop (unresolvable Missing) |
| `Missing` | `Midfield` | 43 | Map to GK/DEF/MID/FWD |
| `Missing` | `Attack` | 28 | Map to GK/DEF/MID/FWD |
| `Missing` | `Defender` | 24 | Map to GK/DEF/MID/FWD |

### 4.2 Missing Club / League & Active Serving Scope (`player_current_features`)
- Total `dcaribou` players: **50,149**
- `current_club_domestic_competition_id IS NULL`: **2,986** (retired, free agents, or non-headline clubs)
- **Active Supported Serving Scope (`player_current_features` rows)**: **536** players (`GB1`: **525**, `EC1`: **11**)
- **Policy**: Populate players.current_league_code with COALESCE(current_club_domestic_competition_id, last_valuation_league_code, 'UNKNOWN') to satisfy NOT NULL on historical players, but NEVER use players.current_league_code to define active API serving scope. Instead, active serving scope is defined exclusively by row existence in player_current_features (players with a valuation in the last ~12 months >= 2025-06-12, not retired/unattached, whose club's 2025 season league is GB1 or EC1): 525 GB1 players and 11 EC1 players (536 total). Live API serving is therefore effectively Premier League (97.9% GB1), while EC1 is primarily an historical evaluation-only OOD slice.

### 4.3 Valuation Cleaning Drop Log
- Raw valuation rows (`dcaribou/player_valuations.csv.gz`): **656,301**
- Dropped (`market_value_in_eur <= 0` or `NULL`): **1**
- Dropped (duplicate `(player_id, valuation_date)`): **0**
- Dropped (`date_of_birth IS NULL` -> `age_at_valuation` unknown): **600**
- Dropped (`position = 'Missing'` after `salimt` backfill): **404**
- Dropped (`valuation_date < 2013-07-01`, prior to appearance coverage window): **100,805**
- Dropped (`valuation_date >= 2026-07-01`, incomplete current cycle): **0**
- **Remaining Clean Valuations (All Leagues, 2013/14–2025/26)**: **554,491**
- Excluded other leagues (`league_at_valuation NOT IN ('GB1', 'EC1')`, including Ecuador Serie B `EC2L`): **534,553**
- **Remaining Clean Target Cohort Valuations (`GB1` + `EC1`, 2013/14–2025/26)**: **19,938** (`GB1`: **19,273** across **2,421** players; `EC1`: **665** across **137** players at **26** Ecuadorian clubs)

### 4.4 Promotion & Relegation Leakage Audit (`league_at_valuation` & `tier_level = 1`)
- Total valuations (`2013/14`–`2025/26`) with static `player_club_domestic_competition_id = 'GB1'`: **27,605**
- True point-in-time `GB1` valuations (club was one of the 20 Premier League clubs in `dcaribou/games.csv.gz` that season): **19,273**
- Relegated/lower-tier club valuations falsely labeled `GB1` by static club table: **8,332 (30.18% of static `GB1` rows)**
- All-league static vs. point-in-time mismatches (`team_competitions_seasons`): **98,092**
- Valuations at ever-Tier-1 Ecuadorian clubs (`2013/14`–`2025/26`) verified in `EC1` (Serie A Tier 1) for that calendar year: **665** (excluded because the club was in `EC2L` Serie B or pre-promotion in that calendar year: **23**)

## 5. July–June Cycle Breakdown & Split Cutoffs (`GB1` vs. `EC1` Survivorship OOD Slice)
| Cycle | `GB1` Valuations | `GB1` Players | `GB1` Valid Contract % | `GB1` Injury History % | `EC1` Valuations | `EC1` Players | `EC1` Valid Contract % | `EC1` Injury History % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2013/2014` | 997 | 528 | 0.0% | 82.7% | 8 | 4 | 0.0% | 75.0% |
| `2014/2015` | 971 | 540 | 0.0% | 85.1% | 10 | 5 | 0.0% | 60.0% |
| `2015/2016` | 1,265 | 638 | 0.2% | 87.6% | 6 | 6 | 0.0% | 66.7% |
| `2016/2017` | 1,676 | 685 | 0.0% | 88.2% | 18 | 11 | 0.0% | 55.6% |
| `2017/2018` | 1,240 | 621 | 0.2% | 90.3% | 33 | 18 | 0.0% | 42.4% |
| `2018/2019` | 1,394 | 629 | 0.1% | 93.0% | 72 | 33 | 0.0% | 59.7% |
| `2019/2020` | 1,498 | 691 | 0.2% | 92.2% | 50 | 34 | 0.0% | 52.0% |
| `2020/2021` | 2,190 | 769 | 0.6% | 91.5% | 83 | 36 | 0.0% | 44.6% |
| `2021/2022` | 1,572 | 749 | 3.3% | 91.9% | 62 | 39 | 0.0% | 48.4% |
| `2022/2023` | 2,105 | 908 | 8.6% | 85.4% | 46 | 43 | 0.0% | 41.3% |
| `2023/2024` | 1,448 | 655 | 28.2% | 91.5% | 125 | 59 | 3.2% | 33.6% |
| `2024/2025` | 1,387 | 577 | 51.6% | 92.2% | 96 | 52 | 11.5% | 18.8% |
| `2025/2026` | 1,530 | 565 | 15.2% | 86.6% | 56 | 35 | 3.6% | 21.4% |

> **Note on `EC1` Survivorship OOD Slice & Live Serving Scope**: With **665 Tier-1 valuations across 137 players at 26 clubs** historically, and only **11 active `EC1` players** vs. **525 active `GB1` players** in `player_current_features`, live API serving is effectively Premier League (`GB1`), while `EC1` is an historical evaluation-only out-of-distribution (OOD) slice of Ecuador-stage valuations for players who reached Transfermarkt-tracked clubs. Phase 3 reports player-level block bootstrap confidence intervals on the `56` test-split `EC1` valuations.

### Temporal Split Cutoffs & Computed Row Counts
| Split | Cycles | Min Valuation Date | Max Valuation Date | Total Valuations | `GB1` Valuations | `EC1` Valuations | `EC1` Players |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `train` | `2013/2014..2022/2023` | `2013-07-02` | `2023-06-30` | 15,296 | 14,908 | 388 | 85 |
| `val_tune` | `2023/2024` | `2023-07-04` | `2024-06-28` | 1,573 | 1,448 | 125 | 59 |
| `val_calib` | `2024/2025` | `2024-07-01` | `2025-06-30` | 1,483 | 1,387 | 96 | 52 |
| `test` | `2025/2026` | `2025-07-04` | `2026-06-03` | 1,586 | 1,530 | 56 | 35 |

## 6. Snapshot-Date Comparison & Single-Snapshot `salimt` Per-Row Contract Audit
- `dcaribou` max valuation date in cohort: **`2026-06-03`**
- `salimt/player_profiles.csv` GitHub commit SHA & snapshot date: **`62ca05786a` (`2025-10-18`)**
- `salimt/player_profiles.csv` max valid `date_of_last_contract_extension`: **`2025-09-12`** (excluding `5` future-dated typo rows `> 2026-07-01`)
- `salimt/player_profiles.csv` max valid `joined` date: **`2025-09-13`**
- Because `salimt/player_profiles.csv` was committed on `2025-10-18` whereas `dcaribou` valuations run through `2026-06-03`, mixing `dcaribou`'s contract expiry with `salimt`'s extension date would leak extensions signed after `2025-10-18`. The consistent per-row filter uses `salimt`'s own `contract_expires > t`, `tm_club_id_at_valuation = salimt.current_club_id`, `max(joined, date_of_last_contract_extension) <= t`, and `t <= 2025-10-18`.

| Cycle | Valuations (`GB1`+`EC1`) | `salimt` Contract Non-Null % | Extended After Val Date (Count) | Extended After Val Date (% of All Rows) | Per-Row Valid Contract (Count) | Per-Row Valid Contract (% of All Rows) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2018/2019` | 1,466 | 60.2% | 275 | 18.8% | **2** | **0.1%** |
| `2019/2020` | 1,548 | 62.0% | 309 | 20.0% | **3** | **0.2%** |
| `2020/2021` | 2,273 | 68.0% | 474 | 20.9% | **13** | **0.6%** |
| `2021/2022` | 1,634 | 71.4% | 339 | 20.7% | **52** | **3.2%** |
| `2022/2023` | 2,151 | 77.2% | 468 | 21.8% | **182** | **8.5%** |
| `2023/2024` | 1,573 | 85.5% | 224 | 14.2% | **412** | **26.2%** |
| `2024/2025` | 1,483 | 89.3% | 145 | 9.8% | **726** | **49.0%** |
| `2025/2026` | 1,586 | 91.6% | 0 | 0.0% | **234** | **14.8%** |

- **Valid Contract Valuations Pre-Test (`2018/19`–`2024/25`)**: **1,390**
- **Valid Contract Valuations in `2025/2026` (`<= 2025-10-18`)**: **234** (`0` rows after `2025-10-18`)
- **Total Single-Snapshot Valid Contract Valuations (`2018/19`–`2025/26`)**: **1,624**
- **Policy**: Keep the main model strictly contract-free across all 19,938 rows. Enforce a single-snapshot contract filter using salimt's own contract_expires, tm_club_id_at_valuation = salimt.current_club_id, max(joined, date_of_last_contract_extension) <= t, and t <= 2025-10-18 (commit 62ca05786a). This yields 1,390 valid rows prior to 2025/2026 (1,624 total including 234 from the October 2025 batch, and 0 rows after 2025-10-18). Because the 2025/2026 test split has zero contract coverage after October 2025, Phase 3 will evaluate the contract-aware ablation on val_tune (2023/2024) and val_calib (2024/2025), and the UI contract field is expected to remain read-only.
