# Phase 1 Pre-Ingestion Data & License Audit Report

## 1. Repository License Verification & Pinned Sources
| Repository | LICENSE File Present | GitHub API SPDX | Note |
| :--- | :--- | :--- | :--- |
| `dcaribou/transfermarkt-datasets` | `True` | `CC0-1.0` | Creative Commons Zero v1.0 Universal |
| `salimt/football-datasets` | `False` | `None` | No LICENSE file in GitHub repository (GitHub API license=null -> all rights reserved by default under copyright law, despite third-party mirrors claiming MIT/CC0). Raw and derived Parquet files must remain gitignored and raw injury records must never be exposed through the API. |

### Pinned Raw Source Files & CSV Reject Audit (`store_rejects=true`)
| Source File | SHA-256 (Prefix) | Size (Bytes) | CSV Rows Read | CSV Reject Count |
| :--- | :--- | ---: | ---: | ---: |
| `dcaribou/competitions.csv.gz` | `8924ddfbc0e9...` | 2,242 | 65 | 0 |
| `dcaribou/clubs.csv.gz` | `aa57ab56e87f...` | 51,281 | 796 | 0 |
| `dcaribou/players.csv.gz` | `d22e407981d5...` | 4,389,958 | 50,149 | 0 |
| `dcaribou/player_valuations.csv.gz` | `608e32e5ad22...` | 7,197,935 | 656,301 | 0 |
| `dcaribou/appearances.csv.gz` | `e096c2b158fc...` | 45,253,638 | 1,894,350 | 0 |
| `dcaribou/games.csv.gz` | `142561989017...` | 4,995,595 | 88,958 | 0 |
| `dcaribou/transfers.csv.gz` | `90326983daf7...` | 5,809,514 | N/A | 0 |
| `salimt/player_profiles.csv` | `7ef9afbacf97...` | 26,380,409 | 92,671 | 0 |
| `salimt/player_market_value.csv` | `e9e1868c883f...` | 23,873,747 | 901,429 | 0 |
| `salimt/player_injuries_updated_20260929.csv` | `c622ed5684ec...` | 9,455,168 | 168,290 | 0 |
| `salimt/team_competitions_seasons.csv` | `83f06c8108e8...` | 11,032,370 | 58,247 | 0 |
| `salimt/player_performances.csv` | `21eda6556654...` | 157,647,289 | 1,878,716 | 3 |

## 2. Canonical Authoritative Source per Table
| Entity / Table | Canonical Source |
| :--- | :--- |
| `players` | dcaribou/players.csv.gz (position 'Missing' backfilled from salimt/player_profiles.csv) |
| `valuations` | dcaribou/player_valuations.csv.gz (sole valuation source; includes current_club_id at valuation date) |
| `appearances` | dcaribou/appearances.csv.gz + dcaribou/games.csv.gz (match-level; player_seasons is display-only) |
| `injuries` | salimt/player_injuries_updated_20260929.csv (lakehouse-only; clipped at valuation date t) |
| `club_season_league_lookup` | dcaribou/games.csv.gz + salimt/team_competitions_seasons.csv + EC1 club registry |

## 3. Entity Resolution Audit (`dcaribou` vs. `salimt`)
### 3.1 Global `dcaribou` Coverage vs. ID-Namespace Verification
- `dcaribou` total players: **50,149**
- Matched in `salimt/player_profiles.csv` by `tm_player_id` (coverage): **30,361 (60.54%)**
- Exact `date_of_birth` agreement among ID-matched rows (ID-namespace verification): **30,230 / 30,361 (99.57%)**

### 3.2 Target Cohort Coverage (`GB1` + `EC1` Clean Players & Valuations)
- Target cohort distinct players: **2,552**
- Target cohort players matched in `salimt/player_profiles.csv` by `tm_player_id`: **2,358 / 2,552 (92.40%)**
- Target cohort valuations matched in `salimt/player_profiles.csv` by `tm_player_id`: **18,951 / 19,961 (94.94%)**
- Exact `date_of_birth` agreement among ID-matched target-cohort players: **2,350 / 2,358 (99.66%)**
- Target cohort players with injury records in `salimt/player_injuries_updated_20260929.csv`: **1,990 / 2,552 (77.98%)**
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

### 4.2 Missing Club / League (`current_league_code NOT NULL`)
- Total `dcaribou` players: **50,149**
- `current_club_domestic_competition_id IS NULL`: **2,986** (retired, free agents, or non-headline clubs)
- **Policy**: Populate players.current_league_code with COALESCE(current_club_domestic_competition_id, last_valuation_league_code, 'UNKNOWN') so retired/free-agent players with valid historical valuations are retained without violating NOT NULL.

### 4.3 Valuation Cleaning Drop Log
- Raw valuation rows (`dcaribou/player_valuations.csv.gz`): **656,301**
- Dropped (`market_value_in_eur <= 0` or `NULL`): **1**
- Dropped (duplicate `(player_id, valuation_date)`): **0**
- Dropped (`date_of_birth IS NULL` -> `age_at_valuation` unknown): **600**
- Dropped (`position = 'Missing'` after `salimt` backfill): **404**
- Dropped (`valuation_date < 2013-07-01`, prior to appearance coverage window): **100,805**
- Dropped (`valuation_date >= 2026-07-01`, incomplete current cycle): **0**
- **Remaining Clean Valuations (All Leagues, 2013/14–2025/26)**: **554,491**
- Excluded other leagues (`league_at_valuation NOT IN ('GB1', 'EC1')`): **534,530**
- **Remaining Clean Target Cohort Valuations (`GB1` + `EC1`, 2013/14–2025/26)**: **19,961** (`GB1`: **19,273** across **2,421** players; `EC1`: **688** across **138** players at **27** Ecuadorian clubs)

### 4.4 Promotion & Relegation Leakage Audit (`league_at_valuation`)
- Total valuations (`2013/14`–`2025/26`) with static `player_club_domestic_competition_id = 'GB1'`: **27,614**
- True point-in-time `GB1` valuations (club was one of the 20 Premier League clubs in `dcaribou/games.csv.gz` that season): **19,282**
- Relegated/lower-tier club valuations falsely labeled `GB1` by static club table: **8,332 (30.17% of static `GB1` rows)**
- All-league static vs. point-in-time mismatches (`team_competitions_seasons`): **99,184**
- Valuations at `ec1_clubs` (`2013/14`–`2025/26`) resolved to `EC1`: **688** (resolved to other codes: **0**)

## 5. July–June Cycle Breakdown & Split Cutoffs (`GB1` vs. `EC1` Survivorship OOD Slice)
| Cycle | `GB1` Valuations | `GB1` Players | `GB1` Contract Non-Null % | `GB1` Injury History % | `EC1` Valuations | `EC1` Players | `EC1` Contract Non-Null % | `EC1` Injury History % |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2013/2014` | 997 | 528 | 36.5% | 82.7% | 8 | 4 | 50.0% | 75.0% |
| `2014/2015` | 971 | 540 | 37.7% | 85.1% | 10 | 5 | 90.0% | 60.0% |
| `2015/2016` | 1,265 | 638 | 42.9% | 87.6% | 6 | 6 | 66.7% | 66.7% |
| `2016/2017` | 1,676 | 685 | 49.6% | 88.2% | 20 | 13 | 85.0% | 55.0% |
| `2017/2018` | 1,240 | 621 | 56.4% | 90.3% | 34 | 18 | 73.5% | 41.2% |
| `2018/2019` | 1,394 | 629 | 64.1% | 93.0% | 75 | 33 | 76.0% | 58.7% |
| `2019/2020` | 1,498 | 691 | 66.4% | 92.2% | 51 | 35 | 76.5% | 52.9% |
| `2020/2021` | 2,190 | 769 | 73.2% | 91.5% | 87 | 37 | 75.9% | 42.5% |
| `2021/2022` | 1,572 | 749 | 79.1% | 91.9% | 65 | 39 | 81.5% | 46.2% |
| `2022/2023` | 2,105 | 908 | 82.9% | 85.4% | 49 | 45 | 85.7% | 42.9% |
| `2023/2024` | 1,448 | 655 | 86.2% | 91.5% | 129 | 60 | 81.4% | 32.6% |
| `2024/2025` | 1,387 | 577 | 92.2% | 92.2% | 98 | 52 | 84.7% | 18.4% |
| `2025/2026` | 1,530 | 565 | 93.7% | 86.6% | 56 | 35 | 85.7% | 21.4% |

> **Note on `EC1` Survivorship Sample**: With **688 valuations across 138 players at 27 clubs**, the `EC1` cohort represents **Ecuador-stage valuations of players who reached Transfermarkt-tracked clubs** (a survivorship-selected out-of-distribution slice rather than a complete census of Liga Pro Ecuador). Evaluation in Phase 3 will report player-level block bootstrap confidence intervals on the `56` test-split `EC1` valuations.

### Temporal Split Cutoffs & Computed Row Counts
| Split | Cycles | Min Valuation Date | Max Valuation Date | Total Valuations | `GB1` Valuations | `EC1` Valuations | `EC1` Players |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `train` | `2013/2014..2022/2023` | `2013-07-02` | `2023-06-30` | 15,313 | 14,908 | 405 | 87 |
| `val_tune` | `2023/2024` | `2023-07-04` | `2024-06-28` | 1,577 | 1,448 | 129 | 60 |
| `val_calib` | `2024/2025` | `2024-07-01` | `2025-06-30` | 1,485 | 1,387 | 98 | 52 |
| `test` | `2025/2026` | `2025-07-04` | `2026-06-03` | 1,586 | 1,530 | 56 | 35 |

## 6. Snapshot-Date Comparison & Per-Row Contemporaneous Contract Audit
- `dcaribou` max valuation date in cohort: **`2026-06-03`**
- `salimt/player_profiles.csv` max valid `date_of_last_contract_extension`: **`2025-09-12`** (excluding `5` future-dated typo rows `> 2026-07-01`)
- `salimt/player_profiles.csv` max valid `joined` date: **`2025-09-13`**
- Because `salimt/player_profiles.csv` was scraped in September 2025 (`2025-09-12`) whereas `dcaribou` valuations run through June 2026 (`2026-06-12`), `salimt` has no extension records after `2025-09-12`.

| Cycle | Valuations (`GB1`+`EC1`) | Static Contract Non-Null % | Extended After Val Date (Count) | Extended After Val Date (% of All Rows) | Per-Row Valid Contract (Count) | Per-Row Valid Contract (% of All Rows) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2018/2019` | 1,469 | 64.7% | 275 | 18.7% | **0** | **0.0%** |
| `2019/2020` | 1,549 | 66.7% | 309 | 19.9% | **0** | **0.0%** |
| `2020/2021` | 2,277 | 73.3% | 474 | 20.8% | **10** | **0.4%** |
| `2021/2022` | 1,637 | 79.2% | 339 | 20.7% | **39** | **2.4%** |
| `2022/2023` | 2,154 | 83.0% | 468 | 21.7% | **166** | **7.7%** |
| `2023/2024` | 1,577 | 85.8% | 224 | 14.2% | **379** | **24.0%** |
| `2024/2025` | 1,485 | 91.7% | 145 | 9.8% | **670** | **45.1%** |
| `2025/2026` | 1,586 | 93.4% | 0 | 0.0% | **1,332** | **84.0%** |

- **Total Per-Row Valid Contract Valuations (`2018/19`–`2025/26`)**: **2,596**
- **Policy**: Keep the main model strictly contract-free across all 19,961 rows. Export per-row validated contract_years_remaining (2,596 rows passing max(joined, date_of_last_contract_extension) <= t and same club) for Phase 3's recent-window contract-aware ablation before deciding whether the UI contract slider is active or read-only.
