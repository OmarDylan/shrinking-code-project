# CSC491 — V2 statistical analysis (T1–T6)

Repetitions: r02, r03, r04, r05. 240 planned skill outcomes; 24 treatment runs.
Python and dependency manifests identical: **True**. Inspect environments.txt before interpreting timing.

## Outcome accounting

| Treatment | PASS / planned | Syntax failures | Blocked | Test FAIL | Complete generation | All skills passed and retained | Direct assigned library imports / generated |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1: Python | 39/40 | 1 | 0 | 0 | 3/4 | 3/4 | N/A |
| T2: Python + reuse | 38/40 | 0 | 0 | 2 | 4/4 | 3/4 | N/A |
| T3: sklearn | 36/40 | 3 | 0 | 1 | 2/4 | 1/4 | 16/37 |
| T4: sklearn + reuse | 40/40 | 0 | 0 | 0 | 4/4 | 4/4 | 36/40 |
| T5: EZR available | 34/40 | 4 | 0 | 2 | 0/4 | 0/4 | 0/36 |
| T6: EZR available + reuse | 34/40 | 1 | 5 | 0 | 3/4 | 3/4 | 0/34 |

## Course rankings

Entries are upper median [Q1, Q3]; n; rank. Rank 0 is best **for that metric only**. A dash means fewer than five observations (no rank). Conditional metrics may favor surviving programs; do not treat them as unconditional superiority.

| Metric | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---:|---:|---:|---:|---:|---:|
| Individual skills passed (of 10) | 10 [10, 10]; n=4; #— | 10 [10, 10]; n=4; #— | 9 [9, 10]; n=4; #— | 10 [10, 10]; n=4; #— | 9 [9, 9]; n=4; #— | 10 [10, 10]; n=4; #— |
| Syntax-valid snapshots (of 10) | 10 [10, 10]; n=4; #— | 10 [10, 10]; n=4; #— | 10 [9, 10]; n=4; #— | 10 [10, 10]; n=4; #— | 9 [9, 9]; n=4; #— | 10 [10, 10]; n=4; #— |
| Skills retained in final implementation (of 10) | 10 [10, 10]; n=4; #— | 10 [10, 10]; n=4; #— | 9 [9, 10]; n=4; #— | 10 [10, 10]; n=4; #— | 9 [9, 9]; n=4; #— | 10 [10, 10]; n=4; #— |
| Final physical LOC, complete generation only | 990 [988, 1003]; n=3; #— | 618 [597, 622]; n=4; #— | 918 [872, 918]; n=2; #— | 522 [509, 579]; n=4; #— | — [—, —]; n=0; #— | 541 [496, 603]; n=3; #— |
| Final physical LOC, all 10 passed and retained only | 990 [988, 1003]; n=3; #— | 618 [597, 622]; n=3; #— | 872 [872, 872]; n=1; #— | 522 [509, 579]; n=4; #— | — [—, —]; n=0; #— | 541 [496, 603]; n=3; #— |
| Gross added physical LOC, complete generation only | 990 [988, 1003]; n=3; #— | 650 [649, 679]; n=4; #— | 918 [872, 918]; n=2; #— | 573 [572, 613]; n=4; #— | — [—, —]; n=0; #— | 565 [528, 639]; n=3; #— |
| Estimated API-equivalent USD, attempted calls | 0.3204 [0.2381, 0.3504]; n=4; #— | 0.8254 [0.8014, 0.8509]; n=4; #— | 0.3162 [0.2402, 0.3407]; n=4; #— | 0.757 [0.7539, 0.7845]; n=4; #— | 0.9393 [0.3343, 1.054]; n=4; #— | 1.417 [1.373, 1.465]; n=4; #— |
| Total input tokens including cache, attempted calls | 3.657e+04 [3.657e+04, 3.658e+04]; n=4; #— | 7.489e+04 [7.365e+04, 7.555e+04]; n=4; #— | 3.679e+04 [3.679e+04, 3.679e+04]; n=4; #— | 7.14e+04 [7.104e+04, 7.287e+04]; n=4; #— | 2.027e+05 [2.027e+05, 2.027e+05]; n=4; #— | 2.375e+05 [2.356e+05, 2.425e+05]; n=4; #— |
| Total output tokens, attempted calls | 2.039e+04 [1.983e+04, 2.093e+04]; n=4; #— | 5.258e+04 [5.068e+04, 5.539e+04]; n=4; #— | 1.936e+04 [1.911e+04, 2.003e+04]; n=4; #— | 4.766e+04 [4.75e+04, 4.983e+04]; n=4; #— | 2.152e+04 [2.132e+04, 2.428e+04]; n=4; #— | 4.929e+04 [4.745e+04, 5.647e+04]; n=4; #— |
| Sum of generation seconds, attempted calls | 140.6 [135.1, 144.6]; n=4; #— | 276 [266, 297.5]; n=4; #— | 132.1 [131.8, 142.1]; n=4; #— | 250.1 [249.6, 259.4]; n=4; #— | 145.2 [143.5, 157.4]; n=4; #— | 261.1 [254.6, 298.7]; n=4; #— |
| Estimated cost, all 10 generated only | 0.3204 [0.2217, 0.3504]; n=3; #— | 0.8254 [0.8014, 0.8509]; n=4; #— | 0.3407 [0.2402, 0.3407]; n=2; #— | 0.757 [0.7539, 0.7845]; n=4; #— | — [—, —]; n=0; #— | 1.417 [1.373, 1.465]; n=3; #— |
| Generation seconds, all 10 generated only | 140.6 [131.8, 144.6]; n=3; #— | 276 [266, 297.5]; n=4; #— | 142.1 [132.1, 142.1]; n=2; #— | 250.1 [249.6, 259.4]; n=4; #— | — [—, —]; n=0; #— | 261.1 [254.6, 298.7]; n=3; #— |
| Sum of primary evaluator wall seconds, all ten passed only | 1.742 [1.738, 1.863]; n=3; #— | 1.686 [1.641, 1.712]; n=3; #— | 19.63 [19.63, 19.63]; n=1; #— | 15.32 [13.8, 18.78]; n=4; #— | — [—, —]; n=0; #— | 1.98 [1.894, 2.271]; n=3; #— |
| Sum of reported execution seconds (skills 2–10), all ten passed only | 0.7174 [0.6985, 0.8278]; n=3; #— | 0.6471 [0.6061, 0.6829]; n=3; #— | 16.35 [16.35, 16.35]; n=1; #— | 12.72 [10.3, 15.17]; n=4; #— | — [—, —]; n=0; #— | 0.9217 [0.8485, 1.212]; n=3; #— |
| Guess MPG MAE, passing snapshots only | 3.998 [3.979, 3.999]; n=4; #— | 4.024 [3.998, 4.024]; n=4; #— | 3.359 [3.259, 3.36]; n=3; #— | 3.359 [3.317, 3.498]; n=4; #— | 3.983 [3.796, 3.991]; n=3; #— | 3.746 [3.712, 3.979]; n=4; #— |
| Guess distance-to-heaven MAE, passing snapshots only | 0.08607 [0.08579, 0.08608]; n=4; #— | 0.08579 [0.08514, 0.08606]; n=4; #— | 0.0774 [0.07605, 0.07756]; n=3; #— | 0.07503 [0.07432, 0.07609]; n=4; #— | 0.08624 [0.08363, 0.09055]; n=3; #— | 0.08392 [0.08385, 0.08579]; n=4; #— |
| Spot F1, passing snapshots only | 1 [0.9524, 1]; n=4; #— | 0.9756 [0.9091, 1]; n=4; #— | 0.9524 [0.9474, 0.9756]; n=4; #— | 0.85 [0.8421, 1]; n=4; #— | 0.9756 [0.9756, 1]; n=4; #— | 0.9091 [0.9091, 0.9756]; n=3; #— |
| Race adaptive best at 10 labels, passing snapshots only | 0.1223 [0.1223, 0.126]; n=4; #— | 0.1159 [0.1139, 0.1323]; n=4; #— | 0.113 [0.113, 0.113]; n=4; #— | 0.113 [0.113, 0.113]; n=4; #— | 0.1534 [0.1186, 0.2576]; n=4; #— | 0.1139 [0.113, 0.1365]; n=3; #— |
| Race random best at 10 labels, passing snapshots only | 0.1556 [0.1556, 0.1556]; n=4; #— | 0.1556 [0.1556, 0.1556]; n=4; #— | 0.1556 [0.1556, 0.1556]; n=4; #— | 0.1556 [0.1556, 0.1556]; n=4; #— | 0.1556 [0.1556, 0.1556]; n=4; #— | 0.1556 [0.1556, 0.1556]; n=3; #— |
| Race random minus adaptive best at 10, passing snapshots only | 0.03329 [0.03329, 0.03329]; n=4; #— | 0.04162 [0.03969, 0.04258]; n=4; #— | 0.04258 [0.04258, 0.04258]; n=4; #— | 0.04258 [0.04258, 0.04258]; n=4; #— | 0.03697 [0.002139, 0.04162]; n=4; #— | 0.04162 [0.01902, 0.04258]; n=3; #— |

## Matched reuse comparisons (same repetition)

| Pair | Metric | Matched n | Median reuse − no reuse | Median % change | Reuse lower / equal / higher |
|---|---|---:|---:|---:|---:|
| T1/T2 | passes | 4 | 0 | 0 | 1/3/0 |
| T1/T2 | complete_loc | 3 | -385 | -38.38 | 3/0/0 |
| T1/T2 | cost | 4 | 0.5317 | 223.3 | 0/0/4 |
| T1/T2 | complete_cost | 3 | 0.505 | 157.6 | 0/0/3 |
| T1/T2 | generation_seconds | 4 | 144.3 | 109.5 | 0/0/4 |
| T1/T2 | complete_generation_seconds | 3 | 144.3 | 109.5 | 0/0/3 |
| T3/T4 | passes | 4 | 1 | 11.11 | 0/1/3 |
| T3/T4 | complete_loc | 2 | -350 | -40.14 | 2/0/0 |
| T3/T4 | cost | 4 | 0.5138 | 213.9 | 0/0/4 |
| T3/T4 | complete_cost | 2 | 0.5138 | 213.9 | 0/0/2 |
| T3/T4 | generation_seconds | 4 | 117.7 | 89.3 | 0/0/4 |
| T3/T4 | complete_generation_seconds | 2 | 116.4 | 88.11 | 0/0/2 |
| T5/T6 | passes | 4 | 1 | 11.11 | 1/0/3 |
| T5/T6 | complete_loc | 0 | — | — | 0/0/0 |
| T5/T6 | cost | 4 | 1.13 | 338.1 | 1/0/3 |
| T5/T6 | complete_cost | 0 | — | — | 0/0/0 |
| T5/T6 | generation_seconds | 4 | 121 | 86.35 | 1/0/3 |
| T5/T6 | complete_generation_seconds | 0 | — | — | 0/0/0 |

Total attempted-call API-equivalent estimated cost: $15.8090.

## Methods and limitations

- Independent unit: one fresh treatment generation sequence on a fixed Auto93 split. Neither test rows, ten skills, nor overlapping Race episodes are independent repetitions.
- Course glossary 6: https://txt.github.io/seai26f/gloss6.html. same() requires absolute pooled Cohen ≤ 0.35, absolute Cliff’s delta ≤ 0.195, scaled two-sample KS ≤ 1.36. Rank by upper median; compare each candidate to the current tier leader, starting at rank 0. Q1/Q3 are sorted order statistics at floor(n×0.25/0.75). No rank with n<5 (analysis convention). These are course-style descriptive tiers, not multiplicity-adjusted significance claims or proof of equality.
- Final LOC uses only complete ten-skill generation sequences; all-pass LOC additionally requires individual and final retention success. Missing or blocked code is not zero LOC. All-outcome costs and time include failed attempts, but blocked steps have no call. Complete-only resource estimates are reported separately.
- Primary test wall time includes evaluator overhead; it is not pure program execution time. Reported execution time sums skills 2–10 only when all nine values exist (Remember has no reported runtime); some timings come from generated program payloads and are not independently verified. Per-skill execution_seconds is retained in all_steps.csv when available. Different cloud hardware may affect timings.
- r02 was the v2 validation run; sensitivity rankings and pairwise comparisons excluding it appear in CSVs. r01 used a different contract and is excluded. Model/library availability is not proof of library use; detected imports are a proxy.
- Fixed dataset, narrowly specified reference tests and one model version limit generalization. No automatic repairs. Do not claim real-world fraud detection or causal repair.
- T7 is not included: genuine manual prudence checks require separately collected human-authored evidence. EZR tuning is a different experiment; do not pool it with frozen V2. No overall seven-treatment ranking is claimed.

Files: run_metrics.csv, all_steps.csv, rankings.csv, pairwise_same.csv, matched_reuse.csv, skill_summary.csv, new_loc_by_skill.svg, treatment_distributions.svg, environments.txt.
