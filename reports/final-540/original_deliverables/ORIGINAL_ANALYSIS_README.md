# External metrics and statistical prudence check

Prepared for the professor’s post-presentation instructions. Nine complete V2 repetitions (R02–R05, R07–R10, R12), 540 planned outcomes; T7 and E1 are single AI-assisted adapters. All values below use archived outputs only. No new Claude calls or generated program execution.

## Customer-facing measurements

The controlled-arm cells are upper medians among passing current-skill snapshots; each reference is a single implementation. Separate metrics can have different median snapshots. No failed/blocked run gets an invented quality value. The per-implementation CSV retains the denominators.

| Skill / metric | T1 | T2 | T3 | T4 | T5 | T6 | T7 | E1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Remember: precision@10 proxy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Remember: recall@10 proxy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Guess: MPG MAE | 3.9793 | 3.9985 | 3.3232 | 3.3585 | 3.9812 | 3.9369 | 4.2750 | 3.9908 |
| Guess: within ±2 MPG | 23.75% | 22.50% | 50.00% | 48.75% | 25.00% | 25.00% | 50.00% | 43.75% |
| Guess: quality-score MAE | 0.0858 | 0.0851 | 0.0760 | 0.0761 | 0.0851 | 0.0856 | 0.1321 | 0.1134 |
| Certify: reference decision accuracy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Certify: false alarms among reference-same | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Certify: misses among reference-different | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Flow: alert precision | 3.23% | 20.00% | 6.67% | 14.29% | 3.23% | 12.50% | 0.00% | 18.75% |
| Flow: alert recall | 25.00% | 25.00% | 25.00% | 25.00% | 25.00% | 25.00% | 0.00% | 75.00% |
| Flow: false alerts | 30 | 4 | 14 | 6 | 30 | 7 | 17 | 13 |
| Flow: missed alerts | 3 | 3 | 3 | 3 | 3 | 3 | 4 | 1 |
| Blame: reference-case accuracy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Justify: reference-case accuracy | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Choose: regret | 0.0192 | 0.0000 | 0.0000 | 0.0000 | 0.0192 | 0.0192 | 0.0192 | 0.0875 |
| Choose: top-three mean quality | 0.1523 | 0.1173 | 0.1173 | 0.1173 | 0.1536 | 0.1518 | 0.1641 | 0.2491 |
| Choose: simulated inspection savings | 87.50% | 87.50% | 87.50% | 87.50% | 87.50% | 87.50% | 87.50% | 87.50% |
| Fix: eligible cars improved | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Fix: mean estimated gain | 0.4147 | 0.4147 | 0.4147 | 0.4147 | 0.4147 | 0.4147 | 0.4147 | 0.4147 |
| Fix: mean feature edits | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 | 2.0000 |
| Spot: precision | 100.00% | 95.24% | 95.24% | 88.89% | 100.00% | 90.91% | 100.00% | 81.25% |
| Spot: recall | 100.00% | 100.00% | 100.00% | 80.00% | 100.00% | 100.00% | 100.00% | 65.00% |
| Spot: F1 | 100.00% | 97.56% | 95.24% | 85.00% | 97.56% | 95.24% | 100.00% | 72.22% |
| Race: mean best at 5 | 0.1550 | 0.1315 | 0.1168 | 0.1252 | 0.1543 | 0.1407 | 0.1531 | 0.2378 |
| Race: mean best at 10 | 0.1223 | 0.1139 | 0.1130 | 0.1130 | 0.1223 | 0.1139 | 0.1313 | 0.1664 |
| Race: mean labels to target | 1.0000 | 1.0500 | 1.0000 | 1.0000 | 1.0000 | 1.1000 | 1.0000 | 2.6500 |

## Reliability must accompany quality

| Skill | T1 | T2 | T3 | T4 | T5 | T6 | T7 | E1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Remember | 6/9 | 5/9 | 7/9 | 7/9 | 9/9 | 9/9 | 1/1 | 1/1 |
| Guess | 9/9 | 5/9 | 7/9 | 7/9 | 8/9 | 9/9 | 1/1 | 1/1 |
| Certify | 7/9 | 5/9 | 6/9 | 7/9 | 2/9 | 9/9 | 1/1 | 1/1 |
| Flow | 9/9 | 5/9 | 9/9 | 7/9 | 7/9 | 9/9 | 1/1 | 1/1 |
| Blame | 9/9 | 4/9 | 7/9 | 7/9 | 9/9 | 7/9 | 1/1 | 1/1 |
| Justify | 9/9 | 4/9 | 9/9 | 7/9 | 9/9 | 7/9 | 1/1 | 1/1 |
| Choose | 9/9 | 5/9 | 8/9 | 7/9 | 8/9 | 8/9 | 1/1 | 1/1 |
| Fix | 9/9 | 5/9 | 9/9 | 7/9 | 9/9 | 8/9 | 1/1 | 1/1 |
| Spot | 9/9 | 5/9 | 9/9 | 6/9 | 9/9 | 8/9 | 1/1 | 1/1 |
| Race | 8/9 | 5/9 | 8/9 | 6/9 | 9/9 | 8/9 | 1/1 | 1/1 |

## What is actually measured (and what is not)

- Remember: the original checker asserts the exact ordered top 10 and best/median/worst, so both proxy precision and recall are 100% for a passing snapshot. Individual prediction/rank vectors were not saved. The proxy uses protocol-defined quality, not purchase prices or expert lists. Financial savings unavailable.
- Guess: absolute MPG error and hit rate are saved per row. Signed predictions are absent, so Spearman rank agreement cannot be computed from these archives.
- Certify: 56 reference-same and 24 reference-different cases per passing snapshot. All passing cases agree; these are fixed reference labels, not a dedicated A/A or independently injected-effect study. New-seed stability is unmeasured.
- Flow: 61 eligible arrivals, four positive observed-score reference alerts. A never-alert baseline scores 57/61 = 93.44% agreement. Precision, recall and false/missed counts are more useful than agreement alone. Reference alerts are not verified deployed drift.
- Blame / Justify: exact required attribution and decision behavior on 80 reference cases. No causal validation, human explanation usefulness or sales conversions. Numerical errors below tolerance should not be treated as meaningful business differences.
- Choose: ten of 80 candidates inspected and three recommended; 87.5% fewer simulated inspections versus exhaustive screening. Regret measures the gap from the best pool car under protocol quality. No actual labor or purchase savings.
- Fix: modeled improvements for 30 eligible cars, not evidence of feasible mechanical changes. All passing snapshots improve all 30 under the common contract.
- Spot: synthetic corruptions in 20 of 80 rows; not verified scams. Use flags for human review.
- Race: 20 fixed episodes, overlapping pools and shared dataset; adaptive vs random under ten labels. Baseline mean best10 .155565, mean labels to target 3.2; no actual dollar ROI.

## Course-style statistics across T1–T7

Same requires Cohen ≤ .35, absolute Cliff ≤ .195, and scaled KS ≤ 1.36. We use the same formula as the final analysis, minimum three observations per group. This is distinct from Certify’s frozen .197 Cliff criterion. These are descriptive distribution diagnostics, not equivalence proofs or causal p-values. No multiplicity adjustment or independent-case inference is claimed.

1. **Repetition-level:** 210 pairwise records for ten primary metrics × 21 pairs. 90 same, 55 not-same, 65 insufficient N (including all 60 T7 comparisons). Each automated arm has at most nine independent generation paths. T7 has one, not 80.
2. **Fixed-case profiles:** 336 pairwise records for 16 metrics × 21 pairs, including T7. Each case has one mean outcome across passing automated snapshots; T7 contributes its single outcome. The 7,216 profile records include E1 for descriptive reference. Case IDs are aligned and reference labels verified where available. Averaging repeated snapshots can smooth distributions relative to a single implementation; interpret only descriptively.
3. **Direct snapshot–T7 diagnostics:** 714 comparisons (417 same, 297 not-same), each using one passing generated snapshot and the one T7 on the same held-out cases. This avoids pooling the same car repeatedly into an artificially large sample. For all cases, signed paired mean differences supplement the marginal-distribution diagnostics.

### Examples: number of passing snapshots whose external distribution is “same” as T7

| Metric | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---:|---:|---:|---:|---:|---:|
| Guess absolute MPG error | 0/9 | 0/5 | 2/7 | 4/7 | 0/8 | 4/9 |
| Guess ±2 MPG hit | 0/9 | 0/5 | 7/7 | 7/7 | 1/8 | 4/9 |
| Flow decision error | 1/9 | 0/5 | 6/9 | 0/7 | 3/7 | 4/9 |
| Spot decision error | 9/9 | 5/5 | 9/9 | 2/6 | 9/9 | 8/8 |
| Race best at 10 | 6/8 | 1/5 | 0/8 | 0/6 | 3/9 | 3/8 |

Remember and Choose lack a meaningful fixed-case numerical comparison in the saved records: Remember has checker status, and Choose has one pool-level regret per implementation. Their repetition-level comparisons are included; T7 remains insufficient N. The professor’s requirement to compare T1–T7 is addressed as far as the available evidence supports, but independent-generation comparisons with T7 are not established.

## Reproduce and submit

Extract this package and the existing final 540-outcome analysis. Keep the seven archived evaluation ZIPs in a single directory. Run:

```bash
python3 external_metrics.py /path/to/CSC491_Final_540_Outcome_Analysis/results /path/to/archives /path/to/new_results
```

The script reads seven ZIPs, verifies 540 outcomes and 440 passes, checks aligned case identifiers, and records source SHA256 hashes. The results have 2,095 metric observations for 460 passing skill snapshots including the two references. The 360 summary rows include 8 conditions. The revised three-page paper is included under revised_paper/; the previous paper and raw evidence remain unchanged. No generated code is run.

Presentation: exactly 15 slides; editable PowerPoint with embedded speaker notes and separate notes. Upload the PPTX to Google Slides if needed; review imported fonts/formatting. The public URL is on slide one. The deck was rendered locally and visually inspected. No GitHub push or Moodle submission performed.

Submission limitations: T7 AI-assisted rather than unaided; nine complete repeats below rubric’s ten-repeat full-credit threshold; original interrupted R06/R11/R13 evidence excluded but retained; no EZR self-tuning experiment; snapshot-derived code changes are not historical Git-tag diffs. Budget checkpoint $30 covered additional batch only, $13.459120 recorded for it; $33.237689 broader known cost excludes two later unknown timeout costs and earlier excluded/auxiliary costs. Do not claim confirmed invoice or original project-wide cap.
