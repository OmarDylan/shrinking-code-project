# Final 540-outcome study and reproduction record

This supplement supersedes the four-repetition results in `../final-study/` for the final research conclusions, without altering that historical study. Controlled treatments T1–T6 use the nine balanced repetitions R02–R05, R07–R10 and R12. T7 is a separate AI-assisted coursework-reuse adapter, not an unaided human baseline.

| Treatment | Current-skill passes / 90 | Complete systems / 9 |
|---|---:|---:|
| T1 Python | 84 | 4 |
| T2 Python + reuse | 48 | 4 |
| T3 scikit-learn | 79 | 1 |
| T4 scikit-learn + reuse | 68 | 6 |
| T5 EZR available | 79 | 1 |
| T6 EZR available + reuse | 82 | 7 |

Of 540 planned outcomes, 440 passed, 32 failed generation, 60 were blocked and 8 failed competency. Estimated generation cost for the balanced repetitions was $32.8602664, not a complete invoice. Reuse reduced final source size in six matched full-generation comparisons, but did not uniformly improve reliability, cost or effort.

## Final deliverables

- [Updated paper](original_deliverables/revised_paper/main.pdf), [LaTeX source](original_deliverables/revised_paper/main.tex)
- [Updated 15-slide presentation](original_deliverables/CSC491_Updated_Final_Presentation.pptx), [speaker notes](original_deliverables/Speaker_Notes.md)
- [Reconstructed outcomes](all_540_outcomes_RECONSTRUCTED.csv), [reproduction report](REPRODUCTION_REPORT.json)
- [Recomputed customer-facing metrics](reproduced_metrics/external_summary.csv), [metric verification report](METRIC_VERIFICATION.json)
- [Previously reported analysis, including E1](original_deliverables/ORIGINAL_ANALYSIS_README.md)

**Evidence notice:** R02–R05 and T7 checker results are retained originals; R07–R10 and R12 checker results are local replays of the original generated code, not the original archived bytes. All 540 competency outcomes match the previously reported statuses. The 9,939 comparable T1–T7 analysis rows match exactly. The original E1 evaluation archive was not recovered; the original paper, deck and E1 summaries are preserved but E1 is not independently reverified. Read this notice alongside the original deliverables. No new LLM generation was performed during reproduction.

## Reproduce the analysis (Python standard library only)

From the repository root:

```bash
python3 reports/final-540/verify.py
```

This verifies source hashes, the unique outcome keys and counts, and recalculates all six T1–T7 external-metric tables into a temporary directory. It compares every row against the previous analysis, excluding E1. It does not execute generated source, call Claude, or modify the repository. The original E1 archive is required to independently repeat the E1 analysis.

`generation_records.zip` contains 540 original metadata records and 448 generated Python snapshots, with per-file hashes in `generation_sha256.json`. Raw provider responses, prompts and batch logs are not included in this publication copy. `evaluation_records.zip` contains the original and replay checker outputs used by the analysis; redundant console logs and machine-specific T7 provenance are omitted. `replay_environment.txt` records the frozen evaluator commit and Docker image. To rerun generated programs, review the source and use the isolated evaluation procedure from the historical evaluator commit; the analysis verifier does not rerun the programs. The exact original Docker image may not be available to a fresh machine. The original local archive and replay driver should be retained separately.

## Limits

Nine repetitions instead of the ten required for full results credit; T7 AI-assisted; E1 not independently reverified; no EZR self-tuning experiment; snapshot-derived code changes are not historical Git-tag diffs. External metrics are fixed-data proxies, not demonstrated financial savings or actual fraud detection. No independent-generation T7 statistical inference, no Spearman rank agreement from absolute errors, no real-dealership validation. See the original analysis README for statistical thresholds, treatment denominators and budget scope.


Publication privacy note: Local home-directory paths in T7 console logs were redacted. Original evaluation values and source snapshots remain unchanged.
