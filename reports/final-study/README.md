# CSC491 / CSC591 — Shrinking Code final-report draft

This package is a **three-page ACM-style research report draft** and reproducible descriptive analysis. **It does not edit the progress-presentation slides.**

## What is provided

- `main.pdf`: three-page ACM-style article, generated from actual R02–R05 and T7 evidence.
- `main.tex`: editable LaTeX source listing Omar Dylan Abou, Henoc Dossou, and Andrew Sergison as team authors. Verify the required report template and obtain teammate approval before final submission.
- `figures/`: complete-pair LOC chart and estimated-generation-cost chart, PDF and PNG variants.
- `data/study_summary.csv`: 6 Claude conditions across 4 independent V2 repetitions.
- `data/paired_reuse.csv`: repetition-aligned comparisons, including missing/incomplete LOC pairs.
- `data/exploratory_same_diagnostics.csv`: descriptive course-style Cohen / Cliff / KS comparisons for Claude pairs **only**. These are *not final ranks*.
- `data/run_metrics.csv`: 24 treatment-run metric records extracted from the uploaded evaluation archive.
- `data/audit.json`: checksum-like expectation and provenance summary.
- `scripts/reproduce_analysis.py`: reconstruct CSVs from the original two evidence archives (Python stdlib only).
- `scripts/exploratory_course_statistics.py`: exploratory study of statistical-effect thresholds (stdlib only).
- `scripts/plot_results.py`: regenerate two figures if matplotlib is installed.
- `SUBMISSION_CHECKLIST.md`: remaining tasks before calling this the final project.

## Evidence behind the draft

`CSC491_R02_R05_EVALUATED.zip` and `CSC491_T7_verified_evidence.zip` were used directly. Neither archive is modified or included in this smaller package. Independent V2 generation repetitions R02–R05 are pooled; pilot R01 and incomplete R06 are **not** pooled. T7 was observed once and its adapter is AI-assisted, not human-only.

## Install into the existing repository (no branch needed)

From the repository's existing main branch:

```bash
unzip -o ~/Downloads/CSC491_Final_Report_Package.zip -d .
find reports/final-study -maxdepth 2 -type f | sort
```

Review and commit **only after** the report wording, authors, dates, and statistical limitations are correct:

```bash
git add reports/final-study
git commit -m "Add ACM-style project report and reproducible study summaries"
git push origin main
```

## Reproduce outputs

If both original evidence archives are on your Desktop:

```bash
python3 reports/final-study/scripts/reproduce_analysis.py \
  --claude-zip "$HOME/Desktop/CSC491_R02_R05_EVALUATED.zip" \
  --t7-zip "$HOME/Desktop/CSC491_T7_verified_evidence.zip" \
  --out reports/final-study/data
python3 reports/final-study/scripts/exploratory_course_statistics.py
```

To recompile the article using TeX Live / Overleaf, use the `acmart` class and run `pdflatex main.tex` twice from `reports/final-study/`. Figures are included, so recompilation does not require regeneration.

## Interpretation rule

Do not report an official seven-treatment `ranks()` result from four Claude repeats and one AI-assisted T7 observation. The analytic scripts intentionally leave treatment ranking out. The course lecture's Cliff threshold for between-treatment `same()` is 0.195; the V2 `Certify` skill's frozen acceptance threshold is 0.197. They serve different purposes.
