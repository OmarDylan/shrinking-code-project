# Safe r02–r05 offline evaluation (no new model calls)

This evaluates the **four completed V2 generation repetitions**, r02, r03, r04, r05. It does **not** generate or retry any code. R06's provider errors and later planned repetitions are unaffected.

Runs 240 skill outcomes, including syntax failures and blocks, through the unchanged V2 evaluator, restricted offline Docker containers in GitHub Actions. Source code stays mounted read-only, network is disabled, container privileges are removed, and the workflow runs without persisted checkout credentials. Container isolation is a precaution, **not a guarantee** against untrusted code. Review code before execution.

The final report uses the existing `scripts/analyze_v2_final.py --first 2 --last 5`, with descriptive matched-pair summaries and no course-style ranks where n < 5. Only T1–T6 appear; T7 needs actual measurements. R01 is a different pilot and is excluded.

Use from project root after unzipping:

```bash
python3 scripts/evaluate_recorded_v2.py
# Review all generated snapshots in .git/csc491-r02-r05-review.txt
python3 scripts/evaluate_recorded_v2.py --reviewed
# Type REVIEWED after genuinely examining sources.
```

A successful run writes `~/Desktop/CSC491_R02_R05_EVALUATED.zip` and echoes `REPORT.md`. You may upload the ZIP to ChatGPT for a slide refresh and deeper interpretation. The review checkpoint may be done in two stages without losing generated results. The new script commits only the files in this patch on existing main, and checks that the remote remains private before pushing. It reuses an existing workflow invocation when run again for the same SHA.

If the workflow fails because of package availability, Docker sandbox constraints or tests, inspect `gh run view <ID> --log-failed`; do not rewrite generated source, silently retry generation, or change the frozen V2 acceptance checks.
