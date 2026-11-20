# CSC491: T7 coursework reuse + AI-assisted adapters — all ten skills

**Provenance.** `src/t7_assisted/prior/eg5/eg5.py` and
`src/t7_assisted/prior/eg6/eg6.py` are unchanged copies of prior user-supplied
coursework. `src/t7_assisted/main.py` is an **AI-assisted** adaptation built on
that coursework. This is a seventh *construction condition*, but is **not** an
unaided human-only baseline. Do not call its generated adapter LOC "human-written
LOC" and do not fabricate human development hours or AI generation token costs.

The original Remember, Guess, Certify, and Flow functions remain in place.
The same file now contains Blame, Justify, Choose, Fix, Spot, and Race.

## Task mapping

| Skill | Algorithm / contract |
|---|---|
| Remember | EG5 normalization and distance-to-heaven |
| Guess | Fixed five-neighbor feature-only predictor from EG5 distances |
| Certify | EG6 statistical functions with frozen V2 Cliff threshold 0.197 |
| Flow | Guess and Certify on streaming windows |
| Blame | EG5 column normalization, quartiles, and feature evidence |
| Justify | Remember and Blame for retrospective explanations |
| Choose | Ten interactive inspections with training-only predicted ranking |
| Fix | Frozen deterministic 5-neighbor repair search, 12 candidate pairs |
| Spot | Training-calibrated outlier scores on four input features only |
| Race | 20 adaptive-vs-random episodes, each with 10 labels per strategy, EG6 comparisons |

## One-command evaluation

Run in the project root **after reviewing all source**:

```bash
python3 scripts/check_t7_all_ten.py
```

The script invokes the *unchanged* `tests/check_*.py` files, creates a new
`experiments/t7-assisted/local-TIMESTAMP/` directory containing checker logs,
per-skill payloads, timestamps, source digests, and `summary.json`, then exits
nonzero if any checker fails. It uses the local `data/splits/seed1` dataset.
A `PASS` for Guess or Spot confirms valid interface/output, not model superiority;
compare their quality fields separately. Race's 20 episodes are not 20 independent
LLM generations. Keep this T7 single observation separate from T1–T6 repetition
statistics; seven-way effect-size ranks require comparable per-treatment samples.

The previous four skills have been verified on the user's actual Auto93 split.
The combined ten-skill package was checked on a **synthetic** 318/80 fixture
while being prepared; results on actual Auto93 remain unverified until the user
runs the bundled evaluator locally.

**Limitations:** New adapter code may exceed the earlier planned 20–50 LOC per
skill. The exact *new code vs reused code* LOC can be computed from the hashes and
physical line counts in the result's `provenance.json`; **development time has
not been independently measured**. This may serve as an AI-assisted comparison
and should not be substituted for a genuine hand-authored prudence check.
