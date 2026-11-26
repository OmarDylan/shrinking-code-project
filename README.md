# Shrinking Code

<!-- CSC491_FINAL_540 -->
## Final study — nine repetitions, 540 outcomes

The [final study and reproducibility supplement](reports/final-540/README.md) is the current report: 440 passes, 32 generation failures, 60 blocked outcomes and 8 competency failures. It includes the updated paper and 15-slide presentation, the original generated source snapshots, and independently reproduced T1–T7 external metrics. The native EZR E1 archive was not recovered and its previously reported results are not independently reverified. T7 is AI-assisted, not an unaided human baseline.

Run `python3 reports/final-540/verify.py` to check the archived evidence and recalculate all six external-metric tables. No LLM calls are required.

---

## Historical four-repetition study (R02–R05)

The original README is preserved below for provenance; its older results are superseded by the final study above.

**CSC 491 — Computer Science, NC State University | Fall 2026**

## Team

| Name | NC State Email |
|---|---|
| Omar Dylan Abou | odabou@ncsu.edu |
| Henoc Dossou | hbdossou@ncsu.edu |
| Andrew Sergison | arsergis@ncsu.edu |

## Research Question

Does reusing existing software reduce the amount of new code needed to solve software engineering tasks, and what are the trade-offs in generation cost, time, and correctness?

This project investigates code reuse using ten data-analysis and software-engineering skills on the Auto93 car dataset.

## The Ten Skills

1. Remember — summarize data
2. Guess — predict values
3. Certify — compare distributions
4. Flow — monitor changes
5. Blame — identify influential features
6. Justify — explain predictions
7. Choose — select promising candidates
8. Fix — improve candidates
9. Spot — detect anomalies
10. Race — search under a limited budget

## Experimental Treatments

| Treatment | Approach |
|---|---|
| T1 | Independent implementation using pure Python |
| T2 | Reused implementation using pure Python |
| T3 | Independent implementation using scikit-learn |
| T4 | Reused implementation using scikit-learn |
| T5 | Independent implementation with EZR available |
| T6 | Reused implementation with EZR available |
| T7 | AI-assisted reuse of existing coursework modules |

T7 was AI-assisted. It is not an unaided human-only baseline.

EZR availability in T5/T6 does not imply that every generated implementation imported or used EZR.

## Experimental Results

### T1–T6

The reported analysis uses four repetitions, R02 through R05, with ten tasks across six treatments.

- Total evaluated outcomes: **240**
- Passed: **221**
- Competency failures: **5**
- Syntax failures: **9**
- Blocked outcomes: **5**
- T4 passed all **40 of 40** evaluated tasks.

Observed matched comparisons show that reuse can reduce new lines of code. However, reuse also introduced additional generation effort, including higher token usage, cost, and time in measured comparisons.

These findings describe the evaluated conditions and should not be treated as universal claims about all software projects.

### T7

T7 reused existing coursework modules through an AI-assisted adapter.

The local T7 evaluation passed all ten frozen V2 skill checks. This is one separate local observation, not a directly comparable set of repeated independent generation trials.

## Research Report

The final three-author research paper is available here:

- [Final research paper (PDF)](reports/final-study/main.pdf)
- [LaTeX source](reports/final-study/main.tex)
- [Report documentation and reproduction notes](reports/final-study/README.md)

## Experimental Evidence

The evaluated R02–R05 records are preserved in:

- [Experimental evidence documentation](experiments/evidence/README.md)
- [R02–R05 V2 results](experiments/evidence/r02-r05-v2/)
- [SHA-256 evidence manifest](experiments/evidence/r02-r05-sha256.csv)

The evidence includes individual task outputs and aggregate analysis files.

## T7 Implementation

- [AI-assisted T7 implementation](src/t7_assisted/main.py)
- [T7 evaluation script](scripts/check_t7_all_ten.py)

## Scope and Limitations

- The reported T1–T6 results use R02–R05.
- R06 was interrupted and is excluded from the reported balanced results.
- The T7 evaluation is a separate AI-assisted local observation.
- Passing a competence check does not establish general performance outside the tested conditions.
- Provider-pilot artifacts and incomplete experiments are not included in the reported comparisons.

## Project Status

The implementation, evaluated experiment evidence, and research report have been completed. The repository preserves the results used for the final study.


## Running the ten skill checks

From the repository root, run all ten frozen V2 competency
checks against the AI-assisted T7 implementation:

```bash
python3 scripts/check_t7_all_ten.py
```

To run an individual skill's acceptance check, use one command:

| Skill | Command |
|---|---|
| Remember | `python3 tests/check_remember.py src/t7_assisted/main.py /tmp/csc491-remember-demo.json` |
| Guess | `python3 tests/check_guess.py src/t7_assisted/main.py /tmp/csc491-guess-demo.json` |
| Certify | `python3 tests/check_certify.py src/t7_assisted/main.py /tmp/csc491-certify-demo.json` |
| Flow | `python3 tests/check_flow.py src/t7_assisted/main.py /tmp/csc491-flow-demo.json` |
| Blame | `python3 tests/check_blame.py src/t7_assisted/main.py /tmp/csc491-blame-demo.json` |
| Justify | `python3 tests/check_justify.py src/t7_assisted/main.py /tmp/csc491-justify-demo.json` |
| Choose | `python3 tests/check_choose.py src/t7_assisted/main.py /tmp/csc491-choose-demo.json` |
| Fix | `python3 tests/check_fix.py src/t7_assisted/main.py /tmp/csc491-fix-demo.json` |
| Spot | `python3 tests/check_spot.py src/t7_assisted/main.py /tmp/csc491-spot-demo.json` |
| Race | `python3 tests/check_race.py src/t7_assisted/main.py /tmp/csc491-race-demo.json` |

These are acceptance-check demonstrations using the frozen
V2 test contracts, not an interactive customer application.

T7 contains AI-assisted adapter code; it is not the
unaided-human baseline specified in the original assignment.

### Step history

Tags `step1` through `step10` point retrospectively to
the original R01 pilot implementation commits, covering
Remember through Race. The labels were added during final
submission preparation, not at the time of those commits.

The separately evaluated R02-R05 V2 runs use different
recorded evidence. Pilot Git tag differences must not
be presented as R02-R05 measured generation results.

Additional `t7-assisted-step01` through
`t7-assisted-step10` tags preserve the T7 development history.
