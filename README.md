# Shrinking Code

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
