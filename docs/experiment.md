# Shrinking-Code Experiment

## Domain

Client: Used-car dealership
Dataset: Auto93
Rows: 398
Training: 318 rows (80%)
Testing: 80 rows (20%)
Requirements: 20
Skills: 10

## Treatments

| ID | Author | Background | Reuse |
|---|---|---|---|
| T1 | Claude | Plain Python | No |
| T2 | Claude | Plain Python | Yes |
| T3 | Claude | Scikit-learn | No |
| T4 | Claude | Scikit-learn | Yes |
| T5 | Claude | ezr | No |
| T6 | Claude | ezr | Yes |
| T7 | Human | Own previous skills | Maximum |

## Protocol

- Use identical training and testing splits across treatments.
- Hide test goals when evaluating prediction.
- Use the same Claude model for T1 through T6.
- Use fresh one-shot Claude calls.
- Disable memory and shared conversational context.
- Preserve prompts, responses, and implementation snapshots.
- Start with one complete pilot.
- Target 10 independent repeats if feasible.
- Record failures instead of hiding them.
- Stop before exceeding the agreed budget.
- Tag every implementation step in Git.

## Measurements

- New lines of code per step
- Total lines of code
- Competency scores
- Tests passed and failed
- Generation time
- Execution time
- LLM cost where available
- Detected implementation errors

## Analysis

- Plot new code per step.
- Compare competency distributions.
- Use same() and ranks() for statistical comparisons.
- Report tied winners and negative results honestly.

## Deliverables

- Public Git repository
- Presentation: 10 minutes maximum, 10 slides maximum
- Three-page ACM transactions report
- Final deadline: November 30, 2026
