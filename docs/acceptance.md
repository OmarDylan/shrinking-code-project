# Acceptance Tests

All treatments must satisfy the same requirements
and evaluation protocol.

## Step 1: Remember

Requirements: 1 and 2.

- Load the training inventory.
- Identify the goal columns.
- Calculate normalization statistics from training data.
- Calculate distance to heaven for observed cars.
- Rank cars from best to worst.
- Identify the ten best cars.
- Identify the best, median and worst representative cars.

Goal directions:
- Lbs-: minimize
- Acc+: maximize
- Mpg+: maximize

Evaluation:
- Use the shared 80/20 split.
- Fit normalization statistics using training rows only.
- Evaluate ranking on the 80 observed test cars.
- Compare results against a common reference calculation.
- Record individual distance errors.

This is a retrospective inventory-ranking task.
Observed goals are available for this skill.
They must be hidden for predictive tasks.

Interface:

python3 main.py remember --train TRAIN --test TEST

Output: one JSON object containing:
- scores: distance to heaven for every test row
- top: indices of the ten best test rows
- best: index of the best test row
- median: index of the median-ranked test row
- worst: index of the worst test row

Indices are zero-based.
Break ranking ties using the original row index.
