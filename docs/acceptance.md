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

## Step 2: Guess

Requirements: 3 and 4.

- Predict MPG for every test car.
- Predict distance to heaven for every test car.
- Use training data only to learn the model.
- Hide all test goal values from the learner.
- Hide Volume for alternating test rows to assess missing inputs.
- Preserve all functionality from step 1.

Evaluation:
- Absolute MPG error for each test car.
- Absolute distance-to-heaven prediction error for each test car.
- Compare with a training-mean prediction baseline.
- Preserve individual errors and execution time.

Interface:

python3 main.py guess --train TRAIN --test MASKED_TEST

Output: one JSON object containing:
- mpg: predicted MPG for every test row
- scores: predicted distance to heaven for every test row

Predictions must be returned in the original test row order.

## Step 3: Certify

Requirements: 5 and 6.

- Compare two distributions of car evaluation scores.
- Calculate Cohen's effect size.
- Calculate Cliff's delta.
- Calculate the normalized KS statistic.
- Decide whether the distributions count as the same.
- Handle ties, unsorted samples and unequal sample sizes.
- Preserve all previous skills.

Thresholds:
- Cohen: 0.35
- Cliffs: 0.197
- KS: 1.36

Evaluation:
- Evaluate 80 comparison cases.
- Include observed car scores and controlled variations.
- Compare against independent reference calculations.
- Record comparison accuracy and numerical errors.

Interface:

python3 main.py certify --train TRAIN --test CASES_JSON

Input: a JSON list of objects containing xs and ys.

Output: one JSON object containing comparisons.

Each comparison contains:
- cohen
- cliffs
- ks
- same

The distributions are already supplied, so no training
is necessary for this skill.
