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

## Step 4: Flow

Requirements: 7 and 8.

- Process arriving cars sequentially.
- Predict MPG and distance to heaven for each car.
- Use only training data and currently available inputs.
- Maintain a rolling window of 20 predicted quality scores.
- Compare the window against historical training quality.
- Raise an alert when the distributions are different.
- Do not raise alerts before the first full window.
- Preserve all previous skills.

Evaluation:
- Individual MPG and distance prediction errors.
- Correctness of statistical alert decisions.
- Agreement with alerts based on observed quality.
- Verify that future arrivals cannot change earlier predictions.
- Record execution time.

Interface:

python3 main.py flow --train TRAIN --test MASKED_TEST

Output: one JSON object containing:
- mpg: predicted MPG for each arriving car
- scores: predicted distance to heaven
- alerts: boolean alert for each arriving car

Window size: 20.

## Step 5: Blame

Requirements: 9 and 10.

- Rank training cars using distance to heaven.
- Build profiles of the best and worst 25% of training cars.
- Identify the weakest goals of each observed test car.
- Rank input characteristics associated with poor cars.
- Exclude ignored columns.
- Preserve all previous skills.

Diagnosis method:
- Goal gaps are absolute distances from normalized heaven.
- Numerical feature scores measure how much closer a
  characteristic is to the poor profile than the good profile.
- Numerical profile centers are normalized group means.
- Symbolic feature scores are positive differences between
  poor-group and good-group relative frequencies.
- Negative feature scores are clipped to zero.
- Missing inputs receive zero evidence.
- Rank scores in descending order, breaking ties by name.

Evaluation:
- Evaluate all 80 observed test cars.
- Learn profiles using training data only.
- Compare diagnostic scores with independent calculations.
- Record diagnostic errors and runtime.

Interface:

python3 main.py blame --train TRAIN --test TEST

Output: one JSON object containing diagnoses.

Each diagnosis contains:
- goals: goal names mapped to their gaps
- features: input names mapped to association scores
- goal_order: goal names ranked by weakness
- feature_order: input names ranked by association

This is retrospective diagnosis, not causal inference.

## Step 6: Justify

Requirements: 11 and 12.

- Explain why one car is recommended over another.
- Explain why a car is accepted or rejected.
- Use observed goals and existing diagnostic evidence.
- Use the median training quality score as acceptance cutoff.
- Recommend the car with the lower distance to heaven.
- Report a tie when quality scores match to 12 decimal places.
- Identify the goal contributing most to the winner's advantage.
- Identify the current car's weakest goal.
- Report an associated feature only when evidence is positive.
- Preserve all previous skills.

Evaluation:
- Evaluate all 80 test cars.
- Compare each car with the next car in the test inventory.
- Compare the last car with the first car.
- Verify recommendation and acceptance decisions.
- Verify that explanations agree with the numerical evidence.
- Record individual results and execution time.

Interface:

python3 main.py justify --train TRAIN --test TEST

Output: one JSON object containing explanations.

Each explanation contains:
- row
- other
- score
- other_score
- cutoff
- accepted
- winner (null for a tie)
- advantage_goal (null for a tie)
- weakest_goal
- top_feature (null when no positive evidence)
- text

Indices are zero-based.

This is retrospective explanation, not causal inference.

## Step 7: Choose

Requirements: 13 and 14.

- Find high-quality cars without inspecting the entire inventory.
- Initially receive only test car input characteristics.
- Request observed goal values through an inspection oracle.
- Inspect exactly 10 distinct test cars.
- Recommend the three best inspected cars.
- Rank recommendations by observed distance to heaven.
- Break ties using the original row index.
- Preserve all previous skills.

Evaluation:
- Use the same 80 test cars for every treatment.
- Learn normalization statistics using training data only.
- Enforce the inspection budget externally.
- Record all inspected and recommended cars.
- Compare against random selection with the same budget.
- Measure regret relative to the best available test car.
- Record execution time.

Interface:

python3 main.py choose --train TRAIN --test MASKED_TEST

The inspection protocol uses JSON lines over stdin and stdout.

The program requests an inspection:
{"inspect": 12}

The evaluator responds with the row index and observed goals.

The program finishes by returning:
{"selected": [12, 5, 19]}

Indices are zero-based.
The selected indices above are illustrative.

Budget: 10 inspections.
Recommendations: 3 cars.

Treatments may use different search strategies.

## Step 8: Fix

Requirements: 15 and 16.

- Identify cars with poor predicted quality.
- Hide all test goal values.
- Reuse training data to estimate quality.
- Change only Clndrs and Volume.
- Keep Model and origin unchanged.
- Consider the first 12 distinct engine configurations
  from the best 25% of training cars.
- Require an estimated improvement of at least 0.05.
- Prefer fewer changed specifications, then smaller
  normalized input distance, then better predicted quality.
- Leave already good cars unchanged.
- Preserve all previous skills.

Evaluation:
- Evaluate all 80 test cars.
- Use a common five-nearest-neighbor scoring model.
- Break neighbor-distance ties at 12 decimal places
  using the original training row order.
- Compare proposed repairs with reference calculations.
- Record estimated improvement and number of changes.
- Record execution time.

Interface:

python3 main.py fix --train TRAIN --test MASKED_TEST

Output: one JSON object containing repairs.

Each repair contains:
- row
- before
- after
- changes
- status

Status values:
- already_good
- improved
- no_candidate

Repairs are hypothetical configuration recommendations.
Estimated improvements are not observed causal effects.

## Step 9: Spot

Requirements: 17 and 18.

- Identify unusual car listings using input characteristics.
- Learn normal behavior from training data only.
- Assign an anomaly score between 0 and 1.
- Higher anomaly scores indicate more unusual cars.
- Determine an anomaly threshold using training data.
- Flag cars whose anomaly scores exceed the threshold.
- Exclude ignored columns and hidden goals.
- Preserve all previous skills.

Evaluation:
- Evaluate all 80 test cars.
- Deliberately corrupt 20 randomly selected listings.
- Introduce abnormal cylinders, engine volume,
  model year and origin values.
- Hide all goal values from the learner.
- Measure precision, recall, F1 and accuracy.
- Record false positives and false negatives.
- Record individual results and execution time.

Interface:

python3 main.py spot --train TRAIN --test MASKED_TEST

Output: one JSON object containing:
- scores: anomaly score for every test car
- alerts: boolean anomaly flag for every test car
- threshold: anomaly detection threshold

Treatments may use different anomaly detection strategies.

Injected anomalies simulate suspicious records.
The dataset does not contain verified fraud labels.
