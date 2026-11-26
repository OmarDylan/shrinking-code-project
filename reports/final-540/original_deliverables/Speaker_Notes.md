# CSC491 updated final presentation — speaker notes

## Slide 1

Introduce the three authors and the central question. The public repository URL is on slide one. This revised presentation uses the nine complete V2 repetitions and the two reference implementations; the final supplemental files still need team-reviewed publication.

## Slide 2

There are twenty client requests, two per skill. We evaluate quality from weight, acceleration and economy. The data do not have vehicle prices, customer conversions or actual fraud labels. Great buys here are protocol-defined high-quality cars, not independently validated bargains.

## Slide 3

These are the original course hypotheses. We distinguish three predictions: smaller complete systems, a decreasing new-code ladder, and fewer errors. External metrics provide the prudence check: even if code becomes smaller, we still need to know if the outputs are useful.

## Slide 4

Same frozen V2 contracts, data split and expected model claude-sonnet-5-5. One fresh generation path per repeat and no repair of syntax failures. Reuse arms extend prior code and check earlier skills. R02–05, R07–10 and R12 completed; R01 is pilot and R06/R11/R13 interrupted. T7 is AI-assisted and does not satisfy an unaided human control. E1 is exploratory.

## Slide 5

Quality is conditional on passing snapshots; we do not impute zero quality for failures. The course-style same diagnostic requires all three limits. The frozen Certify checker separately uses Cliff .197. Compare repetition-level distributions for T1–T6 and fixed-case profiles for T7, with different interpretations. No independent T7 authoring repeats.

## Slide 6

Full means all ten skills and all applicable regression steps pass. T1 has the highest individual pass count; T6 has the highest full delivery count. T2 blocked 36 outcomes. Gross LOC includes partial paths, so a smaller number can be a symptom of missing features. Total cost for these nine runs is $32.86. The total input count includes cached input. Generation minutes sum recorded call durations (not elapsed calendar time). Docker image digests differ so we avoid a controlled execution-speed claim.

## Slide 7

The six complete-generation matches all shrink physical source. One pair failed competency; five were fully delivered. There are only three Python matches, two sklearn matches, one EZR-available match. This is strong descriptive evidence but selected on successful generation. Step-level added shares do not show a uniformly declining ladder. Do not claim all programs became better.

## Slide 8

For Remember, the unchanged checker asserts exact top ten and correct rankings, so the precision@10 and recall@10 proxy is 1 for passing runs. This is NOT expert price-based ground truth. Guess MAE and hit rates are upper medians among passing snapshots; they can come from different runs. T7 matches T3 hit rate but has worse MAE. Certify accuracy is 100% conditional on passing; T5 reliability is only 2/9. Reference same and different cases are not a dedicated A/A or injected-difference experiment; stability across seeds is unmeasured.

## Slide 9

Conditional upper medians, one T7 observation. Flow recall is more important than raw agreement; the median of passing T1–T6 recalls is only 25%. T7 detects none of the four reference alerts. False alerts are staff workload proxies. The reference is observed-score alerts rather than verified real-world drift. Blame and Justify verify reference correctness, not causal explanations or actual buyer comprehension.

## Slide 10

Regret is selected best score minus true best pool score, lower is better. Top-three quality is the mean distance to heaven. Zero median regret does not mean every run finds the best. Fix uses a common frozen model, so the gain is essentially identical across passing implementations. It is a modeled feature counterfactual, not actual repair or savings. Choose gives one aggregate regret per run on one pool, so T7 cannot support a comparable repeated-run statistical test.

## Slide 11

Spot uses 20 deliberately corrupted rows among 80. Precision, recall and F1 shown separately as upper medians of passing snapshots. For Race, all strategies have a budget of ten inspected rows per episode. Lower distance to heaven and fewer inspections to reach the fixed target are desirable. The random baseline is the same frozen random order across arms. These are simulated customer-facing proxies, not actual fraud detection or labor costs.

## Slide 12

The supplement includes 210 repetition-level pairwise diagnostics for ten primary metrics, with T7 explicitly insufficient at that level; 336 fixed-case profile comparisons across all 21 T1–T7 pairs for 16 metrics; and 714 per-snapshot comparisons versus T7. This slide shows the latter. Cases are matched by identifier, not pooled across runs; Cohen, Cliff and KS compare marginal distributions, and paired mean differences are also saved. Cases share one dataset and race pools overlap. The same result is not proof of equivalence. For Remember raw ranking vectors were not retained; Choose has one pool-level regret per implementation. No fabricated statistical test.

## Slide 13

T7 reuses EG5 and EG6 coursework, and has 437 new lines. It is not an unaided human control. E1 calls native EZR table, tree, leaf and acquisition primitives; the V2 adapter must use the specified normalization coefficient and numerical statistics. E1 passed all checks and reproduced on the Mac. It is not necessarily better: its synthetic anomaly F1 and Race quality are worse than several alternatives. Actual EZR usage cannot be inferred from availability.

## Slide 14

The study does not identify a universal winning treatment. T1 passes 84/90 individual skills; T6 delivers 7/9 complete systems but has the highest cost. T7 remains valuable as an understanding and reuse reference. We favor the source-size part of Timm but the rival remains plausible for other predictions. No causal or deployment claims.

## Slide 15

The budget checkpoint was established for the additional batch, not proven for the entire study; all dollar values are API-equivalent estimates and do not represent the invoice. Two later timeout costs are unknown and excluded; the broader cost also excludes pilot R01, interrupted R06 and auxiliary effort. We stopped at nine after repeated interruptions, below the full-credit ten-repeat target. Next work would include real prices, signed prediction capture for rank agreement, separate A/A and controlled injected differences, new seeds and an unaided T7 implementation. Review and publish final evidence with teammates before replacing the earlier submission.
