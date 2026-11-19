# V2 recovery policy (after r02 T5 Flow syntax failure)

The frozen V2 contract, model, prompts and generation harness are unchanged. The
r02 attempt at T5 Flow was retained as a syntax failure; no source was repaired or
regenerated. A completed Claude response from the expected model that fails the
same Python parser used by generation counts as a recorded generation failure.
Missing, corrupted, unexpected-model and infrastructure records remain unresolved
and stop the runner at the step boundary for inspection.

Because T1/T3/T5 have independent implementations, their later tasks continue.
For T2/T4/T6, a confirmed syntax failure blocks later dependent generation steps:
record each blocked step (no call or cost), never substitute an older snapshot.
The same rule applies to subsequent V2 repetitions. Retain raw responses, prompt
hashes, usage, and logs. The evaluator never executes a failed or blocked source.
It reports generation failures, blocked steps and competency test failures
separately. An artifact with all recorded outcomes can succeed as a workflow even
if implementations fail. Missing/unresolved records fail the evaluation job after
writing the available CSVs. Do not count blocked steps as attempted model calls;
include generation failures in first-attempt success denominators. LOC is only
measured for valid generated source; final cumulative LOC can represent a partial
implementation when not all skills exist. No success is imputed for missing data.
