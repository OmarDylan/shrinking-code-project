# Protocol v2: changes after pilot r01

The r01 data and source history are preserved. v2 is not statistically pooled with r01.
The new specification clarifies existing evaluator rules; no evaluator or generated
r01 code was modified. In particular: exact normalization, pooled Cohen variance and
zero-variance convention; diagnostic centers and rounding; upper-middle cutoff;
squared goal gaps; oracle response and ranking; five-neighbor repair model and
candidate tie-breaks; full Race schema and strategy ordering. Some original tests
are stricter than the natural-language customer asks, so report reference agreement
as such, not universal real-world competence. Guess and Spot PASS test valid output;
actual MAE and F1 must be reported separately. Race repeated pools share held-out rows.

All treatments are given the same shared definitions and earlier task specifications;
only reuse conditions receive prior generated source. Assigned background libraries
are available, not required; detected import is an imperfect proxy for actual use.
The pinned EZR code is supplied as in r01; sklearn availability rather than equivalent
source context is used. This makes between-library token cost partially a function
of context size. Preserve the exact prompts, canonical model, source hashes, raw usage,
cache usage, code snapshots, and failures. Estimated API-equivalent cost is not a bill.
Same model actual name must match the pilot, or the attempt is rejected and retained.
No retries or human corrections are mixed into a completed run. For syntax-valid
programs, later reuse steps extend the immediate predecessor even if its tests fail.
Generated code is evaluated on a fresh cloud runner (not a security sandbox);
review source for unintended access or operations before dispatching it. The test
filesystem is accessible to generated code, so data isolation relies in part on
prompt compliance and review; do not claim a hardened oracle. Cloud run timings
are host-specific; the environment and commit are saved. Do not compare host timing
across Mac and Linux as if measured under identical conditions.

A repetition means a new ten-step generation chain for each treatment (60 fresh
calls); a different dataset episode is not a fresh generation chain. Start with r02
as a v2 validation run. Freeze protocol before r03+; no final statistics until there
are sufficient independent comparable runs, preferably the target of 10 per treatment
if feasible. T7 must reflect genuine manual prudence checks; the AI-assisted reference
is not an independent T7. Check the professor's requirement interpretation and budget
before scaling. No automatic new charges or paid headless cloud Claude calls are set up.
