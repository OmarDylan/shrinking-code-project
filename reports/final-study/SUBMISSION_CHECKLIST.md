# CSC491 final-project submission checklist

**Leave the progress presentation unchanged.** This file concerns the research project / report only.

- [x] T1–T6 recorded in four frozen V2 code-generation repetitions (R02–R05).
- [x] Cloud competency checks recorded for 240 planned outcomes: 221 PASS; 9 syntax failures; 5 competency failures; 5 blocked.
- [x] Reuse versus no-reuse complete-LOC comparisons made on matched repetitions; all five available Python/sklearn pairs show lower LOC with reuse.
- [x] API-equivalent generation cost/time and token metrics preserved; failure records included.
- [x] T7 coursework-reuse adapter locally tested: 10/10 PASS in one observed implementation.
- [x] T7 new-vs-reused source measurement preserved: 437 new LOC; 1,500 prior LOC; 1,937 total source footprint.
- [x] ACM-style three-page **draft report** and reproducibility data created.
- [x] Insert all three team authors and NC State email addresses in the report header.
- [ ] Obtain teammate approval of the author list, paper wording, and required report format before final submission.
- [ ] Resolve the **human-only T7 protocol gap**: independently implement and time manual prudence checks, or explicitly get acceptance for the AI-assisted T7 adaptation as a modified condition. Never claim the existing 437 LOC were unaided human work.
- [ ] Record how (or whether) the R06 paid provider error is resolved; R06 remains excluded from frozen R02–R05 analysis until independently completed and evaluated.
- [ ] Confirm with instructor whether four V2 repetitions are sufficient; additional runs are optional based on budget, not implicitly assumed to exist.
- [ ] Confirm the project repository's final public visibility requirement and perform a private-information / credentials review before making it public. The repository was previously described as private.
- [ ] Review the generated-code failure records for any untracked infrastructure condition and verify the evaluation environment is properly documented.
- [ ] Submit the final report and repository via course instructions. `docs/experiment.md` in the archived repository lists **November 30, 2026** as the deadline; verify against the latest course announcement.

## Two major research caveats

1. **Statistical comparability** — only four comparable Claude repetitions; LOC sample counts are as low as two and zero for some arms; T7 has one implementation with different authorship and run environment. No reliable seven-way ranking is yet supported.
2. **Construct validity** — a passing acceptance checker does not measure real-world decision quality or software maintainability. EZR was available in T5/T6 prompts but was not imported directly by successful snapshots. Spot anomalies were synthetically injected; they are not proven fraud.
