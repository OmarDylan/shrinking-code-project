# T7 AI-assisted coursework reuse: Guess + Flow

Builds on the already validated `src/t7_assisted/main.py` Remember and Certify work (commit `e00b1d5`). The earlier EG5 and EG6 modules stay byte-identical and are reused by reference. **This newly prepared adapter code is AI-assisted** and is not an unaided human coding-time measurement.

- **Guess:** uses EG5's training-only `Tbl.distx()` to find 5 nearest labeled training cars by four input fields, and EG5's `Tbl.disty()` training goal scores to estimate held-out d2h. MPG is the mean of the five training labels. Missing Volume is handled by EG5's distance rules. Never reads test goals.
- **Flow:** uses that same fixed five-neighbor model and EG6's `cohen()`, `cliffs()`, and `ks()` to compare all training d2h scores with a rolling window of 20 arriving predicted scores. First 19 alerts are false. All predictions and alerts are prefix-stable.
- **Certify:** as before, uses V2 Cliff <= 0.197 instead of EG6's earlier default <= 0.195.
- **Accuracy:** the frozen Guess checker verifies valid finite outputs and reports MAE; a PASS is **not** proof of better predictive accuracy than baseline. The Flow checker verifies exact alert decisions against its reference statistics, along with prefix consistency.

## Installation

Unzip this package into the existing repository (`unzip -o ... -d .`). It replaces only `src/t7_assisted/main.py`, adds a checker, and this documentation. The previous version is in git commit `e00b1d5` and can be restored if needed. It does not overwrite previous homework, T1–T6 data, frozen checkers, or `src/t7_human/main.py`.

Review code before executing:

```bash
python3 -m py_compile src/t7_assisted/main.py scripts/check_t7_first_four.py
python3 scripts/check_t7_first_four.py
```

If all four pass on Auto93, commit the added adapter and its checker. Keep actual human time and AI-assisted time separate; do not estimate or fabricate them. For separate human-only work, use `src/t7_human/main.py` and produce it independently.

## Scope of local validation by assistant

The frozen checkers were run on synthetic car rows (318 train, 80 test) because the original Auto93 CSV is not included in the uploaded project snapshot. The user's real Auto93 split must still be evaluated locally. No claim is made that the later six T7 skills are implemented or tested.
