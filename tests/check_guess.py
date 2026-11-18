import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time

from pathlib import Path
from check_remember import load, reference, ROOT, TRAIN, TEST, GOALS


def check(program):
  train, test = load(TRAIN), load(TEST)

  actual_mpg = [float(row["Mpg+"]) for row in test]
  actual_scores = reference(train, test)

  baseline_mpg = statistics.mean(
    float(row["Mpg+"]) for row in train
  )
  baseline_score = statistics.mean(reference(train, train))

  masked = []

  for j, row in enumerate(test):
    row = dict(row)

    for goal in GOALS:
      row[goal] = "?"

    if j % 2 == 0:
      row["Volume"] = "?"

    masked.append(row)

  with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "test.csv"

    with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(test[0]))
      writer.writeheader()
      writer.writerows(masked)

    start = time.perf_counter()

    run = subprocess.run(
      [
        sys.executable, program, "guess",
        "--train", str(TRAIN),
        "--test", str(path)
      ],
      capture_output=True,
      text=True,
      check=True,
      timeout=60
    )

    elapsed = time.perf_counter() - start

  result = json.loads(run.stdout)

  mpg = result["mpg"]
  scores = result["scores"]

  assert len(mpg) == len(test)
  assert len(scores) == len(test)
  assert all(math.isfinite(x) and x > 0 for x in mpg)
  assert all(math.isfinite(x) and 0 <= x <= 1 for x in scores)

  errors = []

  for j in range(len(test)):
    errors.append({
      "row": j,
      "partial_inputs": j % 2 == 0,
      "mpg_error": abs(mpg[j] - actual_mpg[j]),
      "disty_error": abs(scores[j] - actual_scores[j]),
      "baseline_mpg_error": abs(baseline_mpg - actual_mpg[j]),
      "baseline_disty_error": abs(
        baseline_score - actual_scores[j]
      )
    })

  summary = {
    key: statistics.mean(row[key] for row in errors)
    for key in [
      "mpg_error",
      "disty_error",
      "baseline_mpg_error",
      "baseline_disty_error"
    ]
  }

  print("Skill: guess")
  print(f"Training rows: {len(train)}")
  print(f"Testing rows: {len(test)}")
  print(f"Partial-input rows: {len(test)//2}")

  for key, value in summary.items():
    print(f"{key}: {value:.4f}")

  print(f"Runtime: {elapsed:.4f}s")
  print("PASS (valid predictions)")

  if len(sys.argv) > 2:
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
      "seed": 1,
      "runtime": elapsed,
      "summary": summary,
      "errors": errors
    }, indent=2) + "\n")

    print(f"Results saved to: {out}")


if __name__ == "__main__":
  check(sys.argv[1])
