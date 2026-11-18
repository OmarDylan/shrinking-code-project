import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time

from pathlib import Path
from check_remember import load, reference, TRAIN, TEST, GOALS
from check_certify import expected

WINDOW = 20


def alerts(history, scores):
  return [
    False if j + 1 < WINDOW else
    not expected(history, scores[j+1-WINDOW:j+1])["same"]
    for j in range(len(scores))
  ]


def check(program):
  train, test = load(TRAIN), load(TEST)

  history = reference(train, train)
  actual_scores = reference(train, test)
  actual_mpg = [float(row["Mpg+"]) for row in test]

  masked = []

  for j, row in enumerate(test):
    row = dict(row)

    for goal in GOALS:
      row[goal] = "?"

    if j % 2 == 0:
      row["Volume"] = "?"

    masked.append(row)

  def run(path, rows):
    with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(test[0]))
      writer.writeheader()
      writer.writerows(rows)

    start = time.perf_counter()

    result = subprocess.run(
      [
        sys.executable, program, "flow",
        "--train", str(TRAIN),
        "--test", str(path)
      ],
      capture_output=True,
      text=True,
      check=True,
      timeout=60
    )

    elapsed = time.perf_counter() - start

    return json.loads(result.stdout), elapsed

  with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "stream.csv"

    result, elapsed = run(path, masked)
    prefix, _ = run(path, masked[:40])

  mpg = result["mpg"]
  scores = result["scores"]
  flags = result["alerts"]

  assert len(mpg) == len(test)
  assert len(scores) == len(test)
  assert len(flags) == len(test)

  assert all(math.isfinite(x) and x > 0 for x in mpg)
  assert all(math.isfinite(x) and 0 <= x <= 1 for x in scores)
  assert all(type(x) is bool for x in flags)

  # Later arrivals must not affect earlier results.
  for key in ["mpg", "scores"]:
    assert len(prefix[key]) == 40
    assert all(
      math.isclose(a, b, abs_tol=1E-9)
      for a, b in zip(prefix[key], result[key][:40])
    )

  assert prefix["alerts"] == flags[:40]

  # Check decisions using independent statistical calculations.
  predicted_flags = alerts(history, scores)
  observed_flags = alerts(history, actual_scores)

  assert flags == predicted_flags

  errors = []

  for j in range(len(test)):
    errors.append({
      "row": j,
      "mpg_error": abs(mpg[j] - actual_mpg[j]),
      "disty_error": abs(scores[j] - actual_scores[j]),
      "alert": flags[j],
      "observed_alert": observed_flags[j],
      "alert_correct": flags[j] == observed_flags[j],
      "window_ready": j + 1 >= WINDOW
    })

  eligible = errors[WINDOW-1:]

  summary = {
    "mpg_error": statistics.mean(
      row["mpg_error"] for row in errors
    ),
    "disty_error": statistics.mean(
      row["disty_error"] for row in errors
    ),
    "alerts": sum(flags),
    "observed_alerts": sum(observed_flags),
    "alert_agreement": statistics.mean(
      row["alert_correct"] for row in eligible
    ),
    "runtime": elapsed
  }

  print("Skill: flow")
  print(f"Training rows: {len(train)}")
  print(f"Streaming cars: {len(test)}")
  print(f"Window size: {WINDOW}")
  print(f"MPG error: {summary['mpg_error']:.4f}")
  print(f"Disty error: {summary['disty_error']:.4f}")
  print(f"Alerts: {summary['alerts']}")
  print(f"Observed alerts: {summary['observed_alerts']}")
  print(f"Alert agreement: {summary['alert_agreement']:.2%}")
  print(f"Runtime: {elapsed:.4f}s")
  print("Prefix consistency: PASS")
  print("Statistical alert decisions: PASS")
  print("PASS")

  if len(sys.argv) > 2:
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(json.dumps({
      "seed": 1,
      "summary": summary,
      "errors": errors
    }, indent=2) + "\n")

    print(f"Results saved to: {out}")


if __name__ == "__main__":
  check(sys.argv[1])
