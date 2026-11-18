import json
import math
import random
import statistics
import subprocess
import sys
import tempfile
import time

from bisect import bisect_right
from pathlib import Path
from check_remember import load, reference, TRAIN, TEST


def expected(xs, ys):
  n, m = len(xs), len(ys)

  pooled = (
    (n - 1) * statistics.variance(xs)
    + (m - 1) * statistics.variance(ys)
  ) / (n + m - 2)

  c = abs(statistics.mean(xs) - statistics.mean(ys))
  c /= math.sqrt(pooled) + 1E-32

  d = abs(sum(
    (x > y) - (x < y)
    for x in xs
    for y in ys
  )) / (n * m)

  a, b = sorted(xs), sorted(ys)

  gap = max(
    abs(bisect_right(a, v) / n - bisect_right(b, v) / m)
    for v in set(a + b)
  )

  k = gap / math.sqrt(1 / n + 1 / m)

  return {
    "cohen": c,
    "cliffs": d,
    "ks": k,
    "same": c <= 0.35 and d <= 0.197 and k <= 1.36
  }


def cases():
  train, test = load(TRAIN), load(TEST)

  known = reference(train, train)
  scores = reference(train, test)

  rng = random.Random(1)
  out = []

  for j in range(76):
    xs = [scores[(j + k) % len(scores)] for k in range(24)]

    if j % 4 == 0:
      ys = list(reversed(xs))

    elif j % 4 == 1:
      ys = [min(1, x + 0.01) for x in xs]

    elif j % 4 == 2:
      ys = [0.9 + 0.1 * x for x in xs]

    else:
      ys = rng.sample(known, 20)

    out.append({"xs": xs, "ys": ys})

  out.extend([
    {"xs": [0.25] * 8, "ys": [0.25] * 5},
    {"xs": [0.25] * 8, "ys": [0.75] * 5},
    {"xs": [0, 0, 1, 1], "ys": [0, 0.5, 0.5, 1]},
    {"xs": [0, 0, 0, 1], "ys": [0, 1, 1, 1, 1]}
  ])

  return out


def check(program):
  samples = cases()
  answers = [expected(c["xs"], c["ys"]) for c in samples]

  with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "cases.json"
    path.write_text(json.dumps(samples))

    start = time.perf_counter()

    run = subprocess.run(
      [
        sys.executable, program, "certify",
        "--train", str(TRAIN),
        "--test", str(path)
      ],
      capture_output=True,
      text=True,
      check=True,
      timeout=60
    )

    elapsed = time.perf_counter() - start

  results = json.loads(run.stdout)["comparisons"]

  assert len(results) == len(answers)

  errors = []

  for j, (actual, want) in enumerate(zip(results, answers)):
    row = {
      "case": j,
      "expected": want["same"],
      "actual": actual["same"],
      "correct": actual["same"] is want["same"]
    }

    for key in ["cohen", "cliffs", "ks"]:
      row[key + "_error"] = abs(actual[key] - want[key])

      assert math.isclose(
        actual[key], want[key],
        rel_tol=1E-9,
        abs_tol=1E-9
      ), (j, key, actual[key], want[key])

    errors.append(row)

  correct = sum(row["correct"] for row in errors)

  assert correct == len(answers)

  summary = {
    "cases": len(answers),
    "correct": correct,
    "accuracy": correct / len(answers),
    "same": sum(row["same"] for row in answers),
    "different": sum(not row["same"] for row in answers),
    "runtime": elapsed
  }

  print("Skill: certify")
  print(f"Comparisons: {len(answers)}")
  print(f"Correct: {correct}/{len(answers)}")
  print(f"Same: {summary['same']}")
  print(f"Different: {summary['different']}")
  print(f"Accuracy: {summary['accuracy']:.2%}")
  print(f"Runtime: {elapsed:.4f}s")
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
