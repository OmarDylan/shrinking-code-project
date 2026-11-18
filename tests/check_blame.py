import json
import math
import statistics
import subprocess
import sys
import time

from pathlib import Path
from check_remember import load, reference, TRAIN, TEST, GOALS


def expected(train, test):
  names = list(train[0])

  features = [
    name for name in names
    if not name.endswith(("X", "+", "-", "!"))
  ]

  stats = {}

  for name in features + list(GOALS):
    if name[0].isupper():
      values = [
        float(row[name]) for row in train
        if row[name] != "?"
      ]

      stats[name] = (
        statistics.mean(values),
        statistics.stdev(values)
      )

  def norm(name, value):
    mu, sd = stats[name]
    z = (float(value) - mu) / (sd + 1E-32)
    z = max(-3, min(3, z))
    return 1 / (1 + math.exp(-1.702 * z))

  scores = reference(train, train)

  ordered = sorted(
    range(len(train)),
    key=lambda j: (round(scores[j], 12), j)
  )

  n = max(1, len(train) // 4)

  good = [train[j] for j in ordered[:n]]
  poor = [train[j] for j in ordered[-n:]]

  def center(rows, name):
    return norm(name, statistics.mean(
      float(row[name]) for row in rows
      if row[name] != "?"
    ))

  def frequency(rows, name, value):
    known = [
      row[name] for row in rows
      if row[name] != "?"
    ]

    return known.count(value) / len(known) if known else 0

  out = []

  for row in test:
    goals = {
      name: abs(norm(name, row[name]) - heaven)
      for name, heaven in GOALS.items()
    }

    evidence = {}

    for name in features:
      value = row[name]

      if value == "?":
        score = 0

      elif name in stats:
        x = norm(name, value)

        score = (
          abs(x - center(good, name))
          - abs(x - center(poor, name))
        )

      else:
        score = (
          frequency(poor, name, value)
          - frequency(good, name, value)
        )

      evidence[name] = max(0, score)

    out.append({
      "goals": goals,
      "features": evidence
    })

  return out


def order(scores):
  return sorted(
    scores, key=lambda k: (-round(scores[k], 10), k)
  )


def check(program):
  train, test = load(TRAIN), load(TEST)
  answers = expected(train, test)

  start = time.perf_counter()

  run = subprocess.run(
    [
      sys.executable, program, "blame",
      "--train", str(TRAIN),
      "--test", str(TEST)
    ],
    capture_output=True,
    text=True,
    check=True,
    timeout=60
  )

  elapsed = time.perf_counter() - start

  results = json.loads(run.stdout)["diagnoses"]

  assert len(results) == len(answers)

  errors = []

  for j, (actual, want) in enumerate(zip(results, answers)):
    row = {"row": j}

    for key in ["goals", "features"]:
      assert set(actual[key]) == set(want[key])

      gaps = []

      for name, value in want[key].items():
        got = actual[key][name]

        assert math.isfinite(got)
        assert 0 <= got <= 1
        assert math.isclose(got, value, abs_tol=1E-9)

        gaps.append(abs(got - value))

      row[key + "_error"] = statistics.mean(gaps)

    assert actual["goal_order"] == order(actual["goals"])
    assert actual["feature_order"] == order(actual["features"])

    row["weakest_goal"] = actual["goal_order"][0]
    row["top_feature"] = actual["feature_order"][0]

    errors.append(row)

  summary = {
    "cases": len(errors),
    "correct": len(errors),
    "goal_error": statistics.mean(
      row["goals_error"] for row in errors
    ),
    "feature_error": statistics.mean(
      row["features_error"] for row in errors
    ),
    "runtime": elapsed
  }

  print("Skill: blame")
  print(f"Training rows: {len(train)}")
  print(f"Diagnosed cars: {len(test)}")
  print(f"Correct: {len(errors)}/{len(test)}")
  print(f"Goal error: {summary['goal_error']:.8f}")
  print(f"Feature error: {summary['feature_error']:.8f}")
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
