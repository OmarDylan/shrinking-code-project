import csv
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time

from functools import lru_cache
from pathlib import Path

from check_remember import load, reference, TRAIN, TEST, GOALS

EDITABLE = ["Clndrs", "Volume"]
MIN_GAIN = 0.05


def expected(train, test):
  features = ["Clndrs", "Volume", "Model", "origin"]

  stats = {
    name: (
      statistics.mean(float(r[name]) for r in train),
      statistics.stdev(float(r[name]) for r in train)
    )
    for name in features
    if name[0].isupper()
  }

  @lru_cache(maxsize=None)
  def norm(name, value):
    mu, sd = stats[name]
    z = (float(value) - mu) / (sd + 1E-32)
    z = max(-3, min(3, z))
    return 1 / (1 + math.exp(-1.702 * z))

  def distance(a, b):
    gaps = []

    for name in features:
      if name in stats:
        gap = norm(name, a[name]) - norm(name, b[name])
      else:
        gap = int(a[name] != b[name])

      gaps.append(gap * gap)

    return math.sqrt(sum(gaps) / len(gaps))

  known = reference(train, train)

  def predict(row):
    near = sorted(
      range(len(train)),
      key=lambda j: (round(distance(row, train[j]), 12), j)
    )[:5]

    return statistics.mean(known[j] for j in near)

  ranked = sorted(
    range(len(train)),
    key=lambda j: (round(known[j], 12), j)
  )

  cutoff = sorted(known)[len(known) // 2]

  good = ranked[:max(1, len(train) // 4)]

  menu = list(dict.fromkeys(
    tuple(float(train[j][name]) for name in EDITABLE)
    for j in good
  ))[:12]

  out = []

  for j, row in enumerate(test):
    before = predict(row)

    answer = {
      "row": j,
      "before": before,
      "after": before,
      "changes": {},
      "status": "already_good"
    }

    if round(before, 10) > round(cutoff, 10):
      answer["status"] = "no_candidate"
      candidates = []

      for values in menu:
        changes = {
          name: value
          for name, value in zip(EDITABLE, values)
          if float(row[name]) != value
        }

        if not changes:
          continue

        new = dict(row)
        new.update(changes)

        after = predict(new)

        if round(before - after, 10) >= MIN_GAIN:
          key = (
            len(changes),
            round(distance(row, new), 10),
            round(after, 10),
            values
          )

          candidates.append((key, after, changes))

      if candidates:
        _, after, changes = min(
          candidates, key=lambda item: item[0]
        )

        answer.update({
          "after": after,
          "changes": changes,
          "status": "improved"
        })

    out.append(answer)

  return out, cutoff


def check(program):
  train, test = load(TRAIN), load(TEST)
  answers, cutoff = expected(train, test)

  masked = []

  for row in test:
    row = dict(row)

    for goal in GOALS:
      row[goal] = "?"

    masked.append(row)

  with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "masked.csv"

    with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(test[0]))
      writer.writeheader()
      writer.writerows(masked)

    start = time.perf_counter()

    run = subprocess.run(
      [
        sys.executable, program, "fix",
        "--train", str(TRAIN),
        "--test", str(path)
      ],
      capture_output=True,
      text=True,
      check=True,
      timeout=60
    )

    elapsed = time.perf_counter() - start

  results = json.loads(run.stdout)["repairs"]

  assert len(results) == len(answers)

  errors = []

  for got, want in zip(results, answers):
    assert got["row"] == want["row"]
    assert got["status"] == want["status"]
    assert got["changes"] == want["changes"]

    for key in ["before", "after"]:
      assert math.isfinite(got[key])
      assert math.isclose(got[key], want[key], abs_tol=1E-9)

    errors.append({
      **got,
      "gain": got["before"] - got["after"],
      "edits": len(got["changes"]),
      "correct": True
    })

  improved = [
    row for row in errors
    if row["status"] == "improved"
  ]

  eligible = [
    row for row in errors
    if row["status"] != "already_good"
  ]

  summary = {
    "cases": len(errors),
    "correct": len(errors),
    "eligible": len(eligible),
    "improved": len(improved),
    "already_good": len(errors) - len(eligible),
    "no_candidate": len(eligible) - len(improved),
    "mean_gain": statistics.mean(
      row["gain"] for row in improved
    ) if improved else 0,
    "mean_edits": statistics.mean(
      row["edits"] for row in improved
    ) if improved else 0,
    "reached_cutoff": sum(
      row["after"] <= cutoff for row in improved
    ),
    "runtime": elapsed
  }

  print("Skill: fix")
  print(f"Evaluated cars: {len(errors)}")
  print(f"Correct: {len(errors)}/{len(test)}")
  print(f"Already good: {summary['already_good']}")
  print(f"Eligible for repair: {summary['eligible']}")
  print(f"Improved: {summary['improved']}")
  print(f"No candidate: {summary['no_candidate']}")
  print(f"Reached quality cutoff: {summary['reached_cutoff']}")
  print(f"Mean estimated gain: {summary['mean_gain']:.4f}")
  print(f"Mean changes: {summary['mean_edits']:.4f}")
  print(f"Runtime: {elapsed:.4f}s")
  print("PASS")

  if improved:
    print("\nExample repair:")
    print(json.dumps(improved[0], indent=2))

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
