import csv
import json
import math
import statistics
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data/splits/seed1/train.csv"
TEST = ROOT / "data/splits/seed1/test.csv"

GOALS = {
  "Lbs-": 0,
  "Acc+": 1,
  "Mpg+": 1
}


def load(path):
  with open(path, newline="") as f:
    return list(csv.DictReader(f))


def reference(train, test):
  stats = {}

  for goal in GOALS:
    values = [float(row[goal]) for row in train]
    stats[goal] = (
      statistics.mean(values),
      statistics.stdev(values)
    )

  def norm(value, goal):
    mu, sd = stats[goal]
    z = (value - mu) / (sd + 1e-32)
    z = max(-3, min(3, z))
    return 1 / (1 + math.exp(-1.702 * z))

  scores = []

  for row in test:
    score = math.sqrt(
      sum(
        (norm(float(row[g]), g) - h) ** 2
        for g, h in GOALS.items()
      ) / len(GOALS)
    )
    scores.append(score)

  return scores


def check(program):
  train = load(TRAIN)
  test = load(TEST)

  expected = reference(train, test)

  run = subprocess.run(
    [
      sys.executable,
      program,
      "remember",
      "--train", str(TRAIN),
      "--test", str(TEST)
    ],
    capture_output=True,
    text=True,
    check=True,
    timeout=60
  )

  result = json.loads(run.stdout)
  scores = result["scores"]

  assert len(scores) == len(expected)

  errors = [
    abs(a - b)
    for a, b in zip(scores, expected)
  ]

  order = sorted(
    range(len(expected)),
    key=lambda j: (expected[j], j)
  )

  assert max(errors) < 1e-7
  assert result["top"] == order[:10]
  assert result["best"] == order[0]
  assert result["median"] == order[len(order)//2]
  assert result["worst"] == order[-1]

  print("Skill: remember")
  print(f"Training rows: {len(train)}")
  print(f"Testing rows: {len(test)}")
  print(f"Mean error: {statistics.mean(errors):.8f}")
  print(f"Maximum error: {max(errors):.8f}")
  print("PASS")


if __name__ == "__main__":
  check(sys.argv[1])
