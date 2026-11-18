import csv
import json
import math
import random
import subprocess
import sys
import tempfile
import time

from pathlib import Path
from check_remember import load, TRAIN, TEST, GOALS


def check(program):
  train, test = load(TRAIN), load(TEST)

  rng = random.Random(1)
  corrupted = rng.sample(range(len(test)), 20)

  modifications = [
    ("Clndrs", "40"),
    ("Volume", "3000"),
    ("Model", "-20"),
    ("origin", "99")
  ]

  labels = {
    j: modifications[k % len(modifications)]
    for k, j in enumerate(corrupted)
  }

  masked = []

  for j, row in enumerate(test):
    row = dict(row)

    for goal in GOALS:
      row[goal] = "?"

    if j in labels:
      name, value = labels[j]
      row[name] = value

    masked.append(row)

  with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "listings.csv"

    with path.open("w", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=list(test[0]))
      writer.writeheader()
      writer.writerows(masked)

    start = time.perf_counter()

    run = subprocess.run(
      [
        sys.executable, program, "spot",
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

  scores = result["scores"]
  alerts = result["alerts"]
  threshold = result["threshold"]

  assert len(scores) == len(test)
  assert len(alerts) == len(test)

  assert math.isfinite(threshold)
  assert 0 <= threshold <= 1

  assert all(math.isfinite(s) and 0 <= s <= 1
             for s in scores)

  assert all(type(a) is bool for a in alerts)
  assert alerts == [s > threshold for s in scores]

  tp = fp = tn = fn = 0
  errors = []

  for j, (score, alert) in enumerate(zip(scores, alerts)):
    actual = j in labels

    if actual and alert:
      tp += 1
    elif actual:
      fn += 1
    elif alert:
      fp += 1
    else:
      tn += 1

    errors.append({
      "row": j,
      "score": score,
      "alert": alert,
      "injected_anomaly": actual,
      "corrupted_feature": labels[j][0] if actual else None,
      "correct": alert == actual
    })

  precision = tp / (tp + fp) if tp + fp else 0
  recall = tp / (tp + fn) if tp + fn else 0
  f1 = (2 * precision * recall / (precision + recall)
        if precision + recall else 0)

  summary = {
    "training_rows": len(train),
    "testing_rows": len(test),
    "injected_anomalies": len(labels),
    "threshold": threshold,
    "true_positive": tp,
    "false_positive": fp,
    "true_negative": tn,
    "false_negative": fn,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "accuracy": (tp + tn) / len(test),
    "runtime": elapsed
  }

  print("Skill: spot")
  print(f"Training rows: {len(train)}")
  print(f"Test listings: {len(test)}")
  print(f"Injected anomalies: {len(labels)}")
  print(f"Threshold: {threshold:.4f}")
  print(f"True positives: {tp}")
  print(f"False positives: {fp}")
  print(f"True negatives: {tn}")
  print(f"False negatives: {fn}")
  print(f"Precision: {precision:.2%}")
  print(f"Recall: {recall:.2%}")
  print(f"F1: {f1:.4f}")
  print(f"Accuracy: {summary['accuracy']:.2%}")
  print(f"Runtime: {elapsed:.4f}s")
  print("PASS (valid anomaly detection output)")

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
