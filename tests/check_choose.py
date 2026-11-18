import csv
import json
import queue
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time

from pathlib import Path
from check_remember import load, reference, TRAIN, TEST, GOALS

BUDGET = 10
TOP = 3


def oracle_run(program, path, test):
  inspected = []
  messages = queue.Queue()

  start = time.perf_counter()
  deadline = start + 60

  with tempfile.TemporaryFile(mode="w+t") as err:
    with subprocess.Popen(
      [
        sys.executable, program, "choose",
        "--train", str(TRAIN),
        "--test", str(path)
      ],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=err,
      text=True,
      bufsize=1
    ) as proc:

      def reader():
        try:
          for line in proc.stdout:
            messages.put(line)
        finally:
          messages.put(None)

      thread = threading.Thread(target=reader, daemon=True)
      thread.start()

      def receive():
        try:
          return messages.get(
            timeout=max(0, deadline - time.perf_counter())
          )
        except queue.Empty:
          raise TimeoutError("Choose exceeded its time limit")

      try:
        while True:
          line = receive()

          if line is None:
            proc.wait(timeout=5)
            err.seek(0)
            raise RuntimeError(
              "Program exited before recommendations: "
              + err.read()
            )

          msg = json.loads(line)

          if "selected" in msg:
            assert set(msg) == {"selected"}
            selected = msg["selected"]
            break

          assert set(msg) == {"inspect"}

          j = msg["inspect"]

          assert type(j) is int
          assert 0 <= j < len(test)
          assert j not in inspected
          assert len(inspected) < BUDGET

          inspected.append(j)

          response = {
            "row": j,
            "goals": {
              name: float(test[j][name])
              for name in GOALS
            }
          }

          proc.stdin.write(json.dumps(response) + "\n")
          proc.stdin.flush()

        proc.stdin.close()

        assert receive() is None, "Unexpected extra output"

        rc = proc.wait(
          timeout=max(0.1, deadline - time.perf_counter())
        )

        if rc != 0:
          err.seek(0)
          raise RuntimeError(err.read())

      finally:
        if proc.poll() is None:
          proc.kill()
          proc.wait()

  elapsed = time.perf_counter() - start

  return inspected, selected, elapsed


def check(program):
  train, test = load(TRAIN), load(TEST)

  scores = reference(train, test)

  def rank(indices):
    return sorted(
      indices,
      key=lambda j: (round(scores[j], 12), j)
    )

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

    inspected, selected, elapsed = oracle_run(
      program, path, test
    )

  assert len(inspected) == BUDGET
  assert isinstance(selected, list)
  assert len(selected) == TOP
  assert all(type(j) is int for j in selected)
  assert selected == rank(inspected)[:TOP]

  pool_best = min(scores)
  median = statistics.median(scores)

  best = scores[selected[0]]
  top_mean = statistics.mean(scores[j] for j in selected)

  rng = random.Random(1)
  baseline = []

  for repeat in range(20):
    picked = rng.sample(range(len(test)), BUDGET)
    winners = rank(picked)[:TOP]

    baseline.append({
      "repeat": repeat,
      "best": scores[winners[0]],
      "top_mean": statistics.mean(
        scores[j] for j in winners
      )
    })

  summary = {
    "pool_size": len(test),
    "inspections": len(inspected),
    "best": best,
    "top_mean": top_mean,
    "pool_best": pool_best,
    "regret": best - pool_best,
    "pool_median": median,
    "random_best_mean": statistics.mean(
      row["best"] for row in baseline
    ),
    "random_top_mean": statistics.mean(
      row["top_mean"] for row in baseline
    ),
    "inspection_savings": 1 - len(inspected) / len(test),
    "runtime": elapsed
  }

  print("Skill: choose")
  print(f"Available cars: {len(test)}")
  print(f"Inspections: {len(inspected)}/{len(test)}")
  print(f"Inspected indices: {inspected}")
  print(f"Recommended indices: {selected}")

  for key in [
    "best", "top_mean", "pool_best", "regret",
    "pool_median", "random_best_mean", "random_top_mean"
  ]:
    print(f"{key}: {summary[key]:.4f}")

  print(f"Inspection savings: {summary['inspection_savings']:.2%}")
  print(f"Runtime: {elapsed:.4f}s")
  print("Budget compliance: PASS")
  print("Recommendation validity: PASS")
  print("PASS")

  if len(sys.argv) > 2:
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(json.dumps({
      "seed": 1,
      "summary": summary,
      "inspected": [
        {"row": j, "score": scores[j]}
        for j in inspected
      ],
      "selected": [
        {"row": j, "score": scores[j]}
        for j in selected
      ],
      "random_baseline": baseline
    }, indent=2) + "\n")

    print(f"Results saved to: {out}")


if __name__ == "__main__":
  check(sys.argv[1])
