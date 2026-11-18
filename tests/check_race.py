import json
import math
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
from check_certify import expected

REPEATS = 20
BUDGET = 10
POOL = 40
STRATEGIES = ["adaptive", "random"]
METRICS = ["best5", "best10", "labels_to_target"]


def check(program):
  train, test = load(TRAIN), load(TEST)

  known = sorted(reference(train, train))
  target = known[len(known) // 4]

  scores = reference(train, test)
  rng = random.Random(1)

  episodes = []

  for _ in range(REPEATS):
    pool = rng.sample(range(len(test)), POOL)

    episodes.append({
      "pool": pool,
      "random_order": rng.sample(pool, BUDGET)
    })

  header = list(test[0])

  rows = [
    ["?" if name in GOALS else row[name]
     for name in header]
    for row in test
  ]

  payload = {
    "header": header,
    "rows": rows,
    "episodes": episodes
  }

  answers = []
  audit = []

  with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "race.json"
    path.write_text(json.dumps(payload))

    start = time.perf_counter()
    deadline = start + 60

    with tempfile.TemporaryFile(mode="w+t") as err:
      with subprocess.Popen(
        [
          sys.executable, program, "race",
          "--train", str(TRAIN),
          "--test", str(path)
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=err,
        text=True,
        bufsize=1
      ) as proc:

        messages = queue.Queue()

        def reader():
          try:
            for line in proc.stdout:
              messages.put(line)
          finally:
            messages.put(None)

        threading.Thread(
          target=reader, daemon=True
        ).start()

        def receive():
          try:
            line = messages.get(
              timeout=max(0, deadline-time.perf_counter())
            )
          except queue.Empty:
            raise TimeoutError("Race exceeded its time limit")

          if line is None:
            err.seek(0)
            raise RuntimeError(err.read() or "Unexpected exit")

          return json.loads(line)

        try:
          for e, episode in enumerate(episodes):
            answer = {"episode": e}

            for strategy in STRATEGIES:
              inspected = []

              for _ in range(BUDGET):
                msg = receive()

                assert set(msg) == {
                  "episode", "strategy", "inspect"
                }

                assert msg["episode"] == e
                assert msg["strategy"] == strategy

                j = msg["inspect"]

                assert type(j) is int
                assert j in episode["pool"]
                assert j not in inspected

                inspected.append(j)

                reply = {
                  "row": j,
                  "goals": {
                    name: float(test[j][name])
                    for name in GOALS
                  }
                }

                proc.stdin.write(json.dumps(reply) + "\n")
                proc.stdin.flush()

              if strategy == "random":
                assert inspected == episode["random_order"]

              values = [scores[j] for j in inspected]

              answer[strategy] = {
                "best5": min(values[:5]),
                "best10": min(values),
                "labels_to_target": next(
                  (k+1 for k, v in enumerate(values)
                   if round(v, 12) <= round(target, 12)),
                  BUDGET + 1
                )
              }

              audit.append({
                "episode": e,
                "strategy": strategy,
                "inspected": inspected,
                "scores": values
              })

            answers.append(answer)

          result = receive()

          proc.stdin.close()

          assert proc.wait(timeout=5) == 0

          assert messages.get(timeout=5) is None

        finally:
          if proc.poll() is None:
            proc.kill()
            proc.wait()

    elapsed = time.perf_counter() - start

  got = result["results"]

  assert len(got) == REPEATS

  for actual, want in zip(got, answers):
    assert actual["episode"] == want["episode"]

    for strategy in STRATEGIES:
      for key in METRICS:
        assert math.isclose(
          actual[strategy][key],
          want[strategy][key],
          abs_tol=1E-9
        )

  comparisons = {}

  print("Skill: race")
  print(f"Episodes: {REPEATS}")
  print(f"Cars per episode: {POOL}")
  print(f"Budget per strategy: {BUDGET}")
  print(f"Quality target: {target:.4f}")

  for metric in METRICS:
    xs = [r["adaptive"][metric] for r in answers]
    ys = [r["random"][metric] for r in answers]

    want = expected(xs, ys)
    actual = result["comparisons"][metric]

    assert actual["same"] is want["same"]

    for key in ["cohen", "cliffs", "ks"]:
      assert math.isclose(
        actual[key], want[key],
        rel_tol=1E-9,
        abs_tol=1E-9
      )

    comparisons[metric] = {
      "adaptive_mean": statistics.mean(xs),
      "random_mean": statistics.mean(ys),
      **want
    }

    print(f"\n{metric}:")
    print(f"  Adaptive: {statistics.mean(xs):.4f}")
    print(f"  Random:   {statistics.mean(ys):.4f}")
    print(f"  Cohen:    {want['cohen']:.4f}")
    print(f"  Cliffs:   {want['cliffs']:.4f}")
    print(f"  KS:       {want['ks']:.4f}")
    print(f"  Same:     {want['same']}")

  print("\nTarget reached:")

  reached = {}

  for strategy in STRATEGIES:
    hits = sum(
      row[strategy]["labels_to_target"] <= BUDGET
      for row in answers
    )

    reached[strategy] = hits

    print(f"  {strategy}: {hits}/{REPEATS}")

  print(f"\nRuntime: {elapsed:.4f}s")
  print("Budget compliance: PASS")
  print("Statistical comparisons: PASS")
  print("PASS")

  if len(sys.argv) > 2:
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(json.dumps({
      "seed": 1,
      "episodes": REPEATS,
      "target": target,
      "runtime": elapsed,
      "target_reached": reached,
      "comparisons": comparisons,
      "results": answers,
      "audit": audit
    }, indent=2) + "\n")

    print(f"Results saved to: {out}")


if __name__ == "__main__":
  check(sys.argv[1])
