import json
import math
import statistics
import subprocess
import sys
import time

from pathlib import Path
from check_remember import load, reference, TRAIN, TEST
from check_blame import expected, order


def check(program):
  train, test = load(TRAIN), load(TEST)

  history = sorted(reference(train, train))
  cutoff = history[len(history) // 2]

  scores = reference(train, test)
  diagnoses = expected(train, test)

  start = time.perf_counter()

  run = subprocess.run(
    [
      sys.executable, program, "justify",
      "--train", str(TRAIN),
      "--test", str(TEST)
    ],
    capture_output=True,
    text=True,
    check=True,
    timeout=60
  )

  elapsed = time.perf_counter() - start

  results = json.loads(run.stdout)["explanations"]

  assert len(results) == len(test)

  errors = []

  for j, actual in enumerate(results):
    k = (j + 1) % len(test)

    a, b = scores[j], scores[k]

    accepted = round(a, 12) <= round(cutoff, 12)

    winner = None if round(a, 12) == round(b, 12) else (
      j if a < b else k
    )

    advantage = None

    if winner is not None:
      loser = k if winner == j else j

      gaps = {
        name: diagnoses[loser]["goals"][name] ** 2
              - diagnoses[winner]["goals"][name] ** 2
        for name in diagnoses[winner]["goals"]
      }

      advantage = order(gaps)[0]

    goals = diagnoses[j]["goals"]
    features = diagnoses[j]["features"]

    weakest = order(goals)[0]
    feature = order(features)[0]
    feature = feature if features[feature] > 1E-10 else None

    assert actual["row"] == j
    assert actual["other"] == k
    assert actual["accepted"] is accepted
    assert actual["winner"] == winner
    assert actual["advantage_goal"] == advantage
    assert actual["weakest_goal"] == weakest
    assert actual["top_feature"] == feature

    for key, value in [
      ("score", a),
      ("other_score", b),
      ("cutoff", cutoff)
    ]:
      assert math.isclose(actual[key], value, abs_tol=1E-9)

    text = actual["text"]
    assert isinstance(text, str) and text.strip()

    verdict = "accepted" if accepted else "rejected"

    assert f"Car {j} {verdict}" in text
    assert f"{a:.3f}" in text
    assert f"{b:.3f}" in text
    assert f"{cutoff:.3f}" in text
    assert weakest in text

    if winner is None:
      assert "tie" in text.lower()
    else:
      assert f"Recommend car {winner}" in text
      assert advantage in text

    if feature is not None:
      assert feature in text

    errors.append({
      "row": j,
      "other": k,
      "score_error": abs(actual["score"] - a),
      "accepted": accepted,
      "winner": winner,
      "correct": True,
      "text": text
    })

  summary = {
    "cases": len(errors),
    "correct": len(errors),
    "accepted": sum(row["accepted"] for row in errors),
    "rejected": sum(not row["accepted"] for row in errors),
    "ties": sum(row["winner"] is None for row in errors),
    "score_error": statistics.mean(
      row["score_error"] for row in errors
    ),
    "runtime": elapsed
  }

  print("Skill: justify")
  print(f"Explained cars: {len(errors)}")
  print(f"Correct: {summary['correct']}/{len(test)}")
  print(f"Accepted: {summary['accepted']}")
  print(f"Rejected: {summary['rejected']}")
  print(f"Ties: {summary['ties']}")
  print(f"Score error: {summary['score_error']:.8f}")
  print(f"Runtime: {elapsed:.4f}s")
  print("PASS")

  print("\nExample explanation:")
  print(results[0]["text"])

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
