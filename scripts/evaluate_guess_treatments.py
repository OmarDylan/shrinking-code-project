import csv
import json
import os
import subprocess
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/summary"
OUT.mkdir(parents=True, exist_ok=True)

env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT / "vendor") + os.pathsep + env.get("PYTHONPATH", "")

records = []

def evaluate(test, source, result=None):
    cmd = [sys.executable, str(ROOT / "tests" / test), str(source)]
    if result is not None:
        cmd.append(str(result))

    try:
        run = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120
        )
        return run.returncode, run.stdout + run.stderr
    except Exception as exc:
        return -1, str(exc)

for t in range(1, 7):
    reuse = t % 2 == 0
    source = ROOT / f"src/t{t}/r01" / (
        "main.py" if reuse else "guess.py"
    )
    metrics_path = ROOT / f"experiments/runs/t{t}/r01/step02/metrics.json"

    if not source.exists() or not metrics_path.exists():
        print(f"T{t}: MISSING SOURCE OR METRICS")
        continue

    metrics = json.loads(metrics_path.read_text())
    usage = metrics.get("usage") or {}

    raw = OUT / f"t{t}-guess-r01.json"
    code, output = evaluate("check_guess.py", source, raw)

    guess_pass = code == 0 and "PASS (valid predictions)" in output

    (OUT / f"t{t}-guess-r01-evaluation.txt").write_text(output)

    remember_pass = None
    if reuse:
        rcode, rout = evaluate("check_remember.py", source)
        remember_pass = rcode == 0 and "PASS" in rout.splitlines()
        (OUT / f"t{t}-remember-regression-r01.txt").write_text(rout)

    summary = {}
    if raw.exists():
        payload = json.loads(raw.read_text())
        summary = payload.get("summary", {})
        summary["runtime"] = payload.get(
            "runtime", summary.get("runtime")
        )

    status = "PASS" if guess_pass and remember_pass is not False else "FAIL"

    record = {
        "treatment": f"T{t}",
        "library": metrics["library"],
        "reuse": reuse,
        "status": status,
        "guess_pass": guess_pass,
        "remember_regression_pass": remember_pass,
        "new_loc": metrics["added_physical_lines"],
        "removed_loc": metrics["removed_physical_lines"],
        "total_loc": metrics["cumulative_physical_lines"],
        "regular_input_tokens": usage.get("input_tokens", 0),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "estimated_cost_usd": metrics["estimated_cost_usd"],
        "generation_seconds": metrics["generation_seconds"],
        "mpg_error": summary.get("mpg_error"),
        "disty_error": summary.get("disty_error"),
        "baseline_mpg_error": summary.get("baseline_mpg_error"),
        "baseline_disty_error": summary.get("baseline_disty_error"),
        "execution_seconds": summary.get("runtime")
    }

    records.append(record)

    print(f"\nT{t}: {status}")
    print(f"  New LOC: {record['new_loc']}")
    print(f"  Total LOC: {record['total_loc']}")
    print(f"  MPG error: {record['mpg_error']}")
    print(f"  Disty error: {record['disty_error']}")
    if reuse:
        print(f"  Remember regression: {remember_pass}")
    if not guess_pass:
        print("  Evaluation error:")
        print(output[-1800:])

if records:
    csv_path = OUT / "guess-r01.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved: {csv_path}")
    print(f"Evaluated {len(records)}/6 treatments")
else:
    raise SystemExit("No treatment records available.")
