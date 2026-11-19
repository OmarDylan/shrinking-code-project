import csv
import json
import os
import subprocess
import sys
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/splits/seed1/train.csv"
TEST = ROOT / "data/splits/seed1/test.csv"

OUT = ROOT / "experiments/summary"
OUT.mkdir(parents=True, exist_ok=True)

rows = []

for t in range(1, 7):
    name = "main.py" if t % 2 == 0 else "remember.py"
    source = ROOT / f"src/t{t}/r01" / name
    metrics_path = (
        ROOT / f"experiments/runs/t{t}/r01/step01/metrics.json"
    )

    if not source.exists() or not metrics_path.exists():
        print(f"T{t}: MISSING SOURCE OR METRICS")
        continue

    metrics = json.loads(metrics_path.read_text())
    usage = metrics.get("usage") or {}

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "vendor") +
        os.pathsep +
        env.get("PYTHONPATH", "")
    )

    command = [
        sys.executable, str(source),
        "remember",
        "--train", str(TRAIN),
        "--test", str(TEST)
    ]

    started = time.perf_counter()

    try:
        execution = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30
        )
        runtime = time.perf_counter() - started

        json.loads(execution.stdout)
        executable = execution.returncode == 0
    except Exception as exc:
        runtime = time.perf_counter() - started
        executable = False
        execution = None
        error = str(exc)

    passed = False
    evaluator_text = ""

    if executable:
        try:
            check = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tests/check_remember.py"),
                    str(source)
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60
            )

            evaluator_text = check.stdout + check.stderr
            passed = (
                check.returncode == 0
                and "PASS" in check.stdout.splitlines()
            )
        except Exception as exc:
            evaluator_text = str(exc)
    else:
        evaluator_text = (
            execution.stderr
            if execution is not None else error
        )

    report = {
        "treatment": f"T{t}",
        "library": metrics["library"],
        "reuse": metrics["reuse"],
        "model": metrics["requested_model"],
        "new_loc": metrics["added_physical_lines"],
        "total_loc": metrics["cumulative_physical_lines"],
        "regular_input_tokens": usage.get("input_tokens", 0),
        "cache_creation_tokens": usage.get(
            "cache_creation_input_tokens", 0
        ),
        "cache_read_tokens": usage.get(
            "cache_read_input_tokens", 0
        ),
        "total_input_tokens": sum([
            usage.get("input_tokens", 0),
            usage.get("cache_creation_input_tokens", 0),
            usage.get("cache_read_input_tokens", 0)
        ]),
        "output_tokens": usage.get("output_tokens", 0),
        "estimated_cost_usd": metrics["estimated_cost_usd"],
        "generation_seconds": metrics["generation_seconds"],
        "execution_seconds": runtime,
        "competency": "PASS" if passed else "FAIL"
    }

    rows.append(report)

    (OUT / f"t{t}-remember-r01-evaluation.txt").write_text(
        evaluator_text
    )

    print(
        f"T{t}: {report['competency']} | "
        f"{report['new_loc']} LOC | "
        f"${report['estimated_cost_usd']:.4f} | "
        f"{report['generation_seconds']:.2f}s generation | "
        f"{runtime:.4f}s execution"
    )

if rows:
    path = OUT / "remember-r01.csv"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nComparison saved to: {path}")
else:
    print("No valid treatment records found.")
    sys.exit(1)
