import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = [
    "remember", "guess", "certify", "flow", "blame",
    "justify", "choose", "fix", "spot", "race"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    skill = SKILLS[args.step - 1]
    trial = f"r{args.repeat:02d}"
    out = ROOT / "experiments" / "summary"
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "vendor") + os.pathsep + env.get("PYTHONPATH", "")

    records = []

    for t in range(1, 7):
        reuse = t % 2 == 0
        source = ROOT / "src" / f"t{t}" / trial / (
            "main.py" if reuse else f"{skill}.py"
        )
        run_dir = ROOT / "experiments" / "runs" / f"t{t}" / trial / f"step{args.step:02d}"
        metrics_file = run_dir / "metrics.json"

        if not source.exists() or not metrics_file.exists():
            print(f"T{t}: Missing source or metrics.")
            continue

        metrics = json.loads(metrics_file.read_text())
        usage = metrics.get("usage") or {}
        checked = range(1, args.step + 1) if reuse else [args.step]

        passed = 0
        checks = []
        primary_summary = {}
        primary_runtime = None

        for number in checked:
            name = SKILLS[number - 1]
            log = out / f"t{t}-{trial}-step{args.step:02d}-{name}.txt"
            result = out / f"t{t}-{trial}-step{args.step:02d}-{name}.json"

            command = [
                sys.executable,
                str(ROOT / "tests" / f"check_{name}.py"),
                str(source)
            ]

            if name != "remember":
                command.append(str(result))

            start = time.perf_counter()

            try:
                run = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=130
                )
                output = run.stdout + run.stderr
                ok = run.returncode == 0 and bool(
                    re.search(r"(?m)^PASS\b", run.stdout)
                )
            except Exception as exc:
                output = f"{type(exc).__name__}: {exc}"
                ok = False

            wall = time.perf_counter() - start
            log.write_text(output)

            if ok:
                passed += 1

            checks.append(f"{name}:{'PASS' if ok else 'FAIL'}")

            if number == args.step and result.exists():
                payload = json.loads(result.read_text())
                primary_summary = payload.get("summary", {})
                primary_runtime = payload.get(
                    "runtime",
                    primary_summary.get("runtime")
                )

            if number == args.step and primary_runtime is None:
                match = re.search(r"Runtime:\s*([0-9.]+)s", output)
                if match:
                    primary_runtime = float(match.group(1))

            print(
                f"  T{t} {name}: "
                f"{'PASS' if ok else 'FAIL'} "
                f"(test wall time {wall:.2f}s)"
            )

        record = {
            "treatment": f"T{t}",
            "repeat": args.repeat,
            "skill": skill,
            "library": metrics.get("library"),
            "reuse": reuse,
            "status": "PASS" if passed == len(checks) else "FAIL",
            "checks": ";".join(checks),
            "new_loc": metrics.get("added_physical_lines"),
            "removed_loc": metrics.get("removed_physical_lines"),
            "total_loc": metrics.get("cumulative_physical_lines"),
            "input_tokens": sum(
                usage.get(k, 0) for k in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens"
                )
            ),
            "cache_creation_tokens": usage.get(
                "cache_creation_input_tokens", 0
            ),
            "cache_read_tokens": usage.get(
                "cache_read_input_tokens", 0
            ),
            "output_tokens": usage.get("output_tokens", 0),
            "estimated_cost_usd": metrics.get("estimated_cost_usd"),
            "generation_seconds": metrics.get("generation_seconds"),
            "execution_seconds": primary_runtime,
            "mpg_error": primary_summary.get("mpg_error"),
            "disty_error": primary_summary.get("disty_error"),
            "accuracy": primary_summary.get("accuracy"),
            "f1": primary_summary.get("f1"),
            "regret": primary_summary.get("regret"),
            "source": str(source.relative_to(ROOT))
        }
        records.append(record)

        print(
            f"T{t}: {record['status']} | "
            f"new LOC {record['new_loc']} | "
            f"total LOC {record['total_loc']}\n"
        )

    if not records:
        raise SystemExit("No generated treatments available.")

    path = out / f"step{args.step:02d}-{trial}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved comparison to {path}")

if __name__ == "__main__":
    main()
