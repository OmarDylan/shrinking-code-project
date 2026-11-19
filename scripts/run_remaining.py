import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS = [
    "remember", "guess", "certify", "flow", "blame",
    "justify", "choose", "fix", "spot", "race"
]


def call(command, timeout=450):
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            command, 124, "",
            f"TIMEOUT: {exc}"
        )


def generate(treatment, step, repeat, cap):
    trial = f"r{repeat:02d}"
    folder = (
        ROOT / "experiments/runs" /
        f"t{treatment}" / trial / f"step{step:02d}"
    )
    metrics_file = folder / "metrics.json"

    if metrics_file.exists():
        try:
            metrics = json.loads(metrics_file.read_text())
            status = metrics.get("status", "unknown")
        except Exception:
            status = "invalid_metrics"

        if status == "generated":
            src = metrics.get("source")
            if not src or not (ROOT / src).exists():
                status = "missing_source"

        return treatment, f"EXISTING ({status})"

    if folder.exists():
        return treatment, "INCOMPLETE EXISTING RUN - NOT OVERWRITTEN"

    command = [
        sys.executable,
        str(ROOT / "scripts/generate.py"),
        "--treatment", str(treatment),
        "--step", str(step),
        "--repeat", str(repeat),
        "--model", "sonnet",
        "--cap", str(cap)
    ]

    result = call(command)

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "batch-log.txt").write_text(
        result.stdout + "\n" + result.stderr
    )

    if result.returncode:
        return treatment, (
            f"GENERATION FAILED ({result.returncode})\n"
            + (result.stderr or result.stdout)[-800:]
        )

    try:
        metrics = json.loads(metrics_file.read_text())
        return treatment, (
            f"GENERATED | "
            f"{metrics['added_physical_lines']} new LOC | "
            f"${metrics['estimated_cost_usd']:.4f} | "
            f"{metrics['generation_seconds']:.2f}s"
        )
    except Exception as exc:
        return treatment, f"METRICS ERROR: {exc}"


def evaluate(step, repeat):
    command = [
        sys.executable,
        str(ROOT / "scripts/evaluate_step.py"),
        "--step", str(step),
        "--repeat", str(repeat)
    ]

    result = call(command, timeout=1500)

    folder = ROOT / "experiments/batch"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / f"step{step:02d}-evaluation.txt").write_text(
        result.stdout + "\n" + result.stderr
    )

    for line in result.stdout.splitlines():
        if line.startswith("T") or "Saved comparison" in line:
            print("  " + line, flush=True)

    if result.returncode:
        print(
            "  EVALUATION ERROR:",
            result.stderr[-1000:],
            flush=True
        )

    return result.returncode


def checkpoint(step, repeat):
    command = [
        "git", "add", "-A", "--",
        "scripts/run_remaining.py",
        "src/t1", "src/t2", "src/t3",
        "src/t4", "src/t5", "src/t6"
    ]
    result = call(command, timeout=60)
    if result.returncode:
        print("Git staging issue:", result.stderr[-500:])

    for treatment in range(1, 7):
        folder = (
            ROOT / "experiments/runs" /
            f"t{treatment}" /
            f"r{repeat:02d}" /
            f"step{step:02d}"
        )

        for name in ["metrics.json", "prompt.txt", "batch-log.txt"]:
            path = folder / name

            if path.exists():
                call([
                    "git", "add", "-f", "--",
                    str(path.relative_to(ROOT))
                ], timeout=60)

    for folder in [
        ROOT / "experiments/summary",
        ROOT / "experiments/batch"
    ]:
        if folder.exists():
            call([
                "git", "add", "-f", "--",
                str(folder.relative_to(ROOT))
            ], timeout=90)

    changed = call([
        "git", "diff", "--cached", "--quiet"
    ], timeout=60)

    if changed.returncode == 1:
        result = call([
            "git", "commit", "-m",
            f"Record all treatments for step {step} pilot"
        ], timeout=120)
        if result.returncode:
            print("Commit failed:", result.stderr[-700:])
        else:
            print(f"  Step {step} committed.", flush=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["generate", "full", "evaluate"],
        default="generate"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cap", type=float, default=0.50)
    parser.add_argument("--start", type=int, default=3)
    parser.add_argument("--end", type=int, default=10)

    args = parser.parse_args()

    print("Project:", ROOT, flush=True)
    print("Mode:", args.mode, flush=True)
    print("Repeat:", args.repeat, flush=True)

    for step in range(args.start, args.end + 1):
        print(
            f"\n{'=' * 55}\n"
            f"STEP {step}/10 - {SKILLS[step-1].upper()}\n"
            f"{'=' * 55}",
            flush=True
        )

        if args.mode != "evaluate":
            with ThreadPoolExecutor(
                max_workers=args.workers
            ) as executor:
                jobs = [
                    executor.submit(
                        generate, treatment, step,
                        args.repeat, args.cap
                    )
                    for treatment in range(1, 7)
                ]

                for job in as_completed(jobs):
                    treatment, status = job.result()
                    print(
                        f"T{treatment}: {status}",
                        flush=True
                    )

        if args.mode in ("full", "evaluate"):
            evaluate(step, args.repeat)

        checkpoint(step, args.repeat)

    print("\n" + "=" * 55)
    print("BATCH COMPLETE")
    print("=" * 55)

    for step in range(args.start, args.end + 1):
        csv_path = (
            ROOT / "experiments/summary" /
            f"step{step:02d}-r{args.repeat:02d}.csv"
        )

        if not csv_path.exists():
            continue

        rows = list(csv.DictReader(csv_path.open()))
        passed = sum(r["status"] == "PASS" for r in rows)

        print(
            f"Step {step:02d}: "
            f"{passed}/{len(rows)} treatments passed"
        )

    status = call(["git", "status", "--short"])
    print("\nGit status:")
    print(status.stdout or "Clean working tree")


if __name__ == "__main__":
    main()
