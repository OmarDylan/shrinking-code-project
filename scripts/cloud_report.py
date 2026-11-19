import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments/summary"
OUT = ROOT / "experiments/overnight"
OUT.mkdir(parents=True, exist_ok=True)

SKILLS = [
    "remember", "guess", "certify", "flow", "blame",
    "justify", "choose", "fix", "spot", "race"
]

def read_csv(name):
    path = SUMMARY / name
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def find_result(t, step):
    if step == 1:
        rows = read_csv("remember-r01.csv")
        field = "competency"
    elif step == 2:
        rows = read_csv("guess-r01.csv")
        field = "status"
    else:
        rows = read_csv(f"step{step:02d}-r01.csv")
        field = "status"

    for row in rows:
        if row.get("treatment") == f"T{t}":
            return row.get(field, "UNKNOWN")

    return "NOT_EVALUATED"

records = []
totals = []

for t in range(1, 7):
    cost = 0.0
    seconds = 0.0
    input_tokens = 0
    output_tokens = 0
    passes = 0
    cumulative = 0

    for step, skill in enumerate(SKILLS, 1):
        path = (
            ROOT / "experiments/runs" /
            f"t{t}/r01/step{step:02d}/metrics.json"
        )

        if not path.exists():
            records.append({
                "treatment": f"T{t}",
                "step": step,
                "skill": skill,
                "status": "MISSING_GENERATION"
            })
            continue

        m = json.loads(path.read_text())
        u = m.get("usage") or {}

        inputs = sum(
            int(u.get(key, 0))
            for key in [
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens"
            ]
        )

        status = find_result(t, step)

        cost += float(m.get("estimated_cost_usd") or 0)
        seconds += float(m.get("generation_seconds") or 0)
        input_tokens += inputs
        output_tokens += int(u.get("output_tokens", 0))

        cumulative = m.get(
            "cumulative_physical_lines", cumulative
        )

        if status == "PASS":
            passes += 1

        records.append({
            "treatment": f"T{t}",
            "step": step,
            "skill": skill,
            "status": status,
            "new_loc": m.get("added_physical_lines"),
            "removed_loc": m.get("removed_physical_lines"),
            "cumulative_loc": cumulative,
            "input_tokens": inputs,
            "output_tokens": u.get("output_tokens"),
            "estimated_cost_usd": m.get("estimated_cost_usd"),
            "generation_seconds": m.get("generation_seconds")
        })

    totals.append({
        "treatment": f"T{t}",
        "passed": passes,
        "tested_skills": 10,
        "final_loc": cumulative,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "estimated_total_cost_usd": round(cost, 6),
        "total_generation_seconds": round(seconds, 3)
    })

def save_csv(path, data):
    keys = list(dict.fromkeys(
        key for record in data for key in record
    ))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

save_csv(OUT / "all_60_generations.csv", records)
save_csv(OUT / "treatment_totals.csv", totals)

lines = [
    "# CSC491 — Six-Treatment Pilot Results",
    "",
    "These are first-run results, not independent repetitions.",
    "T7 is not included.",
    "",
    "## Overall results",
    "",
    "| Treatment | Skills passed | Final LOC | "
    "Input tokens | Output tokens | Estimated cost | "
    "Generation time |",
    "|---|---:|---:|---:|---:|---:|---:|"
]

for r in totals:
    lines.append(
        f"| {r['treatment']} | {r['passed']}/10 | "
        f"{r['final_loc']} | "
        f"{r['total_input_tokens']:,} | "
        f"{r['total_output_tokens']:,} | "
        f"${r['estimated_total_cost_usd']:.4f} | "
        f"{r['total_generation_seconds']:.1f}s |"
    )

lines += [
    "",
    "## Competency by skill",
    "",
    "| Skill | T1 | T2 | T3 | T4 | T5 | T6 |",
    "|---|---|---|---|---|---|---|"
]

for step, skill in enumerate(SKILLS, 1):
    statuses = [
        find_result(t, step) for t in range(1, 7)
    ]
    lines.append(
        f"| {skill} | " + " | ".join(statuses) + " |"
    )

lines += [
    "",
    "## Interpretation limits",
    "",
    "- Only one Claude generation per automated treatment.",
    "- No statistical inference about treatments is justified yet.",
    "- T7 requires separate manual prudence checks.",
    "- Generated programs were not automatically repaired.",
    "- Dollar values are API-equivalent estimated generation costs.",
    "- Source LOC excludes background libraries and evaluators.",
    "- Reuse implementations were evaluated at their historical"
    " step-specific Git snapshots.",
    "- Test execution used a cloud runner, which may have"
    " different dependencies or performance than the Mac.",
    "",
    "## Generated charts",
    "",
    "- `cumulative_loc.png`",
    "- `cumulative_cost.png`",
    ""
]

(OUT / "REPORT.md").write_text("\n".join(lines))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for field, filename, title, ylabel in [
        (
            "cumulative_loc", "cumulative_loc.png",
            "Cumulative generated source LOC",
            "Physical lines"
        ),
        (
            "estimated_cost_usd", "cumulative_cost.png",
            "Cumulative estimated generation cost",
            "USD"
        )
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))

        for t in range(1, 7):
            data = [
                r for r in records
                if r["treatment"] == f"T{t}"
                and r.get(field) is not None
            ]

            xs = [r["step"] for r in data]
            ys = [float(r[field]) for r in data]

            if field == "estimated_cost_usd":
                ys = [
                    sum(ys[:i+1]) for i in range(len(ys))
                ]

            ax.plot(xs, ys, marker="o", label=f"T{t}")

        ax.set(
            title=title, xlabel="Skill number", ylabel=ylabel
        )
        ax.set_xticks(range(1, 11))
        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=150)
        plt.close(fig)

except ImportError as exc:
    print("Charts unavailable:", exc)

print("\n".join(lines))
print("\nResults saved to:", OUT)
