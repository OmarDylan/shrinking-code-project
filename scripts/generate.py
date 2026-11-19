import argparse
import ast
import difflib
import json
import os
import re
import subprocess
import tempfile
import time

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKILLS = [
  "remember", "guess", "certify", "flow", "blame",
  "justify", "choose", "fix", "spot", "race"
]

TREATMENTS = {
  1: ("python", False),
  2: ("python", True),
  3: ("sklearn", False),
  4: ("sklearn", True),
  5: ("ezr", False),
  6: ("ezr", True)
}

REQUIREMENTS = [
  (
    "From hundreds of cars, show me the unusually great buys.",
    "Show me what a typical good car, average car, and poor car look like."
  ),
  (
    "Guess a car's fuel use before I see the final sticker information.",
    "Estimate how promising a newly listed car is with partial specifications."
  ),
  (
    "Tell me when one group of cars is really better than another.",
    "Tell me whether an apparent improvement is large enough to matter."
  ),
  (
    "Judge each new car as it arrives.",
    "Warn me when the quality of newly arriving cars starts looking different."
  ),
  (
    "Tell me why a poor car is slow, heavy, or thirsty.",
    "Tell me which specifications are associated with poor quality."
  ),
  (
    "Explain why you recommended one car instead of another.",
    "Explain why a car was accepted or rejected."
  ),
  (
    "Find me a great car with as few inspections as possible.",
    "Find strong alternatives without inspecting the whole inventory."
  ),
  (
    "Tell me the smallest change that could improve a poor car.",
    "Show me how a car could resemble better cars in the inventory."
  ),
  (
    "Flag a strange listing that might contain a typo or bad measurement.",
    "Show me unusually different cars."
  ),
  (
    "Tell me whether our selection process beats random selection.",
    "Tell me which approach finds good cars with fewer inspections."
  )
]


def contract(step):
  text = (ROOT / "docs/acceptance.md").read_text()

  sections = re.split(r"(?m)^## Step \d+:", text)

  return sections[step].strip()


def background(library):
  if library == "python":
    return "Use Python standard library only."

  if library == "sklearn":
    return (
      "Use scikit-learn and its dependencies where appropriate. "
      "Python standard library is also permitted."
    )

  text = (
    "Use the supplied EZR library where appropriate. "
    "Python standard library is also permitted.\n"
  )

  for name in ["ezr.py", "ezr_eg.py"]:
    source = (ROOT / "vendor" / name).read_text()
    text += f"\n--- {name} ---\n{source}\n"

  return text


def make_prompt(step, library, reuse, previous):
  skill = SKILLS[step-1]

  current = contract(step)

  if not reuse:
    current = current.replace(
      "- Preserve all previous skills.", ""
    )

    current = current.replace(
      "- Preserve all functionality from step 1.", ""
    )

  prompt = f"""
Implement a compact Python program for car inventory analysis.

Dataset header:
Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+

Uppercase initial: numerical column.
Lowercase initial: symbolic column.
X suffix: ignored column.
+ suffix: maximize.
- suffix: minimize.
? means missing.

Current skill: {skill}

Customer requirements:
{REQUIREMENTS[step-1][0]}
{REQUIREMENTS[step-1][1]}

Acceptance contract:
{current}

Follow the specified input and output interface.
Read datasets only from the supplied command-line paths.
Do not hardcode dataset rows or expected test answers.
Do not access the network or unrelated files.
Do not inspect evaluator source.

Return one complete Python source file.
Return only Python code.
Do not include Markdown or explanations.

Background library:
{background(library)}
"""

  if step > 1 and reuse:
    prompt += """
Extend the existing program.
Reuse its structures and functions whenever possible.
Preserve all previous skill interfaces.
Return the complete updated main.py.
"""

    for j in range(1, step):
      prompt += (
        f"\nPrevious skill {SKILLS[j-1]} contract:\n"
        + contract(j) + "\n"
      )

    prompt += "\nExisting program:\n" + previous

  elif step > 1:
    prompt += """
Implement this skill independently.
Do not import or depend on implementations of earlier skills.
"""

  return prompt


def code_only(text):
  text = text.strip()

  if text.startswith("```"):
    lines = text.splitlines()

    if lines[-1].strip() == "```":
      text = "\n".join(lines[1:-1])

  ast.parse(text)

  return text.rstrip() + "\n"


def loc_diff(old, new):
  added = removed = 0

  for line in difflib.ndiff(
    old.splitlines(), new.splitlines()
  ):
    if line.startswith("+ "):
      added += 1
    elif line.startswith("- "):
      removed += 1

  return added, removed


def generate(args):
  number = args.treatment
  step = args.step
  skill = SKILLS[step-1]

  library, reuse = TREATMENTS[number]

  trial = f"r{args.repeat:02d}"

  source_dir = ROOT / "src" / f"t{number}" / trial

  run_dir = (
    ROOT / "experiments/runs" /
    f"t{number}" / trial / f"step{step:02d}"
  )

  if run_dir.exists():
    raise SystemExit(
      f"Run already exists: {run_dir}. "
      "Use another repeat rather than overwriting evidence."
    )

  source_dir.mkdir(parents=True, exist_ok=True)
  run_dir.mkdir(parents=True)

  destination = source_dir / (
    "main.py" if reuse else f"{skill}.py"
  )

  previous = ""

  if reuse and step > 1:
    previous = destination.read_text()

  if destination.exists() and (not reuse or step == 1):
    raise SystemExit(f"Source already exists: {destination}")

  prompt = make_prompt(
    step, library, reuse, previous
  )

  (run_dir / "prompt.txt").write_text(prompt)

  env = os.environ.copy()

  for key in [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY"
  ]:
    env.pop(key, None)

  command = [
    "claude", "-p",
    "--model", args.model,
    "--output-format", "json",
    "--safe-mode",
    "--no-session-persistence",
    "--tools", "",
    "--strict-mcp-config",
    "--mcp-config", '{"mcpServers":{}}',
    "--max-turns", "1",
    "--max-budget-usd", str(args.cap),
    "--system-prompt",
    "You write compact, correct Python programs."
  ]

  print(
    f"T{number} | {trial} | Step {step}: {skill}",
    flush=True
  )

  start = time.perf_counter()

  with tempfile.TemporaryDirectory() as folder:
    run = subprocess.run(
      command,
      input=prompt,
      text=True,
      capture_output=True,
      cwd=folder,
      env=env,
      timeout=300
    )

  elapsed = time.perf_counter() - start

  (run_dir / "response.json").write_text(run.stdout)
  (run_dir / "stderr.txt").write_text(run.stderr)

  try:
    data = json.loads(run.stdout)
  except json.JSONDecodeError:
    data = {}

  metrics = {
    "treatment": number,
    "repeat": args.repeat,
    "step": step,
    "skill": skill,
    "library": library,
    "reuse": reuse,
    "requested_model": args.model,
    "model_usage": data.get("modelUsage"),
    "usage": data.get("usage"),
    "estimated_cost_usd": data.get("total_cost_usd"),
    "generation_seconds": elapsed,
    "returncode": run.returncode,
    "status": "generation_failed"
  }

  try:
    if run.returncode != 0 or data.get("is_error"):
      raise RuntimeError("Claude generation failed")

    source = code_only(data["result"])

    destination.write_text(source)

    (run_dir / "generated.py").write_text(source)

    added, removed = loc_diff(previous, source)
    total = len(source.splitlines())

    metrics.update({
      "added_physical_lines": added,
      "removed_physical_lines": removed,
      "file_physical_lines": total,
      "cumulative_physical_lines": sum(
        len(p.read_text().splitlines())
        for p in source_dir.glob("*.py")
      ),
      "source": str(destination.relative_to(ROOT)),
      "status": "generated"
    })

  except Exception as exc:
    metrics["error"] = str(exc)

  finally:
    (run_dir / "metrics.json").write_text(
      json.dumps(metrics, indent=2) + "\n"
    )

  print(json.dumps(metrics, indent=2))

  if metrics["status"] != "generated":
    raise SystemExit(1)


if __name__ == "__main__":
  p = argparse.ArgumentParser()

  p.add_argument(
    "--treatment", type=int, choices=range(1, 7),
    required=True
  )
  p.add_argument(
    "--step", type=int, choices=range(1, 11),
    required=True
  )
  p.add_argument("--repeat", type=int, default=1)
  p.add_argument("--model", default="sonnet")
  p.add_argument("--cap", type=float, default=0.50)

  generate(p.parse_args())
