import csv
import random
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1

with open(ROOT / "data/auto93.csv", newline="") as f:
  reader = csv.reader(f)
  header = next(reader)
  rows = list(reader)

random.Random(seed).shuffle(rows)

n = int(len(rows) * 0.8)

train = rows[:n]
test = rows[n:]

out = ROOT / "data/splits" / f"seed{seed}"
out.mkdir(parents=True, exist_ok=True)

for name, subset in [("train", train), ("test", test)]:
  with open(out / f"{name}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(subset)

assert len(train) + len(test) == len(rows)

print(f"Seed: {seed}")
print(f"Total: {len(rows)}")
print(f"Training: {len(train)}")
print(f"Testing: {len(test)}")
print(f"Saved to: {out}")
