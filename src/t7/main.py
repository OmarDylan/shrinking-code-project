import argparse
import csv
import json

from math import sqrt, exp

TINY = 1E-32


def thing(s):
  try:
    return int(s)
  except ValueError:
    try:
      return float(s)
    except ValueError:
      return s


def csv_rows(file):
  with open(file, newline="") as src:
    for row in csv.reader(src):
      if row:
        yield [thing(x.strip()) for x in row]


class Num:
  def __init__(i, name, at):
    i.name, i.at = name, at
    i.n = i.mu = i.m2 = 0
    i.heaven = 0 if name.endswith("-") else 1

  def add(i, v):
    if v != "?":
      i.n += 1
      d = v - i.mu
      i.mu += d / i.n
      i.m2 += d * (v - i.mu)

  def div(i):
    return sqrt(max(i.m2, 0) / (i.n - 1)) if i.n > 1 else 0

  def norm(i, v):
    z = (v - i.mu) / (i.div() + TINY)
    z = max(-3, min(3, z))
    return 1 / (1 + exp(-1.702 * z))


class Sym:
  def __init__(i, name, at):
    i.name, i.at = name, at
    i.n, i.has = 0, {}

  def add(i, v):
    if v != "?":
      i.n += 1
      i.has[v] = i.has.get(v, 0) + 1


class Cols:
  def __init__(i, names):
    i.names = names
    i.x, i.y = [], []

    for at, name in enumerate(names):
      if name.endswith("X"):
        continue
      col = Sym(name, at) if name[0].islower() else Num(name, at)
      (i.y if name.endswith(("+", "-")) else i.x).append(col)

  def add(i, row):
    for col in i.x + i.y:
      col.add(row[col.at])


class Tbl:
  def __init__(i, src):
    src = iter(src)
    i.cols = Cols(next(src))
    i.rows = []

    for row in src:
      i.add(row)

  def add(i, row):
    i.cols.add(row)
    i.rows.append(row)

  def distx(i, a, b):
    def gap(col):
      x, y = a[col.at], b[col.at]

      if x == "?" and y == "?":
        return 1

      if isinstance(col, Sym):
        return int(x != y)

      x = col.norm(x) if x != "?" else "?"
      y = col.norm(y) if y != "?" else "?"

      if x == "?":
        x = 0 if y > 0.5 else 1
      if y == "?":
        y = 0 if x > 0.5 else 1

      return abs(x - y)

    return sqrt(sum(gap(col) ** 2 for col in i.cols.x)
                / len(i.cols.x))

  def disty(i, row):
    return sqrt(sum((y.norm(row[y.at]) - y.heaven) ** 2
                    for y in i.cols.y) / len(i.cols.y))


def remember(train, test):
  t = Tbl(csv_rows(train))
  rows = csv_rows(test)

  if next(rows) != t.cols.names:
    raise ValueError("Train and test headers differ")

  scores = [t.disty(row) for row in rows]
  order = sorted(range(len(scores)), key=lambda j: (scores[j], j))

  return {
    "scores": scores,
    "top": order[:10],
    "best": order[0],
    "median": order[len(order) // 2],
    "worst": order[-1]
  }


def guess(train, test, k=5):
  t = Tbl(csv_rows(train))
  rows = csv_rows(test)

  if next(rows) != t.cols.names:
    raise ValueError("Train and test headers differ")

  mpg_at = t.cols.names.index("Mpg+")
  known = [t.disty(row) for row in t.rows]
  mpg, scores = [], []

  for row in rows:
    near = sorted(range(len(t.rows)),
                  key=lambda j: t.distx(row, t.rows[j]))[:k]

    mpg.append(sum(t.rows[j][mpg_at] for j in near) / len(near))
    scores.append(sum(known[j] for j in near) / len(near))

  return {"mpg": mpg, "scores": scores}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("skill", choices=["remember", "guess"])
  p.add_argument("--train", required=True)
  p.add_argument("--test", required=True)
  args = p.parse_args()

  skills = {"remember": remember, "guess": guess}
  print(json.dumps(skills[args.skill](args.train, args.test)))


if __name__ == "__main__":
  main()
