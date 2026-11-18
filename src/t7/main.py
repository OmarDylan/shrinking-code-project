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


def predict_row(t, row, known, mpg_at, k=5, stable=False):
  def key(j):
    d = t.distx(row, t.rows[j])
    return round(d, 12) if stable else d

  near = sorted(range(len(t.rows)), key=key)[:k]

  mpg = sum(t.rows[j][mpg_at] for j in near) / len(near)
  score = sum(known[j] for j in near) / len(near)

  return mpg, score


def predictions(t, test, k=5):
  rows = csv_rows(test)

  if next(rows) != t.cols.names:
    raise ValueError("Train and test headers differ")

  mpg_at = t.cols.names.index("Mpg+")
  known = [t.disty(row) for row in t.rows]

  for row in rows:
    yield predict_row(t, row, known, mpg_at, k)


def guess(train, test, k=5):
  t = Tbl(csv_rows(train))
  mpg, scores = [], []

  for m, s in predictions(t, test, k):
    mpg.append(m)
    scores.append(s)

  return {"mpg": mpg, "scores": scores}


def cohen(xs, ys):
  a, b = Num("", 0), Num("", 0)

  for x in xs:
    a.add(x)
  for y in ys:
    b.add(y)

  pool = ((a.n-1)*a.div()**2 + (b.n-1)*b.div()**2) / (a.n+b.n-2)

  return abs(a.mu - b.mu) / (sqrt(pool) + TINY)


def cliffs(xs, ys):
  gt = lt = j = k = 0

  for x in xs:
    while j < len(ys) and ys[j] < x:
      j += 1
      k = j
    while k < len(ys) and ys[k] == x:
      k += 1
    gt += j
    lt += len(ys) - k

  return abs(gt - lt) / (len(xs) * len(ys))


def ks(xs, ys):
  n, m = len(xs), len(ys)
  i = j = d = 0

  while i < n and j < m:
    v = min(xs[i], ys[j])
    while i < n and xs[i] == v:
      i += 1
    while j < m and ys[j] == v:
      j += 1
    d = max(d, abs(i/n - j/m))

  return d / sqrt((n+m)/(n*m))


def compare(xs, ys):
  xs, ys = sorted(xs), sorted(ys)
  c, d, k = cohen(xs, ys), cliffs(xs, ys), ks(xs, ys)

  return {
    "cohen": c,
    "cliffs": d,
    "ks": k,
    "same": c <= 0.35 and d <= 0.197 and k <= 1.36
  }


def certify(train, test):
  with open(test) as f:
    cases = json.load(f)

  return {"comparisons": [
    compare(case["xs"], case["ys"]) for case in cases
  ]}


def flow(train, test, window=20):
  t = Tbl(csv_rows(train))
  history = [t.disty(row) for row in t.rows]
  mpg, scores, alerts = [], [], []

  for m, s in predictions(t, test):
    mpg.append(m)
    scores.append(s)

    recent = scores[-window:]
    alerts.append(len(recent) == window and
                  not compare(history, recent)["same"])

  return {"mpg": mpg, "scores": scores, "alerts": alerts}


def profiles(t):
  rows = sorted(t.rows, key=lambda r: round(t.disty(r), 12))
  n = max(1, len(rows) // 4)

  return (Tbl([t.cols.names, *rows[:n]]),
          Tbl([t.cols.names, *rows[-n:]]))


def descending(scores):
  return sorted(scores, key=lambda k: (-round(scores[k], 10), k))


def diagnosis(t, row, good, poor):
  goals = {
    y.name: abs(y.norm(row[y.at]) - y.heaven)
    for y in t.cols.y
  }

  features = {}

  for col, g, p in zip(t.cols.x, good.cols.x, poor.cols.x):
    v = row[col.at]

    if v == "?":
      score = 0

    elif isinstance(col, Num):
      x = col.norm(v)
      score = abs(x - col.norm(g.mu)) - abs(x - col.norm(p.mu))

    else:
      score = (p.has.get(v, 0) / max(1, p.n)
               - g.has.get(v, 0) / max(1, g.n))

    features[col.name] = max(0, score)

  return {
    "goals": goals,
    "features": features,
    "goal_order": descending(goals),
    "feature_order": descending(features)
  }


def blame(train, test):
  t = Tbl(csv_rows(train))
  good, poor = profiles(t)
  rows = csv_rows(test)

  if next(rows) != t.cols.names:
    raise ValueError("Train and test headers differ")

  return {"diagnoses": [
    diagnosis(t, row, good, poor) for row in rows
  ]}


def justify(train, test):
  t = Tbl(csv_rows(train))
  good, poor = profiles(t)
  rows = csv_rows(test)

  if next(rows) != t.cols.names:
    raise ValueError("Train and test headers differ")

  rows = list(rows)
  history = sorted(t.disty(row) for row in t.rows)
  cutoff = history[len(history) // 2]
  scores = [t.disty(row) for row in rows]
  ds = [diagnosis(t, row, good, poor) for row in rows]
  out = []

  for j, a in enumerate(scores):
    k = (j + 1) % len(rows)
    b = scores[k]
    accepted = round(a, 12) <= round(cutoff, 12)
    weakest = ds[j]["goal_order"][0]
    feature = ds[j]["feature_order"][0]
    feature = feature if ds[j]["features"][feature] > 1E-10 else None

    winner = advantage = None
    comparison = "The cars tie."

    if round(a, 12) != round(b, 12):
      winner, loser = (j, k) if a < b else (k, j)
      gaps = {
        name: ds[loser]["goals"][name]**2
              - ds[winner]["goals"][name]**2
        for name in ds[winner]["goals"]
      }
      advantage = descending(gaps)[0]
      comparison = f"Recommend car {winner}; largest advantage: {advantage}."

    verdict = "accepted" if accepted else "rejected"
    text = (f"Car {j} {verdict}: quality gap {a:.3f}, "
            f"cutoff {cutoff:.3f}. Car {k} has gap {b:.3f}. "
            f"{comparison} Weakest goal: {weakest}.")

    if feature is not None:
      text += f" Feature associated with poor cars: {feature}."

    out.append({
      "row": j, "other": k, "score": a, "other_score": b,
      "cutoff": cutoff, "accepted": accepted, "winner": winner,
      "advantage_goal": advantage, "weakest_goal": weakest,
      "top_feature": feature, "text": text
    })

  return {"explanations": out}


def choose(train, test, budget=10):
  t = Tbl(csv_rows(train))
  src = csv_rows(test)

  if next(src) != t.cols.names:
    raise ValueError("Train and test headers differ")

  rows = list(src)
  prior = [s for _, s in predictions(t, test)]
  seen = {}

  def estimate(j):
    if not seen:
      return prior[j]

    k = min(seen, key=lambda k: t.distx(rows[j], rows[k]))
    similarity = 1 - t.distx(rows[j], rows[k])
    correction = similarity * (seen[k] - prior[k])

    return prior[j] + correction

  for _ in range(min(budget, len(rows))):
    j = min((j for j in range(len(rows)) if j not in seen),
            key=lambda j: (estimate(j), j))

    print(json.dumps({"inspect": j}), flush=True)
    reply = json.loads(input())

    if reply["row"] != j:
      raise ValueError("Unexpected oracle response")

    for y in t.cols.y:
      rows[j][y.at] = reply["goals"][y.name]

    seen[j] = t.disty(rows[j])

  selected = sorted(seen,
                    key=lambda j: (round(seen[j], 12), j))

  return {"selected": selected[:3]}


def fix(train, test):
  t = Tbl(csv_rows(train))
  good, _ = profiles(t)
  rows = csv_rows(test)

  if next(rows) != t.cols.names:
    raise ValueError("Train and test headers differ")

  names = ["Clndrs", "Volume"]
  ats = [t.cols.names.index(name) for name in names]
  mpg_at = t.cols.names.index("Mpg+")
  known = [t.disty(row) for row in t.rows]
  cutoff = sorted(known)[len(known) // 2]

  menu = list(dict.fromkeys(tuple(r[a] for a in ats)
                            for r in good.rows))[:12]

  def predict(row):
    return predict_row(t, row, known, mpg_at, stable=True)[1]

  out = []

  for j, row in enumerate(rows):
    before = predict(row)
    after = before
    changes = {}
    status = "already_good"

    if round(before, 10) > round(cutoff, 10):
      status = "no_candidate"
      best = None

      for values in menu:
        edits = {name: v for name, a, v in zip(names, ats, values)
                 if row[a] != v}

        if not edits:
          continue

        new = list(row)

        for a, v in zip(ats, values):
          new[a] = v

        score = predict(new)

        key = (len(edits), round(t.distx(row, new), 10),
               round(score, 10), values)

        if round(before - score, 10) >= 0.05:
          if best is None or key < best:
            best, after, changes = key, score, edits
            status = "improved"

    out.append({
      "row": j, "before": before, "after": after,
      "changes": changes, "status": status
    })

  return {"repairs": out}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("skill", choices=["remember", "guess", "certify", "flow", "blame", "justify", "choose", "fix"])
  p.add_argument("--train", required=True)
  p.add_argument("--test", required=True)
  args = p.parse_args()

  skills = {"remember": remember, "guess": guess,
            "certify": certify, "flow": flow, "blame": blame, "justify": justify, "choose": choose, "fix": fix}
  print(json.dumps(skills[args.skill](args.train, args.test)))


if __name__ == "__main__":
  main()
