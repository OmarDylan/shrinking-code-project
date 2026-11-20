"""
EG5: active learning.

USAGE: python3 eg5.py [--key=val ...] [test ...]

OPTIONS:
  --file    data file                = ../data/auto93.csv
  --seed    random seed              = 1234567891
  --round   decimals shown           = 2
  --p       minkowski coefficient    = 2
  --budget  maximum labels           = 50
  --check   holdout test labels      = 5
  --keepf   pool fraction kept       = 0.66
  --leaf    minimum cut size         = 3
  --maxd    maximum tree depth       = 4
  --more    labels bought per round  = 4
"""
import csv, re, sys
sys.dont_write_bytecode = True
from math import floor, sqrt, exp
from types import SimpleNamespace as o
TINY = 1E-32
Seed = 1234567891

def srand(n=1234567891):
  global Seed
  Seed = floor(n) % 2147483647
  if Seed <= 0:
    Seed += 2147483646


def rand(lo=None, hi=None):
  global Seed
  Seed = (16807 * Seed) % 2147483647
  x = Seed / 2147483647
  if lo is None:
    return x
  if hi is None:
    lo, hi = 1, lo
  return lo + floor(
    x * (hi - lo + 1)
  )

def shuffle(rows):
  out = list(rows)
  for at in range(len(out), 1, -1):
    j = rand(at)
    out[at - 1], out[j - 1] = (
      out[j - 1],
      out[at - 1]
    )
  return out

def rounded(v, n=None):
  if not isinstance(v, float):
    return v
  if v.is_integer():
    return int(v)
  n = 10 ** (
    the.round
    if n is None
    else n
  )
  out = floor(v * n + 0.5) / n
  return (
    int(out)
    if out.is_integer()
    else out
  )

def show(x):
  if isinstance(x, list):
    return "{" + " ".join(
      show(v) for v in x
    ) + "}"
  if isinstance(x, float):
    return str(rounded(x))
  return str(x)

def thing(s):
  if (s[1:] if s[:1] == "-" else s).isdigit():
    return int(s)
  try:
    return float(s)
  except ValueError:
    return (
      s == "True"
      or (s != "False" and s)
    )

def settings(doc):
  pat = r"--(\w+)\s+[^=\n]*=\s*(\S+)"
  return o(**{
    k: thing(v)
    for k, v in re.findall(pat, doc)
  })

def adds(src, out=None):
  out = Num() if out is None else out
  for v in src:
    out.add(v)
  return out

class Num:
  def __init__(i, name="", at=0):
    i.at = at
    i.name = name
    i.n = 0
    i.mu = 0
    i.m2 = 0
    i.heaven = (
      0 if name.endswith("-") else 1
    )

  def add(i, v, inc=1):
    if v == "?":
      return v
    i.n += inc
    d = v - i.mu
    i.mu += inc * d / i.n
    i.m2 += (
      inc * d * (v - i.mu)
    )
    return v

  def mid(i):
    return i.mu

  def div(i):
    return (
      0
      if i.n < 2
      else sqrt(
        max(i.m2, 0)
        / (i.n - 1)
      )
    )

  def norm(i, v):
    if v == "?":
      return v
    z = (
      v - i.mu
    ) / (
      i.div() + TINY
    )
    z = max(
      -3,
      min(3, z)
    )
    return 1 / (
      1 + exp(-1.702 * z)
    )

  def dist(i, a, b):
    if a == "?" and b == "?":
      return 1
    a = i.norm(a)
    b = i.norm(b)
    if a == "?":
      a = 0 if b > 0.5 else 1
    if b == "?":
      b = 0 if a > 0.5 else 1
    return abs(a - b)

  def holds(i, x, v):
    return (
      x == "?"
      or x <= v
    )

  def __sub__(i, j):
    n = i.n - j.n
    if n < 1:
      return Num(
        i.name,
        i.at
      )
    d = j.mu - i.mu
    out = Num(
      i.name,
      i.at
    )
    out.heaven = i.heaven
    out.n = n
    out.mu = (
      i.n * i.mu
      - j.n * j.mu
    ) / n
    out.m2 = max(
      0,
      i.m2
      - j.m2
      - d * d * i.n * j.n / n
    )
    return out

  def cuts(i, xy, tot, acc, out):
    xy = sorted(
      xy,
      key=lambda pair: pair[0]
    )
    here = acc()
    for j, pair in enumerate(xy):
      x, y = pair
      here.add(y)
      if (
        j < len(xy) - 1
        and x != xy[j + 1][0]
        and big(j + 1, len(xy))
      ):
        out.append((
          val(
            here,
            tot - here
          ),
          i.at,
          x
        ))

class Sym:
  def __init__(i, name="", at=0):
    i.at = at
    i.name = name
    i.n = 0
    i.has = {}

  def add(i, v, inc=1):
    if v == "?":
      return v
    i.n += inc
    i.has[v] = (
      inc
      + i.has.get(v, 0)
    )
    if i.has[v] <= 0:
      del i.has[v]
    return v

  def dist(i, a, b):
    if a == "?" and b == "?":
      return 1
    return 1 if a != b else 0

  def holds(i, x, v):
    return (
      x == "?"
      or x == v
    )

  def cuts(i, xy, tot, acc, out):
    groups = {}
    for x, y in xy:
      if x not in groups:
        groups[x] = acc()
      groups[x].add(y)
    if len(groups) > 1:
      for key, summary in groups.items():
        if big(
          summary.n,
          len(xy)
        ):
          out.append((
            val(
              summary,
              tot - summary
            ),
            i.at,
            key
          ))

def Col(name, at):
  return (
    Sym(name, at)
    if name[0].islower()
    else Num(name, at)
  )

class Cols:
  def __init__(i, names):
    i.names = names
    i.all = []
    i.x = []
    i.y = []
    i.klass = None
    for at, name in enumerate(names):
      col = Col(
        name,
        at
      )
      i.all.append(col)
      if name.endswith("!"):
        i.klass = col
      elif name.endswith(("+", "-")):
        i.y.append(col)
      elif not name.endswith("X"):
        i.x.append(col)

  def add(i, row, inc=1):
    for col in i.all:
      col.add(
        row[col.at],
        inc
      )
    return row

def minkowski(cols, f):
  d, n = 0, TINY
  for col in cols:
    n += 1
    d += f(col) ** the.p
  return (
    d / n
  ) ** (
    1 / the.p
  )

def val(a, b):
  return (
    a.div() * a.n
    + b.div() * b.n
  ) / (
    a.n + b.n + TINY
  )

def big(lo, n):
  return (
    the.leaf
    <= lo
    <= n - the.leaf
  )

class Tbl:
  def __init__(i, src):
    src = iter(src)
    i.rows = []
    i.cols = Cols(
      next(src)
    )
    i.model = None
    for row in src:
      i.add(row)

  def add(i, row):
    i.rows.append(
      i.cols.add(row)
    )
    return row

  def clone(i, rows=None):
    rows = (
      []
      if rows is None
      else rows
    )
    out = Tbl(
      [i.cols.names]
      + list(rows)
    )
    out.model = i.model
    return out

  def distx(i, row1, row2):
    return minkowski(
      i.cols.x,
      lambda col:
        col.dist(
          row1[col.at],
          row2[col.at]
        )
    )

  def label(i, row):
    xs = [
      row[col.at]
      for col in i.cols.x
    ]
    goals = i.model(xs)
    for j, y in enumerate(i.cols.y):
      row[y.at] = y.add(
        goals[j]
      )
    return row

  def disty(i, row):
    if (
      i.model
      and row[
        i.cols.y[0].at
      ] == "?"
    ):
      i.label(row)
    return minkowski(
      i.cols.y,
      lambda y:
        abs(
          y.norm(
            row[y.at]
          )
          - y.heaven
        )
    )

  def Y(i):
    return lambda row: i.disty(row)

  def poles(i, rows, lo=None, hi=None):
    def far(row):
      return sorted(
        rows,
        key=lambda other:
          i.distx(
            other,
            row
          )
      )[-1]
    if lo is None:
      lo = far(
        rows[
          rand(len(rows)) - 1
        ]
      )
    if hi is None:
      hi = far(lo)
    if i.disty(lo) > i.disty(hi):
      lo, hi = hi, lo
    c = (
      i.distx(lo, hi)
      + TINY
    )

    def project(row):
      return (
        i.distx(lo, row) ** 2
        + c * c
        - i.distx(hi, row) ** 2
      ) / (
        2 * c
      )
    return (
      project,
      lo,
      hi
    )

  def acquire(
    i,
    rows,
    cap,
    lab,
    lo=None,
    hi=None
  ):
    seen = {
      id(row)
      for row in lab
    }
    while (
      len(rows)
      >= 2 * the.leaf
    ):
      more = min(
        the.more,
        cap - len(lab)
      )
      new = []
      for row in rows:
        key = id(row)
        if key in seen:
          new.append(row)
        elif more > 0:
          more -= 1
          seen.add(key)
          lab.append(row)
          new.append(row)
      if len(lab) >= cap:
        return lab
      project, _, _ = i.poles(
        new,
        lo,
        hi
      )
      rows = sorted(
        rows,
        key=project
      )
      keep = max(
        1,
        floor(
          the.keepf
          * len(rows)
        )
      )
      rows = rows[:keep]
    return lab

  def acquirer(i, cap):
    lab = []
    lo = None
    hi = None
    while True:
      before = len(lab)
      lab = i.acquire(
        shuffle(i.rows),
        cap,
        lab,
        lo,
        hi
      )
      if (
        len(lab) >= cap
        or len(lab) >= len(i.rows)
        or len(lab) == before
      ):
        break
      ranked = sorted(
        lab,
        key=i.Y()
      )
      lo = ranked[0]
      hi = ranked[-1]
    return sorted(
      lab,
      key=i.Y()
    )

  def cuts(i, rows, col, Y, acc, out):
    xy = []
    for row in rows:
      if row[col.at] != "?":
        xy.append((
          row[col.at],
          Y(row)
        ))
    tot = adds(
      (
        pair[1]
        for pair in xy
      ),
      acc()
    )
    col.cuts(
      xy,
      tot,
      acc,
      out
    )

  def bestcut(
    i,
    rows,
    Y=None,
    acc=Num
  ):
    Y = (
      i.Y()
      if Y is None
      else Y
    )
    out = []
    for col in i.cols.x:
      i.cuts(
        rows,
        col,
        Y,
        acc,
        out
      )
    return min(
      out,
      key=lambda candidate:
        candidate[0],
      default=None
    )

  def divide(i, rows, col, cut):
    yes = []
    no = []
    for row in rows:
      if col.holds(
        row[col.at],
        cut
      ):
        yes.append(row)
      else:
        no.append(row)
    return yes, no

  def holdout(i, how=None):
    if how is None:
      how = (
        lambda table, cap:
          table.acquirer(cap)
      )
    rows = shuffle(
      i.rows
    )
    n = floor(
      len(rows) / 2
    )
    train = rows[:n]
    test = rows[n:]
    lab = how(
      i.clone(train),
      the.budget
      - the.check
    )
    assert (
      len(lab)
      + the.check
      <= the.budget
    )
    tree = Tree(
      i,
      lab
    )
    top = sorted(
      test,
      key=lambda row:
        tree.leaf(i, row)
    )[:the.check]
    return sorted(
      top,
      key=i.Y()
    )[0]

class Tree:
  def __init__(
    i,
    tbl,
    rows=None,
    Y=None,
    acc=Num,
    lvl=0
  ):
    rows = (
      tbl.rows
      if rows is None
      else rows
    )
    Y = (
      tbl.Y()
      if Y is None
      else Y
    )
    ys = adds(
      (
        Y(row)
        for row in rows
      ),
      acc()
    )
    i.at = None
    i.v = None
    i.yes = None
    i.no = None
    i.mu = ys.mid()
    if (
      len(rows)
      >= 2 * the.leaf
      and lvl < the.maxd
    ):
      best = tbl.bestcut(
        rows,
        Y,
        acc
      )
      if best is not None:
        _, at, cut = best
        col = tbl.cols.all[at]
        yes, no = tbl.divide(
          rows,
          col,
          cut
        )
        if yes and no:
          i.at = at
          i.v = cut
          i.yes = Tree(
            tbl,
            yes,
            Y,
            acc,
            lvl + 1
          )
          i.no = Tree(
            tbl,
            no,
            Y,
            acc,
            lvl + 1
          )

  def leaf(i, tbl, row):
    node = i
    while node.at is not None:
      col = tbl.cols.all[
        node.at
      ]
      if col.holds(
        row[node.at],
        node.v
      ):
        node = node.yes
      else:
        node = node.no
    return node.mu

def Pool():
  rows = [[
    "X1",
    "X2",
    "X3",
    "F1-",
    "F2-"
  ]]
  for _ in range(200):
    rows.append([
      rand(),
      rand(),
      rand(),
      "?",
      "?"
    ])
  t = Tbl(rows)
  def model(xs):
    a = (
      (xs[0] - 0.2) ** 2
      + (xs[1] - 0.8) ** 2
    )
    b = (
      (xs[0] - 0.9) ** 2
      + (xs[2] - 0.5) ** 2
    )
    return [a, b]
  t.model = model
  return t

def csv_rows(file):
  with open(
    file,
    newline="",
    encoding="utf-8-sig"
  ) as src:
    for row in csv.reader(src):
      if row:
        yield [
          thing(x.strip())
          for x in row
        ]

def test_label():
  t = Pool()
  y1 = t.cols.y[0]
  row = t.rows[0]
  assert (
    row[y1.at]
    == "?"
  )
  assert y1.n == 0
  t.disty(row)
  assert (
    row[y1.at]
    != "?"
  )
  assert y1.n == 1
  t.disty(row)
  print(
    "row 1 goals "
    f"{show(row[3:])}; "
    "labels bought so far "
    f"{y1.n}"
  )
  assert y1.n == 1

def test_acquire():
  t = Tbl(
    csv_rows(the.file)
  )
  lab = t.acquirer(
    the.budget
    - the.check
  )
  all_scores = sorted(
    t.Y()(row)
    for row in t.rows
  )
  best = t.disty(
    lab[0]
  )
  mid = all_scores[
    len(all_scores) // 2
  ]
  print(
    f"{len(lab)} labels of "
    f"{len(t.rows)} rows: "
    f"best {best:.3f}, "
    f"median row {mid:.3f}, "
    f"pool best {all_scores[0]:.3f}"
  )
  assert (
    len(lab)
    <= the.budget
    - the.check
  )
  assert best < mid
  assert (
    t.disty(lab[0])
    <= t.disty(lab[-1])
  )

def test_holdout():
  t = Tbl(
    csv_rows(the.file)
  )
  best = t.holdout()
  all_scores = sorted(
    t.Y()(row)
    for row in t.rows
  )
  mid = all_scores[
    len(all_scores) // 2
  ]
  print(
    f"holdout winner "
    f"{t.disty(best):.3f}, "
    f"median row {mid:.3f}, "
    f"pool best {all_scores[0]:.3f}"
  )
  assert (
    t.disty(best)
    < mid
  )

def test_holdouts():
  t = Tbl(
    csv_rows(the.file)
  )
  all_scores = sorted(
    t.Y()(row)
    for row in t.rows
  )
  mid = all_scores[
    len(all_scores) // 2
  ]
  ds = [
    t.disty(
      t.holdout()
    )
    for _ in range(20)
  ]
  ds = sorted(ds)
  wins = sum(
    d < mid
    for d in ds
  )
  print(
    "20 holdouts: "
    f"lo {ds[0]:.3f}, "
    f"mid {ds[len(ds) // 2]:.3f}, "
    f"hi {ds[-1]:.3f}; "
    f"beat the median row "
    f"{wins}/20"
  )
  assert (
    ds[len(ds) // 2]
    < mid
  )
  assert wins >= 18

def test_all():
  tests = [
    test_acquire,
    test_holdout,
    test_holdouts,
    test_label
  ]
  for test in tests:
    srand(the.seed)
    test()
  print("failures: 0")

def main(funs):
  if "-h" in sys.argv:
    return print(__doc__)
  for arg in sys.argv[1:]:
    if (
      arg.startswith("--")
      and "=" in arg
    ):
      key, value = arg[2:].split(
        "=",
        1
      )
      if key in vars(the):
        setattr(
          the,
          key,
          thing(value)
        )
  for arg in sys.argv[1:]:
    if "=" in arg:
      continue
    name = (
      "test_"
      + arg.lstrip("-")
    )
    if name in funs:
      srand(the.seed)
      funs[name]()
the = settings(__doc__)
if __name__ == "__main__":
  main(globals())

# CHECK YOUR PORT
# 1. Replacing min(the.more, cap - len(lab)) with the.more caused acquire() to buy 48 labels instead of the allowed 45. The budget assertion failed:len(lab) <= the.budget - the.check Even though the best score was still 0.087, that result is not meaningful because it exceeded the allowed label budget. If the price of a result is unknown or ignored, we cannot fairly compare it with another active learner under the same budget.
# 2. Over seeds 1..5, keepf=0.5 and more=2 had an average best score of 0.0818, while the defaults keepf=0.66 and more=4 averaged 0.0898. Since lower d2h is better, the week-3 strategy was better by about 0.008. Both had median 0.087, so the difference was small and probably not worth much extra explanation or complexity based on these five runs alone.
# 3. Changing the restart anchors from the best and worst labelled
#    rows to the best and second-best made my best result worse: 0.087 -> 0.113. The projection uses the line between lo and hi as its axis. Best+worst are far apart and give a useful long direction through the data. Best+second-best are usually close, so c=dist(lo,hi) is small and the projection does a poor job spreading/ranking the remaining rows. The next descent then wastes labels exploring nearly the same good region instead of contrasting good vs bad.
# 4. holdout checks that training labels plus the final test checks stay within the total budget. One way to cheat is to buy extra training labels, giving the learner more information than competing methods. In a paper this could appear as better results without reporting equal label costs. Another way is to peek at too many test labels and then report only the best candidate. In a paper this could look like repeated tuning or selection on the holdout/test set while pretending the test set remained unseen.