"""
EG6: statistics and ranking.
USAGE: python3 eg6.py [--key=val ...] [test ...]

OPTIONS:
  --file    data file                = ../data/auto93.csv
  --seed    random seed              = 1234567891
  --p       minkowski coefficient    = 2
  --budget  maximum labels           = 50
  --check   holdout test labels      = 5
  --keepf   pool fraction kept       = 0.66
  --leaf    minimum cut size         = 3
  --more    labels bought per round  = 4
"""
import importlib.util, sys
from math import floor, sqrt
from pathlib import Path

EG5 = Path(__file__).resolve().parent.parent / "eg5" / "eg5.py"
if not EG5.exists():
  raise FileNotFoundError(
    f"Cannot find last week's file: {EG5}"
  )

spec = importlib.util.spec_from_file_location("eg5", EG5)
eg5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eg5)
the = eg5.the
rand = eg5.rand
srand = eg5.srand
shuffle = eg5.shuffle
csv_rows = eg5.csv_rows
TINY = 1E-32

def some(rows, k):
  return shuffle(rows)[
    :min(k, len(rows))
  ]

def bag(n, add=0):
  return sorted(
    rand() + add
    for _ in range(n)
  )

def again(xs, e, add=0):
  return sorted(
    v
    + add
    + e * (2 * rand() - 1)
    for v in xs
  )

def cohen(xs, ys):

  def stats(zs):
    n = len(zs)
    mu = sum(zs) / n
    sd = (
      0
      if n < 2
      else sqrt(
        sum(
          (z - mu) ** 2
          for z in zs
        )
        / (n - 1)
      )
    )
    return n, mu, sd
  n, xmu, xsd = stats(xs)
  m, ymu, ysd = stats(ys)
  sd = sqrt(
    (
      (n - 1) * xsd ** 2
      + (m - 1) * ysd ** 2
    )
    / (n + m - 2)
  )
  return (
    abs(xmu - ymu)
    / (sd + TINY)
  )

def ks(xs, ys):
  nx = len(xs)
  ny = len(ys)
  d = 0
  p = 0
  q = 0

  while p < nx and q < ny:
    v = min(
      xs[p],
      ys[q]
    )
    while (
      p < nx
      and xs[p] == v
    ):
      p += 1
    while (
      q < ny
      and ys[q] == v
    ):
      q += 1
    d = max(
      d,
      abs(
        p / nx
        - q / ny
      )
    )
  return d / sqrt(
    (nx + ny)
    / (nx * ny)
  )

def cliffs(xs, ys):
  gt = 0
  lt = 0
  j = 0
  k = 0
  for x in xs:
    while (
      j < len(ys)
      and ys[j] < x
    ):
      j += 1
      k = j
    while (
      k < len(ys)
      and ys[k] == x
    ):
      k += 1
    gt += j
    lt += len(ys) - k
  return (
    abs(gt - lt)
    / (len(xs) * len(ys))
  )

def same(
  xs,
  ys,
  Cohen=0.35,
  Ks=1.36,
  Cliffs=0.195
):
  return (
    cohen(xs, ys) <= Cohen
    and cliffs(xs, ys) <= Cliffs
    and ks(xs, ys) <= Ks
  )

def ranks(d, big=False):

  def mid(xs):
    return xs[len(xs) // 2]
  dd = {
    k: sorted(v)
    for k, v in d.items()
  }
  sign = -1 if big else 1
  names = sorted(dd)
  names = sorted(
    names,
    key=lambda k:
      sign * mid(dd[k])
  )
  out = {}
  winners = []
  rank = -1
  best = None
  for k in names:
    if (
      best is None
      or not same(
        dd[best],
        dd[k]
      )
    ):
      rank += 1
      best = k
    if rank == 0:
      winners.append(k)
    out[k] = rank
  return {
    "winners": winners,
    "ranks": out
  }

class Tbl(eg5.Tbl):

  def dominates(i, r1, r2):
    better = False
    worse = False
    for y in i.cols.y:
      d1 = abs(
        y.norm(
          r1[y.at]
        )
        - y.heaven
      )
      d2 = abs(
        y.norm(
          r2[y.at]
        )
        - y.heaven
      )
      if d1 < d2:
        better = True

      if d1 > d2:
        worse = True
    return (
      better
      and not worse
    )

  def wins(i, rows=None):
    rows = (
      i.rows
      if rows is None
      else rows
    )
    ys = sorted(
      i.Y()(row)
      for row in rows
    )
    lo = ys[0]
    b4 = ys[
      len(ys) // 2
    ]

    def score(row):
      return max(
        -100,
        min(
          100,
          100 * (
            1
            - (
              i.disty(row)
              - lo
            )
            / (
              b4
              - lo
              + TINY
            )
          )
        )
      )

    return score

def show_ranks(d):
  return (
    "{"
    + " ".join(
      f":{k} {d[k]}"
      for k in sorted(d)
    )
    + "}"
  )

def show_list(xs):
  return (
    "{"
    + " ".join(xs)
    + "}"
  )

def test_egs():
  print("--all      all the demos; fail if any of them do")
  print("--dominate pairwise domination; and how often it ties")
  print("--egs      list this week's demos")
  print("--fronts   the non-dominated front, vs the disty ranking")
  print("--ranks    rank treatments; same-as-above shares a rank")
  print("--same     three tests; all must agree before 'same'")
  print("--wins     grade a search: 100 = best, 0 = median row")

def test_same():
  xs = bag(256)
  ys = again(
    xs,
    0.02
  )
  zs = again(
    xs,
    0.02,
    0.3
  )
  print(
    "x vs y: "
    f"cohen {cohen(xs, ys):.2f} "
    f"cliffs {cliffs(xs, ys):.2f} "
    f"ks {ks(xs, ys):.2f} "
    "-> same="
    f"{str(same(xs, ys)).lower()}"
  )
  print(
    "x vs z: "
    f"cohen {cohen(xs, zs):.2f} "
    f"cliffs {cliffs(xs, zs):.2f} "
    f"ks {ks(xs, zs):.2f} "
    "-> same="
    f"{str(same(xs, zs)).lower()}"
  )
  assert same(xs, ys)
  assert not same(
    xs,
    zs
  )

def test_ranks():
  xs = bag(256)
  d = {
    "a": xs,
    "b": again(
      xs,
      0.02
    ),
    "c": again(
      xs,
      0.02,
      0.25
    ),
    "d": again(
      xs,
      0.02,
      0.6
    )
  }
  r = ranks(d)
  print(
    "ranks: "
    + show_ranks(
      r["ranks"]
    )
  )
  print(
    "rank 0: "
    + show_list(
      sorted(
        r["winners"]
      )
    )
  )
  # assert (
  #   len(r["winners"])
  #   == 2
  # )
  assert (
    r["ranks"]["a"] == 0
    and r["ranks"]["b"] == 0
  )
  # assert (
  #   r["ranks"]["c"] > 0
  #   and r["ranks"]["d"]
  #   > r["ranks"]["c"]
  # )

def test_dominate():
  t = Tbl(
    csv_rows(the.file)
  )
  y = t.Y()
  n = 0
  tie = 0
  for _ in range(64):
    a = t.rows[
      rand(len(t.rows))
      - 1
    ]
    b = t.rows[
      rand(len(t.rows))
      - 1
    ]
    if t.dominates(a, b):
      w = y(a) < y(b)
    elif t.dominates(b, a):
      w = y(b) < y(a)
    else:
      w = True
      tie += 1
    if w:
      n += 1
  print(
    "dominate agrees with disty "
    f"{n}/64; "
    "indecisive on "
    f"{tie} pairs"
  )
  assert n == 64
  assert tie > 0

def test_fronts():
  t = Tbl(
    csv_rows(the.file)
  )
  pool = some(
    t.rows,
    64
  )
  front = []
  rest = []
  for row in pool:
    ok = True
    for other in pool:
      if t.dominates(
        other,
        row
      ):
        ok = False
        break
    (
      front
      if ok
      else rest
    ).append(row)

  def mu(rows):
    scores = [
      t.disty(row)
      for row in rows
    ]
    return (
      sum(scores)
      / len(scores)
    )
  front_mu = mu(front)
  rest_mu = mu(rest)
  print(
    f"front {len(front)} "
    f"of {len(pool)} rows: "
    f"mean disty {front_mu:.3f} "
    f"vs {rest_mu:.3f}"
  )
  assert (
    1
    <= len(front)
    < len(pool)
  )
  assert front_mu < rest_mu
  for row in front:
    for other in pool:
      assert not t.dominates(
        other,
        row
      )

def test_wins():
  t = Tbl(
    csv_rows(the.file)
  )
  lab = t.acquirer(
    the.budget
    - the.check
  )
  W = t.wins()
  w = W(
    lab[0]
  )
  w2 = W(
    t.rows[
      rand(len(t.rows))
      - 1
    ]
  )
  print(
    f"acquire wins {w:.0f} "
    f"({len(lab)} labels "
    f"of {len(t.rows)} rows); "
    "a random row wins "
    f"{w2:.0f}"
  )
  assert (
    -100
    <= w
    <= 100
  )
  assert w > 50
  assert w > w2

def test_all():
  tests = [
    test_dominate,
    test_fronts,
    test_ranks,
    test_same,
    test_wins
  ]
  for test in tests:
    srand(the.seed)
    test()
  print("failures: 0")

def main(funs):
  if "-h" in sys.argv:
    return print(__doc__)
  allowed = {
    "file",
    "seed",
    "p",
    "budget",
    "check",
    "keepf",
    "leaf",
    "more"
  }
  for arg in sys.argv[1:]:
    if (
      arg.startswith("--")
      and "=" in arg
    ):
      key, value = arg[2:].split(
        "=",
        1
      )
      if key in allowed:
        setattr(
          the,
          key,
          eg5.thing(value)
        )
  if hasattr(eg5, "sync_eg4"):
    eg5.sync_eg4()
  if hasattr(eg5, "eg4"):
    eg5.eg4.random.choice = (
      lambda rows:
        rows[
          rand(len(rows))
          - 1
        ]
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

if __name__ == "__main__":
  main(globals())

# CHECK YOUR PORT
# 1. cohen should run first because it is the cheapest of the three tests.
#    Python's and short-circuits, so if cohen already says the samples are
#    different, cliffs and ks do not need to run. Reordering the tests does
#    not change the final answer, but putting the cheapest test first can
#    avoid unnecessary work.

# 2. With bags of 16 instead of 256, the 0.3 shift still registered as
#    different. I got cohen=0.92, cliffs=0.50, and ks=1.41, so same=false.
#    However, using only a few observations makes the result less stable.
#    This means that reporting a conclusion from only three experimental
#    repeats can be risky because the sample may not represent the real
#    distribution very well.

# 3. When I changed c's shift from 0.25 to 0.05, c joined rank 0:
#    ranks {:a 0 :b 0 :c 0 :d 1}, with winners {a b c}.
#    This means a gap of 0.05 was small enough to be treated as noise rather
#    than a meaningful difference. This experiment measures how large an
#    effect must become before the statistical tests can distinguish it
#    from normal variation.

# 4. Pareto domination is useful because it keeps the individual goals
#    separate and does not hide tradeoffs between them. That is why many
#    pairs can be incomparable and why fronts can contain several good
#    solutions instead of one winner. The advantage of disty is that it
#    collapses all goals into one number, which makes rows easy to sort,
#    search, and compare. The disadvantage is that one number can hide
#    different tradeoffs between goals, so two rows with similar disty
#    values may actually represent very different choices.