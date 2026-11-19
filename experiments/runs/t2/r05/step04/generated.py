import sys, csv, math, json
from bisect import bisect_left, bisect_right

GOALS = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def num(row, j):
    if j < 0 or j >= len(row):
        return None
    try:
        v = float(row[j])
        return v if math.isfinite(v) else None
    except ValueError:
        return None

def goal_stats(th, tr):
    stats = []
    for g, _ in GOALS:
        j = th.index(g)
        vals = [v for v in (num(r, j) for r in tr) if v is not None]
        n = len(vals)
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    return stats

def d2h(vals, stats):
    s = 0.0
    for (g, h), v, (m, sd) in zip(GOALS, vals, stats):
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        nv = 1 / (1 + math.exp(-1.702 * z))
        s += (nv - h) ** 2
    return math.sqrt(s / 3)

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]

def build(th, tr, ys, nums, lam=1.0):
    """Ridge on standardized numerics, their squares, origin one-hot. Returns predictor(header,row)."""
    oj = th.index("origin")
    cats = sorted({r[oj] for r in tr if oj < len(r)})
    stat = []
    for nm in nums:
        j = th.index(nm)
        v = [x for x in (num(r, j) for r in tr) if x is not None]
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)) or 1.0
        stat.append((mu, sd))

    def vec(h, row):
        z = []
        for nm, (mu, sd) in zip(nums, stat):
            x = num(row, h.index(nm) if nm in h else -1)
            z.append(0.0 if x is None else (x - mu) / sd)
        o = row[h.index("origin")] if "origin" in h and h.index("origin") < len(row) else None
        return z + [t * t for t in z] + [1.0 if o == c else 0.0 for c in cats] + [1.0]

    X = [vec(th, r) for r in tr]
    d = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) + (lam if i == j and i < d - 1 else 0.0)
          for j in range(d)] for i in range(d)]
    ws = []
    for y in ys:
        b = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(d)]
        ws.append(solve(A, b))
    return lambda h, row: [sum(a * c for a, c in zip(w, vec(h, row))) for w in ws]

def guess(train, test):
    th, tr = load(train)
    eh, te = load(test)
    stats = goal_stats(th, tr)
    gi = [th.index(g) for g, _ in GOALS]
    rows, lm, sc = [], [], []
    for r in tr:
        v = [num(r, j) for j in gi]
        if None in v or v[2] <= 0:
            continue
        rows.append(r)
        lm.append(math.log(v[2]))
        sc.append(d2h(v, stats))
    full = build(th, rows, [lm, sc], ["Clndrs", "Volume", "Model"])
    part = build(th, rows, [lm, sc], ["Clndrs", "Model"])
    vj = eh.index("Volume") if "Volume" in eh else -1
    mpg, scores = [], []
    for r in te:
        f = full if num(r, vj) is not None else part
        p = f(eh, r)
        m = math.exp(max(-5.0, min(8.0, p[0])))
        mpg.append(m if math.isfinite(m) and m > 0 else 1.0)
        s = p[1]
        scores.append(max(0.0, min(1.0, s)) if math.isfinite(s) else 0.5)
    return {"mpg": mpg, "scores": scores}

def remember(train, test):
    th, tr = load(train)
    eh, te = load(test)
    stats = goal_stats(th, tr)
    idx = [eh.index(g) for g, _ in GOALS]
    scores = [d2h([float(row[i]) for i in idx], stats) for row in te]
    order = sorted(range(len(scores)), key=lambda k: (scores[k], k))
    return {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }

def compare(xs, ys):
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((v - mx) ** 2 for v in xs) / (n - 1)
    vy = sum((v - my) ** 2 for v in ys) / (m - 1)
    pooled = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (math.sqrt(pooled) + 1e-32)
    sx, sy = sorted(xs), sorted(ys)
    tot = 0
    for x in xs:
        tot += bisect_left(sy, x) - (m - bisect_right(sy, x))
    cliffs = abs(tot) / (n * m)
    d = 0.0
    for v in set(sx) | set(sy):
        d = max(d, abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m))
    ks = d / math.sqrt(1 / n + 1 / m)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks,
            "same": bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)}

def certify(test):
    with open(test) as f:
        cases = json.load(f)
    return {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}

def flow(train, test):
    out = guess(train, test)
    th, tr = load(train)
    stats = goal_stats(th, tr)
    gi = [th.index(g) for g, _ in GOALS]
    hist = []
    for r in tr:
        v = [num(r, j) for j in gi]
        if None not in v:
            hist.append(d2h(v, stats))
    sc = out["scores"]
    alerts = []
    for i in range(len(sc)):
        if i < 19 or len(hist) < 2:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, sc[i - 19:i + 1])["same"])
    out["alerts"] = alerts
    return out

def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("remember", "guess", "certify", "flow"):
        sys.exit(1)
    train = test = None
    i = 1
    while i < len(a):
        if a[i] == "--train":
            train = a[i + 1]; i += 2
        elif a[i] == "--test":
            test = a[i + 1]; i += 2
        else:
            i += 1
    if a[0] == "certify":
        out = certify(test)
    elif a[0] == "remember":
        out = remember(train, test)
    elif a[0] == "flow":
        out = flow(train, test)
    else:
        out = guess(train, test)
    sys.stdout.write(json.dumps(out))

main()
