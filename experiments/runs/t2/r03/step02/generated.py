import sys, csv, math, json, argparse

GOALS = {"Lbs": 0.0, "Acc": 1.0, "Mpg": 1.0}

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip().rstrip("+-") for h in next(r)]
        return [dict(zip(header, (c.strip() for c in row))) for row in r if row]

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def goals(row):
    g = [num(row.get(k)) for k in GOALS]
    return None if None in g else g

def make_score(train_goals):
    hs = list(GOALS.values())
    stats = []
    for j in range(3):
        col = [r[j] for r in train_goals]
        n = len(col)
        m = sum(col) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))

    def score(r):
        s = 0.0
        for j in range(3):
            m, sd = stats[j]
            z = max(-3.0, min(3.0, (r[j] - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - hs[j]) ** 2
        return math.sqrt(s / 3)
    return score

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            M[c][c] = 1e-12
        for i in range(n):
            if i != c:
                f = M[i][c] / M[c][c]
                for k in range(c, n + 1):
                    M[i][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]

def ridge(X, y, lam=1.0):
    d = len(X[0]) + 1
    X = [[1.0] + x for x in X]
    A = [[sum(x[i] * x[j] for x in X) + (lam if i == j and i > 0 else 0.0)
          for j in range(d)] for i in range(d)]
    b = [sum(x[i] * t for x, t in zip(X, y)) for i in range(d)]
    return solve(A, b)

def remember(train, test):
    score = make_score([g for g in map(goals, train) if g])
    scores = []
    for r in test:
        g = goals(r)
        scores.append(score(g) if g else float("inf"))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[len(order) // 2], "worst": order[-1]}

NUM = ["Clndrs", "Volume", "Model"]

def guess(train, test):
    rows = [(r, goals(r)) for r in train]
    rows = [(r, g) for r, g in rows if g]
    score = make_score([g for _, g in rows])
    ms, sds = {}, {}
    for k in NUM:
        c = [v for v in (num(r.get(k)) for r, _ in rows) if v is not None]
        m = sum(c) / len(c) if c else 0.0
        sd = math.sqrt(sum((v - m) ** 2 for v in c) / max(len(c) - 1, 1)) if c else 1.0
        ms[k], sds[k] = m, sd or 1.0
    origins = sorted({r.get("origin", "?") for r, _ in rows} - {"?", ""})

    def feat(r, vol):
        x = []
        for k in NUM:
            if k == "Volume" and not vol:
                continue
            v = num(r.get(k))
            x.append(((ms[k] if v is None else v) - ms[k]) / sds[k])
        x += [1.0 if r.get("origin") == o else 0.0 for o in origins]
        return x

    ys_mpg = [math.log(max(g[2], 1e-3)) for _, g in rows]
    ys_d = [score(g) for _, g in rows]
    models = {}
    for vol in (True, False):
        X = [feat(r, vol) for r, _ in rows]
        models[vol] = (ridge(X, ys_mpg), ridge(X, ys_d))

    def pred(w, x):
        return w[0] + sum(a * b for a, b in zip(w[1:], x))

    mpg_mean = sum(g[2] for _, g in rows) / len(rows)
    d_mean = sum(ys_d) / len(ys_d)
    mpg, sc = [], []
    for i, r in enumerate(test):
        vol = num(r.get("Volume")) is not None and i % 2 == 1
        x = feat(r, vol)
        wm, wd = models[vol]
        try:
            m = math.exp(max(-5.0, min(6.0, pred(wm, x))))
            d = pred(wd, x)
        except OverflowError:
            m, d = mpg_mean, d_mean
        if not (math.isfinite(m) and m > 0):
            m = mpg_mean
        if not math.isfinite(d):
            d = d_mean
        mpg.append(m)
        sc.append(max(0.0, min(1.0, d)))
    return {"mpg": mpg, "scores": sc}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    train, test = load(a.train), load(a.test)
    out = guess(train, test) if a.cmd == "guess" else remember(train, test)
    sys.stdout.write(json.dumps(out))

main()
