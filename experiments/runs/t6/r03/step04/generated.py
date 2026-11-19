import sys, json, csv, argparse
from math import exp, sqrt, log
from bisect import bisect_left, bisect_right


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if any(c.strip() for c in r)]
    return rows[0], rows[1:]


def goal_stats(names, tr):
    goals = [(i, n[-1]) for i, n in enumerate(names) if n[-1] in "+-"]
    stats = {}
    for i, _ in goals:
        vs = [float(r[i]) for r in tr if r[i] != "?"]
        mu = sum(vs) / len(vs)
        sd = sqrt(sum((v - mu) ** 2 for v in vs) / (len(vs) - 1)) if len(vs) > 1 else 0.0
        stats[i] = (mu, sd)
    return goals, stats


def d2h(r, goals, stats):
    s = 0.0
    for i, sign in goals:
        mu, sd = stats[i]
        z = max(-3.0, min(3.0, (float(r[i]) - mu) / (sd + 1e-32)))
        nv = 1 / (1 + exp(-1.702 * z))
        h = 1.0 if sign == "+" else 0.0
        s += (nv - h) ** 2
    return sqrt(s / len(goals))


def remember(train, test):
    names, tr = load(train)
    _, te = load(test)
    goals, stats = goal_stats(names, tr)
    scores = [d2h(r, goals, stats) for r in te]
    order = sorted(range(len(te)), key=lambda i: (scores[i], i))
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[len(te) // 2], "worst": order[-1]}


def num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c] or 1e-12
        for r in range(n):
            if r != c:
                f = M[r][c] / d
                if f:
                    M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / (M[i][i] or 1e-12) for i in range(n)]


def ridge(X, y, lam=1.0):
    "Standardized ridge regression; returns a predictor."
    n, k = len(X), len(X[0])
    mu = [sum(r[j] for r in X) / n for j in range(k)]
    sd = [sqrt(sum((r[j] - mu[j]) ** 2 for r in X) / n) or 1.0 for j in range(k)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(k)] for r in X]
    ym = sum(y) / n
    A = [[sum(z[i] * z[j] for z in Z) + (lam if i == j else 0.0) for j in range(k)] for i in range(k)]
    b = [sum(z[i] * (t - ym) for z, t in zip(Z, y)) for i in range(k)]
    w = solve(A, b)
    return lambda x: ym + sum(wj * (xj - m) / s for wj, xj, m, s in zip(w, x, mu, sd))


def guess(train, test):
    names, tr = load(train)
    tn, te = load(test)
    goals, stats = goal_stats(names, tr)
    ix = {n: i for i, n in enumerate(names)}
    tix = {n: i for i, n in enumerate(tn)}
    mi = [i for i, n in enumerate(names) if n == "Mpg+"][0]

    def cell(r, n, idx):
        j = idx.get(n)
        return r[j] if j is not None and j < len(r) else "?"

    nums = ["Clndrs", "Volume", "Model"]
    mean = {}
    for n in nums:
        vs = [v for v in (num(cell(r, n, ix)) for r in tr) if v is not None]
        mean[n] = sum(vs) / len(vs) if vs else 0.0
    levels = sorted({cell(r, "origin", ix) for r in tr} - {"?"})

    def feats(r, idx, vol):
        g = {}
        for n in nums:
            v = num(cell(r, n, idx))
            g[n] = mean[n] if v is None else v
        f = [g["Clndrs"], g["Model"]]
        if vol:
            f += [g["Volume"], log(max(g["Volume"], 1.0))]
        o = cell(r, "origin", idx)
        f += [1.0 if o == l else 0.0 for l in levels]
        return f

    good = [r for r in tr if all(num(r[i]) is not None for i, _ in goals)]
    ys_d = [d2h(r, goals, stats) for r in good]
    ys_m = [log(max(float(r[mi]), 1e-6)) for r in good]
    models = {}
    for vol in (True, False):
        X = [feats(r, ix, vol) for r in good]
        models[vol] = (ridge(X, ys_d, 1.0), ridge(X, ys_m, 1.0))
    lo_m, hi_m = min(ys_m), max(ys_m)
    mpg, scores = [], []
    for r in te:
        vol = num(cell(r, "Volume", tix)) is not None
        x = feats(r, tix, vol)
        pd, pm = models[vol]
        scores.append(min(1.0, max(0.0, pd(x))))
        mpg.append(exp(min(hi_m + 0.3, max(lo_m - 0.3, pm(x)))))
    return {"mpg": mpg, "scores": scores}


def compare(xs, ys):
    "Cohen, Cliffs, KS and the same verdict for two samples."
    xs, ys = sorted(float(v) for v in xs), sorted(float(v) for v in ys)
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((v - mx) ** 2 for v in xs) / (n - 1)
    vy = sum((v - my) ** 2 for v in ys) / (m - 1)
    pool = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (sqrt(pool) + 1e-32)
    s = 0
    for x in xs:
        s += bisect_left(ys, x) - (m - bisect_right(ys, x))
    cliffs = abs(s) / (n * m)
    d = max(abs(bisect_right(xs, v) / n - bisect_right(ys, v) / m)
            for v in set(xs) | set(ys))
    ks = d / sqrt(1 / n + 1 / m)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks,
            "same": bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)}


def certify(train, test):
    with open(test, encoding="utf-8-sig") as f:
        cases = json.load(f)
    return {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}


def flow(train, test, win=20):
    "Per-arrival predictions; alert when recent predicted d2h drifts from training history."
    out = guess(train, test)
    names, tr = load(train)
    goals, stats = goal_stats(names, tr)
    hist = [d2h(r, goals, stats) for r in tr
            if all(num(r[i]) is not None for i, _ in goals)]
    sc = out["scores"]
    alerts = []
    for i in range(len(sc)):
        if i < win - 1 or len(hist) < 2:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, sc[i - win + 1:i + 1])["same"])
    return {"mpg": out["mpg"], "scores": sc, "alerts": alerts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["remember", "guess", "certify", "flow"])
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    f = {"remember": remember, "guess": guess, "certify": certify, "flow": flow}[a.cmd]
    sys.stdout.write(json.dumps(f(a.train, a.test)))


if __name__ == "__main__":
    main()
