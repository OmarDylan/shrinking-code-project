import sys, csv, math, json
from bisect import bisect_left, bisect_right

GOALS = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows


def args():
    a = sys.argv[1:]
    d = {}
    for i, x in enumerate(a):
        if x in ("--train", "--test") and i + 1 < len(a):
            d[x[2:]] = a[i + 1]
    return a[0], d


def goal_cols(h):
    return [h.index(g) for g, _ in GOALS]


def goal_vals(rows, idx, skip_missing):
    out = []
    for row in rows:
        v = [row[i] for i in idx]
        if skip_missing and "?" in v:
            continue
        out.append([float(x) for x in v])
    return out


def goal_stats(tr):
    n = len(tr)
    stats = []
    for j in range(3):
        col = [r[j] for r in tr]
        m = sum(col) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in col) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    return stats


def d2h(r, stats):
    s = 0.0
    for j in range(3):
        m, sd = stats[j]
        z = max(-3.0, min(3.0, (r[j] - m) / (sd + 1e-32)))
        nv = 1 / (1 + math.exp(-1.702 * z))
        s += (nv - GOALS[j][1]) ** 2
    return math.sqrt(s / 3)


def remember(d):
    h, rows = load(d["train"])
    stats = goal_stats(goal_vals(rows, goal_cols(h), True))
    h2, rows2 = load(d["test"])
    te = goal_vals(rows2, goal_cols(h2), False)
    scores = [d2h(r, stats) for r in te]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }))


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except ValueError:
        return float("nan")


def predict(d):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    h, rows = load(d["train"])
    h2, rows2 = load(d["test"])
    gi = goal_cols(h)
    stats = goal_stats(goal_vals(rows, gi, True))
    cats = sorted({r[h.index("origin")] for r in rows} - {"?"})

    def feats(hh, rs):
        ci, vi, mi, oi = (hh.index(c) for c in ("Clndrs", "Volume", "Model", "origin"))
        return np.array([[num(r[ci]), num(r[vi]), num(r[mi])] +
                         [1.0 if r[oi] == c else 0.0 for c in cats] for r in rs])

    Xtr, Xte = feats(h, rows), feats(h2, rows2)
    mpg = np.array([num(r[gi[2]]) for r in rows])
    dh = np.array([d2h([num(r[i]) for i in gi], stats)
                   if all(math.isfinite(num(r[i])) for i in gi) else float("nan")
                   for r in rows])

    def fit_pred(y, cols, rowsel):
        ok = np.isfinite(y)
        if ok.sum() < 2:
            return np.full(int(rowsel.sum()), float(np.nanmean(y)) if ok.any() else 0.5)
        m = make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestRegressor(300, min_samples_leaf=2, random_state=0))
        m.fit(Xtr[ok][:, cols], y[ok])
        return m.predict(Xte[rowsel][:, cols])

    n = len(rows2)
    allc = list(range(Xtr.shape[1]))
    noVol = [c for c in allc if c != 1]
    miss = np.array([not math.isfinite(Xte[i, 1]) for i in range(n)], dtype=bool)
    out_m = np.zeros(n)
    out_d = np.zeros(n)
    for sel, cols in ((~miss, allc), (miss, noVol)):
        if sel.any():
            out_m[sel] = fit_pred(mpg, cols, sel)
            out_d[sel] = fit_pred(dh, cols, sel)
    mp = [max(float(x), 1e-3) for x in out_m]
    sc = [min(1.0, max(0.0, float(x))) for x in out_d]
    return mp, sc


def guess(d):
    mp, sc = predict(d)
    print(json.dumps({"mpg": mp, "scores": sc}))


def compare(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n, m = len(xs), len(ys)
    mx, my = math.fsum(xs) / n, math.fsum(ys) / m
    vx = math.fsum((x - mx) ** 2 for x in xs) / (n - 1)
    vy = math.fsum((y - my) ** 2 for y in ys) / (m - 1)
    pooled = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (math.sqrt(pooled) + 1e-32)
    sx, sy = sorted(xs), sorted(ys)
    gt = sum(bisect_left(sy, x) for x in xs)
    lt = sum(m - bisect_right(sy, x) for x in xs)
    cliffs = abs(gt - lt) / (n * m)
    ks = 0.0
    for v in set(xs) | set(ys):
        ks = max(ks, abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m))
    ks /= math.sqrt(1 / n + 1 / m)
    same = cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks, "same": bool(same)}


def certify(d):
    with open(d["test"]) as f:
        cases = json.load(f)
    print(json.dumps({"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}))


def flow(d):
    h, rows = load(d["train"])
    gv = goal_vals(rows, goal_cols(h), True)
    stats = goal_stats(gv)
    hist = [d2h(r, stats) for r in gv]
    mp, sc = predict(d)
    alerts = []
    for i in range(len(sc)):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, sc[i - 19:i + 1])["same"])
    print(json.dumps({"mpg": mp, "scores": sc, "alerts": alerts}))


def main():
    if not sys.argv[1:]:
        sys.exit(1)
    cmd, d = args()
    if cmd == "remember":
        remember(d)
    elif cmd == "guess":
        guess(d)
    elif cmd == "certify":
        certify(d)
    elif cmd == "flow":
        flow(d)
    else:
        sys.exit(1)


main()
