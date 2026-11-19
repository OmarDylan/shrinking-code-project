import csv, sys, json, math, argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
COHEN_T, CLIFFS_T, KS_T = 0.35, 0.197, 1.36
WINDOW = 20


def load(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    head = [h.strip() for h in rows[0]]
    return head, [[c.strip() for c in r] for r in rows[1:]]


def num(s):
    if s == "?" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def column(head, rows, name):
    if name not in head:
        return [None] * len(rows)
    i = head.index(name)
    return [num(r[i]) if i < len(r) else None for r in rows]


def sym(head, rows, name):
    if name not in head:
        return ["?"] * len(rows)
    i = head.index(name)
    return [r[i] if i < len(r) else "?" for r in rows]


def goal_stats(th, tr):
    stats = {}
    for g in GOALS:
        v = [x for x in column(th, tr, g) if x is not None]
        n = len(v)
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        stats[g] = (m, sd)
    return stats


def norm(x, g, stats):
    if x is None:
        return 0.5
    m, sd = stats[g]
    z = max(-3.0, min(3.0, (x - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))


def d2h(vals, stats):
    s = sum((norm(vals[g], g, stats) - h) ** 2 for g, h in GOALS.items())
    return math.sqrt(s / 3)


def remember(a):
    th, tr = load(a.train)
    eh, te = load(a.test)
    stats = goal_stats(th, tr)
    cols = {g: column(eh, te, g) for g in GOALS}
    scores = [d2h({g: cols[g][i] for g in GOALS}, stats) for i in range(len(te))]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }))


def predict_all(a):
    th, tr = load(a.train)
    eh, te = load(a.test)
    stats = goal_stats(th, tr)
    feats = [h for h in th if h and not h.endswith(("X", "+", "-"))]
    numc = [h for h in feats if h[0].isupper()]
    symc = [h for h in feats if not h[0].isupper()]
    med = {}
    for h in numc:
        v = [x for x in column(th, tr, h) if x is not None]
        med[h] = float(np.median(v)) if v else 0.0
    gmed = {}
    if "Clndrs" in th and "Volume" in th:
        d = {}
        for c, v in zip(column(th, tr, "Clndrs"), column(th, tr, "Volume")):
            if c is not None and v is not None:
                d.setdefault(c, []).append(v)
        gmed = {c: float(np.median(v)) for c, v in d.items()}
    cats = {h: sorted(set(sym(th, tr, h))) for h in symc}

    def build(head, rows, mask_vol=False):
        cl = column(head, rows, "Clndrs")
        X = []
        nc = {h: column(head, rows, h) for h in numc}
        sc = {h: sym(head, rows, h) for h in symc}
        for i in range(len(rows)):
            r = []
            for h in numc:
                x = nc[h][i]
                if h == "Volume" and (mask_vol or x is None):
                    x = gmed.get(cl[i]) if cl[i] is not None else None
                if x is None:
                    x = med[h]
                r.append(x)
            for h in symc:
                r.extend(1.0 if sc[h][i] == c else 0.0 for c in cats[h])
            X.append(r)
        return np.array(X, dtype=float).reshape(len(rows), -1)

    Xa = build(th, tr)
    Xb = build(th, tr, True)
    Xt = build(eh, te)
    pred = {}
    for g in GOALS:
        y = column(th, tr, g)
        idx = [i for i, v in enumerate(y) if v is not None]
        yy = np.array([y[i] for i in idx])
        X = np.vstack([Xa[idx], Xb[idx]])
        Y = np.concatenate([yy, yy])
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0)
        m.fit(X, Y)
        pred[g] = m.predict(Xt)
    mpg = [float(v) for v in pred["Mpg+"]]
    scores = [d2h({g: float(pred[g][i]) for g in GOALS}, stats) for i in range(len(te))]
    return mpg, scores, stats, th, tr


def guess(a):
    mpg, scores, _, _, _ = predict_all(a)
    print(json.dumps({"mpg": mpg, "scores": scores}))


def compare(xs, ys):
    x = np.sort(np.asarray(xs, dtype=float))
    y = np.sort(np.asarray(ys, dtype=float))
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return {"cohen": 0.0, "cliffs": 0.0, "ks": 0.0, "same": True}
    v1 = float(np.var(x, ddof=1)) if n > 1 else 0.0
    v2 = float(np.var(y, ddof=1)) if m > 1 else 0.0
    dof = n + m - 2
    pooled = math.sqrt(((n - 1) * v1 + (m - 1) * v2) / dof) if dof > 0 else 0.0
    diff = abs(float(x.mean()) - float(y.mean()))
    if pooled > 0:
        cohen = diff / pooled
    else:
        cohen = 0.0 if diff < 1e-12 else 1e9
    gt = np.searchsorted(y, x, side="left").sum()
    lt = (m - np.searchsorted(y, x, side="right")).sum()
    cliffs = abs(float(gt - lt)) / (n * m)
    allv = np.concatenate([x, y])
    cx = np.searchsorted(x, allv, side="right") / n
    cy = np.searchsorted(y, allv, side="right") / m
    d = float(np.max(np.abs(cx - cy)))
    ks = d * math.sqrt(n * m / (n + m))
    same = bool(cohen < COHEN_T and cliffs < CLIFFS_T and ks < KS_T)
    return {"cohen": float(cohen), "cliffs": float(cliffs), "ks": float(ks), "same": same}


def certify(a):
    with open(a.test) as f:
        cases = json.load(f)
    print(json.dumps({"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}))


def flow(a):
    # Models are fit on training data only; each arrival's prediction depends
    # only on that car's own inputs, so later arrivals cannot alter earlier ones.
    mpg, scores, stats, th, tr = predict_all(a)
    cols = {g: column(th, tr, g) for g in GOALS}
    hist = [d2h({g: cols[g][i] for g in GOALS}, stats)
            for i in range(len(tr)) if all(cols[g][i] is not None for g in GOALS)]
    alerts = []
    for i in range(len(scores)):
        if i + 1 < WINDOW or not hist:
            alerts.append(False)
            continue
        win = scores[i + 1 - WINDOW:i + 1]
        alerts.append(not compare(win, hist)["same"])
    print(json.dumps({"mpg": mpg, "scores": scores, "alerts": alerts}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["remember", "guess", "certify", "flow"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    {"remember": remember, "guess": guess, "certify": certify, "flow": flow}[a.cmd](a)


main()
