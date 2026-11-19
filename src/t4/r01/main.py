import csv, sys, json, math, argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}


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


def guess(a):
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
    print(json.dumps({"mpg": mpg, "scores": scores}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["remember", "guess"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    (remember if a.cmd == "remember" else guess)(a)


main()
