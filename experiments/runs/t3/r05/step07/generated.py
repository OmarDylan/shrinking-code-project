import sys, csv, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

GOALS = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUM = ["Clndrs", "Volume", "Model"]


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    train, test = read(tr), read(te)

    stats = {}
    for g in GOALS:
        vals = [num(r.get(g)) for r in train]
        vals = [v for v in vals if v is not None]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        stats[g] = (m, sd)

    def d2h(d):
        s = 0.0
        for g, h in GOALS.items():
            m, sd = stats[g]
            z = max(-3.0, min(3.0, (d[g] - m) / (sd + 1e-32)))
            n = 1 / (1 + math.exp(-1.702 * z))
            s += (n - h) ** 2
        return math.sqrt(s / 3)

    cats = sorted({r.get("origin", "?") for r in train if r.get("origin", "?") != "?"})
    medians = []
    for c in NUM:
        v = [num(r.get(c)) for r in train]
        v = [x for x in v if x is not None]
        medians.append(float(np.median(v)) if v else 0.0)

    def feat(r):
        f = []
        for c, md in zip(NUM, medians):
            x = num(r.get(c))
            f.append(md if x is None else x)
        o = r.get("origin", "?")
        f += [1.0 if o == k else 0.0 for k in cats]
        return f

    X, y = [], []
    for r in train:
        if all(num(r.get(g)) is not None for g in GOALS):
            X.append(feat(r))
            y.append(d2h({g: num(r[g]) for g in GOALS}))
    n = len(test)
    if X:
        model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
        model.fit(np.array(X), np.array(y))
        pred = model.predict(np.array([feat(r) for r in test]))
    else:
        pred = np.zeros(n)

    order = sorted(range(n), key=lambda i: (pred[i], i))
    picks = order[:min(10, n)]

    seen = {}
    for i in picks:
        print(json.dumps({"inspect": i}), flush=True)
        line = sys.stdin.readline()
        resp = json.loads(line)
        seen[i] = d2h({g: float(resp["goals"][g]) for g in GOALS})

    ranked = sorted(seen, key=lambda i: (round(seen[i], 12), i))[:3]
    print(json.dumps({"selected": ranked}), flush=True)


main()
