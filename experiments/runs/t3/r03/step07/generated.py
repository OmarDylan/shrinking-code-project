import sys, csv, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

GOALS = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]
NUM = ["Clndrs", "Volume", "Model"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def main():
    a = sys.argv
    train = read(a[a.index("--train") + 1])
    test = read(a[a.index("--test") + 1])

    # goal statistics (training only)
    tr = [r for r in train if all(num(r.get(g)) is not None for g, _ in GOALS)]
    stats = {}
    for g, _ in GOALS:
        v = np.array([num(r[g]) for r in tr])
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        stats[g] = (v.mean(), sd)

    def d2h(goals):
        s = 0.0
        for g, h in GOALS:
            m, sd = stats[g]
            z = max(-3.0, min(3.0, (goals[g] - m) / (sd + 1e-32)))
            n = 1 / (1 + math.exp(-1.702 * z))
            s += (n - h) ** 2
        return math.sqrt(s / 3)

    cats = sorted({r["origin"] for r in tr if r.get("origin") not in (None, "?", "")})
    med = {}
    for c in NUM:
        v = [num(r.get(c)) for r in tr]
        v = [x for x in v if x is not None]
        med[c] = float(np.median(v)) if v else 0.0

    def feat(r):
        f = []
        for c in NUM:
            x = num(r.get(c))
            f.append(med[c] if x is None else x)
        f += [1.0 if r.get("origin") == k else 0.0 for k in cats]
        return f

    X = np.array([feat(r) for r in tr])
    y = np.array([d2h({g: num(r[g]) for g, _ in GOALS}) for r in tr])
    Xt = np.array([feat(r) for r in test]) if test else np.zeros((0, len(X[0])))

    k = min(10, len(test))
    if len(X) > 0 and len(test) > 0:
        model = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
        model.fit(X, y)
        pred = model.predict(Xt)
    else:
        pred = np.zeros(len(test))
    order = sorted(range(len(test)), key=lambda i: (pred[i], i))
    picks = order[:k]

    obs = {}
    for i in picks:
        print(json.dumps({"inspect": i}), flush=True)
        line = sys.stdin.readline()
        resp = json.loads(line)
        obs[i] = d2h({g: float(resp["goals"][g]) for g, _ in GOALS})

    ranked = sorted(obs, key=lambda i: (round(obs[i], 12), i))[:3]
    print(json.dumps({"selected": ranked}), flush=True)


main()
