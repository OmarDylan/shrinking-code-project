import sys, csv, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

GOALS = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUM = ["Clndrs", "Volume", "Model"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def main():
    a = sys.argv
    train = read(a[a.index("--train") + 1])
    test = read(a[a.index("--test") + 1])

    rows = [r for r in train if all(fnum(r.get(g)) is not None for g in GOALS)]
    stats = {}
    for g in GOALS:
        v = np.array([float(r[g]) for r in rows])
        stats[g] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)

    def d2h(gl):
        s = 0.0
        for g, h in GOALS.items():
            m, sd = stats[g]
            z = max(-3, min(3, (gl[g] - m) / (sd + 1e-32)))
            s += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
        return math.sqrt(s / 3)

    med = {}
    for c in NUM:
        v = [fnum(r.get(c)) for r in rows]
        v = [x for x in v if x is not None]
        med[c] = float(np.median(v)) if v else 0.0
    cats = sorted({r.get("origin", "?") for r in rows})

    def feat(r):
        f = []
        for c in NUM:
            x = fnum(r.get(c))
            f.append(med[c] if x is None else x)
        o = r.get("origin", "?")
        f += [1.0 if o == k else 0.0 for k in cats]
        return f

    Xtr = np.array([feat(r) for r in rows])
    ytr = np.array([d2h({g: float(r[g]) for g in GOALS}) for r in rows])
    Xte = np.array([feat(r) for r in test])
    n = len(test)
    budget = min(10, n)

    obs = {}
    while len(obs) < budget:
        X, y = Xtr, ytr
        if obs:
            idx = list(obs)
            X = np.vstack([Xtr, Xte[idx]])
            y = np.concatenate([ytr, [obs[i] for i in idx]])
        rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=2,
                                   random_state=len(obs), n_jobs=1).fit(X, y)
        preds = np.array([t.predict(Xte) for t in rf.estimators_])
        score = preds.mean(0) - 0.5 * preds.std(0)
        for i in obs:
            score[i] = np.inf
        i = int(np.argmin(score))
        print(json.dumps({"inspect": i}), flush=True)
        resp = json.loads(sys.stdin.readline())
        obs[i] = d2h({g: float(resp["goals"][g]) for g in GOALS})

    ranked = sorted(obs, key=lambda i: (round(obs[i], 12), i))[:3]
    print(json.dumps({"selected": ranked}), flush=True)


main()
