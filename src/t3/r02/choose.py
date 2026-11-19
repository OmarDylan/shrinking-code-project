import sys, csv, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

G = ["Lbs-", "Acc+", "Mpg+"]
H = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
NUM = ["Clndrs", "Volume", "Model"]


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    train, test = read(tr), read(te)
    train = [r for r in train if all(fnum(r.get(g)) is not None for g in G)]
    st = {}
    for g in G:
        v = np.array([float(r[g]) for r in train])
        st[g] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)

    def d2h(goals):
        s = 0.0
        for g in G:
            m, sd = st[g]
            z = max(-3, min(3, (goals[g] - m) / (sd + 1e-32)))
            n = 1 / (1 + math.exp(-1.702 * z))
            s += (n - H[g]) ** 2
        return math.sqrt(s / 3)

    y = np.array([d2h({g: float(r[g]) for g in G}) for r in train])
    med = {}
    for c in NUM:
        v = [fnum(r.get(c)) for r in train]
        v = [x for x in v if x is not None]
        med[c] = float(np.median(v)) if v else 0.0
    cats = sorted({r.get("origin", "?") for r in train})

    def feat(r):
        f = []
        for c in NUM:
            x = fnum(r.get(c))
            f.append(med[c] if x is None else x)
        o = r.get("origin", "?")
        f += [1.0 if o == k else 0.0 for k in cats]
        return f

    X = np.array([feat(r) for r in train])
    Xt = np.array([feat(r) for r in test])
    m = RandomForestRegressor(n_estimators=200, random_state=0).fit(X, y)
    p = m.predict(Xt)
    order = sorted(range(len(test)), key=lambda i: (p[i], i))
    picks = order[:min(10, len(test))]
    obs = []
    for i in picks:
        print(json.dumps({"inspect": i}), flush=True)
        resp = json.loads(sys.stdin.readline())
        obs.append((round(d2h(resp["goals"]), 12), i))
    obs.sort()
    print(json.dumps({"selected": [i for _, i in obs[:3]]}), flush=True)


main()
