import sys, csv, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

NUM = ["Clndrs", "Volume", "Model"]
GOALS = ["Lbs", "Acc", "Mpg"]


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    names = [h.rstrip("+-X") if h[-1:] in "+-X" else h for h in hdr]
    idx = {n: i for i, n in enumerate(names)}
    return idx, rows


def val(row, idx, n):
    if n not in idx or idx[n] >= len(row):
        return None
    v = row[idx[n]]
    if v == "?" or v == "":
        return None
    return v


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
    ti, trows = load(tr)
    si, srows = load(te)

    X, G = [], []
    for r in trows:
        g = [fnum(val(r, ti, n)) for n in GOALS]
        if any(x is None for x in g):
            continue
        X.append([fnum(val(r, ti, n)) for n in NUM] + [val(r, ti, "origin")])
        G.append(g)
    G = np.array(G)
    mean = G.mean(0)
    sd = G.std(0, ddof=1) if len(G) > 1 else np.zeros(3)
    z = np.clip((G - mean) / (sd + 1e-32), -3, 3)
    nm = 1 / (1 + np.exp(-1.702 * z))
    h = np.array([0, 1, 1])
    d2h = np.sqrt(((nm - h) ** 2).sum(1) / 3)
    Y = np.column_stack([G[:, 2], d2h])

    cats = sorted({x[3] for x in X if x[3] is not None})
    med = []
    for j in range(3):
        v = [x[j] for x in X if x[j] is not None]
        med.append(float(np.median(v)) if v else 0.0)

    def feat(nums, org, cols):
        f = [nums[j] if nums[j] is not None else med[j] for j in cols]
        f += [1.0 if org == c else 0.0 for c in cats]
        return f

    models = {}

    def get_model(cols):
        if cols not in models:
            F = np.array([feat(x[:3], x[3], cols) for x in X])
            m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                      random_state=0, n_jobs=1)
            m.fit(F, Y)
            models[cols] = m
        return models[cols]

    mpg, sc = [], []
    for r in srows:
        nums = [fnum(val(r, si, n)) for n in NUM]
        org = val(r, si, "origin")
        cols = tuple(j for j in range(3) if nums[j] is not None)
        m = get_model(cols)
        p = m.predict(np.array([feat(nums, org, cols)]))[0]
        mpg.append(max(float(p[0]), 1e-3))
        sc.append(min(1.0, max(0.0, float(p[1]))))
    print(json.dumps({"mpg": mpg, "scores": sc}))


main()
