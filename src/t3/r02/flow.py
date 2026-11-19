import csv, sys, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

G = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def d2h_fn(train):
    st = []
    for g, h in G:
        a = np.array([float(r[g]) for r in train])
        st.append((a.mean(), a.std(ddof=1), h))

    def f(vals):
        s = 0.0
        for v, (m, sd, h) in zip(vals, st):
            z = min(3, max(-3, (v - m) / (sd + 1e-32)))
            s += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
        return math.sqrt(s / 3)
    return f


def certify(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    n, m = len(xs), len(ys)
    vx, vy = xs.var(ddof=1), ys.var(ddof=1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(xs.mean() - ys.mean()) / (math.sqrt(pv) + 1e-32)
    cliffs = abs(np.sign(xs[:, None] - ys[None, :]).sum()) / (n * m)
    sx, sy = np.sort(xs), np.sort(ys)
    u = np.unique(np.concatenate([xs, ys]))
    d = np.abs(np.searchsorted(sx, u, "right") / n - np.searchsorted(sy, u, "right") / m).max()
    ks = d / math.sqrt(1 / n + 1 / m)
    same = bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)
    return float(cohen), float(cliffs), float(ks), same


def flow(train_p, test_p):
    tr, te = read(train_p), read(test_p)
    f = d2h_fn(tr)
    hist = np.array([f([float(r[g]) for g, _ in G]) for r in tr])
    origins = sorted({r["origin"] for r in tr})

    def feats(r, vol):
        row = [num(r.get("Clndrs")), num(r.get("Model"))]
        row = [0.0 if v is None else v for v in row]
        row += [1.0 if r.get("origin") == o else 0.0 for o in origins]
        if vol:
            v = num(r.get("Volume"))
            row.append(0.0 if v is None else v)
        return row

    y = np.array([[float(r["Mpg+"]), f([float(r[g]) for g, _ in G])] for r in tr])
    models = {}
    for vol in (True, False):
        X = np.array([feats(r, vol) for r in tr])
        models[vol] = RandomForestRegressor(300, min_samples_leaf=2, random_state=0, n_jobs=1).fit(X, y)
    mpg, sc = [], []
    for r in te:
        vol = num(r.get("Volume")) is not None
        p = models[vol].predict(np.array([feats(r, vol)]))[0]
        mpg.append(float(max(p[0], 1e-3)))
        sc.append(float(min(1.0, max(0.0, p[1]))))
    alerts = []
    for i in range(len(sc)):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not certify(hist, sc[i - 19:i + 1])[3])
    return {"mpg": mpg, "scores": sc, "alerts": alerts}


def main():
    a = sys.argv
    if len(a) < 2 or a[1] != "flow":
        sys.exit(1)
    opts = {a[i]: a[i + 1] for i in range(2, len(a) - 1, 2)}
    print(json.dumps(flow(opts["--train"], opts["--test"])))


main()
