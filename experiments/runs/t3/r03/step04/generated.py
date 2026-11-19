import sys, csv, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def d2h_scores(G, stats):
    tot = 0
    for j, h in enumerate((0, 1, 1)):
        m, s = stats[j]
        z = np.clip((G[:, j] - m) / (s + 1e-32), -3, 3)
        n = 1 / (1 + np.exp(-1.702 * z))
        tot = tot + (n - h) ** 2
    return np.sqrt(tot / 3)


def certify(xs, ys):
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    n, m = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(x.mean() - y.mean()) / (math.sqrt(pv) + 1e-32)
    cliffs = abs(np.sign(x[:, None] - y[None, :]).sum()) / (n * m)
    xs_s, ys_s = np.sort(x), np.sort(y)
    v = np.union1d(x, y)
    fx = np.searchsorted(xs_s, v, side="right") / n
    fy = np.searchsorted(ys_s, v, side="right") / m
    ks = np.abs(fx - fy).max() / math.sqrt(1 / n + 1 / m)
    same = bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)
    return same


def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    train, test = read(tr), read(te)
    origins = sorted({r["origin"] for r in train})

    def feats(rows, vol):
        X = []
        for r in rows:
            f = [num(r["Clndrs"]), num(r["Model"])]
            if vol:
                f.append(num(r.get("Volume")))
            f += [1.0 if r["origin"] == o else 0.0 for o in origins]
            X.append(f)
        return np.array(X, float)

    G = np.array([[num(r["Lbs-"]), num(r["Acc+"]), num(r["Mpg+"])] for r in train])
    stats = [(G[:, j].mean(), G[:, j].std(ddof=1)) for j in range(3)]
    hist = d2h_scores(G, stats)
    mpg_tr = G[:, 2]

    Xf, Xn = feats(train, True), feats(train, False)
    Tf, Tn = feats(test, True), feats(test, False)

    def fit_pred(y):
        out = np.zeros(len(test))
        mf = RandomForestRegressor(300, min_samples_leaf=2, random_state=0, n_jobs=1).fit(Xf, y)
        mn = RandomForestRegressor(300, min_samples_leaf=2, random_state=0, n_jobs=1).fit(Xn, y)
        for i in range(len(test)):
            if np.isnan(Tf[i]).any():
                out[i] = mn.predict(Tn[i:i + 1])[0]
            else:
                out[i] = mf.predict(Tf[i:i + 1])[0]
        return out

    mpg = np.maximum(fit_pred(mpg_tr), 1e-3)
    sc = np.clip(fit_pred(hist), 0, 1)

    alerts = []
    for i in range(len(sc)):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not certify(hist, sc[i - 19:i + 1]))
    print(json.dumps({"mpg": [float(v) for v in mpg],
                      "scores": [float(v) for v in sc],
                      "alerts": alerts}))


if __name__ == "__main__":
    main()
