import sys, csv, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

G = ["Lbs", "Acc", "Mpg"]
H = np.array([0., 1., 1.])


def read(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        names = [h.strip().rstrip("+-").strip() for h in hdr]
        rows = [x for x in r if x]
    idx = {n: i for i, n in enumerate(names)}

    def col(n):
        if n not in idx:
            return [None] * len(rows)
        return [(x[idx[n]].strip() if idx[n] < len(x) else "?") for x in rows]
    return col, len(rows)


def num(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def d2h(vals, mean, sd):
    z = np.clip((vals - mean) / (sd + 1e-32), -3, 3)
    n = 1 / (1 + np.exp(-1.702 * z))
    return np.sqrt(((n - H) ** 2).sum(axis=1) / 3)


def certify(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    n, m = len(xs), len(ys)
    vx, vy = xs.var(ddof=1), ys.var(ddof=1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(xs.mean() - ys.mean()) / (math.sqrt(pv) + 1e-32)
    sy = np.sort(ys)
    less = np.searchsorted(sy, xs, "left")
    leq = np.searchsorted(sy, xs, "right")
    cl = abs(float((less - (m - leq)).sum())) / (n * m)
    u = np.unique(np.concatenate([xs, ys]))
    fx = np.searchsorted(np.sort(xs), u, "right") / n
    fy = np.searchsorted(sy, u, "right") / m
    ks = float(np.abs(fx - fy).max() / math.sqrt(1 / n + 1 / m))
    return cohen, cl, ks, bool(cohen <= 0.35 and cl <= 0.197 and ks <= 1.36)


def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    ct, nt = read(tr)
    cs, ns = read(te)
    Y = np.array([[num(v) for v in ct(g)] for g in G]).T
    mean, sd = Y.mean(0), Y.std(0, ddof=1)
    hist = d2h(Y, mean, sd)
    cats = sorted(set(ct("origin")))

    def feats(c, n):
        cy, vo, mo = (np.array([num(v) for v in c(k)]) for k in ("Clndrs", "Volume", "Model"))
        og = c("origin")
        oh = np.array([[1.0 if (o == k) else 0.0 for k in cats] for o in og]).reshape(n, len(cats))
        return np.column_stack([cy, vo, mo, oh])

    Xt, Xs = feats(ct, nt), feats(cs, ns)
    med = np.nanmedian(Xt, axis=0)
    med = np.where(np.isnan(med), 0, med)

    def fill(X):
        X = X.copy()
        i = np.where(np.isnan(X))
        X[i] = med[i[1]]
        return X
    Xt, Xs_f = fill(Xt), fill(Xs)
    nov = [0, 2] + list(range(3, Xt.shape[1]))
    mA = RandomForestRegressor(300, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(Xt, Y)
    mB = RandomForestRegressor(300, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(Xt[:, nov], Y)
    P = np.zeros((ns, 3))
    miss = np.isnan(Xs[:, 1])
    if (~miss).any():
        P[~miss] = mA.predict(Xs_f[~miss])
    if miss.any():
        P[miss] = mB.predict(Xs_f[miss][:, nov])
    mpg = np.maximum(P[:, 2], 1e-3)
    P[:, 2] = mpg
    sc = np.clip(d2h(P, mean, sd), 0, 1)
    alerts = []
    for i in range(ns):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not certify(hist, sc[i - 19:i + 1])[3])
    print(json.dumps({"mpg": [float(x) for x in mpg],
                      "scores": [float(x) for x in sc],
                      "alerts": alerts}))


main()
