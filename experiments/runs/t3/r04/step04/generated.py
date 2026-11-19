import sys, csv, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

G = ["Lbs-", "Acc+", "Mpg+"]
H = [0, 1, 1]

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")

def load(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))

def same(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    n, m = len(xs), len(ys)
    pv = ((n - 1) * xs.var(ddof=1) + (m - 1) * ys.var(ddof=1)) / (n + m - 2)
    cohen = abs(xs.mean() - ys.mean()) / (math.sqrt(pv) + 1e-32)
    cl = abs(np.sign(xs[:, None] - ys[None, :]).sum()) / (n * m)
    sx, sy = np.sort(xs), np.sort(ys)
    v = np.union1d(xs, ys)
    d = np.abs(np.searchsorted(sx, v, "right") / n - np.searchsorted(sy, v, "right") / m).max()
    ks = d / math.sqrt(1 / n + 1 / m)
    return cohen <= 0.35 and cl <= 0.197 and ks <= 1.36

def main():
    a = sys.argv
    tr = load(a[a.index("--train") + 1])
    te = load(a[a.index("--test") + 1])
    cats = sorted({r["origin"] for r in tr})
    def feat(r):
        return [num(r["Clndrs"]), num(r["Volume"]), num(r["Model"])] + [1.0 if r["origin"] == c else 0.0 for c in cats]
    X = np.array([feat(r) for r in tr])
    Y = np.array([[num(r[g]) for g in G] for r in tr])
    keep = np.isfinite(Y).all(1)
    X, Y = X[keep], Y[keep]
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0, med)
    def fill(M):
        M = M.copy()
        idx = np.where(np.isnan(M))
        M[idx] = med[idx[1]]
        return M
    Xf = fill(X)
    cols_nov = [0, 2] + list(range(3, X.shape[1]))
    mean = Y.mean(0); sd = Y.std(0, ddof=1)
    def d2h(Gv):
        z = np.clip((Gv - mean) / (sd + 1e-32), -3, 3)
        nv = 1 / (1 + np.exp(-1.702 * z))
        return np.sqrt(((nv - np.array(H)) ** 2).sum(1) / 3)
    full = RandomForestRegressor(200, min_samples_leaf=2, random_state=0, n_jobs=1).fit(Xf, Y)
    nov = RandomForestRegressor(200, min_samples_leaf=2, random_state=0, n_jobs=1).fit(Xf[:, cols_nov], Y)
    hist = d2h(Y)
    Xt = np.array([feat(r) for r in te]).reshape(len(te), -1)
    P = np.zeros((len(te), 3))
    for i in range(len(te)):
        row = Xt[i:i + 1]
        miss_vol = math.isnan(row[0, 1])
        row = fill(row)
        P[i] = (nov.predict(row[:, cols_nov]) if miss_vol else full.predict(row))[0]
    mpg = np.maximum(P[:, 2], 1e-3)
    P[:, 2] = mpg
    sc = np.clip(d2h(P), 0, 1) if len(te) else np.array([])
    alerts = []
    for i in range(len(te)):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(bool(not same(hist, sc[i - 19:i + 1])))
    print(json.dumps({"mpg": [float(x) for x in mpg], "scores": [float(x) for x in sc], "alerts": alerts}))

main()
