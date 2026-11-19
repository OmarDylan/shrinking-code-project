import sys, csv, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def load(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def goals(header, rows):
    idx = [header.index(n) for n in ("Lbs-", "Acc+", "Mpg+")]
    return [[float(row[i]) for i in idx] for row in rows]

def stats(G):
    n = len(G)
    means = [sum(g[j] for g in G) / n for j in range(3)]
    sds = [math.sqrt(sum((g[j] - means[j]) ** 2 for g in G) / (n - 1)) if n > 1 else 0.0
           for j in range(3)]
    return means, sds

def d2h(g, means, sds):
    heaven = [0.0, 1.0, 1.0]
    s = 0.0
    for j in range(3):
        z = (g[j] - means[j]) / (sds[j] + 1e-32)
        z = max(-3.0, min(3.0, z))
        v = 1 / (1 + math.exp(-1.702 * z))
        s += (v - heaven[j]) ** 2
    return math.sqrt(s / 3)

def arg(name):
    a = sys.argv
    return a[a.index(name) + 1]

def remember(train, test):
    th, tr = load(train)
    eh, er = load(test)
    G = goals(th, tr)
    E = goals(eh, er)
    means, sds = stats(G)
    scores = [d2h(g, means, sds) for g in E]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else float("nan")
    except ValueError:
        return float("nan")

def feats(header, rows, cats):
    ci, vi, mi, oi = (header.index(n) for n in ("Clndrs", "Volume", "Model", "origin"))
    X = []
    for r in rows:
        X.append([num(r[ci]), num(r[vi]), num(r[mi])] + [1.0 if r[oi] == c else 0.0 for c in cats])
    return np.array(X, dtype=float)

def predict(Xtr, ytr, Xte):
    ps = []
    for m in (RandomForestRegressor(300, min_samples_leaf=2, random_state=0, n_jobs=1),
              GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, random_state=0)):
        ps.append(m.fit(Xtr, ytr).predict(Xte))
    return np.mean(ps, axis=0)

def guess(train, test):
    th, tr = load(train)
    eh, er = load(test)
    keep = []
    for r in tr:
        try:
            g = [float(r[th.index(n)]) for n in ("Lbs-", "Acc+", "Mpg+")]
            if all(math.isfinite(x) for x in g):
                keep.append((r, g))
        except ValueError:
            pass
    tr = [k[0] for k in keep]
    G = [k[1] for k in keep]
    means, sds = stats(G)
    y_mpg = np.array([g[2] for g in G])
    y_d = np.array([d2h(g, means, sds) for g in G])
    cats = sorted({r[th.index("origin")] for r in tr})
    Xtr = feats(th, tr, cats)
    Xte = feats(eh, er, cats)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    fill = lambda X: np.where(np.isnan(X), med, X)
    Xtr = fill(Xtr)
    n = len(Xte)
    mpg = np.zeros(n)
    sc = np.zeros(n)
    hasv = ~np.isnan(Xte[:, 1])
    cols_nov = [0, 2] + list(range(3, Xtr.shape[1]))
    for mask, cols in ((hasv, list(range(Xtr.shape[1]))), (~hasv, cols_nov)):
        if not mask.any():
            continue
        A = Xtr[:, cols]
        B = fill(Xte)[mask][:, cols]
        mpg[mask] = predict(A, y_mpg, B)
        sc[mask] = predict(A, y_d, B)
    mpg = np.where(np.isfinite(mpg) & (mpg > 0), mpg, float(y_mpg.mean()))
    mpg = np.maximum(mpg, 1e-3)
    sc = np.clip(np.nan_to_num(sc, nan=float(y_d.mean())), 0.0, 1.0)
    return {"mpg": [float(x) for x in mpg], "scores": [float(x) for x in sc]}

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "remember"
    train, test = arg("--train"), arg("--test")
    out = guess(train, test) if cmd == "guess" else remember(train, test)
    sys.stdout.write(json.dumps(out))

if __name__ == "__main__":
    main()
