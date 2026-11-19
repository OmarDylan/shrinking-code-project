import sys, csv, json, math
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

def read(p):
    with open(p, newline='') as f:
        r = list(csv.reader(f))
    h = [c.strip() for c in r[0]]
    return h, [[c.strip() for c in row] for row in r[1:] if row]

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else float('nan')
    except Exception:
        return float('nan')

def main():
    a = sys.argv
    tr, te = a[a.index('--train') + 1], a[a.index('--test') + 1]
    h, R = read(tr)
    h2, T = read(te)
    ix = {c: i for i, c in enumerate(h)}
    ix2 = {c: i for i, c in enumerate(h2)}
    cats = sorted({r[ix['origin']] for r in R})

    def feats(rows, idx):
        X = []
        for r in rows:
            g = lambda c: num(r[idx[c]]) if c in idx and idx[c] < len(r) else float('nan')
            o = r[idx['origin']] if 'origin' in idx and idx['origin'] < len(r) else '?'
            X.append([g('Clndrs'), g('Volume'), g('Model')] + [1.0 if o == c else 0.0 for c in cats])
        return np.array(X)

    Xtr = feats(R, ix)
    Xte = feats(T, ix2)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0, med)
    fill = lambda X: np.where(np.isnan(X), med, X)
    Xtr, Xte_f = fill(Xtr), Xte.copy()
    vmiss = np.isnan(Xte[:, 1])
    Xte_f = fill(Xte)

    names = ['Lbs-', 'Acc+', 'Mpg+']
    Y = np.array([[num(r[ix[n]]) for n in names] for r in R])
    ok = ~np.isnan(Y).any(axis=1)
    mean = np.nanmean(Y, axis=0)
    sd = np.array([np.nanstd(Y[:, j], ddof=1) for j in range(3)])
    heaven = np.array([0.0, 1.0, 1.0])

    def d2h(V):
        z = np.clip((V - mean) / (sd + 1e-32), -3, 3)
        n = 1 / (1 + np.exp(-1.702 * z))
        return np.sqrt(((n - heaven) ** 2).sum(axis=1) / 3)

    D = np.full(len(R), np.nan)
    D[ok] = d2h(Y[ok])

    def fitpred(cols, targets, Xa, Xb, log):
        out = []
        for t in targets:
            m = ~np.isnan(t)
            y = np.log(t[m]) if log else t[m]
            ps = []
            for mod in (GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0),
                        RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0)):
                mod.fit(Xa[m][:, cols], y)
                ps.append(mod.predict(Xb[:, cols]))
            out.append(np.mean(ps, axis=0))
        return out

    n = len(T)
    mpg = np.zeros(n)
    sc = np.zeros(n)
    for cols, sel in (([0, 1, 2] + list(range(3, 3 + len(cats))), ~vmiss),
                      ([0, 2] + list(range(3, 3 + len(cats))), vmiss)):
        if not sel.any():
            continue
        Xs = Xte_f[sel]
        pm, = fitpred(cols, [Y[:, 2]], Xtr, Xs, True)
        pm = np.exp(pm)
        pl, = fitpred(cols, [Y[:, 0]], Xtr, Xs, False)
        pa, = fitpred(cols, [Y[:, 1]], Xtr, Xs, False)
        pd, = fitpred(cols, [D], Xtr, Xs, False)
        via = d2h(np.stack([pl, pa, pm], axis=1))
        mpg[sel] = pm
        sc[sel] = 0.5 * via + 0.5 * pd
    mpg = np.where(np.isfinite(mpg) & (mpg > 0), mpg, np.nanmean(Y[:, 2]))
    sc = np.clip(np.nan_to_num(sc, nan=0.5), 0, 1)
    print(json.dumps({"mpg": [float(x) for x in mpg], "scores": [float(x) for x in sc]}))

main()
