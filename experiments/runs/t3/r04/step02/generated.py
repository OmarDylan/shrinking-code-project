import sys, csv, json, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def read(p):
    with open(p, newline='') as f:
        r = list(csv.reader(f))
    h = [c.strip().rstrip('+-X') for c in r[0]]
    return [dict(zip(h, [x.strip() for x in row])) for row in r[1:] if row]

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def main():
    a = sys.argv
    tr = read(a[a.index('--train') + 1])
    te = read(a[a.index('--test') + 1])
    G = ['Lbs', 'Acc', 'Mpg']
    tr = [r for r in tr if all(num(r.get(g)) is not None for g in G)]
    Y = np.array([[float(r[g]) for g in G] for r in tr])
    mu = Y.mean(0)
    sd = Y.std(0, ddof=1) if len(Y) > 1 else np.ones(3)
    z = np.clip((Y - mu) / (sd + 1e-32), -3, 3)
    nm = 1 / (1 + np.exp(-1.702 * z))
    d2h = np.sqrt((nm[:, 0] ** 2 + (nm[:, 1] - 1) ** 2 + (nm[:, 2] - 1) ** 2) / 3)
    mpg = Y[:, 2]
    N = ['Clndrs', 'Volume', 'Model']
    cats = sorted({r.get('origin', '') for r in tr})
    med = {}
    for c in N:
        v = [num(r.get(c)) for r in tr if num(r.get(c)) is not None]
        med[c] = float(np.median(v)) if v else 0.0

    def feat(r, vol=True):
        f = []
        for c in N:
            if c == 'Volume' and not vol:
                continue
            v = num(r.get(c))
            f.append(med[c] if v is None else v)
        f += [1.0 if r.get('origin', '') == k else 0.0 for k in cats]
        return f

    out_m = np.zeros(len(te))
    out_s = np.zeros(len(te))
    for vol in (True, False):
        idx = [i for i, r in enumerate(te)
               if (vol and i % 2 == 1 and num(r.get('Volume')) is not None)
               or (not vol and (i % 2 == 0 or num(r.get('Volume')) is None))]
        if not idx:
            continue
        Xt = np.array([feat(r, vol) for r in tr])
        Xe = np.array([feat(te[i], vol) for i in idx])
        for tgt, out in ((mpg, out_m), (d2h, out_s)):
            m = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                      random_state=0, n_jobs=1).fit(Xt, tgt)
            out[idx] = m.predict(Xe)
    mp = [max(float(x), 1e-3) for x in out_m]
    sc = [min(1.0, max(0.0, float(x))) for x in out_s]
    print(json.dumps({"mpg": mp, "scores": sc}))

main()
