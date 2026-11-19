import sys, csv, json, math
from bisect import bisect_left, bisect_right
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

GOALS = [('Lbs-', 0), ('Acc+', 1), ('Mpg+', 1)]

def read(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return hdr, rows

def fit_stats(vals):
    n = len(vals)
    st = []
    for k in range(3):
        v = [r[k] for r in vals]
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        st.append((m, sd))
    return st

def d2h(vals, stats):
    s = 0.0
    for k, v in enumerate(vals):
        m, sd = stats[k]
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        nv = 1 / (1 + math.exp(-1.702 * z))
        s += (nv - GOALS[k][1]) ** 2
    return math.sqrt(s / 3)

def load_train(path):
    h, rows = read(path)
    ci = [h.index(g) for g, _ in GOALS]
    rows = [r for r in rows if all(r[i] != '?' for i in ci)]
    vals = [[float(r[i]) for i in ci] for r in rows]
    return h, rows, vals

def num(x):
    try:
        return float(x)
    except ValueError:
        return float('nan')

def remember(a):
    train = a[a.index('--train') + 1]
    test = a[a.index('--test') + 1]
    _, _, tr = load_train(train)
    stats = fit_stats(tr)
    h2, te = read(test)
    ci2 = [h2.index(g) for g, _ in GOALS]
    scores = [d2h([float(r[i]) for i in ci2], stats) for r in te]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    out = {"scores": scores, "top": order[:10], "best": order[0],
           "median": order[len(scores) // 2], "worst": order[-1]}
    sys.stdout.write(json.dumps(out))

def predict(train, test):
    h, rows, vals = load_train(train)
    stats = fit_stats(vals)
    mpg = np.array([v[2] for v in vals])
    sc = np.array([d2h(v, stats) for v in vals])
    h2, te = read(test)
    cats = sorted({r[h.index('origin')] for r in rows} - {'?'})
    nf = ['Clndrs', 'Volume', 'Model']

    def feats(hdr, rs):
        X = np.array([[num(r[hdr.index(c)]) for c in nf] for r in rs], dtype=float).reshape(len(rs), 3)
        O = np.array([[1.0 if r[hdr.index('origin')] == c else 0.0 for c in cats]
                      for r in rs]).reshape(len(rs), len(cats))
        return X, O

    Xtr, Otr = feats(h, rows)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def prep(X, O, use):
        X = np.where(np.isnan(X), med, X)
        return np.hstack([X[:, use], O])

    Xte, Ote = feats(h2, te)
    vmiss = np.array([r[h2.index('Volume')] == '?' for r in te], dtype=bool)
    mp = np.zeros(len(te))
    ds = np.zeros(len(te))
    for use, mask in (([0, 1, 2], ~vmiss), ([0, 2], vmiss)):
        if not mask.any():
            continue
        A = prep(Xtr, Otr, use)
        B = prep(Xte[mask], Ote[mask], use)
        kw = dict(n_estimators=200, learning_rate=0.05, max_depth=3,
                  subsample=0.8, random_state=0)
        m1 = GradientBoostingRegressor(**kw).fit(A, np.log(np.maximum(mpg, 1e-6)))
        m2 = GradientBoostingRegressor(**kw).fit(A, sc)
        mp[mask] = np.exp(m1.predict(B))
        ds[mask] = m2.predict(B)
    mp = np.where(np.isfinite(mp) & (mp > 0), mp, float(mpg.mean()))
    ds = np.clip(np.where(np.isfinite(ds), ds, float(sc.mean())), 0.0, 1.0)
    return mp, ds, sc

def guess(a):
    train = a[a.index('--train') + 1]
    test = a[a.index('--test') + 1]
    mp, ds, _ = predict(train, test)
    sys.stdout.write(json.dumps({"mpg": [float(x) for x in mp],
                                 "scores": [float(x) for x in ds]}))

def compare(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    vy = sum((y - my) ** 2 for y in ys) / (m - 1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (math.sqrt(pv) + 1e-32)
    sx, sy = sorted(xs), sorted(ys)
    tot = 0
    for x in xs:
        tot += bisect_left(sy, x) - (m - bisect_right(sy, x))
    cliffs = abs(tot) / (n * m)
    d = 0.0
    for v in sorted(set(sx) | set(sy)):
        d = max(d, abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m))
    ks = d / math.sqrt(1 / n + 1 / m)
    same = bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks, "same": same}

def certify(a):
    test = a[a.index('--test') + 1]
    with open(test) as f:
        cases = json.load(f)
    out = {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}
    sys.stdout.write(json.dumps(out))

def flow(a):
    train = a[a.index('--train') + 1]
    test = a[a.index('--test') + 1]
    mp, ds, sc = predict(train, test)
    hist = [float(x) for x in sc]
    scores = [float(x) for x in ds]
    alerts = []
    for i in range(len(scores)):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, scores[i - 19:i + 1])["same"])
    sys.stdout.write(json.dumps({"mpg": [float(x) for x in mp],
                                 "scores": scores, "alerts": alerts}))

def main():
    a = sys.argv
    cmds = {'remember': remember, 'guess': guess, 'certify': certify, 'flow': flow}
    if len(a) < 2 or a[1] not in cmds:
        sys.exit(1)
    cmds[a[1]](a)

main()
