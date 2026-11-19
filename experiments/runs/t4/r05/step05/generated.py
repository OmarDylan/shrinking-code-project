import sys, csv, json, math
from bisect import bisect_right
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

def train_goals(th, tr):
    keep = []
    for r in tr:
        try:
            g = [float(r[th.index(n)]) for n in ("Lbs-", "Acc+", "Mpg+")]
            if all(math.isfinite(x) for x in g):
                keep.append((r, g))
        except ValueError:
            pass
    return [k[0] for k in keep], [k[1] for k in keep]

def guess(train, test):
    th, tr = load(train)
    eh, er = load(test)
    tr, G = train_goals(th, tr)
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

def svar(v):
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)

def compare(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n, m = len(xs), len(ys)
    pooled = ((n - 1) * svar(xs) + (m - 1) * svar(ys)) / (n + m - 2)
    cohen = abs(sum(xs) / n - sum(ys) / m) / (math.sqrt(pooled) + 1e-32)
    cl = sum((x > y) - (x < y) for x in xs for y in ys)
    cliffs = abs(cl) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    d = max(abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m) for v in set(sx) | set(sy))
    ks = d / math.sqrt(1 / n + 1 / m)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks,
            "same": bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)}

def certify(path):
    with open(path) as f:
        cases = json.load(f)
    return {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}

def flow(train, test):
    out = guess(train, test)
    th, tr = load(train)
    _, G = train_goals(th, tr)
    means, sds = stats(G)
    hist = [d2h(g, means, sds) for g in G]
    sc = out["scores"]
    alerts = []
    for i in range(len(sc)):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, sc[i - 19:i + 1])["same"])
    out["alerts"] = alerts
    return out

def zlogit(v, mean, sd):
    z = (v - mean) / (sd + 1e-32)
    z = max(-3.0, min(3.0, z))
    return 1 / (1 + math.exp(-1.702 * z))

def msd(v):
    if not v:
        return 0.0, 0.0
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, s

def blame(train, test):
    th, tr = load(train)
    eh, er = load(test)
    tr, G = train_goals(th, tr)
    means, sds = stats(G)
    gnames = ("Lbs-", "Acc+", "Mpg+")
    heaven = [0.0, 1.0, 1.0]
    d = [round(d2h(g, means, sds), 12) for g in G]
    order = sorted(range(len(tr)), key=lambda i: (d[i], i))
    q = max(1, len(tr) // 4)
    good = [tr[i] for i in order[:q]]
    poor = [tr[i] for i in order[-q:]]
    numf = ("Clndrs", "Volume", "Model")
    prof = {}
    for f in numf:
        j = th.index(f)
        col = [num(r[j]) for r in tr]
        col = [x for x in col if not math.isnan(x)]
        m, s = msd(col)
        cen = []
        for grp in (good, poor):
            vals = [num(r[j]) for r in grp]
            vals = [x for x in vals if not math.isnan(x)]
            cen.append(zlogit(sum(vals) / len(vals), m, s) if vals else None)
        prof[f] = (m, s, cen[0], cen[1])
    oj = th.index("origin")
    def freq(grp):
        vals = [r[oj] for r in grp if r[oj] != "?" and r[oj] != ""]
        c = {}
        for v in vals:
            c[v] = c.get(v, 0) + 1
        return c, len(vals)
    gf, gn = freq(good)
    pf, pn = freq(poor)
    ehj = {f: eh.index(f) for f in numf}
    eoj = eh.index("origin")
    egi = [eh.index(n) if n in eh else None for n in gnames]
    diags = []
    for r in er:
        feat = {}
        for f in numf:
            x = num(r[ehj[f]])
            m, s, gc, pc = prof[f]
            if math.isnan(x) or gc is None or pc is None:
                feat[f] = 0.0
            else:
                nx = zlogit(x, m, s)
                feat[f] = max(0.0, abs(nx - gc) - abs(nx - pc))
        c = r[eoj]
        if c in ("?", ""):
            feat["origin"] = 0.0
        else:
            a = pf.get(c, 0) / pn if pn else 0.0
            b = gf.get(c, 0) / gn if gn else 0.0
            feat["origin"] = max(0.0, a - b)
        gl = {}
        for j, n in enumerate(gnames):
            v = num(r[egi[j]]) if egi[j] is not None else float("nan")
            gl[n] = 0.0 if math.isnan(v) else abs(zlogit(v, means[j], sds[j]) - heaven[j])
        key = lambda dct: sorted(dct, key=lambda k: (-round(dct[k], 10), k))
        diags.append({"goals": gl, "features": feat,
                      "goal_order": key(gl), "feature_order": key(feat)})
    return {"diagnoses": diags}

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "remember"
    if cmd == "certify":
        out = certify(arg("--test"))
    else:
        train, test = arg("--train"), arg("--test")
        if cmd == "guess":
            out = guess(train, test)
        elif cmd == "flow":
            out = flow(train, test)
        elif cmd == "blame":
            out = blame(train, test)
        else:
            out = remember(train, test)
    sys.stdout.write(json.dumps(out))

if __name__ == "__main__":
    main()
