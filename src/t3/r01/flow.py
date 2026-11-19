import sys, csv, json, time, argparse
import numpy as np
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor

W = 20
ALPHA = 0.05


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in x] for x in r if x]
    cols = []
    for h in hdr:
        ignored = h.endswith("X")
        goal = h[-1] if h[-1] in "+-" else None
        cols.append(dict(name=h, num=h[0].isupper(), ign=ignored, goal=goal))
    data = []
    for x in rows:
        d = {}
        for c, v in zip(cols, x):
            if v in ("?", ""):
                d[c["name"]] = None
            elif c["num"]:
                try:
                    d[c["name"]] = float(v)
                except ValueError:
                    d[c["name"]] = None
            else:
                d[c["name"]] = v
        data.append(d)
    return cols, data


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    cols, tr = load(a.train)
    _, te = load(a.test)
    cmap = {c["name"]: c for c in cols}
    goals = [c["name"] for c in cols if c["goal"]]
    mpg = next(g for g in goals if g.lower().startswith("mpg"))
    inputs = [c["name"] for c in cols if not c["ign"]]
    cats = {c["name"]: sorted({r[c["name"]] for r in tr if r[c["name"]] is not None})
            for c in cols if not c["num"]}

    def feat(r, fc):
        v = []
        for n in fc:
            if cmap[n]["num"]:
                v.append(r[n])
            else:
                v += [1.0 if r[n] == k else 0.0 for k in cats[n]]
        return v

    # normalization for goals
    lo, hi = {}, {}
    for g in goals:
        vals = [r[g] for r in tr if r[g] is not None]
        lo[g], hi[g] = min(vals), max(vals)

    def dist(vals):
        s = 0.0
        for g in goals:
            n = (vals[g] - lo[g]) / ((hi[g] - lo[g]) or 1.0)
            n = min(1.0, max(0.0, n))
            h = 1.0 if cmap[g]["goal"] == "+" else 0.0
            s += (n - h) ** 2
        return (s / len(goals)) ** 0.5

    hist = [dist(r) for r in tr if all(r[g] is not None for g in goals)]

    cache = {}

    def predict(target, r):
        fc = tuple(n for n in inputs if n != target and n != mpg and r.get(n) is not None)
        key = (target, fc)
        if key not in cache:
            rows = [x for x in tr if x[target] is not None and all(x[n] is not None for n in fc)]
            y = np.array([x[target] for x in rows], float)
            if not fc or len(rows) < 5:
                m = float(y.mean()) if len(y) else 0.0
                cache[key] = lambda _r, m=m: m
            else:
                X = np.array([feat(x, fc) for x in rows], float)
                mdl = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=1).fit(X, y)
                cache[key] = lambda _r, mdl=mdl, fc=fc: float(mdl.predict(np.array([feat(_r, fc)], float))[0])
        return cache[key](r)

    mp, sc, al = [], [], []
    win = []
    for r in te:
        pm = predict(mpg, r)
        vals = {}
        for g in goals:
            if g == mpg:
                vals[g] = pm
            elif r.get(g) is not None:
                vals[g] = r[g]
            else:
                vals[g] = predict(g, r)
        s = dist(vals)
        mp.append(pm)
        sc.append(s)
        win.append(s)
        if len(win) > W:
            win.pop(0)
        alert = False
        if len(win) == W and len(hist) > 1:
            alert = bool(ks_2samp(win, hist).pvalue < ALPHA)
        al.append(alert)
    print(json.dumps(dict(mpg=mp, scores=sc, alerts=al, time_seconds=time.time() - t0)))


if __name__ == "__main__":
    main()
