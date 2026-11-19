import sys, json, csv, math, argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor

GOALS = ["Lbs-", "Acc+", "Mpg+"]
HEAVEN = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUM = ["Clndrs", "Volume", "Model"]


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def stats(train, header):
    st = {}
    for g in GOALS:
        vals = [fnum(r[header.index(g)]) for r in train]
        vals = [v for v in vals if v == v]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        st[g] = (m, sd)
    return st


def d2h(goals, st):
    s = 0.0
    for g in GOALS:
        m, sd = st[g]
        z = max(-3.0, min(3.0, (goals[g] - m) / (sd + 1e-32)))
        n = 1 / (1 + math.exp(-1.702 * z))
        s += (n - HEAVEN[g]) ** 2
    return math.sqrt(s / 3)


def cohen_etc(xs, ys):
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    vy = sum((y - my) ** 2 for y in ys) / (m - 1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    c = abs(mx - my) / (math.sqrt(pv) + 1e-32)
    cl = abs(sum((x > y) - (x < y) for x in xs for y in ys)) / (n * m)
    ks = 0.0
    for v in set(xs) | set(ys):
        d = abs(sum(x <= v for x in xs) / n - sum(y <= v for y in ys) / m)
        ks = max(ks, d)
    ks /= math.sqrt(1 / n + 1 / m)
    return {"cohen": c, "cliffs": cl, "ks": ks,
            "same": bool(c <= 0.35 and cl <= 0.197 and ks <= 1.36)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()
    with open(a.train, newline="") as f:
        rd = list(csv.reader(f))
    th, trows = rd[0], [r for r in rd[1:] if r]
    st = stats(trows, th)
    with open(a.test) as f:
        data = json.load(f)
    header, rows, eps = data["header"], data["rows"], data["episodes"]

    tr_d = []
    for r in trows:
        tr_d.append(d2h({g: fnum(r[th.index(g)]) for g in GOALS}, st))
    target = sorted(tr_d)[len(tr_d) // 4]

    cats = sorted({r[th.index("origin")] for r in trows})
    med = {}
    for c in NUM:
        v = [fnum(r[th.index(c)]) for r in trows]
        v = [x for x in v if x == x]
        med[c] = float(np.median(v)) if v else 0.0

    def feat(r, h):
        out = []
        for c in NUM:
            x = fnum(r[h.index(c)])
            out.append(med[c] if x != x else x)
        o = r[h.index("origin")]
        out += [1.0 if o == k else 0.0 for k in cats]
        return out

    Xtr = np.array([feat(r, th) for r in trows])
    ytr = np.array(tr_d)
    Xte = np.array([feat(r, header) for r in rows])

    def ask(ep, strat, idx):
        print(json.dumps({"episode": ep, "strategy": strat, "inspect": idx}), flush=True)
        resp = json.loads(sys.stdin.readline())
        return d2h({g: float(resp["goals"][g]) for g in GOALS}, st)

    def adaptive(ep, pool):
        obs = {}
        while len(obs) < 10:
            if obs:
                ids = list(obs)
                X = np.vstack([Xtr, Xte[ids]])
                y = np.concatenate([ytr, [obs[i] for i in ids]])
                w = np.concatenate([np.ones(len(ytr)), np.full(len(ids), 5.0)])
            else:
                X, y, w = Xtr, ytr, np.ones(len(ytr))
            m = RandomForestRegressor(n_estimators=60, min_samples_leaf=2,
                                      random_state=0, n_jobs=1)
            m.fit(X, y, sample_weight=w)
            rest = [i for i in pool if i not in obs]
            p = m.predict(Xte[rest])
            k = int(np.argmin(p))
            i = rest[k]
            obs[i] = ask(ep, "adaptive", i)
        return list(obs.values())

    results = []
    for e, E in enumerate(eps):
        ad = adaptive(e, E["pool"])
        rv = [ask(e, "random", i) for i in E["random_order"][:10]]

        def summ(v):
            lt = 11
            for k, x in enumerate(v):
                if round(x, 12) <= round(target, 12):
                    lt = k + 1
                    break
            return {"best5": min(v[:5]), "best10": min(v), "labels_to_target": lt}
        results.append({"episode": e, "adaptive": summ(ad), "random": summ(rv)})

    comps = {}
    for k in ["best5", "best10", "labels_to_target"]:
        xs = [float(r["adaptive"][k]) for r in results]
        ys = [float(r["random"][k]) for r in results]
        comps[k] = cohen_etc(xs, ys)
    print(json.dumps({"results": results, "comparisons": comps}), flush=True)


main()
