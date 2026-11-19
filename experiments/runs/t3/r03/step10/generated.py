import sys, json, csv, math, argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor

GOALS = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUM = ["Clndrs", "Volume", "Model"]


def fl(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def load(path):
    with open(path, newline="") as f:
        r = list(csv.reader(f))
    return r[0], r[1:]


def make_norm(h, rows):
    st = {}
    for g in GOALS:
        c = h.index(g)
        v = np.array([float(r[c]) for r in rows])
        st[g] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)

    def d2h(goals):
        s = 0.0
        for g, hv in GOALS.items():
            m, sd = st[g]
            z = max(-3, min(3, (float(goals[g]) - m) / (sd + 1e-32)))
            s += (1 / (1 + math.exp(-1.702 * z)) - hv) ** 2
        return math.sqrt(s / 3)
    return d2h


def stats(xs, ys):
    n, m = len(xs), len(ys)
    x = np.array(xs, float)
    y = np.array(ys, float)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(x.mean() - y.mean()) / (math.sqrt(pv) + 1e-32)
    cl = abs(sum(np.sign(a - b) for a in x for b in y)) / (n * m)
    ks = max(abs((x <= v).sum() / n - (y <= v).sum() / m)
             for v in set(x) | set(y)) / math.sqrt(1 / n + 1 / m)
    return {"cohen": float(cohen), "cliffs": float(cl), "ks": float(ks),
            "same": bool(cohen <= 0.35 and cl <= 0.197 and ks <= 1.36)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()
    h, tr = load(a.train)
    d2h = make_norm(h, tr)
    td = np.array([d2h({g: r[h.index(g)] for g in GOALS}) for r in tr])
    target = sorted(td)[len(tr) // 4]

    ci = [h.index(c) for c in NUM]
    oi = h.index("origin")
    cats = sorted({r[oi] for r in tr if r[oi] != "?"})
    tn = np.array([[fl(r[c]) for c in ci] for r in tr])
    med = np.nanmedian(tn, axis=0)

    def feat(r):
        v = [fl(r[c]) for c in ci]
        v = [med[i] if math.isnan(x) else x for i, x in enumerate(v)]
        return v + [1.0 if r[oi] == c else 0.0 for c in cats]

    Xtr = np.array([feat(r) for r in tr])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    rf = RandomForestRegressor(100, random_state=0, n_jobs=1).fit(Xtr, td)

    with open(a.test) as f:
        data = json.load(f)
    th = data["header"]
    rows = data["rows"]
    hm = {c: i for i, c in enumerate(th)}

    def tfeat(r):
        v = [fl(r[hm[c]]) for c in NUM]
        v = [med[i] if math.isnan(x) else x for i, x in enumerate(v)]
        return v + [1.0 if r[hm["origin"]] == c else 0.0 for c in cats]

    X = (np.array([tfeat(r) for r in rows]) - mu) / sd
    P = rf.predict(X * sd + mu)

    def ask(ep, strat, i):
        print(json.dumps({"episode": ep, "strategy": strat, "inspect": int(i)}), flush=True)
        resp = json.loads(sys.stdin.readline())
        return d2h(resp["goals"])

    def summarize(obs):
        lt = 11
        for k, v in enumerate(obs):
            if round(v, 12) <= round(target, 12):
                lt = k + 1
                break
        return {"best5": min(obs[:5]), "best10": min(obs), "labels_to_target": lt}

    results = []
    for e, ep in enumerate(data["episodes"]):
        pool = list(ep["pool"])
        seen, obs = [], []
        while len(seen) < 10:
            best, bs = None, None
            for i in pool:
                if i in seen:
                    continue
                s = P[i]
                if seen:
                    d2 = np.array([((X[i] - X[j]) ** 2).sum() for j in seen])
                    w = np.exp(-d2 / 2.0)
                    res = np.array([obs[k] - P[j] for k, j in enumerate(seen)])
                    s += (w * res).sum() / (w.sum() + 1.0)
                if bs is None or (s, i) < (bs, best):
                    best, bs = i, s
            seen.append(best)
            obs.append(ask(e, "adaptive", best))
        ad = summarize(obs)
        ro = []
        for i in ep["random_order"][:10]:
            ro.append(ask(e, "random", i))
        results.append({"episode": e, "adaptive": ad, "random": summarize(ro)})

    comps = {}
    for m in ["best5", "best10", "labels_to_target"]:
        comps[m] = stats([r["adaptive"][m] for r in results],
                         [r["random"][m] for r in results])
    print(json.dumps({"results": results, "comparisons": comps}), flush=True)


main()
