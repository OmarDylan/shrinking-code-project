import sys, csv, json, math
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

GOALS = ["Lbs-", "Acc+", "Mpg+"]
HEAVEN = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUMS = ["Clndrs", "Volume", "Model"]


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def certify(xs, ys):
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((a - mx) ** 2 for a in xs) / (n - 1)
    vy = sum((a - my) ** 2 for a in ys) / (m - 1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (math.sqrt(pv) + 1e-32)
    s = sum((a > b) - (a < b) for a in xs for b in ys)
    cl = abs(s) / (n * m)
    ks = max(abs(sum(a <= v for a in xs) / n - sum(b <= v for b in ys) / m)
             for v in set(xs) | set(ys)) / math.sqrt(1 / n + 1 / m)
    return {"cohen": cohen, "cliffs": cl, "ks": ks,
            "same": bool(cohen <= 0.35 and cl <= 0.197 and ks <= 1.36)}


def main():
    a = sys.argv
    train_p = a[a.index("--train") + 1]
    test_p = a[a.index("--test") + 1]
    with open(train_p, newline="") as f:
        rd = list(csv.reader(f))
    th = [h.strip() for h in rd[0]]
    trows = [r for r in rd[1:] if r]
    ti = {h: i for i, h in enumerate(th)}
    stats = {}
    for g in GOALS:
        vals = [float(r[ti[g]]) for r in trows]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))
        stats[g] = (mu, sd)

    def d2h(goals):
        t = 0.0
        for g in GOALS:
            mu, sd = stats[g]
            z = max(-3, min(3, (float(goals[g]) - mu) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            t += (nv - HEAVEN[g]) ** 2
        return math.sqrt(t / 3)

    tscores = [d2h({g: r[ti[g]] for g in GOALS}) for r in trows]
    target = sorted(tscores)[len(trows) // 4]

    cats = sorted({r[ti["origin"]] for r in trows if r[ti["origin"]] != "?"})
    med = {}
    for c in NUMS:
        v = [num(r[ti[c]]) for r in trows]
        v = [x for x in v if x is not None]
        med[c] = float(np.median(v)) if v else 0.0

    def feat(get):
        out = []
        for c in NUMS:
            x = num(get(c))
            out.append(med[c] if x is None else x)
        o = get("origin")
        out += [1.0 if o == k else 0.0 for k in cats]
        return out

    Xtr = np.array([feat(lambda c, r=r: r[ti[c]]) for r in trows])
    ytr = np.array(tscores)

    with open(test_p) as f:
        data = json.load(f)
    hdr = data["header"]
    hi = {h: i for i, h in enumerate(hdr)}
    rows = data["rows"]

    def rowfeat(i):
        return feat(lambda c: rows[i][hi[c]])

    def inspect(ep, strat, idx):
        print(json.dumps({"episode": ep, "strategy": strat, "inspect": idx}))
        sys.stdout.flush()
        resp = json.loads(sys.stdin.readline())
        return d2h(resp["goals"])

    def summarize(obs):
        lab = 11
        for k, v in enumerate(obs):
            if round(v, 12) <= round(target, 12):
                lab = k + 1
                break
        return {"best5": min(obs[:5]), "best10": min(obs), "labels_to_target": lab}

    results = []
    for e, ep in enumerate(data["episodes"]):
        pool = list(ep["pool"])
        P = np.array([rowfeat(i) for i in pool])
        remaining = list(range(len(pool)))
        ox, oy = [], []
        obs = []
        for step in range(10):
            if ox:
                X = np.vstack([Xtr] + [np.array(ox)] * 3)
                y = np.concatenate([ytr, np.tile(oy, 3)])
            else:
                X, y = Xtr, ytr
            m = ExtraTreesRegressor(n_estimators=60, min_samples_leaf=2,
                                    random_state=0, n_jobs=1).fit(X, y)
            pr = m.predict(P[remaining])
            j = remaining[int(np.argmin(pr))]
            remaining.remove(j)
            v = inspect(e, "adaptive", pool[j])
            obs.append(v)
            ox.append(P[j].tolist())
            oy.append(v)
        ad = summarize(obs)
        robs = [inspect(e, "random", i) for i in ep["random_order"][:10]]
        results.append({"episode": e, "adaptive": ad, "random": summarize(robs)})

    comps = {}
    for k in ["best5", "best10", "labels_to_target"]:
        comps[k] = certify([float(r["adaptive"][k]) for r in results],
                           [float(r["random"][k]) for r in results])
    print(json.dumps({"results": results, "comparisons": comps}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
