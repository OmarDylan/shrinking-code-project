import sys, json, csv, math
import numpy as np
from sklearn.ensemble import RandomForestRegressor

H = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
G = list(H)
NUM = ["Clndrs", "Volume", "Model"]


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def main():
    a = sys.argv
    train = a[a.index("--train") + 1]
    test = a[a.index("--test") + 1]
    with open(train, newline="") as f:
        rd = list(csv.reader(f))
    th, trows = rd[0], rd[1:]
    ti = {n: i for i, n in enumerate(th)}
    st = {}
    for g in G:
        v = np.array([float(r[ti[g]]) for r in trows])
        st[g] = (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0)

    def d2h(goals):
        s = 0.0
        for g in G:
            m, sd = st[g]
            z = max(-3, min(3, (goals[g] - m) / (sd + 1e-32)))
            n = 1 / (1 + math.exp(-1.702 * z))
            s += (n - H[g]) ** 2
        return math.sqrt(s / 3)

    ytr = np.array([d2h({g: float(r[ti[g]]) for g in G}) for r in trows])
    target = sorted(ytr)[len(ytr) // 4]
    cats = sorted({r[ti["origin"]] for r in trows})
    means = {}
    for c in NUM:
        v = np.array([fnum(r[ti[c]]) for r in trows])
        means[c] = np.nanmean(v) if not np.all(np.isnan(v)) else 0.0

    def feat(get):
        x = []
        for c in NUM:
            v = fnum(get(c))
            x.append(means[c] if math.isnan(v) else v)
        o = get("origin")
        x += [1.0 if o == k else 0.0 for k in cats]
        return x

    Xtr = np.array([feat(lambda c, r=r: r[ti[c]]) for r in trows])

    with open(test) as f:
        data = json.load(f)
    hd = data["header"]
    hi = {n: i for i, n in enumerate(hd)}
    rows = data["rows"]
    Xall = np.array([feat(lambda c, r=r: r[hi[c]]) for r in rows])

    def inspect(ep, strat, idx):
        print(json.dumps({"episode": ep, "strategy": strat, "inspect": idx}), flush=True)
        resp = json.loads(sys.stdin.readline())
        return d2h({g: float(resp["goals"][g]) for g in G})

    def adaptive(ep, pool):
        obs, seen = [], []
        for _ in range(10):
            if seen:
                X = np.vstack([Xtr] + [Xall[seen]] * 5)
                y = np.concatenate([ytr] + [np.array(obs)] * 5)
            else:
                X, y = Xtr, ytr
            m = RandomForestRegressor(60, random_state=0, min_samples_leaf=2).fit(X, y)
            cand = [i for i in pool if i not in seen]
            p = m.predict(Xall[cand])
            k = cand[int(np.argmin(p))]
            seen.append(k)
            obs.append(inspect(ep, "adaptive", k))
        return obs

    def rnd(ep, order):
        return [inspect(ep, "random", i) for i in order[:10]]

    def summ(o):
        lt = 11
        for i, v in enumerate(o):
            if round(v, 12) <= round(target, 12):
                lt = i + 1
                break
        return {"best5": min(o[:5]), "best10": min(o), "labels_to_target": lt}

    results = []
    for e, ep in enumerate(data["episodes"]):
        ad = adaptive(e, ep["pool"])
        rn = rnd(e, ep["random_order"])
        results.append({"episode": e, "adaptive": summ(ad), "random": summ(rn)})

    def cert(xs, ys):
        n, m = len(xs), len(ys)
        mx, my = sum(xs) / n, sum(ys) / m
        vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
        vy = sum((y - my) ** 2 for y in ys) / (m - 1)
        pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
        co = abs(mx - my) / (math.sqrt(pv) + 1e-32)
        cl = abs(sum((x > y) - (x < y) for x in xs for y in ys)) / (n * m)
        ks = max(abs(sum(x <= v for x in xs) / n - sum(y <= v for y in ys) / m)
                 for v in set(xs) | set(ys)) / math.sqrt(1 / n + 1 / m)
        return {"cohen": co, "cliffs": cl, "ks": ks,
                "same": bool(co <= 0.35 and cl <= 0.197 and ks <= 1.36)}

    comps = {k: cert([r["adaptive"][k] for r in results], [r["random"][k] for r in results])
             for k in ["best5", "best10", "labels_to_target"]}
    print(json.dumps({"results": results, "comparisons": comps}), flush=True)


if __name__ == "__main__":
    main()
