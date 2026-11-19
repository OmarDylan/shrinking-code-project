import sys, json, csv, math, argparse

GOALS = ["Lbs-", "Acc+", "Mpg+"]
HEAVEN = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUMS = ["Clndrs", "Volume", "Model"]


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd


def norm(v, ms):
    m, sd = ms
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))


def cohen(xs, ys):
    n, m = len(xs), len(ys)
    mx, vx = stats(xs)
    my, vy = stats(ys)
    vx *= vx
    vy *= vy
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    return abs(mx - my) / (math.sqrt(pv) + 1e-32)


def cliffs(xs, ys):
    s = 0
    for x in xs:
        for y in ys:
            s += (x > y) - (x < y)
    return abs(s) / (len(xs) * len(ys))


def ks(xs, ys):
    n, m = len(xs), len(ys)
    d = 0
    for v in set(xs) | set(ys):
        d = max(d, abs(sum(x <= v for x in xs) / n - sum(y <= v for y in ys) / m))
    return d / math.sqrt(1 / n + 1 / m)


def compare(xs, ys):
    c, l, k = cohen(xs, ys), cliffs(xs, ys), ks(xs, ys)
    return {"cohen": c, "cliffs": l, "ks": k, "same": bool(c <= 0.35 and l <= 0.197 and k <= 1.36)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()

    with open(a.train, newline="") as f:
        rd = list(csv.reader(f))
    th = rd[0]
    trows = [r for r in rd[1:] if r]
    ti = {h: i for i, h in enumerate(th)}
    gst = {g: stats([float(r[ti[g]]) for r in trows]) for g in GOALS}
    nst = {}
    for c in NUMS:
        vs = [fnum(r[ti[c]]) for r in trows]
        nst[c] = stats([v for v in vs if v is not None])

    def d2h(vals):
        s = sum((norm(vals[g], gst[g]) - HEAVEN[g]) ** 2 for g in GOALS)
        return math.sqrt(s / 3)

    tscore = [d2h({g: float(r[ti[g]]) for g in GOALS}) for r in trows]
    target = sorted(tscore)[len(trows) // 4]

    def feat(get):
        out = []
        for c in NUMS:
            v = fnum(get(c))
            out.append(norm(v, nst[c]) if v is not None else 0.5)
        o = get("origin")
        return out, o

    tfeat = [feat(lambda c, r=r: r[ti[c]]) for r in trows]

    def dist(A, B):
        s = sum((x - y) ** 2 for x, y in zip(A[0], B[0]))
        s += 0.0 if A[1] == B[1] and A[1] != "?" else 1.0
        return math.sqrt(s / 4)

    def knn(F):
        ds = sorted((round(dist(F, t), 12), i) for i, t in enumerate(tfeat))[:5]
        return sum(tscore[i] for _, i in ds) / len(ds)

    with open(a.test) as f:
        data = json.load(f)
    hd = data["header"]
    hi = {h: i for i, h in enumerate(hd)}
    rows = data["rows"]
    fcache = {}
    pcache = {}

    def F(i):
        if i not in fcache:
            fcache[i] = feat(lambda c: rows[i][hi[c]])
        return fcache[i]

    def P(i):
        if i not in pcache:
            pcache[i] = knn(F(i))
        return pcache[i]

    def inspect(ep, strat, idx):
        print(json.dumps({"episode": ep, "strategy": strat, "inspect": idx}), flush=True)
        resp = json.loads(sys.stdin.readline())
        g = resp["goals"]
        return d2h({k: float(g[k]) for k in GOALS})

    def adaptive(ep, pool):
        obs = {}
        order = []
        while len(order) < 10:
            best = None
            for c in pool:
                if c in obs:
                    continue
                adj = 0.0
                if obs:
                    sw = 0.0
                    for o, v in obs.items():
                        w = math.exp(-dist(F(c), F(o)) / 0.2)
                        adj += w * (v - P(o))
                        sw += w
                    adj /= sw + 0.5
                key = (P(c) + adj, c)
                if best is None or key < best:
                    best = key
            c = best[1]
            obs[c] = inspect(ep, "adaptive", c)
            order.append(obs[c])
        return order

    def summ(v):
        lt = 11
        for i, x in enumerate(v):
            if round(x, 12) <= round(target, 12):
                lt = i + 1
                break
        return {"best5": min(v[:5]), "best10": min(v), "labels_to_target": lt}

    results = []
    for e, ep in enumerate(data["episodes"]):
        ad = adaptive(e, ep["pool"])
        rv = [inspect(e, "random", i) for i in ep["random_order"][:10]]
        results.append({"episode": e, "adaptive": summ(ad), "random": summ(rv)})
    comps = {}
    for m in ["best5", "best10", "labels_to_target"]:
        comps[m] = compare([float(r["adaptive"][m]) for r in results],
                           [float(r["random"][m]) for r in results])
    print(json.dumps({"results": results, "comparisons": comps}), flush=True)


main()
