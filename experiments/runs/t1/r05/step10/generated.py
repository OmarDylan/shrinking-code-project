import sys, json, csv, math

def main():
    a = sys.argv
    train_p = a[a.index("--train") + 1]
    test_p = a[a.index("--test") + 1]
    with open(train_p, newline="") as f:
        rd = list(csv.reader(f))
    th = [h.strip() for h in rd[0]]
    trows = [r for r in rd[1:] if r]
    ti = {h: i for i, h in enumerate(th)}
    goals = ["Lbs-", "Acc+", "Mpg+"]
    heaven = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
    st = {}
    for g in goals:
        v = [float(r[ti[g]]) for r in trows]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
        st[g] = (m, sd)

    def norm(v, m, sd):
        z = max(-3, min(3, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    def d2h(gv):
        return math.sqrt(sum((norm(gv[g], *st[g]) - heaven[g]) ** 2 for g in goals) / 3)

    tr_d = [d2h({g: float(r[ti[g]]) for g in goals}) for r in trows]
    target = sorted(tr_d)[len(tr_d) // 4]

    nums = ["Clndrs", "Volume", "Model"]

    def fv(x):
        try:
            return float(x)
        except Exception:
            return None

    nst = {}
    for c in nums:
        v = [fv(r[ti[c]]) for r in trows]
        v = [x for x in v if x is not None]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
        nst[c] = (m, sd)

    def feat(get):
        out = []
        for c in nums:
            x = fv(get(c))
            out.append(None if x is None else norm(x, *nst[c]))
        o = get("origin")
        out.append(None if o is None or str(o).strip() == "?" else str(o).strip())
        return out

    tf = [feat(lambda c, r=r: r[ti[c]]) for r in trows]

    def dist(A, B):
        s = 0.0
        n = 0
        for i in range(4):
            x, y = A[i], B[i]
            if x is None or y is None:
                continue
            s += (x - y) ** 2 if i < 3 else (0 if x == y else 1)
            n += 1
        return math.sqrt(s / n) if n else 1.0

    def predict(F):
        ds = sorted((round(dist(F, t), 12), i) for i, t in enumerate(tf))[:5]
        return sum(tr_d[i] for _, i in ds) / len(ds)

    with open(test_p) as f:
        data = json.load(f)
    hd = data["header"]
    hi = {h: i for i, h in enumerate(hd)}
    rows = data["rows"]
    tef = {}

    def F(i):
        if i not in tef:
            r = rows[i]
            tef[i] = feat(lambda c: r[hi[c]] if c in hi else None)
        return tef[i]

    pc = {}

    def P(i):
        if i not in pc:
            pc[i] = predict(F(i))
        return pc[i]

    def inspect(ep, strat, idx):
        print(json.dumps({"episode": ep, "strategy": strat, "inspect": idx}), flush=True)
        resp = json.loads(sys.stdin.readline())
        return d2h({g: float(resp["goals"][g]) for g in goals})

    def adaptive(ep, pool):
        obs = []
        left = list(pool)
        while len(obs) < 10:
            best = None
            for c in left:
                adj = P(c)
                if obs:
                    ws = [(math.exp(-dist(F(c), F(o)) / 0.2), v - P(o)) for o, v in obs]
                    sw = sum(w for w, _ in ws)
                    adj += sum(w * r for w, r in ws) / (sw + 1.0)
                k = (adj, c)
                if best is None or k < best:
                    best = k
            c = best[1]
            left.remove(c)
            obs.append((c, inspect(ep, "adaptive", c)))
        return [v for _, v in obs]

    def stats(xs):
        n = len(xs)
        m = sum(xs) / n
        return m, sum((x - m) ** 2 for x in xs) / (n - 1)

    def cert(xs, ys):
        n, m = len(xs), len(ys)
        mx, vx = stats(xs)
        my, vy = stats(ys)
        pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
        co = abs(mx - my) / (math.sqrt(pv) + 1e-32)
        cl = abs(sum((x > y) - (x < y) for x in xs for y in ys)) / (n * m)
        ks = max(abs(sum(x <= v for x in xs) / n - sum(y <= v for y in ys) / m)
                 for v in set(xs) | set(ys)) / math.sqrt(1 / n + 1 / m)
        return {"cohen": co, "cliffs": cl, "ks": ks,
                "same": bool(co <= 0.35 and cl <= 0.197 and ks <= 1.36)}

    results = []
    for e, ep in enumerate(data["episodes"]):
        av = adaptive(e, ep["pool"])
        rv = [inspect(e, "random", i) for i in ep["random_order"][:10]]

        def summ(v):
            lt = 11
            for k, x in enumerate(v):
                if round(x, 12) <= round(target, 12):
                    lt = k + 1
                    break
            return {"best5": min(v[:5]), "best10": min(v), "labels_to_target": lt}

        results.append({"episode": e, "adaptive": summ(av), "random": summ(rv)})
    comp = {}
    for k in ["best5", "best10", "labels_to_target"]:
        comp[k] = cert([r["adaptive"][k] for r in results], [r["random"][k] for r in results])
    print(json.dumps({"results": results, "comparisons": comp}), flush=True)

main()
