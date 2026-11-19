import sys, json, csv, math

def main():
    a = sys.argv
    train = a[a.index("--train") + 1]
    test = a[a.index("--test") + 1]
    with open(train, encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r]
    th, trows = [h.strip() for h in rows[0]], rows[1:]
    G = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
    gi = {g: th.index(g) for g in G}
    st = {}
    for g in G:
        v = [float(r[gi[g]]) for r in trows]
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))
        st[g] = (mu, sd)

    def nm(g, v):
        mu, sd = st[g]
        z = max(-3, min(3, (v - mu) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    def d2h(goals):
        return math.sqrt(sum((nm(g, float(goals[g])) - h) ** 2 for g, h in G.items()) / 3)

    tr_d = [d2h({g: r[gi[g]] for g in G}) for r in trows]
    target = sorted(tr_d)[len(tr_d) // 4]

    with open(test) as f:
        T = json.load(f)
    hd = [h.strip() for h in T["header"]]
    R = T["rows"]
    NUM = ["Clndrs", "Volume", "Model"]
    ni = [th.index(c) for c in NUM]
    ti = [hd.index(c) for c in NUM]
    oi_tr, oi_te = th.index("origin"), hd.index("origin")
    ns = []
    for i in ni:
        v = [float(r[i]) for r in trows]
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1))
        ns.append((mu, sd))

    def feat(nums, o):
        out = []
        for x, (mu, sd) in zip(nums, ns):
            try:
                x = float(x)
                z = max(-3, min(3, (x - mu) / (sd + 1e-32)))
                out.append(1 / (1 + math.exp(-1.702 * z)))
            except (ValueError, TypeError):
                out.append(None)
        out.append(None if o in ("?", "", None) else str(o).strip())
        return out

    def fdist(p, q):
        s = 0
        for x, y in zip(p, q):
            if x is None or y is None:
                s += 1
            elif isinstance(x, str):
                s += 0 if x == y else 1
            else:
                s += (x - y) ** 2
        return math.sqrt(s / len(p))

    TF = [feat([r[i] for i in ni], r[oi_tr]) for r in trows]
    cache = {}
    feats = {}

    def F(i):
        if i not in feats:
            r = R[i]
            feats[i] = feat([r[j] for j in ti], r[oi_te])
        return feats[i]

    def base(i):
        if i not in cache:
            f = F(i)
            ds = sorted((fdist(f, t), k) for k, t in enumerate(TF))[:5]
            cache[i] = sum(tr_d[k] for _, k in ds) / len(ds)
        return cache[i]

    def ask(e, s, i):
        print(json.dumps({"episode": e, "strategy": s, "inspect": i}), flush=True)
        resp = json.loads(sys.stdin.readline())
        return d2h(resp["goals"])

    def adaptive(e, pool):
        obs = []
        left = list(pool)
        for _ in range(10):
            def sc(i):
                p = base(i)
                if obs:
                    w = [(1 / (fdist(F(i), F(j)) + 0.1), y - base(j)) for j, y in obs]
                    p += sum(a_ * b_ for a_, b_ in w) / sum(a_ for a_, _ in w)
                return (p, i)
            i = min(left, key=sc)
            left.remove(i)
            obs.append((i, ask(e, "adaptive", i)))
        return [y for _, y in obs]

    def metrics(ys):
        lt = 11
        for k, y in enumerate(ys):
            if round(y, 12) <= round(target, 12):
                lt = k + 1
                break
        return {"best5": min(ys[:5]), "best10": min(ys), "labels_to_target": lt}

    res = []
    for e, ep in enumerate(T["episodes"]):
        ad = adaptive(e, ep["pool"])
        rd = [ask(e, "random", i) for i in ep["random_order"][:10]]
        res.append({"episode": e, "adaptive": metrics(ad), "random": metrics(rd)})

    def cmp(xs, ys):
        n, m = len(xs), len(ys)
        mx, my = sum(xs) / n, sum(ys) / m
        vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
        vy = sum((y - my) ** 2 for y in ys) / (m - 1)
        pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
        c = abs(mx - my) / (math.sqrt(pv) + 1e-32)
        cl = abs(sum((x > y) - (x < y) for x in xs for y in ys)) / (n * m)
        k = max(abs(sum(x <= v for x in xs) / n - sum(y <= v for y in ys) / m)
                for v in set(xs) | set(ys)) / math.sqrt(1 / n + 1 / m)
        return {"cohen": c, "cliffs": cl, "ks": k,
                "same": c <= 0.35 and cl <= 0.197 and k <= 1.36}

    comp = {}
    for m in ("best5", "best10", "labels_to_target"):
        comp[m] = cmp([r["adaptive"][m] for r in res], [r["random"][m] for r in res])
    print(json.dumps({"results": res, "comparisons": comp}), flush=True)

if __name__ == "__main__":
    main()
