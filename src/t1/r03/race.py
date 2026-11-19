import sys, json, csv, math

def main():
    a = sys.argv
    train_p = a[a.index('--train') + 1]
    test_p = a[a.index('--test') + 1]
    with open(train_p, newline='') as f:
        r = list(csv.reader(f))
    hdr, tr = r[0], r[1:]
    col = {h.rstrip('+-'): i for i, h in enumerate(hdr)}
    goals = [('Lbs', 0), ('Acc', 1), ('Mpg', 1)]
    gidx = [col[g] for g, _ in goals]
    st = []
    for gi in gidx:
        v = [float(x[gi]) for x in tr]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((t - m) ** 2 for t in v) / (len(v) - 1))
        st.append((m, sd))

    def nrm(v, m, sd):
        z = max(-3, min(3, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    def d2h(vals):
        s = 0
        for v, (m, sd), (_, h) in zip(vals, st, goals):
            s += (nrm(v, m, sd) - h) ** 2
        return math.sqrt(s / 3)

    tscore = [d2h([float(x[gi]) for gi in gidx]) for x in tr]
    target = sorted(tscore)[len(tr) // 4]

    nums = ['Clndrs', 'Volume', 'Model']
    nst = {}
    for n in nums:
        v = [float(x[col[n]]) for x in tr if x[col[n]] != '?']
        m = sum(v) / len(v)
        nst[n] = (m, math.sqrt(sum((t - m) ** 2 for t in v) / (len(v) - 1)))
    oc = col['origin']

    def feat(row):
        f = []
        for n in nums:
            x = row[col[n]]
            f.append(None if x == '?' else nrm(float(x), *nst[n]))
        o = row[oc]
        f.append(None if o == '?' else o)
        return f

    def dist(p, q):
        s = c = 0
        for i, (x, y) in enumerate(zip(p, q)):
            if x is None or y is None:
                continue
            g = (x - y) if i < 3 else (0 if x == y else 1)
            s += g * g
            c += 1
        return math.sqrt(s / c) if c else 1.0

    tf = [feat(x) for x in tr]

    def predict(f):
        ds = sorted((round(dist(f, g), 12), i) for i, g in enumerate(tf))[:5]
        return sum(tscore[i] for _, i in ds) / len(ds)

    data = json.loads(open(test_p).read())
    th = data['header']
    tcol = {h.rstrip('+-'): i for i, h in enumerate(th)}
    rows = data['rows']

    def tfeat(row):
        f = []
        for n in nums:
            x = row[tcol[n]]
            f.append(None if x == '?' else nrm(float(x), *nst[n]))
        o = row[tcol['origin']]
        f.append(None if o == '?' else o)
        return f

    cache = {}

    def info(i):
        if i not in cache:
            f = tfeat(rows[i])
            cache[i] = (f, predict(f))
        return cache[i]

    def inspect(ep, strat, i):
        print(json.dumps({"episode": ep, "strategy": strat, "inspect": i}), flush=True)
        resp = json.loads(sys.stdin.readline())
        g = resp['goals']
        return d2h([g['Lbs-'], g['Acc+'], g['Mpg+']])

    def summarize(obs):
        lt = 11
        for k, v in enumerate(obs):
            if round(v, 12) <= round(target, 12):
                lt = k + 1
                break
        return {"best5": min(obs[:5]), "best10": min(obs), "labels_to_target": lt}

    results = []
    for ep, e in enumerate(data['episodes']):
        pool = list(e['pool'])
        left = set(pool)
        seen = []  # (idx, obs)
        obs = []
        for _ in range(10):
            best = None
            for c in left:
                f, p = info(c)
                num = den = 0
                for j, o in seen:
                    w = math.exp(-dist(f, info(j)[0]) / 0.2)
                    num += w * (o - info(j)[1])
                    den += w
                adj = p + (num / (den + 1.0) if seen else 0)
                key = (round(adj, 12), c)
                if best is None or key < best[0]:
                    best = (key, c)
            c = best[1]
            left.discard(c)
            o = inspect(ep, "adaptive", c)
            seen.append((c, o))
            obs.append(o)
        ad = summarize(obs)
        robs = [inspect(ep, "random", i) for i in e['random_order'][:10]]
        results.append({"episode": ep, "adaptive": ad, "random": summarize(robs)})

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

    comp = {}
    for k in ("best5", "best10", "labels_to_target"):
        comp[k] = cert([float(x["adaptive"][k]) for x in results],
                       [float(x["random"][k]) for x in results])
    print(json.dumps({"results": results, "comparisons": comp}), flush=True)

main()
