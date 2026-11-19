import sys, json, csv, random
from math import exp, sqrt

def main():
    a = sys.argv
    train = a[a.index('--train') + 1]
    test = a[a.index('--test') + 1]
    with open(train, encoding='utf-8-sig') as f:
        tr = [r for r in csv.reader(f) if r]
    th, trows = [h.strip() for h in tr[0]], tr[1:]
    G = ['Lbs-', 'Acc+', 'Mpg+']
    st = {}
    for g in G:
        v = [float(r[th.index(g)]) for r in trows]
        m = sum(v) / len(v)
        sd = sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
        st[g] = (m, sd)
    nm = lambda v, m, s: 1 / (1 + exp(-1.702 * max(-3, min(3, (v - m) / (s + 1e-32)))))
    def d2h(goals):
        t = 0
        for g in G:
            h = 0 if g[-1] == '-' else 1
            t += (nm(float(goals[g]), *st[g]) - h) ** 2
        return sqrt(t / 3)
    td = sorted(d2h({g: r[th.index(g)] for g in G}) for r in trows)
    target = round(td[len(td) // 4], 12)
    NX = ['Clndrs', 'Volume', 'Model']
    xs = {}
    for c in NX:
        v = [float(r[th.index(c)]) for r in trows if r[th.index(c)].strip() != '?']
        m = sum(v) / len(v)
        xs[c] = (m, sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)))
    D = json.load(open(test))
    hd, rows = D['header'], D['rows']
    def feat(i):
        r, o = rows[i], []
        for c in NX:
            v = r[hd.index(c)]
            try:
                o.append(nm(float(v), *xs[c]))
            except (ValueError, TypeError):
                o.append(None)
        v = r[hd.index('origin')]
        o.append(None if v in ('?', '') else v)
        return o
    def xd(p, q):
        s = 0
        for k in range(4):
            x, y = p[k], q[k]
            if x is None or y is None: s += 1
            elif k == 3: s += x != y
            else: s += (x - y) ** 2
        return sqrt(s / 4)
    def cen(fs):
        o = []
        for k in range(4):
            v = [f[k] for f in fs if f[k] is not None]
            if not v: o.append(None)
            elif k == 3: o.append(max(set(v), key=v.count))
            else: o.append(sum(v) / len(v))
        return o
    def inspect(e, s, i):
        print(json.dumps({"episode": e, "strategy": s, "inspect": i}), flush=True)
        resp = json.loads(sys.stdin.readline())
        return d2h(resp['goals'])
    res = []
    for e, ep in enumerate(D['episodes']):
        pool = ep['pool']
        rng = random.Random(1234 + e)
        F = {i: feat(i) for i in pool}
        lab = {}
        order = []
        for i in rng.sample(pool, 4):
            lab[i] = inspect(e, 'adaptive', i); order.append(i)
        while len(order) < 10:
            rk = sorted(lab, key=lambda i: (lab[i], i))
            n = max(1, int(sqrt(len(rk))))
            cb = cen([F[i] for i in rk[:n]])
            cr = cen([F[i] for i in rk[n:]]) if len(rk) > n else cb
            nxt = min((i for i in pool if i not in lab),
                      key=lambda i: (xd(F[i], cb) - xd(F[i], cr), i))
            lab[nxt] = inspect(e, 'adaptive', nxt); order.append(nxt)
        av = [lab[i] for i in order]
        rv = [inspect(e, 'random', i) for i in ep['random_order'][:10]]
        def m(v):
            l = next((k + 1 for k, x in enumerate(v) if round(x, 12) <= target), 11)
            return {"best5": min(v[:5]), "best10": min(v), "labels_to_target": l}
        res.append({"episode": e, "adaptive": m(av), "random": m(rv)})
    def cmp(x, y):
        n, k = len(x), len(y)
        mx, my = sum(x) / n, sum(y) / k
        vx = sum((t - mx) ** 2 for t in x) / (n - 1)
        vy = sum((t - my) ** 2 for t in y) / (k - 1)
        pv = ((n - 1) * vx + (k - 1) * vy) / (n + k - 2)
        co = abs(mx - my) / (sqrt(pv) + 1e-32)
        cl = abs(sum((p > q) - (p < q) for p in x for q in y)) / (n * k)
        ksv = max(abs(sum(t <= v for t in x) / n - sum(t <= v for t in y) / k)
                  for v in set(x) | set(y)) / sqrt(1 / n + 1 / k)
        return {"cohen": co, "cliffs": cl, "ks": ksv,
                "same": bool(co <= 0.35 and cl <= 0.197 and ksv <= 1.36)}
    comp = {k: cmp([r['adaptive'][k] for r in res], [r['random'][k] for r in res])
            for k in ("best5", "best10", "labels_to_target")}
    print(json.dumps({"results": res, "comparisons": comp}), flush=True)

main()
