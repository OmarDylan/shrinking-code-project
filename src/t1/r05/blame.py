import csv, sys, json, math

def main():
    a = sys.argv[1:]
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]

    def load(p):
        with open(p, newline="") as f:
            r = csv.reader(f)
            h = [x.strip() for x in next(r)]
            rows = [[c.strip() for c in row] for row in r if row]
        return h, [dict(zip(h, row)) for row in rows]

    _, T = load(tr)
    _, E = load(te)
    goals = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
    nums = ["Clndrs", "Volume", "Model"]
    cat = "origin"

    def fl(v):
        if v is None or v == "?" or v == "":
            return None
        try:
            return float(v)
        except ValueError:
            return None

    def stats(vals):
        n = len(vals)
        if n == 0:
            return 0.0, 0.0
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        return m, sd

    def mk(m, sd):
        def f(v):
            z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
            return 1 / (1 + math.exp(-1.702 * z))
        return f

    gn = {}
    for g in goals:
        gn[g] = mk(*stats([fl(r[g]) for r in T]))
    nn = {}
    for c in nums:
        nn[c] = mk(*stats([x for x in (fl(r[c]) for r in T) if x is not None]))

    def d2h(r):
        return math.sqrt(sum((gn[g](fl(r[g])) - h) ** 2 for g, h in goals.items()) / 3)

    order = sorted(range(len(T)), key=lambda i: (round(d2h(T[i]), 12), i))
    q = max(1, len(T) // 4)
    good = [T[i] for i in order[:q]]
    poor = [T[i] for i in order[-q:]]

    centers = {}
    for c in nums:
        cs = []
        for grp in (good, poor):
            v = [x for x in (fl(r[c]) for r in grp) if x is not None]
            cs.append(nn[c](sum(v) / len(v)) if v else None)
        centers[c] = cs

    def freq(grp):
        vals = [r[cat] for r in grp if r[cat] not in ("?", "")]
        d = {}
        for v in vals:
            d[v] = d.get(v, 0) + 1
        n = len(vals)
        return d, n

    gf, gc = freq(good)
    pf, pc = freq(poor)

    out = []
    for r in E:
        gs = {}
        for g, h in goals.items():
            gs[g] = abs(gn[g](fl(r[g])) - h)
        fs = {}
        for c in nums:
            x = fl(r[c])
            gc_, pc_ = centers[c]
            if x is None or gc_ is None or pc_ is None:
                fs[c] = 0.0
            else:
                z = nn[c](x)
                fs[c] = max(0.0, abs(z - gc_) - abs(z - pc_))
        v = r.get(cat, "?")
        if v in ("?", ""):
            fs[cat] = 0.0
        else:
            pfq = pf.get(v, 0) / pc if pc else 0.0
            gfq = gf.get(v, 0) / gc if gc else 0.0
            fs[cat] = max(0.0, pfq - gfq)
        key = lambda d: sorted(d, key=lambda k: (-round(d[k], 10), k))
        out.append({"goals": gs, "features": fs,
                    "goal_order": key(gs), "feature_order": key(fs)})
    print(json.dumps({"diagnoses": out}))

main()
