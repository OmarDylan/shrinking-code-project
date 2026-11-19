import csv, sys, json, math

def load(p):
    with open(p, newline='') as f:
        r = csv.reader(f)
        h = [x.strip() for x in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return h, rows

def stats(v):
    n = len(v)
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, sd

def mk(m, sd):
    def f(v):
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))
    return f

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    h, T = load(tr)
    h2, E = load(te)
    ix = {n: i for i, n in enumerate(h)}
    ix2 = {n: i for i, n in enumerate(h2)}
    goals = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]
    nums = ["Clndrs", "Volume", "Model"]
    cat = "origin"
    gn = {}
    for g, _ in goals:
        gn[g] = mk(*stats([float(r[ix[g]]) for r in T]))
    def d2h(r, ixm):
        return math.sqrt(sum((gn[g](float(r[ixm[g]])) - hv) ** 2 for g, hv in goals) / 3)
    order = sorted(range(len(T)), key=lambda i: (round(d2h(T[i], ix), 12), i))
    q = max(1, len(T) // 4)
    good = [T[i] for i in order[:q]]
    poor = [T[i] for i in order[-q:]]
    def val(r, ixm, c):
        s = r[ixm[c]]
        return None if s == '?' or s == '' else s
    nn, cen = {}, {}
    for c in nums:
        col = [float(val(r, ix, c)) for r in T if val(r, ix, c) is not None]
        nn[c] = mk(*stats(col)) if col else None
        cs = []
        for grp in (good, poor):
            vs = [float(val(r, ix, c)) for r in grp if val(r, ix, c) is not None]
            cs.append(nn[c](sum(vs) / len(vs)) if vs and nn[c] else None)
        cen[c] = cs
    def freq(grp):
        vs = [val(r, ix, cat) for r in grp]
        vs = [v for v in vs if v is not None]
        d = {}
        for v in vs:
            d[v] = d.get(v, 0) + 1
        return d, len(vs)
    gf, gc = freq(good)
    pf, pc = freq(poor)
    out = []
    for r in E:
        gs = {g: abs(gn[g](float(r[ix2[g]])) - hv) for g, hv in goals}
        fs = {}
        for c in nums:
            s = val(r, ix2, c)
            gc_, pc_ = cen[c]
            if s is None or gc_ is None or pc_ is None:
                fs[c] = 0.0
            else:
                x = nn[c](float(s))
                fs[c] = max(0.0, abs(x - gc_) - abs(x - pc_))
        s = val(r, ix2, cat)
        if s is None:
            fs[cat] = 0.0
        else:
            pfr = pf.get(s, 0) / pc if pc else 0.0
            gfr = gf.get(s, 0) / gc if gc else 0.0
            fs[cat] = max(0.0, pfr - gfr)
        so = lambda d: sorted(d, key=lambda k: (-round(d[k], 10), k))
        out.append({"goals": gs, "features": fs, "goal_order": so(gs), "feature_order": so(fs)})
    print(json.dumps({"diagnoses": out}))

main()
