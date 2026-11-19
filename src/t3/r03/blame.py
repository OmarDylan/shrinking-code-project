import csv, sys, json, math

def load(p):
    with open(p, newline='') as f:
        r = csv.reader(f)
        h = [x.strip() for x in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return h, rows

def num(s):
    if s == '?' or s == '':
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
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

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    h, R = load(tr)
    h2, T = load(te)
    gcols = {'Lbs': ('Lbs-', 0.0), 'Acc': ('Acc+', 1.0), 'Mpg': ('Mpg+', 1.0)}
    ix = {x: i for i, x in enumerate(h)}
    ix2 = {x: i for i, x in enumerate(h2)}

    def get(row, idx, name):
        i = idx[name]
        return row[i] if i < len(row) else '?'

    # train rows with fully observed goals
    tg = []
    for row in R:
        g = {k: num(get(row, ix, v[0])) for k, v in gcols.items()}
        if all(x is not None for x in g.values()):
            tg.append((row, g))
    gn = {}
    for k in gcols:
        gn[k] = mk(*stats([g[k] for _, g in tg]))

    def d2h(g):
        return math.sqrt(sum((gn[k](g[k]) - gcols[k][1]) ** 2 for k in gcols) / 3)

    order = sorted(range(len(tg)), key=lambda i: (round(d2h(tg[i][1]), 12), i))
    q = max(1, len(tg) // 4)
    good = [tg[i][0] for i in order[:q]]
    poor = [tg[i][0] for i in order[-q:]]

    nums = ['Clndrs', 'Volume', 'Model']
    nn, ctr = {}, {}
    for c in nums:
        col = [num(get(r, ix, c)) for r, _ in tg]
        nn[c] = mk(*stats([v for v in col if v is not None]))
        cs = []
        for grp in (good, poor):
            vs = [v for v in (num(get(r, ix, c)) for r in grp) if v is not None]
            cs.append(nn[c](sum(vs) / len(vs)) if vs else None)
        ctr[c] = cs

    def freq(grp):
        vals = [get(r, ix, 'origin') for r in grp]
        vals = [v for v in vals if v != '?' and v != '']
        d = {}
        for v in vals:
            d[v] = d.get(v, 0) + 1
        return {k: c / len(vals) for k, c in d.items()}
    gf, pf = freq(good), freq(poor)

    out = []
    for row in T:
        goals = {}
        for k, (nm, hv) in gcols.items():
            v = num(get(row, ix2, nm))
            goals[k] = abs(gn[k](v) - hv) if v is not None else 0.0
        feats = {}
        for c in nums:
            v = num(get(row, ix2, c)) if c in ix2 else None
            gc, pc = ctr[c]
            if v is None or gc is None or pc is None:
                feats[c] = 0.0
            else:
                x = nn[c](v)
                feats[c] = max(0.0, abs(x - gc) - abs(x - pc))
        o = get(row, ix2, 'origin') if 'origin' in ix2 else '?'
        if o == '?' or o == '':
            feats['origin'] = 0.0
        else:
            feats['origin'] = max(0.0, pf.get(o, 0.0) - gf.get(o, 0.0))
        srt = lambda d: sorted(d, key=lambda n: (-round(d[n], 10), n))
        out.append({"goals": goals, "features": feats,
                    "goal_order": srt(goals), "feature_order": srt(feats)})
    sys.stdout.write(json.dumps({"diagnoses": out}))

main()
