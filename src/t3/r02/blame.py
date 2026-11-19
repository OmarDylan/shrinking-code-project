import csv, sys, json, math

def read(p):
    with open(p, newline='') as f:
        r = csv.reader(f)
        h = [x.strip() for x in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return h, rows

def num(s):
    if s == '?' or s == '':
        return None
    try:
        return float(s)
    except ValueError:
        return None

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    h, T = read(tr)
    _, S = read(te)
    ix = {n: i for i, n in enumerate(h)}
    nums = ['Clndrs', 'Volume', 'Model']
    cat = 'origin'
    goals = {'Lbs-': 0.0, 'Acc+': 1.0, 'Mpg+': 1.0}

    stats = {}
    for c in list(goals) + nums:
        vals = [num(r[ix[c]]) for r in T]
        vals = [v for v in vals if v is not None]
        n = len(vals)
        m = sum(vals) / n if n else 0.0
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        stats[c] = (m, sd)

    def norm(c, v):
        m, sd = stats[c]
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    def d2h(r):
        s = sum((norm(g, num(r[ix[g]])) - hv) ** 2 for g, hv in goals.items())
        return math.sqrt(s / 3)

    order = sorted(range(len(T)), key=lambda i: (round(d2h(T[i]), 12), i))
    q = max(1, len(T) // 4)
    good = [T[i] for i in order[:q]]
    poor = [T[i] for i in order[-q:]]

    centers = {}
    for c in nums:
        cs = []
        for grp in (good, poor):
            vs = [num(r[ix[c]]) for r in grp]
            vs = [v for v in vs if v is not None]
            cs.append(norm(c, sum(vs) / len(vs)) if vs else None)
        centers[c] = cs

    def freq(grp):
        vs = [r[ix[cat]] for r in grp if r[ix[cat]] not in ('?', '')]
        d = {}
        for v in vs:
            d[v] = d.get(v, 0) + 1
        return d, len(vs)
    gf, gn = freq(good)
    pf, pn = freq(poor)

    out = []
    for r in S:
        feats = {}
        for c in nums:
            v = num(r[ix[c]])
            gc, pc = centers[c]
            if v is None or gc is None or pc is None:
                feats[c] = 0.0
            else:
                x = norm(c, v)
                feats[c] = max(0.0, abs(x - gc) - abs(x - pc))
        v = r[ix[cat]]
        if v in ('?', ''):
            feats[cat] = 0.0
        else:
            p = pf.get(v, 0) / pn if pn else 0.0
            g = gf.get(v, 0) / gn if gn else 0.0
            feats[cat] = max(0.0, p - g)
        gl = {}
        for g, hv in goals.items():
            gl[g] = abs(norm(g, num(r[ix[g]])) - hv)
        out.append({
            "goals": gl,
            "features": feats,
            "goal_order": sorted(gl, key=lambda n: (-round(gl[n], 10), n)),
            "feature_order": sorted(feats, key=lambda n: (-round(feats[n], 10), n)),
        })
    sys.stdout.write(json.dumps({"diagnoses": out}))

if __name__ == '__main__':
    main()
