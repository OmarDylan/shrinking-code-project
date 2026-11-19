import csv, sys, json, math

def load(p):
    with open(p, newline='') as f:
        r = csv.DictReader(f)
        return [{k.strip(): (v or '').strip() for k, v in row.items()} for row in r]

def num(v):
    if v is None or v == '?' or v == '':
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None

def stats(vals):
    n = len(vals)
    m = sum(vals) / n if n else 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd

def norm(v, st):
    z = (v - st[0]) / (st[1] + 1e-32)
    z = max(-3.0, min(3.0, z))
    return 1 / (1 + math.exp(-1.702 * z))

def main():
    a = sys.argv
    train = load(a[a.index('--train') + 1])
    test = load(a[a.index('--test') + 1])
    goals = {'Lbs-': 0.0, 'Acc+': 1.0, 'Mpg+': 1.0}
    nums = ['Clndrs', 'Volume', 'Model']
    cat = 'origin'
    gst = {g: stats([x for x in (num(r.get(g)) for r in train) if x is not None]) for g in goals}
    nst = {c: stats([x for x in (num(r.get(c)) for r in train) if x is not None]) for c in nums}

    def d2h(r):
        s = 0.0
        for g, h in goals.items():
            v = num(r.get(g))
            s += (norm(v, gst[g]) - h) ** 2 if v is not None else 0.0
        return math.sqrt(s / 3)

    order = sorted(range(len(train)), key=lambda i: (round(d2h(train[i]), 12), i))
    q = max(1, len(train) // 4)
    good = [train[i] for i in order[:q]]
    poor = [train[i] for i in order[-q:]]

    centers = {}
    for c in nums:
        cs = []
        for grp in (good, poor):
            vs = [x for x in (num(r.get(c)) for r in grp) if x is not None]
            cs.append(norm(sum(vs) / len(vs), nst[c]) if vs else None)
        centers[c] = cs

    def freqs(grp):
        vs = [r.get(cat, '') for r in grp if r.get(cat, '') not in ('', '?')]
        d = {}
        for v in vs:
            d[v] = d.get(v, 0) + 1
        n = len(vs)
        return d, n
    gf, gn = freqs(good)
    pf, pn = freqs(poor)

    out = []
    for r in test:
        gs = {}
        for g, h in goals.items():
            v = num(r.get(g))
            gs[g] = abs(norm(v, gst[g]) - h) if v is not None else 0.0
        fs = {}
        for c in nums:
            x = num(r.get(c))
            gc, pc = centers[c]
            if x is None or gc is None or pc is None:
                fs[c] = 0.0
            else:
                nx = norm(x, nst[c])
                fs[c] = max(0.0, abs(nx - gc) - abs(nx - pc))
        v = r.get(cat, '')
        if v in ('', '?'):
            fs[cat] = 0.0
        else:
            p = pf.get(v, 0) / pn if pn else 0.0
            g = gf.get(v, 0) / gn if gn else 0.0
            fs[cat] = max(0.0, p - g)
        key = lambda d: sorted(d, key=lambda k: (-round(d[k], 10), k))
        out.append({"goals": gs, "features": fs,
                    "goal_order": key(gs), "feature_order": key(fs)})
    sys.stdout.write(json.dumps({"diagnoses": out}))

main()
