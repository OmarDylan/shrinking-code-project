import sys, csv, json, math

GOALS = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]
NUM = ["Clndrs", "Volume", "Model"]
CAT = ["origin"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def val(r, k):
    v = r.get(k)
    if v is None:
        return None
    v = v.strip()
    return None if v == "?" or v == "" else v


def fnum(r, k):
    v = val(r, k)
    try:
        return None if v is None else float(v)
    except ValueError:
        return None


def stats(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    return m, sd


def mk_norm(m, sd):
    def norm(v):
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))
    return norm


def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    train = [r for r in read(tr) if all(fnum(r, g) is not None for g, _ in GOALS)]
    test = read(te)

    gn = {}
    for g, _ in GOALS:
        gn[g] = mk_norm(*stats([fnum(r, g) for r in train]))
    fn = {}
    for c in NUM:
        fn[c] = mk_norm(*stats([x for x in (fnum(r, c) for r in train) if x is not None]))

    def d2h(r):
        return math.sqrt(sum((gn[g](fnum(r, g)) - h) ** 2 for g, h in GOALS) / 3)

    order = sorted(range(len(train)), key=lambda i: (round(d2h(train[i]), 12), i))
    q = max(1, len(train) // 4)
    good = [train[i] for i in order[:q]]
    poor = [train[i] for i in order[-q:]]

    centers = {}
    for c in NUM:
        cs = []
        for grp in (good, poor):
            xs = [x for x in (fnum(r, c) for r in grp) if x is not None]
            cs.append(fn[c](sum(xs) / len(xs)) if xs else None)
        centers[c] = cs

    def freq(grp, c):
        vs = [val(r, c) for r in grp]
        vs = [v for v in vs if v is not None]
        d = {}
        for v in vs:
            d[v] = d.get(v, 0) + 1
        n = len(vs)
        return d, n

    cf = {c: (freq(good, c), freq(poor, c)) for c in CAT}

    out = []
    for r in test:
        goals = {}
        for g, h in GOALS:
            x = fnum(r, g)
            goals[g] = abs(gn[g](x) - h) if x is not None else 0.0
        feats = {}
        for c in NUM:
            x = fnum(r, c)
            gc, pc = centers[c]
            if x is None or gc is None or pc is None:
                feats[c] = 0.0
            else:
                z = fn[c](x)
                feats[c] = max(0.0, abs(z - gc) - abs(z - pc))
        for c in CAT:
            v = val(r, c)
            if v is None:
                feats[c] = 0.0
            else:
                (gd, gnn), (pd, pnn) = cf[c]
                gf = gd.get(v, 0) / gnn if gnn else 0.0
                pf = pd.get(v, 0) / pnn if pnn else 0.0
                feats[c] = max(0.0, pf - gf)
        key = lambda d: sorted(d, key=lambda k: (-round(d[k], 10), k))
        out.append({"goals": goals, "features": feats,
                    "goal_order": key(goals), "feature_order": key(feats)})
    sys.stdout.write(json.dumps({"diagnoses": out}))


main()
