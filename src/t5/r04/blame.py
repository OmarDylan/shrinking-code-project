import sys, csv, json, math


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r and any(c.strip() for c in r)]
    return rows[0], rows[1:]


def num(v):
    if v == "?" or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


class N:
    def __init__(self, vals):
        vals = [v for v in vals if v is not None]
        n = len(vals)
        self.mu = sum(vals) / n if n else 0.0
        self.sd = math.sqrt(sum((v - self.mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0

    def norm(self, v):
        z = max(-3.0, min(3.0, (v - self.mu) / (self.sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))


def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    hdr, trows = load(tr)
    _, erows = load(te)
    ix = {h: i for i, h in enumerate(hdr)}
    goals = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]
    nfeat = ["Clndrs", "Volume", "Model"]
    cfeat = ["origin"]
    gn = {g: N([num(r[ix[g]]) for r in trows]) for g, _ in goals}

    def d2h(r):
        s = 0.0
        for g, h in goals:
            s += (gn[g].norm(num(r[ix[g]])) - h) ** 2
        return math.sqrt(s / 3)

    order = sorted(range(len(trows)), key=lambda i: (round(d2h(trows[i]), 12), i))
    q = max(1, len(trows) // 4)
    good = [trows[i] for i in order[:q]]
    poor = [trows[i] for i in order[-q:]]

    fn = {f: N([num(r[ix[f]]) for r in trows]) for f in nfeat}
    cen = {}
    for f in nfeat:
        c = []
        for grp in (good, poor):
            vs = [x for x in (num(r[ix[f]]) for r in grp) if x is not None]
            c.append(fn[f].norm(sum(vs) / len(vs)) if vs else None)
        cen[f] = c

    def freq(grp, f, cat):
        vs = [r[ix[f]] for r in grp if r[ix[f]] != "?" and r[ix[f]] != ""]
        return vs.count(cat) / len(vs) if vs else 0.0

    out = []
    for r in erows:
        gs = {}
        for g, h in goals:
            gs[g] = abs(gn[g].norm(num(r[ix[g]])) - h)
        fs = {}
        for f in nfeat:
            x = num(r[ix[f]])
            gc, pc = cen[f]
            if x is None or gc is None or pc is None:
                fs[f] = 0.0
            else:
                nx = fn[f].norm(x)
                fs[f] = max(0.0, abs(nx - gc) - abs(nx - pc))
        for f in cfeat:
            v = r[ix[f]]
            if v == "?" or v == "":
                fs[f] = 0.0
            else:
                fs[f] = max(0.0, freq(poor, f, v) - freq(good, f, v))
        key = lambda d: sorted(d, key=lambda k: (-round(d[k], 10), k))
        out.append({"goals": gs, "features": fs,
                    "goal_order": key(gs), "feature_order": key(fs)})
    sys.stdout.write(json.dumps({"diagnoses": out}))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "blame":
        main()
