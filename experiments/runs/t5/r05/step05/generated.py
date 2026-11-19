import csv, sys, json, argparse
from math import exp, sqrt

def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r and any(c.strip() for c in r)]
    return rows[0], rows[1:]

def stats(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    mu = sum(vals) / n
    sd = sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd

def mknorm(mu, sd):
    def f(v):
        z = max(-3.0, min(3.0, (v - mu) / (sd + 1e-32)))
        return 1 / (1 + exp(-1.702 * z))
    return f

def num(s):
    return None if s == "?" or s == "" else float(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    hdr, tr = read(a.train)
    _, te = read(a.test)
    ix = {h: i for i, h in enumerate(hdr)}
    goals = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]
    nums = ["Clndrs", "Volume", "Model"]
    cat = "origin"

    gn = {}
    for g, _ in goals:
        gn[g] = mknorm(*stats([float(r[ix[g]]) for r in tr]))
    xn = {}
    for c in nums:
        xn[c] = mknorm(*stats([v for v in (num(r[ix[c]]) for r in tr) if v is not None]))

    def d2h(r):
        return sqrt(sum((gn[g](float(r[ix[g]])) - h) ** 2 for g, h in goals) / 3)

    order = sorted(range(len(tr)), key=lambda i: (round(d2h(tr[i]), 12), i))
    q = max(1, len(tr) // 4)
    good = [tr[i] for i in order[:q]]
    poor = [tr[i] for i in order[-q:]]

    centers = {}
    for c in nums:
        cs = []
        for grp in (good, poor):
            vs = [v for v in (num(r[ix[c]]) for r in grp) if v is not None]
            cs.append(xn[c](sum(vs) / len(vs)) if vs else None)
        centers[c] = cs

    def freqs(grp):
        vs = [r[ix[cat]] for r in grp if r[ix[cat]] != "?"]
        d = {}
        for v in vs:
            d[v] = d.get(v, 0) + 1
        n = len(vs)
        return d, n
    gf, gnn = freqs(good)
    pf, pnn = freqs(poor)

    out = []
    for r in te:
        feats = {}
        for c in nums:
            v = num(r[ix[c]])
            gc, pc = centers[c]
            if v is None or gc is None or pc is None:
                feats[c] = 0.0
            else:
                x = xn[c](v)
                feats[c] = max(0.0, abs(x - gc) - abs(x - pc))
        v = r[ix[cat]]
        if v == "?":
            feats[cat] = 0.0
        else:
            pfq = pf.get(v, 0) / pnn if pnn else 0.0
            gfq = gf.get(v, 0) / gnn if gnn else 0.0
            feats[cat] = max(0.0, pfq - gfq)
        gl = {g: abs(gn[g](float(r[ix[g]])) - h) for g, h in goals}
        out.append({
            "goals": gl,
            "features": feats,
            "goal_order": sorted(gl, key=lambda k: (-round(gl[k], 10), k)),
            "feature_order": sorted(feats, key=lambda k: (-round(feats[k], 10), k)),
        })
    sys.stdout.write(json.dumps({"diagnoses": out}))

main()
