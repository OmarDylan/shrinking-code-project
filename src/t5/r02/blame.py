import csv, sys, json, argparse
from math import exp, sqrt

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
NUMS = ["Clndrs", "Volume", "Model"]
CAT = "origin"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]


def num(s):
    if s == "?" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def stats(vals):
    vals = [v for v in vals if v is not None]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()
    th, tr = load(a.train)
    eh, te = load(a.test)

    def col(h, r, name):
        return r[h.index(name)]

    gn = {g: mknorm(*stats([num(col(th, r, g)) for r in tr])) for g in GOALS}
    fn = {c: mknorm(*stats([num(col(th, r, c)) for r in tr])) for c in NUMS}

    def d2h(h, r):
        s = 0.0
        for g, hv in GOALS.items():
            s += (gn[g](num(col(h, r, g))) - hv) ** 2
        return sqrt(s / 3)

    order = sorted(range(len(tr)), key=lambda i: (round(d2h(th, tr[i]), 12), i))
    q = max(1, len(tr) // 4)
    good = [tr[i] for i in order[:q]]
    poor = [tr[i] for i in order[-q:]]

    centers = {}
    for c in NUMS:
        cs = []
        for grp in (good, poor):
            v = [x for x in (num(col(th, r, c)) for r in grp) if x is not None]
            cs.append(fn[c](sum(v) / len(v)) if v else None)
        centers[c] = cs

    def freqs(grp):
        v = [col(th, r, CAT) for r in grp]
        v = [x for x in v if x != "?" and x != ""]
        d = {}
        for x in v:
            d[x] = d.get(x, 0) + 1
        return d, len(v)

    gf, gc = freqs(good)
    pf, pc = freqs(poor)

    out = []
    for r in te:
        goals = {g: abs(gn[g](num(col(eh, r, g))) - hv) for g, hv in GOALS.items()}
        feats = {}
        for c in NUMS:
            x = num(col(eh, r, c))
            gcn, pcn = centers[c]
            if x is None or gcn is None or pcn is None:
                feats[c] = 0.0
            else:
                nx = fn[c](x)
                feats[c] = max(0.0, abs(nx - gcn) - abs(nx - pcn))
        v = col(eh, r, CAT)
        if v == "?" or v == "":
            feats[CAT] = 0.0
        else:
            g = gf.get(v, 0) / gc if gc else 0.0
            p = pf.get(v, 0) / pc if pc else 0.0
            feats[CAT] = max(0.0, p - g)
        go = sorted(goals, key=lambda k: (-round(goals[k], 10), k))
        fo = sorted(feats, key=lambda k: (-round(feats[k], 10), k))
        out.append({"goals": goals, "features": feats,
                    "goal_order": go, "feature_order": fo})
    sys.stdout.write(json.dumps({"diagnoses": out}))


if __name__ == "__main__":
    main()
