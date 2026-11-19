import sys, csv, json, argparse
from math import exp, sqrt

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
NUMX = ["Clndrs", "Volume", "Model"]


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def goal_stats(names, tr, tnames):
    stats = {}
    for g, h in GOALS.items():
        i = names.index(g)
        xs = [float(r[i]) for r in tr]
        n = len(xs)
        mu = sum(xs) / n
        sd = sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        stats[g] = (tnames.index(g) if tnames is not None else i, mu, sd, h)
    return stats


def d2h(vals, stats):
    s = 0.0
    for v, (_, mu, sd, h) in zip(vals, stats.values()):
        z = max(-3.0, min(3.0, (v - mu) / (sd + 1e-32)))
        s += (1 / (1 + exp(-1.702 * z)) - h) ** 2
    return sqrt(s / 3)


def remember(names, tr, tnames, te):
    stats = goal_stats(names, tr, tnames)
    scores = [d2h([float(r[i]) for i, _, _, _ in stats.values()], stats) for r in te]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    n = len(order)
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[n // 2], "worst": order[-1]}


def guess(names, tr, tnames, te, k=7):
    gi = {g: names.index(g) for g in GOALS}
    tr = [r for r in tr if all(num(r[i]) is not None for i in gi.values())]
    stats = goal_stats(names, tr, None)
    mi = gi["Mpg+"]
    train_d = [d2h([float(r[i]) for i, _, _, _ in stats.values()], stats) for r in tr]
    train_m = [float(r[mi]) for r in tr]
    xi = [(c, names.index(c)) for c in NUMX if c in names]
    oi = names.index("origin") if "origin" in names else None
    tx = {c: tnames.index(c) for c in NUMX + ["origin"] if c in tnames}
    sds = {}
    for c, i in xi:
        v = [num(r[i]) for r in tr if num(r[i]) is not None]
        mu = sum(v) / len(v)
        sds[c] = sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)) or 1.0
    mean_m = sum(train_m) / len(train_m)
    mean_d = sum(train_d) / len(train_d)
    mpg, sc = [], []
    for r in te:
        q = {}
        for c, i in xi:
            if c in tx and tx[c] < len(r):
                q[c] = num(r[tx[c]])
        qo = r[tx["origin"]] if "origin" in tx and tx["origin"] < len(r) else "?"
        ds = []
        for j, t in enumerate(tr):
            s, n = 0.0, 0
            for c, i in xi:
                a, b = q.get(c), num(t[i])
                if a is not None and b is not None:
                    s += ((a - b) / sds[c]) ** 2
                    n += 1
            if oi is not None and qo not in ("?", "") and t[oi] not in ("?", ""):
                s += 0.5 * (qo != t[oi])
                n += 1
            ds.append((sqrt(s / n) if n else 1.0, j))
        ds.sort()
        nb = ds[:k]
        w = [1 / (d + 0.1) for d, _ in nb]
        tw = sum(w)
        m = sum(wi * train_m[j] for wi, (_, j) in zip(w, nb)) / tw if tw else mean_m
        d = sum(wi * train_d[j] for wi, (_, j) in zip(w, nb)) / tw if tw else mean_d
        mpg.append(max(m, 1e-3))
        sc.append(max(0.0, min(1.0, d)))
    return {"mpg": mpg, "scores": sc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    names, tr = load(a.train)
    tnames, te = load(a.test)
    if a.cmd == "guess":
        out = guess(names, tr, tnames, te)
    else:
        out = remember(names, tr, tnames, te)
    print(json.dumps(out))


main()
