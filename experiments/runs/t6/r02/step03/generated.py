import sys, csv, json, argparse
from math import exp, sqrt

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
NUMS = ["Clndrs", "Volume", "Model"]


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]


def load(path):
    head, rows = read(path)
    ix = [head.index(g) for g in GOALS]
    return [[float(r[i]) for i in ix] for r in rows
            if all(r[i] != "?" for i in ix)]


def gstats(tr):
    n = len(tr)
    st = []
    for j in range(3):
        col = [r[j] for r in tr]
        mu = sum(col) / n
        sd = sqrt(sum((v - mu) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        st.append((mu, sd))
    return st


def scorer(stats):
    hs = list(GOALS.values())

    def score(r):
        s = 0.0
        for j in range(3):
            mu, sd = stats[j]
            z = max(-3.0, min(3.0, (r[j] - mu) / (sd + 1e-32)))
            s += (1 / (1 + exp(-1.702 * z)) - hs[j]) ** 2
        return sqrt(s / 3)
    return score


def remember(a):
    tr, te = load(a.train), load(a.test)
    score = scorer(gstats(tr))
    scores = [score(r) for r in te]
    order = sorted(range(len(te)), key=lambda i: (scores[i], i))
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[len(te) // 2], "worst": order[-1]}


def num(v):
    try:
        return float(v)
    except ValueError:
        return None


def guess(a):
    hd, tr = read(a.train)
    hd2, te = read(a.test)
    gi = [hd.index(g) for g in GOALS]
    xi = [hd.index(c) for c in NUMS]
    oi = hd.index("origin")
    xi2 = [hd2.index(c) for c in NUMS]
    oi2 = hd2.index("origin")
    tr = [r for r in tr if all(num(r[i]) is not None for i in gi)]
    score = scorer(gstats([[float(r[i]) for i in gi] for r in tr]))
    X, Y, D = [], [], []
    for r in tr:
        g = [float(r[i]) for i in gi]
        X.append(([num(r[i]) for i in xi], r[oi]))
        Y.append(g[2])
        D.append(score(g))
    sds, mus = [], []
    for j in range(3):
        c = [x[0][j] for x in X if x[0][j] is not None]
        m = sum(c) / len(c)
        mus.append(m)
        sds.append(sqrt(sum((v - m) ** 2 for v in c) / max(1, len(c) - 1)) or 1.0)
    byc = {}
    for x, _, _ in zip(X, Y, D):
        if x[0][0] is not None and x[0][1] is not None:
            byc.setdefault(x[0][0], []).append(x[0][1])
    wt = [1.5, 1.5, 1.0]
    k = 7
    mpg, sc = [], []
    for r in te:
        q = [num(r[i]) for i in xi2]
        o = r[oi2]
        if q[1] is None and q[0] in byc:
            q[1] = sum(byc[q[0]]) / len(byc[q[0]])
        ds = []
        for (x, xo), y, d in zip(X, Y, D):
            s = w = 0.0
            for j in range(3):
                if q[j] is not None and x[j] is not None:
                    s += wt[j] * ((q[j] - x[j]) / sds[j]) ** 2
                    w += wt[j]
            if o != "?" and xo != "?":
                s += 0.5 * (o != xo)
                w += 0.5
            ds.append((sqrt(s / w) if w else 1.0, y, d))
        ds.sort(key=lambda t: t[0])
        nb = ds[:k]
        ws = [1 / (t[0] + 0.05) for t in nb]
        sw = sum(ws)
        m = sum(wi * t[1] for wi, t in zip(ws, nb)) / sw
        s = sum(wi * t[2] for wi, t in zip(ws, nb)) / sw
        mpg.append(max(m, 1.0))
        sc.append(min(1.0, max(0.0, s)))
    return {"mpg": mpg, "scores": sc}


def sign(v):
    return (v > 0) - (v < 0)


def compare(xs, ys):
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((v - mx) ** 2 for v in xs) / (n - 1)
    vy = sum((v - my) ** 2 for v in ys) / (m - 1)
    pooled = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (sqrt(pooled) + 1e-32)
    cl = abs(sum(sign(x - y) for x in xs for y in ys)) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    i = j = 0
    d = 0.0
    for v in sorted(set(xs) | set(ys)):
        while i < n and sx[i] <= v:
            i += 1
        while j < m and sy[j] <= v:
            j += 1
        d = max(d, abs(i / n - j / m))
    ks = d / sqrt(1 / n + 1 / m)
    return {"cohen": cohen, "cliffs": cl, "ks": ks,
            "same": bool(cohen <= 0.35 and cl <= 0.197 and ks <= 1.36)}


def certify(a):
    with open(a.test, encoding="utf-8-sig") as f:
        cases = json.load(f)
    return {"comparisons": [compare([float(v) for v in c["xs"]],
                                    [float(v) for v in c["ys"]])
                            for c in cases]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    if a.cmd == "remember":
        out = remember(a)
    elif a.cmd == "guess":
        out = guess(a)
    elif a.cmd == "certify":
        out = certify(a)
    else:
        sys.exit("unknown command")
    sys.stdout.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
