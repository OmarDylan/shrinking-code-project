import sys, csv, json, math
from bisect import bisect_right

def num(v):
    try:
        return float(v)
    except Exception:
        return None

def solve(A, b):
    n = len(b)
    M = [r[:] + [b[i]] for i, r in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c] or 1e-12
        for r in range(n):
            if r != c:
                f = M[r][c] / d
                if f:
                    for k in range(c, n + 1):
                        M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] or 1e-12) for i in range(n)]

def mean(x): return sum(x) / len(x)
def var(x):
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)

def certify(xs, ys):
    n, m = len(xs), len(ys)
    pv = ((n - 1) * var(xs) + (m - 1) * var(ys)) / (n + m - 2)
    cohen = abs(mean(xs) - mean(ys)) / (math.sqrt(pv) + 1e-32)
    sy = sorted(ys)
    sx = sorted(xs)
    s = 0
    for x in xs:
        lo = bisect_right(sy, x - 0) 
        # count of y < x and y > x
        less = sum(1 for _ in ()) 
        lt = sum(1 for y in sy if y < x)
        gt = sum(1 for y in sy if y > x)
        s += lt - gt
    cliffs = abs(s) / (n * m)
    ks = 0
    for v in set(sx) | set(sy):
        ks = max(ks, abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m))
    ks /= math.sqrt(1 / n + 1 / m)
    return cohen, cliffs, ks, (cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)

def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    with open(tr, newline="") as f:
        train = list(csv.DictReader(f))
    with open(te, newline="") as f:
        test = list(csv.DictReader(f))
    G = ["Lbs-", "Acc+", "Mpg+"]
    train = [r for r in train if all(num(r.get(g)) is not None for g in G)]
    orig = sorted({r["origin"] for r in train})
    nc = ["Clndrs", "Model", "Volume"]
    mu = {c: mean([num(r[c]) for r in train if num(r.get(c)) is not None]) for c in nc}
    sd = {c: (math.sqrt(var([num(r[c]) for r in train if num(r.get(c)) is not None])) or 1) for c in nc}

    def feats(r, vol):
        f = [1.0]
        for c in nc:
            if c == "Volume" and not vol:
                continue
            v = num(r.get(c))
            if v is None:
                v = mu[c]
            f.append((v - mu[c]) / sd[c])
        for o in orig:
            f.append(1.0 if r.get("origin") == o else 0.0)
        return f

    def fit(vol):
        X = [feats(r, vol) for r in train]
        k = len(X[0])
        A = [[sum(x[i] * x[j] for x in X) + (0.5 if i == j and i else 0) for j in range(k)] for i in range(k)]
        ws = []
        for g in G:
            y = [num(r[g]) for r in train]
            ws.append(solve(A, [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(k)]))
        return ws

    W = {True: fit(True), False: fit(False)}
    st = {}
    for g in G:
        v = [num(r[g]) for r in train]
        st[g] = (mean(v), math.sqrt(var(v)))

    def d2h(vals):
        t = 0
        for g, v in zip(G, vals):
            z = max(-3, min(3, (v - st[g][0]) / (st[g][1] + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            h = 0 if g == "Lbs-" else 1
            t += (nv - h) ** 2
        return math.sqrt(t / 3)

    hist = [d2h([num(r[g]) for g in G]) for r in train]
    mpg, scores, alerts = [], [], []
    for i, r in enumerate(test):
        vol = num(r.get("Volume")) is not None
        x = feats(r, vol)
        p = [sum(w * v for w, v in zip(ws, x)) for ws in W[vol]]
        lo = {}
        for j, g in enumerate(G):
            vals = [num(t[g]) for t in train]
            p[j] = max(min(vals), min(max(vals), p[j]))
        p[2] = max(p[2], 1.0)
        mpg.append(p[2])
        scores.append(min(1.0, max(0.0, d2h(p))))
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not certify(hist, scores[-20:])[3])
    print(json.dumps({"mpg": mpg, "scores": scores, "alerts": alerts}))

main()
