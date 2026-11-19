import sys, json, csv
from math import exp, log, sqrt

def read(p):
    with open(p, encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def num(s):
    try:
        return float(s)
    except ValueError:
        return None

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        d = M[i][i] or 1e-12
        for r in range(n):
            if r != i:
                f = M[r][i] / d
                for c in range(i, n + 1):
                    M[r][c] -= f * M[i][c]
    return [M[i][n] / (M[i][i] or 1e-12) for i in range(n)]

def ridge(X, y, lam=1.0):
    k = len(X[0]) + 1
    Z = [[1.0] + x for x in X]
    A = [[sum(z[i] * z[j] for z in Z) + (lam if i == j and i else 0) for j in range(k)] for i in range(k)]
    b = [sum(z[i] * t for z, t in zip(Z, y)) for i in range(k)]
    w = solve(A, b)
    return lambda x: w[0] + sum(a * c for a, c in zip(w[1:], x))

def main():
    a = sys.argv[1:]
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    h, R = read(tr)
    h2, T = read(te)
    ix = {n: i for i, n in enumerate(h)}
    jx = {n: i for i, n in enumerate(h2)}
    goals = ["Lbs-", "Acc+", "Mpg+"]
    R = [r for r in R if all(num(r[ix[g]]) is not None for g in goals)
         and all(num(r[ix[c]]) is not None for c in ("Clndrs", "Model"))]
    G = [[float(r[ix[g]]) for r in R] for g in goals]
    st = []
    for g in G:
        m = sum(g) / len(g)
        sd = sqrt(sum((v - m) ** 2 for v in g) / max(1, len(g) - 1))
        st.append((m, sd))
    def nm(v, s):
        z = max(-3, min(3, (v - s[0]) / (s[1] + 1e-32)))
        return 1 / (1 + exp(-1.702 * z))
    hv = [0, 1, 1]
    d2h = [sqrt(sum((nm(G[k][i], st[k]) - hv[k]) ** 2 for k in range(3)) / 3) for i in range(len(R))]
    mpg = [log(max(v, 1e-6)) for v in G[2]]
    orig = sorted({r[ix["origin"]] for r in R})

    def feats(r, ixm, vol):
        c = num(r[ixm["Clndrs"]]); m = num(r[ixm["Model"]])
        f = [c if c is not None else 0, m if m is not None else 0]
        if vol:
            v = num(r[ixm["Volume"]])
            f += [log(max(v, 1))]
            f += [f[0] * f[2] / 10 if False else 0.0]
        o = r[ixm["origin"]]
        f += [1.0 if o == x else 0.0 for x in orig]
        return f
    def build(vol):
        X = [feats(r, ix, vol) for r in R]
        n = len(X)
        mu = [sum(x[j] for x in X) / n for j in range(len(X[0]))]
        sd = [sqrt(sum((x[j] - mu[j]) ** 2 for x in X) / n) or 1 for j in range(len(X[0]))]
        sc = lambda x: [(x[j] - mu[j]) / sd[j] for j in range(len(x))]
        Xs = [sc(x) for x in X]
        fm, fd = ridge(Xs, mpg, 1.0), ridge(Xs, d2h, 1.0)
        return sc, fm, fd
    models = {True: build(True), False: build(False)}
    mp, sc_out = [], []
    dm = sum(d2h) / len(d2h); mm = sum(G[2]) / len(G[2])
    for r in T:
        v = num(r[jx["Volume"]]) if "Volume" in jx else None
        vol = v is not None
        sc, fm, fd = models[vol]
        x = sc(feats(r, jx, vol))
        m = exp(fm(x)); d = fd(x)
        if not (m == m and m > 0 and m < 1e6): m = mm
        if not (d == d): d = dm
        mp.append(float(m)); sc_out.append(float(min(1.0, max(0.0, d))))
    print(json.dumps({"mpg": mp, "scores": sc_out}))

main()
