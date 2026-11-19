import csv, sys, math

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def read(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c] or 1e-12
        for r in range(n):
            if r != c:
                k = M[r][c] / d
                if k:
                    for j in range(c, n + 1):
                        M[r][j] -= k * M[c][j]
    return [M[i][n] / (M[i][i] or 1e-12) for i in range(n)]

def main():
    a = sys.argv
    tr = read(a[a.index('--train') + 1])
    te = read(a[a.index('--test') + 1])
    goals = [('Lbs', 0), ('Acc', 1), ('Mpg', 1)]
    gk = {}
    for h in (tr[0].keys() if tr else []):
        for g, _ in goals:
            if h.rstrip('+-') == g:
                gk[g] = h
    tr = [r for r in tr if all(num(r.get(gk[g])) is not None for g, _ in goals)]
    G = {g: [float(r[gk[g]]) for r in tr] for g, _ in goals}
    st = {}
    for g, _ in goals:
        v = G[g]; m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
        st[g] = (m, sd)

    def norm(g, v):
        m, sd = st[g]
        z = max(-3, min(3, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    d2h = [math.sqrt(sum((norm(g, G[g][i]) - h) ** 2 for g, h in goals) / 3) for i in range(len(tr))]
    logmpg = [math.log(max(x, 1e-3)) for x in G['Mpg']]
    cats = sorted({r.get('origin', '') for r in tr})
    means = {}
    for c in ('Clndrs', 'Model', 'Volume'):
        v = [num(r.get(c)) for r in tr]
        v = [x for x in v if x is not None]
        means[c] = sum(v) / len(v) if v else 0.0

    def feats(r, vol):
        f = []
        for c in (('Clndrs', 'Volume', 'Model') if vol else ('Clndrs', 'Model')):
            x = num(r.get(c))
            f.append(means[c] if x is None else x)
        if vol:
            f.append(math.log(max(f[1], 1.0)))
        f += [1.0 if r.get('origin', '') == c else 0.0 for c in cats]
        return f

    def fit(vol, y):
        rows = [i for i, r in enumerate(tr) not in () and enumerate(tr) and [(i, r)] and [] or [] ] if False else None
        idx = [i for i, r in enumerate(tr) if (not vol or num(r.get('Volume')) is not None)]
        X = [feats(tr[i], vol) for i in idx]
        Y = [y[i] for i in idx]
        k = len(X[0]); n = len(X)
        mu = [sum(x[j] for x in X) / n for j in range(k)]
        sg = [math.sqrt(sum((x[j] - mu[j]) ** 2 for x in X) / n) or 1.0 for j in range(k)]
        Z = [[(x[j] - mu[j]) / sg[j] for j in range(k)] for x in X]
        ym = sum(Y) / n
        A = [[sum(z[i] * z[j] for z in Z) + (1.0 if i == j else 0.0) for j in range(k)] for i in range(k)]
        b = [sum(z[i] * (yy - ym) for z, yy in zip(Z, Y)) for i in range(k)]
        w = solve(A, b)
        return lambda x: ym + sum(w[j] * (x[j] - mu[j]) / sg[j] for j in range(k))

    models = {}
    for vol in (True, False):
        models[vol] = (fit(vol, logmpg), fit(vol, d2h))
    mpg, sc = [], []
    for r in te:
        vol = num(r.get('Volume')) is not None
        fm, fd = models[vol]
        x = feats(r, vol)
        mpg.append(max(math.exp(max(-5, min(6, fm(x)))), 1e-3))
        sc.append(max(0.0, min(1.0, fd(x))))
    print(__import__('json').dumps({"mpg": mpg, "scores": sc}))

main()
