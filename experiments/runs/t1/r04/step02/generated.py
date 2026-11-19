import sys, csv, math

def load(p):
    with open(p, newline='') as f:
        r = list(csv.reader(f))
    h = [x.strip() for x in r[0]]
    return h, [[c.strip() for c in row] for row in r[1:] if row]

def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        d = M[i][i] or 1e-12
        for j in range(i, n + 1):
            M[i][j] /= d
        for r in range(n):
            if r != i and M[r][i]:
                f = M[r][i]
                for j in range(i, n + 1):
                    M[r][j] -= f * M[i][j]
    return [M[i][n] for i in range(n)]

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    h, R = load(tr)
    th, T = load(te)
    ix = {k: i for i, k in enumerate(h)}
    tx = {k: i for i, k in enumerate(th)}
    goals = ['Lbs-', 'Acc+', 'Mpg+']
    rows = []
    for r in R:
        g = [num(r[ix[k]]) for k in goals]
        if None in g:
            continue
        rows.append((r, g))
    stats = []
    for j in range(3):
        v = [g[j] for _, g in rows]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0
        stats.append((m, sd))
    def d2h(g):
        s = 0
        for j in range(3):
            z = max(-3, min(3, (g[j] - stats[j][0]) / (stats[j][1] + 1e-32)))
            n = 1 / (1 + math.exp(-1.702 * z))
            s += (n - (0 if j == 0 else 1)) ** 2
        return math.sqrt(s / 3)
    origins = sorted({r[ix['origin']] for r, _ in rows if r[ix['origin']] != '?'})
    nums = ['Clndrs', 'Volume', 'Model']
    cm = {}
    for k in nums:
        v = [num(r[ix[k]]) for r, _ in rows]
        v = [x for x in v if x is not None]
        cm[k] = sum(v) / len(v) if v else 0.0
    def feats(r, idx, use_vol):
        f = []
        for k in nums:
            if k == 'Volume' and not use_vol:
                continue
            x = num(r[idx[k]]) if k in idx and idx[k] < len(r) else None
            f.append(cm[k] if x is None else x)
        o = r[idx['origin']] if 'origin' in idx and idx['origin'] < len(r) else '?'
        f += [1.0 if o == x else 0.0 for x in origins]
        return f
    ys = [(math.log(max(g[2], 1e-6)), d2h(g)) for _, g in rows]
    def fit(use_vol):
        X = [feats(r, ix, use_vol) for r, _ in rows]
        p = len(X[0])
        mu = [sum(x[j] for x in X) / len(X) for j in range(p)]
        sd = [math.sqrt(sum((x[j] - mu[j]) ** 2 for x in X) / len(X)) or 1 for j in range(p)]
        Z = [[(x[j] - mu[j]) / sd[j] for j in range(p)] for x in X]
        ws = []
        for t in range(2):
            y = [v[t] for v in ys]
            ym = sum(y) / len(y)
            A = [[sum(z[i] * z[j] for z in Z) + (1.0 if i == j else 0) for j in range(p)] for i in range(p)]
            b = [sum(Z[k][i] * (y[k] - ym) for k in range(len(Z))) for i in range(p)]
            ws.append((solve(A, b), ym))
        return mu, sd, ws
    models = {True: fit(True), False: fit(False)}
    mpg, sc = [], []
    for i, r in enumerate(T):
        uv = i % 2 == 1
        mu, sd, ws = models[uv]
        f = feats(r, tx, uv)
        z = [(f[j] - mu[j]) / sd[j] for j in range(len(f))]
        out = [ym + sum(w * q for w, q in zip(ww, z)) for ww, ym in ws]
        mpg.append(max(1.0, min(100.0, math.exp(out[0]))))
        sc.append(max(0.0, min(1.0, out[1])))
    import json
    print(json.dumps({"mpg": mpg, "scores": sc}))

main()
