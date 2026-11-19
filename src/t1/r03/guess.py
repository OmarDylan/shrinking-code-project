import sys, csv, json, math

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def read(p):
    with open(p, newline='') as f:
        r = list(csv.reader(f))
    return [h.strip() for h in r[0]], [x for x in r[1:] if x]

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

def fit(X, y, lam=1.0):
    n, k = len(X), len(X[0])
    mu = [sum(r[j] for r in X) / n for j in range(k)]
    sd = [math.sqrt(sum((r[j] - mu[j]) ** 2 for r in X) / n) or 1.0 for j in range(k)]
    Z = [[(r[j] - mu[j]) / sd[j] for j in range(k)] for r in X]
    ym = sum(y) / n
    A = [[sum(z[i] * z[j] for z in Z) + (lam if i == j else 0) for j in range(k)] for i in range(k)]
    b = [sum(z[i] * (yy - ym) for z, yy in zip(Z, y)) for i in range(k)]
    w = solve(A, b)
    return lambda x: ym + sum(w[j] * (x[j] - mu[j]) / sd[j] for j in range(k))

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    h, R = read(tr)
    h2, T = read(te)
    ix = {n: i for i, n in enumerate(h)}
    jx = {n: i for i, n in enumerate(h2)}
    g = lambda row, ids, n: row[ids[n]] if n in ids and ids[n] < len(row) else '?'
    rows = []
    for r in R:
        v = [num(g(r, ix, c)) for c in ('Clndrs', 'Volume', 'Model', 'Lbs-', 'Acc+', 'Mpg+')]
        if None in v:
            continue
        rows.append((v, g(r, ix, 'origin').strip()))
    goals = list(zip(*[v[3:] for v, _ in rows]))
    st = []
    for col in goals:
        m = sum(col) / len(col)
        s = math.sqrt(sum((x - m) ** 2 for x in col) / (len(col) - 1))
        st.append((m, s))
    def norm(x, m, s):
        z = max(-3, min(3, (x - m) / (s + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))
    hv = (0, 1, 1)
    d2 = []
    for v, _ in rows:
        d2.append(math.sqrt(sum((norm(v[3 + i], *st[i]) - hv[i]) ** 2 for i in range(3)) / 3))
    cats = sorted({o for _, o in rows})
    cm = sum(v[0] for v, _ in rows) / len(rows)
    mm = sum(v[2] for v, _ in rows) / len(rows)
    vm = sum(v[1] for v, _ in rows) / len(rows)

    def feat(c, vol, m, o, usev):
        f = [c, m, c * c, m * m] + [1.0 if o == k else 0.0 for k in cats]
        if usev:
            f += [vol, math.log(max(vol, 1.0)), vol * vol / 1000.0, vol * c]
        return f

    models = {}
    for usev in (True, False):
        X = [feat(v[0], v[1], v[2], o, usev) for v, o in rows]
        fm = fit(X, [math.log(max(v[5], 1e-3)) for v, _ in rows], 3.0)
        fd = fit(X, d2, 3.0)
        models[usev] = (fm, fd)
    mp, sc = [], []
    lo = min(v[5] for v, _ in rows)
    hi = max(v[5] for v, _ in rows)
    for r in T:
        c = num(g(r, jx, 'Clndrs'))
        m = num(g(r, jx, 'Model'))
        vol = num(g(r, jx, 'Volume'))
        o = g(r, jx, 'origin').strip()
        c = cm if c is None else c
        m = mm if m is None else m
        usev = vol is not None
        x = feat(c, vol if usev else vm, m, o, usev)
        fm, fd = models[usev]
        p = math.exp(fm(x))
        p = min(max(p, lo * 0.7), hi * 1.3)
        mp.append(max(p, 1e-3))
        sc.append(min(1.0, max(0.0, fd(x))))
    sys.stdout.write(json.dumps({"mpg": mp, "scores": sc}))

main()
