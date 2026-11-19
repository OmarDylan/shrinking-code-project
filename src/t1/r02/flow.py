import sys, csv, json, math, bisect

def num(s):
    try:
        return float(s)
    except Exception:
        return None

def load(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    hdr = [h.strip() for h in rows[0]]
    return hdr, [[c.strip() for c in r] for r in rows[1:] if r]

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        d = M[c][c] or 1e-12
        for r in range(n):
            if r != c:
                f = M[r][c] / d
                if f:
                    M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / (M[i][i] or 1e-12) for i in range(n)]

def mean(x): return sum(x) / len(x)

def var(x):
    m = mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)

def certify(xs, ys):
    n, m = len(xs), len(ys)
    pv = ((n - 1) * var(xs) + (m - 1) * var(ys)) / (n + m - 2)
    cohen = abs(mean(xs) - mean(ys)) / (math.sqrt(pv) + 1e-32)
    s = 0
    for x in xs:
        for y in ys:
            s += (x > y) - (x < y)
    cl = abs(s) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    ks = 0
    for v in set(xs) | set(ys):
        d = abs(bisect.bisect_right(sx, v) / n - bisect.bisect_right(sy, v) / m)
        ks = max(ks, d)
    ks /= math.sqrt(1 / n + 1 / m)
    return cohen <= 0.35 and cl <= 0.197 and ks <= 1.36

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    h, R = load(tr)
    th, T = load(te)
    key = lambda n: n.rstrip('+-X?').lower()
    hi = {key(n): i for i, n in enumerate(h)}
    ti = {key(n): i for i, n in enumerate(th)}
    numf = ['clndrs', 'volume', 'model']
    goals = ['lbs', 'acc', 'mpg']
    G = [[float(r[hi[g]]) for r in R] for g in goals]
    st = {}
    for f in numf:
        v = [x for x in (num(r[hi[f]]) for r in R) if x is not None]
        st[f] = (mean(v), math.sqrt(var(v)) or 1.0)
    cats = sorted({r[hi['origin']] for r in R})

    def feat(get, vol):
        x = [1.0]
        for f in numf:
            if f == 'volume' and not vol:
                continue
            v = get(f)
            m, s = st[f]
            x.append(0.0 if v is None else (v - m) / s)
        o = get('origin')
        x += [1.0 if o == c else 0.0 for c in cats]
        return x

    def trget(r): return lambda f: r[hi[f]] if f == 'origin' else num(r[hi[f]])
    def teget(r):
        def g(f):
            if f not in ti: return None
            c = r[ti[f]] if ti[f] < len(r) else '?'
            return c if f == 'origin' else num(c)
        return g

    models = {}
    for vol in (True, False):
        X = [feat(trget(r), vol) for r in R]
        k = len(X[0])
        A = [[sum(x[i] * x[j] for x in X) + (1.0 if i == j and i else 0) for j in range(k)] for i in range(k)]
        models[vol] = [solve(A, [sum(x[i] * y for x, y in zip(X, g)) for i in range(k)]) for g in G]

    stats = [(mean(g), math.sqrt(var(g))) for g in G]
    hv = [0, 1, 1]

    def d2h(vals):
        t = 0
        for v, (m, s), t_h in zip(vals, stats, hv):
            z = max(-3, min(3, (v - m) / (s + 1e-32)))
            t += (1 / (1 + math.exp(-1.702 * z)) - t_h) ** 2
        return math.sqrt(t / 3)

    hist = [d2h(v) for v in zip(*G)]
    mpg, sc, al = [], [], []
    for i, r in enumerate(T):
        g = teget(r)
        vol = g('volume') is not None
        x = feat(g, vol)
        p = [sum(w * v for w, v in zip(m, x)) for m in models[vol]]
        p[2] = max(p[2], 1.0)
        mpg.append(p[2])
        sc.append(d2h(p))
        al.append(False if i < 19 else not certify(hist, sc[-20:]))
    print(json.dumps({"mpg": mpg, "scores": sc, "alerts": al}))

main()
