import sys, csv, json, math

def num(s):
    try:
        return float(s)
    except Exception:
        return None

def read(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def mean(a): return sum(a) / len(a)
def var(a):
    m = mean(a); return sum((x - m) ** 2 for x in a) / (len(a) - 1)

def same(xs, ys):
    n, m = len(xs), len(ys)
    pv = ((n - 1) * var(xs) + (m - 1) * var(ys)) / (n + m - 2)
    c = abs(mean(xs) - mean(ys)) / (math.sqrt(pv) + 1e-32)
    s = sum((x > y) - (x < y) for x in xs for y in ys)
    cl = abs(s) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    import bisect
    ks = max(abs(bisect.bisect_right(sx, v) / n - bisect.bisect_right(sy, v) / m)
             for v in set(xs) | set(sy)) / math.sqrt(1 / n + 1 / m)
    return c <= 0.35 and cl <= 0.197 and ks <= 1.36

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        d = M[i][i] or 1e-12
        for j in range(i, n + 1): M[i][j] /= d
        for r in range(n):
            if r != i:
                f = M[r][i]
                if f:
                    for j in range(i, n + 1): M[r][j] -= f * M[i][j]
    return [M[i][n] for i in range(n)]

def main():
    a = sys.argv
    tr = read(a[a.index('--train') + 1])
    te = read(a[a.index('--test') + 1])
    G = ['Lbs-', 'Acc+', 'Mpg+']
    tr = [r for r in tr if all(num(r[g]) is not None for g in G)]
    st = {}
    for g in G:
        v = [num(r[g]) for r in tr]
        st[g] = (mean(v), math.sqrt(var(v)))
    def d2h(vals):
        t = 0
        for g, v in zip(G, vals):
            z = max(-3, min(3, (v - st[g][0]) / (st[g][1] + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            t += (nv - (0 if g == 'Lbs-' else 1)) ** 2
        return math.sqrt(t / 3)
    hist = [d2h([num(r[g]) for g in G]) for r in tr]
    origins = sorted({r['origin'] for r in tr})
    def feats(r, vol):
        cols = ['Clndrs', 'Model'] + (['Volume'] if vol else [])
        x = [num(r[c]) for c in cols]
        x = [0.0 if v is None else v for v in x]
        return x, [1.0 if r['origin'] == o else 0.0 for o in origins]
    models = {}
    for vol in (True, False):
        raw = [feats(r, vol) for r in tr]
        k = len(raw[0][0])
        ms = [mean([x[0][j] for x in raw]) for j in range(k)]
        sd = [(math.sqrt(var([x[0][j] for x in raw])) or 1) for j in range(k)]
        def vec(f, ms=ms, sd=sd, k=k):
            return [1.0] + [(f[0][j] - ms[j]) / sd[j] for j in range(k)] + f[1]
        X = [vec(f) for f in raw]
        p = len(X[0])
        A = [[sum(x[i] * x[j] for x in X) + (1e-3 if i and i == j else 0) for j in range(p)] for i in range(p)]
        ws = {}
        for g in G:
            y = [num(r[g]) for r in tr]
            ws[g] = solve(A, [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(p)])
        models[vol] = (vec, ws)
    mpg, sc, al = [], [], []
    for i, r in enumerate(te):
        vol = num(r.get('Volume', '?')) is not None
        vec, ws = models[vol]
        x = vec(feats(r, vol))
        pr = {g: sum(w * xi for w, xi in zip(ws[g], x)) for g in G}
        m = pr['Mpg+']
        if not (m > 0.5): m = 0.5
        pr['Mpg+'] = m
        mpg.append(m)
        s = d2h([pr[g] for g in G])
        sc.append(min(1.0, max(0.0, s)))
        al.append(False if i < 19 else not same(hist, sc[-20:]))
    print(json.dumps({"mpg": mpg, "scores": sc, "alerts": al}))

main()
