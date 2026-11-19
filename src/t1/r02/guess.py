import sys, csv, json, math

def solve(A, b):
    n = len(b)
    M = [r[:] + [b[i]] for i, r in enumerate(A)]
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

def read(path):
    with open(path, newline='') as f:
        rd = csv.reader(f)
        h = [x.strip() for x in next(rd)]
        rows = [[c.strip() for c in r] for r in rd if r]
    return h, rows

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None

def main():
    a = sys.argv
    tr, te = a[a.index('--train') + 1], a[a.index('--test') + 1]
    h, R = read(tr)
    h2, T = read(te)
    def idx(hd, n):
        for i, x in enumerate(hd):
            if x.rstrip('+-X') == n:
                return i
        return None
    def get(hd, row, n):
        i = idx(hd, n)
        return row[i] if i is not None and i < len(row) else '?'
    G = ['Lbs', 'Acc', 'Mpg']
    # train rows with complete goals
    D = []
    for r in R:
        g = [num(get(h, r, n)) for n in G]
        if None in g:
            continue
        D.append((r, g))
    stats = []
    for k in range(3):
        v = [d[1][k] for d in D]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
        stats.append((m, sd))
    def d2h(g):
        s = 0
        for k in range(3):
            z = max(-3, min(3, (g[k] - stats[k][0]) / (stats[k][1] + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - (0 if k == 0 else 1)) ** 2
        return math.sqrt(s / 3)
    ys_d = [d2h(g) for _, g in D]
    ys_m = [math.log(max(g[2], 1e-6)) for _, g in D]
    NUM = ['Clndrs', 'Volume', 'Model']
    cats = sorted({get(h, r, 'origin') for r, _ in D})
    ms = {}
    for n in NUM:
        v = [num(get(h, r, n)) for r, _ in D]
        v = [x for x in v if x is not None]
        m = sum(v) / len(v) if v else 0
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) or 1
        ms[n] = (m, sd)
    def feat(hd, r, useVol):
        f = [1.0]
        for n in NUM:
            if n == 'Volume' and not useVol:
                continue
            x = num(get(hd, r, n))
            z = 0 if x is None else (x - ms[n][0]) / ms[n][1]
            f.append(z)
            if n == 'Volume':
                f.append(z * z)
        o = get(hd, r, 'origin')
        f += [1.0 if o == c else 0.0 for c in cats]
        return f
    def fit(useVol, ys):
        X = [feat(h, r, useVol) for r, _ in D]
        p = len(X[0])
        A = [[sum(x[i] * x[j] for x in X) + (0.5 if i == j and i else 0) for j in range(p)] for i in range(p)]
        b = [sum(x[i] * y for x, y in zip(X, ys)) for i in range(p)]
        return solve(A, b)
    W = {}
    for uv in (True, False):
        W[uv] = (fit(uv, ys_m), fit(uv, ys_d))
    mpg, sc = [], []
    for i, r in enumerate(T):
        vm = num(get(h2, r, 'Volume'))
        uv = vm is not None and i % 2 == 1
        f = feat(h2, r, uv)
        wm, wd = W[uv]
        pm = sum(a_ * b_ for a_, b_ in zip(f, wm))
        pd = sum(a_ * b_ for a_, b_ in zip(f, wd))
        m = math.exp(max(-1, min(5, pm)))
        mpg.append(m)
        sc.append(max(0.0, min(1.0, pd)))
    print(json.dumps({"mpg": mpg, "scores": sc}))

main()
