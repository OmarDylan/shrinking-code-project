import sys, json, csv, math

def rd(p):
    with open(p, encoding="utf-8-sig") as f:
        return [[c.strip() for c in r] for r in csv.reader(f) if r]

def num(v):
    try:
        return float(v)
    except Exception:
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
                if f:
                    for c in range(i, n + 1):
                        M[r][c] -= f * M[i][c]
    return [M[i][n] / (M[i][i] or 1e-12) for i in range(n)]

def ridge(X, y, lam=1.0):
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) + (lam if i == j and i else 0)
          for j in range(k)] for i in range(k)]
    b = [sum(r[i] * t for r, t in zip(X, y)) for i in range(k)]
    return solve(A, b)

def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    T, S = rd(tr), rd(te)
    h = T[0]
    ix = {n: i for i, n in enumerate(h)}
    tt, ss = T[1:], S[1:]
    def get(r, n):
        i = ix[n]
        return r[i] if i < len(r) else "?"
    # goal stats
    goals = [("Lbs-", 0), ("Acc+", 1), ("Mpg+", 1)]
    st = {}
    for g, _ in goals:
        v = [num(get(r, g)) for r in tt]
        v = [x for x in v if x is not None]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
        st[g] = (m, sd)
    def d2h(r):
        s = 0
        for g, hv in goals:
            m, sd = st[g]
            z = max(-3, min(3, (num(get(r, g)) - m) / (sd + 1e-32)))
            s += (1 / (1 + math.exp(-1.702 * z)) - hv) ** 2
        return math.sqrt(s / 3)
    ok = [r for r in tt if all(num(get(r, g)) is not None for g, _ in goals)]
    ycols = ["Clndrs", "Volume", "Model"]
    cs = {}
    for c in ycols:
        v = [num(get(r, c)) for r in tt]
        v = [x for x in v if x is not None]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) or 1
        cs[c] = (m, sd)
    orgs = sorted({get(r, "origin") for r in tt})
    def feat(r, cols):
        f = [1.0]
        for c in cols:
            v = num(get(r, c))
            z = 0.0 if v is None else (v - cs[c][0]) / cs[c][1]
            f += [z, z * z]
        o = get(r, "origin")
        f += [1.0 if o == k else 0.0 for k in orgs]
        if len(cols) > 1:
            z = [feat_z(r, c) for c in cols]
            f += [z[0] * z[1], z[1] * z[2], z[0] * z[2]]
        return f
    def feat_z(r, c):
        v = num(get(r, c))
        return 0.0 if v is None else (v - cs[c][0]) / cs[c][1]
    models = {}
    for key, cols in (("full", ycols), ("nov", ["Clndrs", "Model"])):
        X = [feat(r, cols) for r in ok]
        ym = [math.log(max(num(get(r, "Mpg+")), 1e-3)) for r in ok]
        yd = [d2h(r) for r in ok]
        models[key] = (cols, ridge(X, ym), ridge(X, yd))
    mpg, sc = [], []
    for r in ss:
        key = "full" if num(get(r, "Volume")) is not None else "nov"
        cols, wm, wd = models[key]
        f = feat(r, cols)
        pm = math.exp(max(-3, min(5, sum(x * w for x, w in zip(f, wm)))))
        pd = max(0.0, min(1.0, sum(x * w for x, w in zip(f, wd))))
        mpg.append(max(pm, 1e-3))
        sc.append(pd)
    print(json.dumps({"mpg": mpg, "scores": sc}))

main()
