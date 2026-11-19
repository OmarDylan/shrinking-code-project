import sys, csv, json, math

def fl(s):
    try: return float(s)
    except: return None

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        r = [[c.strip() for c in l] for l in csv.reader(f) if l]
    return r[0], r[1:]

def solve(A, b):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        d = M[i][i] or 1e-12
        M[i] = [v / d for v in M[i]]
        for r in range(n):
            if r != i and M[r][i]:
                f = M[r][i]
                M[r] = [a - f * c for a, c in zip(M[r], M[i])]
    return [M[i][n] for i in range(n)]

def same(xs, ys):
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    vy = sum((y - my) ** 2 for y in ys) / (m - 1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    co = abs(mx - my) / (math.sqrt(pv) + 1e-32)
    cl = abs(sum((x > y) - (x < y) for x in xs for y in ys)) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    import bisect
    ks = max(abs(bisect.bisect_right(sx, v) / n - bisect.bisect_right(sy, v) / m)
             for v in set(xs) | set(ys)) / math.sqrt(1 / n + 1 / m)
    return co <= 0.35 and cl <= 0.197 and ks <= 1.36

def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    hn, trr = load(tr)
    _, ter = load(te)
    ix = {n: i for i, n in enumerate(hn)}
    ic, iv, im, io = ix["Clndrs"], ix["Volume"], ix["Model"], ix["origin"]
    il, ia, ip = ix["Lbs-"], ix["Acc+"], ix["Mpg+"]
    goals = [il, ia, ip]
    stats = []
    for g in goals:
        v = [float(r[g]) for r in trr]
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1))
        stats.append((mu, sd))
    def d2h(l, ac, mp):
        t = 0
        for v, (mu, sd), h in zip((l, ac, mp), stats, (0, 1, 1)):
            z = max(-3, min(3, (v - mu) / (sd + 1e-32)))
            t += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
        return math.sqrt(t / 3)
    hist = [d2h(float(r[il]), float(r[ia]), float(r[ip])) for r in trr]
    nums = [ic, iv, im]
    ms = {}
    for c in nums:
        v = [x for x in (fl(r[c]) for r in trr) if x is not None]
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)) or 1
        ms[c] = (mu, sd)
    origs = sorted({r[io] for r in trr})
    def feat(r, use_vol):
        f = [1.0]
        for c in nums:
            if c == iv and not use_vol: continue
            x = fl(r[c])
            mu, sd = ms[c]
            f.append(0.0 if x is None else (x - mu) / sd)
        f += [1.0 if r[io] == o else 0.0 for o in origs]
        return f
    def fit(use_vol, g):
        X = [feat(r, use_vol) for r in trr]
        y = [float(r[g]) for r in trr]
        k = len(X[0])
        A = [[sum(x[i] * x[j] for x in X) + (1.0 if i == j and i else 0) for j in range(k)] for i in range(k)]
        b = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(k)]
        return solve(A, b)
    W = {(uv, g): fit(uv, g) for uv in (True, False) for g in goals}
    minmpg = min(float(r[ip]) for r in trr)
    mpg, scores = [], []
    for r in ter:
        uv = fl(r[iv]) is not None
        f = feat(r, uv)
        p = {g: sum(w * x for w, x in zip(W[(uv, g)], f)) for g in goals}
        m = max(p[ip], minmpg * 0.5, 1e-3)
        mpg.append(m)
        scores.append(min(1.0, max(0.0, d2h(p[il], p[ia], m))))
    alerts = []
    for i in range(len(scores)):
        if i < 19: alerts.append(False)
        else: alerts.append(not same(hist, scores[i - 19:i + 1]))
    print(json.dumps({"mpg": mpg, "scores": scores, "alerts": alerts}))

if __name__ == "__main__":
    main()
