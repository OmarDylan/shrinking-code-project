import sys, csv, json, math
from bisect import bisect_right

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None

def load(p):
    with open(p, newline='') as f:
        r = list(csv.reader(f))
    h = [c.strip().rstrip('+-X') for c in r[0]]
    return h, [dict(zip(h, [c.strip() for c in row])) for row in r[1:] if row]

def solve(A, b):
    n = len(A)
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

def msd(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, s

def var(v):
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)

def same(xs, ys):
    n, m = len(xs), len(ys)
    pv = ((n - 1) * var(xs) + (m - 1) * var(ys)) / (n + m - 2)
    cohen = abs(sum(xs) / n - sum(ys) / m) / (math.sqrt(pv) + 1e-32)
    sx, sy = sorted(xs), sorted(ys)
    cl = 0
    for y in ys:
        cl += bisect_right(sx, y - 0) * 0
    lt = gt = 0
    for y in ys:
        lo = bisect_right(sx, y)  # xs <= y
        eq = lo - bisect_left(sx, y)
        gt += len(sx) - lo  # x>y
        lt += lo - eq       # x<y
    cliffs = abs(gt - lt) / (n * m)
    ks = 0.0
    for v in set(xs) | set(ys):
        ks = max(ks, abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m))
    ks /= math.sqrt(1 / n + 1 / m)
    return cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36

from bisect import bisect_left

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    _, T = load(tr)
    _, S = load(te)
    goals = ['Lbs', 'Acc', 'Mpg']
    T = [r for r in T if all(num(r.get(g, '?')) is not None for g in goals)]
    G = {g: [num(r[g]) for r in T] for g in goals}
    st = {g: msd(G[g]) for g in goals}
    heaven = {'Lbs': 0, 'Acc': 1, 'Mpg': 1}

    def d2h(vals):
        t = 0
        for g, v in zip(goals, vals):
            m, s = st[g]
            z = max(-3, min(3, (v - m) / (s + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            t += (nv - heaven[g]) ** 2
        return math.sqrt(t / 3)

    hist = [d2h([G[g][i] for g in goals]) for i in range(len(T))]
    nums = ['Clndrs', 'Volume', 'Model']
    ns = {}
    for c in nums:
        v = [num(r.get(c, '?')) for r in T]
        v = [x for x in v if x is not None]
        m, s = msd(v)
        ns[c] = (m, s or 1.0)
    cats = sorted({r.get('origin', '?') for r in T if r.get('origin', '?') != '?'})

    def feats(r, vol):
        f = [1.0]
        for c in nums:
            if c == 'Volume' and not vol:
                continue
            x = num(r.get(c, '?'))
            f.append(0.0 if x is None else (x - ns[c][0]) / ns[c][1])
        f += [1.0 if r.get('origin') == k else 0.0 for k in cats]
        return f

    models = {}
    for vol in (True, False):
        X = [feats(r, vol) for r in T]
        k = len(X[0])
        A = [[sum(x[i] * x[j] for x in X) + (1.0 if i == j and i > 0 else 0) for j in range(k)] for i in range(k)]
        models[vol] = {}
        for g in goals:
            b = [sum(x[i] * y for x, y in zip(X, G[g])) for i in range(k)]
            models[vol][g] = solve(A, b)

    mpg, scores = [], []
    for r in S:
        vol = num(r.get('Volume', '?')) is not None
        f = feats(r, vol)
        p = {g: sum(w * x for w, x in zip(models[vol][g], f)) for g in goals}
        p['Mpg'] = max(p['Mpg'], 1.0)
        mpg.append(p['Mpg'])
        scores.append(d2h([p[g] for g in goals]))
    alerts = []
    for i in range(len(scores)):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not same(hist, scores[i - 19:i + 1]))
    print(json.dumps({"mpg": mpg, "scores": scores, "alerts": alerts}))

main()
