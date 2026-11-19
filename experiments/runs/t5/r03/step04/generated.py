import sys, csv, json, math

def num(s):
    try: return float(s)
    except ValueError: return None

def read(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def solve(A, b):
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        d = M[i][i] or 1e-12
        for r in range(n):
            if r != i:
                f = M[r][i] / d
                for c in range(i, n + 1): M[r][c] -= f * M[i][c]
    return [M[i][n] / (M[i][i] or 1e-12) for i in range(n)]

def stats(v):
    n = len(v); m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0
    return m, sd

def mean(v): return sum(v) / len(v)

def same(xs, ys):
    n, m = len(xs), len(ys)
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    vy = sum((y - my) ** 2 for y in ys) / (m - 1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    coh = abs(mx - my) / (math.sqrt(pv) + 1e-32)
    s = sum((x > y) - (x < y) for x in xs for y in ys)
    cl = abs(s) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    import bisect
    k = max(abs(bisect.bisect_right(sx, v) / n - bisect.bisect_right(sy, v) / m)
            for v in set(xs) | set(ys)) / math.sqrt(1 / n + 1 / m)
    return coh <= 0.35 and cl <= 0.197 and k <= 1.36

def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]; te = a[a.index("--test") + 1]
    _, trows = read(tr); _, terows = read(te)
    trows = [r for r in trows if all(num(r[i]) is not None for i in (0, 1, 3, 5, 6, 7))]
    G = [5, 6, 7]
    gs = [stats([float(r[g]) for r in trows]) for g in G]

    def d2h(vals):
        t = 0
        for (m, sd), v, h in zip(gs, vals, (0, 1, 1)):
            z = max(-3, min(3, (v - m) / (sd + 1e-32)))
            t += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
        return math.sqrt(t / 3)

    hist = [d2h([float(r[g]) for g in G]) for r in trows]
    origins = sorted({r[4] for r in trows})
    ms = {c: stats([float(r[c]) for r in trows]) for c in (0, 1, 3)}

    def feats(r, vol):
        f = []
        for c in ((0, 1, 3) if vol else (0, 3)):
            v = num(r[c]); m, sd = ms[c]
            if v is None: v = m
            f.append((v - m) / (sd or 1))
        f += [1.0 if r[4] == o else 0.0 for o in origins]
        return [1.0] + f

    models = {}
    for vol in (True, False):
        X = [feats(r, vol) for r in trows]; p = len(X[0])
        A = [[sum(x[i] * x[j] for x in X) + (0 if i == 0 else 1.0 * (i == j))
              for j in range(p)] for i in range(p)]
        for i in range(1, p): pass
        ws = []
        for g in G:
            y = [float(r[g]) for r in trows]
            b = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(p)]
            ws.append(solve(A, b))
        models[vol] = ws

    mpg, scores, alerts = [], [], []
    for i, r in enumerate(terows):
        vol = num(r[1]) is not None
        x = feats(r, vol)
        pr = [sum(w * v for w, v in zip(ws, x)) for ws in models[vol]]
        pr[2] = max(pr[2], 1e-3)
        mpg.append(pr[2]); scores.append(min(1.0, max(0.0, d2h(pr))))
        alerts.append(bool(i >= 19 and not same(hist, scores[i - 19:i + 1])))
    json.dump({"mpg": mpg, "scores": scores, "alerts": alerts}, sys.stdout)

main()
