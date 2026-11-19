import csv, sys, math, json

GOALS = [('Lbs-', 0), ('Acc+', 1), ('Mpg+', 1)]


def read(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows


def num(row, header, name):
    try:
        i = header.index(name)
        v = float(row[i])
        return v if math.isfinite(v) else None
    except (ValueError, IndexError):
        return None


def goal_vals(header, rows):
    return [[num(r, header, g) for g, _ in GOALS] if all(num(r, header, g) is not None for g, _ in GOALS) else None
            for r in rows]


def goal_stats(tr):
    n = len(tr)
    stats = []
    for j in range(3):
        xs = [v[j] for v in tr]
        m = sum(xs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    return stats


def d2h(v, stats):
    s = 0.0
    for j, (_, h) in enumerate(GOALS):
        m, sd = stats[j]
        z = max(-3.0, min(3.0, (v[j] - m) / (sd + 1e-32)))
        nv = 1 / (1 + math.exp(-1.702 * z))
        s += (nv - h) ** 2
    return math.sqrt(s / 3)


def args():
    a = sys.argv[1:]
    train = test = None
    for i, x in enumerate(a):
        if x == '--train':
            train = a[i + 1]
        elif x == '--test':
            test = a[i + 1]
    return a[0], train, test


def remember(train, test):
    h1, r1 = read(train)
    tr = [v for v in goal_vals(h1, r1) if v is not None]
    stats = goal_stats(tr)
    h2, r2 = read(test)
    scores = [d2h(v, stats) for v in goal_vals(h2, r2)]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({"scores": scores, "top": order[:10], "best": order[0],
                      "median": order[len(scores) // 2], "worst": order[-1]}))


def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


class Ridge:
    def __init__(self, X, y, lam=1.0):
        n, d = len(X), len(X[0])
        self.mu = [sum(r[j] for r in X) / n for j in range(d)]
        self.sd = [math.sqrt(sum((r[j] - self.mu[j]) ** 2 for r in X) / n) or 1.0 for j in range(d)]
        self.ym = sum(y) / n
        Z = [[(r[j] - self.mu[j]) / self.sd[j] for j in range(d)] for r in X]
        A = [[sum(z[i] * z[j] for z in Z) + (lam if i == j else 0.0) for j in range(d)] for i in range(d)]
        b = [sum(z[i] * (yy - self.ym) for z, yy in zip(Z, y)) for i in range(d)]
        self.w = solve(A, b)

    def predict(self, x):
        return self.ym + sum(w * (xi - m) / s for w, xi, m, s in zip(self.w, x, self.mu, self.sd))


def guess(train, test):
    h1, r1 = read(train)
    h2, r2 = read(test)
    origins = sorted({r[h1.index('origin')] for r in r1 if h1.index('origin') < len(r)} - {'?', ''})

    def raw(r, h):
        o = r[h.index('origin')] if h.index('origin') < len(r) else '?'
        return [num(r, h, 'Clndrs'), num(r, h, 'Volume'), num(r, h, 'Model'), o]

    traw = [raw(r, h1) for r in r1]
    means = []
    for j in range(3):
        xs = [t[j] for t in traw if t[j] is not None]
        means.append(sum(xs) / len(xs) if xs else 0.0)

    def feats(t, vol):
        c = t[0] if t[0] is not None else means[0]
        m = t[2] if t[2] is not None else means[2]
        f = [c, m]
        if vol:
            v = t[1] if t[1] is not None else means[1]
            f += [v, math.log(max(v, 1.0))]
        return f + [1.0 if t[3] == o else 0.0 for o in origins]

    gv = goal_vals(h1, r1)
    idx = [i for i, v in enumerate(gv) if v is not None]
    stats = goal_stats([gv[i] for i in idx])
    ymp = [math.log(max(gv[i][2], 1e-6)) for i in idx]
    yd = [d2h(gv[i], stats) for i in idx]
    models = {}
    for vol in (True, False):
        X = [feats(traw[i], vol) for i in idx]
        models[vol] = (Ridge(X, ymp, 2.0), Ridge(X, yd, 2.0))

    mpg, scores = [], []
    lo, hi = min(yd), max(yd)
    for r in r2:
        t = raw(r, h2)
        vol = t[1] is not None
        mm, md = models[vol]
        x = feats(t, vol)
        p = math.exp(max(-5.0, min(6.0, mm.predict(x))))
        s = max(0.0, min(1.0, md.predict(x)))
        mpg.append(p)
        scores.append(s)
    print(json.dumps({"mpg": mpg, "scores": scores}))


def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    cmd, train, test = args()
    if cmd == 'remember':
        remember(train, test)
    elif cmd == 'guess':
        guess(train, test)
    else:
        sys.exit(1)


main()
