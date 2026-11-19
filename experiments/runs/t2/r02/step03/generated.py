import csv, sys, json, math

GOALS = [("Lbs-", 0), ("Acc+", 1), ("Mpg+", 1)]
NUM = ["Clndrs", "Volume", "Model"]


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows


def args():
    a = sys.argv[1:]
    train = test = None
    for i, x in enumerate(a):
        if x == "--train":
            train = a[i + 1]
        elif x == "--test":
            test = a[i + 1]
    return a[0] if a else None, train, test


def goal_vals(path, complete=True):
    h, rows = load(path)
    idx = [h.index(g) for g, _ in GOALS]
    out = []
    for row in rows:
        if any(row[i] == "?" for i in idx):
            if complete:
                continue
        out.append([float(row[i]) if row[i] != "?" else None for i in idx])
    return out


def goal_stats(tr):
    n = len(tr)
    stats = []
    for j in range(3):
        col = [r[j] for r in tr]
        m = sum(col) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    return stats


def d2h(r, stats):
    s = 0.0
    for j, (_, h) in enumerate(GOALS):
        m, sd = stats[j]
        z = max(-3.0, min(3.0, (r[j] - m) / (sd + 1e-32)))
        nv = 1 / (1 + math.exp(-1.702 * z))
        s += (nv - h) ** 2
    return math.sqrt(s / 3)


def remember(train, test):
    tr = goal_vals(train)
    te = goal_vals(test)
    stats = goal_stats(tr)
    scores = [d2h(r, stats) for r in te]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(te) // 2],
        "worst": order[-1],
    }))


def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        if abs(pv) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / pv
                if f:
                    for k in range(c, n + 1):
                        M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def guess(train, test):
    h, rows = load(train)
    gi = [h.index(g) for g, _ in GOALS]
    ni = [h.index(c) for c in NUM]
    oi = h.index("origin")
    rows = [r for r in rows if all(r[i] != "?" for i in gi)]
    gv = [[float(r[i]) for i in gi] for r in rows]
    stats = goal_stats(gv)
    y_mpg = [math.log(max(g[2], 1e-6)) for g in gv]
    y_d = [d2h(g, stats) for g in gv]
    nums = [[num(r[i]) for i in ni] for r in rows]
    cats = sorted({r[oi] for r in rows if r[oi] != "?"})
    ms, ss = [], []
    for j in range(3):
        c = [x[j] for x in nums if x[j] is not None]
        m = sum(c) / len(c) if c else 0.0
        s = math.sqrt(sum((v - m) ** 2 for v in c) / len(c)) if c else 1.0
        ms.append(m)
        ss.append(s or 1.0)

    def feat(nv, o, use):
        f = [1.0]
        for j in use:
            z = 0.0 if nv[j] is None else (nv[j] - ms[j]) / ss[j]
            f += [z, z * z]
        f += [1.0 if o == c else 0.0 for c in cats]
        return f

    cache = {}

    def model(use):
        if use in cache:
            return cache[use]
        X = [feat(nums[i], rows[i][oi], use) for i in range(len(rows))]
        p = len(X[0]) if X else 1
        res = []
        for y in (y_mpg, y_d):
            A = [[sum(x[a] * x[b] for x in X) + (1.0 if a == b and a > 0 else 0.0)
                  for b in range(p)] for a in range(p)]
            bb = [sum(x[a] * t for x, t in zip(X, y)) for a in range(p)]
            res.append(solve(A, bb))
        cache[use] = res
        return res

    th, trows = load(test)
    tni = [th.index(c) for c in NUM]
    toi = th.index("origin")
    mpg, sc = [], []
    for r in trows:
        nv = [num(r[i]) for i in tni]
        use = tuple(j for j in range(3) if nv[j] is not None)
        wm, wd = model(use)
        f = feat(nv, r[toi], use)
        pm = sum(a * b for a, b in zip(wm, f))
        pd = sum(a * b for a, b in zip(wd, f))
        m = math.exp(max(-5.0, min(6.0, pm)))
        mpg.append(m if math.isfinite(m) and m > 0 else 20.0)
        pd = pd if math.isfinite(pd) else 0.5
        sc.append(max(0.0, min(1.0, pd)))
    print(json.dumps({"mpg": mpg, "scores": sc}))


def compare(xs, ys):
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((v - mx) ** 2 for v in xs) / (n - 1)
    vy = sum((v - my) ** 2 for v in ys) / (m - 1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (math.sqrt(pv) + 1e-32)
    s = sum((x > y) - (x < y) for x in xs for y in ys)
    cliffs = abs(s) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    i = j = 0
    d = 0.0
    for v in sorted(set(xs) | set(ys)):
        while i < n and sx[i] <= v:
            i += 1
        while j < m and sy[j] <= v:
            j += 1
        d = max(d, abs(i / n - j / m))
    ks = d / math.sqrt(1 / n + 1 / m)
    same = cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks, "same": same}


def certify(test):
    with open(test) as f:
        cases = json.load(f)
    print(json.dumps({"comparisons": [
        compare([float(v) for v in c["xs"]], [float(v) for v in c["ys"]])
        for c in cases]}))


def main():
    cmd, train, test = args()
    if cmd == "remember":
        remember(train, test)
    elif cmd == "guess":
        guess(train, test)
    elif cmd == "certify":
        certify(test)
    else:
        sys.exit(1)


main()
