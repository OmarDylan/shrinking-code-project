import sys, json, math

def num(s):
    s = s.strip()
    return s if s == "?" else float(s)

def read(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = [l.rstrip("\r\n").split(",") for l in f if l.strip()]
    return [h.strip() for h in rows[0]], [[c.strip() for c in r] for r in rows[1:]]

def ms(v):
    n = len(v); m = sum(v) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0

def same(xs, ys):
    n, m = len(xs), len(ys)
    mx, vx = ms(xs); my, vy = ms(ys)
    pool = ((n - 1) * vx ** 2 + (m - 1) * vy ** 2) / (n + m - 2)
    cohen = abs(mx - my) / (math.sqrt(pool) + 1e-32)
    s = sum((x > y) - (x < y) for x in xs for y in ys)
    cliffs = abs(s) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    import bisect
    d = 0.0
    for v in set(xs) | set(ys):
        d = max(d, abs(bisect.bisect_right(sx, v) / n - bisect.bisect_right(sy, v) / m))
    ks = d / math.sqrt(1 / n + 1 / m)
    return cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36

def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    hd, train = read(tr)
    _, test = read(te)
    ix = {h: i for i, h in enumerate(hd)}
    G = [ix["Lbs-"], ix["Acc+"], ix["Mpg+"]]
    H = [0, 1, 1]
    NF = [ix["Clndrs"], ix["Volume"], ix["Model"]]
    SF = ix["origin"]
    stats = [ms([float(r[g]) for r in train]) for g in G]
    fst = {c: ms([float(r[c]) for r in train]) for c in NF}

    def d2h(vals):
        t = 0.0
        for v, (m, sd), h in zip(vals, stats, H):
            z = max(-3, min(3, (v - m) / (sd + 1e-32)))
            t += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
        return math.sqrt(t / 3)

    T = [([float(r[c]) for c in NF], r[SF], [float(r[g]) for g in G]) for r in train]

    def predict(r):
        q = [num(r[c]) for c in NF]
        o = r[SF]
        ds = []
        for xs, so, ys in T:
            s = k = 0
            for j, c in enumerate(NF):
                if q[j] != "?":
                    s += ((q[j] - xs[j]) / (fst[c][1] + 1e-9)) ** 2; k += 1
            if o != "?":
                s += (o != so); k += 1
            ds.append((math.sqrt(s / max(k, 1)), ys))
        ds.sort(key=lambda t: t[0])
        nb = ds[:7]
        w = [1 / (d + 1e-3) for d, _ in nb]
        W = sum(w)
        return [sum(wi * y[j] for wi, (_, y) in zip(w, nb)) / W for j in range(3)]

    mpg, sc = [], []
    for r in test:
        p = predict(r)
        mpg.append(max(p[2], 1e-6))
        sc.append(min(1.0, max(0.0, d2h(p))))
    hist = [d2h([float(r[g]) for g in G]) for r in train]
    al = []
    for i in range(len(sc)):
        if i < 19:
            al.append(False)
        else:
            al.append(not same(hist, sc[i - 19:i + 1]))
    sys.stdout.write(json.dumps({"mpg": mpg, "scores": sc, "alerts": al}))

main()
