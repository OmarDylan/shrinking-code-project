import sys, json, csv, math, random

def num(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rd = [r for r in csv.reader(f) if r]
    return [h.strip() for h in rd[0]], rd[1:]

def stats(xs):
    n = len(xs); m = sum(xs) / n
    return m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** .5 if n > 1 else 0.0

def nrm(v, m, s):
    z = max(-3, min(3, (v - m) / (s + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))

def cmp(xs, ys):
    n, m = len(xs), len(ys)
    mx, vx = stats(xs); my, vy = stats(ys)
    pv = ((n - 1) * vx ** 2 + (m - 1) * vy ** 2) / (n + m - 2)
    co = abs(mx - my) / (math.sqrt(pv) + 1e-32)
    cl = abs(sum((x > y) - (x < y) for x in xs for y in ys)) / (n * m)
    d = max(abs(sum(a <= v for a in xs) / n - sum(b <= v for b in ys) / m)
            for v in set(xs) | set(ys))
    k = d / math.sqrt(1 / n + 1 / m)
    return {"cohen": co, "cliffs": cl, "ks": k,
            "same": bool(co <= .35 and cl <= .197 and k <= 1.36)}

def main():
    a = sys.argv
    train, test = a[a.index("--train") + 1], a[a.index("--test") + 1]
    h, tr = load(train)
    G = [("Lbs-", 0), ("Acc+", 1), ("Mpg+", 1)]
    gi = [h.index(g) for g, _ in G]
    gs = [stats([float(r[i]) for r in tr]) for i in gi]

    def d2h(vals):
        return math.sqrt(sum((nrm(v, m, s) - hv) ** 2
                             for v, (m, s), (_, hv) in zip(vals, gs, G)) / 3)

    tds = sorted(d2h([float(r[i]) for i in gi]) for r in tr)
    target = round(tds[len(tr) // 4], 12)

    D = json.load(open(test))
    hd, rows = D["header"], D["rows"]
    nc = [hd.index(c) for c in ("Clndrs", "Volume", "Model")]
    oc = hd.index("origin")
    ts = {}
    for c in nc:
        ts[c] = stats([float(r[h.index(hd[c])]) for r in tr if num(r[h.index(hd[c])]) is not None])

    def vec(r):
        v = []
        for c in nc:
            x = num(r[c]); v.append(None if x is None else nrm(x, *ts[c]))
        o = r[oc]; v.append(None if o == "?" else o)
        return v
    X = {}
    def xv(i):
        if i not in X: X[i] = vec(rows[i])
        return X[i]

    def gap(a, b):
        if a is None or b is None: return 1
        return (a != b) if isinstance(a, str) else abs(a - b)

    def cent(idx):
        vs = [xv(i) for i in idx]; c = []
        for j in range(4):
            col = [v[j] for v in vs if v[j] is not None]
            if not col: c.append(None)
            elif j < 3: c.append(sum(col) / len(col))
            else: c.append(max(set(col), key=col.count))
        return c

    def xd(a, b):
        return math.sqrt(sum(gap(p, q) ** 2 for p, q in zip(a, b)) / 4)

    def inspect(e, strat, i):
        print(json.dumps({"episode": e, "strategy": strat, "inspect": i}), flush=True)
        resp = json.loads(sys.stdin.readline())
        g = resp["goals"]
        return d2h([float(g[n]) for n, _ in G])

    rng = random.Random(1234567891)
    res = []
    for e, ep in enumerate(D["episodes"]):
        pool = list(ep["pool"])
        # adaptive
        lab = {}
        order = []
        todo = pool[:]; rng.shuffle(todo)
        for _ in range(4):
            i = todo.pop(); lab[i] = inspect(e, "adaptive", i); order.append(i)
        while len(order) < 10:
            srt = sorted(lab, key=lambda i: lab[i])
            n = max(1, int(math.sqrt(len(srt))))
            cb, cr = cent(srt[:n]), cent(srt[n:] or srt[:n])
            i = min(todo, key=lambda z: xd(xv(z), cr) - xd(xv(z), cb))
            todo.remove(i)
            lab[i] = inspect(e, "adaptive", i); order.append(i)
        A = [lab[i] for i in order]
        R = [inspect(e, "random", i) for i in ep["random_order"][:10]]
        def summ(v):
            t = next((k + 1 for k, x in enumerate(v) if round(x, 12) <= target), 11)
            return {"best5": min(v[:5]), "best10": min(v), "labels_to_target": t}
        res.append({"episode": e, "adaptive": summ(A), "random": summ(R)})
    comp = {k: cmp([r["adaptive"][k] for r in res], [r["random"][k] for r in res])
            for k in ("best5", "best10", "labels_to_target")}
    print(json.dumps({"results": res, "comparisons": comp}), flush=True)

if __name__ == "__main__":
    main()
