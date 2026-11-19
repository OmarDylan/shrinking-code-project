import sys, json, math, argparse, time

def rows_of(path):
    out = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                out.append([s.strip() for s in line.strip().split(",")])
    return out

def num(s):
    try:
        return float(s)
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    t0 = time.time()
    tr, te = rows_of(a.train), rows_of(a.test)
    names = tr[0]
    tr = tr[1:]
    te = te[1:] if te and te[0] == names else te[1:]
    cols = [i for i, s in enumerate(names) if s[-1] != "X"]
    ys = [i for i in cols if names[i][-1] in "+-"]
    xs = [i for i in cols if names[i][-1] not in "+-"]
    isnum = lambda i: names[i][0].isupper()

    stats = {}
    for i in cols:
        if isnum(i):
            v = [num(r[i]) for r in tr if i < len(r) and r[i] != "?" and num(r[i]) is not None]
            n = len(v)
            mu = sum(v) / n if n else 0.0
            sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
            stats[i] = (mu, sd)

    def norm(i, s):
        if s == "?" or s == "":
            return None
        x = num(s)
        if x is None:
            return None
        mu, sd = stats[i]
        z = max(-3, min(3, (x - mu) / (1e-32 + sd)))
        return 1 / (1 + math.exp(-1.7 * z))

    def gaps(r):
        g = {}
        for i in ys:
            n = norm(i, r[i]) if i < len(r) else None
            goal = 0 if names[i][-1] == "-" else 1
            g[names[i]] = 1.0 if n is None else abs(n - goal)
        return g

    def yd(r):
        g = list(gaps(r).values())
        return math.sqrt(sum(x * x for x in g) / len(g)) if g else 0.0

    srt = sorted(tr, key=yd)
    k = max(1, int(len(srt) * 0.25))
    good, poor = srt[:k], srt[-k:]

    def profile(rows):
        p = {}
        for i in xs:
            if isnum(i):
                v = [norm(i, r[i]) for r in rows if i < len(r)]
                v = [x for x in v if x is not None]
                p[i] = sum(v) / len(v) if v else None
            else:
                v = [r[i] for r in rows if i < len(r) and r[i] != "?"]
                c = {}
                for x in v:
                    c[x] = c.get(x, 0) + 1
                p[i] = {x: n / len(v) for x, n in c.items()} if v else {}
        return p

    pg, pp = profile(good), profile(poor)

    out = []
    for r in te:
        g = gaps(r)
        f = {}
        for i in xs:
            s = 0.0
            v = r[i] if i < len(r) else "?"
            if v != "?" and v != "":
                if isnum(i):
                    n = norm(i, v)
                    if n is not None and pg[i] is not None and pp[i] is not None:
                        s = max(0.0, abs(n - pg[i]) - abs(n - pp[i]))
                else:
                    s = max(0.0, pp[i].get(v, 0.0) - pg[i].get(v, 0.0))
            f[names[i]] = s
        out.append({
            "goals": g,
            "features": f,
            "goal_order": sorted(g, key=lambda k: (-g[k], k)),
            "feature_order": sorted(f, key=lambda k: (-f[k], k)),
        })
    print(json.dumps({"diagnoses": out, "runtime": time.time() - t0}))

if __name__ == "__main__":
    main()
