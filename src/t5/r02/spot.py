import sys, json, math

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        rows = [l.rstrip("\r\n").split(",") for l in f if l.strip()]
    return [h.strip() for h in rows[0]], [[c.strip() for c in r] for r in rows[1:]]

def feats(h, rows):
    ix = [h.index(n) for n in ("Clndrs", "Volume", "Model", "origin")]
    out = []
    for r in rows:
        v = []
        for k, i in enumerate(ix):
            c = r[i] if i < len(r) else "?"
            if c == "?" or c == "":
                v.append(None)
            elif k < 3:
                try:
                    v.append(float(c))
                except ValueError:
                    v.append(None)
            else:
                v.append(c)
        out.append(v)
    return out

def main():
    a = sys.argv
    tr, te = a[a.index("--train") + 1], a[a.index("--test") + 1]
    h, R = load(tr)
    X = feats(h, R)
    h2, R2 = load(te)
    T = feats(h2, R2)
    sc = []
    for k in range(3):
        v = sorted(x[k] for x in X if x[k] is not None)
        n = len(v) or 1
        mu = sum(v) / n
        sd = (sum((y - mu) ** 2 for y in v) / max(1, n - 1)) ** .5 if v else 1
        sc.append(sd if sd > 1e-12 else 1.0)

    def d(p, q):
        s = m = 0
        for k in range(4):
            if p[k] is None or q[k] is None:
                continue
            g = abs(p[k] - q[k]) / sc[k] if k < 3 else float(p[k] != q[k])
            s += g * g
            m += 1
        return math.sqrt(s / m) if m else 0.0

    K = min(5, max(1, len(X) - 1))

    def knn(p, skip=None):
        ds = sorted(d(p, q) for i, q in enumerate(X) if i != skip)
        ds = ds[:K]
        return sum(ds) / len(ds) if ds else 0.0

    loo = sorted(knn(p, i) for i, p in enumerate(X))
    med = loo[len(loo) // 2] if loo else 1.0
    med = med if med > 1e-9 else 1e-9
    f = lambda x: x / (x + med)
    thr = f(loo[min(len(loo) - 1, int(0.97 * len(loo)))]) if loo else 0.5
    scores = [f(knn(p)) for p in T]
    scores = [min(1.0, max(0.0, s)) for s in scores]
    thr = min(1.0, max(0.0, thr))
    print(json.dumps({"scores": scores, "alerts": [s > thr for s in scores], "threshold": thr}))

main()
