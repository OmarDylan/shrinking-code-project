import csv, sys, math, json, bisect

def load(p):
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        return [{k.strip(): (v.strip() if v is not None else "?") for k, v in row.items()} for row in r]

NUM = ["Clndrs", "Volume", "Model"]
CAT = "origin"

def main():
    a = sys.argv
    tr = load(a[a.index("--train") + 1])
    te = load(a[a.index("--test") + 1])

    def num(v):
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except Exception:
            return None

    def vec(r):
        c = r.get(CAT, "?")
        return [num(r.get(k, "?")) for k in NUM] + [None if c in ("?", "") else c]

    T = [vec(r) for r in tr]
    Q = [vec(r) for r in te]
    n = len(T)
    mu, sd = [], []
    for j in range(3):
        v = [t[j] for t in T if t[j] is not None]
        m = sum(v) / len(v) if v else 0.0
        s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 1.0
        mu.append(m)
        sd.append(s if s > 1e-9 else 1.0)
    cnt = {}
    for t in T:
        if t[3] is not None:
            cnt[t[3]] = cnt.get(t[3], 0) + 1
    ncat = sum(cnt.values()) or 1

    def dist(x, y):
        s, d = 0.0, 0
        for j in range(3):
            if x[j] is not None and y[j] is not None:
                s += ((x[j] - y[j]) / sd[j]) ** 2
                d += 1
        if x[3] is not None and y[3] is not None:
            s += 0.0 if x[3] == y[3] else 1.0
            d += 1
        return math.sqrt(s / d) if d else 0.0

    def maxz(x):
        z = [0.0]
        for j in range(3):
            if x[j] is not None:
                z.append(abs(x[j] - mu[j]) / sd[j])
        if x[3] is not None:
            z.append(min(4.0, -math.log((cnt.get(x[3], 0) + 0.5) / (ncat + 0.5))) / 1.5)
        return max(z)

    def raw(x, skip=None):
        ds = sorted(dist(x, T[i]) for i in range(n) if i != skip)
        k = ds[:5]
        kn = sum(k) / len(k) if k else 0.0
        return kn + 0.3 * maxz(x)

    tre = sorted(raw(T[i], i) for i in range(n))

    def ecdf(s):
        return min(1.0, max(0.0, bisect.bisect_right(tre, s) / (len(tre) + 1)))

    scores = [ecdf(raw(q)) for q in Q]
    thr = 0.97
    print(json.dumps({"scores": scores, "alerts": [s > thr for s in scores], "threshold": thr}))

main()
