import sys, csv, json, math, bisect

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r]
    return [h.strip() for h in rows[0]], [[c.strip() for c in r] for r in rows[1:]]

def main():
    a = sys.argv
    tr, te = a[a.index("--train") + 1], a[a.index("--test") + 1]
    hn, trr = load(tr)
    ht, ter = load(te)
    cols = [i for i, h in enumerate(hn) if h[-1] not in "X+-!"]
    num = {i: hn[i][0].isupper() for i in cols}
    tidx = {h: i for i, h in enumerate(ht)}

    def parse(r):
        out = {}
        for i in cols:
            j = tidx.get(hn[i], i)
            v = r[j] if j < len(r) else "?"
            if v == "?" or v == "":
                out[i] = None
            elif num[i]:
                try:
                    x = float(v)
                    out[i] = x if math.isfinite(x) else None
                except ValueError:
                    out[i] = None
            else:
                out[i] = v
        return out

    T = [parse(r) for r in trr]
    E = [parse(r) for r in ter]
    st = {}
    for i in cols:
        if not num[i]:
            continue
        xs = sorted(r[i] for r in T if r[i] is not None)
        if not xs:
            st[i] = (0, 1, 1)
            continue
        n = len(xs)
        med = xs[n // 2]
        mu = sum(xs) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / max(1, n - 1)) or 1.0
        dv = sorted(abs(x - med) for x in xs)
        mad = 1.4826 * dv[n // 2]
        if mad < 1e-9:
            mad = sd
        st[i] = (med, sd, mad)
    freq = {}
    for i in cols:
        if not num[i]:
            c = {}
            for r in T:
                if r[i] is not None:
                    c[r[i]] = c.get(r[i], 0) + 1
            freq[i] = c
    N = max(1, len(T))

    def dist(r, q):
        s = k = 0
        for i in cols:
            x, y = r[i], q[i]
            if x is None or y is None:
                continue
            k += 1
            if num[i]:
                s += ((x - y) / st[i][1]) ** 2
            else:
                s += x != y
        return math.sqrt(s / k) if k else 0.0

    def raw(r, skip=None):
        zm = 0.0
        for i in cols:
            x = r[i]
            if x is None:
                continue
            if num[i]:
                zm = max(zm, abs(x - st[i][0]) / st[i][2])
            else:
                f = freq[i].get(x, 0) - (1 if skip is not None else 0)
                p = (f + 0.5) / (N + 1)
                zm = max(zm, max(0.0, -math.log(p) - 3.0))
        ds = sorted(dist(r, q) for j, q in enumerate(T) if j != skip)
        kn = sum(ds[:5]) / max(1, len(ds[:5]))
        return kn + 0.3 * zm

    tr_raw = sorted(raw(r, j) for j, r in enumerate(T))
    n = len(tr_raw)
    q = lambda p: tr_raw[min(n - 1, int(p * n))] if n else 1.0
    c = q(0.9) or 1e-6
    sq = lambda v: v / (v + c)
    thr = sq(q(0.95))
    sc = [min(1.0, max(0.0, sq(raw(r)))) for r in E]
    print(json.dumps({"scores": sc, "alerts": [s > thr for s in sc], "threshold": thr}))

main()
