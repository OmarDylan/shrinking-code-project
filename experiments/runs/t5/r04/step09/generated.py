import sys, csv, json, math, argparse

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        r = [[c.strip() for c in l] for l in csv.reader(f) if l]
    return r[0], r[1:]

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()
    th, tr = load(a.train)
    eh, te = load(a.test)
    cols = [(i, n[0].isupper()) for i, n in enumerate(th) if n[-1] not in "X+-"]
    ti = {n: i for i, n in enumerate(eh)}
    cols = [(i, u, ti.get(th[i], i)) for i, u in cols]  # test index by name

    def vec(row, k):
        out = []
        for i, u, j in cols:
            s = row[j if k else i] if (j if k else i) < len(row) else "?"
            out.append(None if s == "?" or s == "" else (num(s) if u else s))
        return out

    X = [vec(r, 0) for r in tr]
    T = [vec(r, 1) for r in te]
    n = len(X)
    st = []
    for c, (i, u, _) in enumerate(cols):
        vs = [x[c] for x in X if x[c] is not None]
        if u:
            vs.sort()
            m = len(vs)
            med = vs[m // 2] if m else 0.0
            dev = sorted(abs(v - med) for v in vs)
            mad = dev[m // 2] if m else 0.0
            mu = sum(vs) / m if m else 0.0
            sd = math.sqrt(sum((v - mu) ** 2 for v in vs) / max(1, m - 1)) if m else 1.0
            sc = max(1.4826 * mad, 0.5 * sd, 1e-9)
            st.append((med, sc, max(sd, 1e-9)))
        else:
            cnt = {}
            for v in vs:
                cnt[v] = cnt.get(v, 0) + 1
            st.append((cnt, max(1, len(vs)), 1))

    def zmax(x):
        z = 0.0
        for c, (i, u, _) in enumerate(cols):
            v = x[c]
            if v is None:
                continue
            if u:
                z = max(z, abs(v - st[c][0]) / st[c][1])
            else:
                f = (st[c][0].get(v, 0) + 0.5) / (st[c][1] + 0.5)
                z = max(z, -math.log(f) if f < 1 else 0)
        return z

    def d(x, y):
        s = k = 0
        for c, (i, u, _) in enumerate(cols):
            if x[c] is None or y[c] is None:
                continue
            g = (x[c] - y[c]) / st[c][2] if u else float(x[c] != y[c])
            s += g * g
            k += 1
        return math.sqrt(s / k) if k else 0.0

    def knn(x, skip=None):
        ds = sorted(d(x, y) for j, y in enumerate(X) if j != skip)
        ds = ds[:5]
        return sum(ds) / len(ds) if ds else 0.0

    trk = [knn(x, j) for j, x in enumerate(X)]
    ks = sorted(trk)
    kmed = ks[len(ks) // 2] if ks else 1.0
    kmed = max(kmed, 1e-6)

    def raw(x, k):
        return zmax(x) / 3 + k / (3 * kmed)

    tr_raw = sorted(raw(x, k) for x, k in zip(X, trk))
    q = lambda p: tr_raw[min(len(tr_raw) - 1, int(p * len(tr_raw)))] if tr_raw else 1.0
    sc0 = max(2 * q(0.95), 1e-9)
    f = lambda r: min(1.0, max(0.0, 1 - math.exp(-r / sc0)))
    thr = f(q(0.97))
    scores = [f(raw(x, knn(x))) for x in T]
    print(json.dumps({"scores": scores, "alerts": [s > thr for s in scores], "threshold": thr}))

main()
