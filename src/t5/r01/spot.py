import sys, json, math, csv, argparse

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    return [c.strip() for c in rows[0]], [[c.strip() for c in r] for r in rows[1:]]

def num(v):
    try:
        return float(v)
    except Exception:
        return None

def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    th, tr = load(a.train)
    eh, te = load(a.test)

    def used(h):  # input columns only: not ignored (X), not goals (+-!)
        return [n for n in h if n[-1] not in "X+-!"]
    cols = [n for n in used(th) if n in eh]
    nums = [n for n in cols if n[0].isupper()]
    syms = [n for n in cols if not n[0].isupper()]

    def get(h, r, n):
        i = h.index(n)
        return r[i] if i < len(r) else "?"

    T = [{n: get(th, r, n) for n in cols} for r in tr]
    E = [{n: get(eh, r, n) for n in cols} for r in te]

    stats = {}
    for n in nums:
        v = [num(r[n]) for r in T if num(r[n]) is not None]
        v = sorted(v)
        med = v[len(v) // 2]
        mad = sorted(abs(x - med) for x in v)[len(v) // 2] * 1.4826
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1))
        scale = max(mad, 0.25 * sd, 1e-9)
        stats[n] = (med, scale, max(sd, 1e-9))
    freq = {}
    for n in syms:
        c = {}
        for r in T:
            c[r[n]] = c.get(r[n], 0) + 1
        freq[n] = c
    N = len(T)

    def vec(r):
        return [(num(r[n]) - stats[n][0]) / stats[n][2] if num(r[n]) is not None else None for n in nums] + \
               [r[n] for n in syms]

    TV = [vec(r) for r in T]
    K = 5

    def comps(r, skip=None):
        c = []
        for n in nums:
            x = num(r[n])
            c.append(0.0 if x is None else abs(x - stats[n][0]) / stats[n][1])
        for n in syms:
            cnt = freq[n].get(r[n], 0) - (1 if skip is not None else 0)
            c.append(-math.log((cnt + 0.5) / (N + 1)) if r[n] != "?" else 0.0)
        v = vec(r)
        ds = []
        for j, u in enumerate(TV):
            if j == skip:
                continue
            s = 0.0
            for x, y in zip(v, u):
                if x is None or y is None:
                    continue
                s += (x - y) ** 2 if isinstance(x, float) else (0 if x == y else 1)
            ds.append(math.sqrt(s))
        ds.sort()
        k = ds[:K]
        c.append(sum(k) / len(k) if k else 0.0)
        return c

    trc = [comps(r, i) for i, r in enumerate(T)]
    m = len(trc[0])
    norm = [max(q([c[j] for c in trc], 0.97), 1e-9) for j in range(m)]

    def raw(c):
        return max(x / s for x, s in zip(c, norm))

    trraw = [raw(c) for c in trc]
    rt = max(q(trraw, 0.96), 1e-9)
    f = lambda x: x / (x + 1.0)
    thr = f(rt)
    scores = [f(raw(comps(r))) for r in E]
    print(json.dumps({"scores": scores, "alerts": [s > thr for s in scores], "threshold": thr}))

main()
