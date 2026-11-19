import csv, sys, json, argparse
from math import exp, log, sqrt

def num(s):
    try: return float(s)
    except (ValueError, TypeError): return None

def load(path):
    with open(path, encoding="utf-8-sig") as f:
        r = [[c.strip() for c in l] for l in csv.reader(f) if l]
    h = r[0]
    return {n: i for i, n in enumerate(h)}, r[1:]

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i:
                f = M[r][i] / (M[i][i] or 1e-12)
                for c in range(i, n + 1): M[r][c] -= f * M[i][c]
    return [M[i][n] / (M[i][i] or 1e-12) for i in range(n)]

def fit(X, y, lam=1e-2):
    k = len(X[0])
    mu = [sum(r[j] for r in X) / len(X) for j in range(k)]
    sd = [sqrt(sum((r[j] - mu[j]) ** 2 for r in X) / len(X)) or 1 for j in range(k)]
    Z = [[1.0] + [(r[j] - mu[j]) / sd[j] for j in range(k)] for r in X]
    d = k + 1
    A = [[sum(z[i] * z[j] for z in Z) + (lam if i == j and i else 0) for j in range(d)] for i in range(d)]
    b = [sum(z[i] * t for z, t in zip(Z, y)) for i in range(d)]
    w = solve(A, b)
    def pred(x):
        return w[0] + sum(w[j + 1] * (x[j] - mu[j]) / sd[j] for j in range(k))
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--train"); ap.add_argument("--test")
    a = ap.parse_args()
    hi, tr = load(a.train)
    _, te = load(a.test) if False else (None, None)
    ti, te = load(a.test)
    def get(row, idx, n):
        i = idx.get(n)
        return num(row[i]) if i is not None and i < len(row) else None
    G = ["Lbs-", "Acc+", "Mpg+"]
    rows = []
    for r in tr:
        v = [get(r, hi, n) for n in ["Clndrs", "Volume", "Model"] + G]
        if None in v: continue
        rows.append((v, r[hi["origin"]]))
    orgs = sorted({o for _, o in rows})
    cm = sum(v[0] for v, _ in rows) / len(rows)
    mm = sum(v[2] for v, _ in rows) / len(rows)
    vm = sum(v[1] for v, _ in rows) / len(rows)
    oh = lambda o: [1.0 * (o == q) for q in orgs]
    # goal stats and d2h
    st = []
    for j in range(3):
        xs = [v[3 + j] for v, _ in rows]
        m = sum(xs) / len(xs)
        s = sqrt(sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1))
        st.append((m, s))
    def d2h(g):
        t = 0
        for j, x in enumerate(g):
            z = max(-3, min(3, (x - st[j][0]) / (st[j][1] + 1e-32)))
            n = 1 / (1 + exp(-1.702 * z))
            t += (n - (0 if j == 0 else 1)) ** 2
        return sqrt(t / 3)
    vf = fit([[v[0], v[2]] + oh(o) for v, o in rows], [v[1] for v, _ in rows])
    F = lambda c, vol, m, o: [c, log(max(vol, 1.0)), m, c * c] + oh(o)
    X = [F(v[0], v[1], v[2], o) for v, o in rows]
    pl = fit(X, [v[3] for v, _ in rows])
    pa = fit(X, [v[4] for v, _ in rows])
    pm = fit(X, [log(max(v[5], 1e-6)) for v, _ in rows])
    pd = fit(X, [d2h(v[3:]) for v, _ in rows])
    mpg, sc = [], []
    for r in te:
        c = get(r, ti, "Clndrs"); c = cm if c is None else c
        m = get(r, ti, "Model"); m = mm if m is None else m
        o = r[ti["origin"]] if "origin" in ti and ti["origin"] < len(r) else ""
        vol = get(r, ti, "Volume")
        if vol is None: vol = vf([c, m] + oh(o))
        x = F(c, vol, m, o)
        mpg.append(max(0.5, min(100.0, exp(pm(x)))))
        dd = pd(x)
        dd2 = d2h([pl(x), pa(x), mpg[-1]])
        sc.append(max(0.0, min(1.0, 0.5 * dd + 0.5 * dd2)))
    print(json.dumps({"mpg": mpg, "scores": sc}))

main()
