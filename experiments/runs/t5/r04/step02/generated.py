import sys, json, argparse
from math import exp, log, sqrt

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        L = [l.strip().split(",") for l in f if l.strip()]
    return [h.strip() for h in L[0]], [[c.strip() for c in r] for r in L[1:]]

def num(s):
    try: return float(s)
    except (ValueError, TypeError): return None

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i:
                f = M[r][i] / (M[i][i] + 1e-300)
                M[r] = [a - f * c for a, c in zip(M[r], M[i])]
    return [M[i][n] / (M[i][i] + 1e-300) for i in range(n)]

def d2h_fn(G):
    st = []
    for j in range(3):
        c = [g[j] for g in G]; m = sum(c) / len(c)
        sd = sqrt(sum((v - m) ** 2 for v in c) / max(1, len(c) - 1))
        st.append((m, sd))
    def f(g):
        s = 0
        for j, v in enumerate(g):
            z = max(-3, min(3, (v - st[j][0]) / (st[j][1] + 1e-32)))
            s += (1 / (1 + exp(-1.702 * z)) - (0 if j == 0 else 1)) ** 2
        return sqrt(s / 3)
    return f

def guess(tr, te):
    h, R = load(tr); h2, T = load(te)
    I = {n: i for i, n in enumerate(h)}; J = {n: i for i, n in enumerate(h2)}
    gi = [I["Lbs-"], I["Acc+"], I["Mpg+"]]
    rows = []
    for r in R:
        g = [num(r[i]) for i in gi]
        x = [num(r[I[k]]) for k in ("Clndrs", "Volume", "Model")]
        if None in g or None in x: continue
        rows.append((x, r[I["origin"]], g))
    d2 = d2h_fn([r[2] for r in rows])
    Y = [[log(max(r[2][2], 1e-6)), d2(r[2])] for r in rows]
    orgs = sorted({r[1] for r in rows})
    stats = []
    for k in range(3):
        c = [r[0][k] for r in rows]; m = sum(c) / len(c)
        sd = sqrt(sum((v - m) ** 2 for v in c) / len(c)) or 1
        stats.append((m, sd))
    def feat(x, o, vol):
        f = [(x[k] - stats[k][0]) / stats[k][1] for k in range(3) if vol or k != 1]
        return f + [1.0 if o == g else 0.0 for g in orgs]
    models = {}
    for vol in (True, False):
        X = [feat(r[0], r[1], vol) for r in rows]
        Xb = [[1.0] + x for x in X]; n = len(Xb[0])
        W = []
        for t in range(2):
            A = [[sum(x[i] * x[j] for x in Xb) + (0.5 if i == j and i else 0) for j in range(n)] for i in range(n)]
            b = [sum(x[i] * y[t] for x, y in zip(Xb, Y)) for i in range(n)]
            W.append(solve(A, b))
        models[vol] = (X, W)
    def pred(x, o, vol):
        X, W = models[vol]
        f = feat(x, o, vol); fb = [1.0] + f
        ds = sorted((sqrt(sum((a - b) ** 2 for a, b in zip(f, xv))), i) for i, xv in enumerate(X))[:7]
        ws = [1 / (d + 0.1) for d, _ in ds]; sw = sum(ws)
        out = []
        for t in range(2):
            lin = sum(a * b for a, b in zip(W[t], fb))
            kn = sum(w * Y[i][t] for w, (_, i) in zip(ws, ds)) / sw
            out.append((lin + kn) / 2)
        return out
    mu = [sum(y[t] for y in Y) / len(Y) for t in range(2)]
    mpg, sc = [], []
    for r in T:
        x = [num(r[J[k]]) if J[k] < len(r) else None for k in ("Clndrs", "Volume", "Model")]
        o = r[J["origin"]] if J["origin"] < len(r) else "?"
        vol = x[1] is not None
        for k in range(3):
            if x[k] is None: x[k] = stats[k][0]
        try:
            p = pred(x, o, vol)
        except Exception:
            p = mu
        m = exp(p[0])
        if not (m > 0 and m < 1e6): m = exp(mu[0])
        mpg.append(m); sc.append(min(1.0, max(0.0, p[1])))
    return {"mpg": mpg, "scores": sc}

def remember(tr, te):
    _, R = load(tr); h, T = load(te)
    I = {n: i for i, n in enumerate(h)}
    gi = [I["Lbs-"], I["Acc+"], I["Mpg+"]]
    f = d2h_fn([[float(r[i]) for i in gi] for r in R])
    s = [f([float(r[i]) for i in gi]) for r in T]
    order = sorted(range(len(s)), key=lambda i: (s[i], i))
    return {"scores": s, "top": order[:10], "best": order[0],
            "median": order[len(s) // 2], "worst": order[-1]}

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("cmd"); a.add_argument("--train"); a.add_argument("--test")
    a = a.parse_args()
    fn = {"guess": guess, "remember": remember}[a.cmd]
    print(json.dumps(fn(a.train, a.test)))
