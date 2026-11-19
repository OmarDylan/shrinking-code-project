import csv, sys, json, math

NUM = ["Clndrs", "Volume", "Model"]

def load(p):
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in r]

def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def med(a):
    a = sorted(a); n = len(a)
    return a[n // 2] if n % 2 else (a[n // 2 - 1] + a[n // 2]) / 2

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        d = M[i][i] or 1e-12
        for r in range(n):
            if r != i:
                f = M[r][i] / d
                for c in range(i, n + 1):
                    M[r][c] -= f * M[i][c]
    return [M[i][n] / (M[i][i] or 1e-12) for i in range(n)]

def main():
    a = sys.argv[1:]
    tr = load(a[a.index("--train") + 1])
    te = load(a[a.index("--test") + 1])
    cols = [[num(r.get(c, "?")) for r in tr] for c in NUM]
    st = []
    for col in cols:
        v = [x for x in col if x is not None]
        if not v:
            st.append((0, 1, 0, 1)); continue
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) or 1.0
        md = med(v)
        mad = 1.4826 * med([abs(x - md) for x in v])
        if mad < 1e-9:
            mad = sd
        st.append((m, sd, md, mad))
    # correlation matrix on complete rows
    comp = [[(cols[j][i] - st[j][0]) / st[j][1] for j in range(3)]
            for i in range(len(tr)) if all(cols[j][i] is not None for j in range(3))]
    C = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            C[i][j] = sum(r[i] * r[j] for r in comp) / max(1, len(comp) - 1)
        C[i][i] += 0.02
    cnt = {}
    for r in tr:
        o = r.get("origin", "?")
        if o != "?":
            cnt[o] = cnt.get(o, 0) + 1
    tot = sum(cnt.values()) or 1

    def raw(r):
        x = 0.0
        idx, z = [], []
        for j, c in enumerate(NUM):
            v = num(r.get(c, "?"))
            if v is None:
                continue
            x = max(x, abs(v - st[j][2]) / st[j][3])
            idx.append(j); z.append((v - st[j][0]) / st[j][1])
        if idx:
            sub = [[C[i][j] for j in idx] for i in idx]
            y = solve(sub, z)
            d2 = max(0.0, sum(p * q for p, q in zip(z, y)))
            x = max(x, math.sqrt(d2))
        o = r.get("origin", "?")
        if o != "?":
            f = cnt.get(o, 0) / tot
            if f == 0:
                x = max(x, 8.0)
            elif f < 0.02:
                x = max(x, 3.0 - math.log(f / 0.02))
        return x

    f = lambda x: min(1.0, max(0.0, 1 - math.exp(-x / 5.0)))
    ts = sorted(f(raw(r)) for r in tr)
    if ts:
        k = min(len(ts) - 1, int(math.ceil(0.97 * len(ts))) - 1)
        th = ts[max(0, k)]
    else:
        th = 0.5
    sc = [f(raw(r)) for r in te]
    print(json.dumps({"scores": sc, "alerts": [s > th for s in sc], "threshold": th}))

main()
