import sys, csv, json, math

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]

    def load(p):
        with open(p, newline='') as f:
            r = list(csv.reader(f))
        h = [x.strip() for x in r[0]]
        return h, [[c.strip() for c in row] for row in r[1:] if row]

    h, T = load(tr)
    h2, S = load(te)
    ix = {n: i for i, n in enumerate(h)}
    jx = {n: i for i, n in enumerate(h2)}
    goals = ['Lbs-', 'Acc+', 'Mpg+']
    heav = [0, 1, 1]

    def num(s):
        try:
            return float(s)
        except Exception:
            return None

    st = []
    for g in goals:
        v = [num(r[ix[g]]) for r in T]
        v = [x for x in v if x is not None]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
        st.append((m, sd))

    def nm(v, k):
        m, sd = st[k]
        z = max(-3, min(3, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    def d2h(vals):
        return math.sqrt(sum((nm(v, k) - heav[k]) ** 2 for k, v in enumerate(vals)) / 3)

    nf = ['Clndrs', 'Volume', 'Model']
    fs = []
    for f in nf:
        v = [num(r[ix[f]]) for r in T]
        v = [x for x in v if x is not None]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) or 1
        fs.append((m, sd))

    def feat(row, idx):
        out = []
        for k, f in enumerate(nf):
            x = num(row[idx[f]])
            out.append(None if x is None else (x - fs[k][0]) / fs[k][1])
        o = row[idx['origin']]
        out.append(None if o == '?' or o == '' else o)
        return out

    pool = []
    for r in T:
        gv = [num(r[ix[g]]) for g in goals]
        if None in gv:
            continue
        pool.append((feat(r, ix), d2h(gv)))

    def dist(x, y):
        s = c = 0
        for p, q in zip(x, y):
            if p is None or q is None:
                continue
            c += 1
            if isinstance(p, str):
                s += 0 if p == q else 1
            else:
                s += (p - q) ** 2
        return math.sqrt(s / c) if c else 10

    def pred(x):
        ds = sorted((dist(x, f), y) for f, y in pool)[:5]
        w = [1 / (d + 0.05) for d, _ in ds]
        return sum(wi * y for wi, (_, y) in zip(w, ds)) / sum(w)

    F = [feat(r, jx) for r in S]
    seen = {}
    n = len(S)
    for _ in range(min(10, n)):
        best = min((i for i in range(n) if i not in seen), key=lambda i: (pred(F[i]), i))
        print(json.dumps({"inspect": best}), flush=True)
        resp = json.loads(sys.stdin.readline())
        g = resp["goals"]
        d = d2h([g[k] for k in goals])
        seen[best] = d
        pool.append((F[best], d))
    top = sorted(seen, key=lambda i: (round(seen[i], 12), i))[:3]
    print(json.dumps({"selected": top}), flush=True)

main()
