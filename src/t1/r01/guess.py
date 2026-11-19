import sys, json, csv, math, time

def load(path):
    with open(path, newline='') as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def main():
    t0 = time.time()
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    h, train = load(tr)
    h2, test = load(te)
    idx2 = {n: i for i, n in enumerate(h2)}
    goals = [i for i, n in enumerate(h) if n[-1] in '+-']
    feats = [i for i, n in enumerate(h) if n[-1] not in '+-X']
    isnum = {i: h[i][0].isupper() for i in feats}
    mpg_i = next(i for i in goals if h[i].lower().startswith('mpg'))

    def val(row, i):
        j = idx2.get(h[i])
        if j is None or j >= len(row):
            return None
        v = row[j]
        if v == '?' or v == '':
            return None
        return num(v) if isnum[i] else v if i in isnum else None

    def gval(row, i):
        if i >= len(row) or row[i] in ('?', ''):
            return None
        return num(row[i])

    lo, hi = {}, {}
    for i in feats + goals:
        if i in isnum and not isnum[i]:
            continue
        vs = [x for x in (num(r[i]) if i < len(r) and r[i] != '?' else None for r in train) if x is not None]
        lo[i] = min(vs) if vs else 0
        hi[i] = max(vs) if vs else 1

    def nz(i, x):
        r = hi[i] - lo[i]
        return (x - lo[i]) / r if r else 0.0

    def d2h(vals):
        s = 0
        for i in goals:
            n = nz(i, vals[i])
            n = min(1, max(0, n))
            s += (n - (1 if h[i][-1] == '+' else 0)) ** 2
        return math.sqrt(s / len(goals))

    # prepare training
    T = []
    for r in train:
        fv = {}
        for i in feats:
            v = gval(r, i) if isnum[i] else (r[i] if i < len(r) and r[i] != '?' else None)
            if v is not None:
                fv[i] = nz(i, v) if isnum[i] else v
        gv = {i: gval(r, i) for i in goals}
        full = all(v is not None for v in gv.values())
        T.append((fv, gv, d2h(gv) if full else None))
    # baseline means
    gm = {}
    for i in goals:
        vs = [g[i] for _, g, _ in T if g[i] is not None]
        gm[i] = sum(vs) / len(vs) if vs else 0
    ds = [d for _, _, d in T if d is not None]
    dm = sum(ds) / len(ds) if ds else 0.5

    K = 5
    mpg, scores = [], []
    for k, r in enumerate(test):
        q = {}
        for i in feats:
            j = idx2.get(h[i])
            if j is None or j >= len(r) or r[j] in ('?', ''):
                continue
            if isnum[i]:
                x = num(r[j])
                if x is not None:
                    q[i] = nz(i, x)
            else:
                q[i] = r[j]
        nb = []
        for fv, gv, d in T:
            s, c = 0.0, 0
            for i, x in q.items():
                y = fv.get(i)
                if y is None:
                    continue
                s += (x - y) ** 2 if isnum[i] else (0 if x == y else 1)
                c += 1
            nb.append((math.sqrt(s / c) if c else 1.0, fv, gv, d))
        nb.sort(key=lambda z: z[0])

        def avg(sel, getter, default):
            num_, den = 0.0, 0.0
            cnt = 0
            for z in nb:
                v = getter(z)
                if v is None:
                    continue
                w = 1 / (z[0] + 0.05)
                num_ += w * v
                den += w
                cnt += 1
                if cnt >= K:
                    break
            return num_ / den if den else default

        mpg.append(avg(nb, lambda z: z[2][mpg_i], gm[mpg_i]))
        scores.append(avg(nb, lambda z: z[3], dm))
    print(json.dumps({"mpg": mpg, "scores": scores, "time": time.time() - t0}))

if __name__ == '__main__':
    main()
