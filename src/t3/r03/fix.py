import csv, sys, math, json

def load(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def sig(z):
    return 1 / (1 + math.exp(-1.702 * z))

def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd

def main():
    a = sys.argv[1:]
    train_p = a[a.index('--train') + 1]
    test_p = a[a.index('--test') + 1]
    tr = load(train_p)
    te = load(test_p)
    n = len(tr)
    goals = [('Lbs-', 0.0), ('Acc+', 1.0), ('Mpg+', 1.0)]
    gs = {}
    for g, _ in goals:
        gs[g] = stats([float(r[g]) for r in tr])

    def ng(g, v):
        m, sd = gs[g]
        z = max(-3, min(3, (v - m) / (sd + 1e-32)))
        return sig(z)

    d2h = []
    for r in tr:
        s = sum((ng(g, float(r[g])) - h) ** 2 for g, h in goals)
        d2h.append(math.sqrt(s / 3))

    nums = ['Clndrs', 'Volume', 'Model']
    ist = {c: stats([float(r[c]) for r in tr]) for c in nums}

    def nf(c, v):
        m, sd = ist[c]
        z = max(-3, min(3, (v - m) / (sd + 1e-32)))
        return sig(z)

    trf = [([nf(c, float(r[c])) for c in nums], r['origin']) for r in tr]

    def feat(row):
        return ([nf(c, float(row[c])) for c in nums], row['origin'])

    def dist(fa, fb):
        s = sum((x - y) ** 2 for x, y in zip(fa[0], fb[0]))
        s += 0.0 if fa[1] == fb[1] else 1.0
        return math.sqrt(s / 4)

    def predict(f):
        ds = sorted((round(dist(f, t), 12), i) for i, t in enumerate(trf))[:5]
        return sum(d2h[i] for _, i in ds) / len(ds)

    cutoff = sorted(d2h)[n // 2]
    rank = sorted(range(n), key=lambda i: (round(d2h[i], 12), i))
    good = rank[:max(1, n // 4)]
    menu = []
    seen = set()
    for i in good:
        p = (float(tr[i]['Clndrs']), float(tr[i]['Volume']))
        if p not in seen:
            seen.add(p)
            menu.append(p)
            if len(menu) == 12:
                break

    out = []
    for idx, row in enumerate(te):
        f = feat(row)
        before = predict(f)
        if round(before, 10) <= round(cutoff, 10):
            out.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "already_good"})
            continue
        c0 = float(row['Clndrs'])
        v0 = float(row['Volume'])
        best = None
        for nc, nv in menu:
            ch = {}
            if nc != c0:
                ch['Clndrs'] = nc
            if nv != v0:
                ch['Volume'] = nv
            if not ch:
                continue
            e = dict(row)
            e['Clndrs'] = nc
            e['Volume'] = nv
            fe = feat(e)
            after = predict(fe)
            if round(before - after, 10) >= 0.05:
                key = (len(ch), round(dist(f, fe), 10), round(after, 10), (nc, nv))
                if best is None or key < best[0]:
                    best = (key, after, ch)
        if best is None:
            out.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "no_candidate"})
        else:
            out.append({"row": idx, "before": before, "after": best[1], "changes": best[2], "status": "improved"})
    sys.stdout.write(json.dumps({"repairs": out}))

if __name__ == '__main__':
    main()
