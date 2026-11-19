import csv, sys, json, math

def load(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def num(x):
    try:
        return float(x)
    except Exception:
        return None

def main():
    a = sys.argv[1:]
    train_p = a[a.index('--train') + 1]
    test_p = a[a.index('--test') + 1]
    tr, te = load(train_p), load(test_p)
    n = len(tr)

    def stats(vals):
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        return m, sd

    def norm(v, ms):
        m, sd = ms
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    goals = {'Lbs-': 0.0, 'Acc+': 1.0, 'Mpg+': 1.0}
    gs = {g: stats([float(r[g]) for r in tr]) for g in goals}
    d2h = []
    for r in tr:
        s = sum((norm(float(r[g]), gs[g]) - h) ** 2 for g, h in goals.items())
        d2h.append(math.sqrt(s / 3))

    cols = ['Clndrs', 'Volume', 'Model']
    cs = {c: stats([float(r[c]) for r in tr]) for c in cols}

    def feat(r):
        out = []
        for c in cols:
            v = num(r[c])
            out.append(None if v is None else norm(v, cs[c]))
        return out + [r['origin']]

    tf = [feat(r) for r in tr]

    def dist(f, g):
        s = 0.0
        for i in range(3):
            if f[i] is None or g[i] is None:
                s += 1.0
            else:
                s += (f[i] - g[i]) ** 2
        s += 0.0 if f[3] == g[3] else 1.0
        return math.sqrt(s / 4)

    def predict(f):
        ds = sorted((round(dist(f, tf[i]), 12), i) for i in range(n))[:5]
        return sum(d2h[i] for _, i in ds) / len(ds)

    cutoff = sorted(d2h)[n // 2]
    order = sorted(range(n), key=lambda i: (round(d2h[i], 12), i))
    good = order[:max(1, n // 4)]
    menu = []
    seen = set()
    for i in good:
        p = (float(tr[i]['Clndrs']), float(tr[i]['Volume']))
        if p not in seen:
            seen.add(p)
            menu.append(p)
            if len(menu) == 12:
                break

    res = []
    for idx, r in enumerate(te):
        f = feat(r)
        before = predict(f)
        if round(before, 10) <= round(cutoff, 10):
            res.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "already_good"})
            continue
        oc, ov = num(r['Clndrs']), num(r['Volume'])
        best = None
        for nc, nv in menu:
            chg = {}
            if oc is None or nc != oc:
                chg['Clndrs'] = nc
            if ov is None or nv != ov:
                chg['Volume'] = nv
            if not chg:
                continue
            g = list(f)
            g[0] = norm(nc, cs['Clndrs'])
            g[1] = norm(nv, cs['Volume'])
            after = predict(g)
            if round(before - after, 10) >= 0.05:
                key = (len(chg), round(dist(f, g), 10), round(after, 10), (nc, nv))
                if best is None or key < best[0]:
                    best = (key, after, chg)
        if best is None:
            res.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "no_candidate"})
        else:
            res.append({"row": idx, "before": before, "after": best[1], "changes": best[2], "status": "improved"})
    sys.stdout.write(json.dumps({"repairs": res}))

main()
