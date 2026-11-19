import csv, sys, math, json

def read(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd

def norm(v, m, sd):
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))

def main():
    a = sys.argv[1:]
    train_p = a[a.index('--train') + 1]
    test_p = a[a.index('--test') + 1]
    tr, te = read(train_p), read(test_p)
    goals = [('Lbs-', 0.0), ('Acc+', 1.0), ('Mpg+', 1.0)]
    gs = {g: stats([float(r[g]) for r in tr]) for g, _ in goals}
    d2h = []
    for r in tr:
        s = 0.0
        for g, h in goals:
            s += (norm(float(r[g]), *gs[g]) - h) ** 2
        d2h.append(math.sqrt(s / 3))
    cols = ['Clndrs', 'Volume', 'Model']
    ns = {c: stats([float(r[c]) for r in tr]) for c in cols}

    def feat(r, ov=None):
        f = []
        for c in cols:
            v = num(r[c]) if ov is None or c not in ov else ov[c]
            f.append(None if v is None else norm(v, *ns[c]))
        return f, r['origin']

    trf = [feat(r) for r in tr]

    def dist(A, B):
        s = 0.0
        for x, y in zip(A[0], B[0]):
            if x is not None and y is not None:
                s += (x - y) ** 2
        if A[1] != B[1]:
            s += 1
        return math.sqrt(s / 4)

    def pred(F):
        ds = sorted(range(len(tr)), key=lambda i: (round(dist(F, trf[i]), 12), i))[:5]
        return sum(d2h[i] for i in ds) / len(ds)

    n = len(tr)
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

    out = []
    for idx, r in enumerate(te):
        F = feat(r)
        before = pred(F)
        res = {"row": idx, "before": before, "after": before, "changes": {}, "status": "already_good"}
        if round(before, 10) > round(cutoff, 10):
            res["status"] = "no_candidate"
            oc, ov = num(r['Clndrs']), num(r['Volume'])
            best = None
            for c, v in menu:
                ch = {}
                if oc is None or c != oc:
                    ch['Clndrs'] = c
                if ov is None or v != ov:
                    ch['Volume'] = v
                if not ch:
                    continue
                G = feat(r, {'Clndrs': c, 'Volume': v})
                after = pred(G)
                if round(before - after, 10) >= 0.05:
                    key = (len(ch), round(dist(F, G), 10), round(after, 10), (c, v))
                    if best is None or key < best[0]:
                        best = (key, after, ch)
            if best:
                res["status"] = "improved"
                res["after"] = best[1]
                res["changes"] = best[2]
        out.append(res)
    sys.stdout.write(json.dumps({"repairs": out}))

main()
