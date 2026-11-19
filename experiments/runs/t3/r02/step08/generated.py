import sys, csv, json, math

def load(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def fnum(s):
    try:
        return float(s)
    except Exception:
        return None

def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd

def norm(v, ms):
    m, sd = ms
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))

def main():
    a = sys.argv[1:]
    train = load(a[a.index('--train') + 1])
    test = load(a[a.index('--test') + 1])
    goals = [('Lbs-', 0.0), ('Acc+', 1.0), ('Mpg+', 1.0)]
    gs = {g: stats([float(r[g]) for r in train]) for g, _ in goals}
    ins = ['Clndrs', 'Volume', 'Model']
    ist = {c: stats([float(r[c]) for r in train]) for c in ins}

    def d2h(r):
        return math.sqrt(sum((norm(float(r[g]), gs[g]) - h) ** 2 for g, h in goals) / 3)

    def feat(c, v, m, o):
        return (norm(c, ist['Clndrs']), norm(v, ist['Volume']), norm(m, ist['Model']), o)

    def dist(x, y):
        s = sum((x[i] - y[i]) ** 2 for i in range(3)) + (0 if x[3] == y[3] else 1)
        return math.sqrt(s / 4)

    tf = [feat(float(r['Clndrs']), float(r['Volume']), float(r['Model']), r['origin']) for r in train]
    td = [d2h(r) for r in train]

    def pred(x):
        ds = sorted(range(len(tf)), key=lambda i: (round(dist(x, tf[i]), 12), i))
        return sum(td[i] for i in ds[:5]) / len(ds[:5])

    cutoff = sorted(td)[len(td) // 2]
    q = max(1, len(td) // 4)
    good = sorted(range(len(td)), key=lambda i: (round(td[i], 12), i))[:q]
    menu = []
    for i in good:
        p = (float(train[i]['Clndrs']), float(train[i]['Volume']))
        if p not in menu:
            menu.append(p)
        if len(menu) == 12:
            break

    out = []
    for k, r in enumerate(test):
        c, v = float(r['Clndrs']), float(r['Volume'])
        m, o = float(r['Model']), r['origin']
        x = feat(c, v, m, o)
        before = pred(x)
        if round(before, 10) <= round(cutoff, 10):
            out.append({"row": k, "before": before, "after": before, "changes": {}, "status": "already_good"})
            continue
        best = None
        for nc, nv in menu:
            if nc == c and nv == v:
                continue
            y = feat(nc, nv, m, o)
            after = pred(y)
            if round(before - after, 10) >= 0.05:
                ch = {}
                if nc != c:
                    ch['Clndrs'] = nc
                if nv != v:
                    ch['Volume'] = nv
                key = (len(ch), round(dist(x, y), 10), round(after, 10), (nc, nv))
                if best is None or key < best[0]:
                    best = (key, after, ch)
        if best is None:
            out.append({"row": k, "before": before, "after": before, "changes": {}, "status": "no_candidate"})
        else:
            out.append({"row": k, "before": before, "after": best[1], "changes": best[2], "status": "improved"})
    print(json.dumps({"repairs": out}))

main()
