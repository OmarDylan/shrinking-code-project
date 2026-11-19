import csv, sys, math, json

def read(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def num(v):
    try:
        return float(v)
    except Exception:
        return None

def main():
    a = sys.argv[1:]
    train = read(a[a.index('--train') + 1])
    test = read(a[a.index('--test') + 1])
    goals = [('Lbs-', 0), ('Acc+', 1), ('Mpg+', 1)]
    nums = ['Clndrs', 'Volume', 'Model']

    def stats(col):
        v = [float(r[col]) for r in train]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        return m, sd

    st = {c: stats(c) for c in nums + [g for g, _ in goals]}

    def norm(c, v):
        m, sd = st[c]
        z = max(-3, min(3, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    def d2h(r):
        return math.sqrt(sum((norm(g, float(r[g])) - h) ** 2 for g, h in goals) / 3)

    def feat(r):
        return ([num(r[c]) for c in nums], r['origin'])

    tf = [feat(r) for r in train]
    tn = [[norm(c, x) if x is not None else None for c, x in zip(nums, f[0])] for f in tf]
    td = [d2h(r) for r in train]

    def dist(n1, o1, n2, o2):
        s = 0.0
        for x, y in zip(n1, n2):
            g = 1.0 if x is None or y is None else x - y
            s += g * g
        s += 0.0 if o1 == o2 else 1.0
        return math.sqrt(s / 4)

    def pred(n, o):
        ds = sorted((round(dist(n, o, tn[i], tf[i][1]), 12), i) for i in range(len(train)))
        return sum(td[i] for _, i in ds[:5]) / 5

    N = len(train)
    cutoff = sorted(td)[N // 2]
    rank = sorted(range(N), key=lambda i: (round(td[i], 12), i))
    good = rank[:max(1, N // 4)]
    menu = []
    for i in good:
        p = (tf[i][0][0], tf[i][0][1])
        if p not in menu:
            menu.append(p)
        if len(menu) == 12:
            break

    out = []
    for idx, r in enumerate(test):
        raw, o = feat(r)
        n = [norm(c, x) if x is not None else None for c, x in zip(nums, raw)]
        before = pred(n, o)
        if round(before, 10) <= round(cutoff, 10):
            out.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "already_good"})
            continue
        best = None
        for c, v in menu:
            ch = {}
            if c != raw[0]:
                ch['Clndrs'] = c
            if v != raw[1]:
                ch['Volume'] = v
            if not ch:
                continue
            e = list(raw)
            e[0], e[1] = c, v
            en = [norm(cc, x) if x is not None else None for cc, x in zip(nums, e)]
            after = pred(en, o)
            if round(before - after, 10) >= 0.05:
                key = (len(ch), round(dist(n, o, en, o), 10), round(after, 10), (c, v))
                if best is None or key < best[0]:
                    best = (key, after, ch)
        if best:
            out.append({"row": idx, "before": before, "after": best[1], "changes": best[2], "status": "improved"})
        else:
            out.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "no_candidate"})
    sys.stdout.write(json.dumps({"repairs": out}))

main()
