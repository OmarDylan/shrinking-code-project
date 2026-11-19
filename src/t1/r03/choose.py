import sys, csv, json, math

def load(p):
    with open(p, newline='') as f:
        r = list(csv.reader(f))
    return [h.strip() for h in r[0]], [x for x in r[1:] if x]

def num(v):
    try:
        v = v.strip()
        return None if v == '?' or v == '' else float(v)
    except Exception:
        return None

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    th, trows = load(tr)
    eh, erows = load(te)
    G = {'Lbs-': 0, 'Acc+': 1, 'Mpg+': 1}
    stats = {}
    for g in G:
        c = th.index(g)
        vs = [num(r[c]) for r in trows]
        vs = [v for v in vs if v is not None]
        m = sum(vs) / len(vs)
        sd = math.sqrt(sum((v - m) ** 2 for v in vs) / (len(vs) - 1)) if len(vs) > 1 else 0
        stats[g] = (m, sd)

    def d2h(d):
        s = 0
        for g, h in G.items():
            m, sd = stats[g]
            z = max(-3, min(3, (d[g] - m) / (sd + 1e-32)))
            s += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
        return math.sqrt(s / 3)

    nf = ['Clndrs', 'Volume', 'Model']
    ti = {n: th.index(n) for n in nf + ['origin']}
    ei = {n: eh.index(n) for n in nf + ['origin']}
    fs = {}
    for n in nf:
        vs = [num(r[ti[n]]) for r in trows]
        vs = [v for v in vs if v is not None]
        m = sum(vs) / len(vs)
        sd = math.sqrt(sum((v - m) ** 2 for v in vs) / max(1, len(vs) - 1)) or 1
        fs[n] = (m, sd)

    def feat(r, idx):
        f = [(num(r[idx[n]]) if idx[n] < len(r) else None) for n in nf]
        f = [None if v is None else (v - fs[n][0]) / fs[n][1] for v, n in zip(f, nf)]
        o = r[idx['origin']].strip() if idx['origin'] < len(r) else '?'
        return f, o

    T = []
    for r in trows:
        d = {g: num(r[th.index(g)]) for g in G}
        if any(v is None for v in d.values()):
            continue
        f, o = feat(r, ti)
        T.append((f, o, d2h(d)))

    def pred(r):
        f, o = feat(r, ei)
        ds = []
        for tf, to, y in T:
            s = 0
            for x, z in zip(f, tf):
                if x is not None and z is not None:
                    s += (x - z) ** 2
            if o != '?' and to != '?':
                s += 0 if o == to else 1
            ds.append((s, y))
        ds.sort(key=lambda t: t[0])
        k = ds[:7]
        w = [1 / (math.sqrt(s) + 0.1) for s, _ in k]
        return sum(wi * y for wi, (_, y) in zip(w, k)) / sum(w)

    preds = [pred(r) for r in erows]
    order = sorted(range(len(erows)), key=lambda i: (preds[i], i))
    pick = order[:10]
    obs = []
    for i in pick:
        print(json.dumps({"inspect": i}), flush=True)
        line = sys.stdin.readline()
        resp = json.loads(line)
        obs.append((round(d2h(resp["goals"]), 12), i))
    obs.sort()
    print(json.dumps({"selected": [i for _, i in obs[:3]]}), flush=True)

main()
