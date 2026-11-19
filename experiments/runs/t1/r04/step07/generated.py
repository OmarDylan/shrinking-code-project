import csv, sys, math, json

def load(p):
    with open(p, newline='') as f:
        r = csv.reader(f)
        h = [x.strip() for x in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return h, rows

def num(v):
    try:
        if v == '?' or v == '':
            return None
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    h, T = load(tr)
    h2, S = load(te)
    G = ['Lbs-', 'Acc+', 'Mpg+']
    heaven = [0, 1, 1]
    ti = {n: i for i, n in enumerate(h)}
    si = {n: i for i, n in enumerate(h2)}

    def get(row, idx, n):
        i = idx.get(n)
        return row[i] if i is not None and i < len(row) else '?'

    stats = []
    for g in G:
        vs = [num(get(r, ti, g)) for r in T]
        vs = [v for v in vs if v is not None]
        m = sum(vs) / len(vs)
        sd = math.sqrt(sum((v - m) ** 2 for v in vs) / (len(vs) - 1)) if len(vs) > 1 else 0.0
        stats.append((m, sd))

    def d2h(vals):
        s = 0
        for v, (m, sd), hv in zip(vals, stats, heaven):
            z = max(-3, min(3, (v - m) / (sd + 1e-32)))
            n = 1 / (1 + math.exp(-1.702 * z))
            s += (n - hv) ** 2
        return math.sqrt(s / 3)

    NF = ['Clndrs', 'Volume', 'Model']
    fst = {}
    for f in NF:
        vs = [num(get(r, ti, f)) for r in T]
        vs = [v for v in vs if v is not None]
        m = sum(vs) / len(vs) if vs else 0
        sd = math.sqrt(sum((v - m) ** 2 for v in vs) / (len(vs) - 1)) if len(vs) > 1 else 1
        fst[f] = (m, sd or 1)

    def feat(row, idx):
        out = []
        for f in NF:
            v = num(get(row, idx, f))
            out.append(None if v is None else (v - fst[f][0]) / fst[f][1])
        o = get(row, idx, 'origin')
        out.append(None if o == '?' else o)
        return out

    train = []
    for r in T:
        gv = [num(get(r, ti, g)) for g in G]
        if any(v is None for v in gv):
            continue
        train.append((feat(r, ti), d2h(gv)))

    def dist(x, y):
        s = 0.0
        c = 0
        for i in range(3):
            if x[i] is not None and y[i] is not None:
                s += (x[i] - y[i]) ** 2
                c += 1
        if x[3] is not None and y[3] is not None:
            s += 0 if x[3] == y[3] else 1
            c += 1
        return math.sqrt(s * 4 / c) if c else 10.0

    def pred(x):
        ds = sorted((dist(x, f), d) for f, d in train)[:5]
        w = [1 / (d + 0.1) for d, _ in ds]
        return sum(wi * v for wi, (_, v) in zip(w, ds)) / sum(w)

    preds = [(pred(feat(r, si)), i) for i, r in enumerate(S)]
    preds.sort()
    picks = [i for _, i in preds[:min(10, len(S))]]
    seen = []
    for i in picks:
        print(json.dumps({"inspect": i}))
        sys.stdout.flush()
        line = sys.stdin.readline()
        resp = json.loads(line)
        gs = resp["goals"]
        seen.append((round(d2h([float(gs[g]) for g in G]), 12), i))
    seen.sort()
    print(json.dumps({"selected": [i for _, i in seen[:3]]}))
    sys.stdout.flush()

main()
