import sys, json, csv, math

def num(s):
    try:
        return float(s)
    except ValueError:
        return s.strip()

def read(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], [[num(c) for c in r] for r in rows[1:]]

def main():
    a = sys.argv
    tr, te = a[a.index("--train") + 1], a[a.index("--test") + 1]
    hn, train = read(tr)
    tn, test = read(te)
    G = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
    st = {}
    for g in G:
        v = [r[hn.index(g)] for r in train if isinstance(r[hn.index(g)], float)]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0
        st[g] = (m, sd)

    def d2h(goals):
        s = 0
        for g, h in G.items():
            m, sd = st[g]
            z = max(-3, min(3, (goals[g] - m) / (sd + 1e-32)))
            s += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
        return math.sqrt(s / 3)

    feats = ["Clndrs", "Volume", "Model", "origin"]
    cat = {"origin"}
    ti = [hn.index(f) for f in feats]
    ei = [tn.index(f) for f in feats]
    rng = []
    for k, f in enumerate(feats):
        if f in cat:
            rng.append(1)
            continue
        v = [r[ti[k]] for r in train if isinstance(r[ti[k]], float)]
        rng.append((max(v) - min(v)) or 1 if v else 1)

    def fx(r, idx):
        return [r[i] for i in idx]

    pool = []
    for r in train:
        gs = {g: r[hn.index(g)] for g in G}
        if all(isinstance(x, float) for x in gs.values()):
            pool.append((fx(r, ti), d2h(gs), 1.0))
    tx = [fx(r, ei) for r in test]

    def dist(p, q):
        s = 0
        for k, f in enumerate(feats):
            x, y = p[k], q[k]
            if x == "?" or y == "?" or isinstance(x, str) != isinstance(y, str):
                d = 1
            elif f in cat:
                d = float(x != y)
            else:
                d = abs(x - y) / rng[k]
            s += d * d
        return math.sqrt(s / len(feats))

    def pred(x):
        ds = sorted(((dist(x, p[0]), p[1], p[2]) for p in pool), key=lambda t: t[0])[:5]
        w = [p[2] / (d + 0.05) for d, _, p in [(t[0], 0, t[2]) for t in ds]]
        return sum(wi * t[1] for wi, t in zip(w, ds)) / sum(w)

    n = len(test)
    seen = {}
    for _ in range(min(10, n)):
        best = min((i for i in range(n) if i not in seen), key=lambda i: (pred(tx[i]), i))
        print(json.dumps({"inspect": best}), flush=True)
        resp = json.loads(sys.stdin.readline())
        seen[best] = d2h(resp["goals"])
        pool.append((tx[best], seen[best], 3.0))
    top = sorted(seen, key=lambda i: (round(seen[i], 12), i))[:3]
    print(json.dumps({"selected": top}), flush=True)

main()
