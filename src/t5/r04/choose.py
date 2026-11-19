import sys, csv, json, math

def num(s):
    try: return float(s)
    except ValueError: return None

def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if r]
    return [h.strip() for h in rows[0]], [[c.strip() for c in r] for r in rows[1:]]

def main():
    a = sys.argv
    tr, te = a[a.index("--train") + 1], a[a.index("--test") + 1]
    hd, train = read(tr)
    _, test = read(te)
    ix = {h: i for i, h in enumerate(hd)}
    goals = ["Lbs-", "Acc+", "Mpg+"]
    heaven = [0, 1, 1]

    def stats(vals):
        n = len(vals)
        mu = sum(vals) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0
        return mu, sd

    gs = []
    for g in goals:
        vals = [num(r[ix[g]]) for r in train]
        gs.append(stats([v for v in vals if v is not None]))

    def d2h(vs):
        t = 0
        for v, (mu, sd), h in zip(vs, gs, heaven):
            z = max(-3, min(3, (v - mu) / (sd + 1e-32)))
            t += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
        return math.sqrt(t / 3)

    tg = [[num(r[ix[g]]) for g in goals] for r in train]
    keep = [i for i, v in enumerate(tg) if None not in v]
    ydt = {i: d2h(tg[i]) for i in keep}

    nums = ["Clndrs", "Volume", "Model"]
    ns = {}
    for c in nums:
        ns[c] = stats([v for v in (num(r[ix[c]]) for r in train) if v is not None])

    def feat(r):
        f = []
        for c in nums:
            v = num(r[ix[c]])
            mu, sd = ns[c]
            f.append(None if v is None else max(-3, min(3, (v - mu) / (sd + 1e-32))))
        o = r[ix["origin"]]
        f.append(None if o == "?" else o)
        return f

    def dist(p, q):
        s = 0
        for x, y in zip(p, q):
            if x is None or y is None: s += 1
            elif isinstance(x, str): s += x != y
            else: s += (abs(x - y) / 2) ** 2
        return math.sqrt(s / len(p))

    tf = {i: feat(train[i]) for i in keep}
    preds = []
    for r in test:
        f = feat(r)
        ds = sorted((dist(f, tf[i]), i) for i in keep)[:7]
        w = [1 / (d + 0.05) for d, _ in ds]
        preds.append(sum(wi * ydt[i] for wi, (_, i) in zip(w, ds)) / sum(w) if ds else 0.5)

    order = sorted(range(len(test)), key=lambda i: (preds[i], i))[:10]
    obs = {}
    for i in order:
        print(json.dumps({"inspect": i}), flush=True)
        resp = json.loads(sys.stdin.readline())
        g = resp["goals"]
        obs[i] = d2h([g[k] for k in goals])
    top = sorted(obs, key=lambda i: (round(obs[i], 12), i))[:3]
    print(json.dumps({"selected": top}), flush=True)

main()
