import csv, sys, json, math

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def stats(vals):
    n = len(vals)
    mu = sum(vals) / n if n else 0.0
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd

def norm(st, v):
    z = max(-3, min(3, (v - st[0]) / (st[1] + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))

def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    hdr, train = load(tr)
    _, test = load(te)
    ix = {h: i for i, h in enumerate(hdr)}
    goals = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
    nums = ["Clndrs", "Volume", "Model"]
    cat = "origin"
    gst = {g: stats([float(r[ix[g]]) for r in train]) for g in goals}

    def d2h(r):
        return math.sqrt(sum((norm(gst[g], float(r[ix[g]])) - h) ** 2
                             for g, h in goals.items()) / 3)

    order = sorted(range(len(train)), key=lambda i: (round(d2h(train[i]), 12), i))
    q = max(1, len(train) // 4)
    good = [train[i] for i in order[:q]]
    poor = [train[i] for i in order[-q:]]

    def known(rows, c):
        return [r[ix[c]] for r in rows if r[ix[c]] != "?"]

    nst, cen = {}, {}
    for c in nums:
        nst[c] = stats([float(v) for v in known(train, c)])
        cs = []
        for grp in (good, poor):
            k = [float(v) for v in known(grp, c)]
            cs.append(norm(nst[c], sum(k) / len(k)) if k else None)
        cen[c] = cs
    freq = []
    for grp in (good, poor):
        k = known(grp, cat)
        d = {}
        for v in k:
            d[v] = d.get(v, 0) + 1
        freq.append({v: c / len(k) for v, c in d.items()} if k else {})

    out = []
    for r in test:
        gs = {}
        for g, h in goals.items():
            v = r[ix[g]]
            gs[g] = abs(norm(gst[g], float(v)) - h) if v != "?" else 0.0
        fs = {}
        for c in nums:
            v = r[ix[c]]
            gc, pc = cen[c]
            if v == "?" or gc is None or pc is None:
                fs[c] = 0.0
            else:
                x = norm(nst[c], float(v))
                fs[c] = max(0.0, abs(x - gc) - abs(x - pc))
        v = r[ix[cat]]
        fs[cat] = 0.0 if v == "?" else max(0.0, freq[1].get(v, 0.0) - freq[0].get(v, 0.0))
        srt = lambda d: [k for k in sorted(d, key=lambda k: (-round(d[k], 10), k))]
        out.append({"goals": gs, "features": fs,
                    "goal_order": srt(gs), "feature_order": srt(fs)})
    print(json.dumps({"diagnoses": out}))

main()
