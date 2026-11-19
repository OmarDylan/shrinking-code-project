import sys, csv, json, math

def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def num(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def stats(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd

def norm(st, v):
    z = max(-3, min(3, (v - st[0]) / (st[1] + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))

def main():
    a = sys.argv[1:]
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    hn, trows = load(tr)
    tn, terows = load(te)
    ix = {n: i for i, n in enumerate(hn)}
    tx = {n: i for i, n in enumerate(tn)}
    goals = [("Lbs-", 0), ("Acc+", 1), ("Mpg+", 1)]
    gst = {g: stats([num(r[ix[g]]) for r in trows]) for g, _ in goals}
    d2h = []
    for r in trows:
        s = sum((norm(gst[g], num(r[ix[g]])) - h) ** 2 for g, h in goals)
        d2h.append(math.sqrt(s / 3))
    nums = ["Clndrs", "Volume", "Model"]
    nst = {c: stats([num(r[ix[c]]) for r in trows]) for c in nums}
    trf = [[norm(nst[c], num(r[ix[c]])) for c in nums] + [r[ix["origin"]]] for r in trows]
    n = len(trows)

    def feat(c, v, m, o):
        f = []
        for name, val in zip(nums, (c, v, m)):
            f.append(None if val is None else norm(nst[name], val))
        f.append(o)
        return f

    def predict(f):
        ds = []
        for i, t in enumerate(trf):
            s = 0.0
            for k in range(3):
                if f[k] is None or t[k] is None:
                    g = 1.0
                else:
                    g = f[k] - t[k]
                s += g * g
            if f[3] != t[3]:
                s += 1
            ds.append((round(math.sqrt(s / 4), 12), i))
        ds.sort()
        return sum(d2h[i] for _, i in ds[:5]) / 5

    def fdist(f, g):
        s = 0.0
        for k in range(3):
            d = 1.0 if f[k] is None or g[k] is None else f[k] - g[k]
            s += d * d
        if f[3] != g[3]:
            s += 1
        return math.sqrt(s / 4)

    cutoff = sorted(d2h)[n // 2]
    order = sorted(range(n), key=lambda i: (round(d2h[i], 12), i))
    good = order[:max(1, n // 4)]
    menu = []
    seen = set()
    for i in good:
        p = (num(trows[i][ix["Clndrs"]]), num(trows[i][ix["Volume"]]))
        if p not in seen:
            seen.add(p)
            menu.append(p)
            if len(menu) == 12:
                break

    out = []
    for ri, r in enumerate(terows):
        c0 = num(r[tx["Clndrs"]])
        v0 = num(r[tx["Volume"]])
        m0 = num(r[tx["Model"]])
        o0 = r[tx["origin"]]
        f0 = feat(c0, v0, m0, o0)
        before = predict(f0)
        if round(before, 10) <= round(cutoff, 10):
            out.append({"row": ri, "before": before, "after": before, "changes": {}, "status": "already_good"})
            continue
        best = None
        for c, v in menu:
            ch = {}
            if c != c0:
                ch["Clndrs"] = c
            if v != v0:
                ch["Volume"] = v
            if not ch:
                continue
            f1 = feat(c, v, m0, o0)
            after = predict(f1)
            if round(before - after, 10) >= 0.05:
                key = (len(ch), round(fdist(f0, f1), 10), round(after, 10), (c, v))
                if best is None or key < best[0]:
                    best = (key, after, ch)
        if best is None:
            out.append({"row": ri, "before": before, "after": before, "changes": {}, "status": "no_candidate"})
        else:
            out.append({"row": ri, "before": before, "after": best[1], "changes": best[2], "status": "improved"})
    sys.stdout.write(json.dumps({"repairs": out}))

if __name__ == "__main__":
    main()
