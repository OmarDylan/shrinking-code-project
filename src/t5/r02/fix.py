import sys, csv, json, math

def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def num(v):
    try:
        return float(v)
    except ValueError:
        return None

def main():
    a = sys.argv
    tr, te = a[a.index("--train") + 1], a[a.index("--test") + 1]
    h, R = load(tr)
    th, T = load(te)
    hi = {n: i for i, n in enumerate(h)}
    ti = {n: i for i, n in enumerate(th)}
    goals = [("Lbs-", 0), ("Acc+", 1), ("Mpg+", 1)]
    nums = ["Clndrs", "Volume", "Model"]

    def stats(vals):
        vals = [v for v in vals if v is not None]
        n = len(vals)
        mu = sum(vals) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        return mu, sd

    def nz(v, st, k=1.702):
        z = max(-3, min(3, (v - st[0]) / (st[1] + 1e-32)))
        return 1 / (1 + math.exp(-k * z))

    gst = {g: stats([num(r[hi[g]]) for r in R]) for g, _ in goals}
    d2h = []
    for r in R:
        s = 0
        for g, hv in goals:
            s += (nz(num(r[hi[g]]), gst[g]) - hv) ** 2
        d2h.append(math.sqrt(s / 3))
    xst = {c: stats([num(r[hi[c]]) for r in R]) for c in nums}

    def nrm(c, v):
        return nz(v, xst[c], 1.7)

    # feature vectors: (Clndrs, Volume, Model normalized or None, origin)
    def vec(row, idx):
        out = []
        for c in nums:
            v = num(row[idx[c]])
            out.append(None if v is None else nrm(c, v))
        out.append(row[idx["origin"]])
        return out

    def d(x, y):
        s = 0
        for i in range(3):
            if x[i] is None or y[i] is None:
                g = 1
            else:
                g = x[i] - y[i]
            s += g * g
        if x[3] != y[3]:
            s += 1
        return math.sqrt(s / 4)

    TV = [vec(r, hi) for r in R]
    n = len(R)

    def pred(x):
        order = sorted(range(n), key=lambda i: (round(d(x, TV[i]), 12), i))[:5]
        return sum(d2h[i] for i in order) / len(order)

    cutoff = sorted(d2h)[n // 2]
    rank = sorted(range(n), key=lambda i: (round(d2h[i], 12), i))
    good = rank[:max(1, n // 4)]
    menu, seen = [], set()
    for i in good:
        p = (num(R[i][hi["Clndrs"]]), num(R[i][hi["Volume"]]))
        if p not in seen:
            seen.add(p)
            menu.append(p)
            if len(menu) == 12:
                break

    reps = []
    for j, row in enumerate(T):
        x = vec(row, ti)
        before = pred(x)
        if round(before, 10) <= round(cutoff, 10):
            reps.append({"row": j, "before": before, "after": before,
                         "changes": {}, "status": "already_good"})
            continue
        oc, ov = num(row[ti["Clndrs"]]), num(row[ti["Volume"]])
        best = None
        for c, v in menu:
            ch = {}
            if c != oc:
                ch["Clndrs"] = c
            if v != ov:
                ch["Volume"] = v
            if not ch:
                continue
            y = list(x)
            y[0] = nrm("Clndrs", c)
            y[1] = nrm("Volume", v)
            after = pred(y)
            if round(before - after, 10) >= 0.05:
                key = (len(ch), round(d(x, y), 10), round(after, 10), (c, v))
                if best is None or key < best[0]:
                    best = (key, after, ch)
        if best:
            reps.append({"row": j, "before": before, "after": best[1],
                         "changes": best[2], "status": "improved"})
        else:
            reps.append({"row": j, "before": before, "after": before,
                         "changes": {}, "status": "no_candidate"})
    print(json.dumps({"repairs": reps}))

main()
