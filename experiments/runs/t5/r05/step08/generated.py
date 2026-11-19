import sys, json, math

def atom(s):
    s = s.strip()
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    return s

def read(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [l.rstrip("\r\n") for l in f if l.strip()]
    hdr = [h.strip() for h in lines[0].split(",")]
    return hdr, [[atom(c) for c in l.split(",")] for l in lines[1:]]

def stats(vals):
    v = [x for x in vals if x != "?"]
    n = len(v)
    mu = sum(v) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return mu, sd

def nrm(st, v):
    z = max(-3, min(3, (v - st[0]) / (st[1] + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))

def main():
    a = sys.argv
    tr_p, te_p = a[a.index("--train") + 1], a[a.index("--test") + 1]
    h, tr = read(tr_p)
    th, te = read(te_p)
    ti = {n: i for i, n in enumerate(h)}
    ei = {n: i for i, n in enumerate(th)}
    goals = [("Lbs-", 0), ("Acc+", 1), ("Mpg+", 1)]
    gst = {g: stats([r[ti[g]] for r in tr]) for g, _ in goals}
    d2h = []
    for r in tr:
        s = sum((nrm(gst[g], r[ti[g]]) - hv) ** 2 for g, hv in goals)
        d2h.append(math.sqrt(s / 3))
    nums = ["Clndrs", "Volume", "Model"]
    nst = {c: stats([r[ti[c]] for r in tr]) for c in nums}
    def feat(r, idx):
        return [nrm(nst[c], r[idx[c]]) if r[idx[c]] != "?" else None for c in nums] + [r[idx["origin"]]]
    trf = [feat(r, ti) for r in tr]
    def dist(x, y):
        s = 0.0
        for i in range(3):
            if x[i] is None or y[i] is None:
                s += 1
            else:
                s += (x[i] - y[i]) ** 2
        s += 0 if x[3] == y[3] else 1
        return math.sqrt(s / 4)
    def pred(x):
        order = sorted(range(len(trf)), key=lambda i: (round(dist(x, trf[i]), 12), i))[:5]
        return sum(d2h[i] for i in order) / len(order)
    n = len(tr)
    cutoff = sorted(d2h)[n // 2]
    rank = sorted(range(n), key=lambda i: (round(d2h[i], 12), i))[:max(1, n // 4)]
    menu, seen = [], set()
    for i in rank:
        p = (tr[i][ti["Clndrs"]], tr[i][ti["Volume"]])
        if p not in seen:
            seen.add(p)
            menu.append(p)
            if len(menu) == 12:
                break
    out = []
    for k, r in enumerate(te):
        x = feat(r, ei)
        before = pred(x)
        res = {"row": k, "before": before, "after": before, "changes": {}, "status": "already_good"}
        if round(before, 10) > round(cutoff, 10):
            res["status"] = "no_candidate"
            oc, ov = r[ei["Clndrs"]], r[ei["Volume"]]
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
                y[0] = nrm(nst["Clndrs"], c)
                y[1] = nrm(nst["Volume"], v)
                after = pred(y)
                if round(before - after, 10) >= 0.05:
                    key = (len(ch), round(dist(x, y), 10), round(after, 10), (c, v))
                    if best is None or key < best[0]:
                        best = (key, ch, after)
            if best:
                res.update(after=best[2], changes=best[1], status="improved")
        out.append(res)
    sys.stdout.write(json.dumps({"repairs": out}))

if __name__ == "__main__":
    main()
