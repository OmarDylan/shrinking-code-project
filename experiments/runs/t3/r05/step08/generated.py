import sys, csv, json, math, argparse

def load(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def fnum(v):
    return float(v)

def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd

def norm(v, m, sd):
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--train')
    ap.add_argument('--test')
    a = ap.parse_args()
    tr = load(a.train)
    te = load(a.test)
    goals = [('Lbs-', 0.0), ('Acc+', 1.0), ('Mpg+', 1.0)]
    gs = {g: stats([fnum(r[g]) for r in tr]) for g, _ in goals}
    d2h = []
    for r in tr:
        s = sum((norm(fnum(r[g]), *gs[g]) - h) ** 2 for g, h in goals)
        d2h.append(math.sqrt(s / 3))
    nums = ['Clndrs', 'Volume', 'Model']
    ns = {c: stats([fnum(r[c]) for r in tr]) for c in nums}

    def vec(r, over=None):
        v = []
        for c in nums:
            x = fnum(r[c])
            if over and c in over:
                x = over[c]
            v.append(norm(x, *ns[c]))
        return v, r['origin']

    def dist(A, B):
        s = sum((x - y) ** 2 for x, y in zip(A[0], B[0]))
        s += 0.0 if A[1] == B[1] else 1.0
        return math.sqrt(s / 4)

    trv = [vec(r) for r in tr]

    def pred(q):
        ds = sorted((round(dist(q, t), 12), i) for i, t in enumerate(trv))
        top = ds[:5]
        return sum(d2h[i] for _, i in top) / len(top)

    n = len(tr)
    cutoff = sorted(d2h)[n // 2]
    order = sorted(range(n), key=lambda i: (round(d2h[i], 12), i))
    good = order[:max(1, n // 4)]
    menu = []
    seen = set()
    for i in good:
        p = (fnum(tr[i]['Clndrs']), fnum(tr[i]['Volume']))
        if p not in seen:
            seen.add(p)
            menu.append(p)
            if len(menu) == 12:
                break

    out = []
    for idx, r in enumerate(te):
        q = vec(r)
        before = pred(q)
        if round(before, 10) <= round(cutoff, 10):
            out.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "already_good"})
            continue
        c0, v0 = fnum(r['Clndrs']), fnum(r['Volume'])
        best = None
        for (nc, nv) in menu:
            ch = {}
            if nc != c0:
                ch['Clndrs'] = nc
            if nv != v0:
                ch['Volume'] = nv
            if not ch:
                continue
            e = vec(r, ch)
            after = pred(e)
            if round(before - after, 10) >= 0.05:
                key = (len(ch), round(dist(q, e), 10), round(after, 10), (nc, nv))
                if best is None or key < best[0]:
                    best = (key, after, ch)
        if best is None:
            out.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "no_candidate"})
        else:
            out.append({"row": idx, "before": before, "after": best[1], "changes": best[2], "status": "improved"})
    sys.stdout.write(json.dumps({"repairs": out}))

main()
