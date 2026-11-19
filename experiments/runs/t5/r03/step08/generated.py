import sys, csv, json, math

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]

    def load(p):
        with open(p, encoding='utf-8-sig') as f:
            rows = [r for r in csv.reader(f) if r]
        return [h.strip() for h in rows[0]], [[c.strip() for c in r] for r in rows[1:]]

    def num(s):
        return None if s == '?' else float(s)

    hn, tr_rows = load(tr)
    hm, te_rows = load(te)
    ix = {n: i for i, n in enumerate(hn)}
    jx = {n: i for i, n in enumerate(hm)}

    def stats(vals):
        v = [x for x in vals if x is not None]
        n = len(v)
        mu = sum(v) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        return mu, sd

    def nm(st, v):
        z = max(-3, min(3, (v - st[0]) / (st[1] + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    goals = [('Lbs-', 0), ('Acc+', 1), ('Mpg+', 1)]
    gst = {g: stats([num(r[ix[g]]) for r in tr_rows]) for g, _ in goals}
    d2h = []
    for r in tr_rows:
        s = 0
        for g, h in goals:
            s += (nm(gst[g], num(r[ix[g]])) - h) ** 2
        d2h.append(math.sqrt(s / 3))

    nf = ['Clndrs', 'Volume', 'Model']
    fst = {f: stats([num(r[ix[f]]) for r in tr_rows]) for f in nf}

    def feat(r, idx):
        out = []
        for f in nf:
            v = num(r[idx[f]])
            out.append(None if v is None else nm(fst[f], v))
        out.append(r[idx['origin']])
        return out

    def fd(A, B):
        s = 0
        for x, y in zip(A[:3], B[:3]):
            s += 1 if (x is None or y is None) else (x - y) ** 2
        s += 0 if A[3] == B[3] else 1
        return math.sqrt(s / 4)

    TF = [feat(r, ix) for r in tr_rows]
    n = len(tr_rows)

    def predict(F):
        order = sorted(range(n), key=lambda i: (round(fd(F, TF[i]), 12), i))[:5]
        return sum(d2h[i] for i in order) / len(order)

    cutoff = sorted(d2h)[n // 2]
    rank = sorted(range(n), key=lambda i: (round(d2h[i], 12), i))
    good = rank[:max(1, n // 4)]
    menu = []
    seen = set()
    for i in good:
        p = (num(tr_rows[i][ix['Clndrs']]), num(tr_rows[i][ix['Volume']]))
        if p not in seen:
            seen.add(p)
            menu.append(p)
            if len(menu) == 12:
                break

    def fmt(v):
        return int(v) if float(v).is_integer() else v

    reps = []
    for k, r in enumerate(te_rows):
        F = feat(r, jx)
        c0, v0 = num(r[jx['Clndrs']]), num(r[jx['Volume']])
        before = predict(F)
        if round(before, 10) <= round(cutoff, 10):
            reps.append({"row": k, "before": before, "after": before, "changes": {}, "status": "already_good"})
            continue
        best = None
        for (nc, nv) in menu:
            chc = nc != c0
            chv = nv != v0
            if not (chc or chv):
                continue
            G = list(F)
            if chc:
                G[0] = nm(fst['Clndrs'], nc)
            if chv:
                G[1] = nm(fst['Volume'], nv)
            after = predict(G)
            if round(before - after, 10) >= 0.05:
                key = (int(chc) + int(chv), round(fd(F, G), 10), round(after, 10), (nc, nv))
                if best is None or key < best[0]:
                    best = (key, after, chc, chv, nc, nv)
        if best is None:
            reps.append({"row": k, "before": before, "after": before, "changes": {}, "status": "no_candidate"})
        else:
            _, after, chc, chv, nc, nv = best
            ch = {}
            if chc:
                ch['Clndrs'] = fmt(nc)
            if chv:
                ch['Volume'] = fmt(nv)
            reps.append({"row": k, "before": before, "after": after, "changes": ch, "status": "improved"})
    sys.stdout.write(json.dumps({"repairs": reps}))

if __name__ == '__main__':
    main()
