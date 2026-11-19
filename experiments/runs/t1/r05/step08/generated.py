import csv, sys, math, argparse, json

def load(p):
    with open(p, newline='') as f:
        return list(csv.DictReader(f))

def fnum(s):
    return float(s)

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
    goals = ['Lbs-', 'Acc+', 'Mpg+']
    heaven = [0, 1, 1]
    gs = [stats([fnum(r[g]) for r in tr]) for g in goals]
    ins = ['Clndrs', 'Volume', 'Model']
    ist = {c: stats([fnum(r[c]) for r in tr]) for c in ins}

    def d2h(r):
        s = 0.0
        for g, h, (m, sd) in zip(goals, heaven, gs):
            s += (norm(fnum(r[g]), m, sd) - h) ** 2
        return math.sqrt(s / 3)

    def feat(r):
        return ([norm(fnum(r[c]), *ist[c]) for c in ins], r['origin'])

    trf = [feat(r) for r in tr]
    td = [d2h(r) for r in tr]

    def dist(A, B):
        s = sum((x - y) ** 2 for x, y in zip(A[0], B[0]))
        s += 0 if A[1] == B[1] else 1
        return math.sqrt(s / 4)

    def predict(f):
        order = sorted(range(len(tr)), key=lambda i: (round(dist(f, trf[i]), 12), i))[:5]
        return sum(td[i] for i in order) / len(order)

    cutoff = sorted(td)[len(tr) // 2]
    rank = sorted(range(len(tr)), key=lambda i: (round(td[i], 12), i))
    good = rank[:max(1, len(tr) // 4)]
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
        f = feat(r)
        before = predict(f)
        if round(before, 10) <= round(cutoff, 10):
            out.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "already_good"})
            continue
        oc, ov = fnum(r['Clndrs']), fnum(r['Volume'])
        best = None
        for nc, nv in menu:
            ch = {}
            if nc != oc:
                ch['Clndrs'] = nc
            if nv != ov:
                ch['Volume'] = nv
            if not ch:
                continue
            nf = ([norm(nc, *ist['Clndrs']), norm(nv, *ist['Volume']), f[0][2]], f[1])
            after = predict(nf)
            if round(before - after, 10) < 0.05:
                continue
            key = (len(ch), round(dist(f, nf), 10), round(after, 10), (nc, nv))
            if best is None or key < best[0]:
                best = (key, after, ch)
        if best is None:
            out.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "no_candidate"})
        else:
            out.append({"row": idx, "before": before, "after": best[1], "changes": best[2], "status": "improved"})
    sys.stdout.write(json.dumps({"repairs": out}))

main()
