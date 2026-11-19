import sys, json, csv, time, argparse, math

def load(path):
    with open(path, newline='') as f:
        r = list(csv.reader(f))
    return [h.strip() for h in r[0]], [[c.strip() for c in x] for x in r[1:] if x]

def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    a = ap.parse_args()
    H, tr = load(a.train)
    H2, te = load(a.test)
    cols = []
    for i, h in enumerate(H):
        base = h.rstrip('+-X')
        if h.endswith('X'):
            continue
        cols.append((i, base, h[0].isupper(), 1 if h.endswith('+') else -1 if h.endswith('-') else 0))
    idx2 = {h: i for i, h in enumerate(H2)}
    feats = [c for c in cols if c[3] == 0]
    goals = [c for c in cols if c[3] != 0]
    num = lambda s: None if s in ('?', '') else float(s)

    def parse(row):
        d = {}
        for i, n, isnum, _ in cols:
            v = row[i] if i < len(row) else '?'
            d[n] = (num(v) if isnum else (None if v == '?' else v))
        return d

    T = [parse(r) for r in tr]
    T = [d for d in T if all(d[g[1]] is not None for g in goals)]
    rng = {}
    for _, n, isnum, _ in cols:
        if isnum:
            vs = [d[n] for d in T if d[n] is not None]
            rng[n] = (min(vs), (max(vs) - min(vs)) or 1.0)

    def nrm(n, v):
        return (v - rng[n][0]) / rng[n][1]

    def qual(d):
        s = 0
        for _, n, _, s_ in goals:
            x = nrm(n, d[n])
            s += x if s_ > 0 else 1 - x
        return s / len(goals)

    TQ = [qual(d) for d in T]

    def fdist(x, y, names):
        s = 0
        for _, n, isnum, _ in feats:
            if n not in names:
                continue
            u, v = x[n], y[n]
            if u is None or v is None:
                e = 1.0
            elif isnum:
                e = nrm(n, u) - nrm(n, v)
            else:
                e = 0.0 if u == v else 1.0
            s += e * e
        return math.sqrt(s)

    allf = {f[1] for f in feats}

    def pred(x):
        ds = sorted((round(fdist(x, t, allf), 12), i) for i, t in enumerate(T))[:5]
        return sum(TQ[i] for _, i in ds) / len(ds)

    srt = sorted(TQ)
    good_thr = srt[len(srt) // 2]
    order = sorted(range(len(T)), key=lambda i: (-TQ[i], i))
    top = order[:max(1, math.ceil(len(T) * 0.25))]
    cfgs, seen = [], set()
    for i in top:
        k = (T[i]['Clndrs'], T[i]['Volume'])
        if k not in seen:
            seen.add(k)
            cfgs.append(k)
        if len(cfgs) == 12:
            break
    editable = ['Clndrs', 'Volume']
    fnames = [f[1] for f in feats]
    repairs = []
    for ri, row in enumerate(te):
        x = parse(row)
        before = {n: x[n] for n in fnames}
        p0 = pred(x)
        rep = {'row': ri, 'before': before, 'after': dict(before), 'changes': [],
               'status': 'already_good', 'predicted_before': p0,
               'predicted_after': p0, 'estimated_improvement': 0.0}
        if p0 < good_thr:
            best = None
            for c in cfgs:
                y = dict(x)
                y['Clndrs'], y['Volume'] = c
                ch = [n for n in editable if x[n] != y[n]]
                if not ch:
                    continue
                p = pred(y)
                imp = p - p0
                if imp < 0.05 - 1e-12:
                    continue
                dist = fdist(x, y, set(editable))
                key = (len(ch), round(dist, 12), -p)
                if best is None or key < best[0]:
                    best = (key, y, ch, p)
            if best:
                _, y, ch, p = best
                rep['after'] = {n: y[n] for n in fnames}
                rep['changes'] = [{'column': n, 'from': x[n], 'to': y[n]} for n in ch]
                rep['status'] = 'improved'
                rep['predicted_after'] = p
                rep['estimated_improvement'] = p - p0
            else:
                rep['status'] = 'no_candidate'
        repairs.append(rep)
    out = {'repairs': repairs,
           'summary': {'cars': len(repairs),
                       'mean_estimated_improvement': sum(r['estimated_improvement'] for r in repairs) / max(1, len(repairs)),
                       'total_changes': sum(len(r['changes']) for r in repairs),
                       'execution_seconds': time.time() - t0,
                       'note': 'Hypothetical recommendations; estimated improvements are not causal effects.'}}
    print(json.dumps(out))

main()
