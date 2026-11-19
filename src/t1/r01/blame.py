import csv, sys, json, math, argparse

def load(path):
    with open(path, newline='') as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    a = ap.parse_args()
    h, tr = load(a.train)
    _, te = load(a.test)
    cols = [(i, n) for i, n in enumerate(h) if not n.endswith('X')]
    goals = [(i, n) for i, n in cols if n[-1] in '+-']
    ins = [(i, n) for i, n in cols if n[-1] not in '+-']
    isnum = lambda n: n[0].isupper()

    def val(r, i, n):
        if i >= len(r) or r[i] == '?' or r[i] == '':
            return None
        if isnum(n):
            try:
                return float(r[i])
            except ValueError:
                return None
        return r[i]

    rng = {}
    for i, n in goals + ins:
        if isnum(n):
            v = [x for x in (val(r, i, n) for r in tr) if x is not None]
            lo, hi = (min(v), max(v)) if v else (0, 0)
            rng[n] = (lo, hi)

    def norm(x, n):
        lo, hi = rng[n]
        return 0.0 if hi == lo else (x - lo) / (hi - lo)

    def gaps(r):
        g = {}
        for i, n in goals:
            x = val(r, i, n)
            if x is None:
                continue
            g[n] = abs(norm(x, n) - (1.0 if n[-1] == '+' else 0.0))
        return g

    def dist(r):
        g = gaps(r)
        return math.sqrt(sum(v * v for v in g.values()) / len(g)) if g else 1.0

    order = sorted(range(len(tr)), key=lambda k: (dist(tr[k]), k))
    k = max(1, int(len(tr) * 0.25))
    good = [tr[j] for j in order[:k]]
    poor = [tr[j] for j in order[-k:]]

    prof = {}
    for i, n in ins:
        def stats(grp):
            v = [x for x in (val(r, i, n) for r in grp) if x is not None]
            if isnum(n):
                return sum(norm(x, n) for x in v) / len(v) if v else 0.0
            c = {}
            for x in v:
                c[x] = c.get(x, 0) + 1
            return {s: q / len(v) for s, q in c.items()} if v else {}
        prof[n] = (stats(good), stats(poor))

    out = []
    for r in te:
        g = gaps(r)
        f = {}
        for i, n in ins:
            x = val(r, i, n)
            s = 0.0
            if x is not None:
                pg, pp = prof[n]
                if isnum(n):
                    z = norm(x, n)
                    s = abs(z - pg) - abs(z - pp)
                else:
                    s = pp.get(x, 0.0) - pg.get(x, 0.0)
            f[n] = max(0.0, s)
        out.append({
            'goals': g,
            'features': f,
            'goal_order': sorted(g, key=lambda n: (-g[n], n)),
            'feature_order': sorted(f, key=lambda n: (-f[n], n)),
        })
    print(json.dumps({'diagnoses': out}))

main()
