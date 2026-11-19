import sys, csv, json, time, math, argparse

def load(path):
    with open(path, newline='') as f:
        r = list(csv.reader(f))
    h = [x.strip() for x in r[0]]
    rows = [[c.strip() for c in x] for x in r[1:] if x and any(c.strip() for c in x)]
    return h, rows

def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    a = ap.parse_args()
    h, tr = load(a.train)
    h2, te = load(a.test)
    idx2 = {n: i for i, n in enumerate(h2)}
    cols = []
    for i, n in enumerate(h):
        if n.endswith('X'):
            continue
        kind = 'num' if n[0].isupper() else 'sym'
        goal = 0
        if n.endswith('+'):
            goal = 1
        elif n.endswith('-'):
            goal = -1
        cols.append((i, n, kind, goal))
    inputs = [c for c in cols if c[3] == 0]
    goals = [c for c in cols if c[3] != 0]

    def val(s, kind):
        if s is None or s == '?' or s == '':
            return None
        if kind == 'num':
            try:
                return float(s)
            except ValueError:
                return None
        return s

    def getrow(row, hd_idx, c):
        j = hd_idx.get(c[1])
        if j is None or j >= len(row):
            return None
        return val(row[j], c[2])

    tidx = {n: i for i, n in enumerate(h)}
    # training
    lo, hi = {}, {}
    for c in cols:
        if c[2] == 'num':
            v = [x for x in (val(r[c[0]], 'num') for r in tr) if x is not None]
            if v:
                lo[c[1]], hi[c[1]] = min(v), max(v)
    rng = {k: (hi[k] - lo[k]) or 1.0 for k in lo}

    def quality(r):
        s = []
        for c in goals:
            v = val(r[c[0]], 'num')
            if v is None or c[1] not in lo:
                continue
            n = (v - lo[c[1]]) / rng[c[1]]
            s.append(n if c[3] > 0 else 1 - n)
        return sum(s) / len(s) if s else None

    X, Q = [], []
    for r in tr:
        q = quality(r)
        if q is None:
            continue
        X.append([getrow(r, tidx, c) for c in inputs])
        Q.append(q)

    def dist(u, v):
        s = 0.0
        for c, p, q in zip(inputs, u, v):
            if c[2] == 'num':
                if p is None or q is None:
                    d = 1.0
                else:
                    d = abs(p - q) / rng.get(c[1], 1.0)
            else:
                d = 0.0 if (p is not None and p == q) else 1.0
            s += d * d
        return math.sqrt(s)

    def predict(u):
        ds = sorted((round(dist(u, x), 12), i) for i, x in enumerate(X))[:5]
        return sum(Q[i] for _, i in ds) / len(ds) if ds else 0.0

    sq = sorted(Q)
    median = sq[len(sq) // 2] if sq else 0.0
    order = sorted(range(len(Q)), key=lambda i: -Q[i])
    top = order[:max(1, math.ceil(len(Q) * 0.25))] if Q else []
    pos = {c[1]: k for k, c in enumerate(inputs)}
    ic, iv = pos.get('Clndrs'), pos.get('Volume')
    cands, seen = [], set()
    for i in top:
        cfg = (X[i][ic], X[i][iv])
        if cfg in seen:
            continue
        seen.add(cfg)
        cands.append(cfg)
        if len(cands) == 12:
            break

    def fmt(v):
        if isinstance(v, float) and v == int(v):
            return int(v)
        return v

    def d_out(u):
        return {c[1]: fmt(u[k]) for k, c in enumerate(inputs)}

    repairs = []
    for ri, r in enumerate(te):
        u = [getrow(r, idx2, c) for c in inputs]
        p0 = predict(u)
        rep = {'row': ri, 'before': d_out(u), 'after': d_out(u),
               'changes': [], 'status': 'already_good'}
        if p0 < median:
            best = None
            for cfg in cands:
                v = list(u)
                v[ic], v[iv] = cfg
                ch = [k for k in (ic, iv) if u[k] != v[k]]
                if not ch:
                    continue
                p1 = predict(v)
                if p1 - p0 < 0.05 - 1e-12:
                    continue
                dd = 0.0
                for k in ch:
                    nm = inputs[k][1]
                    if u[k] is None:
                        dd += 1.0
                    else:
                        dd += ((u[k] - v[k]) / rng.get(nm, 1.0)) ** 2
                key = (len(ch), math.sqrt(dd), -p1)
                if best is None or key < best[0]:
                    best = (key, v, ch, p1)
            if best is None:
                rep['status'] = 'no_candidate'
            else:
                _, v, ch, p1 = best
                rep['status'] = 'improved'
                rep['after'] = d_out(v)
                rep['changes'] = [{'column': inputs[k][1], 'from': fmt(u[k]), 'to': fmt(v[k])} for k in ch]
                rep['estimated_improvement'] = round(p1 - p0, 6)
        repairs.append(rep)
    out = {'repairs': repairs,
           'note': 'Hypothetical recommendations; estimated improvements are not observed causal effects.',
           'num_changes_total': sum(len(x['changes']) for x in repairs),
           'execution_time_seconds': round(time.time() - t0, 6)}
    print(json.dumps(out))

main()
