import sys, csv, json, math, time, argparse, statistics as st

def load(path):
    with open(path, newline='') as f:
        r = list(csv.reader(f))
    return [h.strip() for h in r[0]], [x for x in r[1:] if x]

def schema(header):
    cols = []
    for i, h in enumerate(header):
        c = h[-1]
        suf = c if c in 'XxX+-' and c in '+-X' else ''
        name = h[:-1] if suf else h
        cols.append((i, name, h[0].isupper(), suf))
    return cols

def num(v):
    v = v.strip()
    if v == '?' or v == '':
        return None
    try:
        return float(v)
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    a = ap.parse_args()
    t0 = time.time()
    hd, tr = load(a.train)
    _, te = load(a.test)
    cols = schema(hd)
    goals = [(i, n, s) for i, n, isn, s in cols if s in '+-' and s]
    feats = [(i, n, isn) for i, n, isn, s in cols if not s]
    stats = {}
    for i, n, s in goals:
        v = [x for x in (num(r[i]) for r in tr) if x is not None]
        stats[i] = (min(v), max(v), st.median(v))

    def gaps(r):
        g = {}
        for i, n, s in goals:
            lo, hi, med = stats[i]
            v = num(r[i])
            if v is None:
                v = med
            rg = (hi - lo) or 1.0
            x = (hi - v) / rg if s == '+' else (v - lo) / rg
            g[n] = min(1.0, max(0.0, x))
        return g

    def score(r):
        g = gaps(r)
        return math.sqrt(sum(x * x for x in g.values()) / len(g))

    trs = [score(r) for r in tr]
    cutoff = st.median(trs)
    overall = sum(trs) / len(trs)

    # feature keying
    keyf = {}
    for i, n, isn in feats:
        if isn:
            vals = sorted(x for x in (num(r[i]) for r in tr) if x is not None)
            if len(set(vals)) <= 12:
                keyf[i] = lambda v: v
            else:
                c1, c2 = vals[len(vals) // 3], vals[2 * len(vals) // 3]
                keyf[i] = lambda v, c1=c1, c2=c2: 0 if v < c1 else (1 if v < c2 else 2)
        else:
            keyf[i] = lambda v: v

    def fk(i, isn, r):
        if isn:
            v = num(r[i])
            return None if v is None else keyf[i](v)
        v = r[i].strip()
        return None if v == '?' or v == '' else v

    groups = {}
    for i, n, isn in feats:
        d = {}
        for r, s in zip(tr, trs):
            k = fk(i, isn, r)
            if k is not None:
                d.setdefault(k, []).append(s)
        groups[i] = d

    def top_feature(r, accepted):
        best, bn = 0.0, None
        for i, n, isn in feats:
            k = fk(i, isn, r)
            if k is None:
                continue
            g = groups[i].get(k, [])
            if len(g) < 5:
                continue
            m = sum(g) / len(g)
            ev = (overall - m) if accepted else (m - overall)
            if ev > best + 1e-12:
                best, bn = ev, n
        return bn

    n = len(te)
    sc = [score(r) for r in te]
    gp = [gaps(r) for r in te]
    out = []
    ok = True
    for j in range(n):
        o = (j + 1) % n
        s, os_ = sc[j], sc[o]
        acc = s <= cutoff
        g, og = gp[j], gp[o]
        weak = max(g, key=lambda k: g[k])
        tie = round(s, 12) == round(os_, 12)
        if tie:
            win, adv = None, None
        else:
            win = j if s < os_ else o
            wg, lg = (g, og) if win == j else (og, g)
            adv = max(wg, key=lambda k: lg[k] ** 2 - wg[k] ** 2)
        tf = top_feature(te[j], acc)
        if tie:
            t = f"Cars {j} and {o} tie on distance to heaven ({s:.6f})."
        else:
            lose = o if win == j else j
            t = (f"Car {win} is recommended over car {lose} (distance {min(s, os_):.6f} vs {max(s, os_):.6f}, lower is better); "
                 f"the largest advantage comes from {adv}.")
        t += (f" Car {j} is {'accepted' if acc else 'rejected'}: score {s:.6f} is "
              f"{'at or below' if acc else 'above'} the median training cutoff {cutoff:.6f}. "
              f"Its weakest goal is {weak} (gap {g[weak]:.3f}).")
        if tf:
            t += f" Feature {tf} is associated with this outcome in training (retrospective, not causal)."
        # verification
        if not tie:
            ok &= (win == (j if s < os_ else o)) and (adv in g)
        ok &= acc == (s <= cutoff)
        out.append(dict(row=j, other=o, score=s, other_score=os_, cutoff=cutoff, accepted=acc,
                        winner=win, advantage_goal=adv, weakest_goal=weak, top_feature=tf, text=t))
    print(json.dumps(dict(explanations=out, count=n, verified=ok,
                          execution_time_seconds=time.time() - t0)))

main()
