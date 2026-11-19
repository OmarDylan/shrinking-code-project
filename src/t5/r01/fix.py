import sys, json, time, argparse
from math import exp, sqrt

def atom(s):
    s = s.strip()
    for f in (int, float):
        try: return f(s)
        except ValueError: pass
    return s

def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = [[atom(c) for c in l.split(",")] for l in f if l.strip()]
    return [str(h) for h in rows[0]], rows[1:]

def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    names, train = load(a.train)
    _, test = load(a.test)
    cols = [i for i, s in enumerate(names) if s[-1] != "X"]
    ys = [i for i in cols if names[i][-1] in "+-"]
    xs = [i for i in cols if i not in ys]
    isnum = lambda i: names[i][0].isupper()
    st = {}
    for i in cols:
        if isnum(i):
            v = [r[i] for r in train if r[i] != "?"]
            mu = sum(v) / max(1, len(v))
            sd = sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0
            st[i] = (mu, sd)

    def norm(i, v):
        mu, sd = st[i]
        z = max(-3, min(3, (v - mu) / (1e-32 + sd)))
        return 1 / (1 + exp(-1.7 * z))

    def gap(i, p, q):
        if p == "?" or q == "?": return 1
        if not isnum(i): return 1 if p != q else 0
        return abs(norm(i, p) - norm(i, q))

    def xd(r, s, ats=None):
        ats = xs if ats is None else ats
        return sqrt(sum(gap(i, r[i], s[i]) ** 2 for i in ats) / len(xs))

    def yd(r):
        return sqrt(sum((norm(i, r[i]) - (0 if names[i][-1] == "-" else 1)) ** 2
                        for i in ys) / len(ys))

    ty = [yd(r) for r in train]

    def pred(r):
        d = sorted((round(xd(r, s), 12), k) for k, s in enumerate(train))[:5]
        return sum(ty[k] for _, k in d) / len(d)

    order = sorted(range(len(train)), key=lambda k: (ty[k], k))
    nbest = max(1, len(train) // 4)
    best = order[:nbest]
    cutoff = ty[best[-1]]
    cfgs, seen = [], set()
    for k in best:
        c = (train[k][0], train[k][1])
        if "?" in c or c in seen: continue
        seen.add(c); cfgs.append(c)
        if len(cfgs) == 12: break
    chg = [0, 1]
    nm = lambda r: {names[i]: r[i] for i in xs}
    out = []
    for n, r in enumerate(test):
        p = pred(r)
        rep = {"row": n, "before": nm(r), "after": nm(r), "changes": [],
               "status": "already_good"}
        if p > cutoff:
            rep["status"] = "no_candidate"
            cand = []
            for c in cfgs:
                s = list(r); s[0], s[1] = c
                ch = [i for i in chg if r[i] != s[i]]
                if not ch: continue
                q = pred(s)
                imp = p - q
                if imp < 0.05: continue
                cand.append(((len(ch), xd(r, s, ch), q), s, ch, imp))
            if cand:
                _, s, ch, imp = min(cand, key=lambda z: z[0])
                rep["after"] = nm(s)
                rep["changes"] = [{"column": names[i], "from": r[i], "to": s[i]} for i in ch]
                rep["status"] = "improved"
                rep["estimated_improvement"] = imp
        rep["predicted_before"] = p
        out.append(rep)
    print(json.dumps({"repairs": out, "seconds": time.time() - t0}))

main()
