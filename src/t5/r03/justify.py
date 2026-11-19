import sys, json, math

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        L = [l.strip().split(",") for l in f if l.strip()]
    h = [x.strip() for x in L[0]]
    rows = []
    for r in L[1:]:
        r = [x.strip() for x in r]
        rows.append(r)
    return h, rows

def num(s):
    if s == "?":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def stats(vs):
    vs = [v for v in vs if v is not None]
    n = len(vs)
    if n == 0:
        return 0.0, 0.0
    mu = sum(vs) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vs) / (n - 1)) if n > 1 else 0.0
    return mu, sd

def mknorm(mu, sd):
    def f(v):
        z = max(-3, min(3, (v - mu) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))
    return f

def main():
    a = sys.argv[1:]
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    h, R = load(tr)
    _, T = load(te)
    G = {"Lbs-": (5, 0), "Acc+": (6, 1), "Mpg+": (7, 1)}
    NF = {"Clndrs": 0, "Volume": 1, "Model": 3}
    CAT = ("origin", 4)
    gn = {g: mknorm(*stats([num(r[i]) for r in R])) for g, (i, _) in G.items()}
    fn = {k: mknorm(*stats([num(r[i]) for r in R])) for k, i in NF.items()}

    def gaps(r):
        return {g: abs(gn[g](num(r[i])) - hv) for g, (i, hv) in G.items()}

    def d2h(r):
        return math.sqrt(sum(v * v for v in gaps(r).values()) / 3)

    ds = [d2h(r) for r in R]
    order = sorted(range(len(R)), key=lambda i: (round(ds[i], 12), i))
    q = max(1, len(R) // 4)
    good = [R[i] for i in order[:q]]
    poor = [R[i] for i in order[-q:]]
    cutoff = sorted(ds)[len(ds) // 2]

    centers = {}
    for k, i in NF.items():
        c = []
        for grp in (good, poor):
            vs = [num(r[i]) for r in grp if num(r[i]) is not None]
            m = sum(vs) / len(vs) if vs else stats([num(r[i]) for r in R])[0]
            c.append(fn[k](m))
        centers[k] = c

    def freq(grp, v):
        kn = [r[CAT[1]] for r in grp if r[CAT[1]] != "?"]
        return kn.count(v) / len(kn) if kn else 0.0

    def feats(r):
        out = {}
        for k, i in NF.items():
            x = num(r[i])
            if x is None:
                out[k] = 0.0
            else:
                nx = fn[k](x)
                out[k] = max(0.0, abs(nx - centers[k][0]) - abs(nx - centers[k][1]))
        v = r[CAT[1]]
        out[CAT[0]] = 0.0 if v == "?" else max(0.0, freq(poor, v) - freq(good, v))
        return out

    n = len(T)
    sc = [d2h(r) for r in T]
    gp = [gaps(r) for r in T]
    res = []
    for j in range(n):
        k = (j + 1) % n
        s, o = sc[j], sc[k]
        fe = feats(T[j])
        goal_order = sorted(gp[j], key=lambda g: (-round(gp[j][g], 10), g))
        feat_order = sorted(fe, key=lambda f: (-round(fe[f], 10), f))
        weakest = goal_order[0]
        top = feat_order[0] if fe[feat_order[0]] > 1e-10 else None
        acc = round(s, 12) <= round(cutoff, 12)
        win = None
        adv = None
        if round(s, 12) != round(o, 12):
            win = j if s < o else k
            los = k if win == j else j
            ad = {g: gp[los][g] ** 2 - gp[win][g] ** 2 for g in G}
            adv = sorted(ad, key=lambda g: (-round(ad[g], 10), g))[0]
        t = "Car %d %s: score %.3f, other score %.3f, cutoff %.3f; weakest goal %s." % (
            j, "accepted" if acc else "rejected", s, o, cutoff, weakest)
        if win is not None:
            t += " Recommend car %d, largest advantage in %s." % (win, adv)
        else:
            t += " Scores tie."
        if top is not None:
            t += " Top feature: %s." % top
        res.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": acc, "winner": win,
                    "advantage_goal": adv, "weakest_goal": weakest,
                    "top_feature": top, "text": t})
    sys.stdout.write(json.dumps({"explanations": res}))

if __name__ == "__main__":
    main()
