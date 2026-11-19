import sys, json, csv, math

GOALS = [("Lbs-", 5, 0), ("Acc+", 6, 1), ("Mpg+", 7, 1)]
NUMF = [("Clndrs", 0), ("Volume", 1), ("Model", 3)]
CATF = ("origin", 4)


def load(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[1:]


def num(s):
    if s == "?" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def stats(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    mu = sum(vals) / n if n else 0.0
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return mu, sd


def mknorm(st):
    mu, sd = st

    def f(v):
        z = max(-3.0, min(3.0, (v - mu) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))
    return f


def build(train):
    gn = {g: mknorm(stats([num(r[i]) for r in train])) for g, i, _ in GOALS}
    fn = {g: mknorm(stats([num(r[i]) for r in train])) for g, i in NUMF}
    return gn, fn


def gaps(row, gn):
    return {g: abs(gn[g](num(row[i])) - h) for g, i, h in GOALS}


def d2h(row, gn):
    return math.sqrt(sum(v * v for v in gaps(row, gn).values()) / 3)


def blame_setup(train, gn, fn):
    ds = [d2h(r, gn) for r in train]
    order = sorted(range(len(train)), key=lambda i: (round(ds[i], 12), i))
    q = max(1, len(train) // 4)
    good = [train[i] for i in order[:q]]
    poor = [train[i] for i in order[-q:]]
    cen = {}
    for g, i in NUMF:
        c = []
        for grp in (good, poor):
            v = [x for x in (num(r[i]) for r in grp) if x is not None]
            c.append(fn[g](sum(v) / len(v)) if v else None)
        cen[g] = c
    freq = []
    for grp in (good, poor):
        k = [r[CATF[1]] for r in grp if r[CATF[1]] not in ("?", "")]
        d = {}
        for x in k:
            d[x] = d.get(x, 0) + 1
        freq.append((d, len(k)))
    return cen, freq


def diagnose(row, gn, fn, cen, freq):
    goals = gaps(row, gn)
    feats = {}
    for g, i in NUMF:
        x = num(row[i])
        gc, pc = cen[g]
        if x is None or gc is None or pc is None:
            feats[g] = 0.0
        else:
            nx = fn[g](x)
            feats[g] = max(0.0, abs(nx - gc) - abs(nx - pc))
    c = row[CATF[1]]
    if c in ("?", ""):
        feats[CATF[0]] = 0.0
    else:
        fr = [(d.get(c, 0) / n if n else 0.0) for d, n in freq]
        feats[CATF[0]] = max(0.0, fr[1] - fr[0])
    key = lambda s: (lambda n: (-round(s[n], 10), n))
    return {"goals": goals, "features": feats,
            "goal_order": sorted(goals, key=key(goals)),
            "feature_order": sorted(feats, key=key(feats))}


def main():
    a = sys.argv[1:]
    cmd = a[0]
    train = load(a[a.index("--train") + 1])
    test = load(a[a.index("--test") + 1])
    gn, fn = build(train)
    if cmd == "remember":
        sc = [d2h(r, gn) for r in test]
        o = sorted(range(len(sc)), key=lambda i: (sc[i], i))
        print(json.dumps({"scores": sc, "top": o[:10], "best": o[0],
                          "median": o[len(sc) // 2], "worst": o[-1]}))
        return
    cen, freq = blame_setup(train, gn, fn)
    diag = [diagnose(r, gn, fn, cen, freq) for r in test]
    if cmd == "blame":
        print(json.dumps({"diagnoses": diag}))
        return
    tr = sorted(d2h(r, gn) for r in train)
    cutoff = tr[len(tr) // 2]
    sc = [d2h(r, gn) for r in test]
    n = len(test)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = sc[j], sc[k]
        acc = round(s, 12) <= round(cutoff, 12)
        win = None
        adv = None
        if round(s, 12) != round(o, 12):
            win = j if s < o else k
            lose = k if win == j else j
            gw, gl = diag[win]["goals"], diag[lose]["goals"]
            ad = {g: gl[g] ** 2 - gw[g] ** 2 for g, _, _ in GOALS}
            adv = sorted(ad, key=lambda g: (-round(ad[g], 10), g))[0]
        d = diag[j]
        weak = d["goal_order"][0]
        tf = d["feature_order"][0]
        top = tf if d["features"][tf] > 1e-10 else None
        t = (f"Car {j} {'accepted' if acc else 'rejected'}: score {s:.3f} vs "
             f"car {k} score {o:.3f}, cutoff {cutoff:.3f}. Weakest goal: {weak}. ")
        if win is not None:
            t += f"Recommend car {win} because of {adv}. "
        else:
            t += "Result is a tie. "
        if top:
            t += f"Top feature: {top}. "
        t += "Retrospective, not causal."
        out.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": acc, "winner": win,
                    "advantage_goal": adv, "weakest_goal": weak,
                    "top_feature": top, "text": t})
    print(json.dumps({"explanations": out}))


main()
