import sys, json, math

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        lines = [l.strip() for l in f if l.strip()]
    hdr = [h.strip() for h in lines[0].split(",")]
    rows = []
    for l in lines[1:]:
        c = [x.strip() for x in l.split(",")]
        r = []
        for i, v in enumerate(c):
            if v == "?":
                r.append(None)
            elif hdr[i][0].isupper():
                r.append(float(v))
            else:
                r.append(v)
        rows.append(r)
    return hdr, rows

def stats(vals):
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

def main():
    a = sys.argv
    tr = a[a.index("--train") + 1]
    te = a[a.index("--test") + 1]
    hdr, train = load(tr)
    _, test = load(te)
    GI = {5: 0.0, 6: 1.0, 7: 1.0}
    gn = {i: hdr[i] for i in GI}
    gnorm = {i: mknorm(stats([r[i] for r in train if r[i] is not None])) for i in GI}

    def gaps(r):
        return {i: abs(gnorm[i](r[i]) - GI[i]) for i in GI}

    def d2h(r):
        return math.sqrt(sum(g * g for g in gaps(r).values()) / 3)

    tscores = [d2h(r) for r in train]
    order = sorted(range(len(train)), key=lambda i: (round(tscores[i], 12), i))
    q = max(1, len(train) // 4)
    good = [train[i] for i in order[:q]]
    poor = [train[i] for i in order[-q:]]
    cutoff = sorted(tscores)[len(train) // 2]

    numf = [0, 1, 3]
    nn = {i: mknorm(stats([r[i] for r in train if r[i] is not None])) for i in numf}
    centers = {}
    for i in numf:
        c = []
        for g in (good, poor):
            vs = [r[i] for r in g if r[i] is not None]
            c.append(nn[i](sum(vs) / len(vs)) if vs else None)
        centers[i] = c

    def freq(g, v):
        vs = [r[4] for r in g if r[4] is not None]
        return vs.count(v) / len(vs) if vs else 0.0

    def diagnose(r):
        ev = {}
        for i in numf:
            gc, pc = centers[i]
            if r[i] is None or gc is None or pc is None:
                ev[hdr[i]] = 0.0
            else:
                x = nn[i](r[i])
                ev[hdr[i]] = max(0.0, abs(x - gc) - abs(x - pc))
        if r[4] is None:
            ev[hdr[4]] = 0.0
        else:
            ev[hdr[4]] = max(0.0, freq(poor, r[4]) - freq(good, r[4]))
        gg = {gn[i]: v for i, v in gaps(r).items()}
        key = lambda d: sorted(d, key=lambda n: (-round(d[n], 10), n))
        return ev, gg, key(gg), key(ev)

    out = []
    n = len(test)
    for j in range(n):
        k = (j + 1) % n
        s, o = d2h(test[j]), d2h(test[k])
        ev, gg, go, fo = diagnose(test[j])
        acc = round(s, 12) <= round(cutoff, 12)
        win = None
        adv = None
        if round(s, 12) != round(o, 12):
            win = j if s < o else k
            lose = k if win == j else j
            gw, gl = gaps(test[win]), gaps(test[lose])
            ad = {gn[i]: gl[i] ** 2 - gw[i] ** 2 for i in GI}
            adv = sorted(ad, key=lambda m: (-round(ad[m], 10), m))[0]
        weak = go[0]
        top = fo[0] if ev[fo[0]] > 1e-10 else None
        t = (f"Car {j} {'accepted' if acc else 'rejected'}: score {s:.3f}, "
             f"other score {o:.3f}, cutoff {cutoff:.3f}. Weakest goal: {weak}. ")
        if win is not None:
            t += f"Recommend car {win} over car {k if win == j else j}; biggest advantage in {adv}. "
        else:
            t += "Scores tie. "
        if top:
            t += f"Top feature: {top}. "
        t += "Retrospective, not causal."
        out.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": acc, "winner": win,
                    "advantage_goal": adv, "weakest_goal": weak,
                    "top_feature": top, "text": t})
    sys.stdout.write(json.dumps({"explanations": out}))

if __name__ == "__main__":
    main()
