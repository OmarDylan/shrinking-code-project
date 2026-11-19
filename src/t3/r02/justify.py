import csv, sys, json, math

GOALS = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUM = ["Clndrs", "Volume", "Model"]
CAT = "origin"


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fl(v):
    try:
        v = v.strip()
        return None if v == "?" or v == "" else float(v)
    except Exception:
        return None


def stats(vals):
    v = [x for x in vals if x is not None]
    n = len(v)
    if n == 0:
        return 0.0, 0.0
    m = sum(v) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, sd


def norm(v, ms):
    m, sd = ms
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))


def main():
    a = sys.argv
    tr = read(a[a.index("--train") + 1])
    te = read(a[a.index("--test") + 1])
    gs = {g: stats([fl(r[g]) for r in tr]) for g in GOALS}
    ns = {c: stats([fl(r[c]) for r in tr]) for c in NUM}

    def d2h(r):
        return math.sqrt(sum((norm(fl(r[g]), gs[g]) - h) ** 2 for g, h in GOALS.items()) / 3)

    def gaps(r):
        return {g: abs(norm(fl(r[g]), gs[g]) - h) for g, h in GOALS.items()}

    td = [d2h(r) for r in tr]
    cutoff = sorted(td)[len(tr) // 2]
    order = sorted(range(len(tr)), key=lambda i: (round(td[i], 12), i))
    q = max(1, len(tr) // 4)
    good = [tr[i] for i in order[:q]]
    poor = [tr[i] for i in order[-q:]]

    centers = {}
    for c in NUM:
        cs = []
        for grp in (good, poor):
            v = [x for x in (fl(r[c]) for r in grp) if x is not None]
            cs.append(norm(sum(v) / len(v), ns[c]) if v else None)
        centers[c] = cs

    def freq(grp):
        v = [r[CAT].strip() for r in grp if r[CAT].strip() != "?"]
        d = {}
        for x in v:
            d[x] = d.get(x, 0) + 1
        return d, len(v)

    gf, gn = freq(good)
    pf, pn = freq(poor)

    def blame(r):
        feats = {}
        for c in NUM:
            x = fl(r[c])
            gc, pc = centers[c]
            if x is None or gc is None or pc is None:
                feats[c] = 0.0
            else:
                nx = norm(x, ns[c])
                feats[c] = max(0.0, abs(nx - gc) - abs(nx - pc))
        cv = r[CAT].strip()
        if cv == "?":
            feats[CAT] = 0.0
        else:
            p = pf.get(cv, 0) / pn if pn else 0.0
            g = gf.get(cv, 0) / gn if gn else 0.0
            feats[CAT] = max(0.0, p - g)
        go = sorted(gaps(r).items(), key=lambda t: (-round(t[1], 10), t[0]))
        fo = sorted(feats.items(), key=lambda t: (-round(t[1], 10), t[0]))
        return [x[0] for x in go], fo

    sc = [d2h(r) for r in te]
    gp = [gaps(r) for r in te]
    n = len(te)
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
            ad = {g: gp[lose][g] ** 2 - gp[win][g] ** 2 for g in GOALS}
            adv = sorted(ad, key=lambda g: (-round(ad[g], 10), g))[0]
        go, fo = blame(te[j])
        weak = go[0]
        top = fo[0][0] if fo[0][1] > 1e-10 else None
        t = "Car %d %s: score %.3f, other score %.3f, cutoff %.3f; weakest goal %s." % (
            j, "accepted" if acc else "rejected", s, o, cutoff, weak)
        if win is not None:
            t += " Recommend car %d; largest advantage goal %s." % (win, adv)
        else:
            t += " Scores tie."
        if top is not None:
            t += " Top feature %s." % top
        out.append({"row": j, "other": k, "score": s, "other_score": o, "cutoff": cutoff,
                    "accepted": acc, "winner": win, "advantage_goal": adv,
                    "weakest_goal": weak, "top_feature": top, "text": t})
    sys.stdout.write(json.dumps({"explanations": out}))


if len(sys.argv) > 1 and sys.argv[1] == "justify":
    main()
