import csv, sys, math, json, argparse

GOALS = ["Lbs-", "Acc+", "Mpg+"]
HEAVEN = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUMF = ["Clndrs", "Volume", "Model"]
CATF = ["origin"]


def read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    if v is None:
        return None
    v = v.strip()
    if v == "?" or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def stats(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd


def mk_norm(m, sd):
    def f(v):
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1.0 / (1.0 + math.exp(-1.702 * z))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()
    tr, te = read(a.train), read(a.test)

    gn = {}
    for g in GOALS:
        m, sd = stats([x for x in (num(r[g]) for r in tr) if x is not None])
        gn[g] = mk_norm(m, sd)
    fn = {}
    for f in NUMF:
        m, sd = stats([x for x in (num(r[f]) for r in tr) if x is not None])
        fn[f] = mk_norm(m, sd)

    def gaps(r):
        return {g: abs(gn[g](num(r[g])) - HEAVEN[g]) for g in GOALS}

    def d2h(r):
        return math.sqrt(sum(v * v for v in gaps(r).values()) / 3)

    trs = [d2h(r) for r in tr]
    order = sorted(range(len(tr)), key=lambda i: (round(trs[i], 12), i))
    q = max(1, len(tr) // 4)
    good = [tr[i] for i in order[:q]]
    poor = [tr[i] for i in order[-q:]]
    cutoff = sorted(trs)[len(tr) // 2]

    centers = {}
    for f in NUMF:
        c = []
        for grp in (good, poor):
            vs = [x for x in (num(r[f]) for r in grp) if x is not None]
            c.append(fn[f](sum(vs) / len(vs)) if vs else None)
        centers[f] = c

    def freq(grp, f, cat):
        vs = [r[f].strip() for r in grp if r[f].strip() not in ("?", "")]
        return vs.count(cat) / len(vs) if vs else 0.0

    def blame(r):
        feats = {}
        for f in NUMF:
            x = num(r[f])
            gc, pc = centers[f]
            if x is None or gc is None or pc is None:
                feats[f] = 0.0
            else:
                nx = fn[f](x)
                feats[f] = max(0.0, abs(nx - gc) - abs(nx - pc))
        for f in CATF:
            c = r[f].strip()
            if c in ("?", ""):
                feats[f] = 0.0
            else:
                feats[f] = max(0.0, freq(poor, f, c) - freq(good, f, c))
        gp = gaps(r)
        go = sorted(GOALS, key=lambda n: (-round(gp[n], 10), n))
        fo = sorted(feats, key=lambda n: (-round(feats[n], 10), n))
        return gp, feats, go, fo

    n = len(te)
    sc = [d2h(r) for r in te]
    gps = [gaps(r) for r in te]
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
            advs = {g: gps[lose][g] ** 2 - gps[win][g] ** 2 for g in GOALS}
            adv = sorted(GOALS, key=lambda g: (-round(advs[g], 10), g))[0]
        _, feats, go, fo = blame(te[j])
        weakest = go[0]
        top = fo[0] if feats[fo[0]] > 1e-10 else None
        t = "Car %d %s: score %.3f, other score %.3f (car %d), cutoff %.3f. Weakest goal: %s." % (
            j, "accepted" if acc else "rejected", s, o, k, cutoff, weakest)
        if win is not None:
            t += " Recommend car %d; largest advantage in %s." % (win, adv)
        else:
            t += " The comparison is a tie."
        if top is not None:
            t += " Top feature: %s." % top
        out.append({"row": j, "other": k, "score": s, "other_score": o, "cutoff": cutoff,
                    "accepted": acc, "winner": win, "advantage_goal": adv,
                    "weakest_goal": weakest, "top_feature": top, "text": t})
    sys.stdout.write(json.dumps({"explanations": out}))


if __name__ == "__main__":
    main()
