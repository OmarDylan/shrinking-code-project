import csv, sys, json, math

GOALS = ["Lbs-", "Acc+", "Mpg+"]
HEAVEN = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUMS = ["Clndrs", "Volume", "Model"]
CAT = "origin"
FEATS = NUMS + [CAT]


def read(path):
    with open(path, newline="") as f:
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


def make_norm(m, sd):
    def f(v):
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1.0 / (1.0 + math.exp(-1.702 * z))
    return f


def main():
    args = sys.argv[1:]
    if not args or args[0] != "justify":
        sys.exit(1)
    train = test = None
    for i, a in enumerate(args):
        if a == "--train":
            train = args[i + 1]
        elif a == "--test":
            test = args[i + 1]
    tr, te = read(train), read(test)

    gn = {}
    for g in GOALS:
        m, sd = stats([num(r[g]) for r in tr if num(r[g]) is not None])
        gn[g] = make_norm(m, sd)

    def gaps(r):
        return {g: abs(gn[g](num(r[g])) - HEAVEN[g]) for g in GOALS}

    def d2h(r):
        return math.sqrt(sum(x * x for x in gaps(r).values()) / 3)

    trd = [d2h(r) for r in tr]
    cutoff = sorted(trd)[len(tr) // 2]

    # blame setup
    order = sorted(range(len(tr)), key=lambda i: (round(trd[i], 12), i))
    q = max(1, len(tr) // 4)
    good = [tr[i] for i in order[:q]]
    poor = [tr[i] for i in order[-q:]]
    fn, centers = {}, {}
    for c in NUMS:
        vals = [num(r[c]) for r in tr if num(r[c]) is not None]
        m, sd = stats(vals)
        fn[c] = make_norm(m, sd)
        cs = []
        for grp in (good, poor):
            gv = [num(r[c]) for r in grp if num(r[c]) is not None]
            cs.append(fn[c](sum(gv) / len(gv)) if gv else None)
        centers[c] = cs

    def freq(grp, cat):
        known = [r[CAT].strip() for r in grp if r[CAT].strip() not in ("?", "")]
        return known.count(cat) / len(known) if known else 0.0

    def evidence(r):
        ev = {}
        for c in NUMS:
            x = num(r[c])
            gc, pc = centers[c]
            if x is None or gc is None or pc is None:
                ev[c] = 0.0
            else:
                nx = fn[c](x)
                ev[c] = max(0.0, abs(nx - gc) - abs(nx - pc))
        cat = r[CAT].strip()
        if cat in ("?", ""):
            ev[CAT] = 0.0
        else:
            ev[CAT] = max(0.0, freq(poor, cat) - freq(good, cat))
        return ev

    scores = [d2h(r) for r in te]
    out = []
    n = len(te)
    for j in range(n):
        k = (j + 1) % n
        s, o = scores[j], scores[k]
        gj, gk = gaps(te[j]), gaps(te[k])
        accepted = round(s, 12) <= round(cutoff, 12)
        winner = None
        adv_goal = None
        if round(s, 12) != round(o, 12):
            winner = j if s < o else k
            wg, lg = (gj, gk) if winner == j else (gk, gj)
            adv = {g: lg[g] ** 2 - wg[g] ** 2 for g in GOALS}
            adv_goal = sorted(GOALS, key=lambda g: (-round(adv[g], 10), g))[0]
        gorder = sorted(GOALS, key=lambda g: (-round(gj[g], 10), g))
        ev = evidence(te[j])
        forder = sorted(FEATS, key=lambda f: (-round(ev[f], 10), f))
        weakest = gorder[0]
        top = forder[0] if ev[forder[0]] > 1e-10 else None
        text = "Car %d %s: score %.3f, other score %.3f, cutoff %.3f; weakest goal %s." % (
            j, "accepted" if accepted else "rejected", s, o, cutoff, weakest)
        if winner is not None:
            text += " Recommend car %d, largest advantage in %s." % (winner, adv_goal)
        else:
            text += " Scores tie."
        if top is not None:
            text += " Top feature %s (retrospective, not causal)." % top
        out.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": accepted, "winner": winner,
                    "advantage_goal": adv_goal, "weakest_goal": weakest,
                    "top_feature": top, "text": text})
    sys.stdout.write(json.dumps({"explanations": out}))


main()
