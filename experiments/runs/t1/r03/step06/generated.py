import csv, sys, json, math, argparse

GOALS = ["Lbs-", "Acc+", "Mpg+"]
HEAVEN = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUMS = ["Clndrs", "Volume", "Model"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(s):
    s = (s or "").strip()
    if s == "?" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def stats(vals):
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd


def make_norm(m, sd):
    def f(v):
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()
    train = read(a.train)
    test = read(a.test)
    gn = {}
    for g in GOALS:
        gn[g] = make_norm(*stats([num(r[g]) for r in train if num(r[g]) is not None]))
    nn = {}
    for c in NUMS:
        nn[c] = make_norm(*stats([num(r[c]) for r in train if num(r[c]) is not None]))

    def gaps(r):
        return {g: abs(gn[g](num(r[g])) - HEAVEN[g]) for g in GOALS}

    def d2h(r):
        return math.sqrt(sum(v * v for v in gaps(r).values()) / 3)

    tr = [d2h(r) for r in train]
    order = sorted(range(len(train)), key=lambda i: (round(tr[i], 12), i))
    q = max(1, len(train) // 4)
    good = [train[i] for i in order[:q]]
    poor = [train[i] for i in order[-q:]]
    centers = {}
    for c in NUMS:
        cs = []
        for grp in (good, poor):
            v = [num(r[c]) for r in grp if num(r[c]) is not None]
            cs.append(nn[c](sum(v) / len(v)) if v else None)
        centers[c] = cs

    def freq(grp, cat):
        k = [r["origin"].strip() for r in grp if r["origin"].strip() not in ("?", "")]
        return k.count(cat) / len(k) if k else 0.0

    def blame(r):
        goals = gaps(r)
        feats = {}
        for c in NUMS:
            x = num(r[c])
            gc, pc = centers[c]
            if x is None or gc is None or pc is None:
                feats[c] = 0.0
            else:
                nx = nn[c](x)
                feats[c] = max(0.0, abs(nx - gc) - abs(nx - pc))
        o = r["origin"].strip()
        feats["origin"] = 0.0 if o in ("?", "") else max(0.0, freq(poor, o) - freq(good, o))
        go = sorted(goals, key=lambda n: (-round(goals[n], 10), n))
        fo = sorted(feats, key=lambda n: (-round(feats[n], 10), n))
        return goals, feats, go, fo

    if a.cmd != "justify":
        sys.exit(1)
    cutoff = sorted(tr)[len(train) // 2]
    n = len(test)
    sc = [d2h(r) for r in test]
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
            gw, gl = gaps(test[win]), gaps(test[lose])
            ad = {g: gl[g] ** 2 - gw[g] ** 2 for g in GOALS}
            adv = sorted(GOALS, key=lambda g: (-round(ad[g], 10), g))[0]
        goals, feats, go, fo = blame(test[j])
        weak = go[0]
        top = fo[0] if feats[fo[0]] > 1e-10 else None
        t = "Car %d %s: score %.3f, other score %.3f, cutoff %.3f. Weakest goal: %s." % (
            j, "accepted" if acc else "rejected", s, o, cutoff, weak)
        if win is not None:
            t += " Recommend car %d; largest advantage in %s." % (win, adv)
        else:
            t += " Result is a tie."
        if top:
            t += " Top feature: %s." % top
        out.append({"row": j, "other": k, "score": s, "other_score": o, "cutoff": cutoff,
                    "accepted": acc, "winner": win, "advantage_goal": adv,
                    "weakest_goal": weak, "top_feature": top, "text": t})
    print(json.dumps({"explanations": out}))


main()
