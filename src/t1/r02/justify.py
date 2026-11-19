import csv, sys, json, math

GOALS = ["Lbs-", "Acc+", "Mpg+"]
HEAVEN = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUM = ["Clndrs", "Volume", "Model"]
CAT = "origin"


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    v = (v or "").strip()
    if v == "" or v == "?":
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
        return 1 / (1 + math.exp(-1.702 * z))
    return f


def main():
    a = sys.argv
    train = read(a[a.index("--train") + 1])
    test = read(a[a.index("--test") + 1])
    gn = {g: mk_norm(*stats([num(r[g]) for r in train if num(r[g]) is not None])) for g in GOALS}
    fn = {c: mk_norm(*stats([num(r[c]) for r in train if num(r[c]) is not None])) for c in NUM}

    def gaps(r):
        return {g: gn[g](num(r[g])) - HEAVEN[g] for g in GOALS}

    def d2h(r):
        return math.sqrt(sum(x * x for x in gaps(r).values()) / 3)

    tr = [d2h(r) for r in train]
    te = [d2h(r) for r in test]
    cutoff = sorted(tr)[len(train) // 2]
    order = sorted(range(len(train)), key=lambda i: (round(tr[i], 12), i))
    q = max(1, len(train) // 4)
    good = [train[i] for i in order[:q]]
    poor = [train[i] for i in order[-q:]]
    centers = {}
    for c in NUM:
        gv = [num(r[c]) for r in good if num(r[c]) is not None]
        pv = [num(r[c]) for r in poor if num(r[c]) is not None]
        centers[c] = (fn[c](sum(gv) / len(gv)) if gv else None,
                      fn[c](sum(pv) / len(pv)) if pv else None)

    def freqs(grp):
        k = [r[CAT].strip() for r in grp if r[CAT].strip() not in ("?", "")]
        d = {}
        for x in k:
            d[x] = d.get(x, 0) + 1
        return d, len(k)
    gf, gnn = freqs(good)
    pf, pnn = freqs(poor)

    def blame(r):
        feats = {}
        for c in NUM:
            x = num(r[c])
            gc, pc = centers[c]
            if x is None or gc is None or pc is None:
                feats[c] = 0.0
            else:
                nx = fn[c](x)
                feats[c] = max(0.0, abs(nx - gc) - abs(nx - pc))
        v = r[CAT].strip()
        if v in ("?", ""):
            feats[CAT] = 0.0
        else:
            p = pf.get(v, 0) / pnn if pnn else 0.0
            g = gf.get(v, 0) / gnn if gnn else 0.0
            feats[CAT] = max(0.0, p - g)
        gl = {g: abs(x) for g, x in gaps(r).items()}
        go = sorted(gl, key=lambda n: (-round(gl[n], 10), n))
        fo = sorted(feats, key=lambda n: (-round(feats[n], 10), n))
        return gl, feats, go, fo

    n = len(test)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = te[j], te[k]
        acc = round(s, 12) <= round(cutoff, 12)
        if round(s, 12) == round(o, 12):
            w = None
        else:
            w = j if s < o else k
        adv = None
        if w is not None:
            l = k if w == j else j
            gw, gl_ = gaps(test[w]), gaps(test[l])
            ad = {g: gl_[g] ** 2 - gw[g] ** 2 for g in GOALS}
            adv = sorted(GOALS, key=lambda g: (-round(ad[g], 10), g))[0]
        _, feats, go, fo = blame(test[j])
        weak = go[0]
        top = fo[0] if feats[fo[0]] > 1e-10 else None
        t = "Car %d %s: score %.3f, other score %.3f, cutoff %.3f; weakest goal %s." % (
            j, "accepted" if acc else "rejected", s, o, cutoff, weak)
        if w is not None:
            t += " Recommend car %d; largest advantage in %s." % (w, adv)
        else:
            t += " The comparison is a tie."
        if top is not None:
            t += " Top feature: %s." % top
        out.append({"row": j, "other": k, "score": s, "other_score": o, "cutoff": cutoff,
                    "accepted": acc, "winner": w, "advantage_goal": adv,
                    "weakest_goal": weak, "top_feature": top, "text": t})
    sys.stdout.write(json.dumps({"explanations": out}))


main()
