import sys, csv, json, math

G = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]
NUM = ["Clndrs", "Volume", "Model"]


def load(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fnum(s):
    s = (s or "").strip()
    if s == "" or s == "?":
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


def mk_norm(m, sd):
    def f(v):
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))
    return f


def main():
    a = sys.argv
    tr = load(a[a.index("--train") + 1])
    te = load(a[a.index("--test") + 1])
    gn = {}
    for g, h in G:
        gn[g] = mk_norm(*stats([fnum(r[g]) for r in tr if fnum(r[g]) is not None]))
    xn = {}
    for c in NUM:
        xn[c] = mk_norm(*stats([fnum(r[c]) for r in tr if fnum(r[c]) is not None]))

    def gaps(r):
        return {g: abs(gn[g](fnum(r[g])) - h) for g, h in G}

    def d2h(r):
        return math.sqrt(sum(v * v for v in gaps(r).values()) / 3)

    tsc = [d2h(r) for r in tr]
    cutoff = sorted(tsc)[len(tr) // 2]
    order = sorted(range(len(tr)), key=lambda i: (round(tsc[i], 12), i))
    q = max(1, len(tr) // 4)
    good = [tr[i] for i in order[:q]]
    poor = [tr[i] for i in order[-q:]]
    centers = {}
    for c in NUM:
        cs = []
        for grp in (good, poor):
            v = [fnum(r[c]) for r in grp if fnum(r[c]) is not None]
            cs.append(xn[c](sum(v) / len(v)) if v else None)
        centers[c] = cs

    def freq(grp, val):
        k = [r["origin"] for r in grp if r["origin"].strip() not in ("", "?")]
        return sum(1 for x in k if x == val) / len(k) if k else 0.0

    def blame(r):
        gp = gaps(r)
        ft = {}
        for c in NUM:
            x = fnum(r[c])
            gc, pc = centers[c]
            if x is None or gc is None or pc is None:
                ft[c] = 0.0
            else:
                nx = xn[c](x)
                ft[c] = max(0.0, abs(nx - gc) - abs(nx - pc))
        o = r["origin"].strip()
        ft["origin"] = 0.0 if o in ("", "?") else max(0.0, freq(poor, r["origin"]) - freq(good, r["origin"]))
        srt = lambda d: sorted(d, key=lambda k: (-round(d[k], 10), k))
        return gp, ft, srt(gp), srt(ft)

    n = len(te)
    sc = [d2h(r) for r in te]
    bl = [blame(r) for r in te]
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
            gw, gl = bl[win][0], bl[lose][0]
            ad = {g: gl[g] ** 2 - gw[g] ** 2 for g, _ in G}
            adv = sorted(ad, key=lambda g: (-round(ad[g], 10), g))[0]
        gp, ft, go, fo = bl[j]
        weak = go[0]
        top = fo[0] if ft[fo[0]] > 1e-10 else None
        t = "Car %d %s: score %.3f, other score %.3f, cutoff %.3f; weakest goal %s." % (
            j, "accepted" if acc else "rejected", s, o, cutoff, weak)
        if win is not None:
            t += " Recommend car %d over car %d, mainly on %s." % (win, k if win == j else j, adv)
        else:
            t += " Scores tie."
        if top:
            t += " Top feature: %s." % top
        out.append({"row": j, "other": k, "score": s, "other_score": o, "cutoff": cutoff,
                    "accepted": acc, "winner": win, "advantage_goal": adv,
                    "weakest_goal": weak, "top_feature": top, "text": t})
    sys.stdout.write(json.dumps({"explanations": out}))


main()
