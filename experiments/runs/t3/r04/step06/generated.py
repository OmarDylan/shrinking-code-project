import csv, sys, json, math

G = ["Lbs-", "Acc+", "Mpg+"]
H = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}
NUM = ["Clndrs", "Volume", "Model"]
CAT = "origin"


def load(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        v = (v or "").strip()
        return None if v in ("", "?") else float(v)
    except ValueError:
        return None


def stats(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd


def norm(v, ms):
    m, sd = ms
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))


def main():
    a = sys.argv
    tr = load(a[a.index("--train") + 1])
    te = load(a[a.index("--test") + 1])
    cols = G + NUM
    st = {c: stats([fnum(r.get(c)) for r in tr]) for c in cols}

    def gaps(r):
        return {g: abs(norm(fnum(r[g]), st[g]) - H[g]) for g in G}

    def d2h(r):
        return math.sqrt(sum(x * x for x in gaps(r).values()) / 3)

    td = [d2h(r) for r in tr]
    cutoff = sorted(td)[len(tr) // 2]
    order = sorted(range(len(tr)), key=lambda i: (round(td[i], 12), i))
    q = max(1, len(tr) // 4)
    good = [tr[i] for i in order[:q]]
    poor = [tr[i] for i in order[-q:]]

    def center(grp, c):
        v = [fnum(r.get(c)) for r in grp]
        v = [x for x in v if x is not None]
        return norm(sum(v) / len(v), st[c]) if v else None

    gc = {c: center(good, c) for c in NUM}
    pc = {c: center(poor, c) for c in NUM}

    def freq(grp, cat):
        k = [r.get(CAT, "").strip() for r in grp]
        k = [x for x in k if x not in ("", "?")]
        return k.count(cat) / len(k) if k else 0.0

    def blame(r):
        gp = gaps(r)
        ev = {}
        for c in NUM:
            x = fnum(r.get(c))
            if x is None or gc[c] is None or pc[c] is None:
                ev[c] = 0.0
            else:
                nx = norm(x, st[c])
                ev[c] = max(0.0, abs(nx - gc[c]) - abs(nx - pc[c]))
        cat = (r.get(CAT) or "").strip()
        if cat in ("", "?"):
            ev[CAT] = 0.0
        else:
            ev[CAT] = max(0.0, freq(poor, cat) - freq(good, cat))
        go = sorted(gp, key=lambda n: (-round(gp[n], 10), n))
        fo = sorted(ev, key=lambda n: (-round(ev[n], 10), n))
        return gp, ev, go, fo

    sc = [d2h(r) for r in te]
    n = len(te)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = sc[j], sc[k]
        acc = round(s, 12) <= round(cutoff, 12)
        winner = None
        adv = None
        if round(s, 12) != round(o, 12):
            winner = j if s < o else k
            loser = k if winner == j else j
            gw = gaps(te[winner])
            gl = gaps(te[loser])
            d = {g: gl[g] ** 2 - gw[g] ** 2 for g in G}
            adv = sorted(G, key=lambda g: (-round(d[g], 10), g))[0]
        gp, ev, go, fo = blame(te[j])
        weak = go[0]
        top = fo[0] if ev[fo[0]] > 1e-10 else None
        t = "Car %d %s: score %.3f, other score %.3f, cutoff %.3f; weakest goal %s." % (
            j, "accepted" if acc else "rejected", s, o, cutoff, weak)
        if winner is not None:
            t += " Recommend car %d (largest advantage: %s)." % (winner, adv)
        else:
            t += " Scores tie."
        if top:
            t += " Top feature: %s." % top
        out.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": acc, "winner": winner,
                    "advantage_goal": adv, "weakest_goal": weak,
                    "top_feature": top, "text": t})
    sys.stdout.write(json.dumps({"explanations": out}))


if __name__ == "__main__":
    main()
