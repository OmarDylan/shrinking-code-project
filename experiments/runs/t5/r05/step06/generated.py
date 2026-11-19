import sys, json, math, argparse

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        L = [l.strip().split(",") for l in f if l.strip()]
    h = [s.strip() for s in L[0]]
    return h, [[None if c.strip() == "?" else c.strip() for c in r] for r in L[1:]]

def stats(vs):
    n = len(vs); m = sum(vs) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vs) / (n - 1)) if n > 1 else 0.0
    return m, sd

def nrm(v, st):
    z = max(-3, min(3, (v - st[0]) / (st[1] + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("--train"); ap.add_argument("--test")
    a = ap.parse_args()
    h, tr = load(a.train)
    _, te = load(a.test)
    G = {5: 0, 6: 1, 7: 1}
    gn = {i: h[i] for i in G}
    gst = {i: stats([float(r[i]) for r in tr if r[i] is not None]) for i in G}
    NUM = [0, 1, 3]
    ist = {i: stats([float(r[i]) for r in tr if r[i] is not None]) for i in NUM}

    def gaps(r):
        return {i: abs(nrm(float(r[i]), gst[i]) - G[i]) for i in G}

    def d2h(r):
        return math.sqrt(sum(v * v for v in gaps(r).values()) / 3)

    td = [d2h(r) for r in tr]
    cutoff = sorted(td)[len(tr) // 2]
    order = sorted(range(len(tr)), key=lambda i: (round(td[i], 12), i))
    q = max(1, len(tr) // 4)
    good = [tr[i] for i in order[:q]]
    poor = [tr[i] for i in order[-q:]]

    cen = {}
    for i in NUM:
        c = []
        for grp in (good, poor):
            vs = [float(r[i]) for r in grp if r[i] is not None]
            c.append(nrm(sum(vs) / len(vs), ist[i]) if vs else None)
        cen[i] = c

    def freq(grp, v):
        k = [r[4] for r in grp if r[4] is not None]
        return k.count(v) / len(k) if k else 0.0

    def feats(r):
        out = {}
        for i in NUM:
            if r[i] is None or cen[i][0] is None or cen[i][1] is None:
                out[h[i]] = 0.0
            else:
                x = nrm(float(r[i]), ist[i])
                out[h[i]] = max(0.0, abs(x - cen[i][0]) - abs(x - cen[i][1]))
        out[h[4]] = 0.0 if r[4] is None else max(0.0, freq(poor, r[4]) - freq(good, r[4]))
        return out

    scores = [d2h(r) for r in te]
    n = len(te)
    res = []
    for j in range(n):
        k = (j + 1) % n
        s, o = scores[j], scores[k]
        gj = {gn[i]: v for i, v in gaps(te[j]).items()}
        gk = {gn[i]: v for i, v in gaps(te[k]).items()}
        fe = feats(te[j])
        gord = sorted(gj, key=lambda x: (-round(gj[x], 10), x))
        ford = sorted(fe, key=lambda x: (-round(fe[x], 10), x))
        acc = round(s, 12) <= round(cutoff, 12)
        win = None; ag = None
        if round(s, 12) != round(o, 12):
            win = j if s < o else k
            w, l = (gj, gk) if win == j else (gk, gj)
            adv = {g: l[g] ** 2 - w[g] ** 2 for g in gj}
            ag = sorted(adv, key=lambda x: (-round(adv[x], 10), x))[0]
        weak = gord[0]
        top = ford[0] if fe[ford[0]] > 1e-10 else None
        t = (f"Car {j} {'accepted' if acc else 'rejected'}: score {s:.3f}, "
             f"other score {o:.3f}, cutoff {cutoff:.3f}. Weakest goal {weak}. ")
        t += f"Recommend car {win}; biggest advantage {ag}. " if win is not None else "Result is a tie. "
        if top:
            t += f"Top feature {top}. "
        t += "Retrospective, not causal."
        res.append({"row": j, "other": k, "score": s, "other_score": o, "cutoff": cutoff,
                    "accepted": acc, "winner": win, "advantage_goal": ag,
                    "weakest_goal": weak, "top_feature": top, "text": t})
    sys.stdout.write(json.dumps({"explanations": res}))

main()
