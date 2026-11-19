import sys, csv, json, math

def load(p):
    with open(p, newline='') as f:
        r = list(csv.reader(f))
    h = [c.strip() for c in r[0]]
    return h, [[c.strip() for c in row] for row in r[1:] if row]

def num(s):
    return None if s == '?' else float(s)

def stats(v):
    v = [x for x in v if x is not None]
    n = len(v)
    m = sum(v) / n if n else 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    return m, sd

def norm(v, st):
    z = (v - st[0]) / (st[1] + 1e-32)
    z = max(-3.0, min(3.0, z))
    return 1 / (1 + math.exp(-1.702 * z))

def main():
    a = sys.argv
    tr = a[a.index('--train') + 1]
    te = a[a.index('--test') + 1]
    h, T = load(tr)
    _, E = load(te)
    ix = {c: i for i, c in enumerate(h)}
    goals = ['Lbs-', 'Acc+', 'Mpg+']
    heaven = {'Lbs-': 0.0, 'Acc+': 1.0, 'Mpg+': 1.0}
    nums = ['Clndrs', 'Volume', 'Model']
    cat = 'origin'
    feats = nums + [cat]
    gst = {g: stats([num(r[ix[g]]) for r in T]) for g in goals}
    nst = {f: stats([num(r[ix[f]]) for r in T]) for f in nums}

    def gaps(r):
        return {g: abs(norm(float(r[ix[g]]), gst[g]) - heaven[g]) for g in goals}

    def d2h(r):
        gp = gaps(r)
        return math.sqrt(sum(x * x for x in gp.values()) / 3)

    td = [d2h(r) for r in T]
    cutoff = sorted(td)[len(T) // 2]
    order = sorted(range(len(T)), key=lambda i: (round(td[i], 12), i))
    q = max(1, len(T) // 4)
    good = [T[i] for i in order[:q]]
    poor = [T[i] for i in order[-q:]]

    centers = {}
    for f in nums:
        c = []
        for grp in (good, poor):
            v = [num(r[ix[f]]) for r in grp]
            v = [x for x in v if x is not None]
            c.append(norm(sum(v) / len(v), nst[f]) if v else None)
        centers[f] = c

    def freq(grp, val):
        k = [r[ix[cat]] for r in grp if r[ix[cat]] != '?']
        return k.count(val) / len(k) if k else 0.0

    def blame(r):
        gp = gaps(r)
        fe = {}
        for f in nums:
            x = num(r[ix[f]])
            gc, pc = centers[f]
            if x is None or gc is None or pc is None:
                fe[f] = 0.0
            else:
                nx = norm(x, nst[f])
                fe[f] = max(0.0, abs(nx - gc) - abs(nx - pc))
        v = r[ix[cat]]
        fe[cat] = 0.0 if v == '?' else max(0.0, freq(poor, v) - freq(good, v))
        go = sorted(gp, key=lambda n: (-round(gp[n], 10), n))
        fo = sorted(fe, key=lambda n: (-round(fe[n], 10), n))
        return gp, fe, go, fo

    sc = [d2h(r) for r in E]
    n = len(E)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = sc[j], sc[k]
        acc = round(s, 12) <= round(cutoff, 12)
        win = None
        adv = None
        if round(s, 12) != round(o, 12):
            win = j if s < o else k
            los = k if win == j else j
            gw = gaps(E[win])
            gl = gaps(E[los])
            ad = {g: gl[g] ** 2 - gw[g] ** 2 for g in goals}
            adv = sorted(goals, key=lambda g: (-round(ad[g], 10), g))[0]
        _, fe, go, fo = blame(E[j])
        weak = go[0]
        top = fo[0] if fe[fo[0]] > 1e-10 else None
        t = f"Car {j} {'accepted' if acc else 'rejected'}: score {s:.3f}, other score {o:.3f}, cutoff {cutoff:.3f}; weakest goal {weak}."
        if win is not None:
            t += f" Recommend car {win} (largest advantage goal {adv})."
        else:
            t += " Result is a tie."
        if top:
            t += f" Top feature {top}."
        out.append({"row": j, "other": k, "score": s, "other_score": o, "cutoff": cutoff,
                    "accepted": acc, "winner": win, "advantage_goal": adv,
                    "weakest_goal": weak, "top_feature": top, "text": t})
    print(json.dumps({"explanations": out}))

if __name__ == '__main__':
    main()
