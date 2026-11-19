import sys, csv, json, time, argparse, math
from statistics import median


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        h = [x.strip() for x in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return h, rows


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    h, tr = load(a.train)
    _, te = load(a.test)
    cols = [i for i, c in enumerate(h) if not c.endswith("X")]
    num = {i for i in cols if h[i][0].isupper()}
    name = lambda i: h[i].rstrip("+-")
    goals = [i for i in cols if h[i].endswith(("+", "-")) and i in num]
    feats = [i for i in cols if i not in goals]

    # imputation values
    fill = {}
    for i in cols:
        v = [x[i] for x in tr if x[i] != "?"]
        if i in num:
            fill[i] = median(float(x) for x in v)
        else:
            fill[i] = max(set(v), key=v.count)

    def clean(rows):
        out = []
        for r in rows:
            out.append({i: (float(r[i]) if r[i] != "?" else fill[i]) if i in num
                        else (r[i] if r[i] != "?" else fill[i]) for i in cols})
        return out

    T, E = clean(tr), clean(te)
    lo = {i: min(r[i] for r in T) for i in goals}
    hi = {i: max(r[i] for r in T) for i in goals}

    def contrib(r):
        c = {}
        for i in goals:
            x = (r[i] - lo[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 0.0
            c[i] = (1 - x) ** 2 if h[i].endswith("+") else x ** 2
        return c

    def score(c):
        return math.sqrt(sum(c.values()) / len(c))

    ts = [score(contrib(r)) for r in T]
    cutoff = median(ts)
    good = [1.0 if s <= cutoff else 0.0 for s in ts]
    gm = sum(good) / len(good)

    # feature evidence from training
    stats = {}
    for i in feats:
        if i in num:
            xs = [r[i] for r in T]
            m = sum(xs) / len(xs)
            sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
            cov = sum((x - m) * (g - gm) for x, g in zip(xs, good)) / len(xs)
            sg = math.sqrt(sum((g - gm) ** 2 for g in good) / len(good))
            corr = cov / (sd * sg) if sd > 0 and sg > 0 else 0.0
            stats[i] = (m, sd, corr)
        else:
            d = {}
            for r, g in zip(T, good):
                d.setdefault(r[i], []).append(g)
            stats[i] = {k: sum(v) / len(v) - gm for k, v in d.items()}

    def top_feature(r):
        best, bv = None, 0.0
        for i in feats:
            if i in num:
                m, sd, corr = stats[i]
                ev = (r[i] - m) / sd * corr if sd > 0 else 0.0
            else:
                ev = stats[i].get(r[i], 0.0)
            if ev > bv + 1e-12:
                best, bv = i, ev
        return best

    cs = [contrib(r) for r in E]
    sc = [score(c) for c in cs]
    n = len(E)
    res = []
    for k in range(n):
        o = (k + 1) % n
        s, so = sc[k], sc[o]
        acc = s <= cutoff
        weak = name(max(goals, key=lambda i: cs[k][i]))
        tf = top_feature(E[k])
        tfn = name(tf) if tf is not None else None
        tie = round(s, 12) == round(so, 12)
        if tie:
            win, adv = None, None
        else:
            win = k if s < so else o
            los = o if win == k else k
            g = max(goals, key=lambda i: cs[los][i] - cs[win][i])
            adv = name(g)
        txt = (f"Car {k} has distance-to-heaven {s:.6f} vs car {o} at {so:.6f}. ")
        if tie:
            txt += "The two cars tie on quality. "
        else:
            txt += (f"Car {win} is recommended because its lower distance to heaven; "
                    f"the largest advantage comes from {adv}. ")
        txt += (f"Car {k} is {'accepted' if acc else 'rejected'}: score {s:.6f} is "
                f"{'at or below' if acc else 'above'} the median training cutoff {cutoff:.6f}. "
                f"Its weakest goal is {weak}. ")
        if tfn:
            txt += f"Associated feature with positive evidence: {tfn} (retrospective, not causal)."
        else:
            txt += "No feature shows positive association evidence."
        res.append({"row": k, "other": o, "score": s, "other_score": so,
                    "cutoff": cutoff, "accepted": acc, "winner": win,
                    "advantage_goal": adv, "weakest_goal": weak,
                    "top_feature": tfn, "text": txt})
    print(json.dumps({"explanations": res, "count": n,
                      "execution_time_seconds": time.time() - t0}))


main()
