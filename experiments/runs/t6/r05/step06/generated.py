import sys, csv, json, argparse
from bisect import bisect_left, bisect_right
from math import exp, sqrt

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
NUMX = ["Clndrs", "Volume", "Model"]


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def goal_stats(names, tr, tnames):
    stats = {}
    for g, h in GOALS.items():
        i = names.index(g)
        xs = [float(r[i]) for r in tr]
        n = len(xs)
        mu = sum(xs) / n
        sd = sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        stats[g] = (tnames.index(g) if tnames is not None else i, mu, sd, h)
    return stats


def d2h(vals, stats):
    s = 0.0
    for v, (_, mu, sd, h) in zip(vals, stats.values()):
        z = max(-3.0, min(3.0, (v - mu) / (sd + 1e-32)))
        s += (1 / (1 + exp(-1.702 * z)) - h) ** 2
    return sqrt(s / 3)


def remember(names, tr, tnames, te):
    stats = goal_stats(names, tr, tnames)
    scores = [d2h([float(r[i]) for i, _, _, _ in stats.values()], stats) for r in te]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    n = len(order)
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[n // 2], "worst": order[-1]}


def guess(names, tr, tnames, te, k=7):
    gi = {g: names.index(g) for g in GOALS}
    tr = [r for r in tr if all(num(r[i]) is not None for i in gi.values())]
    stats = goal_stats(names, tr, None)
    mi = gi["Mpg+"]
    train_d = [d2h([float(r[i]) for i, _, _, _ in stats.values()], stats) for r in tr]
    train_m = [float(r[mi]) for r in tr]
    xi = [(c, names.index(c)) for c in NUMX if c in names]
    oi = names.index("origin") if "origin" in names else None
    tx = {c: tnames.index(c) for c in NUMX + ["origin"] if c in tnames}
    sds = {}
    for c, i in xi:
        v = [num(r[i]) for r in tr if num(r[i]) is not None]
        mu = sum(v) / len(v)
        sds[c] = sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)) or 1.0
    mean_m = sum(train_m) / len(train_m)
    mean_d = sum(train_d) / len(train_d)
    mpg, sc = [], []
    for r in te:
        q = {}
        for c, i in xi:
            if c in tx and tx[c] < len(r):
                q[c] = num(r[tx[c]])
        qo = r[tx["origin"]] if "origin" in tx and tx["origin"] < len(r) else "?"
        ds = []
        for j, t in enumerate(tr):
            s, n = 0.0, 0
            for c, i in xi:
                a, b = q.get(c), num(t[i])
                if a is not None and b is not None:
                    s += ((a - b) / sds[c]) ** 2
                    n += 1
            if oi is not None and qo not in ("?", "") and t[oi] not in ("?", ""):
                s += 0.5 * (qo != t[oi])
                n += 1
            ds.append((sqrt(s / n) if n else 1.0, j))
        ds.sort()
        nb = ds[:k]
        w = [1 / (d + 0.1) for d, _ in nb]
        tw = sum(w)
        m = sum(wi * train_m[j] for wi, (_, j) in zip(w, nb)) / tw if tw else mean_m
        d = sum(wi * train_d[j] for wi, (_, j) in zip(w, nb)) / tw if tw else mean_d
        mpg.append(max(m, 1e-3))
        sc.append(max(0.0, min(1.0, d)))
    return {"mpg": mpg, "scores": sc}


def compare(xs, ys):
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    vy = sum((y - my) ** 2 for y in ys) / (m - 1)
    pool = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (sqrt(pool) + 1e-32)
    sx, sy = sorted(xs), sorted(ys)
    tot = 0
    for x in sx:
        tot += bisect_left(sy, x) - (m - bisect_right(sy, x))
    cliffs = abs(tot) / (n * m)
    d = 0.0
    for v in set(sx) | set(sy):
        d = max(d, abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m))
    ks = d / sqrt(1 / n + 1 / m)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks,
            "same": bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)}


def certify(path):
    with open(path, encoding="utf-8-sig") as f:
        cases = json.load(f)
    return {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}


def flow(names, tr, tnames, te, win=20):
    out = guess(names, tr, tnames, te)
    gi = [names.index(g) for g in GOALS]
    full = [r for r in tr if all(num(r[i]) is not None for i in gi)]
    stats = goal_stats(names, full, None)
    hist = [d2h([float(r[i]) for i in gi], stats) for r in full]
    sc = out["scores"]
    alerts = []
    for i in range(len(sc)):
        if i < win - 1:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, sc[i - win + 1:i + 1])["same"])
    out["alerts"] = alerts
    return out


def nrm(v, mu, sd):
    z = max(-3.0, min(3.0, (v - mu) / (sd + 1e-32)))
    return 1 / (1 + exp(-1.702 * z))


def blame(names, tr, tnames, te):
    gi = [names.index(g) for g in GOALS]
    full = [r for r in tr if all(num(r[i]) is not None for i in gi)]
    stats = goal_stats(names, full, None)
    ds = [d2h([float(r[i]) for i in gi], stats) for r in full]
    order = sorted(range(len(full)), key=lambda i: (round(ds[i], 12), i))
    q = max(1, len(full) // 4)
    good = [full[i] for i in order[:q]]
    poor = [full[i] for i in order[-q:]]

    nx = {}
    for c in NUMX:
        i = names.index(c)
        v = [num(r[i]) for r in full]
        v = [x for x in v if x is not None]
        if not v:
            nx[c] = None
            continue
        mu = sum(v) / len(v)
        sd = sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0

        def center(g, i=i, mu=mu, sd=sd):
            w = [num(r[i]) for r in g]
            w = [x for x in w if x is not None]
            return nrm(sum(w) / len(w), mu, sd) if w else None
        nx[c] = (mu, sd, center(good), center(poor))

    oi = names.index("origin")

    def freqs(g):
        w = [r[oi] for r in g if r[oi] not in ("?", "")]
        f = {}
        for x in w:
            f[x] = f.get(x, 0) + 1
        return {k: v / len(w) for k, v in f.items()}
    fg, fp = freqs(good), freqs(poor)

    tx = {c: tnames.index(c) for c in NUMX + ["origin"] + list(GOALS) if c in tnames}
    out = []
    for r in te:
        goals = {}
        for g, h in GOALS.items():
            v = num(r[tx[g]]) if g in tx and tx[g] < len(r) else None
            if v is None:
                goals[g] = 0.0
            else:
                _, mu, sd, _h = stats[g]
                goals[g] = abs(nrm(v, mu, sd) - h)
        feats = {}
        for c in NUMX:
            ev = 0.0
            v = num(r[tx[c]]) if c in tx and tx[c] < len(r) else None
            if v is not None and nx[c] and nx[c][2] is not None and nx[c][3] is not None:
                mu, sd, gc, pc = nx[c]
                x = nrm(v, mu, sd)
                ev = max(0.0, abs(x - gc) - abs(x - pc))
            feats[c] = ev
        o = r[tx["origin"]] if "origin" in tx and tx["origin"] < len(r) else "?"
        feats["origin"] = 0.0 if o in ("?", "") else max(0.0, fp.get(o, 0.0) - fg.get(o, 0.0))
        srt = lambda d: [k for k in sorted(d, key=lambda k: (-round(d[k], 10), k))]
        out.append({"goals": goals, "features": feats,
                    "goal_order": srt(goals), "feature_order": srt(feats)})
    return {"diagnoses": out}


def justify(names, tr, tnames, te):
    gi = [names.index(g) for g in GOALS]
    full = [r for r in tr if all(num(r[i]) is not None for i in gi)]
    stats = goal_stats(names, full, None)
    td = sorted(d2h([float(r[i]) for i in gi], stats) for r in full)
    cutoff = td[len(full) // 2]
    scores = remember(names, tr, tnames, te)["scores"]
    dg = blame(names, tr, tnames, te)["diagnoses"]
    n = len(te)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = scores[j], scores[k]
        acc = round(s, 12) <= round(cutoff, 12)
        win = None
        adv = None
        if round(s, 12) != round(o, 12):
            win = j if s < o else k
            lose = k if win == j else j
            advs = {g: dg[lose]["goals"][g] ** 2 - dg[win]["goals"][g] ** 2
                    for g in GOALS}
            adv = sorted(advs, key=lambda g: (-round(advs[g], 10), g))[0]
        weak = dg[j]["goal_order"][0]
        f0 = dg[j]["feature_order"][0]
        top = f0 if dg[j]["features"][f0] > 1e-10 else None
        text = (f"Car {j} {'accepted' if acc else 'rejected'}: score {s:.3f}, "
                f"other score {o:.3f}, cutoff {cutoff:.3f}; weakest goal {weak}. ")
        if win is not None:
            text += f"Recommend car {win} over car {k if win == j else j}; biggest advantage {adv}. "
        else:
            text += "Scores tie. "
        if top is not None:
            text += f"Top feature {top}."
        out.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": bool(acc), "winner": win,
                    "advantage_goal": adv, "weakest_goal": weak,
                    "top_feature": top, "text": text.strip()})
    return {"explanations": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    if a.cmd == "certify":
        print(json.dumps(certify(a.test)))
        return
    names, tr = load(a.train)
    tnames, te = load(a.test)
    if a.cmd == "guess":
        out = guess(names, tr, tnames, te)
    elif a.cmd == "flow":
        out = flow(names, tr, tnames, te)
    elif a.cmd == "blame":
        out = blame(names, tr, tnames, te)
    elif a.cmd == "justify":
        out = justify(names, tr, tnames, te)
    else:
        out = remember(names, tr, tnames, te)
    print(json.dumps(out))


main()
