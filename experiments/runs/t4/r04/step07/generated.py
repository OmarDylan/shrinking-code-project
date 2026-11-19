import csv, sys, math, argparse, json
from bisect import bisect_left, bisect_right

G = [("Lbs-", 0), ("Acc+", 1), ("Mpg+", 1)]
F = ["Clndrs", "Volume", "Model"]


def read(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return head, rows


def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def load(path):
    head, rows = read(path)
    idx = [head.index(g) for g, _ in G]
    out = []
    for row in rows:
        vals = [num(row[i]) for i in idx]
        out.append(None if None in vals else vals)
    return out


def get_stats(tr):
    n = len(tr)
    stats = []
    for j in range(3):
        col = [r[j] for r in tr]
        m = sum(col) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    return stats


def norm(v, m, sd):
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))


def d2h(r, stats):
    s = 0.0
    for j, (_, h) in enumerate(G):
        m, sd = stats[j]
        s += (norm(r[j], m, sd) - h) ** 2
    return math.sqrt(s / 3)


def remember(a):
    tr = [r for r in load(a.train) if r is not None]
    te = load(a.test)
    stats = get_stats(tr)
    scores = [d2h(r, stats) for r in te]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }


def guess(a):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    th, trows = read(a.train)
    eh, erows = read(a.test)
    gi = [th.index(g) for g, _ in G]
    ti = [th.index(c) for c in F]
    to = th.index("origin")
    ei = [eh.index(c) for c in F]
    eo = eh.index("origin")

    keep = [r for r in trows if all(num(r[i]) is not None for i in gi)]
    goals = [[float(r[i]) for i in gi] for r in keep]
    stats = get_stats(goals)
    y_mpg = np.array([g[2] for g in goals])
    y_d = np.array([d2h(g, stats) for g in goals])

    cats = sorted({r[to] for r in keep})
    meds = []
    for k in range(3):
        v = [num(r[ti[k]]) for r in keep if num(r[ti[k]]) is not None]
        meds.append(float(np.median(v)) if v else 0.0)

    def feats(row, idxs, oi, use_vol):
        x = []
        for k in range(3):
            if k == 1 and not use_vol:
                continue
            v = num(row[idxs[k]])
            x.append(meds[k] if v is None else v)
        x += [1.0 if row[oi] == c else 0.0 for c in cats]
        return x

    def fit_predict(use_vol, Xte, y):
        Xtr = np.array([feats(r, ti, to, use_vol) for r in keep])
        ps = []
        for m in (RandomForestRegressor(300, min_samples_leaf=2, random_state=0),
                  GradientBoostingRegressor(random_state=0, n_estimators=150,
                                            max_depth=2, learning_rate=0.05)):
            m.fit(Xtr, y)
            ps.append(m.predict(Xte))
        return np.mean(ps, axis=0)

    n = len(erows)
    mpg = [0.0] * n
    sc = [0.0] * n
    for use_vol in (True, False):
        ids = [i for i in range(n)
               if (num(erows[i][ei[1]]) is not None) == use_vol]
        if not ids:
            continue
        X = np.array([feats(erows[i], ei, eo, use_vol) for i in ids])
        pm = fit_predict(use_vol, X, y_mpg)
        pd = fit_predict(use_vol, X, y_d)
        for k, i in enumerate(ids):
            mpg[i] = max(float(pm[k]), 1.0)
            sc[i] = min(1.0, max(0.0, float(pd[k])))
    return {"mpg": mpg, "scores": sc}


def compare(xs, ys):
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((v - mx) ** 2 for v in xs) / (n - 1)
    vy = sum((v - my) ** 2 for v in ys) / (m - 1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (math.sqrt(pv) + 1e-32)
    sx, sy = sorted(xs), sorted(ys)
    tot = 0
    for x in xs:
        tot += bisect_left(sy, x) - (m - bisect_right(sy, x))
    cliffs = abs(tot) / (n * m)
    d = max(abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m)
            for v in set(sx) | set(sy))
    ks = d / math.sqrt(1 / n + 1 / m)
    same = bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks, "same": same}


def certify(a):
    with open(a.test) as f:
        cases = json.load(f)
    return {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}


def flow(a):
    out = guess(a)
    tr = [r for r in load(a.train) if r is not None]
    stats = get_stats(tr)
    hist = [d2h(r, stats) for r in tr]
    sc = out["scores"]
    alerts = []
    for i in range(len(sc)):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, sc[i - 19:i + 1])["same"])
    out["alerts"] = alerts
    return out


def blame(a):
    th, trows = read(a.train)
    eh, erows = read(a.test)
    gi = [th.index(g) for g, _ in G]
    keep = [r for r in trows if all(num(r[i]) is not None for i in gi)]
    goals = [[float(r[i]) for i in gi] for r in keep]
    stats = get_stats(goals)
    ds = [d2h(g, stats) for g in goals]
    order = sorted(range(len(keep)), key=lambda i: (round(ds[i], 12), i))
    q = max(1, len(keep) // 4)
    good = [keep[i] for i in order[:q]]
    poor = [keep[i] for i in order[-q:]]

    def known(s):
        return s != "" and s != "?"

    # numeric inputs
    prof = {}
    for f in F:
        j = th.index(f)
        col = [num(r[j]) for r in keep]
        col = [v for v in col if v is not None]
        if not col:
            prof[f] = None
            continue
        m = sum(col) / len(col)
        sd = math.sqrt(sum((v - m) ** 2 for v in col) / (len(col) - 1)) if len(col) > 1 else 0.0
        cen = []
        for grp in (good, poor):
            gv = [num(r[j]) for r in grp]
            gv = [v for v in gv if v is not None]
            cen.append(norm(sum(gv) / len(gv), m, sd) if gv else None)
        prof[f] = (m, sd, cen[0], cen[1])

    # categorical
    to = th.index("origin")
    freq = []
    for grp in (good, poor):
        vals = [r[to] for r in grp if known(r[to])]
        c = {}
        for v in vals:
            c[v] = c.get(v, 0) + 1
        freq.append((c, len(vals)))

    def cat_ev(v):
        if not known(v):
            return 0.0
        fs = []
        for c, n in freq:
            fs.append(c.get(v, 0) / n if n else 0.0)
        return max(0.0, fs[1] - fs[0])

    egi = [eh.index(g) for g, _ in G]
    efi = {f: eh.index(f) for f in F}
    eo = eh.index("origin")

    def order_of(d):
        return sorted(d, key=lambda k: (-round(d[k], 10), k))

    out = []
    for row in erows:
        goal = {}
        for k, (g, h) in enumerate(G):
            v = num(row[egi[k]])
            goal[g] = 0.0 if v is None else abs(norm(v, *stats[k]) - h)
        feat = {}
        for f in F:
            v = num(row[efi[f]])
            p = prof[f]
            if v is None or p is None or p[2] is None or p[3] is None:
                feat[f] = 0.0
            else:
                x = norm(v, p[0], p[1])
                feat[f] = max(0.0, abs(x - p[2]) - abs(x - p[3]))
        feat["origin"] = cat_ev(row[eo])
        out.append({"goals": goal, "features": feat,
                    "goal_order": order_of(goal),
                    "feature_order": order_of(feat)})
    return {"diagnoses": out}


def justify(a):
    tr = [r for r in load(a.train) if r is not None]
    te = load(a.test)
    stats = get_stats(tr)
    hist = sorted(d2h(r, stats) for r in tr)
    cutoff = hist[len(hist) // 2]
    scores = [d2h(r, stats) for r in te]
    diags = blame(a)["diagnoses"]
    n = len(te)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = scores[j], scores[k]
        acc = bool(round(s, 12) <= round(cutoff, 12))
        win = None
        adv_goal = None
        if round(s, 12) != round(o, 12):
            win = j if s < o else k
            lose = k if win == j else j
            adv = {}
            for gi, (g, h) in enumerate(G):
                m, sd = stats[gi]
                gl = (norm(te[lose][gi], m, sd) - h) ** 2
                gw = (norm(te[win][gi], m, sd) - h) ** 2
                adv[g] = gl - gw
            adv_goal = sorted(adv, key=lambda g: (-round(adv[g], 10), g))[0]
        d = diags[j]
        weak = d["goal_order"][0]
        tf = d["feature_order"][0]
        top = tf if d["features"][tf] > 1e-10 else None
        text = (f"Car {j} {'accepted' if acc else 'rejected'}: score {s:.3f}, "
                f"other score {o:.3f}, cutoff {cutoff:.3f}; weakest goal {weak}. ")
        if win is not None:
            text += f"Recommend car {win} over car {k if win == j else j}, mainly on {adv_goal}. "
        else:
            text += "The comparison is a tie. "
        if top is not None:
            text += f"Top feature {top} (retrospective, not causal)."
        out.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": acc, "winner": win,
                    "advantage_goal": adv_goal, "weakest_goal": weak,
                    "top_feature": top, "text": text.strip()})
    return {"explanations": out}


def choose(a):
    tr = [r for r in load(a.train) if r is not None]
    stats = get_stats(tr)
    pred = guess(a)["scores"]
    n = len(pred)
    order = sorted(range(n), key=lambda i: (pred[i], i))
    seen = {}
    for i in order[:min(10, n)]:
        sys.stdout.write(json.dumps({"inspect": i}) + "\n")
        sys.stdout.flush()
        line = sys.stdin.readline()
        resp = json.loads(line)
        g = resp["goals"]
        seen[i] = d2h([float(g[name]) for name, _ in G], stats)
    ranked = sorted(seen, key=lambda i: (round(seen[i], 12), i))
    return {"selected": ranked[:3]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    if a.cmd == "guess":
        out = guess(a)
    elif a.cmd == "certify":
        out = certify(a)
    elif a.cmd == "flow":
        out = flow(a)
    elif a.cmd == "blame":
        out = blame(a)
    elif a.cmd == "justify":
        out = justify(a)
    elif a.cmd == "choose":
        out = choose(a)
    else:
        out = remember(a)
    print(json.dumps(out))
    sys.stdout.flush()


main()
