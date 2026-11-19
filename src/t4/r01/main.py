import csv, sys, json, math, argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}
COHEN_T, CLIFFS_T, KS_T = 0.35, 0.197, 1.36
WINDOW = 20
BUDGET, PICKS = 10, 3
K_NN, N_CONFIG, MIN_GAIN = 5, 12, 0.05
EDIT = ["Clndrs", "Volume"]
SPOT_COLS = ["Clndrs", "Volume", "Model", "origin"]
SPOT_Q, SPOT_FLOOR = 0.95, 3.0


def load(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    head = [h.strip() for h in rows[0]]
    return head, [[c.strip() for c in r] for r in rows[1:]]


def num(s):
    if s == "?" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def column(head, rows, name):
    if name not in head:
        return [None] * len(rows)
    i = head.index(name)
    return [num(r[i]) if i < len(r) else None for r in rows]


def sym(head, rows, name):
    if name not in head:
        return ["?"] * len(rows)
    i = head.index(name)
    return [r[i] if i < len(r) else "?" for r in rows]


def goal_stats(th, tr):
    stats = {}
    for g in GOALS:
        v = [x for x in column(th, tr, g) if x is not None]
        n = len(v)
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        stats[g] = (m, sd)
    return stats


def norm(x, g, stats):
    if x is None:
        return 0.5
    m, sd = stats[g]
    z = max(-3.0, min(3.0, (x - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))


def d2h(vals, stats):
    s = sum((norm(vals[g], g, stats) - h) ** 2 for g, h in GOALS.items())
    return math.sqrt(s / 3)


def remember(a):
    th, tr = load(a.train)
    eh, te = load(a.test)
    stats = goal_stats(th, tr)
    cols = {g: column(eh, te, g) for g in GOALS}
    scores = [d2h({g: cols[g][i] for g in GOALS}, stats) for i in range(len(te))]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }))


def predict_all(a):
    th, tr = load(a.train)
    eh, te = load(a.test)
    stats = goal_stats(th, tr)
    feats = [h for h in th if h and not h.endswith(("X", "+", "-"))]
    numc = [h for h in feats if h[0].isupper()]
    symc = [h for h in feats if not h[0].isupper()]
    med = {}
    for h in numc:
        v = [x for x in column(th, tr, h) if x is not None]
        med[h] = float(np.median(v)) if v else 0.0
    gmed = {}
    if "Clndrs" in th and "Volume" in th:
        d = {}
        for c, v in zip(column(th, tr, "Clndrs"), column(th, tr, "Volume")):
            if c is not None and v is not None:
                d.setdefault(c, []).append(v)
        gmed = {c: float(np.median(v)) for c, v in d.items()}
    cats = {h: sorted(set(sym(th, tr, h))) for h in symc}

    def build(head, rows, mask_vol=False):
        cl = column(head, rows, "Clndrs")
        X = []
        nc = {h: column(head, rows, h) for h in numc}
        sc = {h: sym(head, rows, h) for h in symc}
        for i in range(len(rows)):
            r = []
            for h in numc:
                x = nc[h][i]
                if h == "Volume" and (mask_vol or x is None):
                    x = gmed.get(cl[i]) if cl[i] is not None else None
                if x is None:
                    x = med[h]
                r.append(x)
            for h in symc:
                r.extend(1.0 if sc[h][i] == c else 0.0 for c in cats[h])
            X.append(r)
        return np.array(X, dtype=float).reshape(len(rows), -1)

    Xa = build(th, tr)
    Xb = build(th, tr, True)
    Xt = build(eh, te)
    pred = {}
    for g in GOALS:
        y = column(th, tr, g)
        idx = [i for i, v in enumerate(y) if v is not None]
        yy = np.array([y[i] for i in idx])
        X = np.vstack([Xa[idx], Xb[idx]])
        Y = np.concatenate([yy, yy])
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0)
        m.fit(X, Y)
        pred[g] = m.predict(Xt)
    mpg = [float(v) for v in pred["Mpg+"]]
    scores = [d2h({g: float(pred[g][i]) for g in GOALS}, stats) for i in range(len(te))]
    return mpg, scores, stats, th, tr


def guess(a):
    mpg, scores, _, _, _ = predict_all(a)
    print(json.dumps({"mpg": mpg, "scores": scores}))


def compare(xs, ys):
    x = np.sort(np.asarray(xs, dtype=float))
    y = np.sort(np.asarray(ys, dtype=float))
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return {"cohen": 0.0, "cliffs": 0.0, "ks": 0.0, "same": True}
    v1 = float(np.var(x, ddof=1)) if n > 1 else 0.0
    v2 = float(np.var(y, ddof=1)) if m > 1 else 0.0
    dof = n + m - 2
    pooled = math.sqrt(((n - 1) * v1 + (m - 1) * v2) / dof) if dof > 0 else 0.0
    diff = abs(float(x.mean()) - float(y.mean()))
    if pooled > 0:
        cohen = diff / pooled
    else:
        cohen = 0.0 if diff < 1e-12 else 1e9
    gt = np.searchsorted(y, x, side="left").sum()
    lt = (m - np.searchsorted(y, x, side="right")).sum()
    cliffs = abs(float(gt - lt)) / (n * m)
    allv = np.concatenate([x, y])
    cx = np.searchsorted(x, allv, side="right") / n
    cy = np.searchsorted(y, allv, side="right") / m
    d = float(np.max(np.abs(cx - cy)))
    ks = d * math.sqrt(n * m / (n + m))
    same = bool(cohen < COHEN_T and cliffs < CLIFFS_T and ks < KS_T)
    return {"cohen": float(cohen), "cliffs": float(cliffs), "ks": float(ks), "same": same}


def certify(a):
    with open(a.test) as f:
        cases = json.load(f)
    print(json.dumps({"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}))


def flow(a):
    mpg, scores, stats, th, tr = predict_all(a)
    cols = {g: column(th, tr, g) for g in GOALS}
    hist = [d2h({g: cols[g][i] for g in GOALS}, stats)
            for i in range(len(tr)) if all(cols[g][i] is not None for g in GOALS)]
    alerts = []
    for i in range(len(scores)):
        if i + 1 < WINDOW or not hist:
            alerts.append(False)
            continue
        win = scores[i + 1 - WINDOW:i + 1]
        alerts.append(not compare(win, hist)["same"])
    print(json.dumps({"mpg": mpg, "scores": scores, "alerts": alerts}))


def diagnose(a):
    th, tr = load(a.train)
    eh, te = load(a.test)
    stats = goal_stats(th, tr)
    gc = {g: column(th, tr, g) for g in GOALS}
    rows = [i for i in range(len(tr)) if all(gc[g][i] is not None for g in GOALS)]
    sc = {i: d2h({g: gc[g][i] for g in GOALS}, stats) for i in rows}
    order = sorted(rows, key=lambda i: (sc[i], i))
    k = max(1, int(len(order) * 0.25)) if order else 0
    good, poor = order[:k], order[len(order) - k:] if k else []

    inputs = [h for h in th if h and not h.endswith(("X", "+", "-"))]
    prof = {}
    for h in inputs:
        if h[0].isupper():
            v = [x for x in column(th, tr, h) if x is not None]
            n = len(v)
            m = sum(v) / n if n else 0.0
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
            st = {h: (m, sd)}
            col = column(th, tr, h)

            def center(grp, col=col, h=h, st=st):
                z = [norm(col[i], h, st) for i in grp if col[i] is not None]
                return sum(z) / len(z) if z else 0.5
            prof[h] = ("n", st, center(good), center(poor))
        else:
            col = sym(th, tr, h)

            def freq(grp, col=col):
                z = [col[i] for i in grp if col[i] not in ("?", "")]
                d = {}
                for s in z:
                    d[s] = d.get(s, 0) + 1
                return {s: c / len(z) for s, c in d.items()} if z else {}
            prof[h] = ("s", freq(good), freq(poor))

    tg = {g: column(eh, te, g) for g in GOALS}
    out = []
    for r in range(len(te)):
        goals = {}
        for g, hv in GOALS.items():
            goals[g] = abs(norm(tg[g][r], g, stats) - hv)
        feats = {}
        for h in inputs:
            p = prof[h]
            if p[0] == "n":
                x = column(eh, [te[r]], h)[0]
                if x is None:
                    s = 0.0
                else:
                    z = norm(x, h, p[1])
                    s = max(0.0, abs(z - p[2]) - abs(z - p[3]))
            else:
                v = sym(eh, [te[r]], h)[0]
                if v in ("?", ""):
                    s = 0.0
                else:
                    s = max(0.0, p[2].get(v, 0.0) - p[1].get(v, 0.0))
            feats[h] = float(s)
        out.append({
            "goals": goals,
            "features": feats,
            "goal_order": sorted(goals, key=lambda n: (-goals[n], n)),
            "feature_order": sorted(feats, key=lambda n: (-feats[n], n)),
        })
    return out, stats, list(sc.values()), tg, te


def blame(a):
    out = diagnose(a)[0]
    print(json.dumps({"diagnoses": out}))


def justify(a):
    diag, stats, hist, tg, te = diagnose(a)
    n = len(te)
    cutoff = float(np.median(hist)) if hist else 0.5
    scores = [d2h({g: tg[g][i] for g in GOALS}, stats) for i in range(n)]
    res = []
    for i in range(n):
        j = (i + 1) % n
        s, o = scores[i], scores[j]
        acc = bool(s <= cutoff)
        d = diag[i]
        weakest = d["goal_order"][0]
        top = None
        if d["feature_order"] and d["features"][d["feature_order"][0]] > 0:
            top = d["feature_order"][0]
        tie = round(s, 12) == round(o, 12)
        winner = adv = None
        if not tie:
            w, l = (i, j) if s < o else (j, i)
            winner = w
            gw, gl = diag[w]["goals"], diag[l]["goals"]
            contrib = {g: gl[g] ** 2 - gw[g] ** 2 for g in GOALS}
            adv = max(GOALS, key=lambda g: contrib[g])
        if tie:
            t = f"Car {i} and car {j} tie on quality (distance to heaven {s:.6f} vs {o:.6f})."
        else:
            t = (f"Car {winner} is recommended over car {j if winner == i else i} "
                 f"because its distance to heaven is lower ({min(s, o):.6f} vs {max(s, o):.6f}); "
                 f"the largest advantage comes from {adv}.")
        t += (f" Car {i} is {'accepted' if acc else 'rejected'}: score {s:.6f} is "
              f"{'at or below' if acc else 'above'} the median training cutoff {cutoff:.6f}."
              f" Its weakest goal is {weakest} (gap {d['goals'][weakest]:.4f}).")
        if top:
            t += f" Feature {top} is most associated with poor cars (evidence {d['features'][top]:.4f})."
        else:
            t += " No input feature shows positive association with poor cars."
        t += " This is retrospective, not causal."
        res.append({"row": i, "other": j, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": acc, "winner": winner,
                    "advantage_goal": adv, "weakest_goal": weakest,
                    "top_feature": top, "text": t})
    print(json.dumps({"explanations": res}))


def find_goals(obj):
    """Locate goal values in an evaluator response (flat or nested)."""
    if isinstance(obj, dict):
        if any(g in obj for g in GOALS):
            out = {}
            for g in GOALS:
                v = obj.get(g)
                out[g] = float(v) if isinstance(v, (int, float)) else num(str(v)) if v is not None else None
            return out
        for v in obj.values():
            r = find_goals(v)
            if r:
                return r
    return None


def choose(a):
    _, pred, stats, _, _ = predict_all(a)
    n = len(pred)
    order = sorted(range(n), key=lambda i: (pred[i], i))
    seen = {}
    for i in order[:min(BUDGET, n)]:
        sys.stdout.write(json.dumps({"inspect": i}) + "\n")
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            break
        try:
            resp = json.loads(line)
        except ValueError:
            continue
        gv = find_goals(resp)
        if gv is None:
            continue
        row = resp.get("row", i) if isinstance(resp, dict) else i
        if not isinstance(row, int):
            row = i
        seen[row] = d2h(gv, stats)
    best = sorted(seen, key=lambda i: (seen[i], i))[:PICKS]
    sys.stdout.write(json.dumps({"selected": best}) + "\n")
    sys.stdout.flush()


def fix(a):
    th, tr = load(a.train)
    eh, te = load(a.test)
    stats = goal_stats(th, tr)
    gc = {g: column(th, tr, g) for g in GOALS}
    ok = [i for i in range(len(tr)) if all(gc[g][i] is not None for g in GOALS)]
    qual = np.array([d2h({g: gc[g][i] for g in GOALS}, stats) for i in ok])
    cutoff = float(np.median(qual)) if len(qual) else 0.5

    feats = [h for h in th if h and not h.endswith(("X", "+", "-"))]
    numc = [h for h in feats if h[0].isupper()]
    symc = [h for h in feats if not h[0].isupper()]
    tn = {h: column(th, tr, h) for h in numc}
    ts = {h: sym(th, tr, h) for h in symc}
    med, rng = {}, {}
    for h in numc:
        v = [x for x in tn[h] if x is not None]
        med[h] = float(np.median(v)) if v else 0.0
        rng[h] = (max(v) - min(v)) if v and max(v) > min(v) else 1.0
    tnum = {h: np.array([med[h] if x is None else x for x in tn[h]], float)[ok] / rng[h] for h in numc}
    tsym = {h: np.array(ts[h], dtype=object)[ok] for h in symc}
    ids = np.arange(len(ok))

    def score(car):
        d2 = np.zeros(len(ok))
        for h in numc:
            d2 += (tnum[h] - car[h] / rng[h]) ** 2
        for h in symc:
            d2 += (tsym[h] != car[h]).astype(float)
        d = np.round(np.sqrt(d2), 12)
        nb = np.lexsort((ids, d))[:K_NN]
        return float(qual[nb].mean()) if len(nb) else 0.5

    order = sorted(range(len(ok)), key=lambda i: (qual[i], i))
    best = order[:max(1, int(len(order) * 0.25))] if order else []
    cc, cv = (column(th, tr, e) for e in EDIT)
    cfgs, seen = [], set()
    for i in best:
        r = ok[i]
        if cc[r] is None or cv[r] is None:
            continue
        key = (cc[r], cv[r])
        if key not in seen:
            seen.add(key)
            cfgs.append(key)
        if len(cfgs) >= N_CONFIG:
            break

    en = {h: column(eh, te, h) for h in numc}
    es = {h: sym(eh, te, h) for h in symc}
    repairs = []
    for r in range(len(te)):
        car = {}
        for h in numc:
            x = en[h][r]
            car[h] = med[h] if x is None else x
        for h in symc:
            car[h] = es[h][r]
        before = dict(car)
        sb = score(car)
        rep = {"row": r, "before": before, "after": dict(before), "changes": 0}
        if sb <= cutoff:
            rep["status"] = "already_good"
            repairs.append(rep)
            continue
        cands = []
        for c, v in cfgs:
            ch = int(car["Clndrs"] != c) + int(car["Volume"] != v)
            if ch == 0:
                continue
            new = dict(car)
            new["Clndrs"], new["Volume"] = c, v
            sa = score(new)
            if sb - sa < MIN_GAIN:
                continue
            dist = math.sqrt(((car["Clndrs"] - c) / rng["Clndrs"]) ** 2 +
                             ((car["Volume"] - v) / rng["Volume"]) ** 2)
            cands.append((ch, dist, sa, new))
        if not cands:
            rep["status"] = "no_candidate"
        else:
            ch, dist, sa, new = min(cands, key=lambda t: (t[0], t[1], t[2]))
            rep["after"] = new
            rep["changes"] = ch
            rep["status"] = "improved"
            rep["improvement"] = sb - sa
            rep["distance"] = dist
        rep["before_score"] = sb
        repairs.append(rep)
    print(json.dumps({"repairs": repairs}))


def spot(a):
    th, tr = load(a.train)
    eh, te = load(a.test)
    # inputs only: ignore X columns, goal columns (+/-)
    cols = [h for h in SPOT_COLS if h in th and not h.endswith(("X", "+", "-"))]
    numc = [h for h in cols if h[0].isupper()]
    symc = [h for h in cols if not h[0].isupper()]

    ref = {}
    for h in numc:
        v = np.array([x for x in column(th, tr, h) if x is not None], float)
        if len(v) == 0:
            ref[h] = (0.0, 1.0)
            continue
        m = float(np.median(v))
        sd = float(v.std()) if len(v) > 1 else 1.0
        ref[h] = (m, max(sd, 1e-6))
    known = {h: set(sym(th, tr, h)) - {"?", ""} for h in symc}

    # volume conditional on cylinders
    groups = {}
    if "Clndrs" in th and "Volume" in th:
        d = {}
        for c, v in zip(column(th, tr, "Clndrs"), column(th, tr, "Volume")):
            if c is not None and v is not None:
                d.setdefault(c, []).append(v)
        allv = [v for vs in d.values() for v in vs]
        gscale = max(float(np.std(allv)) * 0.1, 1e-6) if allv else 1.0
        for c, vs in d.items():
            m = float(np.median(vs))
            mad = 1.4826 * float(np.median(np.abs(np.array(vs) - m)))
            groups[c] = (m, max(mad, 0.1 * abs(m), gscale)) if len(vs) >= 3 else None

    def raw_scores(head, rows):
        nc = {h: column(head, rows, h) for h in numc}
        sc = {h: sym(head, rows, h) for h in symc}
        out = []
        for i in range(len(rows)):
            s = 0.0
            for h in numc:
                x = nc[h][i]
                if x is not None:
                    m, sd = ref[h]
                    s = max(s, abs(x - m) / sd)
            for h in symc:
                v = sc[h][i]
                if v not in ("?", "") and known[h] and v not in known[h]:
                    s = max(s, 10.0)
            if groups and "Clndrs" in nc and "Volume" in nc:
                c, v = nc["Clndrs"][i], nc["Volume"][i]
                if c is not None and v is not None:
                    key = c if c in groups else None
                    if key is None and groups:
                        key = min(groups, key=lambda k: abs(k - c))
                    g = groups.get(key)
                    if g:
                        s = max(s, abs(v - g[0]) / g[1])
            out.append(s)
        return np.array(out, float)

    tr_raw = raw_scores(th, tr)
    thr_raw = SPOT_FLOOR
    if len(tr_raw):
        thr_raw = max(SPOT_FLOOR, float(np.quantile(tr_raw, SPOT_Q)))
    raw = raw_scores(eh, te)
    sq = lambda r: r / (1.0 + r)
    scores = [float(sq(r)) for r in raw]
    thr = float(sq(thr_raw))
    alerts = [bool(s > thr) for s in scores]
    print(json.dumps({"scores": scores, "alerts": alerts, "threshold": thr}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["remember", "guess", "certify", "flow", "blame",
                                    "justify", "choose", "fix", "spot"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    {"remember": remember, "guess": guess, "certify": certify,
     "flow": flow, "blame": blame, "justify": justify, "choose": choose,
     "fix": fix, "spot": spot}[a.cmd](a)


main()
