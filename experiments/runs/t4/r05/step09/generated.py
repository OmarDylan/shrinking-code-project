import sys, csv, json, math
from bisect import bisect_right
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest

def load(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def goals(header, rows):
    idx = [header.index(n) for n in ("Lbs-", "Acc+", "Mpg+")]
    return [[float(row[i]) for i in idx] for row in rows]

def stats(G):
    n = len(G)
    means = [sum(g[j] for g in G) / n for j in range(3)]
    sds = [math.sqrt(sum((g[j] - means[j]) ** 2 for g in G) / (n - 1)) if n > 1 else 0.0
           for j in range(3)]
    return means, sds

def d2h(g, means, sds):
    heaven = [0.0, 1.0, 1.0]
    s = 0.0
    for j in range(3):
        z = (g[j] - means[j]) / (sds[j] + 1e-32)
        z = max(-3.0, min(3.0, z))
        v = 1 / (1 + math.exp(-1.702 * z))
        s += (v - heaven[j]) ** 2
    return math.sqrt(s / 3)

def arg(name):
    a = sys.argv
    return a[a.index(name) + 1]

def remember(train, test):
    th, tr = load(train)
    eh, er = load(test)
    G = goals(th, tr)
    E = goals(eh, er)
    means, sds = stats(G)
    scores = [d2h(g, means, sds) for g in E]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else float("nan")
    except ValueError:
        return float("nan")

def feats(header, rows, cats):
    ci, vi, mi, oi = (header.index(n) for n in ("Clndrs", "Volume", "Model", "origin"))
    X = []
    for r in rows:
        X.append([num(r[ci]), num(r[vi]), num(r[mi])] + [1.0 if r[oi] == c else 0.0 for c in cats])
    return np.array(X, dtype=float)

def predict(Xtr, ytr, Xte):
    ps = []
    for m in (RandomForestRegressor(300, min_samples_leaf=2, random_state=0, n_jobs=1),
              GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, random_state=0)):
        ps.append(m.fit(Xtr, ytr).predict(Xte))
    return np.mean(ps, axis=0)

def train_goals(th, tr):
    keep = []
    for r in tr:
        try:
            g = [float(r[th.index(n)]) for n in ("Lbs-", "Acc+", "Mpg+")]
            if all(math.isfinite(x) for x in g):
                keep.append((r, g))
        except ValueError:
            pass
    return [k[0] for k in keep], [k[1] for k in keep]

def guess(train, test):
    th, tr = load(train)
    eh, er = load(test)
    tr, G = train_goals(th, tr)
    means, sds = stats(G)
    y_mpg = np.array([g[2] for g in G])
    y_d = np.array([d2h(g, means, sds) for g in G])
    cats = sorted({r[th.index("origin")] for r in tr})
    Xtr = feats(th, tr, cats)
    Xte = feats(eh, er, cats)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    fill = lambda X: np.where(np.isnan(X), med, X)
    Xtr = fill(Xtr)
    n = len(Xte)
    mpg = np.zeros(n)
    sc = np.zeros(n)
    hasv = ~np.isnan(Xte[:, 1])
    cols_nov = [0, 2] + list(range(3, Xtr.shape[1]))
    for mask, cols in ((hasv, list(range(Xtr.shape[1]))), (~hasv, cols_nov)):
        if not mask.any():
            continue
        A = Xtr[:, cols]
        B = fill(Xte)[mask][:, cols]
        mpg[mask] = predict(A, y_mpg, B)
        sc[mask] = predict(A, y_d, B)
    mpg = np.where(np.isfinite(mpg) & (mpg > 0), mpg, float(y_mpg.mean()))
    mpg = np.maximum(mpg, 1e-3)
    sc = np.clip(np.nan_to_num(sc, nan=float(y_d.mean())), 0.0, 1.0)
    return {"mpg": [float(x) for x in mpg], "scores": [float(x) for x in sc]}

def svar(v):
    m = sum(v) / len(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1)

def compare(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n, m = len(xs), len(ys)
    pooled = ((n - 1) * svar(xs) + (m - 1) * svar(ys)) / (n + m - 2)
    cohen = abs(sum(xs) / n - sum(ys) / m) / (math.sqrt(pooled) + 1e-32)
    cl = sum((x > y) - (x < y) for x in xs for y in ys)
    cliffs = abs(cl) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    d = max(abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m) for v in set(sx) | set(sy))
    ks = d / math.sqrt(1 / n + 1 / m)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks,
            "same": bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)}

def certify(path):
    with open(path) as f:
        cases = json.load(f)
    return {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}

def flow(train, test):
    out = guess(train, test)
    th, tr = load(train)
    _, G = train_goals(th, tr)
    means, sds = stats(G)
    hist = [d2h(g, means, sds) for g in G]
    sc = out["scores"]
    alerts = []
    for i in range(len(sc)):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, sc[i - 19:i + 1])["same"])
    out["alerts"] = alerts
    return out

def zlogit(v, mean, sd):
    z = (v - mean) / (sd + 1e-32)
    z = max(-3.0, min(3.0, z))
    return 1 / (1 + math.exp(-1.702 * z))

def msd(v):
    if not v:
        return 0.0, 0.0
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m, s

def blame(train, test):
    th, tr = load(train)
    eh, er = load(test)
    tr, G = train_goals(th, tr)
    means, sds = stats(G)
    gnames = ("Lbs-", "Acc+", "Mpg+")
    heaven = [0.0, 1.0, 1.0]
    d = [round(d2h(g, means, sds), 12) for g in G]
    order = sorted(range(len(tr)), key=lambda i: (d[i], i))
    q = max(1, len(tr) // 4)
    good = [tr[i] for i in order[:q]]
    poor = [tr[i] for i in order[-q:]]
    numf = ("Clndrs", "Volume", "Model")
    prof = {}
    for f in numf:
        j = th.index(f)
        col = [num(r[j]) for r in tr]
        col = [x for x in col if not math.isnan(x)]
        m, s = msd(col)
        cen = []
        for grp in (good, poor):
            vals = [num(r[j]) for r in grp]
            vals = [x for x in vals if not math.isnan(x)]
            cen.append(zlogit(sum(vals) / len(vals), m, s) if vals else None)
        prof[f] = (m, s, cen[0], cen[1])
    oj = th.index("origin")
    def freq(grp):
        vals = [r[oj] for r in grp if r[oj] != "?" and r[oj] != ""]
        c = {}
        for v in vals:
            c[v] = c.get(v, 0) + 1
        return c, len(vals)
    gf, gn = freq(good)
    pf, pn = freq(poor)
    ehj = {f: eh.index(f) for f in numf}
    eoj = eh.index("origin")
    egi = [eh.index(n) if n in eh else None for n in gnames]
    diags = []
    for r in er:
        feat = {}
        for f in numf:
            x = num(r[ehj[f]])
            m, s, gc, pc = prof[f]
            if math.isnan(x) or gc is None or pc is None:
                feat[f] = 0.0
            else:
                nx = zlogit(x, m, s)
                feat[f] = max(0.0, abs(nx - gc) - abs(nx - pc))
        c = r[eoj]
        if c in ("?", ""):
            feat["origin"] = 0.0
        else:
            a = pf.get(c, 0) / pn if pn else 0.0
            b = gf.get(c, 0) / gn if gn else 0.0
            feat["origin"] = max(0.0, a - b)
        gl = {}
        for j, n in enumerate(gnames):
            v = num(r[egi[j]]) if egi[j] is not None else float("nan")
            gl[n] = 0.0 if math.isnan(v) else abs(zlogit(v, means[j], sds[j]) - heaven[j])
        key = lambda dct: sorted(dct, key=lambda k: (-round(dct[k], 10), k))
        diags.append({"goals": gl, "features": feat,
                      "goal_order": key(gl), "feature_order": key(feat)})
    return {"diagnoses": diags}

def justify(train, test):
    th, tr = load(train)
    eh, er = load(test)
    tr, G = train_goals(th, tr)
    means, sds = stats(G)
    tsc = sorted(d2h(g, means, sds) for g in G)
    cutoff = tsc[len(tsc) // 2]
    E = goals(eh, er)
    scores = [d2h(g, means, sds) for g in E]
    diags = blame(train, test)["diagnoses"]
    n = len(er)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = scores[j], scores[k]
        acc = bool(round(s, 12) <= round(cutoff, 12))
        winner = None
        adv_goal = None
        if round(s, 12) != round(o, 12):
            winner = j if s < o else k
            loser = k if winner == j else j
            adv = {gn: diags[loser]["goals"][gn] ** 2 - diags[winner]["goals"][gn] ** 2
                   for gn in diags[j]["goals"]}
            adv_goal = sorted(adv, key=lambda g: (-round(adv[g], 10), g))[0]
        d = diags[j]
        weakest = d["goal_order"][0]
        tf = d["feature_order"][0]
        top = tf if d["features"][tf] > 1e-10 else None
        text = (f"Car {j} {'accepted' if acc else 'rejected'}: score {s:.3f}, "
                f"other score {o:.3f}, cutoff {cutoff:.3f}. Weakest goal: {weakest}. ")
        if winner is not None:
            text += f"Recommend car {winner} because of {adv_goal}. "
        else:
            text += "The comparison is a tie. "
        if top is not None:
            text += f"Top feature: {top}."
        out.append({"row": j, "other": k, "score": s, "other_score": o, "cutoff": cutoff,
                    "accepted": acc, "winner": winner, "advantage_goal": adv_goal,
                    "weakest_goal": weakest, "top_feature": top, "text": text.strip()})
    return {"explanations": out}

def choose(train, test, budget=10):
    th, tr = load(train)
    eh, er = load(test)
    tr, G = train_goals(th, tr)
    means, sds = stats(G)
    y_d = np.array([d2h(g, means, sds) for g in G])
    cats = sorted({r[th.index("origin")] for r in tr})
    Xtr = feats(th, tr, cats)
    Xte = feats(eh, er, cats)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    fill = lambda X: np.where(np.isnan(X), med, X)
    pred = predict(fill(Xtr), y_d, fill(Xte))
    pred = np.nan_to_num(pred, nan=float(y_d.mean()))
    order = sorted(range(len(er)), key=lambda i: (pred[i], i))
    seen = {}
    for i in order[:min(budget, len(er))]:
        sys.stdout.write(json.dumps({"inspect": int(i)}) + "\n")
        sys.stdout.flush()
        resp = json.loads(sys.stdin.readline())
        g = resp["goals"]
        seen[int(i)] = d2h([float(g["Lbs-"]), float(g["Acc+"]), float(g["Mpg+"])], means, sds)
    rank = sorted(seen, key=lambda i: (round(seen[i], 12), i))
    sys.stdout.write(json.dumps({"selected": rank[:3]}) + "\n")
    sys.stdout.flush()

def fix(train, test):
    th, tr = load(train)
    eh, er = load(test)
    tr, G = train_goals(th, tr)
    means, sds = stats(G)
    y = np.array([d2h(g, means, sds) for g in G])
    n = len(tr)
    numf = ("Clndrs", "Volume", "Model")
    par = []
    for f in numf:
        j = th.index(f)
        col = [num(r[j]) for r in tr]
        par.append(msd([x for x in col if not math.isnan(x)]))
    def normrow(r, h):
        out = []
        for k, f in enumerate(numf):
            v = num(r[h.index(f)])
            out.append(float("nan") if math.isnan(v) else zlogit(v, par[k][0], par[k][1]))
        return out
    T = np.array([normrow(r, th) for r in tr], dtype=float)
    To = np.array([r[th.index("origin")] for r in tr])
    idx = np.arange(n)
    def pred(nv, o):
        gap = T - np.array(nv)
        gap = np.where(np.isnan(gap), 0.0, gap)
        sq = (gap ** 2).sum(axis=1) + (To != o).astype(float)
        dist = np.sqrt(sq / 4)
        order = np.lexsort((idx, np.round(dist, 12)))
        return float(y[order[:5]].mean())
    cutoff = sorted(y.tolist())[n // 2]
    rk = sorted(range(n), key=lambda i: (round(float(y[i]), 12), i))
    good = rk[:max(1, n // 4)]
    ci, vi = th.index("Clndrs"), th.index("Volume")
    menu = []
    for i in good:
        p = (num(tr[i][ci]), num(tr[i][vi]))
        if math.isnan(p[0]) or math.isnan(p[1]) or p in menu:
            continue
        menu.append(p)
        if len(menu) == 12:
            break
    ec, ev = eh.index("Clndrs"), eh.index("Volume")
    eo = eh.index("origin")
    reps = []
    for i, r in enumerate(er):
        o = r[eo]
        nv = normrow(r, eh)
        before = pred(nv, o)
        res = {"row": i, "before": before, "after": before, "changes": {}, "status": "already_good"}
        if round(before, 10) > round(cutoff, 10):
            oc, ov = num(r[ec]), num(r[ev])
            best = None
            for pc, pv in menu:
                chg = {}
                if pc != oc:
                    chg["Clndrs"] = pc
                if pv != ov:
                    chg["Volume"] = pv
                if not chg:
                    continue
                nn = list(nv)
                if "Clndrs" in chg:
                    nn[0] = zlogit(pc, par[0][0], par[0][1])
                if "Volume" in chg:
                    nn[1] = zlogit(pv, par[1][0], par[1][1])
                after = pred(nn, o)
                if round(before - after, 10) < 0.05:
                    continue
                dd = 0.0
                for k in (0, 1):
                    a, b = nv[k], nn[k]
                    if not (math.isnan(a) or math.isnan(b)):
                        dd += (a - b) ** 2
                dist = math.sqrt(dd / 4)
                key = (len(chg), round(dist, 10), round(after, 10), (pc, pv))
                if best is None or key < best[0]:
                    best = (key, chg, after)
            if best is None:
                res["status"] = "no_candidate"
            else:
                res.update(after=best[2], changes=best[1], status="improved")
        reps.append(res)
    return {"repairs": reps}

def spot(train, test):
    th, tr = load(train)
    eh, er = load(test)
    cats = sorted({r[th.index("origin")] for r in tr if r[th.index("origin")] not in ("?", "")})
    Xtr = feats(th, tr, cats)
    Xte = feats(eh, er, cats)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    fill = lambda X: np.where(np.isnan(X), med, X)
    Xtr, Xte = fill(Xtr), fill(Xte)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    cov = np.cov(Ztr, rowvar=False).reshape(Ztr.shape[1], -1) + 0.05 * np.eye(Ztr.shape[1])
    P = np.linalg.pinv(cov)
    maha = lambda Z: np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z, P, Z), 0.0))
    iso = IsolationForest(n_estimators=300, random_state=0).fit(Ztr)
    detectors = [
        (lambda Z: -iso.score_samples(Z)),
        maha,
        lambda Z: np.abs(Z[:, :3]).max(axis=1),
    ]
    tr_s = [np.sort(f(Ztr)) for f in detectors]
    def combine(Z):
        parts = []
        for f, s in zip(detectors, tr_s):
            v = f(Z)
            parts.append(np.searchsorted(s, v, side="right") / len(s))
        return np.mean(parts, axis=0)
    str_ = combine(Ztr)
    ste = combine(Zte)
    thr = float(np.quantile(str_, 0.95))
    sc = np.clip(np.nan_to_num(ste, nan=0.0), 0.0, 1.0)
    thr = min(max(thr, 0.0), 1.0)
    return {"scores": [float(x) for x in sc], "alerts": [bool(x > thr) for x in sc],
            "threshold": thr}

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "remember"
    if cmd == "certify":
        out = certify(arg("--test"))
    else:
        train, test = arg("--train"), arg("--test")
        if cmd == "choose":
            choose(train, test)
            return
        if cmd == "guess":
            out = guess(train, test)
        elif cmd == "flow":
            out = flow(train, test)
        elif cmd == "blame":
            out = blame(train, test)
        elif cmd == "justify":
            out = justify(train, test)
        elif cmd == "fix":
            out = fix(train, test)
        elif cmd == "spot":
            out = spot(train, test)
        else:
            out = remember(train, test)
    sys.stdout.write(json.dumps(out))

if __name__ == "__main__":
    main()
