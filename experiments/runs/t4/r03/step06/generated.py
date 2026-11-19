import sys, csv, json, math
from bisect import bisect_left, bisect_right
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

GOALS = [('Lbs-', 0), ('Acc+', 1), ('Mpg+', 1)]

def read(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        hdr = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return hdr, rows

def fit_stats(vals):
    n = len(vals)
    st = []
    for k in range(3):
        v = [r[k] for r in vals]
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        st.append((m, sd))
    return st

def nrm(v, m, sd):
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))

def d2h(vals, stats):
    s = 0.0
    for k, v in enumerate(vals):
        m, sd = stats[k]
        s += (nrm(v, m, sd) - GOALS[k][1]) ** 2
    return math.sqrt(s / 3)

def load_train(path):
    h, rows = read(path)
    ci = [h.index(g) for g, _ in GOALS]
    rows = [r for r in rows if all(r[i] != '?' for i in ci)]
    vals = [[float(r[i]) for i in ci] for r in rows]
    return h, rows, vals

def num(x):
    try:
        return float(x)
    except ValueError:
        return float('nan')

def remember(a):
    train = a[a.index('--train') + 1]
    test = a[a.index('--test') + 1]
    _, _, tr = load_train(train)
    stats = fit_stats(tr)
    h2, te = read(test)
    ci2 = [h2.index(g) for g, _ in GOALS]
    scores = [d2h([float(r[i]) for i in ci2], stats) for r in te]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    out = {"scores": scores, "top": order[:10], "best": order[0],
           "median": order[len(scores) // 2], "worst": order[-1]}
    sys.stdout.write(json.dumps(out))

def predict(train, test):
    h, rows, vals = load_train(train)
    stats = fit_stats(vals)
    mpg = np.array([v[2] for v in vals])
    sc = np.array([d2h(v, stats) for v in vals])
    h2, te = read(test)
    cats = sorted({r[h.index('origin')] for r in rows} - {'?'})
    nf = ['Clndrs', 'Volume', 'Model']

    def feats(hdr, rs):
        X = np.array([[num(r[hdr.index(c)]) for c in nf] for r in rs], dtype=float).reshape(len(rs), 3)
        O = np.array([[1.0 if r[hdr.index('origin')] == c else 0.0 for c in cats]
                      for r in rs]).reshape(len(rs), len(cats))
        return X, O

    Xtr, Otr = feats(h, rows)
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def prep(X, O, use):
        X = np.where(np.isnan(X), med, X)
        return np.hstack([X[:, use], O])

    Xte, Ote = feats(h2, te)
    vmiss = np.array([r[h2.index('Volume')] == '?' for r in te], dtype=bool)
    mp = np.zeros(len(te))
    ds = np.zeros(len(te))
    for use, mask in (([0, 1, 2], ~vmiss), ([0, 2], vmiss)):
        if not mask.any():
            continue
        A = prep(Xtr, Otr, use)
        B = prep(Xte[mask], Ote[mask], use)
        kw = dict(n_estimators=200, learning_rate=0.05, max_depth=3,
                  subsample=0.8, random_state=0)
        m1 = GradientBoostingRegressor(**kw).fit(A, np.log(np.maximum(mpg, 1e-6)))
        m2 = GradientBoostingRegressor(**kw).fit(A, sc)
        mp[mask] = np.exp(m1.predict(B))
        ds[mask] = m2.predict(B)
    mp = np.where(np.isfinite(mp) & (mp > 0), mp, float(mpg.mean()))
    ds = np.clip(np.where(np.isfinite(ds), ds, float(sc.mean())), 0.0, 1.0)
    return mp, ds, sc

def guess(a):
    train = a[a.index('--train') + 1]
    test = a[a.index('--test') + 1]
    mp, ds, _ = predict(train, test)
    sys.stdout.write(json.dumps({"mpg": [float(x) for x in mp],
                                 "scores": [float(x) for x in ds]}))

def compare(xs, ys):
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    vy = sum((y - my) ** 2 for y in ys) / (m - 1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (math.sqrt(pv) + 1e-32)
    sx, sy = sorted(xs), sorted(ys)
    tot = 0
    for x in xs:
        tot += bisect_left(sy, x) - (m - bisect_right(sy, x))
    cliffs = abs(tot) / (n * m)
    d = 0.0
    for v in sorted(set(sx) | set(sy)):
        d = max(d, abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m))
    ks = d / math.sqrt(1 / n + 1 / m)
    same = bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks, "same": same}

def certify(a):
    test = a[a.index('--test') + 1]
    with open(test) as f:
        cases = json.load(f)
    out = {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}
    sys.stdout.write(json.dumps(out))

def flow(a):
    train = a[a.index('--train') + 1]
    test = a[a.index('--test') + 1]
    mp, ds, sc = predict(train, test)
    hist = [float(x) for x in sc]
    scores = [float(x) for x in ds]
    alerts = []
    for i in range(len(scores)):
        if i < 19:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, scores[i - 19:i + 1])["same"])
    sys.stdout.write(json.dumps({"mpg": [float(x) for x in mp],
                                 "scores": scores, "alerts": alerts}))

def diagnose(train, test):
    h, rows, vals = load_train(train)
    stats = fit_stats(vals)
    h2, te = read(test)
    d = [d2h(v, stats) for v in vals]
    order = sorted(range(len(rows)), key=lambda i: (round(d[i], 12), i))
    q = max(1, len(rows) // 4)
    good = [rows[i] for i in order[:q]]
    poor = [rows[i] for i in order[-q:]]

    nfe = ['Clndrs', 'Volume', 'Model']
    nst = {}
    cen = {}
    for c in nfe:
        j = h.index(c)
        col = [float(r[j]) for r in rows if r[j] != '?']
        n = len(col)
        m = sum(col) / n if n else 0.0
        sd = math.sqrt(sum((x - m) ** 2 for x in col) / (n - 1)) if n > 1 else 0.0
        nst[c] = (m, sd)
        cs = []
        for g in (good, poor):
            gv = [float(r[j]) for r in g if r[j] != '?']
            cs.append(nrm(sum(gv) / len(gv), m, sd) if gv else None)
        cen[c] = cs
    oj = h.index('origin')
    gk = [r[oj] for r in good if r[oj] != '?']
    pk = [r[oj] for r in poor if r[oj] != '?']

    gi = [h2.index(g) for g, _ in GOALS]
    ti = {c: h2.index(c) for c in nfe}
    toj = h2.index('origin')
    diags = []
    for r in te:
        goals = {}
        for k, (g, hv) in enumerate(GOALS):
            m, sd = stats[k]
            goals[g] = abs(nrm(float(r[gi[k]]), m, sd) - hv)
        feats = {}
        for c in nfe:
            x = r[ti[c]]
            gc, pc = cen[c]
            if x == '?' or gc is None or pc is None:
                feats[c] = 0.0
            else:
                nx = nrm(float(x), *nst[c])
                feats[c] = max(0.0, abs(nx - gc) - abs(nx - pc))
        x = r[toj]
        if x == '?':
            feats['origin'] = 0.0
        else:
            pf = pk.count(x) / len(pk) if pk else 0.0
            gf = gk.count(x) / len(gk) if gk else 0.0
            feats['origin'] = max(0.0, pf - gf)
        diags.append({
            "goals": goals, "features": feats,
            "goal_order": sorted(goals, key=lambda n: (-round(goals[n], 10), n)),
            "feature_order": sorted(feats, key=lambda n: (-round(feats[n], 10), n)),
        })
    return diags, d

def blame(a):
    train = a[a.index('--train') + 1]
    test = a[a.index('--test') + 1]
    diags, _ = diagnose(train, test)
    sys.stdout.write(json.dumps({"diagnoses": diags}))

def justify(a):
    train = a[a.index('--train') + 1]
    test = a[a.index('--test') + 1]
    diags, d = diagnose(train, test)
    cutoff = sorted(d)[len(d) // 2]
    # d2h from goal gaps: sqrt(mean(gap^2)) equals d2h by definition
    scores = [math.sqrt(sum(g * g for g in x["goals"].values()) / 3) for x in diags]
    n = len(scores)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = scores[j], scores[k]
        acc = bool(round(s, 12) <= round(cutoff, 12))
        if round(s, 12) == round(o, 12):
            w = None
        else:
            w = j if s < o else k
        adv = None
        if w is not None:
            l = k if w == j else j
            av = {g: diags[l]["goals"][g] ** 2 - diags[w]["goals"][g] ** 2
                  for g in diags[j]["goals"]}
            adv = sorted(av, key=lambda g: (-round(av[g], 10), g))[0]
        dj = diags[j]
        weak = dj["goal_order"][0]
        tf = dj["feature_order"][0]
        if not dj["features"][tf] > 1e-10:
            tf = None
        text = "Car %d %s (score %.3f, other score %.3f, cutoff %.3f); weakest goal: %s." % (
            j, "accepted" if acc else "rejected", s, o, cutoff, weak)
        if w is not None:
            text += " Recommend car %d; largest advantage: %s." % (w, adv)
        else:
            text += " Result is a tie."
        if tf is not None:
            text += " Top feature: %s." % tf
        out.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": acc, "winner": w,
                    "advantage_goal": adv, "weakest_goal": weak,
                    "top_feature": tf, "text": text})
    sys.stdout.write(json.dumps({"explanations": out}))

def main():
    a = sys.argv
    cmds = {'remember': remember, 'guess': guess, 'certify': certify,
            'flow': flow, 'blame': blame, 'justify': justify}
    if len(a) < 2 or a[1] not in cmds:
        sys.exit(1)
    cmds[a[1]](a)

main()
