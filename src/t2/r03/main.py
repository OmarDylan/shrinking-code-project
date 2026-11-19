import sys, csv, math, json, argparse
from bisect import bisect_left, bisect_right

GOALS = {"Lbs": 0.0, "Acc": 1.0, "Mpg": 1.0}

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip().rstrip("+-") for h in next(r)]
        return [dict(zip(header, (c.strip() for c in row))) for row in r if row]

def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def goals(row):
    g = [num(row.get(k)) for k in GOALS]
    return None if None in g else g

def make_norm(col):
    n = len(col)
    m = sum(col) / n if n else 0.0
    sd = math.sqrt(sum((v - m) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0

    def norm(v):
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))
    return norm

def make_score(train_goals):
    hs = list(GOALS.values())
    norms = [make_norm([r[j] for r in train_goals]) for j in range(3)]

    def score(r):
        s = 0.0
        for j in range(3):
            s += (norms[j](r[j]) - hs[j]) ** 2
        return math.sqrt(s / 3)
    return score

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(M[i][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            M[c][c] = 1e-12
        for i in range(n):
            if i != c:
                f = M[i][c] / M[c][c]
                for k in range(c, n + 1):
                    M[i][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]

def ridge(X, y, lam=1.0):
    d = len(X[0]) + 1
    X = [[1.0] + x for x in X]
    A = [[sum(x[i] * x[j] for x in X) + (lam if i == j and i > 0 else 0.0)
          for j in range(d)] for i in range(d)]
    b = [sum(x[i] * t for x, t in zip(X, y)) for i in range(d)]
    return solve(A, b)

def remember(train, test):
    score = make_score([g for g in map(goals, train) if g])
    scores = []
    for r in test:
        g = goals(r)
        scores.append(score(g) if g else float("inf"))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[len(order) // 2], "worst": order[-1]}

NUM = ["Clndrs", "Volume", "Model"]

def guess(train, test):
    rows = [(r, goals(r)) for r in train]
    rows = [(r, g) for r, g in rows if g]
    score = make_score([g for _, g in rows])
    ms, sds = {}, {}
    for k in NUM:
        c = [v for v in (num(r.get(k)) for r, _ in rows) if v is not None]
        m = sum(c) / len(c) if c else 0.0
        sd = math.sqrt(sum((v - m) ** 2 for v in c) / max(len(c) - 1, 1)) if c else 1.0
        ms[k], sds[k] = m, sd or 1.0
    origins = sorted({r.get("origin", "?") for r, _ in rows} - {"?", ""})

    def feat(r, vol):
        x = []
        for k in NUM:
            if k == "Volume" and not vol:
                continue
            v = num(r.get(k))
            x.append(((ms[k] if v is None else v) - ms[k]) / sds[k])
        x += [1.0 if r.get("origin") == o else 0.0 for o in origins]
        return x

    ys_mpg = [math.log(max(g[2], 1e-3)) for _, g in rows]
    ys_d = [score(g) for _, g in rows]
    models = {}
    for vol in (True, False):
        X = [feat(r, vol) for r, _ in rows]
        models[vol] = (ridge(X, ys_mpg), ridge(X, ys_d))

    def pred(w, x):
        return w[0] + sum(a * b for a, b in zip(w[1:], x))

    mpg_mean = sum(g[2] for _, g in rows) / len(rows)
    d_mean = sum(ys_d) / len(ys_d)
    mpg, sc = [], []
    for i, r in enumerate(test):
        vol = num(r.get("Volume")) is not None and i % 2 == 1
        x = feat(r, vol)
        wm, wd = models[vol]
        try:
            m = math.exp(max(-5.0, min(6.0, pred(wm, x))))
            d = pred(wd, x)
        except OverflowError:
            m, d = mpg_mean, d_mean
        if not (math.isfinite(m) and m > 0):
            m = mpg_mean
        if not math.isfinite(d):
            d = d_mean
        mpg.append(m)
        sc.append(max(0.0, min(1.0, d)))
    return {"mpg": mpg, "scores": sc}

def compare(xs, ys):
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((v - mx) ** 2 for v in xs) / (n - 1)
    vy = sum((v - my) ** 2 for v in ys) / (m - 1)
    pooled = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (math.sqrt(pooled) + 1e-32)
    sx, sy = sorted(xs), sorted(ys)
    tot = 0
    for x in xs:
        tot += bisect_left(sy, x) - (m - bisect_right(sy, x))
    cliffs = abs(tot) / (n * m)
    d = 0.0
    for v in set(sx) | set(sy):
        d = max(d, abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m))
    ks = d / math.sqrt(1 / n + 1 / m)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks,
            "same": bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)}

def certify(cases):
    return {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}

def flow(train, test, W=20):
    g = guess(train, test)
    score = make_score([x for x in map(goals, train) if x])
    hist = [score(x) for x in map(goals, train) if x]
    sc = g["scores"]
    alerts = []
    for i in range(len(sc)):
        if i < W - 1:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, sc[i - W + 1:i + 1])["same"])
    return {"mpg": g["mpg"], "scores": sc, "alerts": alerts}

def blame(train, test):
    rows = [(r, goals(r)) for r in train]
    rows = [(r, g) for r, g in rows if g]
    score = make_score([g for _, g in rows])
    gnorms = [make_norm([g[j] for _, g in rows]) for j in range(3)]
    hs = list(GOALS.values())
    order = sorted(range(len(rows)), key=lambda i: (round(score(rows[i][1]), 12), i))
    q = max(1, len(rows) // 4)
    good = [rows[i][0] for i in order[:q]]
    poor = [rows[i][0] for i in order[-q:]]

    norms, centers = {}, {}
    for k in NUM:
        col = [v for v in (num(r.get(k)) for r, _ in rows) if v is not None]
        if not col:
            norms[k] = None
            continue
        nf = make_norm(col)
        norms[k] = nf
        cs = []
        for grp in (good, poor):
            vs = [v for v in (num(r.get(k)) for r in grp) if v is not None]
            cs.append(nf(sum(vs) / len(vs)) if vs else None)
        centers[k] = cs

    def freqs(grp):
        vs = [r.get("origin") for r in grp if r.get("origin") not in (None, "", "?")]
        f = {}
        for v in vs:
            f[v] = f.get(v, 0) + 1
        return {k: c / len(vs) for k, c in f.items()}
    gf, pf = freqs(good), freqs(poor)

    def order_names(d):
        return sorted(d, key=lambda k: (-round(d[k], 10), k))

    out = []
    for r in test:
        feats = {}
        for k in NUM:
            v = num(r.get(k))
            gc, pc = centers.get(k, (None, None))
            if v is None or norms[k] is None or gc is None or pc is None:
                feats[k] = 0.0
            else:
                x = norms[k](v)
                feats[k] = max(0.0, abs(x - gc) - abs(x - pc))
        o = r.get("origin")
        if o in (None, "", "?"):
            feats["origin"] = 0.0
        else:
            feats["origin"] = max(0.0, pf.get(o, 0.0) - gf.get(o, 0.0))
        gl = {}
        for j, name in enumerate(GOALS):
            v = num(r.get(name))
            gl[name] = abs(gnorms[j](v) - hs[j]) if v is not None else 0.0
        out.append({"goals": gl, "features": feats,
                    "goal_order": order_names(gl),
                    "feature_order": order_names(feats)})
    return {"diagnoses": out}

def justify(train, test):
    tg = [g for g in map(goals, train) if g]
    score = make_score(tg)
    ds = sorted(score(g) for g in tg)
    cutoff = ds[len(ds) // 2]
    scores = remember(train, test)["scores"]
    diag = blame(train, test)["diagnoses"]
    n = len(test)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = scores[j], scores[k]
        acc = round(s, 12) <= round(cutoff, 12)
        if round(s, 12) == round(o, 12):
            w = None
        else:
            w = j if s < o else k
        adv_goal = None
        if w is not None:
            l = k if w == j else j
            adv = {name: diag[l]["goals"][name] ** 2 - diag[w]["goals"][name] ** 2
                   for name in GOALS}
            adv_goal = sorted(adv, key=lambda m: (-round(adv[m], 10), m))[0]
        weakest = diag[j]["goal_order"][0]
        fo = diag[j]["feature_order"][0]
        top = fo if diag[j]["features"][fo] > 1e-10 else None
        text = "Car %d %s: score %.3f, other score %.3f, cutoff %.3f; weakest goal %s." % (
            j, "accepted" if acc else "rejected", s, o, cutoff, weakest)
        if w is not None:
            text += " Recommend car %d; largest advantage in %s." % (w, adv_goal)
        else:
            text += " The comparison is a tie."
        if top is not None:
            text += " Top feature: %s." % top
        text += " Retrospective, not causal."
        out.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": acc, "winner": w,
                    "advantage_goal": adv_goal, "weakest_goal": weakest,
                    "top_feature": top, "text": text})
    return {"explanations": out}

GKEYS = ("Lbs-", "Acc+", "Mpg+")

def make_predictor(train):
    rows = [(r, goals(r)) for r in train]
    rows = [(r, g) for r, g in rows if g]
    score = make_score([g for _, g in rows])
    ms, sds = {}, {}
    for k in NUM:
        c = [v for v in (num(r.get(k)) for r, _ in rows) if v is not None]
        m = sum(c) / len(c) if c else 0.0
        sd = math.sqrt(sum((v - m) ** 2 for v in c) / max(len(c) - 1, 1)) if c else 1.0
        ms[k], sds[k] = m, sd or 1.0
    origins = sorted({r.get("origin", "?") for r, _ in rows} - {"?", ""})

    def feat(r):
        x = []
        for k in NUM:
            v = num(r.get(k))
            x.append(((ms[k] if v is None else v) - ms[k]) / sds[k])
        return x + [1.0 if r.get("origin") == o else 0.0 for o in origins]

    w = ridge([feat(r) for r, _ in rows], [score(g) for _, g in rows])
    return score, (lambda r: w[0] + sum(a * b for a, b in zip(w[1:], feat(r))))

def choose(train, test, budget=10):
    score, predf = make_predictor(train)
    pred = [predf(r) for r in test]
    cand = sorted(range(len(test)), key=lambda i: (pred[i], i))[:budget]
    seen = {}
    for i in cand:
        sys.stdout.write(json.dumps({"inspect": i}) + "\n")
        sys.stdout.flush()
        resp = json.loads(sys.stdin.readline())
        seen[i] = score([float(resp["goals"][k]) for k in GKEYS])
    top = sorted(seen, key=lambda i: (round(seen[i], 12), i))[:3]
    return {"selected": top}

def fix(train, test, K=5):
    rows = [(r, goals(r)) for r in train]
    rows = [(r, g) for r, g in rows if g]
    score = make_score([g for _, g in rows])
    d2h = [score(g) for _, g in rows]
    norms = {}
    for k in NUM:
        col = [v for v in (num(r.get(k)) for r, _ in rows) if v is not None]
        norms[k] = make_norm(col or [0.0])

    def nv(r, k):
        v = num(r.get(k))
        return norms[k](v) if v is not None else None

    tr = [([nv(r, k) for k in NUM], r.get("origin")) for r, _ in rows]

    def dist(a, b):
        s = 0.0
        for x, y in zip(a[0], b[0]):
            if x is None or y is None:
                continue
            s += (x - y) ** 2
        if a[1] != b[1]:
            s += 1.0
        return math.sqrt(s / 4)

    def predict(q):
        ds = sorted(range(len(tr)), key=lambda i: (round(dist(q, tr[i]), 12), i))[:K]
        return sum(d2h[i] for i in ds) / len(ds)

    cutoff = sorted(d2h)[len(d2h) // 2]
    order = sorted(range(len(rows)), key=lambda i: (round(d2h[i], 12), i))
    ng = max(1, len(rows) // 4)
    menu, seen = [], set()
    for i in order[:ng]:
        c, v = num(rows[i][0].get("Clndrs")), num(rows[i][0].get("Volume"))
        if c is None or v is None or (c, v) in seen:
            continue
        seen.add((c, v))
        menu.append((c, v))
        if len(menu) == 12:
            break

    out = []
    for idx, r in enumerate(test):
        q = ([nv(r, k) for k in NUM], r.get("origin"))
        before = predict(q)
        if round(before, 10) <= round(cutoff, 10):
            out.append({"row": idx, "before": before, "after": before,
                        "changes": {}, "status": "already_good"})
            continue
        c0, v0 = num(r.get("Clndrs")), num(r.get("Volume"))
        best = None
        for c, v in menu:
            ch = {}
            if c != c0:
                ch["Clndrs"] = c
            if v != v0:
                ch["Volume"] = v
            if not ch:
                continue
            e = ([norms["Clndrs"](c), norms["Volume"](v), q[0][2]], q[1])
            after = predict(e)
            if round(before - after, 10) >= 0.05:
                key = (len(ch), round(dist(q, e), 10), round(after, 10), (c, v))
                if best is None or key < best[0]:
                    best = (key, after, ch)
        if best is None:
            out.append({"row": idx, "before": before, "after": before,
                        "changes": {}, "status": "no_candidate"})
        else:
            out.append({"row": idx, "before": before, "after": best[1],
                        "changes": best[2], "status": "improved"})
    return {"repairs": out}

def spot(train, test, K=5, Q=0.95):
    ms, sds = {}, {}
    for k in NUM:
        c = [v for v in (num(r.get(k)) for r in train) if v is not None]
        m = sum(c) / len(c) if c else 0.0
        sd = math.sqrt(sum((v - m) ** 2 for v in c) / max(len(c) - 1, 1)) if c else 1.0
        ms[k], sds[k] = m, sd or 1.0

    def vec(r):
        o = r.get("origin")
        return ([(num(r.get(k)) - ms[k]) / sds[k] if num(r.get(k)) is not None else None
                 for k in NUM], None if o in (None, "", "?") else o)

    def dist(a, b):
        s, n = 0.0, 0
        for x, y in zip(a[0], b[0]):
            if x is None or y is None:
                continue
            s += (x - y) ** 2
            n += 1
        if a[1] is not None and b[1] is not None:
            s += 1.0 if a[1] != b[1] else 0.0
            n += 1
        return math.sqrt(s / n) if n else 0.0

    tr = [vec(r) for r in train]

    def raw(q, skip=None):
        ds = sorted(dist(q, t) for i, t in enumerate(tr) if i != skip)[:K]
        return sum(ds) / len(ds) if ds else 0.0

    ref = sorted(raw(t, i) for i, t in enumerate(tr))
    n = max(len(ref), 1)
    scores = [bisect_right(ref, raw(vec(r))) / n for r in test]
    scores = [max(0.0, min(1.0, s)) for s in scores]
    return {"scores": scores, "alerts": [s > Q for s in scores], "threshold": Q}

def race(train, cases, n_insp=10):
    header = [h.strip().rstrip("+-") for h in cases["header"]]
    test = [dict(zip(header, (str(c).strip() for c in row))) for row in cases["rows"]]
    score, predf = make_predictor(train)
    tg = [g for g in map(goals, train) if g]
    target = sorted(score(g) for g in tg)[len(tg) // 4]
    pred = {}

    def p(i):
        if i not in pred:
            pred[i] = predf(test[i])
        return pred[i]

    def inspect(ep, strat, i):
        sys.stdout.write(json.dumps({"episode": ep, "strategy": strat, "inspect": i}) + "\n")
        sys.stdout.flush()
        resp = json.loads(sys.stdin.readline())
        return score([float(resp["goals"][k]) for k in GKEYS])

    def summarize(ds):
        lab = 11
        for pos, d in enumerate(ds, 1):
            if round(d, 12) <= round(target, 12):
                lab = pos
                break
        return {"best5": min(ds[:5]), "best10": min(ds), "labels_to_target": lab}

    results = []
    for e, ep in enumerate(cases["episodes"]):
        pool = list(ep["pool"])
        picks = sorted(pool, key=lambda i: (p(i), i))[:n_insp]
        ad = [inspect(e, "adaptive", i) for i in picks]
        rnd = []
        for i in list(ep["random_order"])[:n_insp]:
            rnd.append(inspect(e, "random", i))
        results.append({"episode": e, "adaptive": summarize(ad), "random": summarize(rnd)})
    comps = {}
    for m in ("best5", "best10", "labels_to_target"):
        comps[m] = compare([r["adaptive"][m] for r in results],
                           [r["random"][m] for r in results])
    return {"results": results, "comparisons": comps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    if a.cmd == "certify":
        with open(a.test) as f:
            out = certify(json.load(f))
    elif a.cmd == "race":
        train = load(a.train)
        with open(a.test) as f:
            out = race(train, json.load(f))
    else:
        train, test = load(a.train), load(a.test)
        if a.cmd == "guess":
            out = guess(train, test)
        elif a.cmd == "flow":
            out = flow(train, test)
        elif a.cmd == "blame":
            out = blame(train, test)
        elif a.cmd == "justify":
            out = justify(train, test)
        elif a.cmd == "choose":
            out = choose(train, test)
        elif a.cmd == "fix":
            out = fix(train, test)
        elif a.cmd == "spot":
            out = spot(train, test)
        else:
            out = remember(train, test)
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()

main()
