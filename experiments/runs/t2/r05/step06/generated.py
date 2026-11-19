import sys, csv, math, json
from bisect import bisect_left, bisect_right

GOALS = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def num(row, j):
    if j < 0 or j >= len(row):
        return None
    try:
        v = float(row[j])
        return v if math.isfinite(v) else None
    except ValueError:
        return None

def goal_stats(th, tr):
    stats = []
    for g, _ in GOALS:
        j = th.index(g)
        vals = [v for v in (num(r, j) for r in tr) if v is not None]
        n = len(vals)
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    return stats

def norm(v, m, sd):
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))

def d2h(vals, stats):
    s = 0.0
    for (g, h), v, (m, sd) in zip(GOALS, vals, stats):
        s += (norm(v, m, sd) - h) ** 2
    return math.sqrt(s / 3)

def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            continue
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]

def build(th, tr, ys, nums, lam=1.0):
    """Ridge on standardized numerics, their squares, origin one-hot. Returns predictor(header,row)."""
    oj = th.index("origin")
    cats = sorted({r[oj] for r in tr if oj < len(r)})
    stat = []
    for nm in nums:
        j = th.index(nm)
        v = [x for x in (num(r, j) for r in tr) if x is not None]
        mu = sum(v) / len(v)
        sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)) or 1.0
        stat.append((mu, sd))

    def vec(h, row):
        z = []
        for nm, (mu, sd) in zip(nums, stat):
            x = num(row, h.index(nm) if nm in h else -1)
            z.append(0.0 if x is None else (x - mu) / sd)
        o = row[h.index("origin")] if "origin" in h and h.index("origin") < len(row) else None
        return z + [t * t for t in z] + [1.0 if o == c else 0.0 for c in cats] + [1.0]

    X = [vec(th, r) for r in tr]
    d = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) + (lam if i == j and i < d - 1 else 0.0)
          for j in range(d)] for i in range(d)]
    ws = []
    for y in ys:
        b = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(d)]
        ws.append(solve(A, b))
    return lambda h, row: [sum(a * c for a, c in zip(w, vec(h, row))) for w in ws]

def guess(train, test):
    th, tr = load(train)
    eh, te = load(test)
    stats = goal_stats(th, tr)
    gi = [th.index(g) for g, _ in GOALS]
    rows, lm, sc = [], [], []
    for r in tr:
        v = [num(r, j) for j in gi]
        if None in v or v[2] <= 0:
            continue
        rows.append(r)
        lm.append(math.log(v[2]))
        sc.append(d2h(v, stats))
    full = build(th, rows, [lm, sc], ["Clndrs", "Volume", "Model"])
    part = build(th, rows, [lm, sc], ["Clndrs", "Model"])
    vj = eh.index("Volume") if "Volume" in eh else -1
    mpg, scores = [], []
    for r in te:
        f = full if num(r, vj) is not None else part
        p = f(eh, r)
        m = math.exp(max(-5.0, min(8.0, p[0])))
        mpg.append(m if math.isfinite(m) and m > 0 else 1.0)
        s = p[1]
        scores.append(max(0.0, min(1.0, s)) if math.isfinite(s) else 0.5)
    return {"mpg": mpg, "scores": scores}

def remember(train, test):
    th, tr = load(train)
    eh, te = load(test)
    stats = goal_stats(th, tr)
    idx = [eh.index(g) for g, _ in GOALS]
    scores = [d2h([float(row[i]) for i in idx], stats) for row in te]
    order = sorted(range(len(scores)), key=lambda k: (scores[k], k))
    return {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }

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

def certify(test):
    with open(test) as f:
        cases = json.load(f)
    return {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}

def flow(train, test):
    out = guess(train, test)
    th, tr = load(train)
    stats = goal_stats(th, tr)
    gi = [th.index(g) for g, _ in GOALS]
    hist = []
    for r in tr:
        v = [num(r, j) for j in gi]
        if None not in v:
            hist.append(d2h(v, stats))
    sc = out["scores"]
    alerts = []
    for i in range(len(sc)):
        if i < 19 or len(hist) < 2:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, sc[i - 19:i + 1])["same"])
    out["alerts"] = alerts
    return out

NUM_IN = ["Clndrs", "Volume", "Model"]
CAT_IN = ["origin"]

def blame(train, test):
    th, tr = load(train)
    eh, te = load(test)
    gi = [th.index(g) for g, _ in GOALS]
    tr = [r for r in tr if None not in [num(r, j) for j in gi]]
    stats = goal_stats(th, tr)
    scored = sorted(
        ((round(d2h([num(r, j) for j in gi], stats), 12), k) for k, r in enumerate(tr)))
    q = max(1, len(tr) // 4)
    good = [tr[k] for _, k in scored[:q]]
    poor = [tr[k] for _, k in scored[-q:]]

    nstat = {}
    for nm in NUM_IN:
        j = th.index(nm)
        v = [x for x in (num(r, j) for r in tr) if x is not None]
        if v:
            mu = sum(v) / len(v)
            sd = math.sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        else:
            mu, sd = 0.0, 0.0
        centers = []
        for grp in (good, poor):
            gv = [x for x in (num(r, j) for r in grp) if x is not None]
            c = sum(gv) / len(gv) if gv else mu
            centers.append(norm(c, mu, sd))
        nstat[nm] = (mu, sd, centers[0], centers[1])

    def known(r, j):
        return r[j] if j < len(r) and r[j] not in ("", "?") else None

    cfreq = {}
    for nm in CAT_IN:
        j = th.index(nm)
        fr = []
        for grp in (good, poor):
            vals = [x for x in (known(r, j) for r in grp) if x is not None]
            cnt = {}
            for x in vals:
                cnt[x] = cnt.get(x, 0) + 1
            fr.append((cnt, len(vals)))
        cfreq[nm] = fr

    def order(d):
        return sorted(d, key=lambda k: (-round(d[k], 10), k))

    out = []
    for r in te:
        feats = {}
        for nm in NUM_IN:
            x = num(r, eh.index(nm) if nm in eh else -1)
            if x is None:
                feats[nm] = 0.0
                continue
            mu, sd, gc, pc = nstat[nm]
            nx = norm(x, mu, sd)
            feats[nm] = max(0.0, abs(nx - gc) - abs(nx - pc))
        for nm in CAT_IN:
            c = known(r, eh.index(nm)) if nm in eh else None
            if c is None:
                feats[nm] = 0.0
                continue
            (gc, gn), (pc, pn) = cfreq[nm]
            gf = gc.get(c, 0) / gn if gn else 0.0
            pf = pc.get(c, 0) / pn if pn else 0.0
            feats[nm] = max(0.0, pf - gf)
        goals = {}
        for (g, h), (m, sd) in zip(GOALS, stats):
            x = num(r, eh.index(g) if g in eh else -1)
            goals[g] = 0.0 if x is None else abs(norm(x, m, sd) - h)
        out.append({"goals": goals, "features": feats,
                    "goal_order": order(goals), "feature_order": order(feats)})
    return {"diagnoses": out}

def justify(train, test):
    th, tr = load(train)
    eh, te = load(test)
    gi = [th.index(g) for g, _ in GOALS]
    trc = [r for r in tr if None not in [num(r, j) for j in gi]]
    stats = goal_stats(th, trc)
    tvals = sorted(d2h([num(r, j) for j in gi], stats) for r in trc)
    cutoff = tvals[len(trc) // 2]
    ei = [eh.index(g) for g, _ in GOALS]
    obs = [[num(r, j) for j in ei] for r in te]
    scores = [d2h(v, stats) for v in obs]
    diags = blame(train, test)["diagnoses"]

    def gaps(v):
        return {g: (norm(x, m, sd) - h) ** 2
                for (g, h), x, (m, sd) in zip(GOALS, v, stats)}

    out = []
    n = len(te)
    for j in range(n):
        k = (j + 1) % n
        s, o = scores[j], scores[k]
        acc = bool(round(s, 12) <= round(cutoff, 12))
        d = diags[j]
        weakest = d["goal_order"][0]
        fo = d["feature_order"][0]
        top = fo if d["features"][fo] > 1e-10 else None
        winner = None
        adv = None
        if round(s, 12) != round(o, 12):
            winner = j if s < o else k
            loser = k if winner == j else j
            gw, gl = gaps(obs[winner]), gaps(obs[loser])
            a = {g: gl[g] - gw[g] for g in gw}
            adv = sorted(a, key=lambda g: (-round(a[g], 10), g))[0]
        text = "Car %d %s: score %.3f, other score %.3f, cutoff %.3f; weakest goal %s." % (
            j, "accepted" if acc else "rejected", s, o, cutoff, weakest)
        if winner is not None:
            text += " Recommend car %d; largest advantage goal %s." % (winner, adv)
        else:
            text += " The comparison is a tie."
        if top is not None:
            text += " Top feature: %s." % top
        out.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": acc, "winner": winner,
                    "advantage_goal": adv, "weakest_goal": weakest,
                    "top_feature": top, "text": text})
    return {"explanations": out}

def main():
    a = sys.argv[1:]
    if not a or a[0] not in ("remember", "guess", "certify", "flow", "blame", "justify"):
        sys.exit(1)
    train = test = None
    i = 1
    while i < len(a):
        if a[i] == "--train":
            train = a[i + 1]; i += 2
        elif a[i] == "--test":
            test = a[i + 1]; i += 2
        else:
            i += 1
    if a[0] == "certify":
        out = certify(test)
    elif a[0] == "remember":
        out = remember(train, test)
    elif a[0] == "flow":
        out = flow(train, test)
    elif a[0] == "blame":
        out = blame(train, test)
    elif a[0] == "justify":
        out = justify(train, test)
    else:
        out = guess(train, test)
    sys.stdout.write(json.dumps(out))

main()
