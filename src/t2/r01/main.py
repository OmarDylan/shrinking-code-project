import csv, sys, json, math, argparse
from bisect import bisect_left, bisect_right

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def num(s):
    if s == "?" or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def goal_stats(h, tr, goals):
    stats = {}
    for i, _ in goals:
        v = [x for x in (num(r[i]) if i < len(r) else None for r in tr) if x is not None]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        stats[i] = (m, sd)
    return stats

def norm(x, m, sd):
    z = (x - m) / (sd + 1e-32)
    z = max(-3.0, min(3.0, z))
    return 1 / (1 + math.exp(-1.702 * z))

def d2h(vals, goals, stats):
    s = 0.0
    for i, hv in goals:
        x = vals.get(i)
        if x is None:
            continue
        s += (norm(x, *stats[i]) - hv) ** 2
    return math.sqrt(s / len(goals))

def fit(X, y, lam=1.0):
    n = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) + (lam if i == j and i > 0 else 0.0)
          for j in range(n)] for i in range(n)]
    b = [sum(x[i] * t for x, t in zip(X, y)) for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda k: abs(A[k][c]))
        A[c], A[p] = A[p], A[c]
        b[c], b[p] = b[p], b[c]
        d = A[c][c] or 1e-12
        for k in range(c + 1, n):
            f = A[k][c] / d
            if f:
                for j in range(c, n):
                    A[k][j] -= f * A[c][j]
                b[k] -= f * b[c]
    w = [0.0] * n
    for c in range(n - 1, -1, -1):
        w[c] = (b[c] - sum(A[c][j] * w[j] for j in range(c + 1, n))) / (A[c][c] or 1e-12)
    return w

def remember(h, tr, h2, te, goals, stats):
    idx2 = {n: i for i, n in enumerate(h2)}
    scores = []
    for r in te:
        vals = {}
        for i, _ in goals:
            j = idx2.get(h[i], i)
            vals[i] = num(r[j]) if j < len(r) else None
        scores.append(d2h(vals, goals, stats))
    order = sorted(range(len(scores)), key=lambda k: (scores[k], k))
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[len(order) // 2], "worst": order[-1]}

def guess(h, tr, h2, te, goals, stats):
    idx2 = {n: i for i, n in enumerate(h2)}
    gi = {i for i, _ in goals}
    nums = [i for i, n in enumerate(h) if i not in gi and n[:1].isupper() and not n.endswith("X")]
    syms = [i for i, n in enumerate(h) if i not in gi and n[:1].islower() and not n.endswith("X")]
    mpg_i = next(i for i, n in enumerate(h) if n.startswith("Mpg"))

    def get(r, i, test=False):
        j = idx2.get(h[i]) if test else i
        return r[j] if j is not None and j < len(r) else "?"

    ns = {}
    for i in nums:
        v = [x for x in (num(get(r, i)) for r in tr) if x is not None]
        m = sum(v) / len(v) if v else 0.0
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 1.0
        ns[i] = (m, sd or 1.0)
    cats = {i: sorted({get(r, i) for r in tr} - {"?", ""}) for i in syms}

    def feat(r, ex, test):
        f = [1.0]
        for i in nums:
            if i in ex:
                continue
            x = num(get(r, i, test))
            f.append(0.0 if x is None else (x - ns[i][0]) / ns[i][1])
        for i in syms:
            v = get(r, i, test)
            f.extend(1.0 if v == c else 0.0 for c in cats[i])
        return f

    tgt = []
    for r in tr:
        vals = {i: num(get(r, i)) for i, _ in goals}
        tgt.append((vals, d2h(vals, goals, stats)))

    models = {}
    def model(ex, key):
        k = (ex, key)
        if k not in models:
            X, y = [], []
            for r, (vals, d) in zip(tr, tgt):
                t = d if key == "d" else vals[key]
                if t is None:
                    continue
                X.append(feat(r, ex, False))
                y.append(t)
            models[k] = fit(X, y) if X else None
        return models[k]

    def pred(r, ex, key):
        w = model(ex, key)
        if w is None:
            return None
        return sum(a * b for a, b in zip(w, feat(r, ex, True)))

    mpg, scores = [], []
    base_m = sum(v for v in (t[0][mpg_i] for t in tgt) if v is not None)
    cnt = sum(1 for t in tgt if t[0][mpg_i] is not None)
    base_m = base_m / cnt if cnt else 0.0
    base_d = sum(t[1] for t in tgt) / len(tgt)
    for r in te:
        ex = frozenset(i for i in nums if num(get(r, i, True)) is None)
        pm = pred(r, ex, mpg_i)
        pd = pred(r, ex, "d")
        mpg.append(max(0.0, pm) if pm is not None else base_m)
        scores.append(min(1.0, max(0.0, pd)) if pd is not None else base_d)
    return {"mpg": mpg, "scores": scores}

# ---- certify skill ----
COHEN_T, CLIFFS_T, KS_T = 0.35, 0.197, 1.36

def _clean(v):
    out = []
    for x in v or []:
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if f == f:
            out.append(f)
    return out

def cohen_d(xs, ys):
    n, m = len(xs), len(ys)
    if n == 0 or m == 0:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    dof = n + m - 2
    sp = math.sqrt((vx + vy) / dof) if dof > 0 else 0.0
    diff = abs(mx - my)
    if sp == 0:
        return 0.0 if diff == 0 else 1e12
    return diff / sp

def cliffs_delta(xs, ys):
    n, m = len(xs), len(ys)
    if n == 0 or m == 0:
        return 0.0
    s = sorted(ys)
    gt = lt = 0
    for x in xs:
        gt += bisect_left(s, x)
        lt += m - bisect_right(s, x)
    return abs(gt - lt) / (n * m)

def ks_norm(xs, ys):
    n, m = len(xs), len(ys)
    if n == 0 or m == 0:
        return 0.0
    a, b = sorted(xs), sorted(ys)
    d = 0.0
    for v in sorted(set(a) | set(b)):
        d = max(d, abs(bisect_right(a, v) / n - bisect_right(b, v) / m))
    return d * math.sqrt(n * m / (n + m))

def is_same(xs, ys):
    return (cohen_d(xs, ys) < COHEN_T and cliffs_delta(xs, ys) < CLIFFS_T
            and ks_norm(xs, ys) < KS_T)

def certify(cases):
    res = []
    for c in cases:
        xs, ys = _clean(c.get("xs")), _clean(c.get("ys"))
        co, cl, ks = cohen_d(xs, ys), cliffs_delta(xs, ys), ks_norm(xs, ys)
        same = co < COHEN_T and cl < CLIFFS_T and ks < KS_T
        res.append({"cohen": co, "cliffs": cl, "ks": ks, "same": bool(same)})
    return {"comparisons": res}

# ---- flow skill ----
WINDOW = 20

def flow(h, tr, h2, te, goals, stats):
    out = guess(h, tr, h2, te, goals, stats)
    hist = []
    for r in tr:
        vals = {i: (num(r[i]) if i < len(r) else None) for i, _ in goals}
        hist.append(d2h(vals, goals, stats))
    sc = out["scores"]
    alerts = []
    for k in range(len(sc)):
        if k + 1 < WINDOW:
            alerts.append(False)
        else:
            alerts.append(not is_same(sc[k + 1 - WINDOW:k + 1], hist))
    out["alerts"] = alerts
    return out

# ---- blame skill ----
def blame(h, tr, h2, te, goals, stats):
    idx2 = {n: i for i, n in enumerate(h2)}
    gi = {i for i, _ in goals}
    ins = [i for i, n in enumerate(h) if i not in gi and not n.endswith("X")
           and (n[:1].isupper() or n[:1].islower())]

    def cell(r, i, test):
        j = idx2.get(h[i]) if test else i
        return r[j] if j is not None and j < len(r) else "?"

    ns = {}
    for i in ins:
        if h[i][:1].isupper():
            v = [x for x in (num(cell(r, i, False)) for r in tr) if x is not None]
            m = sum(v) / len(v) if v else 0.0
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
            ns[i] = (m, sd)

    ds = []
    for r in tr:
        vals = {i: num(cell(r, i, False)) for i, _ in goals}
        ds.append(d2h(vals, goals, stats))
    order = sorted(range(len(tr)), key=lambda k: (ds[k], k))
    q = max(1, len(tr) // 4) if tr else 0
    good = [tr[k] for k in order[:q]]
    poor = [tr[k] for k in order[len(order) - q:]] if q else []

    prof = {}
    for i in ins:
        if i in ns:
            def center(g):
                v = [norm(x, *ns[i]) for x in (num(cell(r, i, False)) for r in g) if x is not None]
                return sum(v) / len(v) if v else None
            prof[i] = (center(good), center(poor))
        else:
            def freq(g):
                c = {}
                for r in g:
                    v = cell(r, i, False)
                    c[v] = c.get(v, 0) + 1
                return {k: n / len(g) for k, n in c.items()} if g else {}
            prof[i] = (freq(good), freq(poor))

    out = []
    for r in te:
        goalsd = {}
        for i, hv in goals:
            x = num(cell(r, i, True))
            goalsd[h[i]] = 0.0 if x is None else abs(norm(x, *stats[i]) - hv)
        feats = {}
        for i in ins:
            s = 0.0
            if i in ns:
                x = num(cell(r, i, True))
                g, p = prof[i]
                if x is not None and g is not None and p is not None:
                    nx = norm(x, *ns[i])
                    s = max(0.0, abs(nx - g) - abs(nx - p))
            else:
                v = cell(r, i, True)
                if v not in ("?", ""):
                    g, p = prof[i]
                    s = max(0.0, p.get(v, 0.0) - g.get(v, 0.0))
            feats[h[i]] = s
        out.append({
            "goals": goalsd,
            "features": feats,
            "goal_order": sorted(goalsd, key=lambda k: (-goalsd[k], k)),
            "feature_order": sorted(feats, key=lambda k: (-feats[k], k)),
        })
    return {"diagnoses": out}

# ---- justify skill ----
def justify(h, tr, h2, te, goals, stats):
    scores = remember(h, tr, h2, te, goals, stats)["scores"]
    diag = blame(h, tr, h2, te, goals, stats)["diagnoses"]
    ds = sorted(d2h({i: num(r[i]) if i < len(r) else None for i, _ in goals}, goals, stats)
                for r in tr)
    n = len(ds)
    cutoff = 0.0 if n == 0 else (ds[n // 2] if n % 2 else (ds[n // 2 - 1] + ds[n // 2]) / 2)
    N = len(te)
    res = []
    for k in range(N):
        o = (k + 1) % N
        s, os_ = scores[k], scores[o]
        accepted = s <= cutoff
        gk, go = diag[k]["goals"], diag[o]["goals"]
        weakest = diag[k]["goal_order"][0] if gk else None
        tie = round(s, 12) == round(os_, 12)
        winner = adv = None
        if not tie:
            winner = k if s < os_ else o
            lose = o if winner == k else k
            gw, gl = diag[winner]["goals"], diag[lose]["goals"]
            adv = min(gw, key=lambda g: (-(gl[g] ** 2 - gw[g] ** 2), g))
        fo = diag[k]["feature_order"]
        top = fo[0] if fo and diag[k]["features"][fo[0]] > 0 else None
        if tie:
            t = f"Cars {k} and {o} tie on quality (d2h {s:.6f} vs {os_:.6f})."
        else:
            t = (f"Car {winner} is recommended over car {o if winner == k else k}: "
                 f"d2h {s:.6f} vs {os_:.6f} (lower is better); "
                 f"the biggest advantage comes from {adv}.")
        t += (f" Car {k} is {'accepted' if accepted else 'rejected'}: d2h {s:.6f} "
              f"{'<=' if accepted else '>'} cutoff {cutoff:.6f}."
              f" Its weakest goal is {weakest}")
        if top:
            t += f"; most associated feature is {top} (retrospective, not causal)."
        else:
            t += "; no feature shows positive evidence."
        res.append({"row": k, "other": o, "score": s, "other_score": os_,
                    "cutoff": cutoff, "accepted": accepted, "winner": winner,
                    "advantage_goal": adv, "weakest_goal": weakest,
                    "top_feature": top, "text": t})
    return {"explanations": res}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    if a.cmd == "certify":
        with open(a.test) as f:
            cases = json.load(f)
        print(json.dumps(certify(cases)))
        return

    h, tr = load(a.train)
    h2, te = load(a.test)
    goals = [(i, 1.0 if n.endswith("+") else 0.0)
             for i, n in enumerate(h) if n.endswith("+") or n.endswith("-")]
    stats = goal_stats(h, tr, goals)
    if a.cmd == "flow":
        out = flow(h, tr, h2, te, goals, stats)
    elif a.cmd == "guess":
        out = guess(h, tr, h2, te, goals, stats)
    elif a.cmd == "blame":
        out = blame(h, tr, h2, te, goals, stats)
    elif a.cmd == "justify":
        out = justify(h, tr, h2, te, goals, stats)
    else:
        out = remember(h, tr, h2, te, goals, stats)
    print(json.dumps(out))

main()
