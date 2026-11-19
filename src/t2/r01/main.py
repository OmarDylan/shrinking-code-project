import csv, sys, json, math, argparse

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
    # vals: dict col -> value or None
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

    # targets: goals and d2h computed from training observed goals only
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    h, tr = load(a.train)
    h2, te = load(a.test)
    goals = [(i, 1.0 if n.endswith("+") else 0.0)
             for i, n in enumerate(h) if n.endswith("+") or n.endswith("-")]
    stats = goal_stats(h, tr, goals)
    if a.cmd == "guess":
        out = guess(h, tr, h2, te, goals, stats)
    else:
        out = remember(h, tr, h2, te, goals, stats)
    print(json.dumps(out))

main()
