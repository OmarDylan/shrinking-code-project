import csv, sys, math, json
from bisect import bisect_right

GOALS = [('Lbs-', 0), ('Acc+', 1), ('Mpg+', 1)]


def read(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows


def num(row, header, name):
    try:
        i = header.index(name)
        v = float(row[i])
        return v if math.isfinite(v) else None
    except (ValueError, IndexError):
        return None


def goal_vals(header, rows):
    return [[num(r, header, g) for g, _ in GOALS] if all(num(r, header, g) is not None for g, _ in GOALS) else None
            for r in rows]


def goal_stats(tr):
    n = len(tr)
    stats = []
    for j in range(3):
        xs = [v[j] for v in tr]
        m = sum(xs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    return stats


def normv(v, m, sd):
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))


def d2h(v, stats):
    s = 0.0
    for j, (_, h) in enumerate(GOALS):
        m, sd = stats[j]
        s += (normv(v[j], m, sd) - h) ** 2
    return math.sqrt(s / 3)


def args():
    a = sys.argv[1:]
    train = test = None
    for i, x in enumerate(a):
        if x == '--train':
            train = a[i + 1]
        elif x == '--test':
            test = a[i + 1]
    return a[0], train, test


def remember(train, test):
    h1, r1 = read(train)
    tr = [v for v in goal_vals(h1, r1) if v is not None]
    stats = goal_stats(tr)
    h2, r2 = read(test)
    scores = [d2h(v, stats) for v in goal_vals(h2, r2)]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({"scores": scores, "top": order[:10], "best": order[0],
                      "median": order[len(scores) // 2], "worst": order[-1]}))


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


class Ridge:
    def __init__(self, X, y, lam=1.0):
        n, d = len(X), len(X[0])
        self.mu = [sum(r[j] for r in X) / n for j in range(d)]
        self.sd = [math.sqrt(sum((r[j] - self.mu[j]) ** 2 for r in X) / n) or 1.0 for j in range(d)]
        self.ym = sum(y) / n
        Z = [[(r[j] - self.mu[j]) / self.sd[j] for j in range(d)] for r in X]
        A = [[sum(z[i] * z[j] for z in Z) + (lam if i == j else 0.0) for j in range(d)] for i in range(d)]
        b = [sum(z[i] * (yy - self.ym) for z, yy in zip(Z, y)) for i in range(d)]
        self.w = solve(A, b)

    def predict(self, x):
        return self.ym + sum(w * (xi - m) / s for w, xi, m, s in zip(self.w, x, self.mu, self.sd))


def predict(train, test):
    h1, r1 = read(train)
    h2, r2 = read(test)
    origins = sorted({r[h1.index('origin')] for r in r1 if h1.index('origin') < len(r)} - {'?', ''})

    def raw(r, h):
        o = r[h.index('origin')] if 'origin' in h and h.index('origin') < len(r) else '?'
        return [num(r, h, 'Clndrs'), num(r, h, 'Volume'), num(r, h, 'Model'), o]

    traw = [raw(r, h1) for r in r1]
    means = []
    for j in range(3):
        xs = [t[j] for t in traw if t[j] is not None]
        means.append(sum(xs) / len(xs) if xs else 0.0)

    def feats(t, vol):
        c = t[0] if t[0] is not None else means[0]
        m = t[2] if t[2] is not None else means[2]
        f = [c, m]
        if vol:
            v = t[1] if t[1] is not None else means[1]
            f += [v, math.log(max(v, 1.0))]
        return f + [1.0 if t[3] == o else 0.0 for o in origins]

    gv = goal_vals(h1, r1)
    idx = [i for i, v in enumerate(gv) if v is not None]
    stats = goal_stats([gv[i] for i in idx])
    ymp = [math.log(max(gv[i][2], 1e-6)) for i in idx]
    yd = [d2h(gv[i], stats) for i in idx]
    models = {}
    for vol in (True, False):
        X = [feats(traw[i], vol) for i in idx]
        models[vol] = (Ridge(X, ymp, 2.0), Ridge(X, yd, 2.0))

    mpg, scores = [], []
    for r in r2:
        t = raw(r, h2)
        vol = t[1] is not None
        mm, md = models[vol]
        x = feats(t, vol)
        p = math.exp(max(-5.0, min(6.0, mm.predict(x))))
        s = max(0.0, min(1.0, md.predict(x)))
        mpg.append(p)
        scores.append(s)
    return mpg, scores


def guess(train, test):
    mpg, scores = predict(train, test)
    print(json.dumps({"mpg": mpg, "scores": scores}))


def compare(xs, ys):
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    vy = sum((y - my) ** 2 for y in ys) / (m - 1)
    pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (math.sqrt(pv) + 1e-32)
    s = 0
    for x in xs:
        for y in ys:
            s += (x > y) - (x < y)
    cliffs = abs(s) / (n * m)
    sx, sy = sorted(xs), sorted(ys)
    d = max(abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m) for v in set(sx) | set(sy))
    ks = d / math.sqrt(1 / n + 1 / m)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks,
            "same": bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)}


def certify(test):
    with open(test) as f:
        cases = json.load(f)
    out = [compare([float(x) for x in c["xs"]], [float(y) for y in c["ys"]]) for c in cases]
    print(json.dumps({"comparisons": out}))


def flow(train, test):
    mpg, scores = predict(train, test)
    h1, r1 = read(train)
    tr = [v for v in goal_vals(h1, r1) if v is not None]
    stats = goal_stats(tr)
    hist = [d2h(v, stats) for v in tr]
    alerts = []
    for i in range(len(scores)):
        if i < 19 or len(hist) < 2:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, scores[i - 19:i + 1])["same"])
    print(json.dumps({"mpg": mpg, "scores": scores, "alerts": alerts}))


NUM_IN = ['Clndrs', 'Volume', 'Model']


def diagnose(train, test):
    h1, r1 = read(train)
    h2, r2 = read(test)
    gv = goal_vals(h1, r1)
    idx = [i for i, v in enumerate(gv) if v is not None]
    stats = goal_stats([gv[i] for i in idx])
    order = sorted(idx, key=lambda i: (round(d2h(gv[i], stats), 12), i))
    q = max(1, len(order) // 4)
    good, poor = order[:q], order[-q:]

    def cat(r, h):
        k = h.index('origin') if 'origin' in h else -1
        v = r[k] if 0 <= k < len(r) else '?'
        return None if v in ('?', '') else v

    nstat, centers = {}, {}
    for name in NUM_IN:
        xs = [num(r1[i], h1, name) for i in idx]
        xs = [x for x in xs if x is not None]
        n = len(xs)
        m = sum(xs) / n if n else 0.0
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        nstat[name] = (m, sd)
        cs = []
        for grp in (good, poor):
            vs = [num(r1[i], h1, name) for i in grp]
            vs = [x for x in vs if x is not None]
            cs.append(normv(sum(vs) / len(vs), m, sd) if vs else None)
        centers[name] = cs

    freqs = []
    for grp in (good, poor):
        vs = [cat(r1[i], h1) for i in grp]
        vs = [x for x in vs if x is not None]
        c = {}
        for x in vs:
            c[x] = c.get(x, 0) + 1
        freqs.append((c, len(vs)))

    def fr(g, c):
        cnt, n = freqs[g]
        return cnt.get(c, 0) / n if n else 0.0

    out = []
    for r in r2:
        goals = {}
        for g, h in GOALS:
            v = num(r, h2, g)
            if v is None:
                goals[g] = 0.0
            else:
                j = [x for x, _ in GOALS].index(g)
                goals[g] = abs(normv(v, *stats[j]) - h)
        feats = {}
        for name in NUM_IN:
            v = num(r, h2, name)
            gc, pc = centers[name]
            if v is None or gc is None or pc is None:
                feats[name] = 0.0
            else:
                nv = normv(v, *nstat[name])
                feats[name] = max(0.0, abs(nv - gc) - abs(nv - pc))
        c = cat(r, h2)
        feats['origin'] = 0.0 if c is None else max(0.0, fr(1, c) - fr(0, c))
        key = lambda d: sorted(d, key=lambda k: (-round(d[k], 10), k))
        out.append({"goals": goals, "features": feats,
                    "goal_order": key(goals), "feature_order": key(feats)})
    return out


def blame(train, test):
    print(json.dumps({"diagnoses": diagnose(train, test)}))


def justify(train, test):
    h1, r1 = read(train)
    tr = [v for v in goal_vals(h1, r1) if v is not None]
    stats = goal_stats(tr)
    hist = sorted(d2h(v, stats) for v in tr)
    cutoff = hist[len(hist) // 2]
    h2, r2 = read(test)
    scores = [d2h(v, stats) for v in goal_vals(h2, r2)]
    diag = diagnose(train, test)
    n = len(scores)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = scores[j], scores[k]
        accepted = round(s, 12) <= round(cutoff, 12)
        winner = None
        adv_goal = None
        if round(s, 12) != round(o, 12):
            winner = j if s < o else k
            loser = k if winner == j else j
            adv = {g: diag[loser]["goals"][g] ** 2 - diag[winner]["goals"][g] ** 2 for g, _ in GOALS}
            adv_goal = sorted(adv, key=lambda g: (-round(adv[g], 10), g))[0]
        weakest = diag[j]["goal_order"][0]
        tf = diag[j]["feature_order"][0]
        top = tf if diag[j]["features"][tf] > 1e-10 else None
        text = "Car %d %s (score %.3f, other score %.3f, cutoff %.3f); weakest goal: %s." % (
            j, "accepted" if accepted else "rejected", s, o, cutoff, weakest)
        if winner is not None:
            text += " Recommend car %d; largest advantage: %s." % (winner, adv_goal)
        else:
            text += " The comparison is a tie."
        if top is not None:
            text += " Top feature: %s." % top
        out.append({"row": j, "other": k, "score": s, "other_score": o, "cutoff": cutoff,
                    "accepted": bool(accepted), "winner": winner, "advantage_goal": adv_goal,
                    "weakest_goal": weakest, "top_feature": top, "text": text})
    print(json.dumps({"explanations": out}))


def choose(train, test):
    h1, r1 = read(train)
    tr = [v for v in goal_vals(h1, r1) if v is not None]
    stats = goal_stats(tr)
    _, pred = predict(train, test)
    n = len(pred)
    order = sorted(range(n), key=lambda i: (pred[i], i))
    picks = order[:min(10, n)]
    seen = []
    for i in picks:
        print(json.dumps({"inspect": i}))
        sys.stdout.flush()
        resp = json.loads(sys.stdin.readline())
        g = resp["goals"]
        v = [float(g[name]) for name, _ in GOALS]
        seen.append((round(d2h(v, stats), 12), i))
    seen.sort()
    print(json.dumps({"selected": [i for _, i in seen[:3]]}))
    sys.stdout.flush()


def main():
    if len(sys.argv) < 2:
        sys.exit(1)
    cmd, train, test = args()
    if cmd == 'remember':
        remember(train, test)
    elif cmd == 'guess':
        guess(train, test)
    elif cmd == 'certify':
        certify(test)
    elif cmd == 'flow':
        flow(train, test)
    elif cmd == 'blame':
        blame(train, test)
    elif cmd == 'justify':
        justify(train, test)
    elif cmd == 'choose':
        choose(train, test)
    else:
        sys.exit(1)


main()
