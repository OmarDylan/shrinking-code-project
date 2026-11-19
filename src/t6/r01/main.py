import sys, json, argparse
from bisect import bisect_left, bisect_right
from math import exp, sqrt

COHEN_T, CLIFFS_T, KS_T = 0.35, 0.197, 1.36
WINDOW = 20


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [l.strip() for l in f if l.strip()]
    names = [s.strip() for s in lines[0].split(",")]
    rows = [[s.strip() for s in l.split(",")] for l in lines[1:]]
    return names, rows


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def goal_stats(names, train, goals):
    stats = {}
    for n, _ in goals:
        i = names.index(n)
        vs = [v for v in (num(r[i]) for r in train if i < len(r)) if v is not None]
        k = len(vs)
        mu = sum(vs) / k if k else 0.0
        sd = sqrt(sum((v - mu) ** 2 for v in vs) / (k - 1)) if k > 1 else 0.0
        stats[n] = (mu, sd)
    return stats


def logi(v, mu, sd):
    z = max(-3.0, min(3.0, (v - mu) / (sd + 1e-32)))
    return 1 / (1 + exp(-1.702 * z))


def remember(a):
    names, train = load(a.train)
    tnames, test = load(a.test)
    goals = [(n, 0.0 if n[-1] == "-" else 1.0) for n in names if n[-1] in "+-"]
    stats = goal_stats(names, train, goals)

    def d2h(r):
        tot = 0.0
        for n, h in goals:
            i = tnames.index(n)
            v = num(r[i]) if i < len(r) else None
            if v is None:
                tot += 1.0
                continue
            mu, sd = stats[n]
            x = logi(v, mu, sd)
            tot += (x - h) ** 2
        return sqrt(tot / len(goals))

    scores = [d2h(r) for r in test]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }))


def model(a, K=7):
    "Fit on training only; return test rows, per-row predictor, training d2h list."
    names, train = load(a.train)
    tnames, test = load(a.test)
    goals = [(n, 0.0 if n[-1] == "-" else 1.0) for n in names if n[-1] in "+-"]
    stats = goal_stats(names, train, goals)
    mi = next((i for i, n in enumerate(names) if n.lower().startswith("mpg")),
              names.index(goals[-1][0]))

    def val(r, i):
        return num(r[i]) if i < len(r) else None

    def d2h_train(r):
        tot = 0.0
        for n, h in goals:
            v = val(r, names.index(n))
            if v is None:
                tot += 1.0
                continue
            mu, sd = stats[n]
            tot += (logi(v, mu, sd) - h) ** 2
        return sqrt(tot / len(goals))

    tr = [(r, val(r, mi), d2h_train(r)) for r in train]
    mp = [t[1] for t in tr if t[1] is not None]
    base_mpg = sum(mp) / len(mp) if mp else 0.0
    base_d = sum(t[2] for t in tr) / len(tr) if tr else 0.0

    xs = []
    for i, n in enumerate(names):
        if n[-1] in "X+-" or n not in tnames:
            continue
        xs.append((i, tnames.index(n), n[0].isupper()))
    cstat = {}
    for i, _, isn in xs:
        if isn:
            vs = [v for v in (val(r, i) for r in train) if v is not None]
            k = len(vs)
            mu = sum(vs) / k if k else 0.0
            sd = sqrt(sum((v - mu) ** 2 for v in vs) / (k - 1)) if k > 1 else 0.0
            cstat[i] = (mu, sd)

    def prep(r, i, isn):
        if i >= len(r) or r[i] in ("?", ""):
            return None
        if isn:
            v = num(r[i])
            return None if v is None else logi(v, *cstat[i])
        return r[i]

    trx = [[prep(r, i, isn) for i, _, isn in xs] for r, _, _ in tr]

    def predict(r):
        q = [prep(r, j, isn) for _, j, isn in xs]
        ds = []
        for k, t in enumerate(trx):
            s = c = 0
            for (_, _, isn), a_, b_ in zip(xs, q, t):
                if a_ is None or b_ is None:
                    continue
                g = abs(a_ - b_) if isn else float(a_ != b_)
                s += g * g
                c += 1
            ds.append((sqrt(s / c) if c else 1.0, k))
        ds.sort()
        nn = [k for _, k in ds[:K]]
        ms = [tr[k][1] for k in nn if tr[k][1] is not None]
        dd = [tr[k][2] for k in nn]
        return (sum(ms) / len(ms) if ms else base_mpg,
                sum(dd) / len(dd) if dd else base_d)

    return test, predict, [t[2] for t in tr]


def guess(a):
    test, predict, _ = model(a)
    out = [predict(r) for r in test]
    print(json.dumps({"mpg": [p[0] for p in out], "scores": [p[1] for p in out]}))


# ---- certify: distribution comparison ----

def cohen_d(xs, ys):
    "Absolute Cohen's d with pooled sample standard deviation."
    n, m = len(xs), len(ys)
    if not n or not m:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((v - mx) ** 2 for v in xs)
    vy = sum((v - my) ** 2 for v in ys)
    df = n + m - 2
    pool = sqrt((vx + vy) / df) if df > 0 else 0.0
    diff = abs(mx - my)
    if diff < 1e-12:
        return 0.0
    return diff / (pool + 1e-32)


def cliffs_delta(xs, ys):
    "Absolute Cliff's delta; xs, ys sorted. Ties count as neither."
    n, m = len(xs), len(ys)
    if not n or not m:
        return 0.0
    gt = lt = 0
    for x in xs:
        gt += bisect_left(ys, x)
        lt += m - bisect_right(ys, x)
    return abs(gt - lt) / (n * m)


def ks_norm(xs, ys):
    "KS distance divided by sqrt((n+m)/(n*m)); compare to 1.36."
    n, m = len(xs), len(ys)
    if not n or not m:
        return 0.0
    d = 0.0
    for v in sorted(set(xs) | set(ys)):
        d = max(d, abs(bisect_right(xs, v) / n - bisect_right(ys, v) / m))
    return d / sqrt((n + m) / (n * m))


def compare(xs, ys):
    xs = sorted(float(v) for v in xs)
    ys = sorted(float(v) for v in ys)
    c, cl, k = cohen_d(xs, ys), cliffs_delta(xs, ys), ks_norm(xs, ys)
    return {"cohen": c, "cliffs": cl, "ks": k,
            "same": bool(c <= COHEN_T and cl <= CLIFFS_T and k <= KS_T)}


def certify(a):
    with open(a.test, encoding="utf-8-sig") as f:
        cases = json.load(f)
    print(json.dumps({"comparisons": [compare(c.get("xs", []), c.get("ys", []))
                                      for c in cases]}))


# ---- flow: sequential judging with drift alerts ----

def flow(a):
    test, predict, hist = model(a)
    hist = sorted(hist)
    mpg, scores, alerts, win = [], [], [], []
    for r in test:  # each prediction depends only on training data and this row
        m, s = predict(r)
        mpg.append(m)
        scores.append(s)
        win.append(s)
        if len(win) > WINDOW:
            win.pop(0)
        alerts.append(len(win) == WINDOW and not compare(win, hist)["same"])
    print(json.dumps({"mpg": mpg, "scores": scores, "alerts": alerts}))


# ---- blame: retrospective diagnosis of poor cars ----

def blame(a):
    names, train = load(a.train)
    tnames, test = load(a.test)
    goals = [(n, 0.0 if n[-1] == "-" else 1.0) for n in names if n[-1] in "+-"]
    stats = goal_stats(names, train, goals)

    def val(r, i):
        return num(r[i]) if 0 <= i < len(r) else None

    def d2h(r):
        tot = 0.0
        for n, h in goals:
            v = val(r, names.index(n))
            tot += 1.0 if v is None else (logi(v, *stats[n]) - h) ** 2
        return sqrt(tot / len(goals))

    ds = [d2h(r) for r in train]
    order = sorted(range(len(train)), key=lambda i: (ds[i], i))
    k = max(1, len(order) // 4)
    good = [train[i] for i in order[:k]]
    poor = [train[i] for i in order[-k:]]

    xs = [(i, n) for i, n in enumerate(names)
          if n[-1] not in "X+-" and n in tnames]
    cstat, prof = {}, {}
    for i, n in xs:
        if n[0].isupper():
            vs = [v for v in (val(r, i) for r in train) if v is not None]
            c = len(vs)
            mu = sum(vs) / c if c else 0.0
            sd = sqrt(sum((v - mu) ** 2 for v in vs) / (c - 1)) if c > 1 else 0.0
            cstat[i] = (mu, sd)

            def center(g, i=i):
                z = [logi(v, *cstat[i]) for v in (val(r, i) for r in g)
                     if v is not None]
                return sum(z) / len(z) if z else None
            prof[i] = (center(good), center(poor))
        else:
            def freq(g, i=i):
                vs = [r[i] for r in g if i < len(r) and r[i] not in ("?", "")]
                d = {}
                for v in vs:
                    d[v] = d.get(v, 0) + 1
                return {v: c / len(vs) for v, c in d.items()}
            prof[i] = (freq(good), freq(poor))

    out = []
    for r in test:
        goalgaps = {}
        for n, h in goals:
            v = val(r, tnames.index(n))
            goalgaps[n] = 1.0 if v is None else abs(logi(v, *stats[n]) - h)
        feats = {}
        for i, n in xs:
            j = tnames.index(n)
            s = 0.0
            if j < len(r) and r[j] not in ("?", ""):
                g, p = prof[i]
                if n[0].isupper():
                    v = num(r[j])
                    if v is not None and g is not None and p is not None:
                        x = logi(v, *cstat[i])
                        s = max(0.0, abs(x - g) - abs(x - p))
                else:
                    s = max(0.0, p.get(r[j], 0.0) - g.get(r[j], 0.0))
            feats[n] = s
        out.append({
            "goals": goalgaps,
            "features": feats,
            "goal_order": sorted(goalgaps, key=lambda n: (-goalgaps[n], n)),
            "feature_order": sorted(feats, key=lambda n: (-feats[n], n)),
        })
    print(json.dumps({"diagnoses": out}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["remember", "guess", "certify", "flow", "blame"])
    ap.add_argument("--train", required=False)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    if a.cmd == "certify":
        certify(a)
        return
    if not a.train:
        ap.error("--train is required")
    {"remember": remember, "guess": guess, "flow": flow,
     "blame": blame}[a.cmd](a)


if __name__ == "__main__":
    main()
