import sys, json, argparse
from bisect import bisect_left, bisect_right
from math import exp, sqrt

COHEN_T, CLIFFS_T, KS_T = 0.35, 0.197, 1.36
WINDOW = 20
BUDGET, PICKS = 10, 3
FIX_K, FIX_CONFIGS, FIX_MIN_GAIN = 5, 12, 0.05
FIX_COLS = ("Clndrs", "Volume")
SPOT_K, SPOT_Q = 5, 0.90


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


def model(a, K=7, rnd=False):
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
            d = sqrt(s / c) if c else 1.0
            ds.append((round(d, 12) if rnd else d, k))
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

def diagnose(a):
    "Return (diagnoses, training d2h list) for every test row."
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
    return out, ds


def blame(a):
    out, _ = diagnose(a)
    print(json.dumps({"diagnoses": out}))


# ---- justify: explain recommendations and acceptance ----

def median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def justify(a):
    diag, hist = diagnose(a)
    cutoff = median(hist)
    n = len(diag)
    scores = []
    for d in diag:
        g = d["goals"]
        scores.append(sqrt(sum(x * x for x in g.values()) / len(g)) if g else 0.0)
    out = []
    for i in range(n):
        j = (i + 1) % n
        s, o = scores[i], scores[j]
        acc = bool(s <= cutoff)
        weakest = diag[i]["goal_order"][0] if diag[i]["goal_order"] else None
        fo = diag[i]["feature_order"]
        top = fo[0] if fo and diag[i]["features"][fo[0]] > 0 else None
        tie = round(s, 12) == round(o, 12)
        if tie:
            win, adv = None, None
        else:
            win = i if s < o else j
            lose = j if win == i else i
            gw, gl = diag[win]["goals"], diag[lose]["goals"]
            adv = sorted(gw, key=lambda k: (-(gl[k] - gw[k]), k))[0]
        verdict = "accepted" if acc else "rejected"
        t = (f"Car {i} has quality score {s:.6f} versus cutoff {cutoff:.6f} "
             f"(median training score), so it is {verdict}. ")
        if tie:
            t += f"Car {i} and car {j} tie ({s:.6f} vs {o:.6f}); no recommendation. "
        else:
            t += (f"Car {win} is recommended over car {j if win == i else i} "
                  f"(lower distance to heaven: {min(s, o):.6f} vs {max(s, o):.6f}); "
                  f"its biggest advantage is {adv}. ")
        t += f"Car {i}'s weakest goal is {weakest}"
        if top:
            t += f"; most associated feature with poor cars: {top}."
        else:
            t += "; no feature shows positive association with poor cars."
        out.append({"row": i, "other": j, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": acc, "winner": win,
                    "advantage_goal": adv, "weakest_goal": weakest,
                    "top_feature": top, "text": t})
    print(json.dumps({"explanations": out}))


# ---- choose: few inspections, best cars ----

def parse_goals(resp, names, goals):
    "Extract observed goal values (in goal order) from an evaluator reply."
    gn = [n for n, _ in goals]
    if isinstance(resp, dict):
        src = resp
        for k in ("goals", "observed", "y", "values"):
            if k in resp:
                src = resp[k]
                break
        if isinstance(src, dict):
            vals = []
            for n in gn:
                v = src.get(n, src.get(n[:-1], src.get(n[:-1].lower())))
                vals.append(num(str(v)) if v is not None else None)
            return vals
        resp = src
    if isinstance(resp, list):
        if len(resp) == len(names):
            return [num(str(resp[names.index(n)])) for n in gn]
        if len(resp) == len(gn):
            return [num(str(v)) for v in resp]
    return [None] * len(gn)


def choose(a):
    names, train = load(a.train)
    tnames, test = load(a.test)
    _, predict, _ = model(a)
    goals = [(n, 0.0 if n[-1] == "-" else 1.0) for n in names if n[-1] in "+-"]
    stats = goal_stats(names, train, goals)
    pred = [predict(r)[1] for r in test]

    # test-row feature space (inputs only), scaled by test spread
    cols = [j for j, n in enumerate(tnames) if n[-1] not in "X+-"]
    sc = {}
    for j in cols:
        if tnames[j][0].isupper():
            vs = [v for v in (num(r[j]) for r in test if j < len(r)) if v is not None]
            mu = sum(vs) / len(vs) if vs else 0.0
            sd = sqrt(sum((v - mu) ** 2 for v in vs) / max(1, len(vs) - 1)) if vs else 1.0
            sc[j] = (mu, sd or 1.0)

    def gap(r, s):
        t = c = 0
        for j in cols:
            x, y = (r[j] if j < len(r) else "?"), (s[j] if j < len(s) else "?")
            if x in ("?", "") or y in ("?", ""):
                continue
            if j in sc:
                p, q = num(x), num(y)
                if p is None or q is None:
                    continue
                g = (p - q) / sc[j][1]
            else:
                g = float(x != y)
            t += g * g
            c += 1
        return sqrt(t / c) if c else 1.0

    obs = {}
    n = len(test)
    for _ in range(min(BUDGET, n)):
        if obs:
            near = {}
            for i in range(n):
                if i not in obs:
                    near[i] = sorted((gap(test[i], test[k]), k) for k in obs)[:3]
        best, bs = None, None
        for i in range(n):
            if i in obs:
                continue
            adj = pred[i]
            if obs:
                w = [(1 / (d + 0.05), obs[k] - pred[k]) for d, k in near[i]]
                adj += sum(x * y for x, y in w) / sum(x for x, _ in w)
            if bs is None or adj < bs:
                best, bs = i, adj
        print(json.dumps({"inspect": best}))
        sys.stdout.flush()
        line = sys.stdin.readline()
        try:
            vals = parse_goals(json.loads(line), names, goals)
        except Exception:
            vals = [None] * len(goals)
        tot = 0.0
        for (gname, h), v in zip(goals, vals):
            tot += 1.0 if v is None else (logi(v, *stats[gname]) - h) ** 2
        obs[best] = sqrt(tot / len(goals))
    sel = sorted(obs, key=lambda i: (obs[i], i))[:PICKS]
    print(json.dumps({"selected": sel}))
    sys.stdout.flush()


# ---- fix: smallest hypothetical change to improve a poor car ----

def fix(a):
    names, train = load(a.train)
    tnames, test = load(a.test)
    _, predict, hist = model(a, FIX_K, True)
    cutoff = median(hist)
    order = sorted(range(len(train)), key=lambda i: (hist[i], i))
    best = order[:max(1, len(order) // 4)]
    fcols = [c for c in FIX_COLS if c in names and c in tnames]
    tidx = {c: tnames.index(c) for c in fcols}
    nidx = {c: names.index(c) for c in fcols}

    # distinct engine configurations from the best 25%, in rank order
    configs, seen = [], set()
    for i in best:
        r = train[i]
        cfg = tuple(r[nidx[c]] if nidx[c] < len(r) else "?" for c in fcols)
        if "?" in cfg or cfg in seen:
            continue
        seen.add(cfg)
        configs.append(cfg)
        if len(configs) >= FIX_CONFIGS:
            break

    st = {}
    for c in fcols:
        vs = [v for v in (num(r[nidx[c]]) for r in train if nidx[c] < len(r))
              if v is not None]
        k = len(vs)
        mu = sum(vs) / k if k else 0.0
        sd = sqrt(sum((v - mu) ** 2 for v in vs) / (k - 1)) if k > 1 else 0.0
        st[c] = (mu, sd)

    def conv(s):
        v = num(s)
        if v is None:
            return None if s in ("?", "") else s
        return int(v) if v == int(v) else v

    inputs = [(j, n) for j, n in enumerate(tnames) if n[-1] not in "X+-"]

    def snap(r):
        return {n: conv(r[j]) if j < len(r) else None for j, n in inputs}

    def same(x, y):
        p, q = num(x), num(y)
        return p == q if p is not None and q is not None else x == y

    def ndist(r, cfg, changed):
        t = 0.0
        for c, new in changed:
            old = r[tidx[c]] if tidx[c] < len(r) else "?"
            p, q = num(old), num(new)
            if p is None or q is None:
                g = 1.0
            else:
                g = abs(logi(p, *st[c]) - logi(q, *st[c]))
            t += g * g
        return sqrt(t / max(1, len(fcols)))

    out = []
    for ri, r in enumerate(test):
        r = list(r)
        s0 = predict(r)[1]
        before = snap(r)
        if s0 <= cutoff:
            out.append({"row": ri, "before": before, "after": before,
                        "changes": [], "status": "already_good",
                        "estimated_improvement": 0.0, "n_changes": 0})
            continue
        pick = None
        for cfg in configs:
            changed = []
            for c, new in zip(fcols, cfg):
                old = r[tidx[c]] if tidx[c] < len(r) else "?"
                if not same(old, new):
                    changed.append((c, new))
            if not changed:
                continue
            r2 = list(r)
            for c, new in changed:
                while len(r2) <= tidx[c]:
                    r2.append("?")
                r2[tidx[c]] = new
            s1 = predict(r2)[1]
            if s0 - s1 < FIX_MIN_GAIN:
                continue
            key = (len(changed), ndist(r, cfg, changed), s1)
            if pick is None or key < pick[0]:
                pick = (key, r2, changed, s1)
        if pick is None:
            out.append({"row": ri, "before": before, "after": before,
                        "changes": [], "status": "no_candidate",
                        "estimated_improvement": 0.0, "n_changes": 0})
            continue
        _, r2, changed, s1 = pick
        out.append({"row": ri, "before": before, "after": snap(r2),
                    "changes": [{"column": c, "from": before[c], "to": conv(new)}
                                for c, new in changed],
                    "status": "improved",
                    "estimated_improvement": s0 - s1,
                    "n_changes": len(changed)})
    print(json.dumps({"repairs": out}))


# ---- spot: anomaly detection on input characteristics ----

def spot(a):
    names, train = load(a.train)
    tnames, test = load(a.test)
    # inputs only: no ignored (X) columns, no goals
    xs = [(i, tnames.index(n), n[0].isupper()) for i, n in enumerate(names)
          if n[-1] not in "X+-" and n in tnames]

    def cell(r, j):
        return r[j] if j < len(r) and r[j] not in ("?", "") else None

    stat, freq = {}, {}
    for i, _, isn in xs:
        vs = [cell(r, i) for r in train]
        vs = [v for v in vs if v is not None]
        if isn:
            f = [v for v in (num(v) for v in vs) if v is not None]
            k = len(f)
            mu = sum(f) / k if k else 0.0
            sd = sqrt(sum((v - mu) ** 2 for v in f) / (k - 1)) if k > 1 else 0.0
            stat[i] = (mu, sd or 1.0)
        else:
            d = {}
            for v in vs:
                d[v] = d.get(v, 0) + 1
            freq[i] = d

    def prep(r, pos):
        out = []
        for (i, j, isn) in xs:
            v = cell(r, i if pos == 0 else j)
            if v is not None and isn:
                v = num(v)
                v = None if v is None else (v - stat[i][0]) / stat[i][1]
            out.append(v)
        return out

    def gapd(p, q):
        s = c = 0
        for (_, _, isn), x, y in zip(xs, p, q):
            if x is None or y is None:
                continue
            g = (x - y) if isn else float(x != y)
            s += g * g
            c += 1
        return sqrt(s / c) if c else 1.0

    def marginal(q):
        m = 0.0
        for (i, _, isn), x in zip(xs, q):
            if x is None:
                continue
            m = max(m, abs(x) if isn else (4.0 if freq[i].get(x, 0) == 0 else 0.0))
        return m

    trq = [prep(r, 0) for r in train]

    def raw(q, skip=None):
        ds = sorted(gapd(q, t) for k, t in enumerate(trq) if k != skip)[:SPOT_K]
        knn = sum(ds) / len(ds) if ds else 0.0
        return knn + 0.5 * marginal(q)

    tr_raw = [raw(q, k) for k, q in enumerate(trq)]
    scale = median(tr_raw) or 1.0

    def squash(v):
        return v / (v + scale)

    tr_s = sorted(squash(v) for v in tr_raw)
    thr = tr_s[min(len(tr_s) - 1, int(SPOT_Q * len(tr_s)))] if tr_s else 0.5
    scores = [squash(raw(prep(r, 1))) for r in test]
    print(json.dumps({"scores": scores,
                      "alerts": [s > thr for s in scores],
                      "threshold": thr}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["remember", "guess", "certify", "flow",
                                    "blame", "justify", "choose", "fix",
                                    "spot"])
    ap.add_argument("--train", required=False)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    if a.cmd == "certify":
        certify(a)
        return
    if not a.train:
        ap.error("--train is required")
    {"remember": remember, "guess": guess, "flow": flow,
     "blame": blame, "justify": justify, "choose": choose,
     "fix": fix, "spot": spot}[a.cmd](a)


if __name__ == "__main__":
    main()
