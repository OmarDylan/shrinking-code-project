import csv, sys, json, argparse
from bisect import bisect_left, bisect_right
from math import exp, sqrt


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return rows[0], rows[1:]

def get(r, i):
    return r[i] if i < len(r) else "?"

def norm(mu, sd, x):
    z = max(-3.0, min(3.0, (x - mu) / (sd + 1e-32)))
    return 1 / (1 + exp(-1.702 * z))

def goal_stats(names, tr):
    goals = [(i, 0 if n[-1] == "-" else 1) for i, n in enumerate(names) if n[-1] in "+-"]
    stats = {}
    for i, _ in goals:
        v = [float(r[i]) for r in tr if get(r, i) != "?"]
        mu = sum(v) / len(v)
        sd = sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        stats[i] = (mu, sd)
    return goals, stats

def make_d2h(goals, stats):
    def nrm(i, x):
        mu, sd = stats[i]
        return norm(mu, sd, x)
    return lambda vals: sqrt(sum((nrm(i, vals[i]) - h) ** 2 for i, h in goals) / len(goals))

def remember(names, tr, te):
    goals, stats = goal_stats(names, tr)
    d2h = make_d2h(goals, stats)
    scores = [d2h({i: float(r[i]) for i, _ in goals}) for r in te]
    order = sorted(range(len(scores)), key=lambda k: (scores[k], k))
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[len(order) // 2], "worst": order[-1]}

def guess(names, tr, te, k=7):
    goals, stats = goal_stats(names, tr)
    d2h = make_d2h(goals, stats)
    gi = {i for i, _ in goals}
    mpg_i = next(i for i, n in enumerate(names) if n.startswith("Mpg"))
    xs = [i for i, n in enumerate(names) if n[-1] not in "+-X" and i not in gi]
    tr = [r for r in tr if all(get(r, i) != "?" for i in gi)]
    sds = {}
    for i in xs:
        if names[i][0].isupper():
            v = [float(r[i]) for r in tr if get(r, i) != "?"]
            mu = sum(v) / len(v)
            sds[i] = sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)) or 1.0
    ty = [(float(r[mpg_i]), d2h({i: float(r[i]) for i in gi})) for r in tr]
    mm = sum(y[0] for y in ty) / len(ty)
    ms = sum(y[1] for y in ty) / len(ty)

    def dist(a, b):
        s = n = 0
        for i in xs:
            p, q = get(a, i), get(b, i)
            if p == "?" or q == "?":
                continue
            s += ((float(p) - float(q)) / sds[i]) ** 2 if i in sds else float(p != q)
            n += 1
        return sqrt(s / n) if n else 1.0

    mpg, scores = [], []
    for r in te:
        near = sorted(range(len(tr)), key=lambda j: dist(r, tr[j]))[:k]
        w = [1 / (dist(r, tr[j]) + 0.05) for j in near]
        sw = sum(w)
        m = sum(wi * ty[j][0] for wi, j in zip(w, near)) / sw
        s = sum(wi * ty[j][1] for wi, j in zip(w, near)) / sw
        mpg.append(m if m > 0 else mm)
        scores.append(min(1.0, max(0.0, s)) if s == s else ms)
    return {"mpg": mpg, "scores": scores}

def compare(xs, ys):
    xs, ys = [float(x) for x in xs], [float(y) for y in ys]
    n, m = len(xs), len(ys)
    mx, my = sum(xs) / n, sum(ys) / m
    vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    vy = sum((y - my) ** 2 for y in ys) / (m - 1)
    pooled = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
    cohen = abs(mx - my) / (sqrt(pooled) + 1e-32)
    sx, sy = sorted(xs), sorted(ys)
    tot = 0
    for x in sx:
        tot += bisect_left(sy, x) - (m - bisect_right(sy, x))
    cliffs = abs(tot) / (n * m)
    d = 0.0
    for v in set(sx) | set(sy):
        d = max(d, abs(bisect_right(sx, v) / n - bisect_right(sy, v) / m))
    ks = d / sqrt(1 / n + 1 / m)
    return {"cohen": cohen, "cliffs": cliffs, "ks": ks,
            "same": bool(cohen <= 0.35 and cliffs <= 0.197 and ks <= 1.36)}

def certify(cases):
    return {"comparisons": [compare(c["xs"], c["ys"]) for c in cases]}

def flow(names, tr, te, win=20):
    g = guess(names, tr, te)
    goals, stats = goal_stats(names, tr)
    d2h = make_d2h(goals, stats)
    hist = [d2h({i: float(r[i]) for i, _ in goals}) for r in tr
            if all(get(r, i) != "?" for i, _ in goals)]
    sc = g["scores"]
    alerts = []
    for t in range(len(sc)):
        if t < win - 1:
            alerts.append(False)
        else:
            alerts.append(not compare(hist, sc[t - win + 1:t + 1])["same"])
    return {"mpg": g["mpg"], "scores": sc, "alerts": alerts}

def blame(names, tr, te):
    goals, stats = goal_stats(names, tr)
    d2h = make_d2h(goals, stats)
    xs = [i for i, n in enumerate(names) if n[-1] not in "+-X"]
    full = [r for r in tr if all(get(r, i) != "?" for i, _ in goals)]
    order = sorted(range(len(full)),
                   key=lambda j: (round(d2h({i: float(full[j][i]) for i, _ in goals}), 12), j))
    q = max(1, len(full) // 4)
    good = [full[j] for j in order[:q]]
    poor = [full[j] for j in order[-q:]]

    info = {}
    for i in xs:
        if names[i][0].isupper():
            v = [float(r[i]) for r in tr if get(r, i) != "?"]
            if not v:
                info[i] = None
                continue
            mu = sum(v) / len(v)
            sd = sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
            gv = [float(r[i]) for r in good if get(r, i) != "?"]
            pv = [float(r[i]) for r in poor if get(r, i) != "?"]
            gc = norm(mu, sd, sum(gv) / len(gv)) if gv else None
            pc = norm(mu, sd, sum(pv) / len(pv)) if pv else None
            info[i] = (mu, sd, gc, pc)
        else:
            gv = [get(r, i) for r in good if get(r, i) != "?"]
            pv = [get(r, i) for r in poor if get(r, i) != "?"]
            info[i] = (gv, pv)

    out = []
    for r in te:
        feats = {}
        for i in xs:
            x = get(r, i)
            ev = 0.0
            if x != "?" and info[i] is not None:
                if names[i][0].isupper():
                    mu, sd, gc, pc = info[i]
                    if gc is not None and pc is not None:
                        nx = norm(mu, sd, float(x))
                        ev = max(0.0, abs(nx - gc) - abs(nx - pc))
                else:
                    gv, pv = info[i]
                    gf = gv.count(x) / len(gv) if gv else 0.0
                    pf = pv.count(x) / len(pv) if pv else 0.0
                    ev = max(0.0, pf - gf)
            feats[names[i]] = ev
        gl = {}
        for i, h in goals:
            x = get(r, i)
            if x == "?":
                gl[names[i]] = 0.0
            else:
                mu, sd = stats[i]
                gl[names[i]] = abs(norm(mu, sd, float(x)) - h)
        key = lambda d: sorted(d, key=lambda n: (-round(d[n], 10), n))
        out.append({"goals": gl, "features": feats,
                    "goal_order": key(gl), "feature_order": key(feats)})
    return {"diagnoses": out}

def justify(names, tr, te):
    goals, stats = goal_stats(names, tr)
    d2h = make_d2h(goals, stats)
    sc = remember(names, tr, te)["scores"]
    full = [r for r in tr if all(get(r, i) != "?" for i, _ in goals)]
    ts = sorted(d2h({i: float(r[i]) for i, _ in goals}) for r in full)
    cutoff = ts[len(full) // 2]
    dg = blame(names, tr, te)["diagnoses"]
    n = len(te)
    out = []
    for j in range(n):
        k = (j + 1) % n
        s, o = sc[j], sc[k]
        acc = round(s, 12) <= round(cutoff, 12)
        winner = None
        adv = None
        if round(s, 12) != round(o, 12):
            winner = j if s < o else k
            loser = k if winner == j else j
            wg, lg = dg[winner]["goals"], dg[loser]["goals"]
            a = {g: lg[g] ** 2 - wg[g] ** 2 for g in wg}
            adv = sorted(a, key=lambda g: (-round(a[g], 10), g))[0]
        weak = dg[j]["goal_order"][0]
        fo = dg[j]["feature_order"]
        top = fo[0] if fo and dg[j]["features"][fo[0]] > 1e-10 else None
        text = (f"Car {j} {'accepted' if acc else 'rejected'}: score {s:.3f}, "
                f"other score {o:.3f}, cutoff {cutoff:.3f}; weakest goal {weak}. ")
        if winner is not None:
            text += f"Recommend car {winner} because of {adv}. "
        else:
            text += "Scores tie. "
        if top is not None:
            text += f"Top feature {top} (retrospective, not causal)."
        out.append({"row": j, "other": k, "score": s, "other_score": o,
                    "cutoff": cutoff, "accepted": bool(acc), "winner": winner,
                    "advantage_goal": adv, "weakest_goal": weak,
                    "top_feature": top, "text": text.strip()})
    return {"explanations": out}

def choose(names, tr, te, budget=10, exploit=7):
    goals, stats = goal_stats(names, tr)
    d2h = make_d2h(goals, stats)
    n = len(te)
    budget = min(budget, n)
    pred = guess(names, tr, te)["scores"]
    rank = sorted(range(n), key=lambda j: (pred[j], j))
    picks = rank[:min(exploit, budget)]
    rest = rank[len(picks):]
    need = budget - len(picks)
    if need > 0 and rest:
        pool = rest[:max(need, 30)]
        step = len(pool) / need
        for t in range(need):
            c = pool[int(t * step)]
            picks.append(c)
    seen = []
    for c in picks:
        if c not in seen:
            seen.append(c)
    for c in rank:
        if len(seen) >= budget:
            break
        if c not in seen:
            seen.append(c)
    obs = {}
    for idx in seen[:budget]:
        print(json.dumps({"inspect": idx}))
        sys.stdout.flush()
        line = sys.stdin.readline()
        resp = json.loads(line)
        g = resp["goals"]
        obs[idx] = d2h({i: float(g[names[i]]) for i, _ in goals})
    top = sorted(obs, key=lambda j: (round(obs[j], 12), j))[:3]
    return {"selected": top}

def fix(names, tr, te, k=5, menu_n=12, min_gain=0.05):
    goals, stats = goal_stats(names, tr)
    d2h = make_d2h(goals, stats)
    full = [r for r in tr if all(get(r, i) != "?" for i, _ in goals)]
    fn = ["Clndrs", "Volume", "Model"]
    ci = [names.index(n) for n in fn]
    oi = names.index("origin")
    ns = []
    for i in ci:
        v = [float(r[i]) for r in tr if get(r, i) != "?"]
        mu = sum(v) / len(v)
        sd = sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        ns.append((mu, sd))

    def vec(c, v, m, o):
        out = []
        for x, (mu, sd) in zip((c, v, m), ns):
            out.append(None if x is None or x == "?" else norm(mu, sd, float(x)))
        out.append(o)
        return out

    def dist(a, b):
        s = 0.0
        for p, q in zip(a[:3], b[:3]):
            s += 1.0 if p is None or q is None else (p - q) ** 2
        s += 1.0 if (a[3] == "?" or b[3] == "?" or a[3] != b[3]) else 0.0
        return sqrt(s / 4)

    tv = [vec(*[get(r, i) for i in ci], get(r, oi)) for r in full]
    td = [d2h({i: float(r[i]) for i, _ in goals}) for r in full]

    def pred(v):
        o = sorted(range(len(tv)), key=lambda j: (round(dist(v, tv[j]), 12), j))[:k]
        return sum(td[j] for j in o) / len(o)

    cutoff = sorted(td)[len(td) // 2]
    rk = sorted(range(len(full)), key=lambda j: (round(td[j], 12), j))
    good = rk[:max(1, len(full) // 4)]
    menu = []
    for j in good:
        p = (float(full[j][ci[0]]), float(full[j][ci[1]]))
        if p not in menu:
            menu.append(p)
        if len(menu) >= menu_n:
            break

    out = []
    for idx, r in enumerate(te):
        c, v, m = (get(r, i) for i in ci)
        o = get(r, oi)
        v0 = vec(c, v, m, o)
        before = pred(v0)
        res = {"row": idx, "before": before, "after": before,
               "changes": {}, "status": "already_good"}
        if round(before, 10) > round(cutoff, 10):
            res["status"] = "no_candidate"
            best = None
            cf = float(c) if c != "?" else None
            vf = float(v) if v != "?" else None
            for nc, nv in menu:
                ch = {}
                if nc != cf:
                    ch[fn[0]] = nc
                if nv != vf:
                    ch[fn[1]] = nv
                if not ch:
                    continue
                v1 = vec(nc, nv, m, o)
                after = pred(v1)
                if round(before - after, 10) >= min_gain:
                    key = (len(ch), round(dist(v0, v1), 10), round(after, 10), (nc, nv))
                    if best is None or key < best[0]:
                        best = (key, after, ch)
            if best:
                res.update(after=best[1], changes=best[2], status="improved")
        out.append(res)
    return {"repairs": out}

def spot(names, tr, te, k=5, q=0.95):
    """Anomaly score: mean kNN distance in standardized input space,
    squashed to [0,1] by raw/(raw+t), t = training LOO q-quantile.
    Threshold is therefore 0.5 (= the training q-quantile)."""
    num = [names.index(n) for n in ("Clndrs", "Volume", "Model")]
    cat = [names.index("origin")]
    st = {}
    for i in num:
        v = [float(r[i]) for r in tr if get(r, i) != "?"]
        mu = sum(v) / len(v)
        sd = sqrt(sum((x - mu) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
        st[i] = (mu, sd or 1.0)

    def vec(r):
        out = []
        for i in num:
            x = get(r, i)
            try:
                out.append(None if x == "?" else (float(x) - st[i][0]) / st[i][1])
            except ValueError:
                out.append(None)
        for i in cat:
            x = get(r, i)
            out.append(None if x == "?" else x)
        return out

    def dist(a, b):
        s = n = 0
        for p, qv in zip(a, b):
            if p is None or qv is None:
                continue
            s += (p - qv) ** 2 if isinstance(p, float) else float(p != qv)
            n += 1
        return sqrt(s / n) if n else 0.0

    tv = [vec(r) for r in tr]

    def raw(v, skip=None):
        d = sorted(dist(v, tv[j]) for j in range(len(tv)) if j != skip)[:k]
        return sum(d) / len(d) if d else 0.0

    base = sorted(raw(tv[j], j) for j in range(len(tv)))
    t = base[min(len(base) - 1, int(q * len(base)))] if base else 1.0
    t = max(t, 1e-12)
    scores = []
    for r in te:
        x = raw(vec(r))
        s = x / (x + t)
        scores.append(min(1.0, max(0.0, s)))
    thr = 0.5
    return {"scores": scores, "alerts": [s > thr for s in scores], "threshold": thr}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    if a.cmd == "certify":
        with open(a.test, encoding="utf-8-sig") as f:
            out = certify(json.load(f))
    elif a.cmd in ("remember", "guess", "flow", "blame", "justify", "choose", "fix", "spot"):
        names, tr = load(a.train)
        _, te = load(a.test)
        if a.cmd == "remember":
            out = remember(names, tr, te)
        elif a.cmd == "guess":
            out = guess(names, tr, te)
        elif a.cmd == "blame":
            out = blame(names, tr, te)
        elif a.cmd == "justify":
            out = justify(names, tr, te)
        elif a.cmd == "choose":
            out = choose(names, tr, te)
        elif a.cmd == "fix":
            out = fix(names, tr, te)
        elif a.cmd == "spot":
            out = spot(names, tr, te)
        else:
            out = flow(names, tr, te)
    else:
        sys.exit(1)
    print(json.dumps(out))
    sys.stdout.flush()

main()
