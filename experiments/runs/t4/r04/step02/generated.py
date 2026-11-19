import csv, sys, math, argparse, json

G = [("Lbs-", 0), ("Acc+", 1), ("Mpg+", 1)]
F = ["Clndrs", "Volume", "Model"]


def read(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return head, rows


def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def load(path):
    head, rows = read(path)
    idx = [head.index(g) for g, _ in G]
    out = []
    for row in rows:
        vals = [num(row[i]) for i in idx]
        out.append(None if None in vals else vals)
    return out


def get_stats(tr):
    n = len(tr)
    stats = []
    for j in range(3):
        col = [r[j] for r in tr]
        m = sum(col) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    return stats


def d2h(r, stats):
    s = 0.0
    for j, (_, h) in enumerate(G):
        m, sd = stats[j]
        z = max(-3.0, min(3.0, (r[j] - m) / (sd + 1e-32)))
        s += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
    return math.sqrt(s / 3)


def remember(a):
    tr = [r for r in load(a.train) if r is not None]
    te = load(a.test)
    stats = get_stats(tr)
    scores = [d2h(r, stats) for r in te]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }


def guess(a):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    th, trows = read(a.train)
    eh, erows = read(a.test)
    gi = [th.index(g) for g, _ in G]
    ti = [th.index(c) for c in F]
    to = th.index("origin")
    ei = [eh.index(c) for c in F]
    eo = eh.index("origin")

    keep = [r for r in trows if all(num(r[i]) is not None for i in gi)]
    goals = [[float(r[i]) for i in gi] for r in keep]
    stats = get_stats(goals)
    y_mpg = np.array([g[2] for g in goals])
    y_d = np.array([d2h(g, stats) for g in goals])

    cats = sorted({r[to] for r in keep})
    meds = []
    for k in range(3):
        v = [num(r[ti[k]]) for r in keep if num(r[ti[k]]) is not None]
        meds.append(float(np.median(v)) if v else 0.0)

    def feats(row, idxs, oi, use_vol):
        x = []
        for k in range(3):
            if k == 1 and not use_vol:
                continue
            v = num(row[idxs[k]])
            x.append(meds[k] if v is None else v)
        x += [1.0 if row[oi] == c else 0.0 for c in cats]
        return x

    def fit_predict(use_vol, Xte, y):
        Xtr = np.array([feats(r, ti, to, use_vol) for r in keep])
        ps = []
        for m in (RandomForestRegressor(300, min_samples_leaf=2, random_state=0),
                  GradientBoostingRegressor(random_state=0, n_estimators=150,
                                            max_depth=2, learning_rate=0.05)):
            m.fit(Xtr, y)
            ps.append(m.predict(Xte))
        return np.mean(ps, axis=0)

    n = len(erows)
    mpg = [0.0] * n
    sc = [0.0] * n
    for use_vol in (True, False):
        ids = [i for i in range(n)
               if (num(erows[i][ei[1]]) is not None) == use_vol]
        if not ids:
            continue
        X = np.array([feats(erows[i], ei, eo, use_vol) for i in ids])
        pm = fit_predict(use_vol, X, y_mpg)
        pd = fit_predict(use_vol, X, y_d)
        for k, i in enumerate(ids):
            mpg[i] = max(float(pm[k]), 1.0)
            sc[i] = min(1.0, max(0.0, float(pd[k])))
    return {"mpg": mpg, "scores": sc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    if a.cmd == "guess":
        out = guess(a)
    else:
        out = remember(a)
    print(json.dumps(out))


main()
