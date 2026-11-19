import sys, json, csv
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor

BUDGET, SENT = 10, 11


def parse_header(h):
    cols = []
    for i, n in enumerate(h):
        if n.endswith("X"):
            continue
        cols.append(dict(i=i, name=n, num=n[0].isupper(),
                         goal=n[-1] if n[-1] in "+-" else None))
    return cols


def tofloat(v):
    try:
        if v is None or str(v).strip() == "?" or v == "":
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def rowvals(r, header):
    if isinstance(r, dict):
        return [r.get(n) for n in header]
    return list(r)


def cliff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.sign(a[:, None] - b[None, :])
    return float(d.mean())


def boot_p(a, b, n=5000, seed=1):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    cnt = 0
    for _ in range(n):
        s = rng.choice(pool, len(pool), replace=True)
        if abs(s[:len(a)].mean() - s[len(a):].mean()) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n + 1)


def compare(ad, rd):
    try:
        mw = float(mannwhitneyu(ad, rd, alternative="two-sided").pvalue)
    except ValueError:
        mw = 1.0
    cd = cliff(ad, rd)
    bp = boot_p(ad, rd)
    return {
        "adaptive_mean": float(np.mean(ad)), "random_mean": float(np.mean(rd)),
        "adaptive_median": float(np.median(ad)), "random_median": float(np.median(rd)),
        "mann_whitney_p": mw,
        "cliffs_delta": cd,
        "cliffs_delta_negligible": abs(cd) < 0.147,
        "bootstrap_p": bp,
        "adaptive_better": bool(np.mean(ad) < np.mean(rd) and mw < 0.05 and abs(cd) >= 0.147 and bp < 0.05),
        "lower_is_better": True,
    }


def main():
    a = sys.argv
    train_path = a[a.index("--train") + 1]
    test_path = a[a.index("--test") + 1]
    with open(train_path, newline="") as f:
        rd = list(csv.reader(f))
    header, trows = rd[0], [r for r in rd[1:] if r]
    cols = parse_header(header)
    goals = [c for c in cols if c["goal"]]
    feats = [c for c in cols if not c["goal"]]
    G = np.array([[tofloat(r[c["i"]]) for c in goals] for r in trows])
    lo, hi = np.nanmin(G, 0), np.nanmax(G, 0)
    span = np.where(hi - lo == 0, 1, hi - lo)
    tgt = np.array([1.0 if c["goal"] == "+" else 0.0 for c in goals])

    def d2h(g):
        g = np.asarray(g, float)
        n = (g - lo) / span
        return float(np.sqrt(np.nanmean((tgt - n) ** 2)))

    ytr = np.array([d2h(g) for g in G])
    target = float(np.percentile(ytr, 25))

    with open(test_path) as f:
        J = json.load(f)
    th = J.get("header", header)
    trows_t = [rowvals(r, th) for r in J["rows"]]
    tidx = {n: i for i, n in enumerate(th)}

    cats = {}
    for c in feats:
        if not c["num"]:
            s = {str(r[c["i"]]) for r in trows}
            s |= {str(r[tidx[c["name"]]]) for r in trows_t if c["name"] in tidx}
            cats[c["name"]] = sorted(s)

    def feat(vals_of):
        out = []
        for c in feats:
            v = vals_of(c)
            if c["num"]:
                out.append(tofloat(v))
            else:
                out.extend(1.0 if str(v) == k else 0.0 for k in cats[c["name"]])
        return out

    Xtr = np.array([feat(lambda c, r=r: r[c["i"]]) for r in trows], float)
    Xte = np.array([feat(lambda c, r=r: r[tidx[c["name"]]] if c["name"] in tidx else None)
                    for r in trows_t], float)
    med = np.nanmedian(Xtr, 0)
    med = np.where(np.isnan(med), 0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)

    def inspect(ep, strat, idx):
        print(json.dumps({"episode": ep, "strategy": strat, "inspect": int(idx)}), flush=True)
        line = sys.stdin.readline()
        r = json.loads(line)
        g = r.get("goals", r.get("observed", r.get("y")))
        if g is None:
            g = r
        if isinstance(g, dict):
            vals = []
            for c in goals:
                n = c["name"]
                v = g.get(n, g.get(n[:-1]))
                vals.append(tofloat(v))
        else:
            vals = [tofloat(v) for v in g]
        return d2h(vals)

    def metrics(seq):
        best, b5, b10, hit = np.inf, None, None, SENT
        for k, q in enumerate(seq, 1):
            best = min(best, q)
            if hit == SENT and q <= target:
                hit = k
            if k == 5:
                b5 = best
        b10 = best
        if b5 is None:
            b5 = best
        return float(b5), float(b10), hit

    results, M = [], {"adaptive": [], "random": []}
    for ep_i, ep in enumerate(J["episodes"]):
        en = ep.get("episode", ep_i)
        pool = list(ep["pool"])
        # adaptive
        seen, obsX, obsY, seq = [], [], [], []
        for _ in range(min(BUDGET, len(pool))):
            X = np.vstack([Xtr] + ([np.array(obsX)] if obsX else []))
            y = np.concatenate([ytr, obsY])
            m = RandomForestRegressor(n_estimators=40, random_state=0, n_jobs=1).fit(X, y)
            cand = [p for p in pool if p not in seen]
            pred = m.predict(Xte[cand])
            pick = cand[int(np.argmin(pred))]
            q = inspect(en, "adaptive", pick)
            seen.append(pick)
            obsX.append(Xte[pick])
            obsY.append(q)
            seq.append(q)
        M["adaptive"].append(metrics(seq))
        # random
        seen, seq = [], []
        for p in ep["random_order"]:
            if p in seen:
                continue
            if len(seen) >= BUDGET:
                break
            seen.append(p)
            seq.append(inspect(en, "random", p))
        M["random"].append(metrics(seq))
        for s in ("adaptive", "random"):
            b5, b10, h = M[s][-1]
            results.append({"episode": en, "strategy": s, "best_at_5": b5,
                            "best_at_10": b10, "inspections_to_target": h})

    comps = {}
    for j, name in enumerate(["best_at_5", "best_at_10", "inspections_to_target"]):
        comps[name] = compare([m[j] for m in M["adaptive"]], [m[j] for m in M["random"]])
    print(json.dumps({"target": target, "results": results, "comparisons": comps}), flush=True)


if __name__ == "__main__":
    main()
