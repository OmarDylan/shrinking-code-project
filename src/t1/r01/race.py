import sys, json, csv, math, random

def load_train(path):
    with open(path, newline='') as f:
        r = list(csv.reader(f))
    return r[0], r[1:]

def num(v):
    try:
        if v is None or v == '?':
            return None
        return float(v)
    except Exception:
        return None

def quantile(xs, q):
    s = sorted(xs)
    p = (len(s) - 1) * q
    lo = int(math.floor(p)); hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (p - lo)

def mann_whitney(a, b):
    n, m = len(a), len(b)
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(allv)
    i = 0; tie = 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        t = j - i + 1
        tie += t ** 3 - t
        i = j + 1
    ra = sum(r for r, (v, g) in zip(ranks, allv) if g == 0)
    u = ra - n * (n + 1) / 2
    N = n + m
    var = n * m / 12 * ((N + 1) - tie / (N * (N - 1))) if N > 1 else 0
    if var <= 0:
        return {"U": u, "p": 1.0}
    z = (u - n * m / 2) / math.sqrt(var)
    return {"U": u, "p": math.erfc(abs(z) / math.sqrt(2))}

def cliffs(a, b):
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    d = (gt - lt) / (len(a) * len(b))
    ad = abs(d)
    mag = "negligible" if ad < 0.147 else "small" if ad < 0.33 else "medium" if ad < 0.474 else "large"
    return {"delta": d, "magnitude": mag}

def bootstrap(a, b, B=2000):
    rng = random.Random(1)
    obs = sum(a) / len(a) - sum(b) / len(b)
    pool = a + b
    n = len(a); cnt = 0
    for _ in range(B):
        s = [rng.choice(pool) for _ in pool]
        d = sum(s[:n]) / n - sum(s[n:]) / (len(s) - n)
        if abs(d) >= abs(obs) - 1e-12:
            cnt += 1
    return {"mean_diff": obs, "p": (cnt + 1) / (B + 1)}

def main():
    if len(sys.argv) < 2 or sys.argv[1] != 'race':
        sys.exit("usage: main.py race --train TRAIN --test CASES_JSON")
    args = sys.argv[2:]
    train = test = None
    for i, a in enumerate(args):
        if a == '--train': train = args[i + 1]
        if a == '--test': test = args[i + 1]
    th, trows = load_train(train)
    with open(test) as f:
        cases = json.load(f)
    header = cases.get("header", th)
    rows = cases["rows"]
    eps = cases["episodes"]

    goals = [i for i, h in enumerate(header) if h[-1] in '+-']
    feats = [i for i, h in enumerate(header) if h[-1] not in '+-X']
    isnum = {i: header[i][0].isupper() for i in feats}
    maxim = {i: header[i][-1] == '+' for i in goals}
    names = {i: header[i][:-1].lower() for i in goals}

    # goal ranges from training
    rng_ = {}
    for i in goals:
        vs = [num(r[i]) for r in trows if i < len(r) and num(r[i]) is not None]
        rng_[i] = (min(vs), max(vs))

    def d2h(vals):
        s = 0.0
        for i, v in zip(goals, vals):
            lo, hi = rng_[i]
            x = (v - lo) / (hi - lo) if hi > lo else 0.0
            s += ((1 if maxim[i] else 0) - x) ** 2
        return math.sqrt(s / len(goals))

    def row_row(r):
        if isinstance(r, dict):
            return [r.get(h) for h in header]
        return r

    tr_d = []
    tr_feat = []
    for r in trows:
        vals = [num(r[i]) for i in goals]
        if any(v is None for v in vals):
            continue
        tr_d.append(d2h(vals))
        tr_feat.append(r)
    target = quantile(tr_d, 0.25)

    # feature ranges
    frange = {}
    for i in feats:
        if isnum[i]:
            vs = [num(r[i]) for r in trows if num(r[i]) is not None]
            frange[i] = (min(vs), max(vs)) if vs else (0, 1)

    def dist(a, b):
        s = 0.0
        for i in feats:
            x, y = a[i], b[i]
            if isnum[i]:
                x, y = num(x), num(y)
                if x is None or y is None:
                    s += 1
                else:
                    lo, hi = frange[i]
                    s += ((x - y) / (hi - lo)) ** 2 if hi > lo else 0
            else:
                s += 0 if (x == y and x not in (None, '?')) else 1
        return math.sqrt(s)

    trow = [row_row(r) for r in rows]

    def inspect(ep, strat, idx):
        print(json.dumps({"episode": ep, "strategy": strat, "inspect": idx}), flush=True)
        line = sys.stdin.readline()
        resp = json.loads(line)
        g = resp.get("goals", resp.get("observed", resp.get("values")))
        if isinstance(g, dict):
            vals = []
            for i in goals:
                for k, v in g.items():
                    if k == header[i] or k.lower() == names[i]:
                        vals.append(float(v)); break
        elif isinstance(g, list):
            vals = [float(v) for v in g]
        else:
            vals = []
            for i in goals:
                for k, v in resp.items():
                    if k == header[i] or k.lower() == names[i]:
                        vals.append(float(v)); break
        return d2h(vals)

    def metrics(seq):
        best = float('inf'); hit = 11; b5 = None
        for n, q in enumerate(seq, 1):
            best = min(best, q)
            if best <= target and hit == 11:
                hit = n
            if n == 5:
                b5 = best
        return {"best_at_5": b5, "best_at_10": best, "inspections_to_target": hit}

    def adaptive(ep, pool):
        labeled = list(zip(tr_feat, tr_d))
        seen = set(); seq = []
        for _ in range(10):
            bestc = None
            for idx in pool:
                if idx in seen:
                    continue
                ds = sorted((dist(trow[idx], l), y) for l, y in labeled)[:5]
                w = [1 / (d + 1e-3) for d, _ in ds]
                pred = sum(wi * y for wi, (_, y) in zip(w, ds)) / sum(w)
                if bestc is None or pred < bestc[0]:
                    bestc = (pred, idx)
            idx = bestc[1]
            seen.add(idx)
            q = inspect(ep, "adaptive", idx)
            seq.append(q)
            labeled.append((trow[idx], q))
        return seq

    results = []
    for n, e in enumerate(eps):
        ep = e.get("episode", n)
        pool = e["pool"]
        a = adaptive(ep, pool)
        rseq = []
        seen = set()
        for idx in e["random_order"]:
            if idx in seen:
                continue
            seen.add(idx)
            rseq.append(inspect(ep, "random", idx))
            if len(rseq) == 10:
                break
        results.append({"episode": ep, "target": target,
                        "adaptive": metrics(a), "random": metrics(rseq)})

    comps = {}
    for m in ("best_at_5", "best_at_10", "inspections_to_target"):
        x = [r["adaptive"][m] for r in results]
        y = [r["random"][m] for r in results]
        comps[m] = {
            "adaptive_mean": sum(x) / len(x), "random_mean": sum(y) / len(y),
            "adaptive_median": quantile(x, .5), "random_median": quantile(y, .5),
            "mann_whitney": mann_whitney(x, y),
            "cliffs_delta": cliffs(x, y),
            "bootstrap": bootstrap(x, y),
        }
    print(json.dumps({"results": results, "comparisons": comps}), flush=True)

main()
