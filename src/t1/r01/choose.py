import csv, json, math, sys, time


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return [h.strip() for h in rows[0]], [[x.strip() for x in r] for r in rows[1:] if r]


def num(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    a = sys.argv[1:]
    if not a or a[0] != "choose":
        sys.exit("usage: main.py choose --train T --test M")
    train_p = a[a.index("--train") + 1]
    test_p = a[a.index("--test") + 1]
    t0 = time.time()
    th, tr = load(train_p)
    eh, te = load(test_p)

    goals = [(i, h) for i, h in enumerate(th) if h[-1] in "+-"]
    feats = [i for i, h in enumerate(th) if h[-1] not in "+-X"]
    eidx = {h: i for i, h in enumerate(eh)}

    # normalization stats from training data only
    stats = {}
    for i, h in enumerate(th):
        if h[0].isupper() and h[-1] != "X":
            v = [x for x in (num(r[i]) for r in tr if i < len(r)) if x is not None]
            if v:
                stats[i] = (min(v), max(v) or 1)
    def nz(i, s):
        lo, hi = stats[i]
        x = num(s)
        return None if x is None else (x - lo) / ((hi - lo) or 1)

    def d2h(vals):
        # vals: raw goal values aligned with goals
        t = 0
        for (i, h), x in zip(goals, vals):
            n = nz(i, str(x))
            if n is None:
                return None
            n = min(max(n, -1), 2)
            t += ((1 - n) if h[-1] == "+" else n) ** 2
        return math.sqrt(t / len(goals))

    def vec(r, hdr_idx):
        out = []
        for i in feats:
            j = hdr_idx(i)
            s = r[j] if j is not None and j < len(r) else "?"
            if i in stats:
                out.append(nz(i, s) if s != "?" else None)
            else:
                out.append(None if s == "?" else s)
        return out

    def dist(x, y):
        t = 0
        for p, q in zip(x, y):
            if p is None or q is None:
                t += 0.25
            elif isinstance(p, str) or isinstance(q, str):
                t += 0.0 if p == q else 1.0
            else:
                t += (p - q) ** 2
        return t

    tv, ty = [], []
    for r in tr:
        y = d2h([r[i] if i < len(r) else "?" for i, _ in goals])
        if y is not None:
            tv.append(vec(r, lambda i: i))
            ty.append(y)
    K = 5
    pred = []
    for r in te:
        x = vec(r, lambda i: eidx.get(th[i]))
        ds = sorted((dist(x, v), y) for v, y in zip(tv, ty))[:K]
        w = [1 / (d + 1e-3) for d, _ in ds]
        pred.append(sum(wi * y for wi, (_, y) in zip(w, ds)) / sum(w) if ds else 0.5)

    order = sorted(range(len(te)), key=lambda i: (pred[i], i))
    budget = min(10, len(te))
    chosen = order[:budget]

    def extract(resp):
        g = resp
        if isinstance(resp, dict):
            for k in ("goals", "observed", "values", "y"):
                if k in resp:
                    g = resp[k]
                    break
        vals = []
        if isinstance(g, dict):
            low = {k.lower(): v for k, v in g.items()}
            for _, h in goals:
                for k in (h, h[:-1], h.lower(), h[:-1].lower()):
                    if k in g or k in low:
                        vals.append(g.get(k, low.get(k)))
                        break
                else:
                    return None
            return vals
        if isinstance(g, list) and len(g) >= len(goals):
            return g[-len(goals):]
        return None

    observed = {}
    for idx in chosen:
        print(json.dumps({"inspect": idx}), flush=True)
        line = sys.stdin.readline()
        if not line:
            break
        try:
            vals = extract(json.loads(line))
            d = d2h(vals) if vals is not None else None
        except Exception:
            d = None
        observed[idx] = d if d is not None else pred[idx]

    best = sorted(observed, key=lambda i: (observed[i], i))[:3]
    print(json.dumps({"selected": best}), flush=True)
    sys.stderr.write(json.dumps({"inspected": list(observed), "selected": best,
                                 "seconds": time.time() - t0}) + "\n")


main()
