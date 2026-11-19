import sys, json, random, argparse
from math import exp, sqrt

def atom(s):
    s = s.strip()
    for f in (int, float):
        try: return f(s)
        except ValueError: pass
    return s

def load_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return [[atom(c) for c in l.split(",")] for l in f if l.strip()]

class Model:
    def __init__(s, header, rows):
        s.h = header
        s.num = {i: h[0].isupper() for i, h in enumerate(header)}
        s.ig = {i for i, h in enumerate(header) if h[-1] == "X"}
        s.y = [i for i, h in enumerate(header) if h[-1] in "+-" and i not in s.ig]
        s.x = [i for i, h in enumerate(header) if i not in s.ig and i not in s.y]
        s.st = {}
        for i in range(len(header)):
            if s.num[i]:
                v = [r[i] for r in rows if i < len(r) and isinstance(r[i], (int, float))]
                n = len(v)
                mu = sum(v) / n if n else 0
                sd = sqrt(sum((a - mu) ** 2 for a in v) / (n - 1)) if n > 1 else 0
                s.st[i] = (mu, sd)

    def norm(s, i, v):
        mu, sd = s.st[i]
        z = max(-3, min(3, (v - mu) / (1e-32 + sd)))
        return 1 / (1 + exp(-1.7 * z))

    def d2h(s, vals):  # vals: dict col index -> value
        ds = []
        for i in s.y:
            v = vals.get(i, "?")
            if not isinstance(v, (int, float)): ds.append(1.0); continue
            g = 0 if s.h[i][-1] == "-" else 1
            ds.append(abs(s.norm(i, v) - g))
        return sqrt(sum(d * d for d in ds) / max(1, len(ds)))

    def xd(s, a, b):
        t = 0
        for i in s.x:
            p, q = a[i], b[i]
            if p == "?" or q == "?": g = 1
            elif s.num[i]: g = abs(s.norm(i, p) - s.norm(i, q))
            else: g = float(p != q)
            t += g * g
        return sqrt(t / max(1, len(s.x)))

def goals_of(m, resp):
    g = None
    for k in ("goals", "y", "ys", "values", "observed"):
        if k in resp: g = resp[k]; break
    if g is None:
        for k, v in resp.items():
            if k not in ("row", "index", "inspect", "episode", "strategy") and isinstance(v, (list, dict)):
                g = v; break
    out = {}
    if isinstance(g, dict):
        for i in m.y:
            n = m.h[i]
            for key in (n, n[:-1], str(i)):
                if key in g: out[i] = g[key]; break
    elif isinstance(g, list):
        if len(g) == len(m.h): out = {i: g[i] for i in m.y}
        else: out = {i: g[k] for k, i in enumerate(m.y) if k < len(g)}
    return out

def inspect(m, ep, strat, idx):
    sys.stdout.write(json.dumps({"episode": ep, "strategy": strat, "inspect": idx}) + "\n")
    sys.stdout.flush()
    line = sys.stdin.readline()
    resp = json.loads(line) if line.strip() else {}
    return m.d2h(goals_of(m, resp))

def run_episode(m, rows, ep, e, target):
    pool = list(e["pool"])
    res = {}
    # random strategy
    seen, sc = [], []
    for idx in e["random_order"]:
        if idx in seen: continue
        seen.append(idx); sc.append(inspect(m, ep, "random", idx))
        if len(seen) == 10: break
    res["random"] = sc
    # adaptive strategy
    rng = random.Random(1000 + ep)
    start = rng.sample(pool, min(4, len(pool)))
    seen, sc = [], []
    for idx in start:
        seen.append(idx); sc.append(inspect(m, ep, "adaptive", idx))
    while len(seen) < 10 and len(seen) < len(pool):
        order = sorted(range(len(seen)), key=lambda k: sc[k])
        nb = max(1, int(sqrt(len(seen))))
        best = [seen[k] for k in order[:nb]]
        rest = [seen[k] for k in order[nb:]] or best
        cand = [i for i in pool if i not in seen]
        def key(i):
            r = rows[i]
            db = sum(m.xd(r, rows[b]) for b in best) / len(best)
            dr = sum(m.xd(r, rows[b]) for b in rest) / len(rest)
            return db - dr
        pick = min(cand, key=key)
        seen.append(pick); sc.append(inspect(m, ep, "adaptive", pick))
    res["adaptive"] = sc
    out = []
    for st in ("adaptive", "random"):
        sc = res[st]
        need = next((k + 1 for k, v in enumerate(sc) if v <= target), 11)
        out.append({"episode": ep, "strategy": st,
                    "best_at_5": min(sc[:5]) if sc else None,
                    "best_at_10": min(sc[:10]) if sc else None,
                    "inspections_to_target": need})
    return out

def cliffs(xs, ys, d=0.197):
    gt = lt = 0
    for x in xs:
        for y in ys:
            gt += x > y; lt += x < y
    return abs(gt - lt) / (len(xs) * len(ys)) <= d

def ks(xs, ys, a=1.36):
    n, m = len(xs), len(ys)
    d = 0
    for v in sorted(set(xs + ys)):
        d = max(d, abs(sum(x <= v for x in xs) / n - sum(y <= v for y in ys) / m))
    return d <= a * sqrt((n + m) / (n * m))

def cohen(xs, ys, d=0.35):
    def ms(v):
        n = len(v); mu = sum(v) / n
        return mu, sum((a - mu) ** 2 for a in v) / max(1, n - 1)
    (a, va), (b, vb) = ms(xs), ms(ys)
    n, m = len(xs), len(ys)
    pool = ((n - 1) * va + (m - 1) * vb) / max(1, n + m - 2)
    return abs(a - b) <= d * sqrt(pool)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    tr = load_csv(a.train)
    m = Model(tr[0], tr[1:])
    ds = sorted(m.d2h({i: r[i] for i in m.y}) for r in tr[1:])
    target = ds[len(ds) // 4]
    with open(a.test) as f: T = json.load(f)
    rows = T["rows"]
    results = []
    for ep, e in enumerate(T["episodes"]):
        results += run_episode(m, rows, ep, e, target)
    comps = {}
    for met in ("best_at_5", "best_at_10", "inspections_to_target"):
        A = [r[met] for r in results if r["strategy"] == "adaptive"]
        R = [r[met] for r in results if r["strategy"] == "random"]
        c, k, h = cliffs(A, R), ks(A, R), cohen(A, R)
        comps[met] = {"adaptive_mean": sum(A) / len(A), "random_mean": sum(R) / len(R),
                      "cliffs_same": c, "ks_same": k, "cohen_same": h,
                      "same": c and k and h,
                      "adaptive_better": (sum(A) < sum(R)) and not (c and k and h)}
    print(json.dumps({"target": target, "results": results, "comparisons": comps}))

main()
