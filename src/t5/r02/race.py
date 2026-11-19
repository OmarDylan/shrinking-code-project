import sys, json, csv, math, heapq, argparse

G = ["Lbs-", "Acc+", "Mpg+"]
NUM = ["Clndrs", "Volume", "Model"]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def sg(v, mu, sd):
    z = max(-3, min(3, (v - mu) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))


def stats(vals):
    n = len(vals)
    mu = sum(vals) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0
    return mu, sd


def d2h(goals, gs):
    s = 0
    for g in G:
        h = 0 if g[-1] == "-" else 1
        s += (sg(goals[g], *gs[g]) - h) ** 2
    return math.sqrt(s / 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()
    with open(a.train, encoding="utf-8-sig") as f:
        rd = [r for r in csv.reader(f) if r]
    th, trows = [h.strip() for h in rd[0]], rd[1:]
    ix = {h: i for i, h in enumerate(th)}
    gs = {g: stats([float(r[ix[g]]) for r in trows]) for g in G}
    tr_d = [d2h({g: float(r[ix[g]]) for g in G}, gs) for r in trows]
    target = sorted(tr_d)[len(tr_d) // 4]
    ns = {}
    for c in NUM:
        vs = [x for x in (num(r[ix[c]]) for r in trows) if x is not None]
        ns[c] = stats(vs)

    def feat(get):
        out = []
        for c in NUM:
            v = num(get(c))
            out.append(None if v is None else sg(v, *ns[c]))
        o = get("origin")
        out.append(None if o is None or str(o).strip() in ("?", "") else str(o).strip())
        return out

    def dist(p, q):
        s = 0
        for x, y in zip(p, q):
            if x is None or y is None:
                s += 1
            elif isinstance(x, str):
                s += x != y
            else:
                s += (x - y) ** 2
        return math.sqrt(s / 4)

    train_f = [(feat(lambda c, r=r: r[ix[c]]), y) for r, y in zip(trows, tr_d)]

    data = json.load(open(a.test))
    h = data["header"]
    hx = {n: i for i, n in enumerate(h)}
    rows = data["rows"]
    tf = {}

    def tfeat(i):
        if i not in tf:
            tf[i] = feat(lambda c: rows[i][hx[c]])
        return tf[i]

    def ask(ep, strat, i):
        print(json.dumps({"episode": ep, "strategy": strat, "inspect": i}), flush=True)
        resp = json.loads(sys.stdin.readline())
        return d2h(resp["goals"], gs)

    def adaptive(ep, pool):
        seen = []  # (feat, y)
        got = {}
        order = []
        for _ in range(10):
            best, bs = None, 1e99
            for c in pool:
                if c in got:
                    continue
                p = tfeat(c)
                ds = heapq.nsmallest(5, ((dist(p, f), y) for f, y in train_f + seen))
                s = sum(y for _, y in ds) / len(ds)
                if s < bs:
                    best, bs = c, s
            y = ask(ep, "adaptive", best)
            got[best] = y
            seen.append((tfeat(best), y))
            order.append(y)
        return order

    def randoms(ep, e):
        return [ask(ep, "random", i) for i in e["random_order"][:10]]

    def summ(ys):
        lt = 11
        for k, y in enumerate(ys):
            if round(y, 12) <= round(target, 12):
                lt = k + 1
                break
        return {"best5": min(ys[:5]), "best10": min(ys), "labels_to_target": lt}

    results = []
    for ep, e in enumerate(data["episodes"]):
        ad = adaptive(ep, e["pool"])
        rn = randoms(ep, e)
        results.append({"episode": ep, "adaptive": summ(ad), "random": summ(rn)})

    def compare(xs, ys):
        n, m = len(xs), len(ys)
        mx, my = sum(xs) / n, sum(ys) / m
        vx = sum((x - mx) ** 2 for x in xs) / (n - 1)
        vy = sum((y - my) ** 2 for y in ys) / (m - 1)
        pv = ((n - 1) * vx + (m - 1) * vy) / (n + m - 2)
        co = abs(mx - my) / (math.sqrt(pv) + 1e-32)
        cl = abs(sum((x > y) - (x < y) for x in xs for y in ys)) / (n * m)
        d = max(abs(sum(x <= v for x in xs) / n - sum(y <= v for y in ys) / m)
                for v in set(xs) | set(ys))
        k = d / math.sqrt(1 / n + 1 / m)
        return {"cohen": co, "cliffs": cl, "ks": k,
                "same": bool(co <= 0.35 and cl <= 0.197 and k <= 1.36)}

    comps = {}
    for met in ["best5", "best10", "labels_to_target"]:
        comps[met] = compare([r["adaptive"][met] for r in results],
                             [r["random"][met] for r in results])
    print(json.dumps({"results": results, "comparisons": comps}), flush=True)


main()
