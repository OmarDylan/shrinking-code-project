import sys, json, argparse
from math import exp, sqrt


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


def guess(a, K=7):
    names, train = load(a.train)
    tnames, test = load(a.test)
    goals = [(n, 0.0 if n[-1] == "-" else 1.0) for n in names if n[-1] in "+-"]
    stats = goal_stats(names, train, goals)
    mi = next((i for i, n in enumerate(names) if n.lower().startswith("mpg")),
              names.index(goals[-1][0]))

    def val(r, i):
        return num(r[i]) if i < len(r) else None

    # training targets (goal values are visible in training only)
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

    # x columns: not ignored, not goals; map to test column by name
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

    mpg_out, sc_out = [], []
    for r in test:
        q = [prep(r, j, isn) for _, j, isn in xs]
        ds = []
        for k, t in enumerate(trx):
            s = c = 0
            for (_, _, isn), a_, b_ in zip(xs, q, t):
                if a_ is None or b_ is None:
                    continue  # missing input: skip that column
                g = abs(a_ - b_) if isn else float(a_ != b_)
                s += g * g
                c += 1
            ds.append((sqrt(s / c) if c else 1.0, k))
        ds.sort()
        nn = [k for _, k in ds[:K]]
        ms = [tr[k][1] for k in nn if tr[k][1] is not None]
        mpg_out.append(sum(ms) / len(ms) if ms else base_mpg)
        dd = [tr[k][2] for k in nn]
        sc_out.append(sum(dd) / len(dd) if dd else base_d)
    print(json.dumps({"mpg": mpg_out, "scores": sc_out}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["remember", "guess"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    (remember if a.cmd == "remember" else guess)(a)


if __name__ == "__main__":
    main()
