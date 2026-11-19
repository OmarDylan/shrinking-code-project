import sys, json, csv, argparse

def load(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r and any(c.strip() for c in r)]
    return rows[0], rows[1:]

def num(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    names, tr = load(a.train)
    tnames, te = load(a.test)
    n = len(names)
    xs = [i for i, s in enumerate(names) if s[-1] not in "XxX+-!" and s[-1] != "X"]
    goals = [i for i, s in enumerate(names) if s[-1] in "+-"]
    isnum = {i: names[i][0].isupper() for i in range(n)}
    mpg = next((i for i in goals if names[i].lower().startswith("mpg")), goals[-1])

    # ranges of numeric columns from training data
    lo, hi = {}, {}
    for i in range(n):
        if isnum[i]:
            v = [num(r[i]) for r in tr if i < len(r) and r[i] != "?" and num(r[i]) is not None]
            if v:
                lo[i], hi[i] = min(v), max(v)
            else:
                lo[i] = hi[i] = 0.0

    # distance to heaven for each train row (training-only normalisation)
    def d2h(r):
        t = 0.0
        for g in goals:
            v = num(r[g])
            if v is None:
                return None
            z = (v - lo[g]) / (hi[g] - lo[g] + 1e-32)
            t += (z - (0 if names[g][-1] == "-" else 1)) ** 2
        return (t / len(goals)) ** 0.5

    train = []
    for r in tr:
        if len(r) < n:
            continue
        m = num(r[mpg])
        d = d2h(r)
        if m is None or d is None:
            continue
        train.append((r, m, d))
    base_m = sum(t[1] for t in train) / len(train)
    base_d = sum(t[2] for t in train) / len(train)

    def dist(r1, r2):
        s, c = 0.0, 0
        for i in xs:
            if i >= len(r1) or i >= len(r2):
                continue
            p, q = r1[i], r2[i]
            if p == "?" or q == "?" or p == "" or q == "":
                continue
            if isnum[i]:
                u, v = num(p), num(q)
                if u is None or v is None:
                    continue
                s += ((u - v) / (hi[i] - lo[i] + 1e-32)) ** 2
            else:
                s += 0.0 if p == q else 1.0
            c += 1
        return (s / c) ** 0.5 if c else 1.0

    K = 7
    mp, sc = [], []
    for r in te:
        if not train:
            mp.append(base_m); sc.append(base_d); continue
        ds = sorted(((dist(r, t[0]), t[1], t[2]) for t in train), key=lambda z: z[0])[:K]
        w = [1.0 / (d + 0.05) for d, _, _ in ds]
        sw = sum(w)
        mp.append(sum(wi * m for wi, (_, m, _) in zip(w, ds)) / sw)
        sc.append(sum(wi * d for wi, (_, _, d) in zip(w, ds)) / sw)

    print(json.dumps({"mpg": mp, "scores": sc}))

if __name__ == "__main__":
    main()
