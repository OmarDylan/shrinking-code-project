import sys, csv, json, math, argparse

GOALS = ["Lbs-", "Acc+", "Mpg+"]
NUMS = ["Clndrs", "Volume", "Model"]
CAT = "origin"


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        hdr = [h.strip() for h in next(rd)]
        rows = [[c.strip() for c in r] for r in rd if r and any(c.strip() for c in r)]
    return hdr, rows


def num(s):
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    th, tr = load(a.train)
    eh, te = load(a.test)
    ti = {h: i for i, h in enumerate(th)}
    ei = {h: i for i, h in enumerate(eh)}

    def get(r, idx, h):
        i = idx.get(h)
        return r[i] if i is not None and i < len(r) else "?"

    # goal stats
    gv = {g: [] for g in GOALS}
    for r in tr:
        vs = [num(get(r, ti, g)) for g in GOALS]
        if all(v is not None for v in vs):
            for g, v in zip(GOALS, vs):
                gv[g].append(v)
    st = {}
    for g in GOALS:
        xs = gv[g]
        n = len(xs)
        mu = sum(xs) / n
        sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        st[g] = (mu, sd)

    def d2h(vals):
        t = 0.0
        for g in GOALS:
            mu, sd = st[g]
            z = max(-3, min(3, (vals[g] - mu) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            h = 0 if g.endswith("-") else 1
            t += (nv - h) ** 2
        return math.sqrt(t / 3)

    # training feature stats
    rng = {}
    for c in NUMS:
        xs = [num(get(r, ti, c)) for r in tr]
        xs = [x for x in xs if x is not None]
        rng[c] = (min(xs), max(xs) - min(xs) or 1.0) if xs else (0, 1.0)

    def feat(r, idx):
        f = []
        for c in NUMS:
            v = num(get(r, idx, c))
            f.append(None if v is None else (v - rng[c][0]) / rng[c][1])
        s = get(r, idx, CAT)
        f.append(None if s == "?" else s)
        return f

    train = []
    for r in tr:
        vs = [num(get(r, ti, g)) for g in GOALS]
        if any(v is None for v in vs):
            continue
        train.append((feat(r, ti), d2h(dict(zip(GOALS, vs)))))

    def dist(a, b):
        t = 0.0
        for x, y in zip(a, b):
            if x is None or y is None:
                g = 1.0
            elif isinstance(x, str) or isinstance(y, str):
                g = float(x != y)
            else:
                g = abs(x - y)
            t += g * g
        return math.sqrt(t / len(a))

    def predict(f, k=5):
        ds = sorted(((dist(f, tf), y) for tf, y in train), key=lambda p: p[0])[:k]
        w = [1 / (d + 0.05) for d, _ in ds]
        return sum(wi * y for wi, (_, y) in zip(w, ds)) / sum(w)

    preds = [(predict(feat(r, ei)), i) for i, r in enumerate(te)]
    preds.sort()
    picks = [i for _, i in preds[:10]]

    seen = {}
    out = sys.stdout
    for i in picks:
        out.write(json.dumps({"inspect": i}) + "\n")
        out.flush()
        line = sys.stdin.readline()
        if not line:
            break
        resp = json.loads(line)
        seen[i] = d2h(resp["goals"])

    ranked = sorted(seen, key=lambda i: (round(seen[i], 12), i))[:3]
    out.write(json.dumps({"selected": ranked}) + "\n")
    out.flush()


if __name__ == "__main__":
    main()
