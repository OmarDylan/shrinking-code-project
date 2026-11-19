import sys, csv, json, random
from math import exp, sqrt

def atom(s):
    s = s.strip()
    for f in (int, float):
        try: return f(s)
        except ValueError: pass
    return s

def load(p):
    with open(p, encoding="utf-8-sig") as f:
        return [[atom(c) for c in r] for r in csv.reader(f) if r and any(c.strip() for c in r)]

def choose(train_path, test_path, budget=10, k=3):
    random.seed(1)
    tr = load(train_path); names = [str(n) for n in tr[0]]; tr = tr[1:]
    te = load(test_path)[1:]
    cols = [i for i, n in enumerate(names) if n[-1] != "X"]
    ys = [i for i in cols if names[i][-1] in "+-"]
    xs = [i for i in cols if i not in ys]
    num = {i for i in cols if names[i][0].isupper()}
    st = {}
    for i in num:  # training-only statistics
        v = [r[i] for r in tr if isinstance(r[i], (int, float))]
        mu = sum(v) / max(1, len(v))
        sd = sqrt(sum((a - mu) ** 2 for a in v) / max(1, len(v) - 1)) if len(v) > 1 else 0
        st[i] = (mu, sd)

    def norm(i, v):
        mu, sd = st[i]
        z = max(-3, min(3, (v - mu) / (sd + 1e-32)))
        return 1 / (1 + exp(-1.7 * z))

    def ydist(r):
        s = 0
        for i in ys:
            g = 0 if names[i][-1] == "-" else 1
            s += (norm(i, r[i]) - g) ** 2
        return sqrt(s / len(ys))

    def gap(i, a, b):
        if a == "?" or b == "?": return 1
        if i in num: return abs(norm(i, a) - norm(i, b))
        return float(a != b)

    def xd(a, b):
        return sqrt(sum(gap(i, a[i], b[i]) ** 2 for i in xs) / max(1, len(xs)))

    def ok(r):
        return all(isinstance(r[i], (int, float)) for i in ys)

    known = [(r, ydist(r)) for r in tr if ok(r)]
    seen = {}  # index -> (row, ydist)

    def parse(resp, idx):
        row = list(te[idx])
        g = None
        for key in ("goals", "y", "ys", "observed", "values", "row"):
            if key in resp: g = resp[key]; break
        if g is None: g = {kk: v for kk, v in resp.items() if kk not in ("row", "index", "idx", "inspect")}
        if isinstance(g, dict):
            for i in ys:
                for kk in (names[i], names[i][:-1]):
                    if kk in g: row[i] = g[kk]; break
        elif isinstance(g, list):
            if len(g) == len(names):
                for i in ys: row[i] = g[i]
            elif len(g) == len(ys):
                for i, v in zip(ys, g): row[i] = v
        return row

    def inspect(idx):
        print(json.dumps({"inspect": idx}), flush=True)
        line = sys.stdin.readline()
        row = parse(json.loads(line), idx)
        if not ok(row): return
        seen[idx] = (row, ydist(row))

    def pred(r):
        pool = known + list(seen.values())
        near = sorted(((xd(r, p), y) for p, y in pool), key=lambda t: t[0])[:5]
        if not near: return 0.5
        w = [1 / (d + 0.05) for d, _ in near]
        return sum(a * y for a, (_, y) in zip(w, near)) / sum(w)

    asked = set()
    n = len(te)
    while len(asked) < min(budget, n):
        left = [i for i in range(n) if i not in asked]
        if len(asked) < 2:
            idx = random.choice(left)
        else:
            idx = min(left, key=lambda i: (pred(te[i]), i))
        asked.add(idx)
        inspect(idx)
    best = sorted(seen, key=lambda i: (seen[i][1], i))[:k]
    print(json.dumps({"selected": best}), flush=True)

def main(a):
    if not a or a[0] != "choose": sys.exit("usage: main.py choose --train T --test M")
    o = {a[i]: a[i + 1] for i in range(1, len(a) - 1, 2)}
    choose(o["--train"], o["--test"])

if __name__ == "__main__":
    main(sys.argv[1:])
