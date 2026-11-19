import sys, json, math, time, argparse

def atom(s):
    s = s.strip()
    for f in (int, float):
        try: return f(s)
        except ValueError: pass
    return s

def rd(path):
    with open(path, encoding="utf-8-sig") as f:
        return [[atom(c) for c in l.split(",")] for l in f if l.strip()]

def cliffs(xs, ys, d=0.197):
    gt = lt = 0
    for x in xs:
        for y in ys:
            if x > y: gt += 1
            elif x < y: lt += 1
    return abs(gt - lt) / (len(xs) * len(ys)) <= d

def ks(xs, ys, a=1.36):
    n, m, i, j, d = len(xs), len(ys), 0, 0, 0
    while i < n and j < m:
        v = min(xs[i], ys[j])
        while i < n and xs[i] <= v: i += 1
        while j < m and ys[j] <= v: j += 1
        d = max(d, abs(i / n - j / m))
    return d <= a * math.sqrt((n + m) / (n * m))

def same(xs, ys):
    xs, ys = sorted(xs), sorted(ys)
    return cliffs(xs, ys) and ks(xs, ys)

def flow(train, test, W=20, K=5):
    t0 = time.time()
    hdr, rows = train[0], train[1:]
    cols = [i for i, h in enumerate(hdr) if h[-1] != "X"]
    ycol = [i for i in cols if hdr[i][-1] in "+-"]
    xcol = [i for i in cols if i not in ycol]
    num = {i: hdr[i][0].isupper() for i in cols}
    st = {}
    for i in cols:
        if num[i]:
            v = [r[i] for r in rows if len(r) > i and isinstance(r[i], (int, float))]
            mu = sum(v) / len(v)
            sd = math.sqrt(sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1))
            st[i] = (mu, sd)

    def norm(i, v):
        mu, sd = st[i]
        z = max(-3, min(3, (v - mu) / (1e-32 + sd)))
        return 1 / (1 + math.exp(-1.7 * z))

    def gap(i, a, b):
        if a == "?" or b == "?" or a is None or b is None: return 1
        if num[i]:
            if isinstance(a, str) or isinstance(b, str): return 1
            return abs(norm(i, a) - norm(i, b))
        return 0 if a == b else 1

    def xd(a, b):
        return sum(gap(i, a[i] if len(a) > i else "?", b[i]) ** 2 for i in xcol)

    goal = {i: 0 if hdr[i][-1] == "-" else 1 for i in ycol}
    mi = next((i for i in ycol if hdr[i].lower().startswith("mpg")), ycol[-1])

    def predict(r, skip=None):
        ds = sorted((xd(r, t), n) for n, t in enumerate(rows) if n != skip)[:K]
        ys = {}
        for i in ycol:
            v = [rows[n][i] for _, n in ds if isinstance(rows[n][i], (int, float))]
            ys[i] = sum(v) / len(v) if v else st[i][0]
        sc = math.sqrt(sum((abs(norm(i, ys[i]) - goal[i])) ** 2 for i in ycol) / len(ycol))
        return ys[mi], sc

    hist = [predict(r, n)[1] for n, r in enumerate(rows)]
    mpg, scores, alerts = [], [], []
    for r in test[1:]:
        m, s = predict(r)
        mpg.append(m); scores.append(s)
        if len(scores) < W: alerts.append(False)
        else: alerts.append(not same(scores[-W:], hist))
    return {"mpg": mpg, "scores": scores, "alerts": alerts,
            "seconds": time.time() - t0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    if a.cmd != "flow":
        sys.exit("unknown command")
    print(json.dumps(flow(rd(a.train), rd(a.test))))

if __name__ == "__main__":
    main()
