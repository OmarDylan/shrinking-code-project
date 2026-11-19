import csv, sys, json, math

NUM = ["Clndrs", "Volume", "Model"]
CAT = "origin"
K = 5


def load(path):
    with open(path, newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            row = []
            for c in NUM:
                v = (r.get(c) or "?").strip()
                try:
                    x = float(v)
                    row.append(x if math.isfinite(x) else None)
                except ValueError:
                    row.append(None)
            o = (r.get(CAT) or "?").strip()
            row.append(None if o == "?" or o == "" else o)
            rows.append(row)
    return rows


def main():
    a = sys.argv
    train = load(a[a.index("--train") + 1])
    test = load(a[a.index("--test") + 1])
    sc = []
    for j in range(3):
        v = [r[j] for r in train if r[j] is not None]
        if len(v) > 1:
            m = sum(v) / len(v)
            sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))
        else:
            sd = 1.0
        sc.append(sd if sd > 1e-9 else 1.0)

    def dist(p, q):
        s = n = 0
        for j in range(3):
            if p[j] is not None and q[j] is not None:
                s += ((p[j] - q[j]) / sc[j]) ** 2
                n += 1
        if p[3] is not None and q[3] is not None:
            s += 0.0 if p[3] == q[3] else 1.0
            n += 1
        return math.sqrt(s / n) if n else 0.0

    def knn(p, skip=None):
        d = sorted(dist(p, q) for i, q in enumerate(train) if i != skip)
        d = d[:K]
        return sum(d) / len(d) if d else 0.0

    tr = sorted(knn(r, i) for i, r in enumerate(train))
    c = tr[len(tr) // 2] if tr else 1.0
    if c <= 1e-12:
        c = (sum(tr) / len(tr)) if tr and sum(tr) > 0 else 1.0

    def score(r):
        return r / (r + c)

    trs = [score(r) for r in tr]
    thr = trs[int(0.95 * (len(trs) - 1))] if trs else 0.5
    thr = min(max(thr, 0.0), 1.0)
    scores = [score(knn(r)) for r in test]
    print(json.dumps({"scores": scores, "alerts": [s > thr for s in scores], "threshold": thr}))


main()
