import sys, csv, json, math

NUM = ["Clndrs", "Volume", "Model"]
CAT = "origin"


def load(path):
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        rows = []
        for r in rd:
            v = []
            for c in NUM:
                try:
                    x = float((r.get(c) or "?").strip())
                    if not math.isfinite(x):
                        x = None
                except ValueError:
                    x = None
                v.append(x)
            o = (r.get(CAT) or "?").strip()
            v.append(None if o in ("?", "") else o)
            rows.append(v)
    return rows


def main():
    a = sys.argv
    tr = load(a[a.index("--train") + 1])
    te = load(a[a.index("--test") + 1])
    sc = []
    for j in range(3):
        xs = [r[j] for r in tr if r[j] is not None]
        if len(xs) > 1:
            m = sum(xs) / len(xs)
            s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
        else:
            s = 1.0
        sc.append(s if s > 1e-12 else 1.0)
    k = 5

    def dist(p, q):
        t = 0.0
        c = 0
        for j in range(3):
            if p[j] is not None and q[j] is not None:
                t += ((p[j] - q[j]) / sc[j]) ** 2
                c += 1
        if p[3] is not None and q[3] is not None:
            t += 0.0 if p[3] == q[3] else 1.0
            c += 1
        return math.sqrt(t / c) if c else None

    def raw(p, skip=None):
        ds = []
        for i, q in enumerate(tr):
            if i == skip:
                continue
            d = dist(p, q)
            if d is not None:
                ds.append(d)
        if not ds:
            return 0.0
        ds.sort()
        kk = min(k, len(ds))
        return sum(ds[:kk]) / kk

    trs = sorted(raw(p, i) for i, p in enumerate(tr))
    if trs:
        pos = 0.96 * (len(trs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(trs) - 1)
        thr = trs[lo] + (trs[hi] - trs[lo]) * (pos - lo)
    else:
        thr = 1.0
    thr = max(thr, 1e-9)
    scores = []
    for p in te:
        r = raw(p)
        scores.append(r / (r + thr))
    out = {"scores": scores, "alerts": [s > 0.5 for s in scores], "threshold": 0.5}
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
