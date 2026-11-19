import sys, json, argparse, bisect
from math import sqrt

FEATS = ["Clndrs", "Volume", "Model", "origin"]


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [l.strip().split(",") for l in f if l.strip()]
    names = [h.strip() for h in lines[0]]
    idx = [names.index(n) for n in FEATS]
    rows = []
    for l in lines[1:]:
        r = []
        for k, i in enumerate(idx):
            v = l[i].strip() if i < len(l) else "?"
            if v == "?" or v == "":
                r.append(None)
            elif k < 3:
                try:
                    r.append(float(v))
                except ValueError:
                    r.append(None)
            else:
                r.append(v)
        rows.append(r)
    return rows


def spot(train, test, K=5, Q=0.95):
    med, sc = [], []
    for k in range(3):
        xs = sorted(r[k] for r in train if r[k] is not None)
        m = xs[len(xs) // 2] if xs else 0.0
        mad = sorted(abs(x - m) for x in xs)
        s = 1.4826 * mad[len(mad) // 2] if mad else 0
        if s < 1e-9:
            mu = sum(xs) / max(1, len(xs))
            s = sqrt(sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)) if xs else 1
        med.append(m)
        sc.append(max(s, 1e-9))

    def d(a, b):
        g = []
        for k in range(3):
            if a[k] is not None and b[k] is not None:
                g.append(min(abs(a[k] - b[k]) / sc[k], 10) ** 2)
        if a[3] is not None and b[3] is not None:
            g.append(float(a[3] != b[3]))
        return sqrt(sum(g) / len(g)) if g else 0.0

    def raw(r, skip=None):
        ds = sorted(d(r, t) for j, t in enumerate(train) if j != skip)[:K]
        knn = sum(ds) / len(ds) if ds else 0.0
        zs = [min(abs(r[k] - med[k]) / sc[k], 10) for k in range(3) if r[k] is not None]
        return knn + 0.25 * (max(zs) if zs else 0.0)

    tr = sorted(raw(r, i) for i, r in enumerate(train))
    n = len(tr)
    ecdf = lambda v: bisect.bisect_right(tr, v) / n
    scores = [min(1.0, max(0.0, ecdf(raw(r)))) for r in test]
    return {"scores": scores, "alerts": [s > Q for s in scores], "threshold": Q}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--train")
    p.add_argument("--test")
    a = p.parse_args()
    if a.cmd == "spot":
        out = spot(load(a.train), load(a.test))
        sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
