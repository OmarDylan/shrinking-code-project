import csv, sys, json, math, argparse

NUM = ["Clndrs", "Volume", "Model"]
GOALS = ["Lbs-", "Acc+", "Mpg+"]
HEAVEN = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}


def read(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in r]


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return m, sd


def norm(v, ms):
    m, sd = ms
    z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
    return 1 / (1 + math.exp(-1.702 * z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train")
    ap.add_argument("--test")
    a = ap.parse_args()
    train = read(a.train)
    test = read(a.test)
    n = len(train)
    gs = {g: stats([float(r[g]) for r in train]) for g in GOALS}
    ns = {c: stats([float(r[c]) for r in train]) for c in NUM}

    def d2h(r):
        return math.sqrt(sum((norm(float(r[g]), gs[g]) - HEAVEN[g]) ** 2 for g in GOALS) / 3)

    def feat(r):
        return ([norm(float(r[c]), ns[c]) for c in NUM], r["origin"])

    tf = [feat(r) for r in train]
    td = [d2h(r) for r in train]

    def dist(fa, fb):
        s = sum((x - y) ** 2 for x, y in zip(fa[0], fb[0]))
        s += 0.0 if fa[1] == fb[1] else 1.0
        return math.sqrt(s / 4)

    def predict(f):
        ds = sorted(range(n), key=lambda i: (round(dist(f, tf[i]), 12), i))[:5]
        return sum(td[i] for i in ds) / len(ds)

    cutoff = sorted(td)[n // 2]
    rank = sorted(range(n), key=lambda i: (round(td[i], 12), i))
    good = rank[:max(1, n // 4)]
    menu = []
    seen = set()
    for i in good:
        p = (float(train[i]["Clndrs"]), float(train[i]["Volume"]))
        if p not in seen:
            seen.add(p)
            menu.append(p)
            if len(menu) == 12:
                break

    out = []
    for idx, r in enumerate(test):
        f = feat(r)
        before = predict(f)
        if round(before, 10) <= round(cutoff, 10):
            out.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "already_good"})
            continue
        oc, ov = float(r["Clndrs"]), float(r["Volume"])
        best = None
        for c, v in menu:
            ch = {}
            if c != oc:
                ch["Clndrs"] = c
            if v != ov:
                ch["Volume"] = v
            if not ch:
                continue
            e = dict(r)
            e["Clndrs"] = c
            e["Volume"] = v
            fe = feat(e)
            after = predict(fe)
            if round(before - after, 10) >= 0.05:
                key = (len(ch), round(dist(f, fe), 10), round(after, 10), (c, v))
                if best is None or key < best[0]:
                    best = (key, after, ch)
        if best is None:
            out.append({"row": idx, "before": before, "after": before, "changes": {}, "status": "no_candidate"})
        else:
            out.append({"row": idx, "before": before, "after": best[1], "changes": best[2], "status": "improved"})
    sys.stdout.write(json.dumps({"repairs": out}))


if __name__ == "__main__":
    main()
