import csv, sys, json, math, argparse

GOALS = {"Lbs-": 0, "Acc+": 1, "Mpg+": 1}


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    idx = [header.index(g) for g in GOALS]
    return idx, rows


def goals(idx, rows):
    out = []
    for row in rows:
        vals = [row[i] for i in idx]
        if any(v == "?" or v == "" for v in vals):
            out.append(None)
        else:
            out.append([float(v) for v in vals])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    ti, trows = load(a.train)
    tr = [g for g in goals(ti, trows) if g is not None]
    n = len(tr)
    means = [sum(g[k] for g in tr) / n for k in range(3)]
    sds = [
        math.sqrt(sum((g[k] - means[k]) ** 2 for g in tr) / (n - 1)) if n > 1 else 0.0
        for k in range(3)
    ]
    heaven = list(GOALS.values())

    si, srows = load(a.test)
    scores = []
    for g in goals(si, srows):
        if g is None:
            scores.append(float("inf"))
            continue
        s = 0.0
        for k in range(3):
            z = (g[k] - means[k]) / (sds[k] + 1e-32)
            z = max(-3.0, min(3.0, z))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - heaven[k]) ** 2
        scores.append(math.sqrt(s / 3))

    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    res = {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }
    sys.stdout.write(json.dumps(res))


if __name__ == "__main__":
    main()
