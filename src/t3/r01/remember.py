import csv, sys, json, math, argparse

GOALS = {"Lbs-": 0.0, "Acc+": 1.0, "Mpg+": 1.0}


def load(path):
    with open(path, newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r]
    return [c.strip() for c in rows[0]], rows[1:]


def val(row, i):
    try:
        s = row[i]
        return None if s == "?" or s == "" else float(s)
    except (IndexError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()

    h, tr = load(a.train)
    h2, te = load(a.test)
    idx = {g: h.index(g) for g in GOALS}
    idx2 = {g: h2.index(g) for g in GOALS}

    stats = {}
    for g in GOALS:
        v = [x for x in (val(r, idx[g]) for r in tr) if x is not None]
        n = len(v)
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        stats[g] = (m, sd)

    def norm(g, x):
        m, sd = stats[g]
        z = max(-3.0, min(3.0, (x - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    scores = []
    for r in te:
        s = 0.0
        for g, hv in GOALS.items():
            x = val(r, idx2[g])
            s += 1.0 if x is None else (norm(g, x) - hv) ** 2
        scores.append(math.sqrt(s / len(GOALS)))

    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    out = {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
