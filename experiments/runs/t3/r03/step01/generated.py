import csv, sys, math, argparse, json

GOALS = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    idx = [header.index(g) for g, _ in GOALS]
    return [[float(row[i]) for i in idx] for row in rows]


def remember(train_path, test_path):
    tr = load(train_path)
    te = load(test_path)
    n = len(tr)
    stats = []
    for j in range(3):
        col = [r[j] for r in tr]
        m = sum(col) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    scores = []
    for r in te:
        s = 0.0
        for j, (_, h) in enumerate(GOALS):
            m, sd = stats[j]
            z = max(-3.0, min(3.0, (r[j] - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - h) ** 2
        scores.append(math.sqrt(s / 3))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    if a.cmd != "remember":
        sys.exit("unknown command")
    sys.stdout.write(json.dumps(remember(a.train, a.test)))


if __name__ == "__main__":
    main()
