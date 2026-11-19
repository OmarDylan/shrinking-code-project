import sys, csv, math, json, argparse

GOALS = {"Lbs": 0.0, "Acc": 1.0, "Mpg": 1.0}

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    idx = {}
    for i, h in enumerate(header):
        base = h.rstrip("+-")
        if base in GOALS:
            idx[base] = i
    data = []
    for row in rows:
        try:
            data.append([float(row[idx[g]]) for g in GOALS])
        except (ValueError, IndexError):
            data.append(None)
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    train = [r for r in load(a.train) if r is not None]
    test = load(a.test)
    hs = list(GOALS.values())
    stats = []
    for j in range(3):
        col = [r[j] for r in train]
        n = len(col)
        m = sum(col) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))

    def score(r):
        s = 0.0
        for j in range(3):
            m, sd = stats[j]
            z = max(-3.0, min(3.0, (r[j] - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - hs[j]) ** 2
        return math.sqrt(s / 3)

    scores = [score(r) if r is not None else float("inf") for r in test]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    out = {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }
    sys.stdout.write(json.dumps(out))

main()
