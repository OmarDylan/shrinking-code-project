import csv, sys, math, argparse, json

G = [("Lbs-", 0), ("Acc+", 1), ("Mpg+", 1)]

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        head = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    idx = [head.index(g) for g, _ in G]
    out = []
    for row in rows:
        try:
            out.append([float(row[i]) for i in idx])
        except ValueError:
            out.append(None)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    a = p.parse_args()
    tr = [r for r in load(a.train) if r is not None]
    te = load(a.test)
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
        for j, (_, h) in enumerate(G):
            m, sd = stats[j]
            z = max(-3.0, min(3.0, (r[j] - m) / (sd + 1e-32)))
            s += (1 / (1 + math.exp(-1.702 * z)) - h) ** 2
        scores.append(math.sqrt(s / 3))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }))

main()
