import csv, sys, json, math

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def main():
    a = sys.argv[1:]
    if not a or a[0] != "remember":
        sys.exit(1)
    train = test = None
    for i, x in enumerate(a):
        if x == "--train":
            train = a[i + 1]
        elif x == "--test":
            test = a[i + 1]
    goals = [("Lbs-", 0), ("Acc+", 1), ("Mpg+", 1)]

    def vals(path):
        h, rows = load(path)
        idx = [h.index(g) for g, _ in goals]
        out = []
        for row in rows:
            if any(row[i] == "?" for i in idx):
                continue
            out.append([float(row[i]) for i in idx])
        return out

    tr = vals(train)
    te = vals(test)
    n = len(tr)
    stats = []
    for j in range(3):
        col = [r[j] for r in tr]
        m = sum(col) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in col) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))

    def score(r):
        s = 0.0
        for j, (_, h) in enumerate(goals):
            m, sd = stats[j]
            z = max(-3.0, min(3.0, (r[j] - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - h) ** 2
        return math.sqrt(s / 3)

    scores = [score(r) for r in te]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(te) // 2],
        "worst": order[-1],
    }))

main()
