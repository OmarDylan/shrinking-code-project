import sys, csv, math, json

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
    i = 1
    while i < len(a):
        if a[i] == "--train":
            train = a[i + 1]; i += 2
        elif a[i] == "--test":
            test = a[i + 1]; i += 2
        else:
            i += 1
    goals = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]
    th, tr = load(train)
    eh, te = load(test)
    stats = []
    for g, _ in goals:
        j = th.index(g)
        vals = []
        for row in tr:
            try:
                vals.append(float(row[j]))
            except (ValueError, IndexError):
                pass
        n = len(vals)
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))
    idx = [eh.index(g) for g, _ in goals]
    scores = []
    for row in te:
        s = 0.0
        for k, (g, h) in enumerate(goals):
            v = float(row[idx[k]])
            m, sd = stats[k]
            z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - h) ** 2
        scores.append(math.sqrt(s / 3))
    order = sorted(range(len(scores)), key=lambda k: (scores[k], k))
    out = {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }
    sys.stdout.write(json.dumps(out))

main()
