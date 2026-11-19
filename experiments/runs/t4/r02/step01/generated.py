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
    for i, x in enumerate(a):
        if x == "--train":
            train = a[i + 1]
        elif x == "--test":
            test = a[i + 1]
    goals = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]

    def goal_cols(h):
        return [h.index(g) for g, _ in goals]

    def goal_vals(rows, idx, skip_missing):
        out = []
        for row in rows:
            v = [row[i] for i in idx]
            if skip_missing and "?" in v:
                continue
            out.append([float(x) for x in v])
        return out

    h, rows = load(train)
    tr = goal_vals(rows, goal_cols(h), True)
    n = len(tr)
    stats = []
    for j in range(3):
        col = [r[j] for r in tr]
        m = sum(col) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in col) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))

    h2, rows2 = load(test)
    te = goal_vals(rows2, goal_cols(h2), False)
    scores = []
    for r in te:
        s = 0.0
        for j in range(3):
            m, sd = stats[j]
            z = max(-3.0, min(3.0, (r[j] - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - goals[j][1]) ** 2
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
