import sys, csv, json, math

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def main():
    a = sys.argv
    train = a[a.index("--train") + 1]
    test = a[a.index("--test") + 1]
    goals = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]
    htr, rtr = load(train)
    hte, rte = load(test)

    def col(h, name):
        if name in h:
            return h.index(name)
        base = name[:-1]
        for i, x in enumerate(h):
            if x.rstrip("+-") == base:
                return i
        raise SystemExit("missing column " + name)

    itr = [col(htr, g) for g, _ in goals]
    ite = [col(hte, g) for g, _ in goals]
    stats = []
    for c in itr:
        v = [float(r[c]) for r in rtr]
        n = len(v)
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))

    scores = []
    for r in rte:
        s = 0.0
        for (g, h), c, (m, sd) in zip(goals, ite, stats):
            z = (float(r[c]) - m) / (sd + 1e-32)
            z = max(-3.0, min(3.0, z))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - h) ** 2
        scores.append(math.sqrt(s / 3))

    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    out = {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(order) // 2],
        "worst": order[-1],
    }
    sys.stdout.write(json.dumps(out))

if __name__ == "__main__":
    main()
