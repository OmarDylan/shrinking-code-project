import sys, csv, math, json

def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def main():
    a = sys.argv[1:]
    train = a[a.index("--train") + 1]
    test = a[a.index("--test") + 1]
    goals = [("Lbs-", 0.0), ("Acc+", 1.0), ("Mpg+", 1.0)]
    htr, tr = load(train)
    hte, te = load(test)

    def col(h, rows, name):
        i = h.index(name)
        return [float(r[i]) for r in rows]

    stats = []
    for name, hv in goals:
        v = col(htr, tr, name)
        n = len(v)
        m = sum(v) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd, hv, col(hte, te, name)))

    def norm(v, m, sd):
        z = max(-3.0, min(3.0, (v - m) / (sd + 1e-32)))
        return 1 / (1 + math.exp(-1.702 * z))

    scores = []
    for i in range(len(te)):
        s = sum((norm(vals[i], m, sd) - hv) ** 2 for m, sd, hv, vals in stats)
        scores.append(math.sqrt(s / 3))

    order = sorted(range(len(te)), key=lambda i: (scores[i], i))
    out = {
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(te) // 2],
        "worst": order[-1],
    }
    sys.stdout.write(json.dumps(out))

if __name__ == "__main__":
    main()
