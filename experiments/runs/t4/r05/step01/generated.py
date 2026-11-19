import sys, csv, json, math

def load(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def goals(header, rows):
    idx = [header.index(n) for n in ("Lbs-", "Acc+", "Mpg+")]
    return [[float(row[i]) for i in idx] for row in rows]

def main():
    a = sys.argv
    train = a[a.index("--train") + 1]
    test = a[a.index("--test") + 1]
    th, tr = load(train)
    eh, er = load(test)
    G = goals(th, tr)
    E = goals(eh, er)
    n = len(G)
    means = [sum(g[j] for g in G) / n for j in range(3)]
    sds = [math.sqrt(sum((g[j] - means[j]) ** 2 for g in G) / (n - 1)) if n > 1 else 0.0
           for j in range(3)]
    heaven = [0.0, 1.0, 1.0]
    scores = []
    for g in E:
        s = 0.0
        for j in range(3):
            z = (g[j] - means[j]) / (sds[j] + 1e-32)
            z = max(-3.0, min(3.0, z))
            v = 1 / (1 + math.exp(-1.702 * z))
            s += (v - heaven[j]) ** 2
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
