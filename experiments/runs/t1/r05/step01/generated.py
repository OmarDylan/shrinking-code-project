import sys, csv, json, math

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
    tr = te = None
    i = 1
    while i < len(a):
        if a[i] == "--train":
            tr = a[i + 1]; i += 2
        elif a[i] == "--test":
            te = a[i + 1]; i += 2
        else:
            i += 1
    h1, train = load(tr)
    h2, test = load(te)
    goals1 = [(j, h[-1]) for j, h in enumerate(h1) if h[-1] in "+-"]
    goals2 = [(j, h[-1]) for j, h in enumerate(h2) if h[-1] in "+-"]
    stats = []
    for j, s in goals1:
        vals = []
        for row in train:
            try:
                vals.append(float(row[j]))
            except (ValueError, IndexError):
                pass
        n = len(vals)
        m = sum(vals) / n
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd, 1.0 if s == "+" else 0.0))
    scores = []
    for row in test:
        t = 0.0
        for (j, _), (m, sd, hv) in zip(goals2, stats):
            z = max(-3.0, min(3.0, (float(row[j]) - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            t += (nv - hv) ** 2
        scores.append(math.sqrt(t / 3))
    order = sorted(range(len(scores)), key=lambda k: (scores[k], k))
    n = len(order)
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[n // 2],
        "worst": order[-1],
    }))

main()
