import csv, sys, math, json

def read(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def main():
    a = sys.argv[1:]
    if not a or a[0] != 'remember':
        sys.exit(1)
    train = test = None
    for i, x in enumerate(a):
        if x == '--train':
            train = a[i + 1]
        elif x == '--test':
            test = a[i + 1]
    goals = [('Lbs-', 0), ('Acc+', 1), ('Mpg+', 1)]

    def cols(header):
        return [header.index(g) for g, _ in goals]

    def vals(header, rows):
        idx = cols(header)
        out = []
        for row in rows:
            try:
                out.append([float(row[i]) for i in idx])
            except (ValueError, IndexError):
                out.append(None)
        return out

    h1, r1 = read(train)
    tr = [v for v in vals(h1, r1) if v is not None]
    n = len(tr)
    stats = []
    for j in range(3):
        xs = [v[j] for v in tr]
        m = sum(xs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
        stats.append((m, sd))

    h2, r2 = read(test)
    te = vals(h2, r2)
    scores = []
    for v in te:
        s = 0.0
        for j, (_, h) in enumerate(goals):
            m, sd = stats[j]
            z = max(-3.0, min(3.0, (v[j] - m) / (sd + 1e-32)))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - h) ** 2
        scores.append(math.sqrt(s / 3))
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    print(json.dumps({
        "scores": scores,
        "top": order[:10],
        "best": order[0],
        "median": order[len(scores) // 2],
        "worst": order[-1],
    }))

main()
