import csv, sys, json, math

def load(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        rows = [[c.strip() for c in row] for row in r if row]
    return header, rows

def goals(header, rows):
    idx = [header.index(n) for n in ("Lbs-", "Acc+", "Mpg+")]
    out = []
    for row in rows:
        try:
            out.append([float(row[i]) for i in idx])
        except (ValueError, IndexError):
            out.append(None)
    return out

def main():
    a = sys.argv
    if len(a) < 2 or a[1] != "remember":
        sys.exit(1)
    train = a[a.index("--train") + 1]
    test = a[a.index("--test") + 1]
    th, tr = load(train)
    eh, te = load(test)
    tg = [g for g in goals(th, tr) if g is not None]
    n = len(tg)
    means, sds = [], []
    for j in range(3):
        col = [g[j] for g in tg]
        m = sum(col) / n
        var = sum((v - m) ** 2 for v in col) / (n - 1) if n > 1 else 0.0
        means.append(m)
        sds.append(math.sqrt(var))
    heaven = [0.0, 1.0, 1.0]
    scores = []
    for g in goals(eh, te):
        s = 0.0
        for j in range(3):
            z = (g[j] - means[j]) / (sds[j] + 1e-32)
            z = max(-3.0, min(3.0, z))
            nv = 1 / (1 + math.exp(-1.702 * z))
            s += (nv - heaven[j]) ** 2
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
