import sys, csv, json, math

def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if any(c.strip() for c in r)]
    return rows[0], rows[1:]

def num(s):
    try:
        return float(s)
    except ValueError:
        return None

def remember(train, test):
    th, trows = load(train)
    sh, srows = load(test)
    goals = [(i, h, 0.0 if h[-1] == "-" else 1.0)
             for i, h in enumerate(th) if h[-1] in "+-"]
    stats = []
    for i, h, heaven in goals:
        vs = [v for v in (num(r[i]) for r in trows if i < len(r)) if v is not None]
        n = len(vs)
        mu = sum(vs) / n if n else 0.0
        sd = math.sqrt(sum((v - mu) ** 2 for v in vs) / (n - 1)) if n > 1 else 0.0
        stats.append((sh.index(h) if h in sh else i, mu, sd, heaven))

    def d2h(row):
        tot = 0.0
        for at, mu, sd, heaven in stats:
            v = num(row[at]) if at < len(row) else None
            if v is None:
                tot += 1.0
                continue
            z = max(-3.0, min(3.0, (v - mu) / (sd + 1e-32)))
            tot += (1 / (1 + math.exp(-1.702 * z)) - heaven) ** 2
        return math.sqrt(tot / len(stats))

    scores = [d2h(r) for r in srows]
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
    return {"scores": scores, "top": order[:10], "best": order[0],
            "median": order[len(order) // 2], "worst": order[-1]}

def main(argv):
    if len(argv) < 1 or argv[0] != "remember":
        sys.exit("usage: main.py remember --train TRAIN --test TEST")
    a = argv[1:]
    train = test = None
    i = 0
    while i < len(a):
        if a[i] == "--train": train = a[i + 1]; i += 2
        elif a[i] == "--test": test = a[i + 1]; i += 2
        else: i += 1
    if not train or not test:
        sys.exit("usage: main.py remember --train TRAIN --test TEST")
    print(json.dumps(remember(train, test)))

if __name__ == "__main__":
    main(sys.argv[1:])
